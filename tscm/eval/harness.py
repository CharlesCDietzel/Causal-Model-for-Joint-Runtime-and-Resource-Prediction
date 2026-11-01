"""Evaluation: the splits that each test one claim, baselines, metrics and ablations.

Baselines implemented here are the ones that need nothing beyond the runs table. Bellamy,
DNNPerf, TraPPM, GPUMemNet and QualX need their authors' models and data; wrap one in a class
with ``fit(train)`` and ``predict(test)`` and it slots into ``evaluate``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from tscm.fit.modular import ModularFit
from tscm.model import GiB, Job, Stage
from tscm.query import forward as fwd
from tscm.query import queries

COVERAGE_TARGET = (0.88, 0.92)
FEATURES = ("s", "m", "M_e", "c_e", "e")


# Metrics


def mape(predicted, observed) -> float:
    predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    return float(np.mean(np.abs(predicted - observed) / observed))


def median_ape(predicted, observed) -> float:
    predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    return float(np.median(np.abs(predicted - observed) / observed))


def coverage(low, high, observed) -> float:
    low, high, observed = (np.asarray(v, float) for v in (low, high, observed))
    return float(np.mean((observed >= low) & (observed <= high)))


def memory_requirement_error(predicted: float, measured: float) -> float:
    return abs(predicted - measured) / measured


def profiling_cost_to_target(
    cumulative_su: np.ndarray,
    mape_after_run: np.ndarray,
    target: float = 0.15,
) -> float:
    """SU spent when MAPE first reaches the target; infinity if it never does."""
    hit = np.flatnonzero(np.asarray(mape_after_run) <= target)
    return float(np.asarray(cumulative_su)[hit[0]]) if hit.size else float("inf")


def deadline_regret(chosen_cost: float, best_cost: float) -> float:
    """Relative extra cost of the chosen configuration over the best feasible one."""
    return (chosen_cost - best_cost) / best_cost


# Splits. Each returns (train, test) run tables.


Split = tuple[pd.DataFrame, pd.DataFrame]


def interpolation_split(runs: pd.DataFrame, test_share: float = 0.25, seed: int = 0) -> Split:
    shuffled = runs.sample(frac=1.0, random_state=seed)
    cut = int(round(len(shuffled) * (1.0 - test_share)))
    return shuffled.iloc[:cut], shuffled.iloc[cut:]


def size_extrapolation_split(runs: pd.DataFrame, s_target: float) -> Split:
    """Train on inputs up to 8% of s_target; test above that, up to full size."""
    return _by_mask(runs, runs["s"] <= 0.08 * s_target)


def regime_extrapolation_split(
    runs: pd.DataFrame,
    max_pressure: pd.Series,
    large_input: float,
) -> Split:
    """Train on rho < 1 plus Phase A step 2; test on rho > 1 at large input."""
    calm = max_pressure < 1.0
    train = runs[calm | (runs["phase"] == "A2")]
    test = runs[~calm & (runs["s"] >= large_input) & (runs["phase"] != "A2")]
    return train, test


def machine_transfer_split(
    runs: pd.DataFrame,
    job_id: str,
    source_types: tuple[str, ...],
    profiled_on: str,
    target_type: str,
) -> Split:
    """All jobs on types A, B and the profiled job on A; test the same job on type C."""
    is_job = runs["job_id"] == job_id
    train = runs[(~is_job & runs["machine_type"].isin(source_types))
                 | (is_job & (runs["machine_type"] == profiled_on))]
    return train, runs[is_job & (runs["machine_type"] == target_type)]


def cold_start_split(runs: pd.DataFrame, job_id: str, n_runs: int, seed: int = 0) -> Split:
    """Other jobs plus ``n_runs`` of the new job; the rest of the new job is the test set."""
    is_job = runs["job_id"] == job_id
    target = runs[is_job].sample(frac=1.0, random_state=seed)
    return pd.concat([runs[~is_job], target.iloc[:n_runs]]), target.iloc[n_runs:]


def _by_mask(runs: pd.DataFrame, mask: pd.Series) -> Split:
    return runs[mask], runs[~mask]


# Baselines


class Baseline(Protocol):
    def fit(self, train: pd.DataFrame) -> Baseline: ...
    def predict(self, test: pd.DataFrame) -> np.ndarray: ...


def _ernest_design(frame: pd.DataFrame) -> np.ndarray:
    s = frame["s"].to_numpy(float) / GiB
    m = frame["m"].to_numpy(float)
    return np.column_stack([np.ones_like(s), s / m, np.log(m), m])


@dataclass
class Ernest:
    """theta_0 + theta_1 s/m + theta_2 log m + theta_3 m, by NNLS."""

    target: str = "T"
    coefficients: np.ndarray | None = None

    def fit(self, train):
        self.coefficients = nnls(_ernest_design(train), train[self.target].to_numpy(float))[0]
        return self

    def predict(self, test):
        return _ernest_design(test) @ self.coefficients


@dataclass
class ErnestGpu:
    """Ernest-style NNLS on GPU count: the multi-GPU scaling baseline."""

    coefficients: np.ndarray | None = None

    @staticmethod
    def _design(N):
        N = np.asarray(N, float)
        return np.column_stack([np.ones_like(N), 1.0 / N, np.log(N), N])

    def fit(self, N, T_step):
        self.coefficients = nnls(self._design(N), np.asarray(T_step, float))[0]
        return self

    def predict(self, N):
        return self._design(N) @ self.coefficients


def _train_gbm(features: np.ndarray, target: np.ndarray) -> lgb.Booster:
    settings = {"objective": "regression", "verbosity": -1, "num_leaves": 7,
                "min_data_in_leaf": 2, "min_data_in_bin": 1, "seed": 0,
                "deterministic": True, "force_row_wise": True,
                # Dozens of rows: OpenMP spin-waiting makes the default ~500x slower.
                "num_threads": 1}
    return lgb.train(settings, lgb.Dataset(features, target, params={"verbose": -1}),
                     num_boost_round=100)


@dataclass
class BlackBoxGbm:
    """The correlational control: one GBM per outcome on Tier 0 features, in log space."""

    target: str = "T"
    model: lgb.Booster | None = None

    def features(self, frame: pd.DataFrame) -> np.ndarray:
        return np.log(frame[list(FEATURES)].to_numpy(float))

    def fit(self, train):
        self.model = _train_gbm(self.features(train),
                                np.log(train[self.target].to_numpy(float)))
        return self

    def predict(self, test):
        return np.exp(self.model.predict(self.features(test)))


@dataclass
class C3OGbm(BlackBoxGbm):
    """C3O's GBM variant: the same model on raw features, so it cannot extrapolate a trend."""

    def features(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[list(FEATURES)].to_numpy(float)


@dataclass
class C3OOptimistic:
    """C3O's optimistic model: a scale-out curve times an inputs-only correction."""

    scale_out: Ernest | None = None
    correction: BlackBoxGbm | None = None

    def fit(self, train):
        self.scale_out = Ernest().fit(train)
        ratio = train.assign(ratio=train["T"].to_numpy(float)
                             / np.maximum(self.scale_out.predict(train), 1e-9))
        self.correction = BlackBoxGbm(target="ratio").fit(ratio)
        return self

    def predict(self, test):
        return self.scale_out.predict(test) * self.correction.predict(test)


@dataclass
class Bell:
    """Bell's selection: parametric Ernest or nearest-neighbor, picked by cross-validation."""

    neighbours: int = 3
    parametric: Ernest | None = None
    train: pd.DataFrame | None = None
    use_parametric: bool = True

    def fit(self, train):
        self.train = train
        self.parametric = Ernest().fit(train)
        errors = {"parametric": [], "local": []}
        for i in range(len(train)):
            held, rest = train.iloc[[i]], train.drop(train.index[i])
            if len(rest) < 2:
                continue
            observed = held["T"].to_numpy(float)
            errors["parametric"].append(mape(Ernest().fit(rest).predict(held), observed))
            errors["local"].append(mape(self._local(rest, held), observed))
        if errors["parametric"]:
            self.use_parametric = np.mean(errors["parametric"]) <= np.mean(errors["local"])
        return self

    def _local(self, reference, query):
        known = np.log(reference[["s", "m"]].to_numpy(float))
        wanted = np.log(query[["s", "m"]].to_numpy(float))
        distances = np.linalg.norm(known[None] - wanted[:, None], axis=2)
        nearest = np.argsort(distances, axis=1)[:, : self.neighbours]
        return reference["T"].to_numpy(float)[nearest].mean(axis=1)

    def predict(self, test):
        if self.use_parametric:
            return self.parametric.predict(test)
        return self._local(self.train, test)


@dataclass
class KsPlusMemory:
    """Peak-memory regression on input size, KS+ style."""

    coefficients: np.ndarray | None = None

    def fit(self, train):
        design = np.column_stack([np.ones(len(train)), train["s"].to_numpy(float) / GiB])
        self.coefficients = nnls(design, train["M_peak"].to_numpy(float))[0]
        return self

    def predict(self, test):
        design = np.column_stack([np.ones(len(test)), test["s"].to_numpy(float) / GiB])
        return design @ self.coefficients


@dataclass
class RuyaMemory:
    """Ruya-style: the linear memory model, raised by its worst under-prediction."""

    model: KsPlusMemory | None = None
    margin: float = 0.0

    def fit(self, train):
        self.model = KsPlusMemory().fit(train)
        self.margin = float(np.max(train["M_peak"].to_numpy(float) - self.model.predict(train)))
        return self

    def predict(self, test):
        return self.model.predict(test) + max(self.margin, 0.0)


# The model under test


def predict_runs(post: fwd.Posterior, cfgs, n_samples: int = 400, seed: int = 0) -> dict:
    """P50 and 90% interval of T and M_peak for each config."""
    out = fwd.run(post, cfgs, n_samples, seed)
    report = {}
    for name in ("T", "M_peak"):
        low, median, high = np.quantile(out[name], [0.05, 0.5, 0.95], axis=1)
        report[name] = {"p50": median, "lo": low, "hi": high}
    return report


@dataclass
class SplitReport:
    name: str
    scores: dict[str, dict[str, float]]


def evaluate(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    tscm_predict: Callable[[pd.DataFrame], dict],
    baselines: dict[str, Baseline] | None = None,
) -> SplitReport:
    """Score the TSCM and each baseline on one split, overall."""
    observed = test["T"].to_numpy(float)
    predicted = tscm_predict(test)
    scores = {"tscm": {
        "mape": mape(predicted["T"]["p50"], observed),
        "median_ape": median_ape(predicted["T"]["p50"], observed),
        "coverage": coverage(predicted["T"]["lo"], predicted["T"]["hi"], observed),
    }}
    if "M_peak" in predicted and test["M_peak"].notna().all():
        scores["tscm"]["memory_mape"] = mape(predicted["M_peak"]["p50"], test["M_peak"])
    for label, model in (baselines or default_baselines()).items():
        fitted = model.fit(train)
        scores[label] = {"mape": mape(fitted.predict(test), observed),
                         "median_ape": median_ape(fitted.predict(test), observed)}
    return SplitReport(name, scores)


def default_baselines() -> dict[str, Baseline]:
    return {
        "ernest": Ernest(),
        "bell": Bell(),
        "c3o_gbm": C3OGbm(),
        "c3o_optimistic": C3OOptimistic(),
        "black_box_gbm": BlackBoxGbm(),
    }


def report_by_family(
    runs: pd.DataFrame,
    predicted: np.ndarray,
) -> dict[str, float]:
    """MAPE per workload family, since gains are expected to differ across them."""
    frame = runs.assign(predicted=predicted)
    return {
        family: mape(group["predicted"], group["T"]) for family, group in frame.groupby("family")
    }


def calibration_ok(value: float) -> bool:
    return COVERAGE_TARGET[0] <= value <= COVERAGE_TARGET[1]


# Ablations


ABLATIONS = (
    "no_regime", "generic_functions", "whole_job", "no_tiers", "no_cross_job_priors",
    "no_phase_a2",
)


def no_regime(fit: ModularFit) -> ModularFit:
    """Set h = 0 and hold g constant: spill, recomputation and GC response all go."""
    params = {name: np.array(value, copy=True) for name, value in fit.params.items()}
    params["lam"][:] = 0.0
    params["theta_r"][:] = 0.0
    params["gamma1"] = np.array(0.0)
    params["gamma2"] = np.array(0.0)
    return ModularFit(params, fit.pinned, {**fit.diagnostics, "ablation": "no_regime"})


def without_phase_a2(runs: pd.DataFrame) -> pd.DataFrame:
    return runs[runs["phase"] != "A2"]


def whole_job_tables(
    job: Job,
    runs: pd.DataFrame,
    stages: pd.DataFrame,
) -> tuple[Job, pd.DataFrame]:
    """Collapse every stage into one, for the whole-job ablation."""
    grouped = stages.groupby("run_id")
    rows = pd.DataFrame({
        "run_id": list(grouped.groups),
        "stage_signature": "job",
        "iteration": 0,
        "parent_signatures": [[] for _ in grouped.groups],
        "stage_type": "job",
        "reads_cache": grouped["reads_cache"].any().to_numpy(),
        "submission_time": 0.0,
        "V_read": grouped["V_read"].sum().to_numpy(),
        "V_shuffle_read": grouped["V_shuffle_read"].sum().to_numpy(),
        "V_in": grouped["V_in"].sum().to_numpy(),
        "V_sh": grouped["V_sh"].sum().to_numpy(),
        "V_out": grouped["V_out"].sum().to_numpy(),
        "remote_shuffle_bytes": grouped["remote_shuffle_bytes"].sum().to_numpy(),
        "n": grouped["n"].sum().to_numpy(),
        "S_memory": grouped["S_memory"].sum().to_numpy(),
        "S": grouped["S"].sum().to_numpy(),
        "g": grouped["g"].mean().to_numpy(),
        "compute_time": grouped["compute_time"].sum().to_numpy(),
        "W_p95": grouped["W_p95"].max().to_numpy(),
        "T_k": grouped["T_k"].sum().to_numpy(),
        "C": grouped["C"].max().to_numpy(),
        "cpu_util": grouped["cpu_util"].mean().to_numpy(),
    })
    collapsed = Job(job.job_id, job.family, (
        Stage("job", kind="fixed", stage_type="job", writes_shuffle=True, writes_output=True,
              reads_cache=job.cached, fixed_tasks=int(rows["n"].iloc[0])),
    ), job.features, job.cached)
    return collapsed, rows


def run_level_nodes(runs: pd.DataFrame, stages: pd.DataFrame) -> pd.DataFrame:
    """Add per-run totals of the mediators, for the generic-functions ablation."""
    grouped = stages.groupby("run_id")
    return runs.assign(
        V_total=runs["run_id"].map(grouped["V_in"].sum()),
        S_total=runs["run_id"].map(grouped["S"].sum()),
        g_mean=runs["run_id"].map(grouped["g"].mean()),
    )


MEDIATORS = ("V_total", "S_total", "g_mean")


@dataclass
class GenericNodes:
    """The generic-functions ablation: the same graph, with a GBM at each node.

    Tier 1 and 2 nodes are learned from Tier 0; T from Tier 0 plus those mediators. Prediction
    feeds predicted mediators forward, so only the functional forms differ from the TSCM.
    Works on run-level totals, from ``run_level_nodes``.
    """

    mediators: dict[str, lgb.Booster] | None = None
    outcome: lgb.Booster | None = None

    @staticmethod
    def _tier0(frame):
        return np.log(frame[list(FEATURES)].to_numpy(float))

    def fit(self, train):
        tier0 = self._tier0(train)
        self.mediators = {
            name: _train_gbm(tier0, np.log1p(train[name].to_numpy(float))) for name in MEDIATORS
        }
        observed = np.column_stack([np.log1p(train[name].to_numpy(float)) for name in MEDIATORS])
        self.outcome = _train_gbm(np.column_stack([tier0, observed]),
                                  np.log(train["T"].to_numpy(float)))
        return self

    def predict(self, test):
        tier0 = self._tier0(test)
        predicted = np.column_stack([self.mediators[name].predict(tier0) for name in MEDIATORS])
        return np.exp(self.outcome.predict(np.column_stack([tier0, predicted])))


def memory_requirement_scores(
    post: fwd.Posterior,
    cfgs,
    measured: np.ndarray,
    n_samples: int = 200,
) -> dict[str, float]:
    """Mean error of both M_req answers against the swept ground truth."""
    tolerance, zero_spill = [], []
    for cfg, truth in zip(cfgs, measured, strict=True):
        answer = queries.memory_requirement(post, cfg, n_samples=n_samples)
        tolerance.append(memory_requirement_error(answer.tolerance_bytes, truth))
        zero_spill.append(memory_requirement_error(answer.zero_spill_bytes, truth))
    return {"tolerance": float(np.mean(tolerance)), "zero_spill": float(np.mean(zero_spill))}


