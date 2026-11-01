"""Cross-job transfer: family priors, machine-type factors and historical-run checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tscm.model import Hardware, Job, default_params, param_specs
from tscm.query.forward import Posterior

SHARED = "*"  # stage type for job-wide parameters
MIN_LOG_SCALE = 0.2
RIDGE_CORRELATION = 0.98


@dataclass
class FamilyPrior:
    """log theta ~ N(mu, sigma) per (parameter, stage type), learned from past jobs.

    Stands for mu_{f,t} and Sigma_{f,t}. Real-valued parameters are stored on their natural
    scale, non-negative ones on the log scale.
    """

    family: str
    stats: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)

    def lookup(self, name: str, stage_type: str = SHARED) -> tuple[float, float] | None:
        return self.stats.get((name, stage_type))


def learn_family_prior(family: str, posteriors: Iterable[Posterior]) -> FamilyPrior:
    """Empirical-Bayes prior from per-job posterior medians, grouped by stage type.

    Fitted per job and then pooled, rather than as one joint NUTS model over all jobs, so
    adding a job costs one fit. With a single job the spread falls back to MIN_LOG_SCALE.
    """
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for post in posteriors:
        job, point = post.job, post.point()
        specs = param_specs(job)
        for name, spec in specs.items():
            if name.startswith(("sigma_", "phi_")):
                continue
            value = np.asarray(point[name], dtype=float)
            if spec.per_stage:
                for k, stage in enumerate(job.stages):
                    values[(name, stage.stage_type)].append(float(value[k]))
            else:
                values[(name, SHARED)].append(float(value))

    prior = FamilyPrior(family)
    for (name, stage_type), observed in values.items():
        spec = _spec_for(name)
        array = np.asarray(observed)
        if spec == "nonneg":
            array = np.log(array[array > 0])
            if array.size == 0:
                continue
        floor = {"nonneg": MIN_LOG_SCALE, "unit": 0.1, "real": 0.5}[spec]
        scale = float(array.std(ddof=1)) if array.size > 1 else floor
        prior.stats[(name, stage_type)] = (float(array.mean()), max(scale, floor))
    return prior


def _spec_for(name: str) -> str:
    real = name.startswith("a_") or name in ("gamma0", "gamma2")
    return "real" if real else ("unit" if name == "g_max" else "nonneg")


def posterior_from_prior(
    job: Job,
    prior: FamilyPrior,
    n_draws: int = 1000,
    seed: int = 0,
    hardware: Hardware | None = None,
) -> Posterior:
    """Draws for a job with no profiling runs: the model predicts from the prior alone."""
    rng = np.random.default_rng(seed)
    specs = param_specs(job)
    base = default_params(job)
    samples = {name: np.repeat(value[None], n_draws, axis=0) for name, value in base.items()}

    def draw(name, stage_type, spec):
        stat = prior.lookup(name, stage_type)
        if stat is None:
            return None
        loc, scale = stat
        values = rng.normal(loc, scale, n_draws)
        if spec.constraint == "nonneg":
            return np.exp(values)
        if spec.constraint == "unit":
            return np.clip(values, 0.02, 0.98)
        return values

    for name, spec in specs.items():
        if spec.per_stage and len(spec.shape) == 1 and spec.shape[0] == len(job.stages):
            for k, stage in enumerate(job.stages):
                values = draw(name, stage.stage_type, spec)
                if values is not None:
                    samples[name][:, k] = values
        elif not spec.per_stage:
            values = draw(name, SHARED, spec)
            if values is not None:
                samples[name] = values
    return Posterior(job, samples, hardware or Hardware(), {"source": "family prior"})


def machine_factors(by_type: dict[str, list[Posterior]], reference: str) -> dict[str, float]:
    """kappa_h: per-type speed on theta_1, relative to ``reference``, from shared jobs.

    Each job compares only with itself across types, so job difficulty cancels out.
    """
    base: dict[tuple[str, str], float] = {}
    for post in by_type.get(reference, []):
        point = post.point()["theta1"]
        for k, stage in enumerate(post.job.stages):
            base[(post.job.job_id, stage.signature)] = float(point[k])

    kappa = {reference: 1.0}
    for name, posts in by_type.items():
        if name == reference:
            continue
        ratios = []
        for post in posts:
            point = post.point()["theta1"]
            for k, stage in enumerate(post.job.stages):
                reference_value = base.get((post.job.job_id, stage.signature), 0.0)
                if reference_value > 0 and point[k] > 0:
                    ratios.append(np.log(float(point[k]) / reference_value))
        if ratios:
            kappa[name] = float(np.exp(np.median(ratios)))
    return kappa


@dataclass(frozen=True)
class OverlapReport:
    correlation: float
    weakly_identified: bool
    message: str


def check_overlap(runs: pd.DataFrame) -> OverlapReport:
    """Flag theta_2 and theta_3 when historical (s, m) pairs lie on a narrow ridge."""
    log_s, log_m = np.log(runs["s"].to_numpy()), np.log(runs["m"].to_numpy())
    if np.ptp(log_s) < 1e-9 or np.ptp(log_m) < 1e-9:
        return OverlapReport(1.0, True, "s or m never varies; theta_2, theta_3 unidentified")
    correlation = float(np.corrcoef(log_s, log_m)[0, 1])
    weak = abs(correlation) > RIDGE_CORRELATION
    message = (
        "m is almost proportional to s; add 1-2 profiling runs off the ridge"
        if weak else "historical (s, m) pairs overlap adequately"
    )
    return OverlapReport(correlation, weak, message)


def load_recorded(runs: pd.DataFrame) -> bool:
    """True when every historical run recorded cluster load, the backdoor condition."""
    historical = runs[runs["source"] == "historical"]
    return historical.empty or bool(historical["cluster_load"].notna().all())
