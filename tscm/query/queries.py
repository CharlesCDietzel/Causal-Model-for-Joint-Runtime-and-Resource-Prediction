"""The supported queries. Each is an intervention do(x) followed by forward passes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from tscm.mechanisms.residual import ResidualLearner, tier0_features
from tscm.mechanisms.tier3 import TERMS
from tscm.model import SPARK_RESERVED, Config, Machine
from tscm.query import forward as fwd
from tscm.query import guards

DEFAULT_TOLERANCE = 0.05
OUTCOMES = ("T", "M_peak", "M_req_zero_spill", "S_total", "u")


@dataclass
class Prediction:
    config: Config
    outcomes: dict[str, dict[str, float]]
    stages: dict[str, dict[str, dict[str, float]]]
    flags: list[guards.Flag]
    draws: dict[str, np.ndarray]


def predict(
    post: fwd.Posterior,
    cfg: Config,
    support: guards.Support | None = None,
    residual: ResidualLearner | None = None,
    n_samples: int = 1000,
    seed: int = 0,
) -> Prediction:
    """Runtime and resources for x: P50, P95 and the 90% interval of every outcome."""
    draws = {k: v[0] for k, v in fwd.run(post, [cfg], n_samples, seed).items()}
    applied = None
    if residual is not None:
        features = tier0_features(cfg)
        applied = residual.applicable(features)
        draws["T"] = draws["T"] * np.exp(residual.predict(features))

    outcomes = {name: _floats(fwd.summarize(draws[name])) for name in OUTCOMES}
    signatures = [s.signature for s in post.job.stages]
    stages = {
        signature: {
            name: _floats(fwd.summarize(draws[name][:, i]))
            for name in ("T_k", "V_in", "V_sh", "S", "g", "u_k", "rho", "W", "n")
        }
        for i, signature in enumerate(signatures)
    }
    flags = []
    if support is not None:
        rho = dict(zip(signatures, np.median(draws["rho"], axis=0).tolist(), strict=True))
        flags = guards.check(cfg, support, rho, applied)
    return Prediction(cfg, outcomes, stages, flags, draws)


def _floats(summary: dict) -> dict[str, float]:
    return {k: float(v) for k, v in summary.items()}


@dataclass
class MemoryRequirement:
    tolerance_bytes: float
    zero_spill_bytes: float
    unlimited_runtime: float
    runtime_at_requirement: float


def memory_requirement(
    post: fwd.Posterior,
    cfg: Config,
    tolerance: float = DEFAULT_TOLERANCE,
    n_samples: int = 400,
    seed: int = 0,
    iterations: int = 24,
) -> MemoryRequirement:
    """Smallest M_e with P50 runtime within ``tolerance`` of unlimited-memory runtime.

    M_e -> infinity is stood in for by a large finite allocation: the GC term contains
    log M_e, so a truly unbounded value would push it to whichever limit gamma_2 points at.
    """
    zero_spill = float(np.median(fwd.run(post, [cfg], n_samples, seed)["M_req_zero_spill"]))
    unlimited = max(16.0 * zero_spill, 16.0 * cfg.M_e)
    median_T = lambda M: float(  # noqa: E731
        np.median(fwd.run(post, [cfg.with_(M_e=M)], n_samples, seed)["T"]))
    T_unlimited = median_T(unlimited)
    limit = (1.0 + tolerance) * T_unlimited

    low, high = np.log(SPARK_RESERVED * 1.01), np.log(unlimited)
    if median_T(float(np.exp(low))) <= limit:
        high = low
    for _ in range(iterations):
        if high - low < 1e-3:
            break
        mid = 0.5 * (low + high)
        if median_T(float(np.exp(mid))) <= limit:
            high = mid
        else:
            low = mid
    required = float(np.exp(high))
    return MemoryRequirement(required, zero_spill, T_unlimited, median_T(required))


@dataclass
class ConfigurationChoice:
    config: Config
    runtime_p50: float
    runtime_p95: float
    cost_p50: float


def cheapest_configuration(
    post: fwd.Posterior,
    base: Config,
    deadline: float,
    machines: Iterable[Machine],
    machine_counts: Iterable[int],
    executor_memories: Iterable[float],
    n_samples: int = 400,
    seed: int = 0,
) -> ConfigurationChoice | None:
    """Grid search: keep P95 T <= deadline, minimize P50 cost T * m * price."""
    grid = [
        base.with_(machine=machine, m=m, M_e=M_e)
        for machine in machines
        for m in machine_counts
        for M_e in executor_memories
        if base.e * M_e <= machine.memory
    ]
    if not grid:
        return None
    T = fwd.run(post, grid, n_samples, seed)["T"]
    p50, p95 = np.quantile(T, [0.5, 0.95], axis=1)
    cost = p50 / 3600.0 * np.array([c.m * c.machine.rate for c in grid])
    feasible = np.flatnonzero(p95 <= deadline)
    if feasible.size == 0:
        return None
    best = feasible[np.argmin(cost[feasible])]
    return ConfigurationChoice(grid[best], float(p50[best]), float(p95[best]), float(cost[best]))


def effect(
    post: fwd.Posterior,
    x: Config,
    x_prime: Config,
    n_samples: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Paired difference of forward passes, do(x') minus do(x), on shared draws."""
    draws = fwd.run(post, [x, x_prime], n_samples, seed)
    return {name: _floats(fwd.summarize(draws[name][1] - draws[name][0])) for name in OUTCOMES}


def decompose(post: fwd.Posterior, cfg: Config, n_samples: int = 400, seed: int = 0) -> dict:
    """Why is it slow? Median seconds per runtime term, and the dominant term per stage."""
    draws = fwd.run(post, [cfg], n_samples, seed)
    report = {}
    for i, stage in enumerate(post.job.stages):
        seconds = {t: float(np.median(draws[f"term_{t}"][0, :, i])) for t in TERMS}
        report[stage.signature] = {"terms": seconds, "dominant": max(seconds, key=seconds.get)}
    return report
