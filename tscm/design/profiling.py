"""Profiling phases A, B and C, for Spark and GPU jobs.

The key idea: memory pressure is a ratio, so shrinking executor memory at a small input
reaches the spill and GC regimes of a full-size run without paying for one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from tscm.model import SPARK_RESERVED

DEFAULT_FRACTIONS = (0.005, 0.01, 0.02)
PRESSURE_TARGETS = (1.5, 3.0)
INTERVAL_TARGET = 0.20  # stop Phase C once the 90% interval is within +/-20%
GPU_WARMUP_STEPS = 10
GPU_MEASURED_STEPS = 50


@dataclass(frozen=True)
class PlannedRun:
    phase: str
    input_fraction: float
    machines: int
    executor_memory: float | None = None
    note: str = ""


def phase_a_step1(
    generous_memory: float,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> list[PlannedRun]:
    """One machine at three input fractions with plenty of memory: Tier 1, omega_1, M_base."""
    return [
        PlannedRun("A", fraction, 1, generous_memory, "single-machine profile")
        for fraction in fractions
    ]


def memory_for_pressure(rho: float, W: float, C: float, c_e: int, f_spark: float = 0.6) -> float:
    """M_e reaching a target pressure: rho = (c_e W + C) / (f_spark (M_e - 300 MB))."""
    return SPARK_RESERVED + (c_e * W + C) / (rho * f_spark)


def phase_a_step2(
    fraction: float,
    W: float,
    C: float,
    c_e: int,
    f_spark: float = 0.6,
    targets: Sequence[float] = PRESSURE_TARGETS,
) -> list[PlannedRun]:
    """Repeat the largest fraction with memory shrunk to rho of about 1.5 and 3."""
    return [
        PlannedRun("A", fraction, 1, memory_for_pressure(rho, W, C, c_e, f_spark),
                   f"memory shrink to rho={rho}")
        for rho in targets
    ]


def _log_det(rows: np.ndarray, ridge: float = 1e-6) -> float:
    information = rows.T @ rows + ridge * np.eye(rows.shape[1])
    return float(np.linalg.slogdet(information)[1])


def ernest_features(fractions: np.ndarray, machines: np.ndarray) -> np.ndarray:
    """The Tier 3 NNLS columns, linearized at the prior: 1, s/m, log m, m."""
    fractions, machines = np.asarray(fractions, float), np.asarray(machines, float)
    columns = [np.ones_like(fractions), fractions / machines, np.log(machines), machines]
    scaled = np.column_stack(columns)
    return scaled / np.maximum(np.abs(scaled).max(axis=0), 1e-12)


def phase_b(
    fractions: Sequence[float] = (0.01, 0.02, 0.04, 0.08),
    machines: Sequence[int] = (2, 4, 8, 16),
    cost=None,
    min_runs: int = 4,
    max_runs: int = 6,
    prior_rows: np.ndarray | None = None,
) -> list[PlannedRun]:
    """Cost-weighted D-optimal design over the fraction x machines grid.

    ``cost(fraction, m)`` is the estimated SU of a candidate, defaulting to fraction * rate
    of use (a bigger input and more machines cost more). Runs are added in pairs at one
    fraction with different m, which enforces the constraint that keeps theta_2 and theta_3
    identifiable: at least two distinct m values per input fraction.
    """
    cost = cost or (lambda fraction, m: max(fraction / m, 1e-3) * m)
    width = ernest_features(np.array([0.01]), np.array([2.0])).shape[1]
    rows = np.zeros((0, width)) if prior_rows is None else np.asarray(prior_rows)
    chosen: list[tuple[float, int]] = []

    def row(fraction, m):
        return ernest_features(np.array([fraction]), np.array([float(m)]))

    while len(chosen) < min_runs:
        best, best_score = None, -np.inf
        # Gain is measured from the current design, ridge included, so it stays positive.
        # A negative gain divided by cost would favor the most expensive candidates.
        base = _log_det(rows)
        for fraction in fractions:
            for a, b in combinations(machines, 2):
                if (fraction, a) in chosen or (fraction, b) in chosen:
                    continue
                gain = _log_det(np.vstack([rows, row(fraction, a), row(fraction, b)])) - base
                score = gain / (cost(fraction, a) + cost(fraction, b))
                if score > best_score:
                    best, best_score = (fraction, a, b), score
        if best is None:
            break
        fraction, a, b = best
        chosen += [(fraction, a), (fraction, b)]
        rows = np.vstack([rows, row(fraction, a), row(fraction, b)])

    used = {fraction for fraction, _ in chosen}
    while len(chosen) < max_runs:
        best, best_score = None, -np.inf
        base = _log_det(rows)
        for fraction in used:
            for m in machines:
                if (fraction, m) in chosen:
                    continue
                score = (_log_det(np.vstack([rows, row(fraction, m)])) - base) / cost(fraction, m)
                if score > best_score:
                    best, best_score = (fraction, m), score
        if best is None or best_score <= 0:
            break
        chosen.append(best)
        rows = np.vstack([rows, row(*best)])
    return [PlannedRun("B", fraction, m, note="D-optimal") for fraction, m in chosen]


def information_gain(
    target_draws: np.ndarray,
    candidate_draws: np.ndarray,
    noise_variance: float,
) -> np.ndarray:
    """Expected drop in target variance from observing each candidate, from posterior draws.

    Shapes are (draws, targets) and (draws, candidates), in log space. Under the Gaussian
    approximation, observing c with noise v removes Cov(y, c)^2 / (Var(c) + v) from Var(y),
    summed here over targets.
    """
    centered_y = target_draws - target_draws.mean(axis=0)
    centered_c = candidate_draws - candidate_draws.mean(axis=0)
    denominator = len(target_draws) - 1
    covariance = centered_y.T @ centered_c / denominator
    variance = np.sum(centered_c**2, axis=0) / denominator
    return np.sum(covariance**2, axis=0) / (variance + noise_variance)


def phase_c_next(
    target_draws: np.ndarray,
    candidate_draws: np.ndarray,
    costs: np.ndarray,
    noise_variance: float,
) -> tuple[int, np.ndarray]:
    """Index of the candidate with the largest expected variance reduction per SU."""
    gain = information_gain(target_draws, candidate_draws, noise_variance)
    per_su = gain / np.maximum(costs, 1e-12)
    return int(np.argmax(per_su)), per_su


def phase_c_done(
    runtime_draws: np.ndarray,
    remaining_budget: float,
    next_cost: float,
    target: float = INTERVAL_TARGET,
) -> tuple[bool, str]:
    """Stop when the 90% interval at the target is within +/-20%, or the budget is spent."""
    low, median, high = np.quantile(runtime_draws, [0.05, 0.5, 0.95])
    half_width = max(median - low, high - median) / median
    if half_width <= target:
        return True, f"90% interval is +/-{half_width:.0%}"
    if next_cost > remaining_budget:
        return True, "budget spent"
    return False, f"90% interval is +/-{half_width:.0%}"


@dataclass(frozen=True)
class PlannedGpuRun:
    phase: str
    gpus_per_machine: int
    machines: int
    batch_size: int
    memory_fraction: float = 1.0  # torch.cuda.set_per_process_memory_fraction
    warmup_steps: int = GPU_WARMUP_STEPS
    measured_steps: int = GPU_MEASURED_STEPS
    note: str = ""


def gpu_phase_a(
    batch_sizes: Sequence[int],
    capped_batch: int | None = None,
    capped_fraction: float = 0.5,
) -> list[PlannedGpuRun]:
    """One GPU at three batch sizes, plus a capped run to locate the OOM boundary."""
    runs = [PlannedGpuRun("A", 1, 1, b, note="fit A_k, G_0, theta_c") for b in batch_sizes]
    runs.append(PlannedGpuRun("A", 1, 1, capped_batch or max(batch_sizes), capped_fraction,
                              note="memory cap, the GPU counterpart of memory shrink"))
    return runs


def gpu_phase_b(
    batch_size: int,
    gpus_per_machine: int,
    machine_counts: Sequence[int] = (2, 4),
) -> list[PlannedGpuRun]:
    """Powers of two up to one machine, then multi-machine, to split PCIe from network."""
    runs, count = [], 2
    while count <= gpus_per_machine:
        runs.append(PlannedGpuRun("B", count, 1, batch_size, note="PCIe all-reduce"))
        count *= 2
    for machines in machine_counts:
        runs.append(PlannedGpuRun("B", gpus_per_machine, machines, batch_size,
                                  note="network all-reduce"))
    return runs
