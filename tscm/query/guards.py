"""Extrapolation guards: flags attached to every prediction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tscm.model import Config

SIZE_LIMIT = 10.0


@dataclass(frozen=True)
class Support:
    """What the training runs covered."""

    max_s: float
    machine_types: frozenset[str]
    param_values: dict[str, frozenset[float]]
    stages_seen_over_pressure: frozenset[str] = frozenset()
    residual_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Flag:
    code: str
    message: str


def check(
    cfg: Config,
    support: Support,
    rho: dict[str, float] | None = None,
    residual_applied: bool | None = None,
) -> list[Flag]:
    """Flags for one configuration. ``rho`` maps stage signature to its predicted pressure."""
    flags: list[Flag] = []
    if cfg.s > SIZE_LIMIT * support.max_s:
        ratio = cfg.s / support.max_s
        flags.append(Flag("size_extrapolation",
                          f"input is {ratio:.1f}x the largest profiled input; low confidence"))

    unseen = sorted(
        signature for signature, value in (rho or {}).items()
        if value > 1.0 and signature not in support.stages_seen_over_pressure
    )
    if unseen:
        flags.append(Flag("unseen_regime",
                          f"rho > 1 predicted for {unseen} but never profiled there; "
                          "add one Phase A step 2 run"))

    if cfg.machine.name not in support.machine_types:
        flags.append(Flag("unseen_configuration",
                          f"machine type {cfg.machine.name} is not in the training data"))
    for name, seen in support.param_values.items():
        value = cfg.p.get(name)
        if value is not None and float(value) not in seen and not _inside(value, seen):
            flags.append(Flag("unseen_configuration",
                              f"parameter {name}={value} is outside the training values"))

    if residual_applied is False:
        flags.append(Flag("residual_off",
                          "query is outside the residual learner's range; parametric form only"))
    return flags


def _inside(value: float, seen: frozenset[float]) -> bool:
    return bool(seen) and min(seen) <= value <= max(seen)


def support_from_runs(runs, rho_by_stage: dict[str, np.ndarray] | None = None) -> Support:
    params = {
        column[2:]: frozenset(float(v) for v in runs[column].dropna().unique())
        for column in runs.columns if column.startswith("p_")
    }
    over = frozenset(k for k, v in (rho_by_stage or {}).items() if np.any(np.asarray(v) > 1.0))
    return Support(
        max_s=float(runs["s"].max()),
        machine_types=frozenset(runs["machine_type"].unique()),
        param_values=params,
        stages_seen_over_pressure=over,
    )
