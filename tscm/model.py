"""Shared types: Tier 0 configuration, the stage DAG and the parameter table.

Every fitter (modular, joint, NumPyro) and the forward pass read the same parameter
table, so a parameter is declared once, in ``param_specs``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

MiB = 1024.0**2
GiB = 1024.0**3
SPARK_RESERVED = 300 * MiB
DEFAULT_MEMORY_FRACTION = 0.6
MAX_LOG_S_REF = GiB  # log s is measured from 1 GiB so that a_k stays near log(V / b_k GiB)


def array_namespace(*values):
    """numpy, unless a JAX array or tracer is involved. Lets mechanisms serve both."""
    return jnp if any(isinstance(v, jax.Array) for v in values) else np


def sigmoid(x):
    xp = array_namespace(x)
    return 1.0 / (1.0 + xp.exp(-xp.clip(x, -500.0, 500.0)))


@dataclass(frozen=True)
class Machine:
    """A node type, reduced to the physical quantities the model uses."""

    name: str
    cores: int
    memory: float
    disk_bw: float
    net_bw: float
    rate: float = 1.0  # SU per node-hour on Chameleon, or an hourly price elsewhere


@dataclass(frozen=True)
class Hardware:
    """Parameters shared across jobs, because they describe hardware rather than a job."""

    kappa: Mapping[str, float] = field(default_factory=dict)  # per-type compute speed factor
    beta_d: float = 0.0
    beta_n: float = 0.0
    beta_c: float = 0.0


@dataclass(frozen=True)
class Config:
    """A Tier 0 configuration x, the argument of do(x)."""

    s: float
    m: int
    machine: Machine
    M_e: float
    e: int = 1
    c_e: int = 4
    p: Mapping[str, float] = field(default_factory=dict)
    d: Mapping[str, float] = field(default_factory=dict)
    partition_bytes: float = 128 * MiB
    shuffle_partitions: int = 200
    f_spark: float = DEFAULT_MEMORY_FRACTION
    L_node: float = 0.0
    L_net: float = 0.0
    L_disk: float = 0.0

    @property
    def N(self) -> int:
        return self.m * self.e * self.c_e

    @property
    def U_e(self) -> float:
        return self.f_spark * max(self.M_e - SPARK_RESERVED, 1.0)

    def with_(self, **changes) -> Config:
        return replace(self, **changes)

    def feature(self, name: str) -> float:
        if name in self.p:
            return float(self.p[name])
        if name in self.d:
            return float(self.d[name])
        raise KeyError(f"feature {name!r} is in neither job parameters nor data characteristics")

    def z(self, names: tuple[str, ...]) -> np.ndarray:
        return np.array([np.log1p(self.feature(n)) for n in names], dtype=float)


StageKind = Literal["input", "shuffle", "fixed"]


@dataclass(frozen=True)
class Stage:
    """One node of the stage DAG. Iterations of a repeated stage share this node."""

    signature: str
    kind: StageKind = "input"
    parents: tuple[str, ...] = ()
    stage_type: str = "scan"
    writes_shuffle: bool = False
    writes_output: bool = False
    reads_cache: bool = False
    repeat_param: str | None = None  # job parameter holding the iteration count
    repeat: float = 1.0
    fixed_tasks: int = 1


@dataclass(frozen=True)
class Job:
    job_id: str
    family: str
    stages: tuple[Stage, ...]
    features: tuple[str, ...] = ()
    cached: bool = False

    def __post_init__(self):
        index = {s.signature: i for i, s in enumerate(self.stages)}
        if len(index) != len(self.stages):
            raise ValueError(f"duplicate stage signatures in job {self.job_id}")
        for stage in self.stages:
            unknown = set(stage.parents) - index.keys()
            if unknown:
                raise ValueError(f"stage {stage.signature} has unknown parents {sorted(unknown)}")
        _ = self.topological  # raises on a cycle

    @cached_property
    def index(self) -> dict[str, int]:
        return {s.signature: i for i, s in enumerate(self.stages)}

    @cached_property
    def topological(self) -> tuple[int, ...]:
        order: list[int] = []
        state: dict[int, int] = {}

        def visit(i: int) -> None:
            if state.get(i) == 2:
                return
            if state.get(i) == 1:
                raise ValueError(f"stage DAG of job {self.job_id} has a cycle")
            state[i] = 1
            for parent in self.stages[i].parents:
                visit(self.index[parent])
            state[i] = 2
            order.append(i)

        for i in range(len(self.stages)):
            visit(i)
        return tuple(order)

    def stages_topological(self) -> list[Stage]:
        return [self.stages[i] for i in self.topological]

    def longest_path(self, weights):
        """Longest path through the DAG, for weights of shape (..., K)."""
        xp = array_namespace(weights)
        dist: dict[int, object] = {}
        for i in self.topological:
            best = 0.0
            for parent in self.stages[i].parents:
                best = xp.maximum(best, dist[self.index[parent]])
            dist[i] = weights[..., i] + best
        result = 0.0
        for value in dist.values():
            result = xp.maximum(result, value)
        return result

    def repeats(self, p: Mapping[str, float]) -> np.ndarray:
        return np.array(
            [float(p.get(s.repeat_param, s.repeat)) if s.repeat_param else s.repeat
             for s in self.stages]
        )

    @cached_property
    def static(self) -> dict[str, np.ndarray]:
        """Per-stage structure as arrays, so a forward pass stays free of Python branching."""
        return {
            "is_input": np.array([s.kind == "input" for s in self.stages]),
            "is_shuffle": np.array([s.kind == "shuffle" for s in self.stages]),
            "fixed_tasks": np.array([float(s.fixed_tasks) for s in self.stages]),
            "writes_shuffle": np.array([s.writes_shuffle for s in self.stages], dtype=float),
            "writes_output": np.array([s.writes_output for s in self.stages], dtype=float),
            "reads_cache": np.array([s.reads_cache for s in self.stages], dtype=float),
        }


Constraint = Literal["real", "nonneg", "unit"]


@dataclass(frozen=True)
class ParamSpec:
    """How one parameter is shaped, constrained, scaled and defaulted.

    ``scale`` is the half-normal scale for non-negative parameters and the normal scale
    for real ones. ``default`` is used when a parameter cannot be identified from the
    runs at hand, for example spill terms before any run has reached rho > 1.
    """

    shape: tuple[int, ...]
    constraint: Constraint
    scale: float
    default: float
    loc: float = 0.0

    @property
    def per_stage(self) -> bool:
        return len(self.shape) >= 1


def param_specs(job: Job) -> dict[str, ParamSpec]:
    K, Z = len(job.stages), len(job.features)
    log_bytes = float(np.log(GiB))
    specs: dict[str, ParamSpec] = {}
    for volume in ("in", "sh", "out"):
        specs[f"a_{volume}"] = ParamSpec((K,), "real", 8.0, log_bytes, loc=log_bytes)
        specs[f"b_{volume}"] = ParamSpec((K,), "nonneg", 1.5, 1.0)
        specs[f"phi_{volume}"] = ParamSpec((K, Z), "real", 1.0, 0.0)
    specs.update(
        omega0=ParamSpec((K,), "nonneg", 256 * MiB, 0.0),
        omega1=ParamSpec((K,), "nonneg", 4.0, 1.0),
        omega2=ParamSpec((K,), "nonneg", 1.0, 0.0),
        lam=ParamSpec((K,), "nonneg", 2.0, 1.0),
        gamma0=ParamSpec((K,), "real", 3.0, -4.0, loc=-3.0),
        theta0=ParamSpec((K,), "nonneg", 10.0, 0.0),
        theta1=ParamSpec((K,), "nonneg", 1e-6, 2e-8),
        theta4=ParamSpec((K,), "nonneg", 3.0, 1.0),
        theta5=ParamSpec((K,), "nonneg", 3.0, 1.0),
        theta_r=ParamSpec((K,), "nonneg", 1e-6, 2e-8),
        # Shared across a job's stages (decision 3).
        tau=ParamSpec((), "nonneg", 0.3, 0.1),
        g_max=ParamSpec((), "unit", 1.0, 0.5),
        gamma1=ParamSpec((), "nonneg", 4.0, 2.0),
        gamma2=ParamSpec((), "real", 2.0, 0.0),
        theta2=ParamSpec((), "nonneg", 10.0, 0.0),
        theta3=ParamSpec((), "nonneg", 1.0, 0.0),
        theta_drv=ParamSpec((), "nonneg", 30.0, 0.0),
        M_base=ParamSpec((), "nonneg", 2 * GiB, 512 * MiB),
        a_c=ParamSpec((), "real", 8.0, log_bytes, loc=log_bytes),
        b_c=ParamSpec((), "nonneg", 1.5, 1.0),
        phi_c=ParamSpec((Z,), "real", 1.0, 0.0),
    )
    for sigma, default in (
        ("sigma_V", 0.05),
        ("sigma_W", 0.15),
        ("sigma_S", 0.25),
        ("sigma_g", 0.25),
        ("sigma_T", 0.08),
        ("sigma_task", 0.25),
        ("sigma_eta", 0.05),
        ("sigma_job", 0.05),
    ):
        specs[sigma] = ParamSpec((), "nonneg", 0.5, default)
    return specs


def default_params(job: Job) -> dict[str, np.ndarray]:
    return {
        name: np.full(spec.shape, spec.default, dtype=float)
        for name, spec in param_specs(job).items()
    }


def config_arrays(job: Job, cfg: Config, hardware: Hardware | None = None) -> dict[str, np.ndarray]:
    """A Config as floats, the form the jitted forward pass and the fitters take."""
    hardware = hardware or Hardware()
    return {
        "log_s": np.log(cfg.s / MAX_LOG_S_REF),
        "m": float(cfg.m),
        "e": float(cfg.e),
        "c_e": float(cfg.c_e),
        "N": float(cfg.N),
        "M_e": float(cfg.M_e),
        "U_e": float(cfg.U_e),
        "f_spark": float(cfg.f_spark),
        "B_disk": float(cfg.machine.disk_bw),
        "B_net": float(cfg.machine.net_bw),
        "partition_bytes": float(cfg.partition_bytes),
        "shuffle_partitions": float(cfg.shuffle_partitions),
        "skew": float(cfg.d.get("skew", 0.0)),
        "kappa": float(hardware.kappa.get(cfg.machine.name, 1.0)),
        "beta_d": float(hardware.beta_d),
        "beta_n": float(hardware.beta_n),
        "beta_c": float(hardware.beta_c),
        "L_node": float(cfg.L_node),
        "L_net": float(cfg.L_net),
        "L_disk": float(cfg.L_disk),
        "z": cfg.z(job.features),
        "repeats": job.repeats(cfg.p),
    }


def stack_configs(xs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([x[key] for x in xs]) for key in xs[0]}
