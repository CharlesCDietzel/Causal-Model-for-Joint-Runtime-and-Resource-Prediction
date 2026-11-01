"""GPU jobs: working set, out-of-memory risk, step time, and the RAPIDS extension terms."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from tscm.model import GiB, array_namespace, sigmoid

CHECKPOINT_RECOMPUTE = 0.33
MEMORY_MARGIN = 1.10  # G_req keeps rho_gpu <= 1 with a 10% margin


# Bytes per parameter for weights, gradients and optimizer state (Adam keeps two moments).
BYTES_PER_PARAMETER = {
    "mixed": (2.0, 2.0, 12.0),
    "fp32": (4.0, 4.0, 8.0),
    "fp64": (8.0, 8.0, 16.0),
}


@dataclass(frozen=True)
class GpuConfig:
    """Tier 0 for a training job. Dataset size sets steps; per-step inputs set time and memory."""

    dataset_size: float
    epochs: float
    b: int
    P: float
    phi: float  # FLOPs per sample
    n_g: int
    m: int
    R_gpu: float
    F_gpu: float
    B_pcie: float
    B_net: float
    precision: str = "mixed"
    shard: str = "ddp"
    ckpt: bool = False
    n_load: int = 4

    @property
    def N(self) -> int:
        return self.m * self.n_g

    @property
    def B_ic(self) -> float:
        """PCIe within a machine, network once N spans more than one (no NVLink on Chameleon)."""
        return self.B_pcie if self.m == 1 else self.B_net

    @property
    def steps(self) -> float:
        return self.epochs * self.dataset_size / (self.b * self.N)

    def with_(self, **changes) -> GpuConfig:
        return replace(self, **changes)


def sharding_divisor(shard: str, N: int, betas: tuple[float, float, float]) -> float:
    """Effective z so that P (beta_w + beta_g + beta_o) / z is the per-GPU state.

    ZeRO stage 1 shards optimizer state, stage 2 also gradients, stage 3 and FSDP everything.
    """
    beta_w, beta_g, beta_o = betas
    total = sum(betas)
    held = {
        "ddp": total,
        "zero1": beta_w + beta_g + beta_o / N,
        "zero2": beta_w + (beta_g + beta_o) / N,
        "zero3": total / N,
        "fsdp": total / N,
    }
    if shard not in held:
        raise ValueError(f"unknown sharding mode {shard!r}")
    return total / held[shard]


def working_set(cfg: GpuConfig, A, G0, alpha_ckpt=0.5):
    """G_k = P (beta_w + beta_g + beta_o) / z + b A_k + G_0, per phase if A is an array."""
    betas = BYTES_PER_PARAMETER[cfg.precision]
    z = sharding_divisor(cfg.shard, cfg.N, betas)
    activation = A * (alpha_ckpt if cfg.ckpt else 1.0)
    return cfg.P * sum(betas) / z + cfg.b * activation + G0


def oom_probability(G, R_gpu, tau_g):
    return sigmoid((G / R_gpu - 1.0) / tau_g)


def step_time_terms(cfg: GpuConfig, th: dict) -> dict:
    """The compute/load bottleneck and the communication bracket of T_step, in seconds."""
    xp = array_namespace(*th.values())
    beta_g = BYTES_PER_PARAMETER[cfg.precision][1]
    N = float(cfg.N)
    compute = th["theta_c"] * cfg.b * cfg.phi / cfg.F_gpu * (
        1.0 + th["kappa_ckpt"] * float(cfg.ckpt))
    load = th["theta_d"] * cfg.b / cfg.n_load
    allreduce = th["theta_a"] * 2.0 * (N - 1.0) / N * cfg.P * beta_g / cfg.B_ic
    latency = th["theta_l"] * np.log(N)
    return {
        "setup": th["theta_0"],
        "compute": compute,
        "load": load,
        "busy": xp.maximum(compute, load),
        "communication": (1.0 - th["omega"]) * (allreduce + latency),
    }


def step_time(cfg: GpuConfig, th: dict):
    terms = step_time_terms(cfg, th)
    return terms["setup"] + terms["busy"] + terms["communication"]


def job_runtime(cfg: GpuConfig, th: dict):
    return th["theta_s"] + cfg.steps * step_time(cfg, th)


def utilization(cfg: GpuConfig, th: dict):
    return step_time_terms(cfg, th)["compute"] / step_time(cfg, th)


def required_memory(G_peak: float) -> float:
    return MEMORY_MARGIN * G_peak


def largest_batch(cfg: GpuConfig, A: float, G0: float, alpha_ckpt: float = 0.5) -> int:
    """The largest batch fitting a fixed GPU: G_k <= R_gpu with the same 10% margin."""
    betas = BYTES_PER_PARAMETER[cfg.precision]
    z = sharding_divisor(cfg.shard, cfg.N, betas)
    state = cfg.P * sum(betas) / z
    activation = A * (alpha_ckpt if cfg.ckpt else 1.0)
    return max(int((cfg.R_gpu / MEMORY_MARGIN - state - G0) // activation), 0)


def rapids_transfer_term(theta6, V, n, B_pcie):
    """Host-to-device transfer added inside the waves bracket (extension after v1)."""
    return theta6 * V / (n * B_pcie)


DEFAULT_GPU_PARAMS = {
    "theta_0": 0.01,
    "theta_c": 1.0,
    "theta_d": 1e-3,
    "theta_a": 1.0,
    "theta_l": 1e-3,
    "theta_s": 30.0,
    "omega": 0.2,
    "kappa_ckpt": CHECKPOINT_RECOMPUTE,
    "A": 16e6,
    "G0": 1 * GiB,
    "alpha_ckpt": 0.5,
    "tau_g": 0.03,
    "sigma_step": 0.05,
    "sigma_G": 0.03,
}
