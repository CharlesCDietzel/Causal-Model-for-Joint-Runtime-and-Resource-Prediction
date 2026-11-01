"""The Monte Carlo forward pass: do(x), then push parameters and noise through the graph."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from tscm.mechanisms import tier1, tier2, tier3
from tscm.model import Config, Hardware, Job, config_arrays, stack_configs

NOISE_KEYS = ("V_in", "V_sh", "V_out", "S", "g", "T")


def structural_pass(job: Job, x, th, noise=None, tasks=None):
    """One forward pass through the tiers for a single parameter draw.

    With ``noise=None`` this is the deterministic mean path, which the joint refinement
    differentiates. Passing ``noise`` adds the node noise and the zero-inflated spill draw.
    ``tasks`` overrides the predicted task counts: their ceil makes the loss discontinuous
    in V, which breaks line searches, and they are known exactly at observed configs.
    """
    st = job.static
    zero = 0.0 * x["log_s"]
    e = noise or {}
    eps = lambda key: e.get(key, zero)  # noqa: E731

    # Tier 1
    V_in = jnp.exp(tier1.log_volume(th["a_in"], th["b_in"], th["phi_in"], x["log_s"], x["z"])
                   + eps("V_in"))
    V_sh = st["writes_shuffle"] * jnp.exp(
        tier1.log_volume(th["a_sh"], th["b_sh"], th["phi_sh"], x["log_s"], x["z"]) + eps("V_sh"))
    V_out = st["writes_output"] * jnp.exp(
        tier1.log_volume(th["a_out"], th["b_out"], th["phi_out"], x["log_s"], x["z"])
        + eps("V_out"))
    n = tasks if tasks is not None else tier1.tasks(
        V_in, x["partition_bytes"], x["shuffle_partitions"],
        st["is_input"], st["is_shuffle"], st["fixed_tasks"])
    C_total = float(job.cached) * jnp.exp(
        tier1.log_volume(th["a_c"], th["b_c"], th["phi_c"], x["log_s"], x["z"]))

    # Tier 2
    W = tier2.working_set(V_in, n, x["skew"], th["omega0"], th["omega1"], th["omega2"])
    C = tier2.cache_per_executor(C_total, x["m"], x["e"])
    rho = tier2.memory_pressure(x["c_e"], W, C, x["U_e"])
    base = tier2.spill_base(V_in, V_sh, st["writes_shuffle"])
    if noise is None:
        S = th["lam"] * base * tier2.h(rho, th["tau"])
    else:
        spills = e["u_spill"] < tier2.spill_probability(rho, th["tau"])
        S = spills * tier2.spill_amount(rho, th["lam"], base) * jnp.exp(eps("S"))
    g = tier2.gc_fraction(rho, x["M_e"], th["gamma0"], th["gamma1"], th["gamma2"], th["g_max"])
    g = jnp.clip(g * jnp.exp(eps("g")), 0.0, 0.95)
    # Share of cached partitions recomputed per iteration, from cache pressure.
    r = st["reads_cache"] * tier2.h(C / x["U_e"], th["tau"])

    # Tier 3
    B_disk, B_net = tier3.effective_bandwidths(
        x["B_disk"], x["B_net"], x["beta_d"], x["beta_n"], x["L_disk"], x["L_net"])
    speed = tier3.compute_speed(x["kappa"], x["beta_c"], x["L_node"])
    terms = tier3.runtime_terms(V_in, V_sh, S, g, r, n, x["N"], x["m"], B_disk, B_net, th, speed)
    run_effect = jnp.exp(e.get("eta", zero) + eps("T"))
    T_k = sum(terms.values()) * run_effect
    u_k = tier3.cpu_utilization(terms, T_k, n, x["N"])

    weighted = T_k * x["repeats"]
    T = th["theta_drv"] + job.longest_path(weighted)
    return {
        "T": T,
        "T_k": T_k,
        "M_peak": tier3.peak_memory(th["M_base"], x["U_e"], x["c_e"], W, C),
        "M_req_zero_spill": tier3.zero_spill_memory(x["c_e"], W, C, x["f_spark"]),
        "S_total": jnp.sum(S * x["repeats"]),
        "u": jnp.sum(u_k * weighted) / jnp.sum(weighted),
        "V_in": V_in,
        "V_sh": V_sh,
        "V_out": V_out,
        "n": n,
        "W": W,
        "C": C * jnp.ones(()),
        "rho": rho,
        "S": S,
        "g": g,
        "r": r,
        "u_k": u_k,
        **{f"term_{name}": value * run_effect for name, value in terms.items()},
    }


def draw_noise(job: Job, th, key):
    K = len(job.stages)
    keys = jax.random.split(key, len(NOISE_KEYS) + 2)
    sigma = {"V_in": "sigma_V", "V_sh": "sigma_V", "V_out": "sigma_V",
             "S": "sigma_S", "g": "sigma_g", "T": "sigma_T"}
    noise = {
        name: th[sigma[name]] * jax.random.normal(keys[i], (K,))
        for i, name in enumerate(NOISE_KEYS)
    }
    noise["u_spill"] = jax.random.uniform(keys[-2], (K,))
    noise["eta"] = th["sigma_eta"] * jax.random.normal(keys[-1], ())
    return noise


_COMPILED: dict[tuple[Job, bool], object] = {}


def _compiled(job: Job, stochastic: bool):
    """Jitted pass over (configs, parameter draws). Keyed by job, whose DAG is static."""
    cached = _COMPILED.get((job, stochastic))
    if cached is not None:
        return cached

    def one(x, th, key):
        return structural_pass(job, x, th, draw_noise(job, th, key) if stochastic else None)

    over_draws = jax.vmap(one, in_axes=(None, 0, 0))
    fn = jax.jit(jax.vmap(over_draws, in_axes=(0, None, None)))
    _COMPILED[(job, stochastic)] = fn
    return fn


@dataclass
class Posterior:
    """Parameter draws of shape (draws, ...), from NUTS or a single point estimate."""

    job: Job
    samples: dict[str, np.ndarray]
    hardware: Hardware = field(default_factory=Hardware)
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        return len(next(iter(self.samples.values())))

    @classmethod
    def from_point(cls, job: Job, params: dict, hardware: Hardware | None = None, **diagnostics):
        samples = {k: np.asarray(v, dtype=float)[None] for k, v in params.items()}
        return cls(job, samples, hardware or Hardware(), diagnostics)

    def point(self) -> dict[str, np.ndarray]:
        return {k: np.median(v, axis=0) for k, v in self.samples.items()}


def run(
    post: Posterior,
    cfgs: list[Config],
    n_samples: int = 1000,
    seed: int = 0,
    noise: bool = True,
) -> dict[str, np.ndarray]:
    """All nodes for each config, shaped (configs, n_samples, ...).

    The same seed reuses the same draws for every config, which gives a paired comparison
    for do(x) vs do(x').
    """
    rng = np.random.default_rng(seed)
    index = rng.integers(0, post.n_draws, size=n_samples)
    draws = {k: v[index] for k, v in post.samples.items()}
    keys = jax.random.split(jax.random.PRNGKey(seed), n_samples)
    x = stack_configs([config_arrays(post.job, cfg, post.hardware) for cfg in cfgs])
    return {k: np.asarray(v) for k, v in _compiled(post.job, noise)(x, draws, keys).items()}


def summarize(values: np.ndarray) -> dict[str, float | np.ndarray]:
    """P50, P95 and the 90% interval over the draw axis (the first one)."""
    q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95], axis=0)
    return {"p50": q50, "p95": q95, "lo": q05, "hi": q95}
