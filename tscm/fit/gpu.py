"""Fitting and prediction for GPU training jobs.

Same shape as the Spark path: least squares for the initialization, NumPyro for the posterior,
samples for every reported quantity. Memory fails hard here, so Tier 2 yields OOM risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.optimize import least_squares, nnls

from tscm.mechanisms import gpu
from tscm.mechanisms.gpu import GpuConfig

FREE_STEP_PARAMS = ("theta_0", "theta_c", "theta_d", "theta_a", "theta_l")


@dataclass
class GpuRuns:
    """Step-limited profiling runs of one job."""

    cfgs: list[GpuConfig]
    T_step: np.ndarray
    G_peak: np.ndarray
    oom: np.ndarray  # True where the capped run failed; G_peak is then a lower bound

    def __post_init__(self):
        self.T_step = np.asarray(self.T_step, dtype=float)
        self.G_peak = np.asarray(self.G_peak, dtype=float)
        self.oom = np.asarray(self.oom, dtype=bool)


def fit_gpu_modular(runs: GpuRuns) -> dict[str, float]:
    th = dict(gpu.DEFAULT_GPU_PARAMS)
    _fit_memory(runs, th)
    _fit_step_time(runs, th)
    _fit_oom_scale(runs, th)
    return th


def _fit_memory(runs: GpuRuns, th: dict) -> None:
    """G_peak - state = b A + G_0 is linear; capped runs do not constrain it."""
    use = ~runs.oom
    if not use.any():
        return
    plain = [c.with_(ckpt=False) for c in runs.cfgs]
    state = np.array([gpu.working_set(c, 0.0, 0.0) for c in plain])
    batches = np.array([float(c.b) for c in runs.cfgs])
    ckpt = np.array([c.ckpt for c in runs.cfgs], dtype=bool)
    rows = use & ~ckpt
    if np.unique(batches[rows]).size >= 2:
        design = np.column_stack([batches[rows], np.ones(rows.sum())])
        solution = nnls(design, runs.G_peak[rows] - state[rows])[0]
        th["A"], th["G0"] = float(solution[0]), float(solution[1])
    with_ckpt = use & ckpt
    if with_ckpt.any():
        activation = (runs.G_peak[with_ckpt] - state[with_ckpt] - th["G0"]) / batches[with_ckpt]
        th["alpha_ckpt"] = float(np.clip(np.median(activation) / th["A"], 0.01, 1.0))
    residual = np.log(runs.G_peak[use]) - np.log(
        [gpu.working_set(c, th["A"], th["G0"], th["alpha_ckpt"])
         for c, ok in zip(runs.cfgs, use, strict=True) if ok])
    if residual.size > 2:
        th["sigma_G"] = float(max(np.std(residual), 0.005))


def _fit_step_time(runs: GpuRuns, th: dict) -> None:
    use = np.isfinite(runs.T_step) & (runs.T_step > 0)
    cfgs = [c for c, ok in zip(runs.cfgs, use, strict=True) if ok]
    log_y = np.log(runs.T_step[use])
    has_ckpt = any(c.ckpt for c in cfgs)
    multi = any(c.N > 1 for c in cfgs)
    scales = np.array([0.1, 1.0, 1e-3, 1.0, 1e-2, 1.0, 1.0])

    def unpack(values):
        scaled = values * scales
        return {**th, **dict(zip(FREE_STEP_PARAMS, scaled[:5], strict=True)),
                "omega": scaled[5] if multi else th["omega"],
                "kappa_ckpt": scaled[6] if has_ckpt else th["kappa_ckpt"]}

    def residual(values):
        fitted = np.array([gpu.step_time(c, unpack(values)) for c in cfgs])
        return log_y - np.log(np.maximum(fitted, 1e-9))

    start = np.array([th[n] for n in FREE_STEP_PARAMS] + [th["omega"], th["kappa_ckpt"]]) / scales
    upper = np.full(7, np.inf)
    upper[5] = 0.95
    result = least_squares(residual, np.maximum(start, 1e-6), bounds=(0.0, upper), method="trf")
    th.update(unpack(result.x))
    if len(cfgs) > 2:
        th["sigma_step"] = float(max(np.std(residual(result.x)), 0.005))


def _fit_oom_scale(runs: GpuRuns, th: dict) -> None:
    """tau_g from the capped run, the only run near the boundary on purpose."""
    if not runs.oom.any() or runs.oom.all():
        return
    pressure = np.array([
        gpu.working_set(c, th["A"], th["G0"], th["alpha_ckpt"]) / c.R_gpu for c in runs.cfgs])

    def residual(log_tau):
        p = np.clip(gpu.sigmoid((pressure - 1.0) / np.exp(log_tau[0])), 1e-9, 1.0 - 1e-9)
        return np.sqrt(-np.where(runs.oom, np.log(p), np.log(1.0 - p)))

    solved = least_squares(residual, [np.log(th["tau_g"])], bounds=(np.log(1e-3), np.log(0.5)))
    th["tau_g"] = float(np.exp(solved.x[0]))


_HALF_NORMAL = {
    "theta_0": 1.0, "theta_c": 5.0, "theta_d": 0.1, "theta_a": 5.0, "theta_l": 0.5,
    "kappa_ckpt": 1.0, "sigma_step": 0.5, "sigma_G": 0.5,
}


def _model(runs: GpuRuns, init: dict):
    th = {name: numpyro.sample(name, dist.HalfNormal(scale))
          for name, scale in _HALF_NORMAL.items()}
    th["omega"] = numpyro.sample("omega", dist.Uniform(0.0, 0.98))
    A = numpyro.sample("A", dist.LogNormal(jnp.log(max(init["A"], 1.0)), 1.0))
    G0 = numpyro.sample("G0", dist.LogNormal(jnp.log(max(init["G0"], 1.0)), 1.0))
    T = []
    G = []
    for c in runs.cfgs:
        T.append(gpu.step_time(c, th))
        G.append(gpu.working_set(c, A, G0, init["alpha_ckpt"]))
    T, G = jnp.stack(T), jnp.stack(G)
    timed = np.isfinite(runs.T_step) & (runs.T_step > 0)
    numpyro.factor("T_step", jnp.sum(dist.Normal(jnp.log(T[timed]), th["sigma_step"]).log_prob(
        np.log(runs.T_step[timed]))))
    measured = ~runs.oom
    numpyro.factor("G_peak", jnp.sum(dist.Normal(jnp.log(G[measured]), th["sigma_G"]).log_prob(
        np.log(runs.G_peak[measured]))))


@dataclass
class GpuPosterior:
    samples: dict[str, np.ndarray]
    point: dict[str, float]
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        return len(next(iter(self.samples.values())))

    def draw(self, i: int) -> dict[str, float]:
        return {**self.point, **{k: float(v[i]) for k, v in self.samples.items()}}


def fit_gpu_bayes(
    runs: GpuRuns,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
) -> GpuPosterior:
    point = fit_gpu_modular(runs)
    values = {name: jnp.maximum(point[name], 1e-3 * scale) for name, scale in _HALF_NORMAL.items()}
    values |= {"omega": jnp.clip(point["omega"], 0.01, 0.97), "A": point["A"], "G0": point["G0"]}
    kernel = NUTS(_model, init_strategy=init_to_value(values=values), target_accept_prob=0.9)
    parallel = jax.local_device_count() >= num_chains
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="parallel" if parallel else "vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), runs, point, extra_fields=("diverging",))
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    return GpuPosterior(samples, point,
                        {"divergences": int(np.sum(mcmc.get_extra_fields()["diverging"]))})


def predict_gpu(post: GpuPosterior, cfg: GpuConfig, n_samples: int = 1000, seed: int = 0) -> dict:
    """Step time, job runtime, peak and required GPU memory, utilization and OOM risk."""
    rng = np.random.default_rng(seed)
    index = rng.integers(0, post.n_draws, n_samples)
    outcomes: dict[str, list[float]] = {
        name: [] for name in ("T_step", "T", "G_peak", "G_req", "u_gpu", "P_oom")
    }
    for i in index:
        th = post.draw(int(i))
        noise = float(np.exp(rng.normal(0.0, th["sigma_step"])))
        T_step = float(gpu.step_time(cfg, th)) * noise
        G = float(gpu.working_set(cfg, th["A"], th["G0"], th["alpha_ckpt"]))
        G *= float(np.exp(rng.normal(0.0, th["sigma_G"])))
        outcomes["T_step"].append(T_step)
        outcomes["T"].append(th["theta_s"] + cfg.steps * T_step)
        outcomes["G_peak"].append(G)
        outcomes["G_req"].append(gpu.required_memory(G))
        outcomes["u_gpu"].append(float(gpu.utilization(cfg, th)))
        outcomes["P_oom"].append(float(gpu.oom_probability(G, cfg.R_gpu, th["tau_g"])))
    report = {}
    for name, values in outcomes.items():
        low, median, high = np.quantile(values, [0.05, 0.5, 0.95])
        report[name] = {"p50": float(median), "p95": float(high), "lo": float(low),
                        "hi": float(high)}
    report["largest_batch"] = gpu.largest_batch(
        cfg, post.point["A"], post.point["G0"], post.point["alpha_ckpt"])
    return report
