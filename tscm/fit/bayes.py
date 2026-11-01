"""The NumPyro model, the single fitting path for v1 (decision 4).

The likelihood factorizes along the graph: each node is conditioned on its observed parents,
with task-level likelihoods for the working set and per-task costs (decision 3). The modular
fit supplies the initialization and says which parameters the data cannot identify.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.stats import norm
from numpyro.diagnostics import summary
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, init_to_value

from tscm.fit.data import FitData
from tscm.fit.modular import ModularFit, cached_per_executor, fit_modular, recompute_fraction
from tscm.fit.transfer import SHARED, FamilyPrior
from tscm.mechanisms import tier1, tier2, tier3
from tscm.model import param_specs
from tscm.query.forward import Posterior

LOG_FLOOR = 1e-30


class SoftplusHalfNormal(dist.HalfNormal):
    """HalfNormal unconstrained by softplus rather than log.

    Terms like theta_2 can sit near zero. In log space their curvature then changes by
    orders of magnitude between theta near zero and theta near its scale, which forces tiny
    NUTS steps; softplus stays linear away from zero. Sites are standardized by the spec's
    scale, so that "away from zero" means the same thing for every parameter.
    """

    support = constraints.softplus_positive


# Pins that mean "these runs cannot see it", which a family prior can answer. Other pins are
# structural (a stage with no shuffle write) and hold whatever the prior says.
RELEASED_BY_PRIOR = {"tau", "b_in", "b_sh", "b_out", "b_c", "gamma1", "gamma2", "g_max"}


def _sample_parameters(data: FitData, fit: ModularFit, prior: FamilyPrior | None):
    """Sample the free elements of every parameter; the rest stay at the modular value."""
    job = data.job
    th = {}
    for name, spec in param_specs(job).items():
        value = jnp.asarray(fit.params[name], dtype=float)
        free = ~np.asarray(fit.pinned[name], dtype=bool)
        if spec.constraint == "nonneg" and name.startswith("sigma_"):
            th[name] = numpyro.sample(name, SoftplusHalfNormal(spec.scale))
            continue
        stage_scoped = spec.shape == (len(job.stages),)
        types = [s.stage_type for s in job.stages] if stage_scoped else None
        if prior is not None and name in RELEASED_BY_PRIOR and not stage_scoped:
            free = free | (prior.lookup(name, SHARED) is not None)
        elif prior is not None and name in RELEASED_BY_PRIOR:
            free = free | np.array([prior.lookup(name, t) is not None for t in types])
        if not free.any():
            th[name] = numpyro.deterministic(name, value)
            continue

        indices = np.flatnonzero(free.ravel())
        if stage_scoped:
            stats = [prior.lookup(name, types[i]) if prior else None for i in indices]
        else:
            shared = prior.lookup(name, SHARED) if prior and spec.shape == () else None
            stats = [shared] * indices.size
        informed = np.array([s is not None for s in stats], dtype=bool)

        flat = value.ravel()
        if informed.any():
            loc = np.array([s[0] for s in stats if s is not None])
            scale = np.array([s[1] for s in stats if s is not None])
            if spec.constraint == "real":
                family = dist.Normal(loc, scale)
            elif spec.constraint == "unit":
                family = dist.TruncatedNormal(loc, scale, low=0.02, high=0.98)
            else:
                family = dist.LogNormal(loc, scale)
            flat = flat.at[indices[informed]].set(numpyro.sample(f"{name}__prior", family))
        if (~informed).any():
            size = int((~informed).sum())
            if spec.constraint == "real":
                family = dist.Normal(spec.loc, spec.scale).expand([size])
            elif spec.constraint == "unit":
                family = dist.Uniform(0.02, 0.98).expand([size])
            else:
                family = SoftplusHalfNormal(1.0).expand([size])
            sampled = numpyro.sample(f"{name}__free", family)
            if spec.constraint == "nonneg":
                sampled = sampled * spec.scale
            flat = flat.at[indices[~informed]].set(sampled)
        th[name] = numpyro.deterministic(name, flat.reshape(spec.shape))
    return th


def _gaussian_log(name, log_y, mean, sigma, weight):
    numpyro.factor(name, jnp.sum(weight * dist.Normal(mean, sigma).log_prob(log_y)))


def model(data: FitData, fit: ModularFit, prior: FamilyPrior | None = None):
    th = _sample_parameters(data, fit, prior)
    job, x, r, k = data.job, data.x, data.r, data.k
    row_weight = data.weight[r]

    # Tier 1
    # One observation per (run, stage): iterations repeat the same volume rather than
    # measuring it again, and counting them separately would overstate the evidence.
    for volume, observed in (("in", data.V_in), ("sh", data.V_sh), ("out", data.V_out)):
        per_run = np.nan_to_num(data.stage_run_mean(observed))
        run_rows, stage_rows = np.nonzero(per_run > 0)
        if run_rows.size:
            mean = tier1.log_volume(
                th[f"a_{volume}"][stage_rows], th[f"b_{volume}"][stage_rows],
                th[f"phi_{volume}"][stage_rows], x["log_s"][run_rows], x["z"][run_rows])
            _gaussian_log(f"V_{volume}", np.log(per_run[run_rows, stage_rows]), mean,
                          th["sigma_V"], data.weight[run_rows])

    per_run_cache = np.zeros(data.n_runs)
    np.maximum.at(per_run_cache, r, data.C)
    unsaturated = (per_run_cache > 0) & (per_run_cache < 0.95 * x["U_e"])
    if job.cached and unsaturated.any():
        total = per_run_cache[unsaturated] * x["m"][unsaturated] * x["e"][unsaturated]
        mean = tier1.log_volume(th["a_c"], th["b_c"], th["phi_c"], x["log_s"][unsaturated],
                                x["z"][unsaturated])
        _gaussian_log("C_total", np.log(total), mean, th["sigma_V"], data.weight[unsaturated])

    # Tier 2: working set per task, with spilling tasks as lower bounds (decision 2).
    spilled = data.S > 0
    has_memory = data.t_memory > 0
    if has_memory.any():
        t_k, t_r = data.t_k[has_memory], data.t_r[has_memory]
        predicted = tier2.working_set(
            data.t_v_in[has_memory], 1.0, x["skew"][t_r],
            th["omega0"][t_k], th["omega1"][t_k], th["omega2"][t_k])
        mean = jnp.log(jnp.maximum(predicted, 1.0))
        log_y = np.log(data.t_memory[has_memory])
        censored = spilled[data.t_row[has_memory]]
        weight = data.weight[t_r]
        exact = dist.Normal(mean, th["sigma_W"]).log_prob(log_y)
        lower = norm.logsf((log_y - mean) / th["sigma_W"])
        numpyro.factor("W", jnp.sum(weight * jnp.where(censored, lower, exact)))

    W = tier2.working_set(data.V_in, data.n, x["skew"][r],
                          th["omega0"][k], th["omega1"][k], th["omega2"][k])
    C = cached_per_executor(data, th)
    rho = tier2.memory_pressure(x["c_e"][r], W, C[r], x["U_e"][r])

    # Spill is zero-inflated (decision 1).
    logits = (rho - 1.0) / th["tau"]
    numpyro.factor("S_positive", jnp.sum(
        row_weight * dist.Bernoulli(logits=logits).log_prob(spilled.astype(float))))
    if spilled.any():
        writes = job.static["writes_shuffle"][k]
        base = np.where(writes > 0, data.V_sh, data.V_in)
        amount = tier2.spill_amount(rho, th["lam"][k], base)
        _gaussian_log("S_amount", np.log(data.S[spilled]),
                      jnp.log(jnp.maximum(amount[spilled], 1.0)), th["sigma_S"],
                      row_weight[spilled])

    has_gc = data.g > 0
    if has_gc.any():
        predicted = tier2.gc_fraction(rho, x["M_e"][r], th["gamma0"][k], th["gamma1"],
                                      th["gamma2"], th["g_max"])
        _gaussian_log("g", np.log(data.g[has_gc]),
                      jnp.log(jnp.maximum(predicted[has_gc], LOG_FLOOR)), th["sigma_g"],
                      row_weight[has_gc])

    # Tier 3: per-task costs, then stage wall clock, then job runtime.
    B_disk, B_net = tier3.effective_bandwidths(
        x["B_disk"], x["B_net"], x["beta_d"], x["beta_n"], x["L_disk"], x["L_net"])
    speed = tier3.compute_speed(x["kappa"], x["beta_c"], x["L_node"])
    r_row = recompute_fraction(data, th)
    if data.t_duration.size:
        t_k, t_r, t_row = data.t_k, data.t_r, data.t_row
        features = tier3.task_features(
            data.t_v_in, data.t_v_sh, data.t_spill, data.g[t_row], r_row[t_row], x["m"][t_r],
            B_disk[t_r], B_net[t_r])
        predicted = (th["theta1"][t_k] * speed[t_r] * features["compute"]
                     + th["theta_r"][t_k] * speed[t_r] * features["recompute"]
                     + th["theta4"][t_k] * features["spill"]
                     + th["theta5"][t_k] * features["shuffle"])
        _gaussian_log("task", np.log(data.t_duration), jnp.log(jnp.maximum(predicted, LOG_FLOOR)),
                      th["sigma_task"], data.weight[t_r])

    # Centered: each run has many stage rows, so eta is well identified, and the non-centered
    # form would bend it into a funnel against sigma_eta and collapse the NUTS step size.
    eta = numpyro.sample("eta", dist.Normal(0.0, th["sigma_eta"]).expand([data.n_runs]))
    stage_params = {name: th[name][k]
                    for name in ("theta0", "theta1", "theta_r", "theta4", "theta5")}
    terms = tier3.runtime_terms(
        data.V_in, data.V_sh, data.S, data.g, r_row, data.n, x["N"][r], x["m"][r], B_disk[r],
        B_net[r], stage_params | {"theta2": th["theta2"], "theta3": th["theta3"]}, speed[r])
    predicted = sum(terms.values())
    _gaussian_log("T_k", np.log(data.T_k), jnp.log(jnp.maximum(predicted, LOG_FLOOR)) + eta[r],
                  th["sigma_T"], row_weight)

    T_k = np.nan_to_num(data.stage_run_mean(data.T_k))
    paths = np.array([job.longest_path(T_k[i] * x["repeats"][i]) for i in range(data.n_runs)])
    _gaussian_log("T", np.log(data.T), jnp.log(th["theta_drv"] + paths), th["sigma_job"],
                  data.weight)

    observed_peak = data.M_peak > 0
    if observed_peak.any():
        largest = jax.ops.segment_max(x["c_e"][r] * W, jnp.asarray(r), num_segments=data.n_runs)
        peak = th["M_base"] + jnp.minimum(x["U_e"], largest + C)
        _gaussian_log("M_peak", np.log(data.M_peak[observed_peak]),
                      jnp.log(peak[observed_peak]), th["sigma_job"], data.weight[observed_peak])


def _initial_values(fit: ModularFit, data: FitData, prior: FamilyPrior | None) -> dict:
    """Modular estimates on the sites the model samples, kept off the support boundary."""
    values = {"eta": jnp.zeros(data.n_runs)}
    substitution = numpyro.handlers.seed(
        lambda: _sample_parameters(data, fit, prior), rng_seed=0)
    trace = numpyro.handlers.trace(substitution).get_trace()
    for site, entry in trace.items():
        if entry["type"] != "sample":
            continue
        name, _, kind = site.partition("__")
        spec = param_specs(data.job)[name]
        if not kind:
            values[site] = jnp.maximum(jnp.asarray(fit.params[name]), 1e-3 * spec.scale)
            continue
        current = jnp.asarray(fit.params[name]).ravel()
        free = ~np.asarray(fit.pinned[name], dtype=bool).ravel()
        size = entry["value"].size
        chosen = current[np.flatnonzero(free)[:size]] if free.sum() >= size else \
            jnp.full(size, spec.default)
        if spec.constraint == "nonneg":
            chosen = jnp.maximum(chosen, 1e-3 * spec.scale)
            if kind == "free":
                chosen = chosen / spec.scale
        elif spec.constraint == "unit":
            chosen = jnp.clip(chosen, 0.03, 0.97)
        values[site] = chosen.reshape(entry["value"].shape)
    return values


def fit_bayes(
    data: FitData,
    fit: ModularFit | None = None,
    prior: FamilyPrior | None = None,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    max_tree_depth: int = 7,
) -> Posterior:
    """NUTS over the job's parameters: 4 chains of 1,000 draws by default."""
    fit = fit or fit_modular(data)
    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=_initial_values(fit, data, prior)),
        target_accept_prob=0.8,
        max_tree_depth=max_tree_depth,
        # a_k with b_k and theta_2 with theta_3 are strongly correlated; a diagonal mass
        # matrix saturates the tree depth on them.
        dense_mass=True,
    )
    parallel = jax.local_device_count() >= num_chains
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="parallel" if parallel else "vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), data, fit, prior, extra_fields=("diverging",))

    by_chain = mcmc.get_samples(group_by_chain=True)
    names = list(param_specs(data.job))
    samples = {}
    for name in names:
        values = by_chain[name]
        samples[name] = np.asarray(values).reshape(-1, *values.shape[2:])
    checks = {name: values for name, values in by_chain.items()
              if "__" in name or name.startswith("sigma_")}
    stats = summary(checks, group_by_chain=True)
    r_hat = [float(np.nanmax(s["r_hat"])) for s in stats.values()]
    n_eff = [float(np.nanmin(s["n_eff"])) for s in stats.values()]
    diagnostics = {
        **{k: v for k, v in fit.diagnostics.items()},
        "max_r_hat": max(r_hat),
        "min_n_eff": min(n_eff),
        "divergences": int(np.sum(mcmc.get_extra_fields()["diverging"])),
    }
    return Posterior(data.job, samples, data.hardware, diagnostics)
