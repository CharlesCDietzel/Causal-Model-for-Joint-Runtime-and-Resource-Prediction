"""Modular fitting: each node on its observed parents, in topological order.

These point estimates initialize the NumPyro model. The six-column stage NNLS survives only
as an initialization for stages without task metrics, and as the Ernest baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, nnls

from tscm.fit.data import FitData
from tscm.mechanisms import tier1, tier2, tier3
from tscm.model import array_namespace, default_params, param_specs

SIGMA_FLOOR = 0.01
LOG_EPS = 1e-12
MAPE_LIMIT = 0.25
THROUGHPUT_RANGE = (1e6, 2e9)  # bytes per second per core implied by theta_1
MAX_EXPANSION = 64.0  # largest plausible in-memory expansion factor omega_1


@dataclass
class ModularFit:
    params: dict[str, np.ndarray]
    # True where a parameter is not identifiable and must stay at its current value.
    pinned: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def fit_modular(data: FitData) -> ModularFit:
    params = default_params(data.job)
    pinned = {
        name: np.zeros(spec.shape, dtype=bool) for name, spec in param_specs(data.job).items()
    }
    fit = ModularFit(params, pinned)
    _fit_tier1(data, fit)
    _fit_working_set(data, fit)
    _fit_spill_and_gc(data, fit)
    _fit_runtime(data, fit)
    _fit_memory_base(data, fit)
    _fit_noise(data, fit)
    fit.diagnostics.update(model_checks(data, fit.params))
    return fit


def _active_features(data: FitData) -> np.ndarray:
    """z columns that vary and are not just log s again (record count usually is)."""
    z, log_s = data.x["z"], data.x["log_s"]
    active = np.zeros(z.shape[1], dtype=bool)
    for j in range(z.shape[1]):
        if np.ptp(z[:, j]) < 1e-9:
            continue
        collinear = np.ptp(log_s) > 1e-9 and abs(np.corrcoef(z[:, j], log_s)[0, 1]) > 0.999
        active[j] = not collinear
    return active


def _robust_loglinear(log_y, log_s, z, weight):
    """Huber regression of log y on log s and z with b >= 0: (a, b, phi)."""
    Z = z.shape[1]
    if np.ptp(log_s) < 1e-9:
        # One input size says nothing about b; keep the linear default.
        b = 1.0
        return float(np.mean(log_y - b * log_s)), b, np.zeros(Z), True

    def residual(theta):
        fitted = theta[0] + theta[1] * log_s + z @ theta[2:]
        return np.sqrt(weight) * (log_y - fitted)

    start = np.concatenate([[np.mean(log_y - log_s), 1.0], np.zeros(Z)])
    lower = np.concatenate([[-np.inf, 0.0], np.full(Z, -np.inf)])
    result = least_squares(residual, start, bounds=(lower, np.inf), loss="huber", f_scale=0.1)
    return float(result.x[0]), float(result.x[1]), result.x[2:], False


def _fit_tier1(data: FitData, fit: ModularFit) -> None:
    active = _active_features(data)
    Z = data.x["z"].shape[1]
    th, pinned = fit.params, fit.pinned
    for volume, observed in (("in", data.V_in), ("sh", data.V_sh), ("out", data.V_out)):
        pinned[f"phi_{volume}"][:, ~active] = True
        for k in range(len(data.job.stages)):
            rows = (data.k == k) & (observed > 0)
            if not rows.any():
                # A stage generating its data reads nothing; the 1 GiB default would invent
                # input for it. Shuffle and output volumes are masked by the stage flags.
                th[f"a_{volume}"][k] = th[f"b_{volume}"][k] = 0.0
                pinned[f"a_{volume}"][k] = pinned[f"b_{volume}"][k] = True
                pinned[f"phi_{volume}"][k] = True
                continue
            r = data.r[rows]
            z = data.x["z"][r][:, active]
            a, b, phi, fixed_b = _robust_loglinear(
                np.log(observed[rows]), data.x["log_s"][r], z, data.weight[r])
            th[f"a_{volume}"][k], th[f"b_{volume}"][k] = a, b
            full = np.zeros(Z)
            full[active] = phi
            th[f"phi_{volume}"][k] = full
            pinned[f"b_{volume}"][k] = fixed_b

    # Total cached bytes, from runs whose cache was not evicted (saturated readings are capped).
    pinned["phi_c"][~active] = True
    per_run = np.zeros(data.n_runs)
    np.maximum.at(per_run, data.r, data.C)
    has_cache = per_run > 0
    if not data.job.cached or not has_cache.any():
        pinned["a_c"] = np.array(True)
        pinned["b_c"] = np.array(True)
        pinned["phi_c"][:] = True
        return
    unsaturated = has_cache & (per_run < 0.95 * data.x["U_e"])
    use = unsaturated if unsaturated.any() else has_cache
    total = per_run[use] * data.x["m"][use] * data.x["e"][use]
    z = data.x["z"][use][:, active]
    a, b, phi, fixed_b = _robust_loglinear(
        np.log(total), data.x["log_s"][use], z, data.weight[use])
    th["a_c"], th["b_c"] = np.array(a), np.array(b)
    th["phi_c"] = np.zeros(Z)
    th["phi_c"][active] = phi
    pinned["b_c"] = np.array(fixed_b)
    fit.diagnostics["cache_from_saturated_runs"] = not unsaturated.any()


def _fit_stage_working_set(v, y, skew, censored, omega2, scale0, skew_varies):
    """omega_0..2 for one stage from task log peaks y, and the residuals of exact tasks."""
    scales = np.array([scale0, 1.0, 1.0])

    def residual(theta):
        w0, w1, w2 = theta * scales
        predicted = np.log(tier2.working_set(v, 1.0, skew, w0, w1, w2) + 1.0)
        gap = y - predicted
        # Spilling tasks are capped, so they only say that W is at least this large.
        return np.where(censored, np.maximum(gap, 0.0), gap)

    ratio = float(np.median(np.exp(y) / np.maximum(v, 1.0)))
    start = np.array([0.1, np.clip(ratio, 1e-3, 0.9 * MAX_EXPANSION), omega2])
    # Fixed buffers on tiny tasks look like a huge expansion factor. The cap sends that
    # memory to omega_0; left free, it would scale with V and fake a memory cliff.
    upper = np.array([np.inf, MAX_EXPANSION, np.inf if skew_varies else 1e-12])
    result = least_squares(residual, start, bounds=(0.0, upper), loss="huber", f_scale=0.3)
    return result.x * scales, residual(result.x)[~censored]


def _fit_working_set(data: FitData, fit: ModularFit) -> None:
    th, pinned = fit.params, fit.pinned
    scale0 = param_specs(data.job)["omega0"].scale
    skew = data.x["skew"]
    skew_varies = np.ptp(skew) > 1e-9
    spilled_row = data.S > 0
    sigma_residuals = []

    for k in range(len(data.job.stages)):
        rows = (data.t_k == k) & (data.t_memory > 0)
        if not rows.any():
            # Map-only stages report no execution memory: their working set is really zero.
            th["omega0"][k] = th["omega1"][k] = th["omega2"][k] = 0.0
            pinned["omega0"][k] = pinned["omega1"][k] = pinned["omega2"][k] = True
            continue
        v, y = data.t_v_in[rows], np.log(data.t_memory[rows])
        task_skew = skew[data.t_r[rows]]
        censored = spilled_row[data.t_row[rows]]
        if not skew_varies:
            pinned["omega2"][k] = True
            th["omega2"][k] = 0.0
        omega, gaps = _fit_stage_working_set(
            v, y, task_skew, censored, th["omega2"][k], scale0, skew_varies)
        th["omega0"][k], th["omega1"][k], th["omega2"][k] = omega
        sigma_residuals.append(gaps)

    if sigma_residuals:
        pooled = np.concatenate(sigma_residuals)
        if pooled.size > 1:
            th["sigma_W"] = np.array(max(float(np.std(pooled)), SIGMA_FLOOR))


def observed_pressure(data: FitData, th: dict) -> np.ndarray:
    """rho per stage execution, from observed V and n but fitted W and C.

    Observed W is capped once a stage spills, which is exactly when rho matters, so the
    fitted working set stands in for it (with the same reasoning for evicted cache).
    """
    xp = array_namespace(*th.values())
    r, k = data.r, data.k
    W = tier2.working_set(data.V_in, data.n, data.x["skew"][r],
                          th["omega0"][k], th["omega1"][k], th["omega2"][k])
    C = cached_per_executor(data, th)[r]
    return xp.asarray(tier2.memory_pressure(data.x["c_e"][r], W, C, data.x["U_e"][r]))


def cached_per_executor(data: FitData, th: dict):
    if not data.job.cached:
        return np.zeros(data.n_runs)
    xp = array_namespace(*th.values())
    log_total = tier1.log_volume(th["a_c"], th["b_c"], th["phi_c"], data.x["log_s"], data.x["z"])
    return xp.exp(log_total) / (data.x["m"] * data.x["e"])


def recompute_fraction(data: FitData, th: dict):
    """Share of cached partitions recomputed, per stage execution."""
    reads = data.job.static["reads_cache"][data.k]
    C = cached_per_executor(data, th)[data.r]
    return reads * tier2.h(C / data.x["U_e"][data.r], th["tau"])


def _fit_spill_and_gc(data: FitData, fit: ModularFit) -> None:
    th, pinned = fit.params, fit.pinned
    K = len(data.job.stages)
    rho = observed_pressure(data, th)
    st = data.job.static
    base = np.where(st["writes_shuffle"][data.k] > 0, data.V_sh, data.V_in)
    spilled = data.S > 0
    fit.diagnostics["spill_observed"] = bool(spilled.any())

    if spilled.any():
        # Without both outcomes the threshold is not identifiable, only the amount is.
        tau_free = (~spilled).any()
        seen = np.isin(np.arange(K), data.k[spilled])
        tau_prior = float(np.log(th["tau"]))

        def residual(theta):
            tau = np.exp(theta[0])
            lam = np.exp(theta[1:])
            amount = np.maximum(tier2.spill_amount(rho, lam[data.k], base), 1.0)
            gap = (np.log(data.S[spilled]) - np.log(amount[spilled]))
            p = np.clip(tier2.spill_probability(rho, tau), 1e-9, 1.0 - 1e-9)
            deviance = -(np.where(spilled, np.log(p), np.log(1.0 - p)))
            # Runs either side of rho = 1 separate perfectly, which would send tau to zero.
            return np.concatenate([gap, np.sqrt(deviance), [0.1 * (theta[0] - tau_prior)]])

        start = np.concatenate([[np.log(th["tau"])], np.log(np.maximum(th["lam"], 1e-3))])
        lower = np.concatenate([[np.log(0.01) if tau_free else start[0] - 1e-9],
                                np.full(K, -12.0)])
        upper = np.concatenate([[np.log(2.0) if tau_free else start[0] + 1e-9], np.full(K, 6.0)])
        result = least_squares(residual, start, bounds=(lower, upper), method="trf")
        th["tau"] = np.array(float(np.exp(result.x[0])))
        lam = np.exp(result.x[1:])
        # Stages that never spilled borrow the job's typical spill rate.
        th["lam"] = np.where(seen, lam, float(np.median(lam[seen])))
        pinned["tau"] = np.array(not tau_free)
        gaps = residual(result.x)[: int(spilled.sum())]
        if gaps.size > 1:
            th["sigma_S"] = np.array(max(float(np.std(gaps)), SIGMA_FLOOR))
    else:
        pinned["tau"] = np.array(True)

    observed = data.g > 0
    if not observed.any():
        th["gamma0"][:] = -8.0
        pinned["gamma0"][:] = True
        for name in ("gamma1", "gamma2", "g_max"):
            pinned[name] = np.array(True)
        return

    M_e = data.x["M_e"][data.r][observed]
    rows_rho = rho[observed]
    rows_k = data.k[observed]
    log_g = np.log(data.g[observed])

    def gc_residual(theta):
        gamma0, gamma1, gamma2, g_max = theta[:K], theta[K], theta[K + 1], theta[K + 2]
        predicted = tier2.gc_fraction(rows_rho, M_e, gamma0[rows_k], gamma1, gamma2, g_max)
        gap = log_g - np.log(np.maximum(predicted, LOG_EPS))
        # Weak pull to the defaults: gamma_1 and g_max are unidentified until GC nears its cap.
        return np.concatenate([gap, [0.02 * (gamma1 - 2.0), 0.02 * gamma2, 0.2 * (g_max - 0.5)]])

    start = np.concatenate([th["gamma0"], [th["gamma1"], th["gamma2"], th["g_max"]]])
    lower = np.concatenate([np.full(K, -30.0), [0.0, -5.0, 0.02]])
    upper = np.concatenate([np.full(K, 15.0), [20.0, 5.0, 0.98]])
    result = least_squares(gc_residual, start, bounds=(lower, upper), method="trf")
    th["gamma0"] = result.x[:K]
    th["gamma1"], th["gamma2"], th["g_max"] = (np.array(v) for v in result.x[K:K + 3])
    gaps = gc_residual(result.x)[: observed.sum()]
    if gaps.size > 1:
        th["sigma_g"] = np.array(max(float(np.std(gaps)), SIGMA_FLOOR))


def _bandwidths(data: FitData):
    x = data.x
    return tier3.effective_bandwidths(
        x["B_disk"], x["B_net"], x["beta_d"], x["beta_n"], x["L_disk"], x["L_net"])


def _speed(data: FitData):
    return tier3.compute_speed(data.x["kappa"], data.x["beta_c"], data.x["L_node"])


def _fit_runtime(data: FitData, fit: ModularFit) -> None:
    """Per-task terms from task durations, then theta_0 with theta_2, theta_3 shared."""
    th = fit.params
    K = len(data.job.stages)
    B_disk, B_net = _bandwidths(data)
    speed = _speed(data)
    r_row = recompute_fraction(data, th)
    defaults = {name: param_specs(data.job)[name].default for name in ("theta4", "theta5")}

    for k in range(K):
        rows = data.t_k == k
        if not rows.any():
            _fit_stage_nnls(data, fit, k)
            continue
        t_r, t_row = data.t_r[rows], data.t_row[rows]
        features = tier3.task_features(
            data.t_v_in[rows], data.t_v_sh[rows], data.t_spill[rows], data.g[t_row], r_row[t_row],
            data.x["m"][t_r], B_disk[t_r], B_net[t_r])
        y = data.t_duration[rows]
        design = np.column_stack([
            features["compute"] * speed[t_r],
            features["recompute"] * speed[t_r],
            features["spill"],
            features["shuffle"],
        ])
        # Weighting by 1/y turns NNLS's absolute error into relative error.
        scaled = design / y[:, None]
        norm = np.maximum(np.linalg.norm(scaled, axis=0), 1e-30)
        solution, _ = nnls(scaled / norm, np.ones_like(y))
        theta1, theta_r, theta4, theta5 = solution / norm
        # A column that was all zeros says nothing, so its default stands instead of zero.
        th["theta1"][k] = theta1
        th["theta_r"][k] = theta_r if np.any(design[:, 1]) else theta1
        fit.pinned["theta_r"][k] = not data.job.stages[k].reads_cache
        th["theta4"][k] = theta4 if np.any(design[:, 2]) else defaults["theta4"]
        th["theta5"][k] = theta5 if np.any(design[:, 3]) else defaults["theta5"]

    _fit_stage_overheads(data, fit, r_row, B_disk, B_net, speed)


def _fit_stage_nnls(data: FitData, fit: ModularFit, k: int) -> None:
    """Six-column NNLS for a stage with no task metrics, such as a historical run."""
    th = fit.params
    rows = data.k == k
    r = data.r[rows]
    B_disk, B_net = _bandwidths(data)
    design = tier3.nnls_features(
        data.V_in[rows], data.V_sh[rows], data.S[rows], data.g[rows], data.n[rows],
        data.x["N"][r], data.x["m"][r], B_disk[r], B_net[r])
    y = data.T_k[rows]
    scaled = design / y[:, None]
    norm = np.maximum(np.linalg.norm(scaled, axis=0), 1e-30)
    solution = nnls(scaled / norm, np.ones_like(y))[0] / norm
    th["theta1"][k], th["theta4"][k], th["theta5"][k] = solution[[1, 4, 5]]
    th["theta_r"][k] = solution[1]


def _fit_stage_overheads(data: FitData, fit: ModularFit, r_row, B_disk, B_net, speed) -> None:
    th = fit.params
    K = len(data.job.stages)
    r, k = data.r, data.k
    terms = tier3.runtime_terms(
        data.V_in, data.V_sh, data.S, data.g, r_row, data.n, data.x["N"][r], data.x["m"][r],
        B_disk[r], B_net[r],
        {name: th[name][k] for name in ("theta1", "theta_r", "theta4", "theta5", "theta0")}
        | {"theta2": 0.0, "theta3": 0.0},
        speed[r])
    wave_time = terms["compute"] + terms["recompute"] + terms["spill"] + terms["shuffle"]
    y = data.T_k - wave_time
    log_m, m = np.log(data.x["m"][r]), data.x["m"][r]
    design = np.column_stack([np.eye(K)[k], log_m, m])
    if np.ptp(m) < 1e-9:
        # With one machine count, theta_2 and theta_3 cannot be told from theta_0.
        design[:, K:] = 0.0
    weight = 1.0 / data.T_k
    solution = nnls(design * weight[:, None], y * weight)[0]
    th["theta0"] = solution[:K]
    th["theta2"], th["theta3"] = np.array(solution[K]), np.array(solution[K + 1])

    predicted = wave_time + th["theta0"][k] + th["theta2"] * log_m + th["theta3"] * m
    fit.diagnostics["stage_log_residuals"] = np.log(data.T_k) - np.log(predicted)
    T_k = np.nan_to_num(data.stage_run_mean(data.T_k))
    T_paths = np.array([
        data.job.longest_path(T_k[i] * data.x["repeats"][i]) for i in range(data.n_runs)])
    th["theta_drv"] = np.array(max(float(np.median(data.T - T_paths)), 0.0))


def _fit_memory_base(data: FitData, fit: ModularFit) -> None:
    th = fit.params
    observed = data.M_peak > 0
    if not observed.any():
        return
    W = tier2.working_set(data.V_in, data.n, data.x["skew"][data.r],
                          th["omega0"][data.k], th["omega1"][data.k], th["omega2"][data.k])
    largest = np.zeros(data.n_runs)
    np.maximum.at(largest, data.r, W * data.x["c_e"][data.r])
    C = cached_per_executor(data, th)
    heap = np.minimum(data.x["U_e"], largest + C)
    th["M_base"] = np.array(max(float(np.median((data.M_peak - heap)[observed])), 0.0))


def _fit_noise(data: FitData, fit: ModularFit) -> None:
    th = fit.params
    residuals = fit.diagnostics.pop("stage_log_residuals")
    run_mean = np.zeros(data.n_runs)
    counts = np.bincount(data.r, minlength=data.n_runs)
    np.add.at(run_mean, data.r, residuals)
    run_mean /= np.maximum(counts, 1)
    within = residuals - run_mean[data.r]
    K_eff = max(float(np.mean(counts)), 1.0)
    dof = max(within.size - data.n_runs, 1)
    sigma_T = float(np.sqrt(np.sum(within**2) / dof))
    th["sigma_T"] = np.array(max(sigma_T, SIGMA_FLOOR))
    # Correlated stage residuals identify the per-run effect.
    between = float(np.var(run_mean)) - sigma_T**2 / K_eff
    th["sigma_eta"] = np.array(max(np.sqrt(max(between, 0.0)), SIGMA_FLOOR))

    volume_gaps = []
    for name, observed in (("in", data.V_in), ("sh", data.V_sh)):
        rows = observed > 0
        fitted = tier1.log_volume(th[f"a_{name}"][data.k[rows]], th[f"b_{name}"][data.k[rows]],
                                  th[f"phi_{name}"][data.k[rows]], data.x["log_s"][data.r[rows]],
                                  data.x["z"][data.r[rows]])
        volume_gaps.append(np.log(observed[rows]) - fitted)
    gaps = np.concatenate(volume_gaps)
    if gaps.size > 2:
        th["sigma_V"] = np.array(max(float(np.std(gaps)), SIGMA_FLOOR))

    if data.t_duration.size > 2:
        B_disk, B_net = _bandwidths(data)
        speed = _speed(data)
        r_row = recompute_fraction(data, th)
        t_k, t_r, t_row = data.t_k, data.t_r, data.t_row
        features = tier3.task_features(
            data.t_v_in, data.t_v_sh, data.t_spill, data.g[t_row], r_row[t_row],
            data.x["m"][t_r], B_disk[t_r], B_net[t_r])
        predicted = (th["theta1"][t_k] * speed[t_r] * features["compute"]
                     + th["theta_r"][t_k] * speed[t_r] * features["recompute"]
                     + th["theta4"][t_k] * features["spill"]
                     + th["theta5"][t_k] * features["shuffle"])
        gap = np.log(data.t_duration) - np.log(np.maximum(predicted, 1e-9))
        th["sigma_task"] = np.array(max(float(np.std(gap)), SIGMA_FLOOR))

    if data.n_runs > 1:
        T_k = np.nan_to_num(data.stage_run_mean(data.T_k))
        paths = np.array([
            data.job.longest_path(T_k[i] * data.x["repeats"][i]) for i in range(data.n_runs)])
        gap = np.log(data.T) - np.log(paths + th["theta_drv"])
        th["sigma_job"] = np.array(max(float(np.std(gap)), SIGMA_FLOOR))


def model_checks(data: FitData, th: dict) -> dict:
    """Stages over the MAPE limit, and theta_1 values that are physically implausible."""
    from tscm.query.forward import Posterior, run

    post = Posterior.from_point(data.job, th, data.hardware)
    out = run(post, data.cfgs, n_samples=1, noise=False)
    predicted = out["T_k"][:, 0]
    observed = data.stage_run_mean(data.T_k)
    with np.errstate(invalid="ignore", divide="ignore"):
        errors = np.abs(predicted - observed) / observed
    mape = np.nanmean(errors, axis=0)
    signatures = [s.signature for s in data.job.stages]
    implausible = []
    for k, signature in enumerate(signatures):
        theta1 = float(th["theta1"][k])
        if theta1 > 0 and not THROUGHPUT_RANGE[0] <= 1.0 / theta1 <= THROUGHPUT_RANGE[1]:
            implausible.append(signature)
    return {
        "stage_mape": dict(zip(signatures, mape.tolist(), strict=True)),
        "stages_over_mape_limit": [s for s, e in zip(signatures, mape, strict=True)
                                   if e > MAPE_LIMIT],
        "implausible_theta1": implausible,
    }


def fit_interference(
    solo_T: np.ndarray,
    shared_T: np.ndarray,
    L_node: np.ndarray,
    L_net: np.ndarray,
    L_disk: np.ndarray,
) -> dict[str, float]:
    """beta_c, beta_n, beta_d from solo/co-located calibration pairs of one configuration.

    Treats the slowdown of a pair as additive in the loads, which holds to first order and
    is enough to anchor the betas. NNLS keeps them non-negative.
    """
    slowdown = np.asarray(shared_T) / np.asarray(solo_T) - 1.0
    design = np.column_stack([L_node, L_net, L_disk])
    norm = np.maximum(np.linalg.norm(design, axis=0), 1e-30)
    beta = nnls(design / norm, slowdown)[0] / norm
    return {"beta_c": float(beta[0]), "beta_n": float(beta[1]), "beta_d": float(beta[2])}
