"""Tier 3: stage runtime, its decomposition, and the memory outcomes."""

from __future__ import annotations

import numpy as np

from tscm.model import SPARK_RESERVED, array_namespace

# Terms of the stage runtime equation, in the order "why is it slow?" reports them.
TERMS = ("compute", "recompute", "spill", "shuffle", "fixed", "coordination")


def waves(n, N):
    xp = array_namespace(n, N)
    return xp.ceil(n / N)


def effective_bandwidths(B_disk, B_net, beta_d, beta_n, L_disk, L_net):
    return B_disk / (1.0 + beta_d * L_disk), B_net / (1.0 + beta_n * L_net)


def compute_speed(kappa, beta_c, L_node):
    """Multiplier on theta_1: machine-type speed factor times co-location slowdown."""
    return kappa * (1.0 + beta_c * L_node)


def task_features(v_in, v_sh, spill, g, r, m, B_disk, B_net):
    """Per-task cost features, each multiplying one of theta_1, theta_r, theta_4, theta_5."""
    return {
        "compute": v_in / (1.0 - g),
        "recompute": r * v_in / (1.0 - g),
        "spill": spill / B_disk,
        "shuffle": v_sh / B_net * (m - 1.0) / m,
    }


def runtime_terms(V_in, V_sh, S, g, r, n, N, m, B_disk, B_net, th, speed=1.0):
    """Each term of the stage runtime equation, in seconds."""
    xp = array_namespace(V_in, g, th["theta1"])
    w = waves(n, N)
    per_task = task_features(V_in / n, V_sh / n, S / n, g, r, m, B_disk, B_net)
    return {
        "compute": w * th["theta1"] * speed * per_task["compute"],
        "recompute": w * th["theta_r"] * speed * per_task["recompute"],
        "spill": w * th["theta4"] * per_task["spill"],
        "shuffle": w * th["theta5"] * per_task["shuffle"],
        "fixed": th["theta0"] * xp.ones_like(w),
        "coordination": (th["theta2"] * xp.log(m) + th["theta3"] * m) * xp.ones_like(w),
    }


def cpu_utilization(terms, T, n, N):
    """Compute time over stage time, scaled by the share of slots a wave actually fills."""
    xp = array_namespace(T)
    occupancy = xp.minimum(n / (waves(n, N) * N), 1.0)
    return xp.clip(occupancy * (terms["compute"] + terms["recompute"]) / T, 0.0, 1.0)


def peak_memory(M_base, U_e, c_e, W, C):
    """M_peak = M_base + min(U_e, max_k c_e W_k + C)."""
    xp = array_namespace(M_base, W)
    return M_base + xp.minimum(U_e, xp.max(c_e * W, axis=-1) + C)


def zero_spill_memory(c_e, W, C, f_spark):
    """The closed-form M_req, needing only Tiers 1 and 2."""
    xp = array_namespace(W)
    return SPARK_RESERVED + (xp.max(c_e * W, axis=-1) + C) / f_spark


def nnls_features(V_in, V_sh, S, g, n, N, m, B_disk, B_net) -> np.ndarray:
    """The six-column Ernest-style design for one stage, ordered theta_0..theta_5.

    Kept as the initialization and the baseline; see Fitting procedure, step 3.
    """
    w = np.ceil(n / N)
    per_task = task_features(V_in / n, V_sh / n, S / n, g, 0.0, m, B_disk, B_net)
    return np.column_stack(
        [
            np.ones_like(w),
            w * per_task["compute"],
            np.log(m),
            m,
            w * per_task["spill"],
            w * per_task["shuffle"],
        ]
    )
