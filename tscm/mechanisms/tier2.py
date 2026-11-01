"""Tier 2: working set, memory pressure, spill, recomputation and GC."""

from __future__ import annotations

from tscm.model import GiB, array_namespace, sigmoid

MIN_PRESSURE = 1e-9


def working_set(V, n, skew, omega0, omega1, omega2):
    """W_k = omega0 + omega1 (V / n) (1 + omega2 skew)."""
    return omega0 + omega1 * (V / n) * (1.0 + omega2 * skew)


def cache_per_executor(C_total, m, e):
    return C_total / (m * e)


def memory_pressure(c_e, W, C, U_e):
    return (c_e * W + C) / U_e


def excess(rho):
    """1 - 1/rho clamped at zero, the share of demand that does not fit.

    The design's h(rho) leaves this unclamped, which makes spill negative for rho < 1.
    """
    xp = array_namespace(rho)
    return xp.maximum(1.0 - 1.0 / xp.maximum(rho, MIN_PRESSURE), 0.0)


def spill_probability(rho, tau):
    return sigmoid((rho - 1.0) / tau)


def h(rho, tau):
    return excess(rho) * spill_probability(rho, tau)


def spill_amount(rho, lam, base):
    """Bytes spilled given that the stage spills; h(rho) is this times P(S > 0)."""
    return lam * base * excess(rho)


def spill_base(V_in, V_sh, writes_shuffle):
    """V_sh, or V_in for a stage writing no shuffle (a final sort still spills)."""
    xp = array_namespace(V_in, V_sh, writes_shuffle)
    return xp.where(writes_shuffle > 0, V_sh, V_in)


def gc_fraction(rho, M_e, gamma0, gamma1, gamma2, g_max):
    xp = array_namespace(rho, gamma0, gamma1)
    log_rho = xp.log(xp.maximum(rho, MIN_PRESSURE))
    return g_max * sigmoid(gamma0 + gamma1 * log_rho + gamma2 * xp.log(M_e / GiB))
