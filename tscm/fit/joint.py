"""Optional end-to-end refinement: feed predicted parents downstream to reduce compounding.

Parameters are optimized in an unconstrained space (softplus for non-negative ones), starting
from the modular fit. The node errors are the same squared log errors the modular steps
minimize, but here Tier 3 sees the Tier 1 and Tier 2 predictions rather than observations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from tscm.fit.data import FitData
from tscm.fit.modular import ModularFit
from tscm.model import param_specs
from tscm.query.forward import structural_pass

S_FLOOR = 2.0**20  # spill below 1 MiB is noise; the floor keeps log S defined at zero
G_FLOOR = 1e-4
PROXIMAL = 1e-3


def _softplus_inverse(y):
    return np.log(np.expm1(np.maximum(y, 1e-8)))


def _sigmoid_inverse(y):
    y = np.clip(y, 1e-6, 1.0 - 1e-6)
    return np.log(y / (1.0 - y))


def refine(data: FitData, fit: ModularFit, max_iter: int = 200) -> ModularFit:
    """Minimize the sum of squared log errors over all nodes. Returns a new ModularFit."""
    specs = {k: v for k, v in param_specs(data.job).items() if not k.startswith("sigma_")}
    init = {name: np.asarray(fit.params[name], dtype=float) for name in specs}

    def to_free(name, value):
        spec = specs[name]
        if spec.constraint == "nonneg":
            return _softplus_inverse(value / spec.scale)
        if spec.constraint == "unit":
            return _sigmoid_inverse(value)
        return (value - spec.loc) / spec.scale

    def from_free(name, value):
        spec = specs[name]
        if spec.constraint == "nonneg":
            return jax.nn.softplus(value) * spec.scale
        if spec.constraint == "unit":
            return jax.nn.sigmoid(value)
        return value * spec.scale + spec.loc

    def constrain(free):
        th = {
            name: jnp.where(fit.pinned[name], init[name], from_free(name, free[name]))
            for name in specs
        }
        return th | {k: jnp.asarray(v) for k, v in fit.params.items() if k.startswith("sigma_")}

    start = {name: to_free(name, init[name]) for name in specs}
    flat_start, unravel = ravel_pytree(start)
    observed = _aggregates(data)
    x = {k: jnp.asarray(v) for k, v in data.x.items()}
    weight = jnp.asarray(data.weight)
    n = jnp.asarray(np.nan_to_num(data.stage_run_mean(data.n), nan=1.0))
    forward = jax.vmap(
        lambda run_x, run_n, th: structural_pass(data.job, run_x, th, tasks=run_n),
        (0, 0, None))

    def loss(flat):
        out = forward(x, n, constrain(unravel(flat)))
        total = 0.0
        for name, floor in (("V_in", 0.0), ("V_sh", 0.0), ("T_k", 0.0), ("g", G_FLOOR),
                            ("S", S_FLOOR)):
            target = observed[name]
            mask = target > 0
            safe_target = jnp.where(mask, target, 1.0)
            # The maximum guards log(0) for a run whose predicted node is exactly zero.
            prediction = jnp.where(mask, jnp.maximum(out[name], 1e-30), 1.0)
            error = jnp.log(prediction + floor) - jnp.log(safe_target + floor)
            total += jnp.sum(weight[:, None] * mask * error**2)
        total += jnp.sum(weight * (jnp.log(out["T"]) - jnp.log(observed["T"])) ** 2)
        has_memory = observed["M_peak"] > 0
        peak = jnp.where(has_memory, observed["M_peak"], 1.0)
        memory_gap = jnp.log(out["M_peak"]) - jnp.log(peak)
        total += jnp.sum(weight * has_memory * memory_gap**2)
        # Keeps parameters the data cannot see at their modular values.
        return total + PROXIMAL * jnp.sum((flat - flat_start) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss))

    best = {"loss": np.inf, "flat": np.asarray(flat_start)}

    def objective(flat):
        value, grad = value_and_grad(jnp.asarray(flat))
        if float(value) < best["loss"]:
            best.update(loss=float(value), flat=np.array(flat))
        return float(value), np.asarray(grad, dtype=float)

    minimize(objective, np.asarray(flat_start), jac=True, method="L-BFGS-B",
             options={"maxiter": max_iter})
    refined = {k: np.asarray(v) for k, v in constrain(unravel(jnp.asarray(best["flat"]))).items()}
    params = {**fit.params, **refined}
    diagnostics = {**fit.diagnostics, "joint_loss": best["loss"],
                   "joint_start_loss": float(loss(flat_start))}
    return ModularFit(params, fit.pinned, diagnostics)


def _aggregates(data: FitData) -> dict[str, jnp.ndarray]:
    """Observed nodes as (runs, stages) means, with zero where a stage did not run."""
    aggregate = {
        name: jnp.asarray(np.nan_to_num(data.stage_run_mean(values)))
        for name, values in (("V_in", data.V_in), ("V_sh", data.V_sh), ("T_k", data.T_k),
                             ("g", data.g), ("S", data.S))
    }
    return aggregate | {"T": jnp.asarray(data.T), "M_peak": jnp.asarray(data.M_peak)}
