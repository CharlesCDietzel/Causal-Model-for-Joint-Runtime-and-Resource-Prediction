"""Tiered structural causal model for joint runtime and resource prediction."""

import jax
import numpyro

# Byte counts reach 1e13, past what float32 resolves when computing task counts.
jax.config.update("jax_enable_x64", True)
# NUTS chains run in parallel only with several host devices, fixed when the backend starts.
numpyro.set_host_device_count(4)

__version__ = "0.1.0"
