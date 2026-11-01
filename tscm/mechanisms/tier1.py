"""Tier 1: data volumes and task counts. Nothing here depends on machines or memory."""

from __future__ import annotations

from tscm.model import array_namespace


def log_volume(a, b, phi, log_s, z):
    """log V_k = a_k + b_k log s + phi_k' z(d, p)."""
    xp = array_namespace(a, b, phi, log_s, z)
    return a + b * log_s + xp.sum(phi * z, axis=-1)


def tasks(V_in, partition_bytes, shuffle_partitions, is_input, is_shuffle, fixed_tasks):
    """n_k = ceil(V_in / partition size) for input stages, else the configured partitions."""
    xp = array_namespace(V_in, partition_bytes, shuffle_partitions)
    by_size = xp.maximum(xp.ceil(V_in / partition_bytes), 1.0)
    return xp.where(is_input, by_size, xp.where(is_shuffle, shuffle_partitions, fixed_tasks))
