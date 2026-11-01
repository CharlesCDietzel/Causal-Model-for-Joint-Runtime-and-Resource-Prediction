"""Data characteristics d, computed once per input from a 0.1% sample."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

SAMPLE_RATE = 0.001
HLL_PRECISION = 14


def hyperloglog(values: pd.Series, precision: int = HLL_PRECISION) -> float:
    """Distinct-count estimate, with the small-range correction from the HLL paper."""
    m = 1 << precision
    hashes = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64)
    buckets = (hashes >> np.uint64(64 - precision)).astype(np.int64)
    rest = hashes << np.uint64(precision)
    # Position of the first set bit; frexp's exponent is the bit length of the float value.
    bit_length = np.frexp(rest.astype(np.float64))[1].astype(np.int64)
    rank = np.where(rest == 0, 64 - precision + 1, 64 - bit_length + 1)
    registers = np.zeros(m, dtype=np.int64)
    np.maximum.at(registers, buckets, rank)

    alpha = 0.7213 / (1.0 + 1.079 / m)
    estimate = alpha * m * m / np.sum(2.0 ** (-registers.astype(np.float64)))
    zeros = int(np.sum(registers == 0))
    if estimate <= 2.5 * m and zeros:
        return float(m * np.log(m / zeros))
    return float(estimate)


def top_key_share(values: pd.Series, top_fraction: float = 0.01) -> float:
    """Share of records held by the top 1% of keys, the design's skew measure."""
    counts = values.value_counts(sort=True)
    if counts.empty:
        return 0.0
    top = max(ceil(top_fraction * len(counts)), 1)
    return float(counts.iloc[:top].sum() / counts.sum())


def sparsity(frame: pd.DataFrame) -> float:
    """Fraction of zero values over numeric columns."""
    numeric = frame.select_dtypes(include="number")
    if numeric.size == 0:
        return 0.0
    return float((numeric.to_numpy() == 0).mean())


def profile_frame(
    sample: pd.DataFrame,
    key: str | None = None,
    sample_rate: float = 1.0,
    size_bytes: float | None = None,
) -> dict[str, float]:
    """d features from an in-memory sample. ``sample_rate`` scales record count to the input."""
    records = len(sample)
    features = {
        "record_count": records / sample_rate,
        "sparsity": sparsity(sample),
    }
    if size_bytes is not None and records:
        features["mean_record_size"] = size_bytes / (records / sample_rate)
    elif records:
        features["mean_record_size"] = float(sample.memory_usage(deep=True).sum() / records)
    if key is not None:
        features["distinct_keys"] = hyperloglog(sample[key])
        features["skew"] = top_key_share(sample[key])
    return features


def profile_parquet(
    path: str | Path,
    key: str | None = None,
    sample_rate: float = SAMPLE_RATE,
    seed: int = 0,
) -> dict[str, float]:
    """Sample Parquet rows batch by batch, so the input never has to fit in memory."""
    dataset = ds.dataset(str(path), format="parquet")
    rng = np.random.default_rng(seed)
    chosen = []
    total_rows = 0
    for batch in dataset.to_batches():
        total_rows += batch.num_rows
        mask = rng.random(batch.num_rows) < sample_rate
        if mask.any():
            chosen.append(batch.to_pandas()[mask])
    sample = pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame()
    size = sum(Path(f).stat().st_size for f in dataset.files)
    features = profile_frame(sample, key, 1.0, size_bytes=None)
    features["record_count"] = float(total_rows)
    if total_rows:
        features["mean_record_size"] = size / total_rows
    return features
