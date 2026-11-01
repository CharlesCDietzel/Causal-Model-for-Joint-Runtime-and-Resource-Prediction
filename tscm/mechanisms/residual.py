"""Monotone GBM on log residuals, used only inside the observed support."""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from tscm.model import Config

FEATURES = ("log_s", "log_m", "log_M_e", "c_e", "e")
# Residuals may only grow with input size; the other features are left unconstrained.
MONOTONE = (1, 0, 0, 0, 0)


def tier0_features(cfg: Config) -> np.ndarray:
    return np.array([np.log(cfg.s), np.log(cfg.m), np.log(cfg.M_e), cfg.c_e, cfg.e], dtype=float)


@dataclass
class ResidualLearner:
    """Corrects one Tier 3 node. Outside the training range it returns exactly zero."""

    monotone: tuple[int, ...] = MONOTONE
    params: dict = field(default_factory=dict)
    model: lgb.Booster | None = None
    low: np.ndarray | None = None
    high: np.ndarray | None = None

    def fit(self, X: np.ndarray, log_residual: np.ndarray) -> ResidualLearner:
        X = np.asarray(X, dtype=float)
        self.low, self.high = X.min(axis=0), X.max(axis=0)
        settings = {
            "objective": "regression",
            "verbosity": -1,
            "learning_rate": 0.1,
            "num_leaves": 4,
            "min_data_in_leaf": max(2, len(X) // 10),
            "min_data_in_bin": 1,
            "monotone_constraints": list(self.monotone),
            "seed": 0,
            "deterministic": True,
            "force_row_wise": True,
            # Dozens of rows: OpenMP spin-waiting makes the default thread count ~500x slower.
            "num_threads": 1,
            **self.params,
        }
        data = lgb.Dataset(X, label=np.asarray(log_residual, dtype=float), params={"verbose": -1})
        self.model = lgb.train(settings, data, num_boost_round=50)
        return self

    def applicable(self, features: np.ndarray) -> bool:
        if self.model is None:
            return False
        return bool(np.all(features >= self.low) and np.all(features <= self.high))

    def predict(self, features: np.ndarray) -> float:
        if not self.applicable(features):
            return 0.0
        return float(self.model.predict(features.reshape(1, -1))[0])
