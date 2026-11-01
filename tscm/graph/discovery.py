"""Constrained discovery among resource-state mediators, pooled over a workload family.

Procedure: remove deterministic nodes, residualize on fixed parents, run PC with the tiers as
background knowledge, keep edges stable over bootstrap resamples of whole jobs, and send edges
contradicting the skeleton to a human. Below about 30 jobs, all of that is skipped.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from tscm.graph.skeleton import DISCOVERY_NODES, Skeleton, default_skeleton

MIN_JOBS = 30
DETERMINISTIC_R2 = 0.99
STABILITY = 0.7
BOOTSTRAPS = 200
ALPHA = 0.05


def _ols_residual(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficient
    total = float(np.sum((y - y.mean()) ** 2))
    return residual, 1.0 - float(np.sum(residual**2)) / total if total > 0 else 1.0


def _available_parents(frame: pd.DataFrame, node: str, skeleton: Skeleton) -> list[str]:
    return [p for p in skeleton.parents(node) if p in frame.columns]


def _fixed_parents(frame: pd.DataFrame, node: str, skeleton: Skeleton) -> list[str]:
    """Tier 0 and Tier 1 columns present, whether or not they are direct parents."""
    return [c for c in frame.columns if c != node and _tier_or_none(skeleton, c) in (0, 1)]


def _tier_or_none(skeleton: Skeleton, node: str) -> int | None:
    try:
        return skeleton.tier(node)
    except KeyError:
        return None


def remove_deterministic(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    skeleton: Skeleton,
    threshold: float = DETERMINISTIC_R2,
) -> list[str]:
    """Nodes that are near-exact functions of their fixed parents (R^2 > 0.99).

    They stay in the model as function nodes but would violate faithfulness in CI tests.
    """
    deterministic = []
    for node in candidates:
        parents = _available_parents(frame, node, skeleton)
        if parents:
            _, r2 = _ols_residual(frame[node].to_numpy(), frame[parents].to_numpy())
            if r2 > threshold:
                deterministic.append(node)
    return deterministic


def residualize(frame: pd.DataFrame, nodes: list[str], skeleton: Skeleton) -> pd.DataFrame:
    """Each node minus what Tier 0 and Tier 1 explain, leaving only unexplained structure."""
    result = {}
    for node in nodes:
        parents = _fixed_parents(frame, node, skeleton)
        result[node] = _ols_residual(frame[node].to_numpy(), frame[parents].to_numpy())[0]
    return pd.DataFrame(result, index=frame.index)


def _background(names: list[str], skeleton: Skeleton | None) -> BackgroundKnowledge | None:
    if skeleton is None:
        return None
    knowledge = BackgroundKnowledge()
    nodes = {name: GraphNode(name) for name in names}
    for name, node in nodes.items():
        knowledge.add_node_to_tier(node, skeleton.tier(name))
    for parent in names:
        for child in names:
            if parent != child and skeleton.forbidden(parent, child):
                knowledge.add_forbidden_by_node(nodes[parent], nodes[child])
    return knowledge


def run_pc(
    residuals: pd.DataFrame,
    skeleton: Skeleton | None,
    alpha: float = ALPHA,
    test: str = "fisherz",
) -> set[tuple[str, str]]:
    """Directed edges from one PC run. An unoriented edge appears in both directions.

    ``skeleton=None`` is the unconstrained variant, used by the no-tiers ablation.
    ``test`` is any causal-learn CI test; "kci" is the kernel test the design names.
    """
    names = list(residuals.columns)
    graph = pc(
        residuals.to_numpy(), alpha, test, background_knowledge=_background(names, skeleton),
        show_progress=False, verbose=False, node_names=names,
    ).G.graph
    edges = set()
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            # causal-learn encodes a -> b as graph[a, b] == -1 and graph[b, a] == 1.
            if graph[i, j] == -1 and graph[j, i] == 1:
                edges.add((a, b))
            elif graph[i, j] == -1 and graph[j, i] == -1 and i < j:
                edges.update({(a, b), (b, a)})
    return edges


@dataclass
class DiscoveryResult:
    edges: list[tuple[str, str]] = field(default_factory=list)
    frequency: dict[tuple[str, str], float] = field(default_factory=dict)
    deterministic: list[str] = field(default_factory=list)
    needs_review: list[tuple[str, str]] = field(default_factory=list)
    skipped: str | None = None


def stability_selection(
    frame: pd.DataFrame,
    jobs: pd.Series,
    nodes: list[str],
    skeleton: Skeleton | None,
    bootstraps: int = BOOTSTRAPS,
    threshold: float = STABILITY,
    alpha: float = ALPHA,
    seed: int = 0,
    test: str = "fisherz",
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], float]]:
    """PC on bootstrap resamples of whole jobs, not runs, which are correlated within a job."""
    rng = np.random.default_rng(seed)
    unique = jobs.unique()
    groups = {job: np.flatnonzero(jobs.to_numpy() == job) for job in unique}
    counts: Counter[tuple[str, str]] = Counter()
    for _ in range(bootstraps):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([groups[job] for job in chosen])
        sample = frame.iloc[rows].reset_index(drop=True)
        residuals = residualize(sample, nodes, skeleton or default_skeleton())
        counts.update(run_pc(residuals, skeleton, alpha, test))

    frequency = {edge: count / bootstraps for edge, count in counts.items()}
    kept = []
    for (a, b), share in frequency.items():
        reverse = frequency.get((b, a), 0.0)
        adjacency = share + reverse if reverse < share or (reverse == share and a < b) else 0.0
        # Keep at the threshold on adjacency, oriented by the majority direction.
        if adjacency >= threshold and (share > reverse or (share == reverse and a < b)):
            kept.append((a, b))
    return sorted(kept), frequency


def discover(
    frame: pd.DataFrame,
    job_column: str = "job_id",
    skeleton: Skeleton | None = None,
    candidates: tuple[str, ...] = DISCOVERY_NODES,
    bootstraps: int = BOOTSTRAPS,
    seed: int = 0,
    force: bool = False,
    test: str = "fisherz",
) -> DiscoveryResult:
    """The full procedure on a frame of log-scale node columns, one row per stage execution."""
    skeleton = skeleton or default_skeleton()
    jobs = frame[job_column]
    if jobs.nunique() < MIN_JOBS and not force:
        return DiscoveryResult(skipped=f"{jobs.nunique()} jobs < {MIN_JOBS}: using the skeleton")

    numeric = frame.drop(columns=[job_column])
    present = tuple(node for node in candidates if node in numeric.columns)
    deterministic = remove_deterministic(numeric, present, skeleton)
    nodes = [node for node in present if node not in deterministic]
    edges, frequency = stability_selection(
        numeric, jobs, nodes, skeleton, bootstraps, seed=seed, test=test)
    new_edges = [edge for edge in edges if edge not in skeleton.required]
    review = [(a, b) for a, b in new_edges if not skeleton.open_for_discovery(a, b)]
    return DiscoveryResult(new_edges, frequency, deterministic, review)


@dataclass
class DomainLog:
    """Human decisions on edges contradicting the skeleton, kept as an append-only log."""

    path: Path

    def record(self, edge: tuple[str, str], accepted: bool, reason: str, sign: int = 0) -> None:
        entry = {"parent": edge[0], "child": edge[1], "accepted": accepted, "reason": reason,
                 "sign": sign}
        with open(self.path, "a") as handle:
            handle.write(json.dumps(entry) + "\n")

    def decisions(self) -> list[dict]:
        if not Path(self.path).exists():
            return []
        with open(self.path) as handle:
            return [json.loads(line) for line in handle if line.strip()]
