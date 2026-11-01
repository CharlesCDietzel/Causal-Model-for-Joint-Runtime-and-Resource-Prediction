import numpy as np
import pandas as pd
import pytest

from tscm.graph import discovery
from tscm.graph.skeleton import default_skeleton


@pytest.fixture(scope="module")
def skeleton():
    return default_skeleton()


def test_required_edges_form_a_valid_dag(skeleton):
    skeleton.validate()


def test_required_edges_never_point_to_a_lower_tier(skeleton):
    for parent, child in skeleton.required:
        assert skeleton.tier(parent) <= skeleton.tier(child)


@pytest.mark.parametrize("parent, child", [
    ("T_k", "s"), ("rho", "M_e"),    # anything into Tier 0
    ("T_k", "S"), ("S", "V_in"),     # higher tier to lower tier
    ("m", "V_in"), ("M_e", "V_sh"), ("h", "V_out"),  # the plan-change test
])
def test_forbidden_edges(skeleton, parent, child):
    assert skeleton.forbidden(parent, child)


@pytest.mark.parametrize("parent, child, expected", [
    ("S", "g", True), ("g", "u", True),  # within Tier 2, and Tier 2 into Tier 3
    ("rho", "S", False),                 # already required
    ("V_in", "W", False),                # not a discovery edge
    ("u", "rho", False),                 # CPU -> pressure contradicts the tiers
])
def test_what_is_open_for_discovery(skeleton, parent, child, expected):
    assert skeleton.open_for_discovery(parent, child) is expected


def make_frame(jobs=40, rows=30, seed=0, with_deterministic=False):
    """Planted S -> g on top of a shared Tier 0 cause, with u independent."""
    rng = np.random.default_rng(seed)
    frames = []
    for job in range(jobs):
        offset = rng.normal(0.0, 0.2)
        s = rng.normal(0.0, 1.0, rows)
        S = 0.7 * s + offset + rng.normal(0.0, 1.0, rows)
        g = 0.8 * S + 0.3 * s + rng.normal(0.0, 0.5, rows)
        u = 0.5 * s + rng.normal(0.0, 1.0, rows)
        frame = {"job_id": f"j{job}", "s": s, "S": S, "g": g, "u": u}
        if with_deterministic:
            frame["W"] = 2.0 * s + 1e-6 * rng.normal(0.0, 1.0, rows)
        frames.append(pd.DataFrame(frame))
    return pd.concat(frames, ignore_index=True)


def test_too_few_jobs_skips_discovery():
    result = discovery.discover(make_frame(jobs=10))
    assert result.skipped and "10 jobs" in result.skipped and result.edges == []


def test_deterministic_nodes_are_excluded(skeleton):
    frame = make_frame(jobs=2, with_deterministic=True).drop(columns="job_id")
    frame = frame.rename(columns={"s": "V_in"})
    assert discovery.remove_deterministic(frame, ("W", "S"), skeleton) == ["W"]


def test_residualizing_removes_the_shared_cause(skeleton):
    frame = make_frame(jobs=40).drop(columns="job_id")
    residuals = discovery.residualize(frame, ["S", "u"], skeleton)
    assert abs(np.corrcoef(residuals["S"], residuals["u"])[0, 1]) < 0.1
    assert abs(np.corrcoef(frame["S"], frame["u"])[0, 1]) > 0.1


def test_discovery_finds_the_planted_edge_and_only_that():
    result = discovery.discover(make_frame(), bootstraps=20)
    assert result.skipped is None
    pairs = {frozenset(edge) for edge in result.edges}
    assert frozenset(("S", "g")) in pairs
    assert frozenset(("S", "u")) not in pairs and frozenset(("g", "u")) not in pairs


def test_tiers_orient_edges_into_tier_3(skeleton):
    residuals = pd.DataFrame(np.random.default_rng(0).normal(size=(800, 2)), columns=["g", "u"])
    residuals["u"] = residuals["u"] + 2.0 * residuals["g"]
    assert discovery.run_pc(residuals, skeleton) == {("g", "u")}


def test_unconstrained_pc_cannot_orient_a_lone_edge():
    # The no-tiers ablation: without background knowledge the direction stays open.
    residuals = pd.DataFrame(np.random.default_rng(0).normal(size=(800, 2)), columns=["g", "u"])
    residuals["u"] = residuals["u"] + 2.0 * residuals["g"]
    assert discovery.run_pc(residuals, None) == {("g", "u"), ("u", "g")}


def test_stability_selection_bootstraps_jobs_not_rows():
    frame = make_frame(jobs=6, rows=40)
    numeric = frame.drop(columns="job_id")
    edges, frequency = discovery.stability_selection(
        numeric, frame["job_id"], ["S", "g", "u"], default_skeleton(), bootstraps=8)
    assert all(0.0 <= share <= 1.0 for share in frequency.values())
    assert edges == sorted(edges)


def test_domain_decisions_are_logged(tmp_path):
    log = discovery.DomainLog(tmp_path / "decisions.jsonl")
    assert log.decisions() == []
    log.record(("u", "rho"), False, "contradicts the tiers")
    log.record(("S", "g"), True, "spill raises GC", sign=1)
    assert [d["accepted"] for d in log.decisions()] == [False, True]
