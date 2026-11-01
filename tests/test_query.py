import numpy as np
import pytest

from tscm.model import Config, GiB, Hardware, Machine, MiB
from tscm.query import forward as fwd
from tscm.query import guards, queries
from tscm.simulate import DEFAULT_MACHINE


@pytest.fixture(scope="module")
def post(job, truth):
    return fwd.Posterior.from_point(job, truth)


def config(**changes):
    base = dict(s=40 * GiB, m=8, machine=DEFAULT_MACHINE, M_e=24 * GiB, e=2, c_e=4,
                p={"k": 8, "max_iter": 10}, d={"skew": 0.1})
    return Config(**{**base, **changes})


def mean_path(post, cfg):
    return {k: v[0, 0] for k, v in fwd.run(post, [cfg], 1, noise=False).items()}


def test_machines_and_memory_do_not_change_volumes(post):
    # The forbidden edge m, M_e, h -> V_k.
    a = mean_path(post, config())
    b = mean_path(post, config(m=32, M_e=4 * GiB))
    for node in ("V_in", "V_sh", "V_out"):
        np.testing.assert_allclose(a[node], b[node])


def test_more_machines_reduce_cached_data_per_executor(post):
    fewer, more = mean_path(post, config(m=8)), mean_path(post, config(m=16))
    assert more["C"] == pytest.approx(fewer["C"] / 2)


def test_shrinking_memory_crosses_into_spill_and_slows_the_job(post):
    roomy = mean_path(post, config())
    tight = mean_path(post, config(M_e=1 * GiB))
    assert np.all(roomy["rho"] < 1.0) and roomy["S_total"] == 0.0
    assert np.any(tight["rho"] > 1.0) and tight["S_total"] > 0.0
    assert tight["T"] > 1.5 * roomy["T"]


def test_iterations_scale_the_iterative_path(post):
    short = mean_path(post, config(p={"k": 8, "max_iter": 5}))
    long = mean_path(post, config(p={"k": 8, "max_iter": 10}))
    assert long["T"] > 1.5 * short["T"]


def test_interference_slows_runs_only_when_load_is_present(job, truth):
    hardware = Hardware(beta_c=0.5, beta_d=1e-9, beta_n=1e-9)
    post = fwd.Posterior.from_point(job, truth, hardware)
    solo = mean_path(post, config())
    busy = mean_path(post, config(L_node=2.0, L_net=1e9, L_disk=1e9))
    assert busy["T"] > solo["T"]
    assert fwd.Posterior.from_point(job, truth).job == job


def test_kappa_scales_compute_for_a_slower_machine_type(job, truth):
    slow = Machine("slow", 48, 192 * GiB, 500 * MiB, 1.2 * GiB)
    post = fwd.Posterior.from_point(job, truth, Hardware(kappa={"slow": 2.0}))
    a = mean_path(post, config())
    b = mean_path(post, config(machine=slow))
    assert b["term_compute"][1] == pytest.approx(2.0 * a["term_compute"][1])


def test_predict_reports_intervals_and_stages(post):
    result = queries.predict(post, config(), n_samples=200)
    T = result.outcomes["T"]
    assert T["lo"] < T["p50"] < T["hi"] and T["p95"] == T["hi"]
    assert set(result.stages) == {"scan", "assign", "update"}


def test_same_seed_gives_the_same_draws(post):
    a = fwd.run(post, [config()], 50, seed=3)["T"]
    b = fwd.run(post, [config()], 50, seed=3)["T"]
    np.testing.assert_array_equal(a, b)


def test_effect_is_paired_so_noise_cancels(post):
    report = queries.effect(post, config(), config(c_e=4), n_samples=100)
    assert report["T"]["p50"] == 0.0 and report["T"]["hi"] == 0.0


def test_memory_requirement_is_near_the_runtime_knee(post):
    cfg = config()
    answer = queries.memory_requirement(post, cfg, n_samples=100)
    assert answer.runtime_at_requirement <= 1.06 * answer.unlimited_runtime
    below = queries.fwd.run(post, [cfg.with_(M_e=answer.tolerance_bytes * 0.5)], 100)["T"]
    assert np.median(below) > 1.05 * answer.unlimited_runtime


def test_cheapest_configuration_meets_the_deadline(post):
    cheap = Machine("cheap", 48, 192 * GiB, 500 * MiB, 1.2 * GiB, rate=1.0)
    pricey = Machine("pricey", 48, 192 * GiB, 500 * MiB, 1.2 * GiB, rate=3.0)
    deadline = 400.0
    choice = queries.cheapest_configuration(
        post, config(), deadline, [cheap, pricey], [2, 4, 8, 16], [8 * GiB, 24 * GiB], 200)
    assert choice is not None
    assert choice.runtime_p95 <= deadline
    assert choice.config.machine.name == "cheap"


def test_cheapest_configuration_skips_memory_the_machine_lacks(post):
    choice = queries.cheapest_configuration(
        post, config(), 1e9, [DEFAULT_MACHINE], [4], [400 * GiB], 50)
    assert choice is None


def test_impossible_deadline_returns_none(post):
    assert queries.cheapest_configuration(
        post, config(), 0.001, [DEFAULT_MACHINE], [8], [24 * GiB], 50) is None


def test_decomposition_reports_the_dominant_term(post):
    tight = queries.decompose(post, config(M_e=1 * GiB), 50)
    assert tight["assign"]["dominant"] in ("recompute", "compute", "spill")
    assert set(tight["scan"]["terms"]) == {
        "compute", "recompute", "spill", "shuffle", "fixed", "coordination"}


def support(**changes):
    base = dict(max_s=4 * GiB, machine_types=frozenset({DEFAULT_MACHINE.name}),
                param_values={"k": frozenset({4.0, 8.0})},
                stages_seen_over_pressure=frozenset({"scan"}))
    return guards.Support(**{**base, **changes})


def codes(flags):
    return {f.code for f in flags}


def test_guard_size_extrapolation_is_past_ten_times():
    assert "size_extrapolation" in codes(guards.check(config(s=41 * GiB), support()))
    assert "size_extrapolation" not in codes(guards.check(config(s=39 * GiB), support()))


def test_guard_unseen_regime_only_for_stages_never_over_pressure():
    flags = guards.check(config(s=4 * GiB), support(), {"scan": 2.0, "assign": 0.5})
    assert "unseen_regime" not in codes(flags)
    flags = guards.check(config(s=4 * GiB), support(), {"scan": 2.0, "assign": 1.2})
    assert "unseen_regime" in codes(flags)


def test_guard_unseen_configuration():
    other = Machine("other", 8, 32 * GiB, 1e8, 1e9)
    assert "unseen_configuration" in codes(guards.check(config(s=GiB, machine=other), support()))
    outside = config(s=GiB, p={"k": 64})
    assert "unseen_configuration" in codes(guards.check(outside, support()))
    between = config(s=GiB, p={"k": 6})
    assert "unseen_configuration" not in codes(guards.check(between, support()))


def test_guard_residual_off():
    assert "residual_off" in codes(guards.check(config(s=GiB), support(), None, False))
