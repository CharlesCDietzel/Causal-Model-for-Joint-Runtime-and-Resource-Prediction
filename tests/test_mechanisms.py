import numpy as np
import pytest

from tscm.mechanisms import tier1, tier2, tier3
from tscm.mechanisms.residual import ResidualLearner
from tscm.model import SPARK_RESERVED, Config, GiB, Job, MiB, Stage
from tscm.simulate import DEFAULT_MACHINE


def test_h_is_never_negative_below_pressure_one():
    # The design's unclamped form gives h(0.5) < 0, meaning negative spill.
    rho = np.array([1e-6, 0.1, 0.5, 0.99, 1.0])
    assert np.all(tier2.h(rho, 0.1) == 0.0)


def test_h_approaches_excess_well_above_one():
    assert tier2.h(np.array(3.0), 0.05) == pytest.approx(1.0 - 1.0 / 3.0)


def test_spill_amount_times_probability_is_h():
    rho = np.array([1.1, 1.5, 3.0])
    tau, lam, base = 0.2, 2.0, 100.0
    amount = tier2.spill_amount(rho, lam, base) * tier2.spill_probability(rho, tau)
    assert amount == pytest.approx(lam * base * tier2.h(rho, tau))


def test_spill_base_falls_back_to_input_without_shuffle_write():
    base = tier2.spill_base(np.array([10.0, 10.0]), np.array([0.0, 4.0]), np.array([0.0, 1.0]))
    assert base.tolist() == [10.0, 4.0]


def test_gc_is_bounded_by_g_max_and_increases_with_pressure():
    rho = np.array([0.1, 1.0, 10.0, 1e6])
    g = tier2.gc_fraction(rho, 4 * GiB, -2.0, 1.5, -0.1, 0.4)
    assert np.all(np.diff(g) > 0) and np.all(g < 0.4)


def test_task_counts_follow_the_stage_kind():
    V = np.array([130 * MiB, 130 * MiB, 130 * MiB])
    n = tier1.tasks(V, 128 * MiB, 200.0, np.array([True, False, False]),
                    np.array([False, True, False]), np.array([1.0, 1.0, 7.0]))
    assert n.tolist() == [2.0, 200.0, 7.0]


def test_stage_runtime_matches_the_design_equation_by_hand():
    th = {"theta0": 2.0, "theta1": 1e-8, "theta_r": 0.0, "theta4": 3.0, "theta5": 1.5,
          "theta2": 0.7, "theta3": 0.1}
    V_in, V_sh, S, g, n, N, m, B_disk, B_net = 8e9, 4e9, 1e9, 0.2, 40.0, 16.0, 4.0, 2e8, 1e9
    terms = tier3.runtime_terms(V_in, V_sh, S, g, 0.0, n, N, m, B_disk, B_net, th)
    w = np.ceil(n / N)
    expected = 2.0 + w * (
        1e-8 * V_in / (n * (1 - g)) + 3.0 * S / (n * B_disk)
        + 1.5 * V_sh / (n * B_net) * (m - 1) / m
    ) + 0.7 * np.log(m) + 0.1 * m
    assert sum(terms.values()) == pytest.approx(expected)


def test_single_machine_has_no_remote_shuffle_cost():
    features = tier3.task_features(1.0, 100.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert features["shuffle"] == 0.0


def test_interference_scales_bandwidth_and_compute():
    B_disk, B_net = tier3.effective_bandwidths(100.0, 200.0, 0.5, 1.0, 2.0, 1.0)
    assert (B_disk, B_net) == (50.0, 100.0)
    assert tier3.compute_speed(1.2, 0.25, 2.0) == pytest.approx(1.8)


def test_cpu_utilization_counts_unfilled_slots():
    terms = {"compute": 10.0, "recompute": 0.0}
    assert tier3.cpu_utilization(terms, 10.0, 4.0, 16.0) == pytest.approx(0.25)


def test_zero_spill_memory_closed_form():
    W = np.array([1 * GiB, 2 * GiB])
    value = tier3.zero_spill_memory(4.0, W, 3 * GiB, 0.6)
    assert value == pytest.approx(SPARK_RESERVED + (8 * GiB + 3 * GiB) / 0.6)


def test_peak_memory_is_capped_by_usable_memory():
    W = np.array([10 * GiB])
    assert tier3.peak_memory(GiB, 4 * GiB, 4.0, W, 0.0) == pytest.approx(5 * GiB)


def test_longest_path_takes_the_slower_branch_of_a_diamond():
    job = Job("j", "f", (
        Stage("a"), Stage("b", parents=("a",)), Stage("c", parents=("a",)),
        Stage("d", parents=("b", "c")),
    ))
    assert job.longest_path(np.array([1.0, 5.0, 2.0, 1.0])) == 7.0


def test_longest_path_broadcasts_over_draws():
    job = Job("j", "f", (Stage("a"), Stage("b", parents=("a",))))
    values = job.longest_path(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert values.tolist() == [3.0, 7.0]


def test_cycles_and_unknown_parents_are_rejected():
    with pytest.raises(ValueError, match="cycle"):
        Job("j", "f", (Stage("a", parents=("b",)), Stage("b", parents=("a",))))
    with pytest.raises(ValueError, match="unknown parents"):
        Job("j", "f", (Stage("a", parents=("z",)),))


def test_repeats_read_the_iteration_parameter():
    job = Job("j", "f", (Stage("a"), Stage("b", repeat_param="max_iter")))
    assert job.repeats({"max_iter": 7}).tolist() == [1.0, 7.0]


def test_derived_config_quantities():
    cfg = Config(s=GiB, m=4, machine=DEFAULT_MACHINE, M_e=2 * GiB, e=2, c_e=3)
    assert cfg.N == 24
    assert cfg.U_e == pytest.approx(0.6 * (2 * GiB - 300 * MiB))


def test_residual_is_exactly_zero_outside_the_training_range():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (60, 5))
    learner = ResidualLearner().fit(X, X[:, 0])
    assert learner.applicable(np.full(5, 0.5))
    assert learner.predict(np.full(5, 0.5)) != 0.0
    outside = np.array([2.0, 0.5, 0.5, 0.5, 0.5])
    assert not learner.applicable(outside)
    assert learner.predict(outside) == 0.0


def test_residual_respects_the_monotone_constraint():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, (200, 5))
    learner = ResidualLearner().fit(X, -3.0 * X[:, 0] + rng.normal(0, 0.1, 200))
    grid = np.column_stack([np.linspace(0.05, 0.95, 30), np.full((30, 4), 0.5)])
    values = [learner.predict(row) for row in grid]
    assert np.all(np.diff(values) >= -1e-9)
