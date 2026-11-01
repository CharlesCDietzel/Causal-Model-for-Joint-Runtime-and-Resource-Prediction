import numpy as np
import pandas as pd
import pytest

from tscm.eval import harness
from tscm.fit import modular
from tscm.fit.data import build_fit_data
from tscm.model import Config, GiB
from tscm.query import forward as fwd
from tscm.simulate import DEFAULT_MACHINE, simulate_runs


def test_metrics():
    assert harness.mape([110.0, 90.0], [100.0, 100.0]) == pytest.approx(0.1)
    assert harness.median_ape([100.0, 120.0, 200.0], [100.0, 100.0, 100.0]) == pytest.approx(0.2)
    assert harness.coverage([0.0, 0.0], [1.0, 1.0], [0.5, 2.0]) == 0.5
    assert harness.memory_requirement_error(12.0, 10.0) == pytest.approx(0.2)
    assert harness.deadline_regret(12.0, 10.0) == pytest.approx(0.2)


def test_profiling_cost_stops_at_the_first_run_under_target():
    assert harness.profiling_cost_to_target([5, 10, 20], [0.4, 0.14, 0.1]) == 10.0
    assert harness.profiling_cost_to_target([5], [0.4]) == float("inf")


def test_calibration_target():
    assert harness.calibration_ok(0.9) and not harness.calibration_ok(0.8)


@pytest.fixture(scope="module")
def grid(job, truth):
    base = dict(machine=DEFAULT_MACHINE, c_e=4, e=2, p={"k": 8, "max_iter": 4}, d={"skew": 0.1})
    cfgs = [Config(s=s * GiB, m=m, M_e=24 * GiB, **base)
            for s in (2, 4, 8, 16, 40, 100) for m in (2, 4, 8, 16)]
    return simulate_runs(job, truth, cfgs, seed=5)


def test_splits_partition_the_runs(grid):
    runs = grid[0]
    train, test = harness.interpolation_split(runs)
    assert len(train) + len(test) == len(runs) and not set(train.run_id) & set(test.run_id)
    train, test = harness.size_extrapolation_split(runs, s_target=200 * GiB)
    assert train["s"].max() <= 16 * GiB < test["s"].min()


def test_regime_split_keeps_memory_shrink_runs_in_training():
    runs = pd.DataFrame({"s": [1.0, 100.0, 1.0], "phase": ["B", "B", "A2"]})
    pressure = pd.Series([0.5, 2.0, 3.0])
    train, test = harness.regime_extrapolation_split(runs, pressure, 50.0)
    assert sorted(train.index) == [0, 2] and list(test.index) == [1]


def test_machine_transfer_split_holds_out_the_target_type():
    runs = pd.DataFrame({
        "job_id": ["x", "x", "y", "y", "y"],
        "machine_type": ["a", "c", "a", "b", "c"],
    })
    train, test = harness.machine_transfer_split(runs, "x", ("a", "b"), "a", "c")
    assert sorted(train.index) == [0, 2, 3] and list(test.index) == [1]


def test_cold_start_split_reveals_only_n_runs():
    runs = pd.DataFrame({"job_id": ["x"] * 5 + ["y"] * 2})
    train, test = harness.cold_start_split(runs, "x", 2)
    assert (train["job_id"] == "x").sum() == 2 and len(test) == 3


def test_ernest_recovers_its_own_form():
    runs = pd.DataFrame({
        "s": np.array([1.0, 2.0, 4.0, 8.0, 1.0, 8.0]) * GiB,
        "m": [1, 2, 4, 8, 8, 1],
    })
    s, m = runs["s"] / GiB, runs["m"]
    runs["T"] = 5.0 + 20.0 * s / m + 2.0 * np.log(m) + 0.5 * m
    fitted = harness.Ernest().fit(runs)
    np.testing.assert_allclose(fitted.coefficients, [5.0, 20.0, 2.0, 0.5], atol=1e-6)


def test_ernest_gpu_extrapolates_a_scaling_curve():
    N = np.array([1, 2, 4, 8])
    fitted = harness.ErnestGpu().fit(N, 1.0 + 10.0 / N)
    assert fitted.predict([16])[0] == pytest.approx(1.0 + 10.0 / 16, rel=0.05)


def test_c3o_gbm_uses_raw_features_and_black_box_uses_log(grid):
    runs = grid[0]
    assert harness.C3OGbm().features(runs)[0, 0] == runs["s"].iloc[0]
    assert harness.BlackBoxGbm().features(runs)[0, 0] == pytest.approx(np.log(runs["s"].iloc[0]))


def test_every_baseline_runs_on_a_split(grid):
    train, test = harness.interpolation_split(grid[0])
    for name, model in harness.default_baselines().items():
        predicted = model.fit(train).predict(test)
        assert len(predicted) == len(test) and np.isfinite(predicted).all(), name


def test_memory_baselines(grid):
    train, test = harness.interpolation_split(grid[0])
    plain = harness.KsPlusMemory().fit(train)
    safe = harness.RuyaMemory().fit(train)
    assert (safe.predict(test) >= plain.predict(test)).all()


def test_evaluate_scores_the_model_against_the_baselines(job, truth, grid):
    train, test = harness.size_extrapolation_split(grid[0], s_target=200 * GiB)
    post = fwd.Posterior.from_point(job, truth)
    cfgs = [Config(s=row.s, m=row.m, machine=DEFAULT_MACHINE, M_e=row.M_e, e=2, c_e=4,
                   p={"k": 8, "max_iter": 4}, d={"skew": 0.1}) for row in test.itertuples()]
    report = harness.evaluate("size", train, test, lambda _: harness.predict_runs(post, cfgs))
    assert set(report.scores) == {"tscm", *harness.default_baselines()}
    assert report.scores["tscm"]["mape"] < 0.1  # the true parameters must score well
    assert report.scores["tscm"]["mape"] < report.scores["c3o_gbm"]["mape"]


def test_family_report():
    runs = pd.DataFrame({"family": ["a", "a", "b"], "T": [10.0, 10.0, 10.0]})
    report = harness.report_by_family(runs, np.array([11.0, 11.0, 20.0]))
    assert report == {"a": pytest.approx(0.1), "b": pytest.approx(1.0)}


def test_no_regime_ablation_removes_the_memory_response(job, tables):
    fit = modular.fit_modular(build_fit_data(job, *tables))
    ablated = harness.no_regime(fit)
    assert fit.params["lam"].max() > 0  # the input is left alone
    cfg = Config(s=4 * GiB, m=1, machine=DEFAULT_MACHINE, M_e=1.7 * GiB, e=2, c_e=4,
                 p={"k": 8, "max_iter": 4}, d={"skew": 0.1})
    out = fwd.run(fwd.Posterior.from_point(job, ablated.params), [cfg], 1, noise=False)
    assert out["S_total"].item() == 0.0


def test_phase_a2_ablation_filters_runs():
    runs = pd.DataFrame({"phase": ["A", "A2", "B"]})
    assert harness.without_phase_a2(runs)["phase"].tolist() == ["A", "B"]


def test_whole_job_ablation_is_one_stage(job, grid):
    runs, stages, _ = grid
    collapsed, rows = harness.whole_job_tables(job, runs, stages)
    assert len(collapsed.stages) == 1 and len(rows) == len(runs)
    first_run = stages[stages.run_id == rows.run_id.iloc[0]]
    assert rows["V_in"].iloc[0] == pytest.approx(first_run["V_in"].sum())
    assert modular.fit_modular(build_fit_data(collapsed, runs, rows)).params["theta1"].shape == (1,)


def test_generic_functions_ablation_chains_through_mediators(grid):
    runs = harness.run_level_nodes(grid[0], grid[1])
    train, test = harness.interpolation_split(runs)
    model = harness.GenericNodes().fit(train)
    assert set(model.mediators) == set(harness.MEDIATORS)
    predicted = model.predict(test)
    assert len(predicted) == len(test) and np.isfinite(predicted).all()


def test_memory_requirement_scores_both_answers(job, truth):
    post = fwd.Posterior.from_point(job, truth)
    cfg = Config(s=8 * GiB, m=2, machine=DEFAULT_MACHINE, M_e=24 * GiB, e=2, c_e=4,
                 p={"k": 8, "max_iter": 2}, d={"skew": 0.1})
    scores = harness.memory_requirement_scores(post, [cfg], np.array([6 * GiB]), n_samples=40)
    assert set(scores) == {"tolerance", "zero_spill"}
