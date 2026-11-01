import numpy as np
import pandas as pd
import pytest

from tscm.fit import bayes, joint, modular, transfer
from tscm.fit.data import HISTORICAL_WEIGHT, build_fit_data, build_job
from tscm.model import Config, GiB
from tscm.query import forward as fwd
from tscm.simulate import DEFAULT_MACHINE, simulate_runs


@pytest.fixture(scope="module")
def built(tables):
    runs, stages, _ = tables
    return build_job("kmeans", "iterative_ml", runs, stages, features=("k",),
                     iteration_param="max_iter")


@pytest.fixture(scope="module")
def data(built, tables):
    runs, stages, tasks = tables
    return build_fit_data(built, runs, stages, tasks)


@pytest.fixture(scope="module")
def fit(data):
    return modular.fit_modular(data)


def test_build_job_recovers_the_dag(built):
    by_name = {s.signature: s for s in built.stages}
    assert by_name["scan"].kind == "input"
    assert by_name["assign"].kind == "shuffle"
    assert by_name["assign"].stage_type == "iteration_body"
    assert by_name["assign"].repeat_param == "max_iter"
    assert by_name["assign"].reads_cache
    assert by_name["update"].parents == ("assign",)
    assert built.cached


def test_build_job_breaks_iteration_cycles(tables):
    runs, stages, _ = tables
    looped = stages.copy()
    looped["parent_signatures"] = [
        ["update"] if s == "assign" else list(p)
        for s, p in zip(looped["stage_signature"], looped["parent_signatures"], strict=True)
    ]
    job = build_job("kmeans", "f", runs, looped)  # raises on a cycle if not broken
    assert job.topological


@pytest.mark.parametrize("name, tolerance", [
    ("a_in", 0.02), ("b_in", 0.02), ("omega1", 0.1), ("theta1", 0.03), ("theta_r", 0.05),
])
def test_identifiable_stage_parameters_are_recovered(fit, truth, name, tolerance):
    k = 1  # the iteration body, the stage with the most evidence
    assert fit.params[name][k] == pytest.approx(truth[name][k], rel=tolerance)


def test_shared_gc_and_noise_terms_are_recovered(fit, truth):
    assert fit.params["gamma1"] == pytest.approx(truth["gamma1"], rel=0.05)
    assert fit.params["sigma_T"] == pytest.approx(truth["sigma_T"], rel=0.25)
    assert fit.params["M_base"] == pytest.approx(truth["M_base"], rel=0.05)


def test_censoring_keeps_the_working_set_unbiased(job, truth):
    # Under heavy spill every task reads as capped; ignoring that underestimates omega_1.
    base = dict(machine=DEFAULT_MACHINE, c_e=4, e=2, p={"k": 8, "max_iter": 2}, d={"skew": 0.1})
    cfgs = [Config(s=s * GiB, m=1, M_e=M * GiB, **base)
            for s in (2, 4, 8) for M in (48, 2)]
    runs, stages, tasks = simulate_runs(job, truth, cfgs, seed=3)
    data = build_fit_data(job, runs, stages, tasks)
    params = modular.fit_modular(data).params
    assert params["omega1"][1] == pytest.approx(truth["omega1"][1], rel=0.15)


def test_stage_without_any_input_is_not_given_the_default_volume(job, tables):
    # Regression from real Spark: a generating stage must not inherit 1 GiB of invented input.
    runs, stages, tasks = tables
    silent = stages.copy()
    silent.loc[silent["stage_signature"] == "scan", ["V_in", "V_read"]] = 0.0
    data = build_fit_data(job, runs, silent, tasks[tasks["stage_signature"] != "scan"])
    params = modular.fit_modular(data).params
    assert np.exp(params["a_in"][0]) < 2.0
    assert params["b_in"][0] == 0.0


def test_fixed_buffers_do_not_become_a_huge_expansion_factor(job, tables):
    # Regression from real Spark: hash aggregation holds a constant buffer on tiny inputs.
    runs, stages, tasks = tables
    small = tasks.copy()
    small["v_in"] = 200.0
    small["peak_execution_memory"] = 2.5e8
    params = modular.fit_modular(build_fit_data(job, runs, stages, small)).params
    assert params["omega1"].max() <= modular.MAX_EXPANSION
    assert params["omega0"][1] > 1e8


def test_unidentified_parameters_are_pinned(job, tables):
    runs, stages, tasks = tables
    single_size = runs[runs["s"] == runs["s"].iloc[0]]
    data = build_fit_data(job, single_size, stages, tasks)
    result = modular.fit_modular(data)
    assert result.pinned["b_in"].all() and result.params["b_in"].tolist() == [1.0, 1.0, 1.0]


def test_no_spill_keeps_defaults_and_pins_tau(job, truth):
    base = dict(machine=DEFAULT_MACHINE, c_e=4, e=2, p={"k": 8, "max_iter": 2}, d={"skew": 0.1})
    cfgs = [Config(s=s * GiB, m=m, M_e=48 * GiB, **base) for s in (1, 2, 4) for m in (2, 8)]
    data = build_fit_data(job, *simulate_runs(job, truth, cfgs, seed=2))
    result = modular.fit_modular(data)
    assert not result.diagnostics["spill_observed"]
    assert bool(result.pinned["tau"])


def test_historical_runs_are_down_weighted(job, truth, tables):
    runs, stages, tasks = tables
    marked = runs.assign(source=["historical"] * 3 + ["profiling"] * (len(runs) - 3))
    data = build_fit_data(job, marked, stages, tasks)
    assert data.weight[:3].tolist() == [HISTORICAL_WEIGHT] * 3
    assert data.weight[3:].min() == 1.0


def test_model_checks_flag_bad_fits(data, fit):
    bad = {k: np.array(v, copy=True) for k, v in fit.params.items()}
    bad["theta1"] = bad["theta1"] * 100.0
    report = modular.model_checks(data, bad)
    assert report["stages_over_mape_limit"]
    assert report["implausible_theta1"]


def test_joint_refinement_never_worsens_its_objective(data, fit):
    refined = joint.refine(data, fit, max_iter=60)
    assert refined.diagnostics["joint_loss"] < refined.diagnostics["joint_start_loss"]
    for name, pinned in fit.pinned.items():
        if name.startswith("sigma_"):
            continue
        np.testing.assert_allclose(np.asarray(refined.params[name])[pinned],
                                   np.asarray(fit.params[name])[pinned])


def test_interference_betas_are_recovered():
    rng = np.random.default_rng(0)
    L_node = rng.uniform(0, 3, 8)
    L_net = rng.uniform(0, 2e9, 8)
    L_disk = rng.uniform(0, 1e9, 8)
    solo = np.full(8, 100.0)
    shared = solo * (1.0 + 0.1 * L_node + 2e-10 * L_net + 5e-10 * L_disk)
    beta = modular.fit_interference(solo, shared, L_node, L_net, L_disk)
    assert beta["beta_c"] == pytest.approx(0.1, rel=1e-6)
    assert beta["beta_n"] == pytest.approx(2e-10, rel=1e-6)
    assert beta["beta_d"] == pytest.approx(5e-10, rel=1e-6)


def test_family_prior_and_cold_start(job, truth):
    posteriors = []
    for scale in (0.8, 1.0, 1.25):
        params = {k: np.array(v, copy=True) for k, v in truth.items()}
        params["theta1"] = params["theta1"] * scale
        posteriors.append(fwd.Posterior.from_point(job, params))
    prior = transfer.learn_family_prior("iterative_ml", posteriors)
    mu, sigma = prior.lookup("theta1", "iteration_body")
    assert np.exp(mu) == pytest.approx(truth["theta1"][1], rel=0.05)
    assert sigma >= transfer.MIN_LOG_SCALE

    cold = transfer.posterior_from_prior(job, prior, n_draws=200)
    cfg = Config(s=4 * GiB, m=4, machine=DEFAULT_MACHINE, M_e=24 * GiB, e=2, c_e=4,
                 p={"k": 8, "max_iter": 5}, d={"skew": 0.1})
    draws = fwd.run(cold, [cfg], 200)["T"]
    expected = fwd.run(fwd.Posterior.from_point(job, truth), [cfg], 1, noise=False)["T"]
    assert np.quantile(draws, 0.05) < expected.item() < np.quantile(draws, 0.95)


def test_machine_factor_is_relative_to_the_reference(job, truth):
    faster = {k: np.array(v, copy=True) for k, v in truth.items()}
    faster["theta1"] = faster["theta1"] * 0.5
    kappa = transfer.machine_factors({
        "a": [fwd.Posterior.from_point(job, truth)],
        "b": [fwd.Posterior.from_point(job, faster)],
    }, reference="a")
    assert kappa == {"a": 1.0, "b": pytest.approx(0.5)}


def test_overlap_check_flags_a_ridge():
    ridge = pd.DataFrame({"s": [1.0, 2.0, 4.0, 8.0], "m": [2, 4, 8, 16]})
    spread = pd.DataFrame({"s": [1.0, 1.0, 8.0, 8.0], "m": [2, 16, 2, 16]})
    assert transfer.check_overlap(ridge).weakly_identified
    assert not transfer.check_overlap(spread).weakly_identified


def test_unrecorded_cluster_load_is_detected():
    runs = pd.DataFrame({"source": ["historical", "profiling"], "cluster_load": [np.nan, 0.0]})
    assert not transfer.load_recorded(runs)


@pytest.mark.slow
def test_nuts_recovers_parameters_and_converges(data, fit, truth):
    post = bayes.fit_bayes(data, fit, num_warmup=250, num_samples=250, num_chains=2)
    assert post.samples["theta1"].shape == (500, 3)
    assert np.median(post.samples["theta1"][:, 1]) == pytest.approx(truth["theta1"][1], rel=0.05)
    assert np.median(post.samples["gamma1"]) == pytest.approx(truth["gamma1"], rel=0.1)
    # Pinned elements stay exactly at the modular value.
    assert np.ptp(post.samples["theta_r"][:, 0]) == 0.0
    # Spill terms are wider than the task-level compute terms, as the evidence is thinner.
    relative = post.samples["theta4"].std(0) / post.samples["theta4"].mean(0)
    assert relative.min() > 5 * post.samples["theta1"].std(0)[1] / truth["theta1"][1]


@pytest.mark.slow
def test_nuts_accepts_a_family_prior(data, fit):
    point = fwd.Posterior.from_point(data.job, fit.params)
    prior = transfer.learn_family_prior("iterative_ml", [point])
    post = bayes.fit_bayes(data, fit, prior, num_warmup=100, num_samples=100, num_chains=2)
    assert post.n_draws == 200
    assert np.isfinite(post.samples["theta1"]).all()
