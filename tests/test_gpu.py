import numpy as np
import pytest

from tscm.fit import gpu as gpu_fit
from tscm.mechanisms import gpu
from tscm.mechanisms.gpu import GpuConfig
from tscm.model import GiB


def config(**changes):
    base = dict(dataset_size=1e6, epochs=1, b=32, P=25e6, phi=4e9, n_g=1, m=1, R_gpu=16 * GiB,
                F_gpu=1e13, B_pcie=16e9, B_net=1.2e9)
    return GpuConfig(**{**base, **changes})


TRUTH = {**gpu.DEFAULT_GPU_PARAMS, "theta_0": 0.02, "theta_c": 1.2, "theta_d": 2e-3,
         "theta_a": 1.5, "theta_l": 5e-3, "omega": 0.3, "A": 40e6, "G0": 1.5 * GiB,
         "tau_g": 0.03}


def test_ddp_keeps_all_state_on_every_gpu():
    cfg = config(n_g=4)
    assert gpu.working_set(cfg, 0.0, 0.0) == pytest.approx(25e6 * 16)


@pytest.mark.parametrize("shard, expected", [
    ("fsdp", 16.0 / 4), ("zero3", 16.0 / 4),
    ("zero1", 2.0 + 2.0 + 12.0 / 4), ("zero2", 2.0 + 14.0 / 4),
])
def test_sharding_reduces_per_gpu_state(shard, expected):
    cfg = config(n_g=4, shard=shard)
    assert gpu.working_set(cfg, 0.0, 0.0) == pytest.approx(25e6 * expected)


def test_unknown_sharding_is_rejected():
    with pytest.raises(ValueError, match="sharding mode"):
        gpu.working_set(config(shard="zero9"), 0.0, 0.0)


def test_checkpointing_shrinks_activations_and_adds_recompute():
    plain, checkpointed = config(), config(ckpt=True)
    assert gpu.working_set(checkpointed, 1e6, 0.0) < gpu.working_set(plain, 1e6, 0.0)
    th = TRUTH
    expected = gpu.step_time_terms(plain, th)["compute"] * (1.0 + th["kappa_ckpt"])
    assert gpu.step_time_terms(checkpointed, th)["compute"] == pytest.approx(expected)


def test_oom_probability_crosses_half_at_the_limit():
    assert gpu.oom_probability(10.0, 10.0, 0.03) == pytest.approx(0.5)
    assert gpu.oom_probability(9.0, 10.0, 0.01) < 1e-3
    assert gpu.oom_probability(11.0, 10.0, 0.01) > 0.999


def test_interconnect_switches_from_pcie_to_network():
    assert config(n_g=4, m=1).B_ic == 16e9
    assert config(n_g=4, m=2).B_ic == 1.2e9


def test_step_time_breaks_when_training_leaves_one_machine():
    one = gpu.step_time(config(n_g=4, m=1), TRUTH)
    two = gpu.step_time(config(n_g=2, m=2), TRUTH)  # same N, slower interconnect
    assert two > 1.5 * one


def test_step_time_is_limited_by_the_slower_of_compute_and_loading():
    th = {**TRUTH, "theta_d": 10.0}
    terms = gpu.step_time_terms(config(n_load=1), th)
    assert terms["busy"] == terms["load"] > terms["compute"]


def test_one_gpu_has_no_communication():
    assert gpu.step_time_terms(config(), TRUTH)["communication"] == 0.0


def test_steps_follow_dataset_size_and_runtime_sums():
    cfg = config(n_g=2, b=50, dataset_size=1000, epochs=2)
    assert cfg.steps == 20.0
    expected = TRUTH["theta_s"] + 20.0 * gpu.step_time(cfg, TRUTH)
    assert gpu.job_runtime(cfg, TRUTH) == pytest.approx(expected)


def test_requirement_has_a_ten_percent_margin():
    assert gpu.required_memory(10 * GiB) == pytest.approx(11 * GiB)


def test_largest_batch_fits_with_the_margin_and_the_next_does_not():
    cfg = config()
    b = gpu.largest_batch(cfg, TRUTH["A"], TRUTH["G0"])
    fits = gpu.working_set(cfg.with_(b=b), TRUTH["A"], TRUTH["G0"])
    over = gpu.working_set(cfg.with_(b=b + 1), TRUTH["A"], TRUTH["G0"])
    assert fits * gpu.MEMORY_MARGIN <= cfg.R_gpu < over * gpu.MEMORY_MARGIN


def test_rapids_transfer_term():
    assert gpu.rapids_transfer_term(2.0, 100.0, 4.0, 5.0) == 10.0


def simulated_runs(seed=0):
    rng = np.random.default_rng(seed)
    cfgs = [config(b=b) for b in (8, 16, 32, 64)]
    cfgs += [config(b=32, n_g=n) for n in (2, 4)]
    cfgs += [config(b=32, n_g=4, m=m) for m in (2, 4)]
    cfgs += [config(b=64, n_g=2, m=2), config(b=16, n_g=4, m=2), config(b=128)]
    T = [gpu.step_time(c, TRUTH) * np.exp(rng.normal(0, 0.01)) for c in cfgs]
    G = [gpu.working_set(c, TRUTH["A"], TRUTH["G0"]) * np.exp(rng.normal(0, 0.005)) for c in cfgs]
    capped = config(b=64, R_gpu=2.5 * GiB)
    cfgs.append(capped)
    T.append(np.nan)
    G.append(2.5 * GiB)
    oom = [False] * (len(cfgs) - 1) + [True]
    return gpu_fit.GpuRuns(cfgs, T, G, oom)


def test_memory_parameters_are_recovered():
    th = gpu_fit.fit_gpu_modular(simulated_runs())
    assert th["A"] == pytest.approx(TRUTH["A"], rel=0.03)
    assert th["G0"] == pytest.approx(TRUTH["G0"], rel=0.05)


def test_step_time_fit_predicts_held_out_scaling():
    th = gpu_fit.fit_gpu_modular(simulated_runs())
    unseen = config(b=48, n_g=2, m=4)
    assert gpu.step_time(unseen, th) == pytest.approx(gpu.step_time(unseen, TRUTH), rel=0.1)


def test_noise_estimate_is_kept():
    # Regression: a rebinding inside the fit used to lose sigma_step.
    th = gpu_fit.fit_gpu_modular(simulated_runs())
    assert th["sigma_step"] != gpu.DEFAULT_GPU_PARAMS["sigma_step"]


@pytest.mark.slow
def test_bayes_prediction_reports_every_gpu_outcome():
    post = gpu_fit.fit_gpu_bayes(simulated_runs(), num_warmup=150, num_samples=150, num_chains=2)
    report = gpu_fit.predict_gpu(post, config(b=32, n_g=4, m=2), n_samples=60)
    assert {"T_step", "T", "G_peak", "G_req", "u_gpu", "P_oom", "largest_batch"} == set(report)
    expected = gpu.step_time(config(b=32, n_g=4, m=2), TRUTH)
    assert report["T_step"]["p50"] == pytest.approx(expected, rel=0.15)
    assert report["P_oom"]["p95"] < 0.01
