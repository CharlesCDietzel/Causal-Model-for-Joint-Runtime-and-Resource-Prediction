"""A synthetic Spark cluster: draws runs from the structural model with known parameters.

Used for parameter-recovery tests and the worked example, since those need ground truth.
Because it shares the model's equations, results on it validate the fitting and query code,
not the model's fit to real Spark jobs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tscm.collect.store import STAGE_COLUMNS, TASK_COLUMNS, run_row
from tscm.mechanisms import tier3
from tscm.model import Config, GiB, Hardware, Job, Machine, MiB, Stage, default_params
from tscm.query.forward import Posterior, run

DEFAULT_MACHINE = Machine(
    "compute_skylake", cores=48, memory=192 * GiB, disk_bw=500 * MiB, net_bw=1.2 * GiB)


def example_job(cached: bool = True) -> Job:
    """A K-Means-like job: scan and cache, a shuffled iteration body, and a summary stage."""
    return Job(
        job_id="kmeans",
        family="iterative_ml",
        stages=(
            Stage("scan", kind="input", stage_type="scan", writes_shuffle=True),
            Stage("assign", kind="shuffle", parents=("scan",), stage_type="iteration_body",
                  writes_shuffle=True, reads_cache=cached, repeat_param="max_iter"),
            Stage("update", kind="shuffle", parents=("assign",), stage_type="aggregation",
                  writes_output=True),
        ),
        features=("k",),
        cached=cached,
    )


def example_params(job: Job) -> dict[str, np.ndarray]:
    """Plausible ground truth for ``example_job`` (and any three-stage job)."""
    th = default_params(job)
    log_gib = np.log(GiB)
    th["a_in"][:] = [log_gib, log_gib - 0.5, log_gib - 1.5]
    th["b_in"][:] = [1.0, 1.0, 0.9]
    th["a_sh"][:] = [log_gib - 0.5, log_gib - 1.5, 0.0]
    th["b_sh"][:] = [1.0, 0.9, 0.0]
    th["a_out"][:] = [0.0, 0.0, log_gib - 3.0]
    th["b_out"][:] = [0.0, 0.0, 0.8]
    th["omega0"][:] = [32 * MiB, 64 * MiB, 48 * MiB]
    th["omega1"][:] = [1.2, 2.5, 3.0]
    th["omega2"][:] = 0.5
    th["lam"][:] = [0.9, 1.2, 1.0]
    th["gamma0"][:] = [-3.5, -3.0, -3.2]
    th["theta0"][:] = [1.0, 0.5, 0.5]
    th["theta1"][:] = [4e-8, 8e-8, 1e-7]
    th["theta4"][:] = [1.5, 2.0, 1.8]
    th["theta5"][:] = [1.2, 1.5, 1.0]
    th["theta_r"][:] = [4e-8, 3e-7, 1e-7]
    th.update(
        tau=np.array(0.08), g_max=np.array(0.45), gamma1=np.array(2.5), gamma2=np.array(-0.2),
        theta2=np.array(1.5), theta3=np.array(0.15), theta_drv=np.array(6.0),
        M_base=np.array(1.5 * GiB), a_c=np.array(log_gib + 0.2), b_c=np.array(1.0),
        sigma_V=np.array(0.02), sigma_W=np.array(0.08), sigma_S=np.array(0.1),
        sigma_g=np.array(0.1), sigma_T=np.array(0.04), sigma_task=np.array(0.1),
        sigma_eta=np.array(0.03), sigma_job=np.array(0.02),
    )
    return th


def simulate_runs(
    job: Job,
    params: dict[str, np.ndarray],
    cfgs: list[Config],
    hardware: Hardware | None = None,
    seed: int = 0,
    tasks_per_stage: int = 32,
    source: str = "profiling",
    phase: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(runs, stages, tasks) tables in the store's schema, one run per config."""
    hardware = hardware or Hardware()
    post = Posterior.from_point(job, params, hardware)
    rng = np.random.default_rng(seed)
    run_rows, stage_rows, task_rows = [], [], []

    for index, cfg in enumerate(cfgs):
        out = {k: v[0, 0] for k, v in run(post, [cfg], 1, seed + index).items()}
        run_id = f"{job.job_id}-{seed}-{index}"
        run_rows.append(run_row(
            cfg, run_id, job.job_id, job.family, T=float(out["T"]),
            M_peak=float(out["M_peak"]), source=source, phase=phase,
            node_seconds=float(out["T"]) * cfg.m))

        C_observed = min(float(out["C"]), cfg.U_e)
        clock = 0.0
        for k, stage in enumerate(job.stages):
            n = float(out["n"][k])
            speed = tier3.compute_speed(
                hardware.kappa.get(cfg.machine.name, 1.0), hardware.beta_c, cfg.L_node)
            B_disk, B_net = tier3.effective_bandwidths(
                cfg.machine.disk_bw, cfg.machine.net_bw, hardware.beta_d, hardware.beta_n,
                cfg.L_disk, cfg.L_net)
            repeats = int(round(job.repeats(cfg.p)[k]))
            for iteration in range(repeats):
                # Later iterations jitter around the pass; the run effect stays shared.
                jitter = float(np.exp(rng.normal(0.0, params["sigma_T"]))) if iteration else 1.0
                T_k = float(out["T_k"][k]) * jitter
                count = int(min(n, tasks_per_stage))
                share = rng.lognormal(0.0, 0.1, count)
                share /= share.mean()
                v_in = float(out["V_in"][k]) / n * share
                v_sh = float(out["V_sh"][k]) / n * share
                stage_spill = float(out["S"][k])
                if iteration:
                    stage_spill *= float(np.exp(rng.normal(0.0, params["sigma_S"])))
                spill = stage_spill / n * share
                g = float(out["g"][k])
                r = float(out["r"][k])
                memory = (float(params["omega0"][k]) + float(params["omega1"][k]) * v_in
                          * (1.0 + float(params["omega2"][k]) * cfg.d.get("skew", 0.0))
                          ) * rng.lognormal(0.0, float(params["sigma_W"]), count)
                if out["S"][k] > 0:
                    memory = np.minimum(memory, cfg.U_e / cfg.c_e)
                features = tier3.task_features(v_in, v_sh, spill, g, r, cfg.m, B_disk, B_net)
                duration = (
                    float(params["theta1"][k]) * speed * features["compute"]
                    + float(params["theta_r"][k]) * speed * features["recompute"]
                    + float(params["theta4"][k]) * features["spill"]
                    + float(params["theta5"][k]) * features["shuffle"]
                ) * rng.lognormal(0.0, float(params["sigma_task"]), count)
                stage_rows.append({
                    "run_id": run_id,
                    "stage_signature": stage.signature,
                    "iteration": iteration,
                    "parent_signatures": list(stage.parents),
                    "stage_type": stage.stage_type,
                    "reads_cache": stage.reads_cache,
                    "submission_time": clock,
                    "V_read": float(out["V_in"][k]) if stage.kind == "input" else 0.0,
                    "V_shuffle_read": float(out["V_in"][k]) if stage.kind == "shuffle" else 0.0,
                    "V_in": float(out["V_in"][k]),
                    "V_sh": float(out["V_sh"][k]),
                    "V_out": float(out["V_out"][k]),
                    "remote_shuffle_bytes": float(out["V_in"][k]) * (cfg.m - 1) / cfg.m,
                    "n": n,
                    "S_memory": stage_spill * 3.0,
                    "S": stage_spill,
                    "g": g,
                    "compute_time": float(out["term_compute"][k]),
                    "W_p95": float(np.quantile(memory, 0.95)),
                    "T_k": T_k,
                    "C": C_observed,
                    "cpu_util": float(out["u_k"][k]),
                })
                clock += T_k
                task_rows.append(pd.DataFrame({
                    "run_id": run_id,
                    "stage_signature": stage.signature,
                    "iteration": iteration,
                    "task_id": np.arange(count),
                    "duration": duration,
                    "gc_time": duration * g,
                    "cpu_time": duration * (1.0 - g),
                    "peak_execution_memory": memory,
                    "v_in": v_in,
                    "v_sh": v_sh,
                    "spill": spill,
                }))

    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(stage_rows, columns=list(STAGE_COLUMNS)),
        pd.concat(task_rows, ignore_index=True)[list(TASK_COLUMNS)],
    )
