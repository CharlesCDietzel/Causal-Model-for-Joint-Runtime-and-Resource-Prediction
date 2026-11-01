"""From store tables to the arrays every fitter reads."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tscm.collect.store import config_from_row
from tscm.model import Config, Hardware, Job, Stage, config_arrays, stack_configs

HISTORICAL_WEIGHT = 0.5
MAX_TASKS_PER_STAGE_RUN = 16


def build_job(
    job_id: str,
    family: str,
    runs: pd.DataFrame,
    stages: pd.DataFrame,
    features: tuple[str, ...] = (),
    iteration_param: str | None = None,
) -> Job:
    """Derive the job's stage DAG from observed stages.

    Collapsing iterations onto one signature turns an iterative loop into a cycle, so
    parents are kept only when they come earlier in first-submission order.
    """
    ordered = stages.sort_values(["submission_time", "iteration"], kind="stable")
    signatures = list(dict.fromkeys(ordered["stage_signature"]))
    position = {signature: i for i, signature in enumerate(signatures)}
    partitions = runs.set_index("run_id")["partition_bytes"]
    shuffle_partitions = runs.set_index("run_id")["shuffle_partitions"]

    result = []
    for signature in signatures:
        rows = stages[stages["stage_signature"] == signature]
        parents = {p for listed in rows["parent_signatures"] for p in listed}
        earlier = position[signature]
        parents = tuple(sorted(p for p in parents if position.get(p, earlier) < earlier))
        iterations = rows.groupby("run_id")["iteration"].max() + 1
        repeated = bool((iterations > 1).any())

        size_tasks = np.maximum(
            np.ceil(rows["V_in"].to_numpy() / rows["run_id"].map(partitions).to_numpy()), 1.0)
        n = rows["n"].to_numpy()
        if (rows["V_shuffle_read"] > 0).all() and np.array_equal(
                n, rows["run_id"].map(shuffle_partitions).to_numpy()):
            kind = "shuffle"
        elif np.array_equal(n, size_tasks):
            kind = "input"
        elif np.unique(n).size == 1:
            kind = "fixed"
        else:
            kind = "input"

        result.append(Stage(
            signature=signature,
            kind=kind,
            parents=parents,
            stage_type="iteration_body" if repeated else str(rows["stage_type"].mode().iloc[0]),
            writes_shuffle=bool((rows["V_sh"] > 0).any()),
            writes_output=bool((rows["V_out"] > 0).any()),
            reads_cache=bool(rows["reads_cache"].any()),
            repeat_param=iteration_param if repeated else None,
            repeat=float(iterations.mean()),
            fixed_tasks=int(n[0]),
        ))
    return Job(job_id, family, tuple(result), features, cached=bool((stages["C"] > 0).any()))


@dataclass
class FitData:
    """Observations aligned on run index r and stage index k."""

    job: Job
    hardware: Hardware
    run_ids: list[str]
    cfgs: list[Config]
    x: dict[str, np.ndarray]
    weight: np.ndarray  # per run; historical runs are down-weighted
    T: np.ndarray
    M_peak: np.ndarray
    # One row per stage execution.
    r: np.ndarray
    k: np.ndarray
    V_in: np.ndarray
    V_sh: np.ndarray
    V_out: np.ndarray
    n: np.ndarray
    S: np.ndarray
    g: np.ndarray
    T_k: np.ndarray
    C: np.ndarray
    # One row per sampled task.
    t_r: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    t_k: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    t_row: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    t_duration: np.ndarray = field(default_factory=lambda: np.empty(0))
    t_memory: np.ndarray = field(default_factory=lambda: np.empty(0))
    t_v_in: np.ndarray = field(default_factory=lambda: np.empty(0))
    t_v_sh: np.ndarray = field(default_factory=lambda: np.empty(0))
    t_spill: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def n_runs(self) -> int:
        return len(self.run_ids)

    def stage_run_mean(self, values: np.ndarray) -> np.ndarray:
        """Mean over iterations, as a (runs, stages) array; NaN where a stage did not run."""
        K = len(self.job.stages)
        total = np.zeros((self.n_runs, K))
        count = np.zeros((self.n_runs, K))
        np.add.at(total, (self.r, self.k), values)
        np.add.at(count, (self.r, self.k), 1.0)
        with np.errstate(invalid="ignore"):
            return total / count


def build_fit_data(
    job: Job,
    runs: pd.DataFrame,
    stages: pd.DataFrame,
    tasks: pd.DataFrame | None = None,
    hardware: Hardware | None = None,
    seed: int = 0,
) -> FitData:
    hardware = hardware or Hardware()
    runs = runs.reset_index(drop=True)
    run_ids = runs["run_id"].tolist()
    run_index = {run_id: i for i, run_id in enumerate(run_ids)}
    cfgs = [config_from_row(row) for row in runs.to_dict("records")]

    stages = stages[stages["run_id"].isin(run_index)
                    & stages["stage_signature"].isin(job.index)].reset_index(drop=True)
    r = stages["run_id"].map(run_index).to_numpy(dtype=int)
    k = stages["stage_signature"].map(job.index).to_numpy(dtype=int)
    weight = np.where(runs["source"].to_numpy() == "historical", HISTORICAL_WEIGHT, 1.0)
    data = FitData(
        job=job,
        hardware=hardware,
        run_ids=run_ids,
        cfgs=cfgs,
        x=stack_configs([config_arrays(job, cfg, hardware) for cfg in cfgs]),
        weight=weight,
        T=runs["T"].to_numpy(dtype=float),
        M_peak=runs["M_peak"].to_numpy(dtype=float),
        r=r,
        k=k,
        V_in=stages["V_in"].to_numpy(dtype=float),
        V_sh=stages["V_sh"].to_numpy(dtype=float),
        V_out=stages["V_out"].to_numpy(dtype=float),
        n=stages["n"].to_numpy(dtype=float),
        S=stages["S"].to_numpy(dtype=float),
        g=stages["g"].to_numpy(dtype=float),
        T_k=stages["T_k"].to_numpy(dtype=float),
        C=stages["C"].to_numpy(dtype=float),
    )
    if tasks is None or tasks.empty:
        return data

    # Sampling caps the likelihood size; decision 3 needs tasks as observations, not all of them.
    keep = tasks[tasks["run_id"].isin(run_index) & tasks["stage_signature"].isin(job.index)]
    keep = keep.groupby(["run_id", "stage_signature", "iteration"], group_keys=False).sample(
        frac=1.0, random_state=seed)
    keep = keep.groupby(["run_id", "stage_signature", "iteration"]).head(MAX_TASKS_PER_STAGE_RUN)
    keep = keep[keep["duration"] > 0]
    row_of = stages.reset_index().set_index(["run_id", "stage_signature", "iteration"])["index"]
    rows = row_of.reindex(
        pd.MultiIndex.from_frame(keep[["run_id", "stage_signature", "iteration"]])).to_numpy()
    matched = ~np.isnan(rows.astype(float))
    keep, rows = keep[matched], rows[matched].astype(int)

    data.t_r = keep["run_id"].map(run_index).to_numpy(dtype=int)
    data.t_k = keep["stage_signature"].map(job.index).to_numpy(dtype=int)
    data.t_row = rows
    data.t_duration = keep["duration"].to_numpy(dtype=float)
    data.t_memory = keep["peak_execution_memory"].to_numpy(dtype=float)
    data.t_v_in = keep["v_in"].to_numpy(dtype=float)
    data.t_v_sh = keep["v_sh"].to_numpy(dtype=float)
    data.t_spill = keep["spill"].to_numpy(dtype=float)
    return data
