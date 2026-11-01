"""The runs / stages / tasks Parquet tables, and ingestion of listener output.

The design names two tables. Task-level likelihoods (decisions 2 and 3) need each task's
peak memory and duration, which a per-stage row cannot hold, so tasks get a third.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tscm.collect import signatures
from tscm.model import Config, GiB, Machine

RUN_COLUMNS = (
    "run_id", "job_id", "family", "s", "m", "machine_type", "c_h", "R_h", "B_disk", "B_net",
    "rate", "e", "c_e", "M_e", "f_spark", "partition_bytes", "shuffle_partitions",
    "start_time", "end_time", "T", "M_peak", "cluster_load", "lease_id", "node_partition",
    "sharing_mode", "co_runners", "co_runner_shuffle_rate", "co_runner_disk_rate",
    "L_node", "L_net", "L_disk", "node_seconds", "source", "phase",
)
STAGE_COLUMNS = (
    "run_id", "stage_signature", "iteration", "parent_signatures", "stage_type", "reads_cache",
    "submission_time", "V_read", "V_shuffle_read", "V_in", "V_sh", "V_out", "remote_shuffle_bytes",
    "n", "S_memory", "S", "g", "compute_time", "W_p95", "T_k", "C", "cpu_util",
)
TASK_COLUMNS = (
    "run_id", "stage_signature", "iteration", "task_id", "duration", "gc_time", "cpu_time",
    "peak_execution_memory", "v_in", "v_sh", "spill",
)
TABLES = {"runs": RUN_COLUMNS, "stages": STAGE_COLUMNS, "tasks": TASK_COLUMNS}


@dataclass(frozen=True)
class Store:
    """Each append writes one Parquet part, so concurrent runs never rewrite a table."""

    root: Path

    def append(self, table: str, frame: pd.DataFrame) -> None:
        missing = set(TABLES[table]) - set(frame.columns)
        if missing:
            raise ValueError(f"{table} frame is missing columns {sorted(missing)}")
        directory = Path(self.root) / table
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(directory / f"part-{uuid.uuid4().hex}.parquet", index=False)

    def load(self, table: str, job_id: str | None = None) -> pd.DataFrame:
        parts = sorted((Path(self.root) / table).glob("*.parquet"))
        if not parts:
            return pd.DataFrame(columns=list(TABLES[table]))
        frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if job_id is None:
            return frame
        if table == "runs":
            return frame[frame["job_id"] == job_id].reset_index(drop=True)
        runs = self.load("runs", job_id)["run_id"]
        return frame[frame["run_id"].isin(runs)].reset_index(drop=True)


def machine_from_row(row) -> Machine:
    return Machine(
        name=row["machine_type"],
        cores=int(row["c_h"]),
        memory=float(row["R_h"]),
        disk_bw=float(row["B_disk"]),
        net_bw=float(row["B_net"]),
        rate=float(row.get("rate", 1.0)),
    )


def config_from_row(row) -> Config:
    """Rebuild the Config of a run. Job parameters and d features live in p_* and d_* columns."""
    prefixed = lambda prefix: {  # noqa: E731
        key[len(prefix):]: float(value) for key, value in row.items()
        if key.startswith(prefix) and pd.notna(value)
    }
    return Config(
        s=float(row["s"]),
        m=int(row["m"]),
        machine=machine_from_row(row),
        M_e=float(row["M_e"]),
        e=int(row["e"]),
        c_e=int(row["c_e"]),
        p=prefixed("p_"),
        d=prefixed("d_"),
        partition_bytes=float(row["partition_bytes"]),
        shuffle_partitions=int(row["shuffle_partitions"]),
        f_spark=float(row["f_spark"]),
        L_node=float(row.get("L_node", 0.0)),
        L_net=float(row.get("L_net", 0.0)),
        L_disk=float(row.get("L_disk", 0.0)),
    )


def run_row(cfg: Config, run_id: str, job_id: str, family: str, **fields) -> dict:
    """A runs-table row for a configuration, with defaults for lease bookkeeping."""
    row = {
        "run_id": run_id,
        "job_id": job_id,
        "family": family,
        "s": cfg.s,
        "m": cfg.m,
        "machine_type": cfg.machine.name,
        "c_h": cfg.machine.cores,
        "R_h": cfg.machine.memory,
        "B_disk": cfg.machine.disk_bw,
        "B_net": cfg.machine.net_bw,
        "rate": cfg.machine.rate,
        "e": cfg.e,
        "c_e": cfg.c_e,
        "M_e": cfg.M_e,
        "f_spark": cfg.f_spark,
        "partition_bytes": cfg.partition_bytes,
        "shuffle_partitions": cfg.shuffle_partitions,
        "start_time": np.nan,
        "end_time": np.nan,
        "T": np.nan,
        "M_peak": np.nan,
        "cluster_load": 0.0,
        "lease_id": "",
        "node_partition": "",
        "sharing_mode": "partitioned",
        "co_runners": "[]",
        "co_runner_shuffle_rate": 0.0,
        "co_runner_disk_rate": 0.0,
        "L_node": cfg.L_node,
        "L_net": cfg.L_net,
        "L_disk": cfg.L_disk,
        "node_seconds": np.nan,
        "source": "profiling",
        "phase": "",
        **{f"p_{k}": v for k, v in cfg.p.items()},
        **{f"d_{k}": v for k, v in cfg.d.items()},
    }
    row.update(fields)
    return row


def read_events(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ingest_listener(
    events: list[dict],
    cfg: Config,
    run_id: str,
    job_id: str,
    family: str,
    **run_fields,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Turn one application's listener events into (runs, stages, tasks) rows.

    Only each stage's last attempt is kept, and failed attempts are dropped, so the
    tables describe the execution that actually produced the result.
    """
    plans = {e["execution_id"]: e["physical_plan"] for e in events if e["event"] == "sql"}
    sql_of_stage = {
        stage_id: e.get("sql_execution_id")
        for e in events if e["event"] == "job"
        for stage_id in e["stage_ids"]
    }
    finished = {}
    for event in events:
        if event["event"] == "stage" and not event.get("failed"):
            finished[event["stage_id"]] = event  # later attempts overwrite earlier ones

    ordered = sorted(finished.values(), key=lambda e: (e["submission_time"], e["stage_id"]))
    raw_signature = {}
    for event in ordered:
        plan = plans.get(sql_of_stage.get(event["stage_id"]))
        raw_signature[event["stage_id"]] = signatures.stage_signature(
            scopes=event.get("rdd_scopes", []),
            call_site=event.get("call_site", ""),
            rdd_types=event.get("rdd_types", []),
            sql_plan=plan,
        )
    indexed = signatures.assign_iterations(
        [(event["stage_id"], raw_signature[event["stage_id"]]) for event in ordered])

    tasks_by_stage: dict[tuple[int, int], list[dict]] = {}
    for event in events:
        if event["event"] == "task" and event.get("successful", True):
            key = (event["stage_id"], event["stage_attempt"])
            tasks_by_stage.setdefault(key, []).append(event)

    stage_rows, task_rows = [], []
    peak_memory = 0.0
    for event in ordered:
        stage_id = event["stage_id"]
        signature, iteration = indexed[stage_id]
        tasks = tasks_by_stage.get((stage_id, event["attempt"]), [])
        run_time = sum(t["executor_run_time_ms"] for t in tasks)
        gc = sum(t["jvm_gc_time_ms"] for t in tasks)
        read = sum(t["input_bytes"] for t in tasks)
        shuffle_read = sum(t["shuffle_local_bytes"] + t["shuffle_remote_bytes"] for t in tasks)
        peaks = [t["peak_execution_memory"] for t in tasks]
        window = max((event["completion_time"] - event["submission_time"]) / 1000.0, 1e-3)
        cpu_seconds = sum(t["executor_cpu_time_ns"] for t in tasks) / 1e9
        stage_rows.append({
            "run_id": run_id,
            "stage_signature": signature,
            "iteration": iteration,
            "parent_signatures": [
                indexed[p][0] for p in event.get("parent_ids", []) if p in indexed
            ],
            "stage_type": signatures.classify_stage(event.get("rdd_scopes", [])),
            "reads_cache": bool(event.get("reads_cache", False)),
            "submission_time": event["submission_time"] / 1000.0,
            "V_read": float(read),
            "V_shuffle_read": float(shuffle_read),
            # V_in counts shuffle reads: a shuffle-reading stage reads no input files, and
            # log V_k would otherwise be undefined for every such stage.
            "V_in": float(read + shuffle_read),
            "V_sh": float(sum(t["shuffle_write_bytes"] for t in tasks)),
            "V_out": float(sum(t["output_bytes"] for t in tasks)),
            "remote_shuffle_bytes": float(sum(t["shuffle_remote_bytes"] for t in tasks)),
            "n": float(event["num_tasks"]),
            "S_memory": float(sum(t["memory_bytes_spilled"] for t in tasks)),
            "S": float(sum(t["disk_bytes_spilled"] for t in tasks)),
            "g": gc / run_time if run_time > 0 else 0.0,
            "compute_time": cpu_seconds,
            "W_p95": float(np.quantile(peaks, 0.95)) if peaks else 0.0,
            "T_k": window,
            "C": float(event.get("storage_memory_max", 0.0)),
            "cpu_util": min(cpu_seconds / (window * cfg.N), 1.0),
        })
        peak_memory = max(peak_memory, float(event.get("executor_peak_memory", 0.0)))
        for task in tasks:
            task_rows.append({
                "run_id": run_id,
                "stage_signature": signature,
                "iteration": iteration,
                "task_id": task["task_id"],
                "duration": task["executor_run_time_ms"] / 1000.0,
                "gc_time": task["jvm_gc_time_ms"] / 1000.0,
                "cpu_time": task["executor_cpu_time_ns"] / 1e9,
                "peak_execution_memory": float(task["peak_execution_memory"]),
                "v_in": float(task["input_bytes"] + task["shuffle_local_bytes"]
                              + task["shuffle_remote_bytes"]),
                "v_sh": float(task["shuffle_write_bytes"]),
                "spill": float(task["disk_bytes_spilled"]),
            })

    span = [e for e in events if e["event"] in ("app_start", "app_end")]
    start = min((e["time"] for e in span), default=float("nan")) / 1000.0
    end = max((e["time"] for e in span), default=float("nan")) / 1000.0
    run = run_row(cfg, run_id, job_id, family, start_time=start, end_time=end, T=end - start,
                  M_peak=peak_memory, node_seconds=(end - start) * cfg.m, **run_fields)
    return (
        pd.DataFrame([run]),
        pd.DataFrame(stage_rows, columns=list(STAGE_COLUMNS)),
        pd.DataFrame(task_rows, columns=list(TASK_COLUMNS)),
    )


def human_bytes(value: float) -> str:
    return f"{value / GiB:.2f} GiB"
