from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tscm.collect import data_profile, gpu_metrics, signatures, spec, store
from tscm.model import Config, GiB, MiB
from tscm.simulate import DEFAULT_MACHINE

FIXTURES = Path(__file__).parent / "fixtures"
REAL_RUNS = ("spark_events_2000000.jsonl", "spark_events_4000000.jsonl")


def config(s=1e8):
    return Config(s=s, m=1, machine=DEFAULT_MACHINE, M_e=2 * GiB, e=1, c_e=4, shuffle_partitions=8)


def ingest(name):
    return store.ingest_listener(store.read_events(FIXTURES / name), config(), name, "real", "sql")


@pytest.fixture(scope="module")
def real():
    return [ingest(name) for name in REAL_RUNS]


# Real listener output, captured from Spark 3.5.1 on two input sizes.


def test_real_runs_have_identical_signatures_across_input_sizes(real):
    # Regression: plans of cached data contain a JVM identity hash that changes every run,
    # which gave each run its own signatures.
    first, second = (set(tables[1]["stage_signature"]) for tables in real)
    assert first == second


def test_real_plans_normalize_equal_across_runs():
    plans = [
        [e["physical_plan"] for e in store.read_events(FIXTURES / name) if e["event"] == "sql"]
        for name in REAL_RUNS
    ]
    assert len(plans[0]) == len(plans[1]) > 0
    for a, b in zip(*plans, strict=True):
        assert signatures.normalize_plan(a) == signatures.normalize_plan(b)


def test_real_events_fill_the_tables(real):
    runs, stages, tasks = real[0]
    assert len(stages) > 10 and len(tasks) > len(stages)
    assert (stages["T_k"] > 0).all() and (runs["T"] > 0).all() and (runs["M_peak"] > 0).all()
    assert (stages["C"] > 0).any() and stages["reads_cache"].any()
    assert (stages["V_shuffle_read"] > 0).any()


def test_shuffle_reads_count_as_stage_input(real):
    _, stages, _ = real[0]
    shuffle = stages[stages["V_shuffle_read"] > 0]
    assert (shuffle["V_in"] == shuffle["V_read"] + shuffle["V_shuffle_read"]).all()


def test_repeated_real_stages_get_iteration_indices(real):
    _, stages, _ = real[0]
    assert stages["iteration"].max() >= 1
    keys = stages[["stage_signature", "iteration"]].drop_duplicates()
    assert len(keys) == len(stages)


def test_real_parents_are_signatures_of_earlier_stages(real):
    _, stages, _ = real[0]
    known = set(stages["stage_signature"])
    linked = [p for parents in stages["parent_signatures"] for p in parents]
    assert linked and set(linked) <= known


def minimal_events(failed=False):
    stage = {"event": "stage", "stage_id": 0, "attempt": 0, "num_tasks": 1, "parent_ids": [],
             "rdd_scopes": ["Scan"], "rdd_types": ["R"], "call_site": "c", "reads_cache": False,
             "submission_time": 1000, "completion_time": 3000, "failed": failed,
             "storage_memory_max": 0, "executor_peak_memory": 5}
    task = {"event": "task", "stage_id": 0, "stage_attempt": 0, "task_id": 1, "successful": True,
            "executor_run_time_ms": 1500, "executor_cpu_time_ns": int(1e9), "jvm_gc_time_ms": 150,
            "peak_execution_memory": 7, "memory_bytes_spilled": 0, "disk_bytes_spilled": 0,
            "input_bytes": 100, "output_bytes": 0, "shuffle_write_bytes": 0,
            "shuffle_local_bytes": 0, "shuffle_remote_bytes": 0}
    return [
        {"event": "app_start", "time": 0}, task, stage, {"event": "app_end", "time": 5000},
    ]


def test_failed_attempts_are_dropped():
    _, stages, _ = store.ingest_listener(minimal_events(failed=True), config(), "r", "j", "f")
    assert stages.empty


def test_gc_fraction_is_a_ratio_of_sums():
    runs, stages, tasks = store.ingest_listener(minimal_events(), config(), "r", "j", "f")
    assert stages["g"].iloc[0] == pytest.approx(0.1)
    assert stages["T_k"].iloc[0] == pytest.approx(2.0)
    assert runs["T"].iloc[0] == pytest.approx(5.0)
    assert tasks["duration"].iloc[0] == pytest.approx(1.5)


# Signatures.


def test_normalization_drops_what_changes_between_runs():
    a = "Range (0, 2000000, step=1) #12L [plan_id=24] Foo@165391b6 'x' /data/points-200"
    b = "Range (0, 4000000, step=1) #98L [plan_id=91] Foo@898aa19 'y' /data/points-400"
    assert signatures.normalize_plan(a) == signatures.normalize_plan(b)


def test_normalization_keeps_operators_and_key_expressions():
    a = signatures.normalize_plan("HashAggregate(keys=[user#1])")
    b = signatures.normalize_plan("HashAggregate(keys=[item#1])")
    assert a != b


def test_rdd_signature_uses_call_site_and_types():
    a = signatures.stage_signature([], "map at Job.scala:42", ["MapPartitionsRDD"])
    b = signatures.stage_signature([], "map at Job.scala:42", ["MapPartitionsRDD"])
    c = signatures.stage_signature([], "reduce at Job.scala:42", ["MapPartitionsRDD"])
    assert a == b != c


def test_stages_of_one_plan_are_separated_by_their_operators():
    a = signatures.stage_signature(["Exchange"], sql_plan="plan")
    b = signatures.stage_signature(["HashAggregate"], sql_plan="plan")
    assert a != b


def test_iterations_are_assigned_in_submission_order():
    result = signatures.assign_iterations([(5, "x"), (6, "y"), (7, "x"), (8, "x")])
    assert result[5] == ("x", 0) and result[7] == ("x", 1) and result[8] == ("x", 2)
    assert result[6] == ("y", 0)


@pytest.mark.parametrize("scopes, expected", [
    (["BroadcastHashJoin"], "join"), (["HashAggregate"], "aggregation"),
    (["Exchange"], "shuffle_map"), (["Scan parquet"], "scan"), (["mapPartitions"], "other"),
])
def test_stage_classification(scopes, expected):
    assert signatures.classify_stage(scopes) == expected


# Store.


def test_store_round_trip_and_job_filter(tmp_path):
    table = store.Store(tmp_path)
    for job in ("a", "b"):
        runs, stages, tasks = store.ingest_listener(minimal_events(), config(), job, job, "f")
        table.append("runs", runs)
        table.append("stages", stages)
        table.append("tasks", tasks)
    assert len(table.load("runs")) == 2
    assert table.load("stages", "a")["run_id"].tolist() == ["a"]
    assert table.load("stages", "a")["parent_signatures"].iloc[0].tolist() == []


def test_store_rejects_missing_columns(tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        store.Store(tmp_path).append("runs", pd.DataFrame({"run_id": ["x"]}))


def test_loading_an_empty_store_gives_the_schema(tmp_path):
    assert list(store.Store(tmp_path).load("tasks").columns) == list(store.TASK_COLUMNS)


def test_config_round_trips_through_a_run_row():
    cfg = Config(s=5e9, m=4, machine=DEFAULT_MACHINE, M_e=8 * GiB, e=2, c_e=3,
                 p={"k": 4.0}, d={"skew": 0.2}, shuffle_partitions=64, L_node=1.0)
    assert store.config_from_row(store.run_row(cfg, "r", "j", "f")) == cfg


# Job specification.


DESIGN_EXAMPLE = """
job: kmeans
family: iterative_ml
framework: spark
entry: jobs/kmeans.py
params:
  k: {type: int, profile: [4, 8]}
  max_iter: {type: int, profile: [10]}
data:
  path: data/points/
  features: [record_count, mean_record_size, sparsity]
profile:
  input_fractions: [0.005, 0.01, 0.02, 0.04, 0.08]
  machines: [2, 4, 8, 16]
"""


def test_the_design_example_parses(tmp_path):
    path = tmp_path / "kmeans.yaml"
    path.write_text(DESIGN_EXAMPLE)
    parsed = spec.load_spec(path)
    assert parsed.job == "kmeans" and parsed.machines == (2, 4, 8, 16)
    assert parsed.params[0] == spec.ParamSpec("k", "int", (4, 8))


@pytest.mark.parametrize("change, message", [
    ({"framework": "flink"}, "framework"),
    ({"params": {"k": {"type": "int", "profile": ["four"]}}}, "not int"),
    ({"params": {"k": {"type": "int", "profile": [True]}}}, "not int"),
    ({"params": {"k": {"type": "complex"}}}, "type must be"),
    ({"data": {"features": []}}, "data.path"),
    ({"data": {"path": "x", "features": ["vibes"]}}, "unknown data feature"),
    ({"profile": {"input_fractions": [1.5], "machines": [2]}}, r"\(0, 1\]"),
    ({"profile": {"input_fractions": [0.1], "machines": [0]}}, "positive integers"),
])
def test_bad_specs_are_rejected(change, message):
    import yaml

    raw = {**yaml.safe_load(DESIGN_EXAMPLE), **change}
    with pytest.raises(spec.SpecError, match=message):
        spec.parse_spec(raw)


def test_all_spec_problems_are_listed_together():
    with pytest.raises(spec.SpecError) as caught:
        spec.parse_spec({"framework": "flink"})
    message = str(caught.value)
    assert "missing required key 'job'" in message and "framework must" in message


# Data characteristics.


def test_hyperloglog_is_accurate_at_both_ranges():
    for count in (500, 200_000):
        estimate = data_profile.hyperloglog(pd.Series(np.arange(count)))
        assert estimate == pytest.approx(count, rel=0.03)


def test_top_key_share_detects_skew():
    uniform = pd.Series(np.arange(1000))
    assert data_profile.top_key_share(uniform) == pytest.approx(0.01)
    skewed = pd.Series([0] * 900 + list(range(1, 101)))
    assert data_profile.top_key_share(skewed) > 0.89


def test_profile_frame_scales_record_count_by_the_sample_rate():
    frame = pd.DataFrame({"key": [1, 1, 2, 3], "x": [0.0, 1.0, 0.0, 2.0]})
    features = data_profile.profile_frame(frame, "key", sample_rate=0.001, size_bytes=4000.0)
    assert features["record_count"] == 4000.0
    assert features["mean_record_size"] == 1.0


def test_profile_parquet_samples_without_loading_everything(tmp_path):
    frame = pd.DataFrame({"key": np.arange(20_000) % 100, "x": np.ones(20_000)})
    frame.to_parquet(tmp_path / "part.parquet")
    features = data_profile.profile_parquet(tmp_path, "key", sample_rate=0.1)
    assert features["record_count"] == 20_000.0
    assert features["distinct_keys"] == pytest.approx(100, rel=0.05)


def test_empty_sample_does_not_crash():
    assert data_profile.profile_frame(pd.DataFrame({"k": []}), "k")["record_count"] == 0.0


# GPU telemetry.


def test_gpu_sample_parsers():
    assert gpu_metrics.parse_nvidia_smi("0, 1024, 87").memory_mib == 1024.0
    assert gpu_metrics.parse_nvidia_smi("index, memory.used, util") is None
    assert gpu_metrics.parse_dcgm("GPU 1  2048  55  0.1  1  2  3").gpu == 1
    assert gpu_metrics.parse_dcgm("#Entity FB_USED") is None


def test_step_recorder_discards_warmup():
    recorder = gpu_metrics.StepRecorder(warmup=2)
    for _ in range(5):
        with recorder.step() as step:
            with step.phase("data_loading"):
                pass
    assert len(recorder.records) == 3
    summary = recorder.summary()
    assert summary["T_step"] >= summary["data_loading"] >= 0.0


def test_sampler_prefers_dcgm_when_installed(monkeypatch):
    monkeypatch.setattr(gpu_metrics.shutil, "which", lambda name: "/usr/bin/dcgmi")
    assert gpu_metrics.sampler_command()[0] == "dcgmi"
    monkeypatch.setattr(gpu_metrics.shutil, "which", lambda name: None)
    assert gpu_metrics.sampler_command()[0] == "nvidia-smi"


def test_human_bytes():
    assert store.human_bytes(512 * MiB) == "0.50 GiB"
