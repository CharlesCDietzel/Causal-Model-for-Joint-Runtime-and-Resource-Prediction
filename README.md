# Causal Model for Joint Runtime and Resource Prediction

A tiered structural causal model (TSCM) that predicts a job's runtime and resource use, per
stage, for any input size, machine count, machine type and memory allocation. It covers batch
Spark jobs and distributed PyTorch training. The design is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned in `.python-version`.

```bash
uv sync                      # core package
uv sync --extra chameleon    # adds python-chi for lease management
uv sync --extra gpu          # adds torch for step and memory hooks
uv run pytest -m "not slow"  # fast tests
uv run pytest                # includes NUTS fits
uv run python examples/quickstart.py
```

The Scala listener builds separately; see
[tscm/collect/listener/README.md](tscm/collect/listener/README.md).

## Layout

| Path | Contents |
| --- | --- |
| `tscm/model.py` | Tier 0 config, stage DAG, and the parameter table every fitter shares |
| `tscm/collect/` | Listener, stage signatures, data characteristics, GPU telemetry, Parquet store, job specs |
| `tscm/graph/` | Tiers, required and forbidden edges, constrained discovery |
| `tscm/mechanisms/` | Structural equations for each tier, GPU jobs and the residual learner |
| `tscm/fit/` | Modular fit, joint refinement, NumPyro model, cross-job transfer, GPU fitting |
| `tscm/design/` | Profiling phases, lease packing and SU budget, Chameleon integration |
| `tscm/query/` | Monte Carlo forward pass, the supported queries, extrapolation guards |
| `tscm/eval/` | Splits, baselines, metrics and ablations |
| `tscm/simulate.py` | Synthetic cluster with known parameters, for tests and the example |

## Where the code differs from ARCHITECTURE.md

The design could not be implemented literally in these places.

| Topic | Design | Code | Why |
| --- | --- | --- | --- |
| `h(ρ)` | `(1 − 1/ρ) σ(...)` | First factor clamped at 0 | Unclamped, spill is negative for ρ < 1 |
| Stage input | `V_in` is `inputMetrics.bytesRead` | Adds shuffle reads | Shuffle-reading stages read no input files, so `log V` is undefined for them |
| Cached-data recompute | Described, but no term in `T_k` | Added `θ_r · r_k · V_in / (n_k (1 − g_k))` in the waves bracket | The memory cliff of the top-priority workload family has no other path into runtime |
| `U_storage` | Undefined | The whole unified pool, `U_e` | Storage can borrow all of it while execution is idle |
| Storage | Two tables | Adds a `tasks` table | Decisions 2 and 3 need per-task peak memory and duration |
| `u_k` | Compute time / `T_k` | Also scaled by slots filled, `n_k / (w_k N)` | Otherwise 4 tasks on 16 slots report full utilization |
| Spill base | `V_sh` | Falls back to `V_in` when a stage writes no shuffle | A final sort still spills |
| Task counts | Input or shuffle stages | Adds `fixed`, for stages whose count never changes | RDD jobs set partitions outside `spark.sql.shuffle.partitions` |
| Family priors | Hierarchical model | Empirical Bayes from per-job posteriors | Adding a job costs one fit, not a refit of the family |
| `M_e → ∞` | Unlimited memory | 16× the larger of the zero-spill estimate and `M_e` | `log M_e` in the GC term has no finite limit |

Normalization of stage signatures also strips JVM identity hashes (`Foo@165391b6`), which the
design does not mention. Plans of cached data contain them, and they differ on every run.

## Status

**Tested with real Spark.** The listener compiles against Spark 3.5.1, and real local-mode runs
were ingested and fitted. Two of those event logs are the fixtures in `tests/fixtures/`. The
sample is small: four input sizes of one job, where 3 of 12 stages exceeded the 25% MAPE check
in-sample (worst 74%). Two runs with memory shrunk to 450 and 600 MB died with a JVM
`OutOfMemoryError` instead of spilling, so Phase A step 2 is unproven. Local mode shares one heap
between driver and executor, which may be the cause, but this needs checking on a cluster.

**Tested on simulated data only.** Fitting, queries and profiling design have only been checked
against `tscm/simulate.py`. That simulator shares the model's equations, so these tests show the
code recovers known parameters, not that the model describes real Spark jobs.

**Never run against real systems:**

- Multi-machine Spark: real runs so far are single-machine local mode, so `θ_2`, `θ_3` and the
  shuffle terms have never seen real data.
- GPU: telemetry and hooks are unit-tested on parsers and timers. No GPU has run them.
- Chameleon: lease calls are tested against a stand-in for python-chi. The shell scripts have
  not been executed.
- `sbt package`: the listener was compiled directly with `scalac`, not through `build.sbt`.

**Known gaps:**

- NUTS takes about 5 minutes for 13 runs on 16 cores, not the design's "seconds to a minute".
  It converges (R̂ 1.03), but about 3% of draws diverge. The stiff direction is `θ_2`, `θ_3`
  and the per-run effects, which a run-level shift can trade off against each other.
- Edges found by discovery are reported and logged, but not yet added to the structural
  equations. Discovery is skipped below 30 jobs, the expected v1 setting.
- The residual learner corrects job runtime only, not each Tier 3 node.
- Baselines needing their authors' models or data are not included: Bellamy, DNNPerf, TraPPM,
  GPUMemNet and QualX. `eval/harness.py` accepts them through its `Baseline` protocol. Bell, C3O,
  KS+ and Ruya are simplified reimplementations.
- The Spark RAPIDS extension has its transfer term only, as planned for after v1.
