# Causal Model for Joint Runtime and Resource Prediction — Architecture Draft

Oct 31, 2026 · Charles Dietzel

## Overview

The model is a tiered structural causal model (TSCM) that predicts a job's runtime and resource use, per stage, for any input size, machine count, machine type and memory allocation. It is fitted from 8–12 cheap profiling runs, or from historical runs when they exist.

Four design choices address the weaknesses of a plain causal-discovery approach:

- **Most of the graph is fixed by domain knowledge.** Configuration variables are root nodes; data-volume mediators, resource-state mediators and outcomes sit in fixed tiers. Discovery runs only among resource-state mediators, which keeps it feasible with few runs.
- **Structural functions encode scaling laws.** Each node has a parametric form derived from how distributed jobs scale (Ernest's terms, data-volume power laws, memory-pressure thresholds). A residual learner is allowed only inside the observed support.
- **Memory regimes are modeled explicitly.** A memory-pressure ratio drives spill, GC and runtime through threshold functions, so the model can predict behavior in regimes that small-input runs would never reach.
- **Stages, not whole jobs, are the unit.** Spark's stage DAG gives the structural skeleton; job runtime is the critical-path sum of stage runtimes.

**Outputs:** per-stage and job runtime; peak executor memory; minimum memory requirement (a counterfactual query); CPU utilization; shuffle, spill and I/O bytes; GC fraction; prediction intervals for each. For GPU jobs, outputs also include peak and required GPU memory, GPU utilization and out-of-memory risk.

**Scope of v1:** batch Spark jobs on CPU and distributed GPU training jobs (PyTorch DDP or FSDP) on homogeneous clusters, one job at a time. Spark with the RAPIDS plugin follows as an extension after v1. Heterogeneous clusters and streaming are out of scope. Profiling and evaluation runs may share a cluster to save budget; the shared-cluster section below models the resulting interference.

## Problem formulation and notation

A job j runs as a DAG of stages k = 1…K_j under a configuration x. The goal is to learn P(Y | do(x)) for every outcome Y, per stage and for the whole job.

**Configuration (Tier 0, set by us or recorded):**

| Symbol | Meaning | Unit |
| --- | --- | --- |
| s | Input size | bytes |
| m | Number of worker machines | count |
| h | Machine type, expanded to cores c_h, memory R_h, disk and network bandwidth B_disk, B_net | categorical + numeric |
| e, c_e, M_e | Executors per machine, cores per executor, memory per executor | count, count, bytes |
| p | Job parameters (iterations, k, convergence threshold, partitions) | mixed |
| d | Data characteristics from a sample: record count, mean record size, distinct-key estimate, key skew (top-1% key share), sparsity | mixed |

Derived quantities used throughout: total task slots N = m · e · c_e, and usable execution memory per executor U_e = f_spark · (M_e − 300 MB), where f_spark is spark.memory.fraction (default 0.6).

**Mediators and outcomes (per stage k):**

| Symbol | Meaning | Unit |
| --- | --- | --- |
| V_k^in, V_k^sh, V_k^out | Stage input, shuffle-write and output volume | bytes |
| n_k | Number of tasks | count |
| W_k | Working set per task | bytes |
| ρ_k | Memory pressure ratio | unitless |
| S_k | Bytes spilled to disk | bytes |
| g_k | Fraction of task time in garbage collection | 0–1 |
| u_k | Mean CPU utilization of allocated cores | 0–1 |
| T_k | Stage wall-clock runtime | seconds |

**Job-level outcomes:** runtime T (critical path over T_k), peak executor memory M_peak, memory requirement M_req, total spill, mean CPU utilization, and, for GPU jobs, peak GPU memory G_peak, GPU memory requirement G_req and GPU utilization u_gpu.

**Additional variables for GPU jobs:**

| Symbol | Meaning | Tier |
| --- | --- | --- |
| n_g | GPUs per machine (N = m · n_g in total) | 0 |
| R_gpu, F_gpu, B_ic | GPU memory, peak throughput, interconnect bandwidth (PCIe within a machine, since Chameleon nodes lack NVLink; network across machines) | 0 |
| b, P, precision | Per-GPU batch size, model parameter count, numeric precision | 0 |
| shard, ckpt | Sharding mode (DDP, FSDP or ZeRO stage) and activation-checkpointing flag | 0 |
| G_k | GPU working set per GPU in phase k | 2 |
| ρ_gpu | GPU memory pressure G_k / R_gpu | 2 |
| T_step | Time per training step | 3 |

For training jobs, a "stage" is a phase of the training step (data loading, forward and backward, gradient communication); job runtime is steps × step time plus setup. Input size for training splits in two: dataset size sets steps per epoch and total runtime, while per-step inputs (batch size, sequence length or image size) set step time and memory.

All volumes and times are modeled in log space unless stated otherwise, so errors are relative.

## Causal skeleton

The graph has four tiers, and edges only point from lower to higher tiers. Configuration is set by us (or recorded), so it has no parents; discovery is confined to resource-state mediators.

*Diagram (see the online doc for the rendered version):*

```
Tier 0 · Config        Tier 1 · Volumes     Tier 2 · State (learned)   Tier 3 · Outcomes
- Input size s         - Stage volumes V    - Working set W            - Stage runtime
- Machines m           - Task count n         -> Memory pressure ρ     - Job runtime T
- Machine type h       - Cached data C           -> Spill S           - CPU utilization
- Executor memory                                -> GC fraction g     - Peak memory
- Job params p                              - GPU working set          - Min. memory
- Data traits d                               -> GPU pressure          - GPU step time
- GPUs and batch                                                       - OOM risk
- Co-location L                                                        - Peak GPU memory

Edges run left to right only. Configuration also enters Tiers 2 and 3 directly.
No edge points back into Tier 0, so a query do(x) is a plain forward pass.
```

Arrows between tiers stand for the fixed edges listed below; arrows inside a tier are within-tier edges.

**Required edges (fixed, from domain knowledge):**

- s, d, p → V_k (data volumes depend on input and parameters, not on machines)
- V_k, partition settings → n_k
- V_k, n_k, skew → W_k; W_k, C, c_e, M_e → ρ_k
- ρ_k → S_k, g_k; V_k^sh → S_k
- V_k, S_k, g_k, n_k, m, c_e, h → T_k; T_k → T along the stage DAG
- T_k components → u_k; W_k, C, M_e → M_peak and M_req

**Forbidden edges:**

- Anything into Tier 0 (outcomes never change the configuration).
- Any edge from a higher tier to a lower tier.
- m, M_e, h → V_k (machines and memory do not change data volumes; this is the test that catches plan changes such as adaptive query execution).

**Open for discovery:** edges among Tier 2 nodes beyond those above (e.g., GC → CPU contention, spill → GC), and Tier 2 → Tier 3 edges not in the required list.

**GPU jobs use the same tiers.** GPU count and type, batch size, model size and sharding mode join Tier 0; GPU working set and GPU memory pressure join Tier 2; step time, peak GPU memory, out-of-memory risk and GPU utilization join Tier 3 (equations under GPU jobs in Structural functions).

## Graph learning

Discovery is restricted to edges among Tier 2 mediators and from Tier 2 into Tier 3; everything else is fixed by the skeleton. It runs on data pooled across many jobs, not per job, because a single job's 8–12 runs cannot support it.

**Procedure:**

1. **Remove deterministic nodes before testing.** For each candidate node, regress it on its fixed parents. If R² > 0.99 (in log space), mark it deterministic, keep it as a function node and exclude it from conditional-independence tests. This avoids faithfulness violations from near-exact relationships such as V^sh = f(s).
2. **Residualize.** Replace each remaining Tier 2 and Tier 3 variable by its residual after regressing on Tier 0 and Tier 1 parents. Discovery then looks only for structure the skeleton does not already explain.
3. **Run constrained discovery** on the residuals with the tier constraints as background knowledge. Default: PC with a kernel CI test (KCI) or Fisher-z on log residuals; alternative: score-based GES or NOTEARS with an edge mask. causal-learn's PC accepts tiered background knowledge.
4. **Stability selection.** Repeat on 200 bootstrap resamples (resampling whole jobs, not runs). Keep an edge if it appears in at least 70% of resamples; orient it by the majority direction.
5. **Domain check.** Show edges that contradict the default skeleton (e.g., CPU → memory pressure) to a human before accepting them. Log the decision.

**Granularity.** One global graph is shared by all jobs in a workload family (e.g., iterative ML, SQL, graph). Per-job graphs are not learned; per-job differences live in the structural-function parameters.

**When data is too small** (fewer than about 30 distinct jobs), skip steps 3–5 and use the default skeleton as is. This is the expected v1 setting.

## Structural functions

Every node has a parametric form with non-negative, interpretable parameters, so the model extrapolates along known scaling laws rather than along whatever a flexible regressor learned. Parameters are per job and per stage unless noted; the Cross-job transfer section shares them across jobs through priors.

### Tier 1: data volumes and task counts

Volumes depend on input size, data characteristics and job parameters, but not on machines or memory:

```latex
\log V_k = a_k + b_k \log s + \boldsymbol{\phi}_k^{\top} \mathbf{z}(d, p), \qquad b_k \ge 0
```

Here z(d, p) holds log-transformed data characteristics and parameters (e.g., log distinct keys, log k). Most stages have b_k ≈ 1; joins and aggregations often do not. For iterative jobs, a stage repeated per iteration shares one set of parameters.

Task counts are deterministic: n_k = ceil(V_k^in / partition size) for input stages, and the configured shuffle partitions for shuffle stages.

### Tier 2: memory and resource state

Working set per task grows with partition size, inflated by deserialization and skew:

```latex
W_k = \omega_{0,k} + \omega_{1,k} \, \frac{V_k}{n_k} \left(1 + \omega_{2,k} \, \mathrm{skew}\right), \qquad \omega \ge 0
```

ω_1 is the in-memory expansion factor; it can be measured directly in single-machine profiling. Cached data per executor is C = (sum of cached dataset sizes in memory) / (m · e).

Memory pressure is the ratio of demand to usable execution memory:

```latex
\rho_k = \frac{c_e \, W_k + C}{U_e}
```

Spill and recomputation switch on smoothly once pressure exceeds 1. With σ the logistic function and τ a learned smoothness:

```latex
h(\rho) = \left(1 - \frac{1}{\rho}\right) \sigma\!\left(\frac{\rho - 1}{\tau}\right), \qquad S_k = \lambda_k \, V_k^{sh} \, h(\rho_k)
```

The same h, applied to cache pressure C / U_storage, gives the fraction of cached partitions recomputed per iteration.

Because S_k = 0 whenever a stage fits in memory, spill is modeled as zero-inflated: P(S_k > 0) = σ((ρ_k − 1) / τ) and, when positive, log S_k follows the equation above with log-normal noise.

GC time is bounded and increases with pressure:

```latex
g_k = g_{\max} \, \sigma\!\left(\gamma_{0,k} + \gamma_{1} \log \rho_k + \gamma_{2} \log M_e\right), \qquad \gamma_1 \ge 0
```

If discovery (see Graph learning) finds extra Tier 2 edges, each gets an additive term in log space on the child, with a sign constraint set by the domain check.

### Tier 3: stage runtime

Stage runtime extends Ernest from "input size over machines" to the stage's own volumes, waves and memory regime. With waves w_k = ceil(n_k / N):

```latex
T_k = \theta_{0,k} + w_k \left[ \theta_{1,k} \frac{V_k^{in}}{n_k (1 - g_k)} + \theta_{4,k} \frac{S_k}{n_k B_{disk}} + \theta_{5,k} \frac{V_k^{sh}}{n_k B_{net}} \cdot \frac{m-1}{m} \right] + \theta_{2,k} \log m + \theta_{3,k} \, m
```

- θ_1: compute cost per byte, inflated by GC.
- θ_4: spill cost; zero in the in-memory regime because S_k = 0.
- θ_5: shuffle cost; (m − 1)/m is the share of shuffle data read remotely.
- θ_2, θ_3: Ernest's aggregation and per-machine coordination terms.

Given observed V, S, g and n, T_k is linear in θ, so it is fitted with non-negative least squares exactly as Ernest is.

Job runtime is the longest path through the stage DAG plus driver overhead: T = θ_drv + max over paths of the sum of T_k. For a linear chain this is a sum; for iterative jobs, iterations × per-iteration path.

### Tier 3: resource outcomes

- **CPU utilization** follows from the runtime decomposition: u_k = compute time / T_k, using the compute term above. It is derived, not separately fitted.
- **Peak executor memory:** M_peak = M_base + min(U_e, max over k of c_e · W_k + C), with M_base fitted per job (JVM and user memory).
- **Memory requirement** is a counterfactual query on the fitted model: the smallest M_e whose predicted runtime is within a tolerance ε (default 5%) of the runtime with unlimited memory. This tolerance version is the primary answer once Phase A step 2 has fitted the spill and GC terms. The zero-spill version (the closed form below) needs only Tiers 1–2 and is reported alongside it, as an early estimate available after Phase A step 1. Zero spill can err both ways: too high when cheap spills are avoided at a large memory cost, too low when GC overhead rises with ρ just under 1.

```latex
M_{req} = 300\,\mathrm{MB} + \frac{\max_k (c_e W_k) + C}{f_{spark}}
```

### GPU jobs

GPU jobs reuse the four tiers, with GPU memory as the scarce resource. The key difference is that exceeding GPU memory usually fails a training job rather than spilling, so the model predicts out-of-memory risk instead of spill.

**Tier 2: GPU working set.** Per GPU, in phase k:

```latex
G_k = \frac{P \, (\beta_w + \beta_g + \beta_o)}{z} + b \, A_k + G_0
```

- β_w, β_g, β_o: bytes per parameter for weights, gradients and optimizer state (2, 2 and 12 for mixed-precision Adam), fixed from the configuration rather than fitted.
- z: 1 for DDP; the number of GPUs when that state is fully sharded (FSDP or ZeRO stage 3); intermediate for ZeRO stages 1–2.
- A_k: activation bytes per sample, fitted from 2–3 batch sizes; smaller when activation checkpointing is on.
- G_0: CUDA context, allocator fragmentation and workspace, fitted per job and GPU type.

GPU memory pressure is ρ_gpu = G_k / R_gpu. For training, the out-of-memory probability is P(OOM) = σ((ρ_gpu − 1) / τ_g). Where exceeding memory spills instead (CPU offload, Spark RAPIDS), the spill function h(ρ) above is reused with host-transfer bandwidth.

**Tier 3: step time** for data-parallel training on N GPUs, with Φ the FLOPs per sample:

```latex
T_{step} = \theta_0 + \max\!\left(\theta_c \frac{b \, \Phi}{F_{gpu}} (1 + \kappa \, \mathrm{ckpt}),\; \theta_d \frac{b}{N_{load}}\right) + (1 - \omega) \left[ \theta_a \frac{2(N-1)}{N} \frac{P \beta_g}{B_{ic}} + \theta_l \log N \right]
```

- The max term: GPU compute versus input loading (N_load loader workers); checkpointing adds recomputation κ (about 0.33 for full checkpointing).
- The bracket: ring all-reduce of gradients plus a latency term that mirrors Ernest's log m; ω is the fitted share of communication overlapped with compute.
- B_ic is PCIe bandwidth within a machine (no NVLink on the target Chameleon nodes) and network bandwidth once N spans more than one machine, which creates the scaling break seen when training leaves a single node.
- Job runtime = setup θ_s + steps × T_step, with steps = epochs × dataset size / (b × N).

**Spark with RAPIDS (extension after v1).** The Spark stage equations apply unchanged, with three edits: θ_1 is fitted separately for operators that run on the GPU; the GPU memory pool replaces U_e in ρ_k; and a host-to-device transfer term θ_6 · V_k / (n_k · B_pcie) is added inside the waves bracket.

**GPU outcomes.**

- G_peak = max over phases of G_k.
- G_req: the smallest GPU memory with ρ_gpu ≤ 1 at a 10% margin. For a fixed GPU type, the more useful form of this query is the largest batch size that fits.
- u_gpu = compute time / T_step, derived from the step-time decomposition.

### Residual learner

For each Tier 3 node, fit a monotone gradient-boosted model (e.g., LightGBM with monotone constraints) to the log residual of the parametric form, using Tier 0 features. It is applied only when the query lies inside the training range on every feature; outside that range the residual is set to zero, so extrapolation uses the parametric form alone.

## Noise model and uncertainty

Each stochastic node gets multiplicative log-normal noise, and uncertainty is propagated through the graph by Monte Carlo so that every prediction comes with an interval.

```latex
\log Y_i = \log f_i(\mathrm{pa}(Y_i); \boldsymbol{\theta}_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma_i^2)
```

- **Node noise σ_i** is estimated from repeated runs at the same configuration (run each profiling configuration twice where budget allows) and from fit residuals.
- **Parameter uncertainty** comes from the NumPyro posterior (see Fitting procedure); bootstrap refits of the NNLS initialization serve only as a quick check.
- **Independence assumption:** noise terms are independent across nodes given parents. Correlated noise between stages of the same run (e.g., a slow node affects every stage) is captured by a shared per-run random effect η_run added to every log T_k.
- **Deterministic nodes** (task counts, derived CPU utilization, M_req closed form) carry no noise of their own; uncertainty reaches them only through parents.

Reported quantities: P50, P95 and a 90% interval for each outcome, plus an out-of-support flag (see Prediction, counterfactual queries and guards). Calibration target: 90% intervals should contain 88–92% of held-out observations.

## Profiling and experiment design

A new job with no history gets three profiling phases, about 8–12 runs in total. The key idea is that memory pressure is a ratio, so shrinking executor memory at small input size reaches the spill and GC regimes that a full-size run would hit, without paying for full-size runs.

**Phase A: single-machine profile (4–5 runs).**

1. Run on one machine at 3 input fractions (default 0.5%, 1%, 2% of s_target) with generous executor memory. This fits Tier 1 (a_k, b_k), the expansion factor ω_1 and M_base.
2. Repeat the largest fraction with M_e reduced so that predicted ρ is about 1.5 and about 3. This fits τ, λ and the GC parameters, so the threshold behavior is observed rather than assumed.

**Phase B: multi-machine design (4–6 runs).**

- Candidates: input fractions {1%, 2%, 4%, 8%} × machines {2, 4, 8, 16}, on the target machine type.
- Choose runs by D-optimal design on the Tier 3 NNLS features (as Ernest does), weighting each candidate by its estimated cost (predicted runtime × m × hourly price, or SU rate on Chameleon).
- Constraint: at least two distinct m values per input fraction, so θ_2 and θ_3 are identifiable.

**Phase C: active refinement (0–4 runs, optional).**

- Given the user's target region (e.g., s_target, m in 8–64), compute the posterior predictive variance of log T and M_req across that region.
- Pick the candidate run with the largest expected variance reduction per dollar; refit; repeat.
- Stop when the 90% interval for T at the target is narrower than ±20%, or the budget is spent.

**Other machine types.** Profiling runs must use the target machine type in v1. Supporting other types needs either a short calibration benchmark per type (as PerfOrator does with calibration queries) or cross-job transfer (see Cross-job transfer and historical data).

**GPU jobs.** Training runs execute a fixed number of steps (default 10 warmup plus 50 measured) instead of full training; per-step metrics stabilize quickly, so this cuts profiling cost sharply.

- Phase A: one GPU at 3 batch sizes to fit A_k, G_0 and θ_c. One more run with GPU memory capped (torch.cuda.set_per_process_memory_fraction) locates the out-of-memory boundary without needing a smaller GPU, the GPU counterpart of the memory-shrink trick.
- Phase B: GPU counts in powers of two up to the GPUs per machine, then 2 and 4 machines, to separate PCIe from network all-reduce costs.
- In the RAPIDS extension, Spark jobs follow the Spark phases, with the GPU memory pool shrunk in Phase A step 2 instead of executor memory.

## Shared-cluster profiling under a fixed budget

Profiling and evaluation runs are packed onto shared leases to cut service-unit (SU) use. Co-location is a controlled, recorded variable, so its effect is measured and removed rather than left as noise.

**How SUs are spent.** Assuming SUs accrue per reserved node for the whole lease (check each node type's rate in the allocation):

```latex
\mathrm{SU} = \sum_{\text{leases}} n_{\text{nodes}} \times \text{lease hours} \times \text{rate}_h
```

Idle reserved nodes and provisioning time cost as much as running jobs. The goal is to keep every reserved node busy and end each lease as soon as its batch finishes.

**Three sharing modes, in order of preference:**

| Mode | How | Interference | Use for |
| --- | --- | --- | --- |
| Partitioned | Split one lease's nodes into disjoint groups, each running its own Spark cluster at the same time (e.g., 16 nodes as 1×8 + 2×4) | Shared switch and uplink only | Default for Phase B, Phase C and evaluation runs |
| Back-to-back | Reuse the same cluster for consecutive runs without releasing the lease | None between runs, once caches are cleared | Runs that need the whole lease; replicates |
| Node-shared | Several small runs on the same nodes with cgroup-pinned cores and memory | Memory bandwidth, CPU cache, disk and network | Phase A single-machine profiles only |

Isolation rules: give each partition its own HDFS (or a local copy of the input) so storage traffic stays inside the partition. Between back-to-back runs, drop the OS page cache and restart Spark executors to clear JVM state.

GPU nodes are shared by whole GPUs (CUDA_VISIBLE_DEVICES), not within a GPU, unless the GPU supports MIG partitions. Runs on different GPUs of the same machine count as node-shared, since they share PCIe, CPU cores and host memory; GPU evaluation runs get whole machines.

**Model changes.** Add three interference variables to Tier 0 for every run:

- L_node: number of co-runners on the same nodes (0 in partitioned and back-to-back modes).
- L_net: aggregate shuffle rate of co-runners on the same switch, from their measured shuffle bytes over time.
- L_disk: aggregate disk rate of co-runners on the same nodes.

They enter the stage runtime equation through effective bandwidths and a compute slowdown:

```latex
B_{disk}^{eff} = \frac{B_{disk}}{1 + \beta_d L_{disk}}, \qquad B_{net}^{eff} = \frac{B_{net}}{1 + \beta_n L_{net}}, \qquad \theta_{1,k}^{eff} = \theta_{1,k} (1 + \beta_c L_{node})
```

The β values are non-negative and shared across jobs, because they describe the hardware. Since we choose which runs share a lease, co-location is an intervention: assign co-runner pairings at random within a batch so that L is not correlated with job configuration. Deployment predictions use do(L = 0), or the expected load where the target cluster is shared.

**Calibration runs.** Per hardware type, run 2–3 pairs of one configuration solo and co-located (in each sharing mode used) to anchor the β values. Keep at least one solo run per job as an anchor; if its residual differs from that job's co-located runs by more than the model predicts, flag the job's interference terms for review.

**Budget-aware scheduler.** Inputs: candidate runs from experiment design, each with a node count and a predicted P75 duration (from the current posterior, or the family prior for a new job); available node types; SU rate per type; remaining budget.

1. Group runs by hardware type, one lease per type per batch.
2. Choose the lease size: the smallest node count that fits the largest run, rounded so partitions pack evenly (powers of two match the {2, 4, 8, 16} design grid).
3. Pack runs into partition × time slots with longest-first greedy strip packing; fill idle node gaps with Phase A single-machine runs.
4. Estimate SU = lease nodes × makespan × rate. If it exceeds the per-batch cap, drop the candidate with the lowest expected information gain per SU and repack.
5. Run the batch, refit the model and re-predict durations before scheduling the next batch.

Experiment design (Phases B and C) now ranks candidates by expected information gain per SU instead of per dollar. Lease creation and release can be scripted with Chameleon's python-chi library.

**Budget tracking.** Each run records its lease, partition and attributed node-seconds. The scheduler keeps a running SU total and stops issuing batches when the remainder falls below a reserve (default 10%) kept for re-runs and evaluation.

**Chameleon rates and leases.** SU estimates use 1 SU per node-hour for CPU nodes and 2 for GPU nodes unless a lease reports otherwise. Development batches use short leases of a few hours to a day; the scheduler caps any lease at the 7-day maximum and splits longer plans into consecutive leases (see Pre-implementation decisions).

**Evaluation runs.** Ground-truth runs (full-size test runs and memory sweeps for M_req) use partitioned or back-to-back mode only, never node-shared, so interference does not contaminate test labels. Pack the full-size test runs of different jobs into the same lease to share its cost.

## Instrumentation and data schema

All model variables come from a Spark listener plus node-level monitoring; nothing requires changing the job's code.

| Variable | Source | Aggregation per stage |
| --- | --- | --- |
| V^in, V^out | Task metrics inputMetrics.bytesRead, outputMetrics.bytesWritten | Sum |
| V^sh | shuffleWriteMetrics.bytesWritten | Sum |
| Remote shuffle bytes | shuffleReadMetrics.remoteBytesRead | Sum (checks the (m−1)/m term) |
| n_k | Stage info numTasks | As is |
| S_k | memoryBytesSpilled, diskBytesSpilled | Sum |
| g_k | jvmGCTime / executorRunTime | Ratio of sums |
| Compute time | executorCpuTime | Sum |
| W_k (observed proxy) | peakExecutionMemory per task | P95 over tasks |
| T_k | Stage submission to completion time | Wall clock |
| M_peak | Executor metrics JVMHeapMemory, OnHeapExecutionMemory, OnHeapStorageMemory, ProcessTreeJVMRSSMemory | Max per executor, then max |
| C | Storage memory used by cached RDDs | Max per executor |
| CPU utilization (check) | Node exporter or cgroup CPU counters | Mean over stage window |
| G_peak (GPU jobs) | DCGM or nvidia-smi memory.used | Max |

ProcessTreeJVMRSSMemory needs spark.executor.processTreeMetrics.enabled=true; executor metric polling should be set to about 1 second (spark.executor.metrics.pollingInterval).

**GPU jobs add:** torch.cuda.max_memory_allocated and max_memory_reserved per step window; DCGM fields for framebuffer memory used, GPU utilization, SM activity and PCIe bytes (and NVLink bytes on nodes that have it), sampled each second; per-step time split into data loading, forward and backward, and communication (PyTorch profiler or step hooks); NCCL all-reduce time. Spark RAPIDS jobs add the plugin's per-stage GPU metrics, such as GPU semaphore wait time and spill to host memory.

**Data characteristics d** are computed once per input from a 0.1% sample: record count, mean record size, HyperLogLog distinct count of the main key, share of records held by the top 1% of keys, and fraction of zero values for numeric data.

**Matching stages across runs.** Spark stage ids differ between runs. Each stage gets a signature. SQL jobs hash the physical-plan operator chain (operator names and key expressions, not literal values); RDD jobs hash the call site plus lineage operator types. A signature that repeats within a job gets iteration indices 0, 1, 2 and so on. Models are keyed by job id and stage signature.

**Storage.** Two Parquet tables:

- runs: run_id, job_id, all Tier 0 fields, d features, start and end time, cluster load indicator, lease id, node partition, sharing mode, co-runner run ids and their measured shuffle and disk rates, attributed node-seconds, source (profiling or historical).
- stages: run_id, stage_signature, iteration, parent signatures, and every per-stage metric above.

## Fitting procedure

Fitting is modular first, then joint: each node is fitted on its observed parents in topological order, and an optional end-to-end pass corrects compounding error.

1. **Tier 1.** For each stage, robust linear regression (Huber loss) of log V_k on log s and z(d, p), with b_k ≥ 0.
2. **Tier 2.** Fit W_k from task-level peakExecutionMemory against V_k / n_k and skew. Tasks in stages without spill are exact observations; tasks in spilling stages are capped at usable memory and enter as lower bounds (censored likelihood). Then fit (τ, λ) and the GC parameters by bounded nonlinear least squares (scipy least_squares, method trf) using observed ρ, S and g. Runs from Phase A step 2 are what make τ identifiable.
3. **Tier 3.** Fit the per-task compute, spill and shuffle terms from task-level durations, then fit stage-level T_k with θ_2 and θ_3 shared across all stages of the job. The six-column NNLS fit per stage is kept only as the initialization and the Ernest baseline. Fit θ_drv on job-level residuals.
4. **Noise.** Estimate σ_i from residuals and replicate runs; estimate the per-run random-effect variance from correlated stage residuals.
5. **Joint refinement (optional).** Implement the whole forward pass in JAX or PyTorch with softplus-parameterized non-negative parameters. Initialize from steps 1–4 and minimize the sum over nodes of squared log errors, feeding predicted (not observed) parents downstream. This reduces error compounding when predictions chain through Tier 1 → Tier 2 → Tier 3.
6. **Residual learners.** Fit last, on residuals of the refined model.

**Default fitting path for v1 (NumPyro); steps 1–5 above provide its initialization.** Write the same model in NumPyro: half-normal priors on non-negative parameters, hierarchical priors from the Cross-job transfer section, log-normal likelihoods per node. Use NUTS with 4 chains × 1,000 draws. A job with 10 runs and 6 stages has roughly 100 parameters, so sampling should take seconds to a minute on a CPU.

**Model checks after every fit:** posterior predictive check on held-out runs; flag any stage whose MAPE exceeds 25% for manual review; confirm θ values are physically plausible (e.g., θ_1 implies a per-core throughput between 1 MB/s and 2 GB/s).

## Prediction, counterfactual queries and guards

Every query is an intervention do(x) on Tier 0 followed by a forward pass through the graph, repeated over parameter and noise samples.

**Forward pass for one configuration x:**

1. Draw parameters (posterior sample or bootstrap refit) and a run effect η_run.
2. Compute Tier 1: V_k and n_k for every stage.
3. Compute Tier 2: W_k, C, ρ_k, S_k, g_k.
4. Compute Tier 3: T_k with exact ceil for waves; T by longest path; u_k, M_peak.
5. Repeat 1,000 times; report P50, P95 and the 90% interval.

**Supported queries:**

| Query | How it is answered |
| --- | --- |
| Runtime and resources for x | Forward pass |
| Memory requirement for (s, m, h) | Bisection on M_e: smallest value with P50 runtime within 5% of the M_e → ∞ runtime; the zero-spill closed form is reported alongside |
| Cheapest configuration meeting deadline D | Grid over machine type, m and M_e; keep configurations with P95 T ≤ D; minimize P50 cost = T × m × price |
| Effect of changing one knob | Difference of forward passes, do(x) vs do(x′), with shared samples |
| Why is it slow? | Decompose T_k into its four terms and report the dominant one per stage |

**Extrapolation guards.** Each prediction carries flags that the caller can act on:

- **Size extrapolation:** s more than 10× the largest profiled input. Intervals are still reported but marked low confidence.
- **Unseen regime:** any stage with predicted ρ > 1 when no profiling run had ρ > 1 for that stage. Suggest one Phase A step 2 run.
- **Unseen configuration:** machine type or job parameter values outside training data.
- **Residual off:** the residual learner was disabled because the query is outside its training range.

## Cross-job transfer and historical data

Jobs share the graph and draw their parameters from family-level priors, so a new job starts from a good guess and needs fewer profiling runs. Historical runs are used too, but treated as observational data.

**Hierarchical parameters.** Group jobs into families (e.g., iterative ML, SQL aggregation, graph processing) and group stages by operator type (scan, shuffle-map, aggregation, join, iteration body). For any parameter θ of stage type t in family f:

```latex
\log \theta_{j,k} \sim \mathcal{N}(\mu_{f,t}, \, \Sigma_{f,t})
```

μ and Σ are learned from all past jobs. With no profiling runs, the model predicts from the prior; each run then narrows the posterior. Evaluate how many runs are needed to reach a target accuracy, versus no-transfer.

**Machine-type transfer.** Make hardware enter only through physical quantities (c_h, R_h, B_disk, B_net) and a per-type compute speed factor κ_h on θ_1. κ_h is learned from all jobs run on that type, so a job profiled on one type can be predicted on another.

**Historical runs are observational.** Users chose those configurations, often scaling m with s, and often under different cluster load. Three steps make them usable:

1. **Record the confounders.** Store cluster load (concurrent jobs, node CPU at submission) and treat it as a Tier 0 variable. Unrecorded load is the main hidden confounder.
2. **Check overlap.** If historical (s, m) pairs lie on a narrow ridge (m almost proportional to s), flag θ_2 and θ_3 as weakly identified and add 1–2 profiling runs off the ridge.
3. **Down-weight by trust.** Give historical runs a likelihood weight below 1 (default 0.5) relative to controlled profiling runs; tune on held-out data.

Because all Tier 0 variables are observed and nothing downstream causes them, adjusting for them satisfies the backdoor criterion, provided cluster load is recorded. If it is not, report that historical-only predictions may be biased.

## Pre-implementation decisions

These decisions are fixed for v1, and the sections above already reflect them. Items under "Deferred to implementation" can be settled while coding.

| # | Topic | Decision |
| --- | --- | --- |
| 1 | Spill is often zero | Spill is zero-inflated: a logistic model for P(S_k > 0) given ρ, and a log-normal model for the amount when positive. The threshold is fitted from Phase A step 2 runs. Spill stays out of the log-space noise model. |
| 2 | Working set is capped when a stage spills | W_k is fitted on stage executions without spill; spilling ones enter as lower bounds in a censored likelihood. peakExecutionMemory excludes user objects, which fall into M_base. |
| 3 | Few observations per stage | Tier 2 terms and the per-task compute term are fitted from task-level metrics, so each task is an observation. θ_2 and θ_3 are shared across a job's stages. Hierarchical priors are used from the first fitting milestone. |
| 4 | One fitting path | NumPyro is the single fitting path. Modular NNLS fits serve only as initialization and as the Ernest baseline. |
| 5 | Spark environment | Spark 3.5 with the standalone cluster manager inside each lease; Adaptive Query Execution disabled; input copied to a per-partition HDFS or local disk, whichever sets up faster for that job. |
| 6 | Stage signatures | SQL jobs: hash of the physical-plan operator chain (operator names and key expressions, not literal values). RDD jobs: hash of call site plus lineage operator types. A signature that repeats within a job gets iteration indices 0, 1, 2 and so on. |
| 7 | Input size for GPU training | Split into dataset size, which sets steps per epoch and total runtime, and per-step inputs (batch size, sequence length or image size), which set step time and memory. |
| 8 | Job specification | One YAML file per job declares the entry point, job parameters p with types, data characteristics d to extract, profiling sweep values and the workload family. |
| 9 | Chameleon environment | See below. |

**Example job specification:**

```yaml
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
```

**Chameleon environment.**

- **No NVLink.** GPUs in one machine communicate over PCIe, so B_ic is PCIe bandwidth within a machine and network bandwidth across machines. Communication is costlier than on NVLink systems, so the multi-GPU scaling terms matter more.
- **SU rates.** Planning assumes 1 SU per node-hour for CPU nodes and 2 SU per node-hour for GPU nodes, confirmed per lease. Examples: 8 CPU nodes for 24 hours cost 192 SU; 4 GPU nodes for 24 hours cost 192 SU; 8 GPU nodes for a full 7-day lease cost 2,688 SU, about 13% of the 20,000 SU budget.
- **Lease length.** Development uses short leases of a few hours to a day; only the final evaluation may approach the 7-day maximum. The scheduler caps a planned lease at 7 days and splits longer plans into consecutive leases.
- **Node types unknown in advance.** The scheduler reads available node types at lease time. On first use of a type, a short calibration run measures memory, disk and network bandwidth and fills that type's physical quantities (c_h, R_h, B_disk, B_net).

**Deferred to implementation:**

- Unified memory: Spark lets execution memory evict cached storage, so c_e · W_k + C overstates pressure. Start with the sum; refine if residuals show bias.
- Experiment design: linearize the model at the prior for D-optimal selection; estimate Phase C information gain from posterior samples.
- GPU details: FLOPs per sample from PyTorch's flop counter; z for ZeRO stages 1–2; separating all-reduce time from compute time.
- Memory-sweep ground truth: sweep only at target configurations, with coarse steps refined near the knee.
- Success thresholds: set target MAPE and interval-coverage numbers per milestone before milestone 2.

## Implementation plan

The system is a Python package with a Scala Spark listener; each module maps to one section above.

**Stack:** Python 3.11; Spark 3.5 (standalone cluster manager, AQE off); NumPy, SciPy (nnls, least_squares); JAX + NumPyro for joint and Bayesian fits; LightGBM for residuals; causal-learn for constrained PC; pandas + PyArrow for storage. Listener in Scala, emitting JSON lines per stage.

```
tscm/
  collect/
    listener/            # Scala SparkListener: stage + executor metrics -> JSONL
    signatures.py        # stage signature hashing, iteration index
    data_profile.py      # sample-based d features (HLL, skew)
    gpu_metrics.py       # DCGM sampling, PyTorch memory and step hooks
    store.py             # runs / stages Parquet tables
    spec.py              # job specification YAML schema and validation
  graph/
    skeleton.py          # tiers, required and forbidden edges
    discovery.py         # residualize, PC with background knowledge, stability selection
  mechanisms/
    tier1.py             # volume power laws, task counts
    tier2.py             # working set, pressure, spill, GC
    tier3.py             # stage runtime NNLS, critical path, memory outcomes
    gpu.py               # GPU working set, OOM risk, step time, RAPIDS terms
    residual.py          # monotone GBM with support check
  fit/
    modular.py           # topological per-node fitting
    joint.py             # JAX end-to-end refinement
    bayes.py             # NumPyro hierarchical model
  design/
    profiling.py         # phases A, B, C; D-optimal selection; cost weighting
    packing.py           # lease sizing, strip packing, SU budget tracking
    chameleon.py         # lease create/release, partition setup (python-chi)
  query/
    forward.py           # Monte Carlo forward pass
    queries.py           # M_req bisection, deadline search, effect, decomposition
    guards.py            # extrapolation flags
  eval/
    harness.py           # splits, baselines, metrics
```

**Core forward pass (pseudocode):**

```python
def forward(job, x, params, eta_run):
    out = {}
    for k in job.stages_topological():
        V_in, V_sh = tier1.volumes(k, x.s, x.d, x.p, params[k])
        n = tier1.tasks(k, V_in, x)
        W = tier2.working_set(V_in, n, x.d.skew, params[k])
        C = tier2.cache_per_executor(job, x, params)
        rho = (x.c_e * W + C) / x.U_e
        S = params[k].lam * V_sh * tier2.h(rho, params.tau)
        g = tier2.gc_fraction(rho, x.M_e, params[k])
        T = tier3.stage_runtime(V_in, V_sh, S, g, n, x, params[k]) * exp(eta_run)
        out[k] = dict(V_in=V_in, V_sh=V_sh, n=n, W=W, rho=rho, S=S, g=g, T=T)
    T_job = params.theta_drv + job.longest_path(lambda k: out[k]["T"])
    M_peak = params.M_base + min(x.U_e, max(x.c_e * o["W"] for o in out.values()) + C)
    return T_job, M_peak, out
```

**Build order (suggested milestones):**

1. Listener, signatures and storage; collect profiling data for 5 benchmark jobs.
2. NumPyro fit of Tiers 1–3 with task-level likelihoods and hierarchical priors, without discovery or residuals; compare with Ernest on runtime.
3. Memory regime: Phase A step 2 runs, h and GC fits, M_req query.
4. Uncertainty: check interval calibration on held-out runs.
5. Machine-type factor and refinement of the cross-job priors.
6. Constrained discovery and residual learners, once 30+ jobs are collected.

GPU milestones follow: PyTorch DDP and FSDP training first (milestone 7), then the Spark RAPIDS extension on NDS (milestone 8).

## Evaluation plan

The central claim to test is that the TSCM extrapolates better than correlational models, to larger inputs, new memory regimes and new machine types, while matching them in interpolation.

**Workloads:** Spark MLlib (K-Means, logistic regression, GBT), SQL (TPC-DS queries), graph (PageRank), Sort and Grep. GPU (v1): step-limited PyTorch DDP and FSDP training of the smaller MLPerf Training reference models (RetinaNet, DLRMv2, RGAT, on subsampled or synthetic data) plus TorchBench or Hugging Face models such as ResNet-50, BERT-base and a GPT-2-size model. Extension: Spark RAPIDS on NDS, the TPC-DS-derived benchmark in NVIDIA's spark-rapids-benchmarks repository, across several scale factors. Collect a new dataset with resource metrics, since public runtime datasets (C3O, Bell) lack them. Enel's experiment repository is a starting point for the listener.

**Workload priorities.** The largest accuracy gains are expected where runtime or memory changes abruptly with scale; smooth scalers are already handled well by Ernest-style models. Gains should appear mainly in extrapolation, not interpolation. These are hypotheses for the evaluation to test.

| Priority | Family | Examples | Why baselines struggle | Model features that target it | Expected gain |
| --- | --- | --- | --- | --- | --- |
| 1 | Iterative ML with in-memory caching | K-Means, logistic regression / SGD, GBT, ALS | Memory cliff: once cached data no longer fits, every iteration recomputes or rereads it. Maros et al. measured Ernest errors of 126.7–187.0% on K-Means; Bellamy's largest gains were on K-Means and SGD | Memory pressure ρ, spill and recompute threshold h(ρ), memory-shrink profiling runs | Largest |
| 2 | Shuffle-heavy SQL and graph | Join- and aggregation-heavy TPC-DS queries, PageRank | Shuffle volume grows non-linearly and depends on data characteristics (e.g., unique pages in PageRank); spill begins at larger inputs | Stage-level volume laws, distinct-key and skew features, spill function | Large, mainly when extrapolating to larger inputs |
| 3 | Multi-node GPU training near its memory limit | FSDP and DDP training spanning 2+ machines, batch sizes near the limit | Step time breaks when communication moves from PCIe to the network; jobs fail past the GPU memory limit; GPU baselines model a single GPU only | Interconnect switch in the step-time equation, out-of-memory model, largest-feasible-batch query | Moderate to large, on multi-node scaling and batch-limit queries |
| 4 | Simple, regular jobs (controls) | Sort, Grep, map-only jobs, scan-dominated queries | Little: they scale predictably (Ernest error 1.5–1.7% on TPC-DS Query 26; 5.82% on Sort in C3O's evaluation) | None specific | Little or none; included to show no added error or profiling cost |

Implications for the evaluation design:

- Weight the workload mix toward families 1–3; choose join- and aggregation-heavy SQL queries rather than scan-dominated ones.
- Make test runs cross the regime boundaries: datasets that stop fitting in memory, shuffles that start spilling, training that spans more than one machine.
- Report every metric per family as well as overall; the regime-extrapolation split is where families 1 and 2 should show their gains most clearly.

**Splits (each tests one claim):**

| Split | Train on | Test on |
| --- | --- | --- |
| Interpolation | Profiling grid | Held-out grid points |
| Size extrapolation | Inputs up to 8% of s_target | 2×, 5×, 10× the largest training input, up to full size |
| Regime extrapolation | Runs with ρ < 1 only, plus Phase A step 2 | Runs with ρ > 1 at large input |
| Machine-type transfer | All jobs on types A, B; profiled job on A | Same job on type C |
| Cold start | Other jobs only | New job with 0, 2, 4, 8 runs |

**Baselines:** Ernest (NNLS), Bell, C3O's GBM and optimistic models, Bellamy for runtime; Ruya-style memory model and a peak-memory regression (KS+ style) for memory; DNNPerf, TraPPM and GPUMemNet for per-GPU memory and step time, using their public datasets (TraPPM, GPUMemNet) plus our runs; Ernest-style NNLS on GPU count as the multi-GPU scaling baseline, since the GPU baselines predict single-GPU behavior only; for the RAPIDS extension, NVIDIA's QualX model as an industry reference (it is trained on NDS, so NDS results favor it); a black-box multi-output GBM on Tier 0 features as the correlational control.

**Metrics:** MAPE and median APE for T and M_peak; error of M_req against the measured minimum memory without slowdown (found by sweep); 90% interval coverage; profiling cost to reach 15% MAPE; regret of the deadline query (cost of chosen vs best configuration).

**Ablations:**

- No regime model (set h = 0, constant g).
- Generic structural functions (GBM per node) instead of scaling laws.
- Whole-job instead of per-stage model.
- No tiers (unconstrained discovery).
- No cross-job priors.
- No Phase A step 2 (memory-shrink runs).

## Open questions and risks

| Risk | Why it matters | Mitigation or test |
| --- | --- | --- |
| Volume laws break at scale (e.g., joins, skewed keys) | Tier 1 errors propagate to every downstream node | Fit b_k from 3+ input fractions; flag stages whose log-log fit has curvature |
| Memory shrinking does not reproduce full-scale behavior | Phase A step 2 is the regime-learning mechanism | Validate on 2–3 jobs by running full-size at low memory once |
| Adaptive Query Execution changes the plan with size | Stage DAG and signatures differ across runs | AQE is disabled in v1 (see Pre-implementation decisions) |
| Multi-tenant interference | Adds noise and hidden confounding | Co-location is controlled and recorded (shared-cluster section); solo anchor runs check the interference terms |
| Stage straggling and data skew | Wave model assumes similar task durations | Add a max-task term using skew; compare to wave-only model |
| Discovery finds spurious edges | Wrong structure hurts transfer | Stability selection and domain check; ablate discovery |
| Heterogeneous clusters | Single h per run assumed | Out of scope for v1 |

**Questions to settle early:**

- Decided: report both memory requirements. The runtime-tolerance version (smallest memory within 5% of unlimited-memory runtime) is primary once the spill and GC terms are fitted; the zero-spill version is an early estimate from Tiers 1–2. Evaluation ground truth uses the tolerance definition. GPU training uses a hard limit with a safety margin instead, since exceeding GPU memory fails the job.
- Decided: workload families are prioritized as iterative ML with caching, shuffle-heavy SQL and graph, multi-node GPU training, then simple jobs as controls (see Workload priorities in the Evaluation plan). These families also define the stage types for the hierarchical priors.
- Decided: PyTorch DDP and FSDP training comes first for GPU evaluation, because it has runnable benchmarks and published GPU baselines with public data; Spark RAPIDS on NDS follows as an extension. Still open: which subset of MLPerf, TorchBench and Hugging Face models fits the SU budget.
