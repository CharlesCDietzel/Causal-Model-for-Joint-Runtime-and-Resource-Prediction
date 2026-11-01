"""End to end on the simulator: profile, fit, then answer each supported query.

    uv run python examples/quickstart.py

The simulator stands in for a cluster, so this shows the workflow, not real-world accuracy.
"""

from tscm.design import profiling
from tscm.fit.bayes import fit_bayes
from tscm.fit.data import build_fit_data, build_job
from tscm.fit.modular import fit_modular
from tscm.model import Config, GiB
from tscm.query import guards, queries
from tscm.query.forward import Posterior
from tscm.simulate import DEFAULT_MACHINE, example_job, example_params, simulate_runs

S_TARGET = 200 * GiB
EXECUTOR = dict(machine=DEFAULT_MACHINE, e=2, c_e=4, p={"k": 8, "max_iter": 10}, d={"skew": 0.1})


def planned_configs() -> list[Config]:
    """Phase A step 1, memory-shrink runs at the largest fraction, then a Phase B design."""
    plan = profiling.phase_a_step1(48 * GiB)
    largest = max(run.input_fraction for run in plan)
    cached_per_executor = 1.22 * largest * S_TARGET / 2  # what step 1 would have measured
    plan += profiling.phase_a_step2(largest, W=0.05 * GiB, C=cached_per_executor, c_e=4)
    plan += profiling.phase_b()
    return [
        Config(s=run.input_fraction * S_TARGET, m=run.machines,
               M_e=run.executor_memory or 24 * GiB, **EXECUTOR)
        for run in plan
    ]


def main(bayes: bool = False) -> None:
    truth_job = example_job()
    runs, stages, tasks = simulate_runs(truth_job, example_params(truth_job), planned_configs())
    job = build_job("kmeans", "iterative_ml", runs, stages, features=("k",),
                    iteration_param="max_iter")
    data = build_fit_data(job, runs, stages, tasks)
    fit = fit_modular(data)
    # NUTS is the v1 fitting path, and takes minutes; the modular fit is its initialization.
    post = fit_bayes(data, fit) if bayes else Posterior.from_point(job, fit.params)

    support = guards.support_from_runs(runs, {"assign": [1.5]})
    target = Config(s=S_TARGET, m=32, M_e=24 * GiB, **EXECUTOR)
    result = queries.predict(post, target, support)
    T = result.outcomes["T"]
    print(f"runtime at {target.m} machines: {T['p50']:.0f}s "
          f"(90% interval {T['lo']:.0f}-{T['hi']:.0f}s)")
    print(f"peak executor memory: {result.outcomes['M_peak']['p50'] / GiB:.1f} GiB")
    for flag in result.flags:
        print(f"flag: {flag.code}: {flag.message}")

    memory = queries.memory_requirement(post, target, n_samples=100)
    print(f"memory requirement: {memory.tolerance_bytes / GiB:.1f} GiB "
          f"(zero-spill estimate {memory.zero_spill_bytes / GiB:.1f} GiB)")

    choice = queries.cheapest_configuration(
        post, target, deadline=1800.0, machines=[DEFAULT_MACHINE],
        machine_counts=[8, 16, 32, 64], executor_memories=[12 * GiB, 24 * GiB, 48 * GiB],
        n_samples=100)
    if choice:
        print(f"cheapest within 30 min: m={choice.config.m}, "
              f"M_e={choice.config.M_e / GiB:.0f} GiB, {choice.cost_p50:.1f} SU")

    for signature, report in queries.decompose(post, target, 100).items():
        print(f"stage {signature}: dominated by {report['dominant']}")


if __name__ == "__main__":
    main()
