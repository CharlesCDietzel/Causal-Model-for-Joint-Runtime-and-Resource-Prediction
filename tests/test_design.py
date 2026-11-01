import sys
import types

import numpy as np
import pytest

from tscm.design import chameleon, packing, profiling
from tscm.design.packing import RunRequest
from tscm.model import SPARK_RESERVED, GiB


def run(run_id, nodes=1, hours=1.0, phase="B", gain=1.0, node_type="cpu", evaluation=False):
    return RunRequest(run_id, "job", nodes, hours, node_type, 1.0, gain, phase, evaluation)


# Phases.


def test_phase_a_step1_uses_three_fractions_on_one_machine():
    runs = profiling.phase_a_step1(48 * GiB)
    assert [r.input_fraction for r in runs] == [0.005, 0.01, 0.02]
    assert all(r.machines == 1 for r in runs)


def test_memory_shrink_reaches_the_target_pressure():
    W, C, c_e, f = 0.5 * GiB, 2 * GiB, 4, 0.6
    for run_ in profiling.phase_a_step2(0.02, W, C, c_e, f):
        U_e = f * (run_.executor_memory - SPARK_RESERVED)
        rho = (c_e * W + C) / U_e
        assert rho == pytest.approx(float(run_.note.split("=")[1]))


def test_phase_b_keeps_two_machine_counts_per_fraction():
    plan = profiling.phase_b()
    assert 4 <= len(plan) <= 6
    by_fraction = {}
    for run_ in plan:
        by_fraction.setdefault(run_.input_fraction, set()).add(run_.machines)
    assert all(len(machines) >= 2 for machines in by_fraction.values())


def test_phase_b_prefers_cheap_runs_when_costs_differ():
    expensive_large = lambda fraction, m: fraction * m * 1000.0  # noqa: E731
    plan = profiling.phase_b(cost=expensive_large, min_runs=2, max_runs=2)
    assert min(r.input_fraction for r in plan) == 0.01


def test_information_gain_rewards_informative_candidates():
    rng = np.random.default_rng(0)
    target = rng.normal(0.0, 1.0, (2000, 1))
    informative = target + rng.normal(0.0, 0.1, (2000, 1))
    irrelevant = rng.normal(0.0, 1.0, (2000, 1))
    gain = profiling.information_gain(target, np.hstack([informative, irrelevant]), 0.01)
    assert gain[0] > 50 * gain[1]
    assert gain[0] == pytest.approx(1.0, rel=0.1)


def test_phase_c_divides_gain_by_cost():
    rng = np.random.default_rng(0)
    target = rng.normal(0.0, 1.0, (2000, 1))
    candidates = np.hstack([target, target])
    best, per_su = profiling.phase_c_next(target, candidates, np.array([10.0, 1.0]), 0.01)
    assert best == 1 and per_su[1] > per_su[0]


def test_phase_c_stops_on_interval_or_budget():
    narrow = np.random.default_rng(0).normal(100.0, 2.0, 5000)
    wide = np.random.default_rng(0).normal(100.0, 40.0, 5000)
    assert profiling.phase_c_done(narrow, 100.0, 1.0)[0]
    done, reason = profiling.phase_c_done(wide, 0.5, 1.0)
    assert done and reason == "budget spent"
    assert not profiling.phase_c_done(wide, 100.0, 1.0)[0]


def test_gpu_phases():
    runs = profiling.gpu_phase_a([8, 16, 32])
    assert len(runs) == 4 and runs[-1].memory_fraction < 1.0
    assert all(r.warmup_steps == 10 and r.measured_steps == 50 for r in runs)
    scaling = profiling.gpu_phase_b(16, 4)
    assert [(r.gpus_per_machine, r.machines) for r in scaling] == [(2, 1), (4, 1), (4, 2), (4, 4)]


# Packing.


def test_design_su_examples():
    plan = lambda nodes, hours, rate: packing.BatchPlan(  # noqa: E731
        "t", nodes, rate, [packing.Slot(run("r", nodes, hours), 0.0, hours, tuple(range(nodes)))])
    assert plan(8, 24.0, packing.CPU_RATE).su == 192
    assert plan(4, 24.0, packing.GPU_RATE).su == 192
    assert plan(8, 168.0, packing.GPU_RATE).su == 2688
    assert plan(8, 168.0, packing.GPU_RATE).su / 20_000 == pytest.approx(0.1344)


@pytest.mark.parametrize("largest, expected", [(1, 1), (2, 2), (3, 4), (8, 8), (9, 16)])
def test_lease_size_rounds_to_a_power_of_two(largest, expected):
    assert packing.lease_size([run("a", largest)]) == expected


def overlaps(a, b):
    return bool(set(a.nodes) & set(b.nodes)) and a.start < b.end and b.start < a.end


def test_packing_never_double_books_a_node():
    runs = [run(f"r{i}", nodes=[1, 2, 4, 8][i % 4], hours=1.0 + i % 3) for i in range(12)]
    slots = packing.pack(runs, 8, seed=1)
    assert len(slots) == len(runs)
    for i, a in enumerate(slots):
        assert len(a.nodes) == a.run.nodes
        for b in slots[i + 1:]:
            assert not overlaps(a, b)


def test_phase_a_runs_fill_idle_gaps_without_extending_the_lease():
    runs = [run("big", nodes=4, hours=4.0), run("half", nodes=2, hours=4.0),
            run("short", nodes=4, hours=1.0)]
    fillers = [run(f"a{i}", phase="A", hours=1.0) for i in range(2)]
    without = packing.BatchPlan("t", 4, 1.0, packing.pack(runs, 4))
    combined = packing.BatchPlan("t", 4, 1.0, packing.pack(runs + fillers, 4))
    assert combined.makespan == without.makespan
    assert combined.utilization() > without.utilization()


def test_only_phase_a_runs_may_be_node_shared():
    runs = [run("big", nodes=2, hours=2.0)] + [run(f"a{i}", phase="A") for i in range(4)]
    slots = packing.pack(runs, 2, node_share=2)
    shared = [s for s in slots if s.mode == packing.NODE_SHARED]
    assert shared and all(s.run.phase == "A" for s in shared)


def test_evaluation_runs_are_never_node_shared():
    runs = [run("e", phase="A", evaluation=True)] + [run(f"a{i}", phase="A") for i in range(3)]
    slots = packing.pack(runs, 1, node_share=2)
    evaluation = next(s for s in slots if s.run.run_id == "e")
    assert evaluation.mode != packing.NODE_SHARED and not evaluation.co_runners


def test_co_location_is_randomized_by_seed_within_a_duration_class():
    runs = [run(f"r{i}", nodes=2, hours=1.0) for i in range(8)]
    orders = {tuple(s.run.run_id for s in packing.pack(runs, 4, seed=seed)) for seed in range(6)}
    assert len(orders) > 1


def test_longest_classes_go_first_whatever_the_seed():
    runs = [run("short", nodes=4, hours=1.0), run("long", nodes=4, hours=8.0)]
    for seed in range(5):
        assert packing.pack(runs, 4, seed=seed)[0].run.run_id == "long"


def test_budget_cap_drops_the_lowest_gain_per_su():
    runs = [run("good", nodes=2, hours=2.0, gain=10.0), run("bad", nodes=2, hours=2.0, gain=0.1)]
    plan = packing.plan_batch(runs, su_cap=5.0)
    assert [s.run.run_id for s in plan.slots] == ["good"]
    assert [r.run_id for r in plan.dropped] == ["bad"]
    assert plan.su <= 5.0


def test_a_batch_is_one_hardware_type_per_lease():
    with pytest.raises(ValueError, match="one lease per hardware type"):
        packing.plan_batch([run("a", node_type="x"), run("b", node_type="y")])
    plans = packing.plan_batches([run("a", node_type="x"), run("b", node_type="y")])
    assert {p.node_type for p in plans} == {"x", "y"}


def test_long_plans_split_into_consecutive_leases():
    slots = [packing.Slot(run(f"r{i}", 1, 100.0), i * 100.0, (i + 1) * 100.0, (0,))
             for i in range(3)]
    leases = packing.split_lease(packing.BatchPlan("t", 1, 1.0, slots))
    assert [len(p.slots) for p in leases] == [1, 1, 1]
    assert all(p.makespan <= packing.MAX_LEASE_HOURS for p in leases)
    assert sum(p.su for p in leases) == pytest.approx(300.0)


def test_a_run_longer_than_a_lease_is_an_error():
    slot = packing.Slot(run("r", 1, 200.0), 0.0, 200.0, (0,))
    with pytest.raises(ValueError, match="longer than one lease"):
        packing.split_lease(packing.BatchPlan("t", 1, 1.0, [slot]))


def test_interference_variables_follow_co_location():
    a = packing.Slot(run("a", 2), 0.0, 1.0, (0, 1))
    same_node = packing.Slot(run("b", 1), 0.0, 1.0, (1,))
    same_switch = packing.Slot(run("c", 1), 0.0, 1.0, (2,))
    later = packing.Slot(run("d", 1), 2.0, 3.0, (0,))
    slots = [a, same_node, same_switch, later]
    rates = {"b": (10.0, 1.0), "c": (5.0, 2.0), "d": (100.0, 100.0)}
    assert packing.interference(a, slots, rates) == {"L_node": 1.0, "L_net": 15.0, "L_disk": 1.0}


def test_attributed_node_seconds():
    assert packing.attributed_node_seconds(packing.Slot(run("a", 2), 0.0, 1.5, (0, 1))) == 10800.0


def test_budget_keeps_the_reserve():
    budget = packing.Budget(total=1000.0)
    assert budget.available == 900.0
    assert budget.can_issue(900.0) and not budget.can_issue(901.0)
    budget.charge("lease", 850.0)
    assert not budget.can_issue(100.0)
    assert budget.log == [("lease", 850.0)]


# Chameleon. No account here, so lease calls run against a stand-in for python-chi.


@pytest.fixture
def fake_chi(monkeypatch):
    calls = {}

    class Lease:
        def __init__(self, name, duration=None):
            calls["init"] = (name, duration)

        def add_node_reservation(self, amount=None, node_type=None):
            calls["reservation"] = (amount, node_type)

        def submit(self, wait_for_active=True, idempotent=False):
            calls["submit"] = (wait_for_active, idempotent)

        def delete(self):
            calls["delete"] = True

    module = types.ModuleType("chi")
    module.lease = types.SimpleNamespace(Lease=Lease)
    module.hardware = types.SimpleNamespace(get_node_types=lambda: ["b", "a"])
    monkeypatch.setitem(sys.modules, "chi", module)
    return calls


def test_lease_lifecycle(fake_chi):
    reservation = chameleon.create_lease("batch-1", "compute_skylake", 8, 12.0)
    chameleon.release_lease(reservation)
    assert fake_chi["reservation"] == (8, "compute_skylake")
    assert fake_chi["init"][1].total_seconds() == 12 * 3600
    assert fake_chi["submit"] == (True, True) and fake_chi["delete"]


def test_lease_longer_than_seven_days_is_refused(fake_chi):
    with pytest.raises(ValueError, match="split the plan"):
        chameleon.create_lease("x", "t", 1, 169.0)
    assert "init" not in fake_chi


def test_node_types_are_read_at_lease_time(fake_chi):
    assert chameleon.available_node_types() == ["a", "b"]


def test_partitions_are_disjoint():
    groups = chameleon.partition_hosts([f"h{i}" for i in range(16)], [8, 4, 4])
    assert [len(g) for g in groups] == [8, 4, 4]
    assert len({h for g in groups for h in g}) == 16
    with pytest.raises(ValueError, match="partitions need"):
        chameleon.partition_hosts(["h"], [2])


def test_spark_setup_disables_aqe_and_attaches_the_listener():
    script = chameleon.spark_partition_script(["a", "b"])
    assert "spark.sql.adaptive.enabled false" in script
    assert "org.apache.spark.tscm.TscmListener" in script
    assert "spark.executor.processTreeMetrics.enabled true" in script
    assert script.count("start-worker.sh") == 2


def test_between_runs_clears_caches_and_restarts_executors():
    script = chameleon.between_runs_script(["a"])
    assert "drop_caches" in script and "start-worker.sh" in script


def test_calibration_parsing_fills_physical_quantities():
    output = "cores=48\nmemory_kib=1048576\ndisk_write_kib_s=1024\nnet_bits_s=8000\n"
    machine = chameleon.machine_from_calibration(
        "t", chameleon.parse_calibration(output), gpus=2)
    assert machine.cores == 48 and machine.memory == GiB
    assert machine.disk_bw == 1024 * 1024 and machine.net_bw == 1000.0
    assert machine.rate == packing.GPU_RATE
