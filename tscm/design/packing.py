"""Budget-aware scheduler: lease sizing, strip packing and SU tracking.

Runs are packed into node x time slots on one lease per hardware type. Co-location is an
intervention here: run order within a duration class is randomized, so that who ends up
next to whom does not correlate with job configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil, floor, log2

import numpy as np

MAX_LEASE_HOURS = 7 * 24.0
DEFAULT_RESERVE = 0.10
CPU_RATE = 1.0
GPU_RATE = 2.0
PARTITIONED, BACK_TO_BACK, NODE_SHARED = "partitioned", "back_to_back", "node_shared"


@dataclass(frozen=True)
class RunRequest:
    """A candidate run from experiment design."""

    run_id: str
    job_id: str
    nodes: int
    hours_p75: float  # predicted P75 duration
    node_type: str
    rate: float = CPU_RATE
    information_gain: float = 0.0
    phase: str = "B"
    evaluation: bool = False

    @property
    def phase_a(self) -> bool:
        return self.phase == "A" and self.nodes == 1 and not self.evaluation

    @property
    def gain_per_su(self) -> float:
        return self.information_gain / max(self.nodes * self.hours_p75 * self.rate, 1e-12)


@dataclass(frozen=True)
class Slot:
    run: RunRequest
    start: float
    end: float
    nodes: tuple[int, ...]
    mode: str = PARTITIONED
    co_runners: tuple[str, ...] = ()


@dataclass
class BatchPlan:
    node_type: str
    lease_nodes: int
    rate: float
    slots: list[Slot]
    dropped: list[RunRequest] = field(default_factory=list)

    @property
    def makespan(self) -> float:
        return max((slot.end for slot in self.slots), default=0.0)

    @property
    def su(self) -> float:
        return self.lease_nodes * self.makespan * self.rate

    def utilization(self) -> float:
        used = sum((s.end - s.start) * len(s.nodes) for s in self.slots)
        return used / max(self.lease_nodes * self.makespan, 1e-12)


def lease_size(runs: list[RunRequest]) -> int:
    """Smallest power of two fitting the largest run, matching the {2, 4, 8, 16} grid."""
    largest = max(run.nodes for run in runs)
    return 1 << ceil(log2(largest)) if largest > 1 else 1


def _duration_class(run: RunRequest) -> int:
    return floor(log2(max(run.hours_p75, 1e-6)))


def _ordered(runs: list[RunRequest], rng: np.random.Generator) -> list[RunRequest]:
    """Longest first, in factor-of-two classes, random within a class."""
    jitter = rng.permutation(len(runs))
    order = sorted(range(len(runs)), key=lambda i: (-_duration_class(runs[i]), jitter[i]))
    return [runs[i] for i in order]


class _Timeline:
    """Busy intervals per (node, lane). A lane is one cgroup-pinned share of a node."""

    def __init__(self, nodes: int, lanes: int):
        self.nodes, self.lanes = nodes, lanes
        self.busy: dict[tuple[int, int], list[tuple[float, float]]] = {
            (n, lane): [] for n in range(nodes) for lane in range(lanes)
        }

    def free_after(self, node: int) -> float:
        return max((iv[1] for lane in range(self.lanes) for iv in self.busy[(node, lane)]),
                   default=0.0)

    def occupy(self, node: int, lanes: range, start: float, end: float) -> None:
        for lane in lanes:
            self.busy[(node, lane)].append((start, end))
            self.busy[(node, lane)].sort()

    def gap(self, node: int, lane: int, duration: float, horizon: float) -> float | None:
        """Earliest start of an idle gap on a lane that ends by ``horizon``."""
        cursor = 0.0
        for start, end in self.busy[(node, lane)]:
            if start - cursor >= duration:
                return cursor
            cursor = max(cursor, end)
        return cursor if cursor + duration <= horizon else None


def pack(
    runs: list[RunRequest],
    lease_nodes: int,
    seed: int = 0,
    node_share: int = 1,
) -> list[Slot]:
    """Longest-first greedy strip packing, then Phase A runs into the idle gaps left over.

    With ``node_share`` > 1, up to that many Phase A runs may share a node (node-shared mode).
    Everything else, evaluation runs above all, takes whole nodes.
    """
    rng = np.random.default_rng(seed)
    timeline = _Timeline(lease_nodes, node_share)
    slots: list[Slot] = []
    fillers = [r for r in runs if r.phase_a]
    for run in _ordered([r for r in runs if not r.phase_a], rng):
        ranked = sorted(range(lease_nodes), key=timeline.free_after)
        chosen = tuple(sorted(ranked[: run.nodes]))
        start = max(timeline.free_after(node) for node in chosen)
        end = start + run.hours_p75
        for node in chosen:
            timeline.occupy(node, range(node_share), start, end)
        slots.append(Slot(run, start, end, chosen))

    horizon = max((slot.end for slot in slots), default=0.0)
    for run in _ordered(fillers, rng):
        placed = _fill_gap(timeline, run, horizon, lease_nodes, node_share)
        if placed is None:
            node = min(range(lease_nodes), key=timeline.free_after)
            start = timeline.free_after(node)
            placed = (node, 0, start)
        node, lane, start = placed
        end = start + run.hours_p75
        timeline.occupy(node, range(lane, lane + 1), start, end)
        horizon = max(horizon, end)
        slots.append(Slot(run, start, end, (node,)))
    return _label_sharing(slots)


def _fill_gap(timeline, run, horizon, lease_nodes, node_share):
    best = None
    for node in range(lease_nodes):
        for lane in range(node_share):
            start = timeline.gap(node, lane, run.hours_p75, horizon)
            if start is not None and (best is None or start < best[2]):
                best = (node, lane, start)
    return best


def _label_sharing(slots: list[Slot]) -> list[Slot]:
    """Record each run's mode and who overlapped with it on the same nodes."""
    labeled = []
    for slot in slots:
        same_node = [
            other.run.run_id for other in slots
            if other is not slot and set(other.nodes) & set(slot.nodes)
            and other.start < slot.end and slot.start < other.end
        ]
        reused = any(
            other is not slot and set(other.nodes) & set(slot.nodes) and other.end <= slot.start
            for other in slots
        )
        mode = NODE_SHARED if same_node else BACK_TO_BACK if reused else PARTITIONED
        labeled.append(replace(slot, mode=mode, co_runners=tuple(same_node)))
    return labeled


def plan_batch(
    runs: list[RunRequest],
    su_cap: float | None = None,
    seed: int = 0,
    node_share: int = 1,
) -> BatchPlan:
    """Pack one hardware type's runs, dropping the lowest gain-per-SU run until under the cap."""
    types = {run.node_type for run in runs}
    if len(types) != 1:
        raise ValueError(f"a batch uses one lease per hardware type, got {sorted(types)}")
    remaining = list(runs)
    dropped: list[RunRequest] = []
    while True:
        lease_nodes = lease_size(remaining)
        plan = BatchPlan(remaining[0].node_type, lease_nodes, remaining[0].rate,
                         pack(remaining, lease_nodes, seed, node_share), dropped)
        if su_cap is None or plan.su <= su_cap or len(remaining) == 1:
            return plan
        worst = min(remaining, key=lambda run: run.gain_per_su)
        remaining.remove(worst)
        dropped.append(worst)


def plan_batches(
    runs: list[RunRequest],
    su_cap: float | None = None,
    seed: int = 0,
    node_share: int = 1,
) -> list[BatchPlan]:
    """One lease per hardware type. The SU cap is shared across them by requested work."""
    groups: dict[str, list[RunRequest]] = {}
    for run in runs:
        groups.setdefault(run.node_type, []).append(run)
    total = sum(r.nodes * r.hours_p75 * r.rate for r in runs)
    plans = []
    for group in groups.values():
        share = sum(r.nodes * r.hours_p75 * r.rate for r in group) / max(total, 1e-12)
        cap = su_cap * share if su_cap is not None else None
        plans.append(plan_batch(group, cap, seed, node_share))
    return plans


def split_lease(plan: BatchPlan, max_hours: float = MAX_LEASE_HOURS) -> list[BatchPlan]:
    """Cap a planned lease at 7 days: consecutive leases, keeping each run whole.

    A run must fit inside one lease, so a window closes at the last run that fits. A later
    run overlapping that boundary starts the next lease from time zero.
    """
    if plan.makespan <= max_hours:
        return [plan]
    leases: list[BatchPlan] = []
    pending = sorted(plan.slots, key=lambda slot: slot.start)
    while pending:
        origin = pending[0].start
        fits = [s for s in pending if s.end - origin <= max_hours]
        if not fits:
            raise ValueError(f"run {pending[0].run.run_id} is longer than one lease")
        rest = [s for s in pending if s.end - origin > max_hours]
        shifted = [replace(s, start=s.start - origin, end=s.end - origin) for s in fits]
        leases.append(BatchPlan(plan.node_type, plan.lease_nodes, plan.rate, shifted))
        pending = rest
    return leases


def attributed_node_seconds(slot: Slot) -> float:
    return len(slot.nodes) * (slot.end - slot.start) * 3600.0


def interference(slot: Slot, slots: list[Slot], rates: dict[str, tuple[float, float]]) -> dict:
    """L_node, L_net, L_disk for one run. ``rates`` maps run id to (shuffle, disk) bytes/s.

    L_net counts every co-runner on the lease (they share a switch); L_node and L_disk
    only those on the same nodes.
    """
    overlapping = [
        other for other in slots
        if other is not slot and other.start < slot.end and slot.start < other.end
    ]
    same_node = [o for o in overlapping if set(o.nodes) & set(slot.nodes)]
    return {
        "L_node": float(len(same_node)),
        "L_net": sum(rates.get(o.run.run_id, (0.0, 0.0))[0] for o in overlapping),
        "L_disk": sum(rates.get(o.run.run_id, (0.0, 0.0))[1] for o in same_node),
    }


@dataclass
class Budget:
    """Running SU total. Issuing stops once only the reserve would remain."""

    total: float = 20_000.0
    reserve_fraction: float = DEFAULT_RESERVE
    spent: float = 0.0
    log: list[tuple[str, float]] = field(default_factory=list)

    @property
    def available(self) -> float:
        return self.total * (1.0 - self.reserve_fraction) - self.spent

    def can_issue(self, su: float) -> bool:
        return su <= self.available

    def charge(self, lease_id: str, su: float) -> None:
        self.spent += su
        self.log.append((lease_id, su))
