"""The causal skeleton: four tiers, required edges, forbidden edges and what is left open."""

from __future__ import annotations

from dataclasses import dataclass

TIERS: dict[int, tuple[str, ...]] = {
    0: ("s", "m", "h", "M_e", "e", "c_e", "p", "d", "partitions", "skew",
        "n_g", "b", "P", "precision", "shard", "ckpt", "R_gpu", "F_gpu", "B_ic",
        "L_node", "L_net", "L_disk"),
    1: ("V_in", "V_sh", "V_out", "n", "C"),
    2: ("W", "rho", "S", "g", "G", "rho_gpu"),
    3: ("T_k", "T", "u", "M_peak", "M_req", "T_step", "G_peak", "G_req", "OOM", "u_gpu"),
}

REQUIRED_EDGES: frozenset[tuple[str, str]] = frozenset(
    {(a, b) for a in ("s", "d", "p") for b in ("V_in", "V_sh", "V_out")}
    | {("V_in", "n"), ("partitions", "n")}
    | {("V_in", "W"), ("n", "W"), ("skew", "W")}
    | {("W", "rho"), ("C", "rho"), ("c_e", "rho"), ("M_e", "rho")}
    | {("rho", "S"), ("rho", "g"), ("V_sh", "S")}
    | {(a, "T_k") for a in ("V_in", "V_sh", "S", "g", "n", "m", "c_e", "h")}
    | {("T_k", "T"), ("T_k", "u")}
    | {(a, b) for a in ("W", "C", "M_e") for b in ("M_peak", "M_req")}
    # GPU jobs use the same tiers.
    | {("P", "G"), ("b", "G"), ("shard", "G"), ("ckpt", "G"), ("precision", "G"),
       ("G", "rho_gpu"), ("R_gpu", "rho_gpu"), ("rho_gpu", "OOM"), ("G", "G_peak"),
       ("G", "G_req"), ("b", "T_step"), ("F_gpu", "T_step"), ("B_ic", "T_step"),
       ("n_g", "T_step"), ("m", "T_step"), ("T_step", "T"), ("T_step", "u_gpu")}
)

# Machines and memory do not change data volumes; this test catches plan changes such as AQE.
_CONFIG_NOT_INTO_VOLUMES = ("m", "M_e", "h")
_VOLUMES = ("V_in", "V_sh", "V_out")

# Discovery only runs among these. W is a function of its parents, and the design's
# residualization leaves only what the fixed parents do not explain.
DISCOVERY_NODES: tuple[str, ...] = ("W", "rho", "S", "g", "T_k", "u", "M_peak")


@dataclass(frozen=True)
class Skeleton:
    tiers: dict[int, tuple[str, ...]]
    required: frozenset[tuple[str, str]]

    def tier(self, node: str) -> int:
        for tier, nodes in self.tiers.items():
            if node in nodes:
                return tier
        raise KeyError(f"unknown node {node!r}")

    def forbidden(self, parent: str, child: str) -> bool:
        if self.tier(child) == 0:
            return True  # outcomes never change the configuration
        if self.tier(parent) > self.tier(child):
            return True
        return parent in _CONFIG_NOT_INTO_VOLUMES and child in _VOLUMES

    def open_for_discovery(self, parent: str, child: str) -> bool:
        """Tier 2 to Tier 2 and Tier 2 to Tier 3 edges beyond the required ones."""
        if (parent, child) in self.required or self.forbidden(parent, child):
            return False
        return self.tier(parent) == 2 and self.tier(child) in (2, 3)

    def parents(self, node: str) -> list[str]:
        return sorted(parent for parent, child in self.required if child == node)

    def validate(self) -> None:
        for parent, child in self.required:
            if self.forbidden(parent, child):
                raise ValueError(f"required edge {parent} -> {child} is forbidden")
        order = {node: tier for tier, nodes in self.tiers.items() for node in nodes}
        children: dict[str, list[str]] = {}
        for parent, child in self.required:
            children.setdefault(parent, []).append(child)
        state: dict[str, int] = {}

        def visit(node: str) -> None:
            if state.get(node) == 1:
                raise ValueError(f"cycle through {node}")
            if state.get(node) == 2:
                return
            state[node] = 1
            for child in children.get(node, ()):
                visit(child)
            state[node] = 2

        for node in order:
            visit(node)


def default_skeleton() -> Skeleton:
    skeleton = Skeleton(TIERS, REQUIRED_EDGES)
    skeleton.validate()
    return skeleton
