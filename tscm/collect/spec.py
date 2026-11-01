"""The job specification: one YAML file per job (decision 8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

FRAMEWORKS = ("spark", "pytorch")
PARAM_TYPES = {"int": int, "float": (int, float), "str": str, "bool": bool}
DATA_FEATURES = (
    "record_count", "mean_record_size", "distinct_keys", "skew", "sparsity",
    "dataset_size", "sequence_length", "image_size",
)


class SpecError(ValueError):
    """A job specification that cannot be used, with every problem listed."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    profile: tuple[Any, ...] = ()


@dataclass(frozen=True)
class JobSpec:
    job: str
    family: str
    framework: str
    entry: str
    params: tuple[ParamSpec, ...]
    data_path: str
    features: tuple[str, ...]
    input_fractions: tuple[float, ...]
    machines: tuple[int, ...]
    extra: dict[str, Any] = field(default_factory=dict)


def _problems(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return ["the specification must be a mapping"]
    problems = []
    for key in ("job", "family", "framework", "entry", "data", "profile"):
        if key not in raw:
            problems.append(f"missing required key {key!r}")
    if raw.get("framework") not in FRAMEWORKS and "framework" in raw:
        problems.append(f"framework must be one of {FRAMEWORKS}, got {raw['framework']!r}")

    for name, declared in (raw.get("params") or {}).items():
        if not isinstance(declared, dict) or declared.get("type") not in PARAM_TYPES:
            problems.append(f"parameter {name}: type must be one of {sorted(PARAM_TYPES)}")
            continue
        expected = PARAM_TYPES[declared["type"]]
        for value in declared.get("profile", []):
            # bool is an int subclass, but True is not a valid int-typed sweep value.
            if not isinstance(value, expected) or (
                    isinstance(value, bool) and declared["type"] != "bool"):
                problems.append(f"parameter {name}: {value!r} is not {declared['type']}")

    data = raw.get("data") or {}
    if "data" in raw and "path" not in data:
        problems.append("data.path is required")
    for feature in data.get("features", []):
        if feature not in DATA_FEATURES:
            problems.append(f"unknown data feature {feature!r}")

    profile = raw.get("profile") or {}
    fractions = profile.get("input_fractions", [])
    machines = profile.get("machines", [])
    if "profile" in raw and not fractions:
        problems.append("profile.input_fractions is required")
    if "profile" in raw and not machines:
        problems.append("profile.machines is required")
    if not all(isinstance(f, (int, float)) and 0 < f <= 1 for f in fractions):
        problems.append("input fractions must lie in (0, 1]")
    if not all(isinstance(m, int) and m > 0 for m in machines):
        problems.append("machine counts must be positive integers")
    return problems


def parse_spec(raw: Any) -> JobSpec:
    problems = _problems(raw)
    if problems:
        raise SpecError("; ".join(problems))
    known = {"job", "family", "framework", "entry", "params", "data", "profile"}
    return JobSpec(
        job=raw["job"],
        family=raw["family"],
        framework=raw["framework"],
        entry=raw["entry"],
        params=tuple(
            ParamSpec(name, declared["type"], tuple(declared.get("profile", ())))
            for name, declared in (raw.get("params") or {}).items()
        ),
        data_path=raw["data"]["path"],
        features=tuple(raw["data"].get("features", ())),
        input_fractions=tuple(float(f) for f in raw["profile"]["input_fractions"]),
        machines=tuple(raw["profile"]["machines"]),
        extra={k: v for k, v in raw.items() if k not in known},
    )


def load_spec(path: str | Path) -> JobSpec:
    with open(path) as handle:
        return parse_spec(yaml.safe_load(handle))
