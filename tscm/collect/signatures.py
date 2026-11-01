"""Stage signatures, which match stages across runs whose stage ids differ.

SQL jobs hash the physical-plan operator chain, keeping operator names and key expressions
but not literal values. RDD jobs hash the call site plus lineage operator types. A signature
repeating within a job gets iteration indices 0, 1, 2 and so on.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable

_EXPR_ID = re.compile(r"#\d+L?")
_OBJECT_ID = re.compile(r"@[0-9a-f]{4,}\b")  # JVM identity hash, new in every application
_PLAN_ID = re.compile(r"\[(?:plan_id|id)=\d+\]|\(\d+\)|\*\(\d+\)|\bcodegen id\s*:\s*\d+")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[LDFSBY]?(?![\w.])")
_PATH = re.compile(r"\b(?:hdfs|file|s3a?)://\S+|/[\w./-]+")
_SPACE = re.compile(r"\s+")


def normalize_plan(plan: str) -> str:
    """Strip what changes between runs: literals, expression ids, paths and statistics."""
    lines = []
    for line in plan.splitlines():
        if line.startswith(("== ", "(")) and "Statistics" in line:
            continue
        text = _PATH.sub("<path>", line)
        text = _QUOTED.sub("?", text)
        text = _PLAN_ID.sub("", text)
        text = _EXPR_ID.sub("", text)
        text = _OBJECT_ID.sub("", text)
        text = _NUMBER.sub("?", text)
        lines.append(_SPACE.sub(" ", text).strip())
    return "\n".join(line for line in lines if line)


def _digest(*parts: Iterable[str] | str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update((part if isinstance(part, str) else "\x1f".join(part)).encode())
        hasher.update(b"\x1e")
    return hasher.hexdigest()[:16]


def stage_signature(
    scopes: list[str],
    call_site: str = "",
    rdd_types: list[str] | None = None,
    sql_plan: str | None = None,
) -> str:
    """Signature of one stage.

    ``scopes`` are the operator names of the stage's RDDs. Hashing them alongside the
    normalized plan separates the stages of a single SQL execution, which share a plan.
    """
    if sql_plan is not None:
        return _digest("sql", sorted(scopes), normalize_plan(sql_plan))
    return _digest("rdd", _normalize_call_site(call_site), sorted(rdd_types or []))


def _normalize_call_site(call_site: str) -> str:
    return _NUMBER.sub("?", call_site.split("\n", 1)[0]).strip()


def assign_iterations(stages: list[tuple[int, str]]) -> dict[int, tuple[str, int]]:
    """Map stage id to (signature, iteration), for stages in submission order."""
    seen: dict[str, int] = defaultdict(int)
    result = {}
    for stage_id, signature in stages:
        result[stage_id] = (signature, seen[signature])
        seen[signature] += 1
    return result


_TYPE_PATTERNS = (
    ("join", re.compile(r"join", re.I)),
    ("aggregation", re.compile(r"aggregate|reducebykey|groupby|combine", re.I)),
    ("shuffle_map", re.compile(r"exchange|shuffle|repartition|sort", re.I)),
    ("scan", re.compile(r"scan|textfile|parquet|hadooprdd|filerdd", re.I)),
)


def classify_stage(scopes: Iterable[str]) -> str:
    """Operator type for the hierarchical priors, taken from the stage's operator names."""
    text = " ".join(scopes)
    for name, pattern in _TYPE_PATTERNS:
        if pattern.search(text):
            return name
    return "other"
