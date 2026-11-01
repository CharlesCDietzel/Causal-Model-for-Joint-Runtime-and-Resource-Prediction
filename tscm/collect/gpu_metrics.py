"""GPU telemetry: DCGM or nvidia-smi sampling, and PyTorch memory and step hooks.

torch is imported on first use, so this module loads without the ``gpu`` extra.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

# DCGM field ids: framebuffer used, GPU util, SM activity, PCIe tx/rx, NVLink total.
DCGM_FIELDS = (252, 203, 1002, 1009, 1010, 1011)
SAMPLE_MS = 1000
PHASES = ("data_loading", "forward_backward", "communication")


def sampler_command(sample_ms: int = SAMPLE_MS) -> list[str]:
    """DCGM when installed, otherwise nvidia-smi, which still gives memory.used and util."""
    if shutil.which("dcgmi"):
        fields = ",".join(str(f) for f in DCGM_FIELDS)
        return ["dcgmi", "dmon", "-e", fields, "-d", str(sample_ms)]
    return [
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits", f"--loop-ms={sample_ms}",
    ]


@dataclass
class GpuSample:
    gpu: int
    memory_mib: float
    utilization: float


def parse_nvidia_smi(line: str) -> GpuSample | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3 or not parts[0].isdigit():
        return None
    return GpuSample(int(parts[0]), float(parts[1]), float(parts[2]))


def parse_dcgm(line: str) -> GpuSample | None:
    """One `dcgmi dmon` row: 'GPU 0  <fb used> <gpu util> ...'. Headers return None."""
    parts = line.split()
    if len(parts) < 4 or parts[0] != "GPU" or not parts[1].isdigit():
        return None
    return GpuSample(int(parts[1]), float(parts[2]), float(parts[3]))


class GpuSampler:
    """Background sampler with a process lifetime around the measured window."""

    def __init__(self, output: Path, sample_ms: int = SAMPLE_MS):
        self.output, self.sample_ms = Path(output), sample_ms
        self._process: subprocess.Popen | None = None
        self._handle = None

    def __enter__(self) -> GpuSampler:
        self._handle = open(self.output, "w")
        self._process = subprocess.Popen(
            sampler_command(self.sample_ms), stdout=self._handle, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc) -> None:
        self._process.terminate()
        self._process.wait(timeout=10)
        self._handle.close()

    def peak_memory_bytes(self) -> float:
        peak = 0.0
        with open(self.output) as handle:
            for line in handle:
                sample = parse_dcgm(line) or parse_nvidia_smi(line)
                if sample is not None:
                    peak = max(peak, sample.memory_mib * 1024.0**2)
        return peak


@dataclass
class StepRecorder:
    """Per-step time split by phase, plus allocator peaks, for step-limited profiling runs.

    Mark phases from the training loop:

        with recorder.step() as step:
            with step.phase("data_loading"): batch = next(loader)
            with step.phase("forward_backward"): loss = model(batch); loss.backward()
            with step.phase("communication"): optimizer.step()

    The first ``warmup`` steps are discarded. Runs execute 10 warmup plus 50 measured steps.
    """

    warmup: int = 10
    records: list[dict] = field(default_factory=list)
    _count: int = 0

    def step(self) -> _Step:
        return _Step(self)

    def _finish(self, phases: dict[str, float], total: float) -> None:
        self._count += 1
        if self._count > self.warmup:
            self.records.append({"step": self._count, "total": total, **phases, **_allocator()})

    def summary(self) -> dict[str, float]:
        if not self.records:
            return {}
        count = len(self.records)
        mean = lambda key: sum(r.get(key, 0.0) for r in self.records) / count  # noqa: E731
        return {
            "T_step": mean("total"),
            **{name: mean(name) for name in PHASES},
            "G_allocated": max(r.get("allocated", 0.0) for r in self.records),
            "G_reserved": max(r.get("reserved", 0.0) for r in self.records),
        }

    def dump(self, path: Path) -> None:
        with open(path, "w") as handle:
            for record in self.records:
                handle.write(json.dumps(record) + "\n")


class _Step:
    def __init__(self, recorder: StepRecorder):
        self.recorder = recorder
        self.phases: dict[str, float] = {}

    def __enter__(self) -> _Step:
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.recorder._finish(self.phases, time.perf_counter() - self.started)

    def phase(self, name: str) -> _Phase:
        return _Phase(self, name)


class _Phase:
    def __init__(self, step: _Step, name: str):
        self.step, self.name = step, name

    def __enter__(self) -> None:
        self.started = time.perf_counter()

    def __exit__(self, *exc) -> None:
        elapsed = time.perf_counter() - self.started
        self.step.phases[self.name] = self.step.phases.get(self.name, 0.0) + elapsed


def _allocator() -> dict[str, float]:
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated": float(torch.cuda.max_memory_allocated()),
        "reserved": float(torch.cuda.max_memory_reserved()),
    }


def cap_gpu_memory(fraction: float, device: int = 0) -> None:
    """The memory-shrink trick for GPUs: locate the OOM boundary without a smaller GPU."""
    import torch

    torch.cuda.set_per_process_memory_fraction(fraction, device)
