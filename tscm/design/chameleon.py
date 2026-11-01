"""Chameleon leases via python-chi, and the shell setup each lease partition needs.

python-chi is imported on first use, so the rest of the package works without the
``chameleon`` extra. Lease calls need a Chameleon project and credentials in the environment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from tscm.design.packing import CPU_RATE, GPU_RATE, MAX_LEASE_HOURS, BatchPlan
from tscm.model import GiB, Machine, MiB

CALIBRATION_FIO_SECONDS = 30
CALIBRATION_IPERF_SECONDS = 15


def available_node_types() -> list[str]:
    """Node types are unknown in advance, so the scheduler reads them at lease time."""
    from chi import hardware

    return sorted(hardware.get_node_types())


def create_lease(name: str, node_type: str, nodes: int, hours: float, wait: bool = True):
    """Reserve ``nodes`` nodes for ``hours`` hours. Spends SUs from when it becomes active."""
    if hours > MAX_LEASE_HOURS:
        raise ValueError(f"a lease cannot exceed {MAX_LEASE_HOURS} hours; split the plan first")
    from chi import lease

    reservation = lease.Lease(name, duration=timedelta(hours=hours))
    reservation.add_node_reservation(amount=nodes, node_type=node_type)
    reservation.submit(wait_for_active=wait, idempotent=True)
    return reservation


def create_lease_for(plan: BatchPlan, name: str, wait: bool = True):
    return create_lease(name, plan.node_type, plan.lease_nodes, plan.makespan, wait)


def release_lease(reservation) -> None:
    """End the lease as soon as its batch finishes: idle nodes cost as much as busy ones."""
    reservation.delete()


def su_rate(gpu_count: int) -> float:
    return GPU_RATE if gpu_count > 0 else CPU_RATE


def partition_hosts(hosts: Sequence[str], sizes: Sequence[int]) -> list[list[str]]:
    """Split a lease's hosts into disjoint groups, each running its own Spark cluster."""
    if sum(sizes) > len(hosts):
        raise ValueError(f"partitions need {sum(sizes)} hosts, lease has {len(hosts)}")
    groups, cursor = [], 0
    for size in sizes:
        groups.append(list(hosts[cursor:cursor + size]))
        cursor += size
    return groups


def spark_partition_script(
    hosts: Sequence[str],
    spark_home: str = "/opt/spark",
    extra_conf: dict[str, str] | None = None,
) -> str:
    """Start a standalone Spark 3.5 cluster on one partition, with AQE off (decision 5).

    The first host is the master; it also runs a worker so the partition has len(hosts)
    workers, matching the machine count the run is recorded with.
    """
    master, conf = hosts[0], {
        "spark.sql.adaptive.enabled": "false",
        "spark.executor.processTreeMetrics.enabled": "true",
        "spark.executor.metrics.pollingInterval": "1s",
        "spark.extraListeners": "org.apache.spark.tscm.TscmListener",
        **(extra_conf or {}),
    }
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cat > {spark_home}/conf/spark-defaults.conf <<'CONF'",
        *(f"{key} {value}" for key, value in conf.items()),
        "CONF",
        f"{spark_home}/sbin/start-master.sh -h {master}",
    ]
    lines += [
        f"ssh {host} {spark_home}/sbin/start-worker.sh spark://{master}:7077" for host in hosts
    ]
    return "\n".join(lines) + "\n"


def between_runs_script(hosts: Sequence[str], spark_home: str = "/opt/spark") -> str:
    """Back-to-back isolation: drop the page cache and restart executors for fresh JVMs."""
    master = hosts[0]
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for host in hosts:
        lines.append(f"ssh {host} 'sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches'")
        lines.append(f"ssh {host} {spark_home}/sbin/stop-worker.sh")
        lines.append(f"ssh {host} {spark_home}/sbin/start-worker.sh spark://{master}:7077")
    return "\n".join(lines) + "\n"


def node_shared_command(command: str, cores: str, memory_bytes: int, name: str) -> str:
    """Pin a Phase A run to cores and memory with a transient cgroup scope."""
    return (f"sudo systemd-run --scope --unit=tscm-{name} -p AllowedCPUs={cores} "
            f"-p MemoryMax={memory_bytes} {command}")


def gpu_visible_devices(indices: Sequence[int]) -> str:
    """GPU nodes are shared by whole GPUs."""
    return "CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in indices)


def calibration_script(peer: str) -> str:
    """First use of a node type: measure the physical quantities the model reads."""
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo "cores=$(nproc)"
echo "memory_kib=$(awk '/MemTotal/ {{print $2}}' /proc/meminfo)"
fio --name=tscm --rw=write --bs=1M --size=4G --direct=1 --ioengine=libaio \\
    --runtime={CALIBRATION_FIO_SECONDS} --time_based --output-format=terse --terse-version=3 \\
    | awk -F';' '{{print "disk_write_kib_s=" $48}}'
iperf3 -c {peer} -t {CALIBRATION_IPERF_SECONDS} -J \\
    | python3 -c "import json,sys; d=json.load(sys.stdin); \\
print('net_bits_s=%d' % d['end']['sum_received']['bits_per_second'])"
"""


@dataclass(frozen=True)
class Calibration:
    cores: int
    memory_bytes: float
    disk_bytes_s: float
    net_bytes_s: float


def parse_calibration(output: str) -> Calibration:
    values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    return Calibration(
        cores=int(values["cores"]),
        memory_bytes=float(values["memory_kib"]) * 1024.0,
        disk_bytes_s=float(values["disk_write_kib_s"]) * 1024.0,
        net_bytes_s=float(values["net_bits_s"]) / 8.0,
    )


def machine_from_calibration(name: str, result: Calibration, gpus: int = 0) -> Machine:
    return Machine(name, result.cores, result.memory_bytes, result.disk_bytes_s,
                   result.net_bytes_s, su_rate(gpus))


def default_executor_layout(machine: Machine, c_e: int = 4) -> dict[str, float]:
    """Executors per machine and memory per executor, leaving 4 GiB for the OS and daemons."""
    e = max(machine.cores // c_e, 1)
    return {"e": e, "c_e": c_e, "M_e": max((machine.memory - 4 * GiB) / e, 512 * MiB)}
