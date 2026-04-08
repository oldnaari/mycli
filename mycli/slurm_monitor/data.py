"""Dataclasses and Slurm/nvidia-smi parsing."""

import re
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GpuInfo:
    index: int
    user: str | None = None
    utilization: int = 0
    mem_used: int = 0
    mem_total: int = 0
    is_drained: bool = False
    start_time: datetime | None = None


@dataclass
class JobInfo:
    job_id: str
    user: str
    partition: str
    nodes: list[str] = field(default_factory=list)
    gpu_count: int = 0
    state: str = ""
    priority: int = 0
    num_nodes: int = 1
    start_time: datetime | None = None


@dataclass
class NodeInfo:
    name: str
    state: str = "unknown"
    total_gpus: int = 0
    partition: str = ""
    gpus: list[GpuInfo] = field(default_factory=list)
    jobs: list[JobInfo] = field(default_factory=list)


@dataclass
class ClusterState:
    nodes: dict[str, NodeInfo] = field(default_factory=dict)
    pending_jobs: list[JobInfo] = field(default_factory=list)
    current_user: str = ""


# ---------------------------------------------------------------------------
# Shell helpers (blocking)
# ---------------------------------------------------------------------------

def run_cmd(cmd: str, timeout: int = 15) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Slurm parsing
# ---------------------------------------------------------------------------

def parse_sinfo(output: str) -> dict[str, NodeInfo]:
    nodes: dict[str, NodeInfo] = {}
    for line in output.strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[0]
        state = parts[1].rstrip("*").lower()
        partition = parts[2].rstrip("*")
        gres = parts[3]
        total_gpus = 0
        m = re.search(r"gpu(?::[^:,(]+)*:(\d+)", gres)
        if m:
            total_gpus = int(m.group(1))
        if name in nodes:
            # Node in multiple partitions — append partition name
            existing = nodes[name]
            if partition not in existing.partition.split(","):
                existing.partition += f",{partition}"
        else:
            nodes[name] = NodeInfo(name=name, state=state, total_gpus=total_gpus,
                                   partition=partition)
    return nodes


def parse_squeue(output: str) -> list[JobInfo]:
    jobs: list[JobInfo] = []
    for line in output.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 9:
            continue
        job_id, user, partition, nodelist, tres, state, priority, num_nodes, start = (
            p.strip() for p in parts[:9]
        )
        gpu_count = 0
        m = re.search(r"gpu(?::[^:,(]+)*:(\d+)", tres)
        if m:
            gpu_count = int(m.group(1))
        start_time = None
        try:
            start_time = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError):
            pass
        jobs.append(JobInfo(
            job_id=job_id, user=user, partition=partition,
            nodes=[nodelist] if nodelist else [], gpu_count=gpu_count,
            state=state.upper(),
            priority=int(priority) if priority.isdigit() else 0,
            num_nodes=int(num_nodes) if num_nodes.isdigit() else 1,
            start_time=start_time,
        ))
    return jobs


def expand_nodelist(nodelist: str) -> list[str]:
    out = run_cmd(f"scontrol show hostnames {nodelist}")
    return [n.strip() for n in out.strip().splitlines() if n.strip()]


# ---------------------------------------------------------------------------
# nvidia-smi (blocking)
# ---------------------------------------------------------------------------

def parse_nvidia_smi(output: str) -> list[tuple[int, int, int, int]]:
    """Return list of (index, utilization%, mem_used_MiB, mem_total_MiB)."""
    results = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                results.append((int(parts[0]), int(parts[1]),
                                int(parts[2]), int(parts[3])))
            except ValueError:
                continue
    return results


_ssh_semaphore = threading.Semaphore(20)


def fetch_nvidia_smi(node: str) -> tuple[str, str | None]:
    """Blocking SSH call to fetch nvidia-smi data from a node."""
    with _ssh_semaphore:
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                 "-o", "BatchMode=yes", node,
                 "nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return (node, result.stdout.strip())
        except Exception:
            pass
    return (node, None)


def apply_nvidia_smi(state: ClusterState, node_name: str, output: str) -> None:
    """Apply nvidia-smi output to a single node in-place."""
    if node_name not in state.nodes:
        return
    node = state.nodes[node_name]
    for gidx, util, mem_used, mem_total in parse_nvidia_smi(output):
        if 0 <= gidx < len(node.gpus):
            node.gpus[gidx].utilization = util
            node.gpus[gidx].mem_used = mem_used
            node.gpus[gidx].mem_total = mem_total


# ---------------------------------------------------------------------------
# Build cluster state (blocking — no SSH)
# ---------------------------------------------------------------------------

def refresh_slurm_state(current_user: str) -> ClusterState:
    """Fetch sinfo + squeue and build cluster state. No nvidia-smi."""
    sinfo_out = run_cmd("sinfo -N -o '%N %T %P %G' --noheader")
    squeue_out = run_cmd("squeue -o '%i|%u|%P|%N|%b|%T|%Q|%D|%S' --noheader")

    nodes = parse_sinfo(sinfo_out)
    all_jobs = parse_squeue(squeue_out)

    pending_jobs: list[JobInfo] = []
    for job in all_jobs:
        if job.state == "PENDING":
            pending_jobs.append(job)
            continue
        if job.state != "RUNNING":
            continue
        expanded: list[str] = []
        for nl in job.nodes:
            if nl and nl != "(null)":
                expanded.extend(expand_nodelist(nl))
        job.nodes = expanded
        for node_name in expanded:
            if node_name in nodes:
                nodes[node_name].jobs.append(job)

    for node in nodes.values():
        is_drain = "drain" in node.state or "down" in node.state
        node.gpus = [GpuInfo(index=i, is_drained=is_drain) for i in range(node.total_gpus)]
        idx = 0
        sorted_jobs = sorted(node.jobs, key=lambda j: j.job_id)
        # First pass: assign GPUs for jobs with known gpu_count
        for job in sorted_jobs:
            if job.gpu_count > 0:
                for _ in range(job.gpu_count):
                    if idx < len(node.gpus):
                        node.gpus[idx].user = job.user
                        node.gpus[idx].start_time = job.start_time
                        idx += 1
        # Second pass: jobs with gpu_count=0 (GRES=N/A) get remaining free GPUs
        zero_jobs = [j for j in sorted_jobs if j.gpu_count == 0]
        if zero_jobs:
            free_gpus = [g for g in node.gpus if g.user is None and not g.is_drained]
            per_job = max(1, len(free_gpus) // len(zero_jobs)) if free_gpus else 0
            fi = 0
            for job in zero_jobs:
                for _ in range(per_job):
                    if fi < len(free_gpus):
                        free_gpus[fi].user = job.user
                        free_gpus[fi].start_time = job.start_time
                        fi += 1

    pending_jobs.sort(key=lambda j: j.priority, reverse=True)
    return ClusterState(nodes=nodes, pending_jobs=pending_jobs, current_user=current_user)
