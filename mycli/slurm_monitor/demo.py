"""Demo data for testing without a Slurm cluster."""

import os

from .data import ClusterState, GpuInfo, JobInfo, NodeInfo


def build_demo_state() -> ClusterState:
    user = os.environ.get("USER", "me")
    nodes: dict[str, NodeInfo] = {}

    def _node(name: str, state: str, gpus: list[GpuInfo], partition: str = "a100",
              jobs: list[JobInfo] | None = None):
        nodes[name] = NodeInfo(name=name, state=state, total_gpus=len(gpus),
                               partition=partition, gpus=gpus, jobs=jobs or [])

    def _gpu(idx: int, usr: str | None = None, util: int = 0,
             mu: int = 0, mt: int = 81920, drained: bool = False) -> GpuInfo:
        return GpuInfo(index=idx, user=usr, utilization=util,
                       mem_used=mu, mem_total=mt, is_drained=drained)

    _node("gpu-node-01", "mixed", [
        _gpu(0, user, 85, 45000), _gpu(1, user, 92, 60000),
        _gpu(2, "alice", 40, 30000), _gpu(3, None, 0, 0),
    ])
    _node("gpu-node-02", "idle", [_gpu(i) for i in range(4)])
    _node("gpu-node-03", "mixed", [
        _gpu(0, "bob", 60, 50000), _gpu(1, "bob", 2, 1000),
        _gpu(2, None, 0, 0), _gpu(3, None, 0, 0),
    ])
    _node("gpu-node-04", "drained", [_gpu(i, drained=True) for i in range(4)])
    _node("gpu-node-05", "allocated", [
        _gpu(0, "carol", 95, 70000), _gpu(1, "carol", 88, 65000),
        _gpu(2, "dave", 76, 55000), _gpu(3, "dave", 91, 72000),
    ])
    _node("gpu-node-06", "mixed", [
        _gpu(0, user, 50, 40000), _gpu(1, "eve", 3, 500),
        _gpu(2, "eve", 70, 50000), _gpu(3, None, 0, 0),
        _gpu(4, None, 0, 0), _gpu(5, "frank", 45, 35000),
        _gpu(6, "frank", 80, 60000), _gpu(7, None, 0, 0),
    ])
    _node("gpu-node-07", "idle", [_gpu(i) for i in range(8)])
    _node("gpu-node-08", "mixed", [
        _gpu(0, "grace", 10, 8000), _gpu(1, "grace", 15, 12000),
        _gpu(2, None, 0, 0), _gpu(3, None, 0, 0),
    ])

    pending = [
        JobInfo("9001", "alice", "train", gpu_count=4, num_nodes=1, state="PENDING", priority=1000),
        JobInfo("9002", "bob", "eval", gpu_count=2, num_nodes=1, state="PENDING", priority=800),
        JobInfo("9003", user, "train", gpu_count=8, num_nodes=2, state="PENDING", priority=600),
    ]
    return ClusterState(nodes=nodes, pending_jobs=pending, current_user=user)
