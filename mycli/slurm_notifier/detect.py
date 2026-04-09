"""Job tracking and failure detection via squeue + sacct."""

from dataclasses import dataclass

from mycli.slurm_monitor.data import JobInfo, parse_squeue, run_cmd


FAILURE_STATES = {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"}


@dataclass
class FailedJobInfo:
    job_id: str
    job_name: str
    state: str
    exit_code: str
    nodes: str
    start_time: str
    end_time: str
    elapsed: str


class JobTracker:
    def __init__(self, user: str, job_ids: list[str] | None = None):
        self.user = user
        self.job_ids = set(job_ids) if job_ids else None
        self.known_running: dict[str, JobInfo] = {}
        self._initialized = False

    def poll(self) -> list[FailedJobInfo]:
        """One poll cycle. Returns newly-detected failed jobs."""
        squeue_out = run_cmd(
            f"squeue -u {self.user} -o '%i|%u|%P|%N|%b|%T|%Q|%D|%S' --noheader"
        )
        current_jobs = parse_squeue(squeue_out)
        active = {j.job_id: j for j in current_jobs
                  if j.state in ("RUNNING", "PENDING")}

        if self.job_ids:
            active = {k: v for k, v in active.items() if k in self.job_ids}

        if not self._initialized:
            self.known_running = active
            self._initialized = True
            ids = ", ".join(sorted(active)) if active else "none found"
            print(f"  Tracking {len(active)} active job(s): {ids}")
            return []

        disappeared = set(self.known_running) - set(active)
        self.known_running = active

        if not disappeared:
            return []

        return self._check_sacct(disappeared)

    def _check_sacct(self, job_ids: set[str]) -> list[FailedJobInfo]:
        ids_str = ",".join(job_ids)
        sacct_out = run_cmd(
            f"sacct --parsable2 --noheader "
            f"--format=JobID,JobName,State,ExitCode,Start,End,Elapsed,NodeList "
            f"-j {ids_str}"
        )
        failed = []
        for line in sacct_out.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 8:
                continue
            job_id, job_name, state, exit_code, start, end, elapsed, nodes = parts[:8]
            # Skip sub-jobs (e.g. "1351512.batch")
            if "." in job_id:
                continue
            state_base = state.split()[0].rstrip("+")
            if state_base in FAILURE_STATES:
                failed.append(FailedJobInfo(
                    job_id=job_id, job_name=job_name, state=state,
                    exit_code=exit_code, nodes=nodes,
                    start_time=start, end_time=end, elapsed=elapsed,
                ))
        return failed
