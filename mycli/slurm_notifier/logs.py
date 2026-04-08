"""Log file discovery and tail extraction."""

from pathlib import Path


class LogFinder:
    def __init__(self, log_dir: str, tail_lines: int = 200):
        self.log_dir = Path(log_dir).expanduser()
        self.tail_lines = tail_lines

    def find_log(self, job_id: str) -> Path | None:
        """Find the log file for a given SLURM job ID."""
        if not self.log_dir.exists():
            return None
        matches = list(self.log_dir.glob(f"*{job_id}*.log"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Prefer exact boundary match: {id}-*.log or *-{id}.log
            for m in matches:
                stem = m.stem
                if stem.startswith(f"{job_id}-") or stem.endswith(f"-{job_id}"):
                    return m
            return matches[0]
        return None

    def find_and_tail(self, job_id: str) -> str | None:
        """Find log file and return its tail."""
        path = self.find_log(job_id)
        if path is None:
            return None
        try:
            lines = path.read_text(errors="replace").splitlines()
            tail = lines[-self.tail_lines:] if len(lines) > self.tail_lines else lines
            return "\n".join(tail)
        except OSError:
            return None
