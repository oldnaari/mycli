"""Failure analysis via claude CLI."""

import subprocess

from .detect import FailedJobInfo


_PROMPT_WITH_LOG = """\
You are diagnosing a SLURM training job failure. Be concise: 2-4 sentences max.
Focus on root cause, not symptoms. If uncertain, say so.

Job ID: {job_id}
Job Name: {job_name}
State: {state}
Exit Code: {exit_code}
Nodes: {nodes}
Elapsed: {elapsed}

--- LOG TAIL (last lines) ---
{log_tail}
"""

_PROMPT_NO_LOG = """\
You are diagnosing a SLURM training job failure. Be concise: 2-4 sentences max.
No log file was found. Based on the metadata and exit code, provide any insights.

Job ID: {job_id}
Job Name: {job_name}
State: {state}
Exit Code: {exit_code}
Nodes: {nodes}
Elapsed: {elapsed}
"""


class FailureAnalyzer:
    def analyze(self, job: FailedJobInfo, log_tail: str | None) -> str:
        if log_tail:
            prompt = _PROMPT_WITH_LOG.format(
                job_id=job.job_id, job_name=job.job_name,
                state=job.state, exit_code=job.exit_code,
                nodes=job.nodes, elapsed=job.elapsed,
                log_tail=log_tail,
            )
        else:
            prompt = _PROMPT_NO_LOG.format(
                job_id=job.job_id, job_name=job.job_name,
                state=job.state, exit_code=job.exit_code,
                nodes=job.nodes, elapsed=job.elapsed,
            )

        try:
            result = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            return f"(Claude analysis unavailable: exit code {result.returncode})"
        except FileNotFoundError:
            return "(Claude CLI not found)"
        except subprocess.TimeoutExpired:
            return "(Claude analysis timed out)"
        except Exception as e:
            return f"(Claude analysis failed: {e})"
