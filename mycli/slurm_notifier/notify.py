"""Push notification via ntfy.sh."""

import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

from .detect import FailedJobInfo


_LOG_FILE = Path.home() / ".slurm-notifier-last.log"

_STATUS_TAG = {
    "FAILED": "x",
    "TIMEOUT": "hourglass",
    "CANCELLED": "no_entry_sign",
    "NODE_FAIL": "warning",
    "OUT_OF_MEMORY": "boom",
}


class Notifier:
    def __init__(self, ntfy_topic: str | None, dry_run: bool = False):
        self.ntfy_topic = ntfy_topic
        self.dry_run = dry_run

    def send(self, job: FailedJobInfo, analysis: str) -> None:
        title, body = self._format_message(job, analysis)
        self._write_log(title, body)

        if self.dry_run:
            print(f"--- Notification (dry run) ---")
            print(f"{title}\n")
            print(body)
            print("------------------------------")
            return

        if not self.ntfy_topic:
            print("No ntfy topic configured, skipping notification", file=sys.stderr)
            return

        state_base = job.state.split()[0].rstrip("+")
        tag = _STATUS_TAG.get(state_base, "red_circle")

        try:
            url = f"https://ntfy.sh/{self.ntfy_topic}"
            data = body.encode()
            req = Request(url, data=data, headers={
                "Title": title,
                "Tags": tag,
            })
            with urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    print(f"ntfy notification failed ({resp.status})", file=sys.stderr)
        except URLError as e:
            print(f"ntfy notification error: {e}", file=sys.stderr)
        except Exception as e:
            print(f"ntfy notification error: {e}", file=sys.stderr)

    def _format_message(self, job: FailedJobInfo, analysis: str) -> tuple[str, str]:
        state_base = job.state.split()[0].rstrip("+")

        title = f"Job {job.job_id} ({job.job_name}) {state_base}"

        body = (
            f"State: {job.state}\n"
            f"Exit Code: {job.exit_code}\n"
            f"Nodes: {job.nodes}\n"
            f"Elapsed: {job.elapsed}\n"
            f"\n"
            f"Analysis:\n{analysis}"
        )

        return title, body

    def _write_log(self, title: str, body: str) -> None:
        try:
            _LOG_FILE.write_text(f"{title}\n\n{body}\n")
        except OSError:
            pass
