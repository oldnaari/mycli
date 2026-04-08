"""Slack webhook notification."""

import json
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError

from .detect import FailedJobInfo


_STATUS_EMOJI = {
    "FAILED": ":x:",
    "TIMEOUT": ":hourglass:",
    "CANCELLED": ":no_entry_sign:",
    "NODE_FAIL": ":warning:",
    "OUT_OF_MEMORY": ":boom:",
}


class SlackNotifier:
    def __init__(self, webhook_url: str | None, dry_run: bool = False):
        self.webhook_url = webhook_url
        self.dry_run = dry_run

    def send(self, job: FailedJobInfo, analysis: str) -> None:
        message = self._format_message(job, analysis)
        if self.dry_run:
            print("--- Slack Notification (dry run) ---")
            print(message["text"])
            print()
            for block in message.get("blocks", []):
                if block["type"] == "section" and "text" in block:
                    print(block["text"]["text"])
                elif block["type"] == "section" and "fields" in block:
                    for f in block["fields"]:
                        print(f["text"])
                elif block["type"] == "header":
                    print(block["text"]["text"])
            print("------------------------------------")
            return

        if not self.webhook_url:
            print("No webhook URL configured, skipping notification", file=sys.stderr)
            return

        try:
            data = json.dumps(message).encode()
            req = Request(
                self.webhook_url, data=data,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    print(f"Slack notification failed ({resp.status})", file=sys.stderr)
        except URLError as e:
            print(f"Slack notification error: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Slack notification error: {e}", file=sys.stderr)

    def _format_message(self, job: FailedJobInfo, analysis: str) -> dict:
        state_base = job.state.split()[0].rstrip("+")
        emoji = _STATUS_EMOJI.get(state_base, ":red_circle:")
        fallback = f"{emoji} Job {job.job_id} ({job.job_name}) {job.state}"

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} Job Failed: {job.job_id}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Job Name:*\n{job.job_name}"},
                    {"type": "mrkdwn", "text": f"*State:*\n{job.state}"},
                    {"type": "mrkdwn", "text": f"*Exit Code:*\n{job.exit_code}"},
                    {"type": "mrkdwn", "text": f"*Elapsed:*\n{job.elapsed}"},
                    {"type": "mrkdwn", "text": f"*Nodes:*\n{job.nodes}"},
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Analysis:*\n{analysis}",
                },
            },
        ]

        return {"text": fallback, "blocks": blocks}
