"""CLI entry point and main polling loop."""

import argparse
import os
import sys
import time

from .analyze import FailureAnalyzer
from .detect import JobTracker
from .logs import LogFinder
from .notify import Notifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="slurm-notifier",
        description="Monitor SLURM jobs and send push notifications on failure.",
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=60,
        help="Polling interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--log-dir", default="~/experiment-logs/text-logs/",
        help="Directory to search for job log files",
    )
    parser.add_argument(
        "--tail-lines", type=int, default=200,
        help="Number of log lines to send to Claude for analysis (default: 200)",
    )
    parser.add_argument(
        "--ntfy-topic",
        default=os.environ.get("NTFY_TOPIC"),
        help="ntfy.sh topic name (or set NTFY_TOPIC env var)",
    )
    parser.add_argument(
        "--job", "-j", nargs="+",
        help="Track specific job ID(s) instead of all running jobs",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print notifications to stdout instead of sending",
    )
    parser.add_argument(
        "--bash-init", action="store_true",
        help="Print bash completion script and exit. Usage: eval \"$(slurm-notifier --bash-init)\"",
    )
    return parser.parse_args()


_BASH_COMPLETION = """\
_slurm_notifier() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    if [[ "$prev" == "-j" || "$prev" == "--job" ]]; then
        COMPREPLY=($(compgen -W "$(squeue -u $USER -h -o '%i' 2>/dev/null)" -- "$cur"))
    fi
}
complete -F _slurm_notifier slurm-notifier
"""


def main():
    args = parse_args()

    if args.bash_init:
        print(_BASH_COMPLETION)
        return

    if not args.dry_run and not args.ntfy_topic:
        print(
            "Error: --ntfy-topic or NTFY_TOPIC env var required "
            "(or use --dry-run)",
            file=sys.stderr,
        )
        sys.exit(1)

    user = os.environ.get("USER", "unknown")
    tracker = JobTracker(user=user, job_ids=args.job)
    log_finder = LogFinder(args.log_dir, args.tail_lines)
    analyzer = FailureAnalyzer()
    notifier = Notifier(args.ntfy_topic, dry_run=args.dry_run)

    print(f"slurm-notifier started (user={user}, interval={args.interval}s)")

    while True:
        try:
            failed_jobs = tracker.poll()
            for job in failed_jobs:
                print(f"  Detected failure: job {job.job_id} ({job.job_name}) -> {job.state}")
                log_tail = log_finder.find_and_tail(job.job_id)
                if log_tail:
                    print(f"  Found log, analyzing with Claude...")
                else:
                    print(f"  No log file found, analyzing metadata only...")
                analysis = analyzer.analyze(job, log_tail)
                notifier.send(job, analysis)
        except KeyboardInterrupt:
            print("\nslurm-notifier stopped.")
            break
        except Exception as e:
            print(f"Error in poll cycle: {e}", file=sys.stderr)

        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nslurm-notifier stopped.")
            break
