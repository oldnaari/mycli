#!/usr/bin/env python3

import argparse
import os
import stat
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import paramiko
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)


def parse_remote_path(remote: str):
    """Parse username@host:path into (user, host, remote_dir, dir_name)."""
    if ":" not in remote:
        print(f"Invalid remote path: {remote}", file=sys.stderr)
        print(
            "Expected format: username@address-or-name:directory/parent/directory-name",
            file=sys.stderr,
        )
        sys.exit(1)

    user_host, remote_dir = remote.split(":", 1)
    remote_dir = remote_dir.rstrip("/")
    if not remote_dir:
        print("Remote directory path cannot be empty.", file=sys.stderr)
        sys.exit(1)

    if "@" in user_host:
        user, host = user_host.split("@", 1)
    else:
        user = None
        host = user_host

    dir_name = remote_dir.rsplit("/", 1)[-1]
    return user, host, remote_dir, dir_name


def connect(user, host, console):
    """Establish SSH connection using agent/keys and ~/.ssh/config."""
    ssh_config = paramiko.SSHConfig()
    config_path = Path.home() / ".ssh" / "config"
    if config_path.exists():
        with open(config_path) as f:
            ssh_config.parse(f)

    host_cfg = ssh_config.lookup(host)
    actual_host = host_cfg.get("hostname", host)
    actual_user = user or host_cfg.get("user", os.getenv("USER"))
    actual_port = int(host_cfg.get("port", 22))

    key_filename = host_cfg.get("identityfile")

    # Handle ProxyJump and ProxyCommand from ~/.ssh/config
    sock = None
    proxy_command = host_cfg.get("proxycommand")
    if not proxy_command:
        proxy_jump = host_cfg.get("proxyjump")
        if proxy_jump:
            # Convert ProxyJump to an equivalent ProxyCommand
            # ProxyJump can be user@host:port or just host
            jump_cfg = ssh_config.lookup(proxy_jump)
            jump_host = jump_cfg.get("hostname", proxy_jump)
            jump_user = jump_cfg.get("user", actual_user)
            jump_port = int(jump_cfg.get("port", 22))
            jump_key = jump_cfg.get("identityfile")
            jump_key_flag = f"-i {jump_key[0]} " if jump_key else ""
            proxy_command = (
                f"ssh -W %h:%p {jump_key_flag}"
                f"-p {jump_port} {jump_user}@{jump_host}"
            )

    if proxy_command:
        # Expand %h, %p, %r tokens
        proxy_command = proxy_command.replace("%h", actual_host)
        proxy_command = proxy_command.replace("%p", str(actual_port))
        proxy_command = proxy_command.replace("%r", actual_user)
        console.print(
            f"Using proxy: {proxy_command}",
            style="bright_black",
        )
        sock = paramiko.ProxyCommand(proxy_command)

    console.print(
        f"Connecting to {actual_user}@{actual_host}:{actual_port}...",
        style="bright_black",
    )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=actual_host,
        port=actual_port,
        username=actual_user,
        key_filename=key_filename,
        allow_agent=True,
        look_for_keys=True,
        sock=sock,
    )
    return client


def sftp_walk(sftp, remote_dir):
    """Recursively walk a remote directory. Yields (dirpath, dirnames, filenames)."""
    queue = deque([remote_dir])
    while queue:
        dirpath = queue.popleft()
        try:
            entries = sftp.listdir_attr(dirpath)
        except IOError:
            continue

        dirs = []
        files = []
        for entry in entries:
            if stat.S_ISDIR(entry.st_mode):
                dirs.append(entry.filename)
                queue.append(f"{dirpath}/{entry.filename}")
            else:
                files.append(entry.filename)

        yield dirpath, dirs, files


def scan_remote(sftp, remote_dir, console):
    """Scan remote directory, return (relative_dirs, relative_files)."""
    console.print("Scanning remote directory...", style="bright_black")

    rel_dirs = []
    rel_files = []
    prefix = remote_dir + "/"

    for dirpath, dirs, files in sftp_walk(sftp, remote_dir):
        if dirpath == remote_dir:
            rel_prefix = ""
        else:
            rel_prefix = dirpath[len(prefix):] + "/"

        for d in dirs:
            rel_dirs.append(f"{rel_prefix}{d}")
        for f in files:
            rel_files.append(f"{rel_prefix}{f}")

    console.print(
        f"Found {len(rel_dirs)} directories, {len(rel_files)} files.",
        style="bright_black",
    )
    return rel_dirs, rel_files


_thread_local = threading.local()


def download_file(transport, remote_dir, local_base, rel_path):
    """Download a single file via a thread-local SFTP channel."""
    if not hasattr(_thread_local, "sftp"):
        _thread_local.sftp = paramiko.SFTPClient.from_transport(transport)

    remote_path = f"{remote_dir}/{rel_path}"
    local_path = str(local_base / rel_path)
    try:
        _thread_local.sftp.get(remote_path, local_path)
        return rel_path, True, None
    except Exception as e:
        return rel_path, False, str(e)


def download_files(transport, remote_dir, local_base, files, workers, console, label="Downloading"):
    """Download files in parallel with a progress bar. Returns list of (rel_path, error)."""
    errors = []
    if not files:
        return errors

    error_count = 0
    with Progress(
        TextColumn("[bright_black]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[bright_black]eta"),
        TimeRemainingColumn(),
        TextColumn("[red]{task.fields[errors]} errors"),
        console=console,
    ) as progress:
        task = progress.add_task(label, total=len(files), errors=0)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(download_file, transport, remote_dir, local_base, f): f
                for f in files
            }
            for future in as_completed(futures):
                rel, ok, err = future.result()
                if not ok:
                    error_count += 1
                    errors.append((rel, err))
                progress.update(task, advance=1, errors=error_count)

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Download a remote directory via SFTP.",
    )
    parser.add_argument(
        "remote",
        help="Remote path: username@host:directory/path",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=10,
        help="Parallel download workers (default: 10)",
    )
    args = parser.parse_args()

    console = Console(highlight=False)

    user, host, remote_dir, dir_name = parse_remote_path(args.remote)
    local_base = Path.home() / "Downloads" / dir_name

    console.print(f"Target: ~/Downloads/{dir_name}", style="bright_black")

    # Connect
    client = connect(user, host, console)
    transport = client.get_transport()

    try:
        # Scan remote directory
        sftp = paramiko.SFTPClient.from_transport(transport)
        rel_dirs, rel_files = scan_remote(sftp, remote_dir, console)
        sftp.close()

        # Create local directory structure
        local_base.mkdir(parents=True, exist_ok=True)
        for d in rel_dirs:
            (local_base / d).mkdir(parents=True, exist_ok=True)

        # Download files in parallel
        errors = download_files(
            transport, remote_dir, local_base, rel_files, args.workers, console,
        )

        # Retry failed files
        if errors:
            console.print(
                f"\n{len(errors)} files failed, retrying...",
                style="bright_black",
            )
            retry_files = [rel for rel, _ in errors]
            errors2 = download_files(
                transport, remote_dir, local_base, retry_files, args.workers, console,
                label="Retrying",
            )
            if errors2:
                console.print(f"\n{len(errors2)} files still failed:", style="red")
                for rel, err in errors2:
                    console.print(f"  {rel}: {err}", style="red")
            else:
                console.print("All retries succeeded.", style="bright_black")
        else:
            console.print("Done.", style="bright_black")
    finally:
        client.close()
