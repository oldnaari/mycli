#!/usr/bin/env python3

import sys
import os


def main():
    if len(sys.argv) != 2:
        print("Usage: deepest-path <path>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    if path.startswith("~"):
        path = os.path.expanduser(path)

    parts = [p for p in path.split("/") if p]
    is_absolute = path.startswith("/")

    existing_parts = []
    missing_parts = []
    current = "/" if is_absolute else ""

    found_missing = False
    for part in parts:
        if found_missing:
            missing_parts.append(part)
        else:
            test_path = current.rstrip("/") + "/" + part
            if os.path.exists(test_path):
                current = test_path
                existing_parts.append(part)
            else:
                found_missing = True
                missing_parts.append(part)

    sep = "/" if is_absolute else ""
    existing_path = sep + "/".join(existing_parts)

    if not sys.stdout.isatty():
        print(existing_path)
        sys.exit(0)

    from rich.console import Console
    from rich.text import Text

    console = Console(highlight=False)
    out = Text()

    if not existing_parts:
        out.append(sep + "/".join(missing_parts), style="red")
    else:
        if missing_parts:
            first_missing = missing_parts[0]
            rest_missing = missing_parts[1:]

            out.append(existing_path + "/", style="bright_green")
            out.append(first_missing, style="bright_yellow")
            if rest_missing:
                out.append("/" + "/".join(rest_missing), style="red")
        else:
            out.append(existing_path, style="bright_green")

    console.print(out)
