#!/usr/bin/env python3
"""One command-line entry point for the preserved scGPT/scGenePT workflows."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the preserved attention workflow for the selected model family. "
            "Remaining arguments are passed unchanged to its backend."
        ),
        epilog=(
            "scGPT paths: --model-dir, --data-dir, --data-file, "
            "--mouse2human-pkl, --output-base. scGenePT paths: --scgenept_repo, "
            "--models_dir, --run_dir, --checkpoint, --output_dir. "
            "Use explicit absolute paths; see README.md for examples. "
            "The launcher preserves the current working directory and uses "
            "the active Python environment. --help and --dry-run do not import models."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model-family", required=True, choices=("scgpt", "scgenept"),
        help="Select the model-specific loading and extraction workflow.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the shell-quoted command without running the backend.",
    )
    args, backend_args = parser.parse_known_args(argv)
    if backend_args[:1] == ["--"]:
        backend_args = backend_args[1:]
    backend = Path(__file__).resolve().parent / "_backends" / (
        args.model_family + "_attention.py"
    )
    if not backend.is_file():
        parser.error("Missing attention backend: " + str(backend))
    command = [sys.executable, str(backend), *backend_args]
    if args.dry_run:
        print(shlex.join(command))
        return 0
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as exc:
        parser.error("Could not launch attention backend: " + str(exc))


if __name__ == "__main__":
    sys.exit(main())
