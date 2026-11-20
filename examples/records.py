"""Provenance and atomic JSON writes for the measurement records in notes/."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def revision() -> dict:
    """The git and jj commits measured, and whether the working copy had changes."""
    root = Path(__file__).resolve().parents[1]

    def read(*args):
        try:
            return subprocess.check_output(
                args, cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = read("git", "status", "--porcelain")
    return {
        "git_commit": read("git", "rev-parse", "HEAD"),
        "jj_commit": read(
            "jj", "log", "--ignore-working-copy", "-r", "@", "--no-graph", "-T", "commit_id"
        ),
        "working_copy_dirty": None if status is None else bool(status),
    }


def write_json(path: Path, result: dict) -> None:
    """Write `result` to `path` atomically, rejecting NaN and infinity."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
