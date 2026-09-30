"""What every execution records about the code that produced it."""
import subprocess
from pathlib import Path
from typing import Any, Dict

from bench import paths


def git_state(root: Path = paths.ROOT) -> Dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
