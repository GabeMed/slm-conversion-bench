"""What every execution records about the code that produced it, and nothing about the machine."""
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from bench import paths


def git_state(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or paths.ROOT

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def scrub(text: str) -> str:
    """Remove the origin from free text that reaches a record (SPEC S1: anonymize with respect to
    origin): the repository path and the home directory become placeholders."""
    return text.replace(str(paths.ROOT), "<repo>").replace(str(Path.home()), "~")
