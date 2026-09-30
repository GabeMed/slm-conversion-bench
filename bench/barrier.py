"""The test barrier (REQ-013): no execution touches the `test` split before the pre-registration is published.

Published means: `prereg/HASH` exists, is committed with no local change, and the same content is
on `origin/main` as this clone last saw it (the remote-tracking ref a push updates). Every
command that executes something on a split calls `ensure_split_allowed` before reading it.
"""
import subprocess
from pathlib import Path
from typing import Optional

from bench import paths

PREREG_HASH = "prereg/HASH"


class TestSplitLocked(RuntimeError):
    __test__ = False  # not a pytest test class


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def prereg_published(root: Optional[Path] = None) -> str:
    """Empty string when published; otherwise the reason it is not."""
    root = root or paths.ROOT
    rel = PREREG_HASH
    if not (root / rel).is_file():
        return f"{rel} does not exist: run `bench prereg` and push it before touching the test split"
    if _git(root, "ls-files", "--error-unmatch", rel).returncode != 0:
        return f"{rel} is not committed"
    if _git(root, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        return f"{rel} has uncommitted changes"
    remote = _git(root, "rev-parse", "--verify", "--quiet", f"origin/main:{rel}")
    if remote.returncode != 0:
        return f"{rel} is not on origin/main: push it before touching the test split"
    local = _git(root, "hash-object", rel)
    if remote.stdout.strip() != local.stdout.strip():
        return f"{rel} differs from origin/main"
    return ""


def ensure_split_allowed(split: str, root: Optional[Path] = None) -> None:
    if split != "test":
        return
    reason = prereg_published(root)
    if reason:
        raise TestSplitLocked(f"refusing to touch the test split: {reason}")
