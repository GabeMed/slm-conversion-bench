"""The test barrier (REQ-013): no execution touches the `test` split unless the pre-registration is
published **and** the execution uses exactly what was pre-registered.

The pre-registration (written by `bench prereg`) is two files:
- `prereg/manifest.json`: {"spec_sha256", "config_sha256", "splits_sha256", "data_manifest_sha256",
  "commit", ...}, the sha256 of SPEC.md, of the merged configuration (`config_sha256`), of
  data/splits.json and of data/MANIFEST.json, and the commit they were registered at;
- `prereg/HASH`: one line, the sha256 of `prereg/manifest.json`'s bytes.

Published means both files are committed with no local change and are identical in the commit
origin's `main` is at, as a fetch just brought it (never a remote-tracking ref, which may be stale);
the fetch times out and never prompts for credentials. On top of that, the execution's configuration, the splits, SPEC.md and the data
manifest must still hash to what was registered. Every command that executes something on a split,
and the split accessor itself, calls `ensure_split_allowed`.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from bench import paths

PREREG_HASH = "prereg/HASH"
PREREG_MANIFEST = "prereg/manifest.json"
REGISTERED = {"spec_sha256": "SPEC.md", "splits_sha256": "data/splits.json",
              "data_manifest_sha256": "data/MANIFEST.json"}


FETCH_TIMEOUT_S = 60  # a stalled network must not hold a command forever


class TestSplitLocked(RuntimeError):
    __test__ = False  # not a pytest test class


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch_origin_main(root: Path) -> Optional[str]:
    """The commit origin's main is at now: what this fetch brought (FETCH_HEAD). A remote-tracking ref
    is not used: a clone whose refspec does not map main would leave it stale. The fetch never prompts
    (a background run would wait forever) and times out."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    try:
        fetched = subprocess.run(["git", "-C", str(root), "fetch", "--quiet", "origin", "main"], capture_output=True,
                                 text=True, timeout=FETCH_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return None
    if fetched.returncode != 0:
        return None
    head = _git(root, "rev-parse", "--verify", "--quiet", "FETCH_HEAD^{commit}")
    return head.stdout.strip() if head.returncode == 0 else None


def _published(root: Path, rel: str, remote_commit: str) -> str:
    if not (root / rel).is_file():
        return f"{rel} does not exist: run `bench prereg` and push it before touching the test split"
    if _git(root, "ls-files", "--error-unmatch", rel).returncode != 0:
        return f"{rel} is not committed"
    if _git(root, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        return f"{rel} has uncommitted changes"
    remote = _git(root, "rev-parse", "--verify", "--quiet", f"{remote_commit}:{rel}")
    if remote.returncode != 0:
        return f"{rel} is not on origin/main: push it before touching the test split"
    if remote.stdout.strip() != _git(root, "hash-object", rel).stdout.strip():
        return f"{rel} differs from origin/main"
    return ""


def prereg_published(config: Optional[Dict[str, Any]] = None, root: Optional[Path] = None) -> str:
    """Empty string when the test split may be touched with `config`; otherwise the reason not."""
    root = root or paths.ROOT
    remote_commit = _fetch_origin_main(root)
    if remote_commit is None:
        return "could not fetch origin/main to confirm the pre-registration is published"
    for rel in (PREREG_MANIFEST, PREREG_HASH):
        reason = _published(root, rel, remote_commit)
        if reason:
            return reason
    if (root / PREREG_HASH).read_text().strip() != _sha256(root / PREREG_MANIFEST):
        return f"{PREREG_HASH} is not the sha256 of {PREREG_MANIFEST}"
    try:
        registered = json.loads((root / PREREG_MANIFEST).read_text())
    except ValueError:
        registered = None
    if not isinstance(registered, dict):
        return f"{PREREG_MANIFEST} is not a JSON object: register again with `bench prereg`"
    if config is None:
        return "an execution on the test split must present its configuration to the barrier"
    from bench.contracts.config import config_sha256
    if registered.get("config_sha256") != config_sha256(config):
        return "the configuration differs from the pre-registered one"
    for key, rel in REGISTERED.items():
        if registered.get(key) != _sha256(root / rel):
            return f"{rel} differs from the pre-registered one"
    return ""


def ensure_split_allowed(split: str, config: Optional[Dict[str, Any]] = None, root: Optional[Path] = None) -> None:
    if split != "test":
        return
    reason = prereg_published(config, root)
    if reason:
        raise TestSplitLocked(f"refusing to touch the test split: {reason}")
