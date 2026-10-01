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
from typing import Any, Dict, Optional, Tuple

from bench import paths

PREREG_HASH = "prereg/HASH"
PREREG_MANIFEST = "prereg/manifest.json"
# what the registration hashes besides the configuration, relative to the root, as bench.paths lays it out
REGISTERED = {"spec_sha256": "SPEC.md", "splits_sha256": paths.SPLITS.relative_to(paths.ROOT).as_posix(),
              "data_manifest_sha256": paths.DATA_MANIFEST.relative_to(paths.ROOT).as_posix()}


FETCH_TIMEOUT_S = 60  # a stalled network must not hold a command forever


class TestSplitLocked(RuntimeError):
    __test__ = False  # not a pytest test class


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_origin_main(root: Path) -> Tuple[Optional[str], str]:
    """(the commit origin's main branch is at now, "") or (None, why not): what this fetch brought
    (FETCH_HEAD), never a remote-tracking ref, which a clone whose refspec does not map main leaves stale.
    A failed fetch refuses even when an earlier one left a FETCH_HEAD. The branch is named in full (a tag
    called main would win over a short name). The fetch never prompts (a background run would wait
    forever) and times out. What every check against the published state reads: the barrier, and the
    registry's check that its commits are pushed."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    try:
        fetched = subprocess.run(["git", "-C", str(root), "fetch", "--quiet", "origin", "refs/heads/main"],
                                 capture_output=True, text=True, timeout=FETCH_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return None, f"the fetch of origin/main timed out after {FETCH_TIMEOUT_S} s"
    if fetched.returncode != 0:
        detail = (fetched.stderr.strip().splitlines() or ["no detail"])[-1]
        return None, f"could not fetch origin/main ({detail})"
    head = _git(root, "rev-parse", "--verify", "--quiet", "FETCH_HEAD^{commit}")
    if head.returncode != 0 or not head.stdout.strip():  # an empty FETCH_HEAD would read as the index
        return None, "the fetch of origin/main left no commit to compare with"
    return head.stdout.strip(), ""


def read_registered(root: Path) -> Optional[Dict[str, Any]]:
    """prereg/manifest.json as a mapping, or None when it is not one (unreadable, not JSON, not an object):
    the one reader of the registration, for the barrier, eval, the report and `bench prereg`."""
    try:
        registered = json.loads((root / PREREG_MANIFEST).read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return registered if isinstance(registered, dict) else None


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
    remote_commit, why_not = fetch_origin_main(root)
    if remote_commit is None:
        return f"{why_not}: the pre-registration cannot be confirmed as published"
    for rel in (PREREG_MANIFEST, PREREG_HASH):
        reason = _published(root, rel, remote_commit)
        if reason:
            return reason
    if (root / PREREG_HASH).read_bytes().strip() != _sha256(root / PREREG_MANIFEST).encode():
        return f"{PREREG_HASH} is not the sha256 of {PREREG_MANIFEST}"
    registered = read_registered(root)
    if registered is None:
        return f"{PREREG_MANIFEST} is not a JSON object: register again with `bench prereg --replace`"
    if config is None:
        return "an execution on the test split must present its configuration to the barrier"
    from bench.contracts.config import config_sha256
    if registered.get("config_sha256") != config_sha256(config):
        return "the configuration differs from the pre-registered one"
    for key, rel in REGISTERED.items():
        if registered.get(key) != _sha256(root / rel):
            return f"{rel} differs from the pre-registered one"
    return ""


def prereg_hash_in_force(config: Dict[str, Any], root: Optional[Path] = None) -> str:
    """The pre-registration in force for touching or reading the test with `config`: its HASH, once the
    barrier has found it published, intact, and matching the configuration, the splits, SPEC.md and the
    data manifest. The one definition every consumer of the test uses (the runner, eval, the report)."""
    root = root or paths.ROOT
    ensure_split_allowed("test", config, root)
    return (root / PREREG_HASH).read_text().strip()


def ensure_split_allowed(split: str, config: Optional[Dict[str, Any]] = None, root: Optional[Path] = None) -> None:
    if split != "test":
        return
    reason = prereg_published(config, root)
    if reason:
        raise TestSplitLocked(f"refusing to touch the test split: {reason}")
