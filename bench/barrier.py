"""The test barrier (REQ-013): no execution touches the `test` split unless the pre-registration is
published **and** the execution uses exactly what was pre-registered.

The pre-registration (written by `bench prereg`) is two files:
- `prereg/manifest.json`: {"spec_sha256", "config_sha256", "splits_sha256", "data_manifest_sha256",
  "commit", ...}, the sha256 of SPEC.md, of the merged configuration (`config_sha256`), of
  data/splits.json and of data/MANIFEST.json, and the commit they were registered at;
- `prereg/HASH`: one line, the sha256 of `prereg/manifest.json`'s bytes.

Published means both files are committed with no local change and are identical on `origin/main`
after a fetch. On top of that, the execution's configuration, the splits, SPEC.md and the data
manifest must still hash to what was registered. Every command that executes something on a split,
and the split accessor itself, calls `ensure_split_allowed`.
"""
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from bench import paths

PREREG_HASH = "prereg/HASH"
PREREG_MANIFEST = "prereg/manifest.json"
REGISTERED = {"spec_sha256": "SPEC.md", "splits_sha256": "data/splits.json",
              "data_manifest_sha256": "data/MANIFEST.json"}


class TestSplitLocked(RuntimeError):
    __test__ = False  # not a pytest test class


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _published(root: Path, rel: str) -> str:
    if not (root / rel).is_file():
        return f"{rel} does not exist: run `bench prereg` and push it before touching the test split"
    if _git(root, "ls-files", "--error-unmatch", rel).returncode != 0:
        return f"{rel} is not committed"
    if _git(root, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        return f"{rel} has uncommitted changes"
    remote = _git(root, "rev-parse", "--verify", "--quiet", f"origin/main:{rel}")
    if remote.returncode != 0:
        return f"{rel} is not on origin/main: push it before touching the test split"
    if remote.stdout.strip() != _git(root, "hash-object", rel).stdout.strip():
        return f"{rel} differs from origin/main"
    return ""


def prereg_published(config: Optional[Dict[str, Any]] = None, root: Optional[Path] = None) -> str:
    """Empty string when the test split may be touched with `config`; otherwise the reason not."""
    root = root or paths.ROOT
    if _git(root, "fetch", "--quiet", "origin", "main").returncode != 0:
        return "could not fetch origin/main to confirm the pre-registration is published"
    for rel in (PREREG_MANIFEST, PREREG_HASH):
        reason = _published(root, rel)
        if reason:
            return reason
    if (root / PREREG_HASH).read_text().strip() != _sha256(root / PREREG_MANIFEST):
        return f"{PREREG_HASH} is not the sha256 of {PREREG_MANIFEST}"
    registered = json.loads((root / PREREG_MANIFEST).read_text())
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
