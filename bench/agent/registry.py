"""The record of every execution on the test split (REQ-013; design §1.2) and the set of call sites
observed on train and calib (REQ-001; SPEC 7.1).

Test executions, in git (the history is the record, no ledger file):
- an execution on `test` starts only when every earlier commit of `registry/` is on origin/main;
- before an execution on `test` starts, `registry/test/<run_id>.intent.json` is committed (run id,
  type, arm, the `prereg/HASH` it runs under, time, code commit);
- when it ends, whatever its status, its `manifest.json` is committed as
  `registry/test/<run_id>.manifest.json`.
Only these two files are committed, never `calls.jsonl`, predictions or CHESS's outputs. A run that
dies before it ends (a killed process, a lost machine) leaves an intent with no manifest.

Call sites: `registry/call_sites.json` is the union of the call sites seen by `done` runs on train
and calib (`bench call-sites <run_id>...` adds runs to it). An execution on `test` only starts when
the file is committed with no local change, and any call site outside it aborts the execution
(`hooks`), which its manifest flags.
"""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from bench import barrier, paths
from bench.agent.hooks import HarnessError
from bench.contracts.calls import CALL_SITES

CALL_SITES_FILE = "registry/call_sites.json"
TEST_DIR = "registry/test"
SOURCE_SPLITS = ("train", "calib")


class RegistryError(HarnessError):
    """The registry could not be read or written: a failure of the harness."""


SECRET_MIN_LENGTH = 8  # a key is longer; shorter values are placeholders ("x", "1", "EMPTY"), and
# replacing every "1" of a record would corrupt it
_KEY_LIKE = re.compile(r"\b(sk|hf)[-_][A-Za-z0-9_\-*.]{6,}")


def redact(text: str, config: Dict[str, Any]) -> str:
    """What is committed to a public repository never carries a credential: the values of every
    credential variable of the configuration (API keys and `headers_env` values, C2's
    `credential_envs`), and anything shaped like a key (provider error bodies sometimes echo a
    masked one)."""
    from bench.contracts.config import credential_envs
    values = {os.environ.get(name) for name in credential_envs(config)}
    for value in sorted((v for v in values if v and len(v) >= SECRET_MIN_LENGTH), key=len, reverse=True):
        text = text.replace(value, "<redacted>")  # the longest first: one value may contain another
    return _KEY_LIKE.sub("<redacted>", text)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(paths.ROOT), *args], capture_output=True, text=True)


def _commit(rel: str, message: str) -> None:
    """Commit exactly one file, leaving anything else staged or changed as it was."""
    for args in (("add", "--", rel), ("commit", "-q", "-m", message, "--only", "--", rel)):
        result = _git(*args)
        if result.returncode != 0:
            raise RegistryError(f"git {args[0]} {rel} failed: {result.stderr.strip() or result.stdout.strip()}")


def _write(rel: str, value: Any) -> None:
    path = paths.ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- the call sites seen on train and calib

def update_call_sites(run_ids: List[str]) -> Path:
    """Add the call sites of these runs (done, on train or calib) to registry/call_sites.json."""
    path = paths.ROOT / CALL_SITES_FILE
    registry = json.loads(path.read_text()) if path.exists() else {"call_sites": [], "sources": {}}
    for run_id in run_ids:
        manifest_path = paths.RUNS / run_id / "manifest.json"
        if not manifest_path.exists():
            raise RegistryError(f"no run {run_id}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("split") not in SOURCE_SPLITS:
            raise RegistryError(f"{run_id} ran on {manifest.get('split')!r}: only train and calib runs register call sites")
        if manifest.get("status") != "done":
            raise RegistryError(f"{run_id} is {manifest.get('status')!r}, not 'done'")
        seen = manifest.get("call_sites_seen")
        if not isinstance(seen, list) or set(seen) - set(CALL_SITES):
            raise RegistryError(f"{run_id}: the manifest has no valid call_sites_seen")
        registry["sources"][run_id] = {"type": manifest["type"], "arm": manifest.get("arm"),
                                       "split": manifest["split"], "call_sites_seen": sorted(seen)}
    registry["call_sites"] = sorted({site for source in registry["sources"].values() for site in source["call_sites_seen"]})
    _write(CALL_SITES_FILE, registry)
    return path


def check_keys(key_envs: List[str]) -> None:
    """Every credential an execution on test uses (API keys and header values) is long enough to be redacted from its public record: a
    shorter one would stay verbatim wherever a provider echoes it (redacting it would corrupt the
    record instead: every "1" of it for a key "1")."""
    short = sorted({name for name in key_envs if name and 0 < len(os.environ.get(name, "")) < SECRET_MIN_LENGTH})
    if short:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {', '.join(short)} is shorter than "
                                      f"{SECRET_MIN_LENGTH} characters, too short to be redacted from the public "
                                      f"record without corrupting it")


def registry_pushed() -> None:
    """Every earlier registry commit is on origin/main (checked after the barrier's fetch): a record
    that lives only in a local clone is no record."""
    result = _git("log", "--oneline", "origin/main..HEAD", "--", "registry/")
    if result.returncode != 0:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: cannot compare registry/ with origin/main "
                                      f"({result.stderr.strip()})")
    unpushed = result.stdout.strip().splitlines()
    if unpushed:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {len(unpushed)} commit(s) of registry/ are "
                                      f"not on origin/main: push the registry first")


def registered_call_sites() -> Dict[str, Any]:
    """The committed set, for an execution on test: {"call_sites": [...], "sha256": ...}."""
    path = paths.ROOT / CALL_SITES_FILE
    if not path.is_file():
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {CALL_SITES_FILE} does not exist "
                                      f"(`bench call-sites` on the train and calib runs, then commit it)")
    if _git("ls-files", "--error-unmatch", CALL_SITES_FILE).returncode != 0:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {CALL_SITES_FILE} is not committed")
    if _git("diff", "--quiet", "HEAD", "--", CALL_SITES_FILE).returncode != 0:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {CALL_SITES_FILE} has uncommitted changes")
    registry_pushed()
    call_sites = json.loads(path.read_text()).get("call_sites") or []
    if not call_sites:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {CALL_SITES_FILE} registers no call site")
    return {"call_sites": call_sites, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# ---------------------------------------------------------------- executions on test

def prereg_hash() -> str:
    """The `prereg/HASH` an execution on test runs under (the barrier has already checked it)."""
    return (paths.ROOT / barrier.PREREG_HASH).read_text().strip()


def commit_intent(manifest: Dict[str, Any]) -> str:
    """Nothing in the intent comes from a model or a provider: no redaction needed."""
    rel = f"{TEST_DIR}/{manifest['run_id']}.intent.json"
    _write(rel, {key: manifest.get(key) for key in
                 ("run_id", "type", "arm", "engine", "split", "source_run_id", "prereg_hash", "commit", "started_at")})
    _commit(rel, f"registry: intent of test execution {manifest['run_id']}")
    return rel


def commit_manifest(run_dir: Path, config: Dict[str, Any]) -> str:
    rel = f"{TEST_DIR}/{run_dir.name}.manifest.json"
    (paths.ROOT / rel).write_text(redact((run_dir / "manifest.json").read_text(), config))
    _commit(rel, f"registry: manifest of test execution {run_dir.name}")
    return rel
