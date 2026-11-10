"""The record of every execution on the test split (REQ-013; design §1.2) and the set of call sites
observed on train and calib (REQ-001; SPEC 7.1).

Test executions, in git (the history is the record, no ledger file):
- an execution on `test` starts only when every earlier commit of `registry/` is on origin/main;
- before an execution on `test` starts, `registry/test/<run_id>.intent.json` is committed (run id,
  type, arm, the `prereg/HASH` it runs under, time, code commit), with its manifest as it starts
  (`status: running`), **and pushed to origin's main**; the
  execution starts only once a fetch finds that commit there, so the intent is public before the
  first question, whatever happens to the machine afterwards;
- when it ends, whatever its status, its `manifest.json` replaces that one at
  `registry/test/<run_id>.manifest.json`.
Only these two files are committed, never `calls.jsonl`, predictions or CHESS's outputs (the manifest
records their sha256 when the execution ends, `predictions_sha256` and `calls_sha256`, and why it did
not end `done`, `end_reason`). A run that dies before it ends (a killed process, a lost machine) leaves
its manifest at `running` (one started before that rule, an intent with no manifest). Nothing is removed: every attempt of a configuration stays listed
(`attempts_of`), and a later one either resumes the latest (its intent names it, `resumed_from`) or
states the defect that makes it start again (`redo_reason`).

Call sites: `registry/call_sites.json` is the union of the call sites seen by `done` runs on train
and calib (`bench call-sites <run_id>...` adds runs to it). An execution on `test` only starts when
the file is committed with no local change, and any call site outside it aborts the execution
(`hooks`), which its manifest flags.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench import barrier, paths
from bench.agent.hooks import HarnessError
from bench.contracts.calls import CALL_SITES
from bench.provenance import SECRET_MIN_LENGTH, redact  # noqa: F401  (redact: re-exported)

CALL_SITES_FILE = "registry/call_sites.json"
TEST_DIR = "registry/test"
SOURCE_SPLITS = ("train", "calib")
# Another git process (an editor, a `git status` in another shell, a merge in this checkout) holds the
# index or a ref for a moment: wait for it, a bounded number of times, rather than fail a finished run.
LOCK_WAITS_S = (0.5, 1, 2, 4, 8, 15)
_LOCKED = ("index.lock", "cannot lock ref")


class RegistryError(HarnessError):
    """The registry could not be read or written: a failure of the harness."""




def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(paths.ROOT), *args], capture_output=True, text=True)


def _git_waiting_for_locks(*args: str) -> subprocess.CompletedProcess:
    """`_git`, tried again while another git process holds the index or a ref (`LOCK_WAITS_S`, about
    30 s in all); any other failure, or a lock that outlasts the waits, is returned as it is."""
    for wait in (*LOCK_WAITS_S, None):
        result = _git(*args)
        if result.returncode == 0 or wait is None or not any(m in result.stderr for m in _LOCKED):
            return result
        time.sleep(wait)
    return result


def _commit(rel: str, message: str) -> None:
    """Commit exactly one file, leaving anything else staged or changed as it was."""
    _commit_all([rel], message)


def _commit_all(rels: List[str], message: str) -> None:
    """Commit exactly these files, in one commit, leaving anything else staged or changed as it was."""
    for args in (("add", "--", *rels), ("commit", "-q", "-m", message, "--only", "--", *rels)):
        result = _git_waiting_for_locks(*args)
        if result.returncode != 0:
            raise RegistryError(f"git {args[0]} {' '.join(rels)} failed: {result.stderr.strip() or result.stdout.strip()}")


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


def registry_pushed() -> str:
    """Every earlier registry commit is on origin's main as a fetch finds it now (the barrier's
    `fetch_origin_main`, never a remote-tracking ref that may be stale): a record that lives only in a
    local clone is no record. Returns the commit origin's main is at."""
    remote_commit, why_not = barrier.fetch_origin_main(paths.ROOT)
    if remote_commit is None:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {why_not}: the registry cannot be "
                                      f"confirmed as pushed")
    result = _git("log", "--oneline", f"{remote_commit}..HEAD", "--", "registry/")
    if result.returncode != 0:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: cannot compare registry/ with origin/main "
                                      f"({result.stderr.strip()})")
    unpushed = result.stdout.strip().splitlines()
    if unpushed:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {len(unpushed)} commit(s) of registry/ are "
                                      f"not on origin/main: push the registry first")
    return remote_commit


def registered_call_sites() -> Dict[str, Any]:
    """The committed set, for an execution on test: {"call_sites": [...], "sha256": ..., "origin": the
    commit origin's main is at, the published registry}."""
    path = paths.ROOT / CALL_SITES_FILE
    if not path.is_file():
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {CALL_SITES_FILE} does not exist "
                                      f"(`bench call-sites` on the train and calib runs, then commit it)")
    if _git("ls-files", "--error-unmatch", CALL_SITES_FILE).returncode != 0:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {CALL_SITES_FILE} is not committed")
    if _git("diff", "--quiet", "HEAD", "--", CALL_SITES_FILE).returncode != 0:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {CALL_SITES_FILE} has uncommitted changes")
    origin = registry_pushed()
    call_sites = json.loads(path.read_text()).get("call_sites") or []
    if not call_sites:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {CALL_SITES_FILE} registers no call site")
    return {"call_sites": call_sites, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "origin": origin}


def done_runs_of(run_type: str, arm: Optional[str], engine: Optional[str], prereg_hash: str, ref: str) -> List[str]:
    """The test executions the registry at `ref` records as `done` for this configuration, keyed as the
    report keys one (type, arm, engine, pre-registration): more than one is "several runs" there, and that
    arm's comparisons get no verdict (SPEC 6.1)."""
    listed = _git("ls-tree", "-r", "--name-only", ref, TEST_DIR)
    if listed.returncode != 0:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: cannot list {TEST_DIR} at {ref} "
                                      f"({listed.stderr.strip()})")
    found = []
    for rel in listed.stdout.split():
        if not rel.endswith(".manifest.json"):
            continue
        shown = _git("show", f"{ref}:{rel}")
        if shown.returncode != 0:
            raise barrier.TestSplitLocked(f"refusing to touch the test split: cannot read {rel} at {ref}")
        ended = json.loads(shown.stdout)
        if ended.get("status") == "done" and (ended.get("type"), ended.get("arm"), ended.get("engine"),
                                              ended.get("prereg_hash")) == (run_type, arm, engine, prereg_hash):
            found.append(ended.get("run_id") or Path(rel).name.rsplit(".", 2)[0])
    return sorted(found)


def _show_json(ref: str, rel: str) -> Optional[Dict[str, Any]]:
    shown = _git("show", f"{ref}:{rel}")
    if shown.returncode != 0:
        return None
    try:
        value = json.loads(shown.stdout)
    except ValueError:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: {rel} at {ref} is not JSON") from None
    return value if isinstance(value, dict) else None


def published_manifest(run_id: str, ref: str) -> Optional[Dict[str, Any]]:
    """The manifest the registry at `ref` holds for a test execution, or None when it holds none."""
    return _show_json(ref, f"{TEST_DIR}/{run_id}.manifest.json")


def attempts_of(run_type: str, arm: Optional[str], engine: Optional[str], prereg_hash: str, ref: str) -> List[Dict[str, Any]]:
    """Every test execution of this configuration the registry at `ref` records, keyed as `done_runs_of`
    keys one: [{"run_id", "started_at", "status"}] by start, `status` being its manifest's, or None for
    an intent with no manifest (a run that died before it ended, started before its manifest was
    committed with the intent). A killed session's status reads `running`. Nothing is ever removed from the
    registry: each attempt stays listed."""
    listed = _git("ls-tree", "-r", "--name-only", ref, TEST_DIR)
    if listed.returncode != 0:
        raise barrier.TestSplitLocked(f"refusing to touch the test split: cannot list {TEST_DIR} at {ref} "
                                      f"({listed.stderr.strip()})")
    found = []
    for rel in listed.stdout.split():
        if not rel.endswith(".intent.json"):
            continue
        intent = _show_json(ref, rel)
        if intent is None:
            raise barrier.TestSplitLocked(f"refusing to touch the test split: cannot read {rel} at {ref}")
        if (intent.get("type"), intent.get("arm"), intent.get("engine"), intent.get("prereg_hash")) != \
                (run_type, arm, engine, prereg_hash):
            continue
        run_id = intent.get("run_id") or Path(rel).name.rsplit(".", 2)[0]
        ended = published_manifest(run_id, ref)
        found.append({"run_id": run_id, "started_at": intent.get("started_at"),
                      "status": None if ended is None else ended.get("status")})
    return sorted(found, key=lambda a: (a["started_at"] or "", a["run_id"]))


# ---------------------------------------------------------------- executions on test

INTENT_FIELDS = ("run_id", "type", "arm", "engine", "split", "source_run_id", "prereg_hash", "commit", "started_at")


def commit_intent(manifest: Dict[str, Any], config: Dict[str, Any], **extra: Any) -> str:
    """Commit the intent, with the manifest as it starts (`status: running`), and publish both: nothing
    runs on `test` until origin's main shows them. A session killed outright leaves that manifest in the
    registry, so it is resumable as any stopped one (`resume.recover_killed`). `extra` (a resume's source,
    a redo's reason, the earlier attempts, a killed source's outputs) is published with the intent, before
    the first question. Nothing in the intent comes from a model or a provider: no redaction needed; the
    manifest is redacted as it always is."""
    rel = f"{TEST_DIR}/{manifest['run_id']}.intent.json"
    _write(rel, {**{key: manifest.get(key) for key in INTENT_FIELDS}, **extra})
    started = f"{TEST_DIR}/{manifest['run_id']}.manifest.json"
    (paths.ROOT / started).write_text(redact(json.dumps(manifest, indent=2, ensure_ascii=False), config) + "\n")
    _commit_all([rel, started], f"registry: intent of test execution {manifest['run_id']}")
    _publish(_git("rev-parse", "HEAD").stdout.strip(), manifest["run_id"])
    return rel


def _publish(commit: str, run_id: str) -> None:
    """Push `commit` (the intent, with whatever of HEAD origin's main lacks) to origin's main, never
    forced, and confirm it by what a fetch finds there (`barrier.fetch_origin_main`): the push's own
    answer is not the evidence. Like the fetch, the push never prompts and times out. On any failure
    the execution is refused; the intent stays committed here, the record of an attempt, to push by hand."""
    refusal = (f"refusing to run {run_id} on the test split: its intent is committed but not published "
               f"(%s): nothing was run; `git push origin HEAD:main`, then start the execution again")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    try:
        pushed = subprocess.run(["git", "-C", str(paths.ROOT), "push", "--quiet", "origin", f"{commit}:refs/heads/main"],
                                capture_output=True, text=True, timeout=barrier.FETCH_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        raise barrier.TestSplitLocked(refusal % f"the push timed out after {barrier.FETCH_TIMEOUT_S} s") from None
    if pushed.returncode != 0:
        detail = (pushed.stderr.strip().splitlines() or ["no detail"])[-1]
        raise barrier.TestSplitLocked(refusal % f"the push to origin/main failed: {detail}")
    remote_commit, why_not = barrier.fetch_origin_main(paths.ROOT)
    if remote_commit is None:
        raise barrier.TestSplitLocked(refusal % why_not)
    if _git("merge-base", "--is-ancestor", commit, remote_commit).returncode != 0:
        raise barrier.TestSplitLocked(refusal % "origin/main does not contain it after the push")


def commit_manifest(run_dir: Path, config: Dict[str, Any]) -> str:
    rel = f"{TEST_DIR}/{run_dir.name}.manifest.json"
    (paths.ROOT / rel).write_text(redact((run_dir / "manifest.json").read_text(), config))
    _commit(rel, f"registry: manifest of test execution {run_dir.name}")
    return rel
