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

An execution on the test split (`bench run`, `bench replay`) also calls `ensure_execution_allowed`: the
code that runs is the code that was registered. The working tree is clean (`git status --porcelain`
lists nothing: no modified and no untracked file; what .gitignore ignores does not count, see
`code_in_force`), HEAD descends from the commit the registration recorded and differs from it only in
the records that change by design after it (`RECORDS`: prereg/ and registry/), and the execution code
(`EXECUTION_CODE`: the agent, CHESS, the serving plan, the curation) hashes, file by file, to what the
registration recorded under `execution_code`.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bench import paths

PREREG_HASH = "prereg/HASH"
PREREG_MANIFEST = "prereg/manifest.json"
# what the registration hashes besides the configuration, relative to the root, as bench.paths lays it out
REGISTERED = {"spec_sha256": "SPEC.md", "splits_sha256": paths.SPLITS.relative_to(paths.ROOT).as_posix(),
              "data_manifest_sha256": paths.DATA_MANIFEST.relative_to(paths.ROOT).as_posix()}


# The code that executes on the test split, beyond the analysis code the registration already hashes
# (bench/prereg.py ANALYSIS_CODE): the agent and its harness, the patched CHESS it drives, the serving
# plan of the SLM (what is served, from which revision, with which adapters) and the curation that
# wrote the adapters' datasets. Every tracked file under these paths, by sha256.
EXECUTION_CODE = ("bench/agent", "bench/curate.py", "vendor/chess", "modal_apps/common.py", "modal_apps/serve_vllm.py",
                  "modal_apps/deploy.yaml", "runpod_apps")
# What changes after the registration by design: the registration itself (its timestamp, a deviation
# line) and the registry of test executions, which every test execution commits to.
RECORDS = ("prereg", "registry")
# Where Python that runs lives (the execution and the analysis code): an ignored .py file there is refused.
CODE_ROOTS = ("bench", "vendor", "modal_apps", "runpod_apps")


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


def tracked_code(root: Path, prefixes: Tuple[str, ...]) -> Dict[str, str]:
    """The sha256 of every tracked file under `prefixes`, by path relative to the root."""
    listed = _git(root, "ls-files", "--", *prefixes)
    if listed.returncode != 0:
        raise TestSplitLocked(f"refusing to touch the test split: cannot list the code ({listed.stderr.strip()})")
    return {rel: _sha256(root / rel) for rel in sorted(listed.stdout.splitlines())}


def execution_code(root: Optional[Path] = None) -> Dict[str, str]:
    """The execution code's hashes, as the registration records them (`execution_code`)."""
    return tracked_code(root or paths.ROOT, EXECUTION_CODE)


def ignored_python(root: Path, prefixes: Tuple[str, ...]) -> List[str]:
    """Python files under `prefixes` that .gitignore hides: they would run while no commit holds them, and
    a clean `git status` does not show them."""
    listed = _git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "--", *prefixes)
    return sorted(rel for rel in listed.stdout.splitlines() if rel.endswith(".py"))


def _outside_records() -> List[str]:
    return ["--", ".", *(f":(exclude){rel}" for rel in RECORDS)]


def same_code(commit: str, other: str, root: Optional[Path] = None) -> bool:
    """Whether two commits hold the same tree outside RECORDS (both must exist here)."""
    root = root or paths.ROOT
    return _git(root, "diff", "--quiet", "--no-renames", commit, other, *_outside_records()).returncode == 0


def code_in_force(registered: Dict[str, Any], root: Optional[Path] = None) -> str:
    """Empty string when the checkout is the registered code; otherwise the reason not.

    Dirty is what `git status --porcelain` lists: a modified, staged, deleted or untracked file. What
    .gitignore ignores does not count: runs/ is what executions write (one test run would make the next
    dirty), data/raw and data/bird_dev are inputs data/MANIFEST.json hashes and every run checks
    (`data.check_database`), train/adapters and train/datasets are served or read by the facts the
    configuration names, by hash, and reports/ is regenerated. judgments/ is not ignored: a judgment the
    registered configuration names is committed before the registration (RUNBOOK step 12), so an
    untracked one at test time is a fact no registered commit holds, and counts. A Python file .gitignore
    hides under the execution or analysis code would run unseen: refused too."""
    from bench.provenance import git_state
    root = root or paths.ROOT
    state = git_state(root)
    if state["dirty"]:
        shown = ", ".join(f"{e['status'].strip()} {e['path']}" for e in state["dirty_paths"][:5])
        more = state["dirty_paths_total"] - min(5, len(state["dirty_paths"]))
        return (f"the working tree is not clean ({shown}{f' and {more} more' if more > 0 else ''}): a test execution "
                f"runs only the registered code, committed")
    hidden = ignored_python(root, CODE_ROOTS)
    if hidden:
        return f"Python code ignored by git would run: {', '.join(hidden[:5])}"
    commit = registered.get("commit")
    if not isinstance(commit, str) or _git(root, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        return f"the registered commit {commit!r} is not in this repository"
    if _git(root, "merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        return f"HEAD does not descend from the registered commit {commit[:12]}"
    changed = _git(root, "diff", "--name-only", "--no-renames", commit, "HEAD", *_outside_records()).stdout.split()
    if changed:
        return (f"HEAD differs from the registered commit {commit[:12]} outside {', '.join(RECORDS)}: "
                f"{', '.join(changed[:5])}" + (f" and {len(changed) - 5} more" if len(changed) > 5 else ""))
    recorded = registered.get("execution_code")
    if not isinstance(recorded, dict):
        return "the registration records no execution code (`execution_code`): register again with a `bench prereg` that does"
    now = execution_code(root)
    differ = sorted(rel for rel in set(recorded) | set(now) if recorded.get(rel) != now.get(rel))
    if differ:
        return (f"the execution code differs from the registered one: {', '.join(differ[:5])}"
                + (f" and {len(differ) - 5} more" if len(differ) > 5 else ""))
    return ""


def ensure_execution_allowed(config: Dict[str, Any], root: Optional[Path] = None) -> str:
    """The barrier for an execution on the test split (`bench run`, `bench replay`): the registration
    in force (`prereg_hash_in_force`), and the checkout is exactly the registered code (`code_in_force`).
    Returns the registration's HASH."""
    root = root or paths.ROOT
    in_force = prereg_hash_in_force(config, root)
    reason = code_in_force(read_registered(root) or {}, root)
    if reason:
        raise TestSplitLocked(f"refusing to touch the test split: {reason}")
    return in_force


def registered_commit(root: Optional[Path] = None) -> Optional[str]:
    commit = (read_registered(root or paths.ROOT) or {}).get("commit")
    return commit if isinstance(commit, str) else None
