"""Read-only report of every precondition of RUNBOOK step 12 (`bench prereg`) and of the test barrier
(`bench/barrier.py`), for a checkout given as an argument. It writes nothing anywhere: git runs with
optional locks off (no index refresh), origin is read with `git ls-remote` (no fetch, no FETCH_HEAD),
and no module of the checkout is imported (no bytecode written there).

    python scripts/prereg_readiness.py [<checkout>] [--config config.yaml]

Each line is OK, BLOCK (bench prereg or the first test run would refuse, or step 12 says to stop),
WARN (worth a look) or LATER (a step after `bench prereg`: the timestamp, the push). Exit code 1 when
any BLOCK is printed.
"""
import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
       "GIT_SSH_COMMAND": os.environ.get("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")}
ANALYSIS_CODE = ("bench/judge", "bench/evaluate.py", "bench/data.py", "bench/report.py", "bench/contracts",
                 "bench/paths.py")
CODE_ROOTS = ("bench", "vendor", "modal_apps", "runpod_apps")
results: List[Tuple[str, str, str]] = []


def report(level: str, what: str, detail: str = "") -> None:
    results.append((level, what, detail))


def git(root: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, env=ENV, timeout=timeout)


def ok(root: Path, *args: str) -> bool:
    return git(root, *args).returncode == 0


def arm_facts(root: Path) -> Dict[str, Tuple[str, ...]]:
    """ARM_FACTS of the checkout's router, read without importing it."""
    tree = ast.parse((root / "bench/contracts/router.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ARM_FACTS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("bench/contracts/router.py has no ARM_FACTS")


def nulls(node: Any, where: str = "") -> List[str]:
    if node is None:
        return [where]
    items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else []
    return [p for k, v in items for p in nulls(v, f"{where}.{k}" if where else str(k))]


def check_git_state(root: Path) -> Optional[str]:
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    head = git(root, "rev-parse", "HEAD").stdout.strip()
    report("OK" if branch == "main" else "BLOCK", "on branch main", f"HEAD is {branch} {head[:12]}")
    remote = git(root, "ls-remote", "origin", "refs/heads/main")
    remote_sha = remote.stdout.split()[0] if remote.returncode == 0 and remote.stdout.strip() else None
    if remote_sha is None:
        report("WARN", "origin/main readable", (remote.stderr.strip().splitlines() or ["no answer"])[-1])
        return None
    known = ok(root, "cat-file", "-e", f"{remote_sha}^{{commit}}")
    if not known:
        report("BLOCK", "origin/main is in this clone", f"origin/main is {remote_sha[:12]}, not fetched here: "
               f"pull before registering")
        return remote_sha
    behind = git(root, "rev-list", "--count", f"HEAD..{remote_sha}").stdout.strip()
    ahead = git(root, "rev-list", "--count", f"{remote_sha}..HEAD").stdout.strip()
    report("OK" if behind == "0" else "BLOCK", "HEAD contains origin/main",
           f"{behind} commit(s) behind origin/main {remote_sha[:12]}, {ahead} ahead")
    return remote_sha


def check_d3(root: Path) -> None:
    barrier = (root / "bench/barrier.py").read_text()
    prereg = (root / "bench/prereg.py").read_text()
    has = "def code_in_force" in barrier and "def local_excludes" in barrier and "execution_code" in prereg
    report("OK" if has else "BLOCK", "D3 is in the checkout (execution_code, clean-tree gate)",
           "" if has else "bench/barrier.py lacks code_in_force/local_excludes: merge PR #37 and pull first")


def check_tree(root: Path) -> None:
    gitignore = (root / ".gitignore").read_text().splitlines() if (root / ".gitignore").is_file() else []
    for pattern in ("runs/", ".claude/"):
        report("OK" if pattern in [l.strip() for l in gitignore] else "BLOCK", f".gitignore has {pattern}",
               "" if pattern in gitignore else "D3 adds .claude/; without it emptying .git/info/exclude shows .claude/")
    status = subprocess.run(["git", "-c", "core.excludesFile=/dev/null", "-C", str(root), "status", "--porcelain"],
                            capture_output=True, text=True, env=ENV)
    lines = [l for l in status.stdout.splitlines() if l.strip()]
    untracked_judgments = [l for l in lines if l[3:].startswith("judgments")]
    report("OK" if not lines else "BLOCK", "clean tree (git -c core.excludesFile=/dev/null status --porcelain)",
           "; ".join(lines[:8]) + (f" and {len(lines) - 8} more" if len(lines) > 8 else ""))
    if untracked_judgments:
        files = git(root, "ls-files", "--others", "--exclude-standard", "--", "judgments").stdout.split()
        report("BLOCK", "judgments/ committed", f"{len(files)} untracked file(s) under judgments/: "
               "`git add config.yaml judgments/ && git commit` (RUNBOOK step 12)")
    exclude = Path(git(root, "rev-parse", "--git-path", "info/exclude").stdout.strip())
    exclude = exclude if exclude.is_absolute() else root / exclude
    entries = [l.strip() for l in exclude.read_text().splitlines()
               if l.strip() and not l.strip().startswith("#")] if exclude.is_file() else []
    report("OK" if not entries else "BLOCK", ".git/info/exclude has no entry (shared by every worktree)",
           f"{exclude}: {', '.join(entries[:6])}" + (f" and {len(entries) - 6} more" if len(entries) > 6 else ""))
    hidden = [rel for rel in git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "--",
                                 *CODE_ROOTS).stdout.splitlines() if rel.endswith(".py")]
    report("OK" if not hidden else "BLOCK", "no ignored Python under the code", ", ".join(hidden[:5]))


def check_config(root: Path, config_rel: str) -> Optional[Dict[str, Any]]:
    path = root / config_rel
    raw = yaml.safe_load(path.read_text())
    if raw.get("extends"):
        report("WARN", "config has no extends", f"extends {raw['extends']}: every file of the chain must be tracked")
    tracked = ok(root, "ls-files", "--error-unmatch", "--", config_rel)
    report("OK" if tracked else "BLOCK", f"{config_rel} tracked")
    config = raw
    report("OK" if (config.get("cost") or {}).get("slo_from") else "BLOCK", "cost.slo_from set",
           str((config.get("cost") or {}).get("slo_from")))
    stats = config.get("stats") or {}
    good = isinstance(stats.get("n_boot"), int) and isinstance(stats.get("pilot_size"), int) and \
        isinstance(stats.get("pilot_mix"), dict) and sum(stats["pilot_mix"].values()) == stats["pilot_size"]
    report("OK" if good else "BLOCK", "stats.n_boot, pilot_size, pilot_mix consistent")
    arms = config.get("arms") or {}
    facts = arm_facts(root)
    for arm, names in sorted(facts.items()):
        for name in names:
            value = (arms.get(arm) or {}).get(name)
            if not value:
                report("BLOCK", f"arms.{arm}.{name} set", "null: " + ("J7's allocation fact, by a config PR"
                                                                         if name == "allocation" else "decide it"))
                continue
            fact = root / value
            if not fact.is_file():
                report("BLOCK", f"arms.{arm}.{name} exists", value)
            elif not ok(root, "ls-files", "--error-unmatch", "--", value):
                report("BLOCK", f"arms.{arm}.{name} tracked by git", f"{value} is untracked: commit judgments/")
            elif hashlib.sha256(fact.read_bytes()).hexdigest() != fact.parent.name:
                report("WARN", f"arms.{arm}.{name} stored by its sha256", f"{value}: directory name differs")
            else:
                report("OK", f"arms.{arm}.{name}", value)
    return config


def check_b1k(root: Path, config: Dict[str, Any]) -> None:
    found = sorted((root / "judgments/b1k").glob("*/choice.json"))
    if len(found) != 1:
        report("BLOCK", "exactly one b1k choice", f"{len(found)} found")
        return
    path = found[0]
    rel = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    payload = json.loads(raw)
    k = ((config.get("arms") or {}).get("B1") or {}).get("few_shot", {}).get("k")
    problems = []
    if hashlib.sha256(raw).hexdigest() != path.parent.name:
        problems.append("not stored by its sha256")
    if (payload.get("result") or {}).get("k") != k:
        problems.append(f"choice k={payload.get('result', {}).get('k')!r} but arms.B1.few_shot.k={k!r}")
    if not ok(root, "ls-files", "--error-unmatch", "--", rel):
        problems.append("untracked: commit judgments/")
    report("OK" if not problems else "BLOCK", "b1k: one choice, by sha256, k matches, tracked",
           f"{rel}" + (": " + "; ".join(problems) if problems else ""))
    # the rule and pilot ids are checked by bench prereg itself (bench/prereg.py b1k_choice)


def check_analysis_code(root: Path) -> None:
    stage0 = (root / "prereg/STAGE0").read_text().strip() if (root / "prereg/STAGE0").is_file() else None
    if not stage0:
        report("BLOCK", "prereg/STAGE0 present")
        return
    changed = git(root, "diff", "--name-only", stage0, "HEAD", "--", *ANALYSIS_CODE).stdout.split()
    deviations = (root / "prereg/DEVIATIONS.md").read_text() if (root / "prereg/DEVIATIONS.md").is_file() else ""
    named = [rel for rel in changed if rel in deviations or rel.split("/")[-1] in deviations]
    missing = [rel for rel in changed if rel not in named]
    report("OK" if not missing else "BLOCK", "analysis code changed since Stage 0 is named in prereg/DEVIATIONS.md",
           f"{len(changed)} changed; not named: {', '.join(missing)}" if missing else f"{len(changed)} changed, all named")
    report("WARN", "a file named once may have changed again later",
           "a name match is not proof each later change has its line: read `git log STAGE0..HEAD -- <file>`")


def check_after(root: Path, remote_sha: Optional[str]) -> None:
    report("OK" if shutil.which("ots") else "BLOCK", "`ots` (OpenTimestamps client) on PATH",
           "" if shutil.which("ots") else "pipx install opentimestamps-client (or uv tool install); free, "
                                           "talks to public calendar servers only")
    manifest, hashf, ots = (root / "prereg/manifest.json", root / "prereg/HASH", root / "prereg/HASH.ots")
    if not manifest.is_file():
        report("LATER", "bench prereg", "prereg/manifest.json not written yet")
        return
    intact = hashf.is_file() and hashf.read_text().strip() == hashlib.sha256(manifest.read_bytes()).hexdigest()
    report("OK" if intact else "BLOCK", "prereg/HASH is the sha256 of prereg/manifest.json")
    report("OK" if ots.is_file() else "LATER", "prereg/HASH.ots (third-party timestamp) present")
    if remote_sha and ok(root, "cat-file", "-e", f"{remote_sha}^{{commit}}"):
        on_origin = ok(root, "cat-file", "-e", f"{remote_sha}:prereg/HASH")
        same = on_origin and git(root, "rev-parse", f"{remote_sha}:prereg/HASH").stdout.strip() == \
            git(root, "hash-object", "prereg/HASH").stdout.strip()
        report("OK" if same else "LATER", "prereg/HASH published on origin/main", "" if same else "git push origin main")
    unpushed = git(root, "log", "--oneline", f"{remote_sha}..HEAD").stdout.strip().splitlines() if remote_sha else []
    report("OK" if not unpushed else "LATER", "nothing unpushed", "; ".join(unpushed[:5]))


def check_registry(root: Path) -> None:
    rel = "registry/call_sites.json"
    committed = ok(root, "ls-files", "--error-unmatch", "--", rel) and ok(root, "diff", "--quiet", "HEAD", "--", rel)
    report("OK" if committed else "BLOCK", f"{rel} committed, no local change (a test run reads it)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    remote_sha = check_git_state(root)
    check_d3(root)
    check_tree(root)
    config = check_config(root, args.config)
    if config is not None:
        check_b1k(root, config)
        left = [p for p in nulls(config) if not p.startswith("arms.")]
        report("WARN", "null values bench prereg will register as undecided (it prints, does not refuse)",
               ", ".join(left))
    check_analysis_code(root)
    check_registry(root)
    check_after(root, remote_sha)
    width = max(len(w) for _, w, _ in results)
    for level, what, detail in results:
        print(f"{level:5}  {what:{width}}  {detail}".rstrip())
    return 1 if any(level == "BLOCK" for level, _, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
