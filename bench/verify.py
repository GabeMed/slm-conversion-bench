"""`bench verify`: every stored judgment the report or an arm's fact reads is what today's code and
configuration compute.

A judgment is a pure function over facts, and its result names what it read (`reads`: the
executions, the results of other judgments, the configuration keys; design §6.2). So a stored
result needs no trust: it is recomputed from its `reads`, with the code and the configuration of
the working tree, and the canonical bytes must be the stored ones. A result computed with another
parameter or another version of the code, or edited afterwards, does not come out the same.

Results are stored by content (`judgments/<J>/<sha256>/result.json`), so the comparison is of
addresses: the stored file must be the content its directory names, and the recomputation must
land on that same directory. The address of a recomputation is the sha256 of the bytes just
computed, never of a file found on disk.

What is verified:
- the judgments the report plan names (`bench/report.py`): `j5`, `j6`, `j7`, `j8`, `per_call`,
  `teacher_train_cost`, the `cost` of every arm and every `format` entry;
- the judgments behind the facts `config.yaml › arms` points at: a `choice` is written by J6,
  `centroids` by J5, an `allocation` by J7 (`adapters` is training's output, S5, not a judgment).
  A fact stands when **at least one** stored result that wrote it verifies, with everything that
  result built on: an older result that reached the same fact under other settings is not read by
  anything, and does not count against it. A fact with no such result is a divergence;
- every judgment one of those built on (a result named in `reads`), and so on.

J1 and J4 are not stored: the report computes them itself.

A divergence is one of: the stored file is not the content its directory names; its `reads` do not
name what the judgment needs; the judgment cannot run on them today (it refuses, or an execution
it read is gone); or it runs and gives other bytes. `bench verify` leaves `judgments/` as it found
it: what a divergent recomputation wrote is removed, and running the judgment again (`bench judge`)
is how to keep and compare it.
"""
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from bench import paths
from bench.contracts.config import ConfigError
from bench.contracts.facts import FactError, read_fact
from bench.data import DataError
from bench.judge.base import JudgmentError, read_json_judgment, read_result, relative

# the judgment that writes each fact, and where its result names the fact it wrote
FACT_WRITERS: Dict[str, Tuple[str, Callable[[Dict[str, Any]], Any]]] = {
    "choice": ("J6", lambda result: (result.get("choice_fact") or {}).get("sha256")),
    "centroids": ("J5", lambda result: (result.get("centroids") or {}).get("sha256")),
    "allocation": ("J7", lambda result: (result.get("allocation_fact") or {}).get("sha256")),
}
PLAN_KEYS = ("j5", "j6", "j7", "j8", "per_call", "teacher_train_cost")
REFUSALS = (JudgmentError, FactError, ConfigError, DataError, OSError)


def result_path(judgment: str, sha: str) -> Path:
    return Path(paths.ROOT) / "judgments" / judgment / sha / "result.json"


def _absolute(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(paths.ROOT) / path


def _fact_path(name: str, sha: str) -> str:
    """The file of a fact a result names by its sha256, whichever step wrote it."""
    found = sorted((Path(paths.ROOT) / "judgments").glob(f"*/{sha}/{name}.json"))
    if len(found) != 1:
        raise JudgmentError(f"expected one {name} fact {sha}, found {len(found)}")
    return str(found[0])


# ---------------------------------------------------------------- what is verified

def plan_results(plan: Dict[str, Any]) -> List[str]:
    """The judgment results a report plan names."""
    named = [plan[key] for key in PLAN_KEYS if plan.get(key)]
    named += [spec["cost"] for spec in (plan.get("arms") or {}).values() if spec.get("cost")]
    named += list((plan.get("format") or {}).values())
    return named


def fact_writers(config: Dict[str, Any]) -> Tuple[List[Tuple[str, str, List[Path]]], List[str]]:
    """([(the fact as the configuration names it, its judgment, the stored results that wrote it)],
    the facts that cannot be read) over the facts `arms` points at."""
    facts, divergences = [], []
    for arm, settings in sorted((config.get("arms") or {}).items()):
        for name, (judgment, written) in FACT_WRITERS.items():
            if not (settings or {}).get(name):
                continue
            label = f"arms.{arm}.{name} ({settings[name]})"
            try:
                _, sha = read_fact(str(_absolute(settings[name])), name)
            except (FactError, OSError) as e:
                divergences.append(f"{label}: {e}")
                continue
            writers = []
            for path in sorted((Path(paths.ROOT) / "judgments" / judgment).glob("*/result.json")):
                try:
                    if written(json.loads(path.read_bytes()).get("result") or {}) == sha:
                        writers.append(path)
                except (ValueError, AttributeError):  # not a result: it wrote nothing
                    continue
            facts.append((label, judgment, writers))
    return facts, divergences


def built_on(reads: Any) -> Iterator[Path]:
    """The results of other judgments a result's `reads` names (`base.result_reference`)."""
    if isinstance(reads, dict):
        if set(reads) == {"judgment", "sha256"}:
            yield result_path(reads["judgment"], reads["sha256"])
        else:
            for value in reads.values():
                yield from built_on(value)


# ---------------------------------------------------------------- recomputing one judgment from its reads

def _run_id(reads: Dict[str, Any], key: str, required: bool = True):
    return reads[key]["run_id"] if required or key in reads else None


def _built_on(reads: Dict[str, Any], key: str, required: bool = True):
    return str(result_path(reads[key]["judgment"], reads[key]["sha256"])) if required or key in reads else None


def _j2(reads, result, config):
    from bench.judge import j2
    if "run" in reads:
        return j2.run, (_run_id(reads, "run"),)
    return j2.run, (None, _run_id(reads, "replay"), _run_id(reads, "replay_eval", False), _run_id(reads, "teacher_eval", False))


def _j3(reads, result, config):
    from bench.judge import j3
    return j3.run, (_run_id(reads, "run"), _run_id(reads, "eval", False), _built_on(reads, "j8", False), config)


def _j5(reads, result, config):
    from bench.judge import j5
    return j5.run, (_run_id(reads, "curate"), _run_id(reads, "embed"), _run_id(reads, "embed_calib", False), config)


def _j6(reads, result, config):
    from bench.judge import j6
    candidates = [value for key, value in reads.items() if key != "config"]
    teacher_evals = {_run_id(candidate, "teacher_eval") for candidate in candidates}
    if len(teacher_evals) != 1:
        raise KeyError("teacher_eval")
    zeroshots = {_run_id(candidate, "replay"): _run_id(candidate, "replay_eval") for candidate in candidates}
    return j6.run, (zeroshots, teacher_evals.pop(), config)


def _j7(reads, result, config):
    from bench.judge import j7
    replays = {engine: (_run_id(reads[engine], "replay"), _run_id(reads[engine], "replay_eval")) for engine in ("cheap_alt", "slm")}
    self_replay = _run_id(reads, "teacher_self_replay")
    return (lambda *arguments: j7.run(*arguments, teacher_self_replay=self_replay)), (
        _fact_path("centroids", result["centroids"]), _fact_path("adapters", result["adapters"]), replays,
        _run_id(reads, "teacher_eval"), _built_on(reads, "j8"), _built_on(reads, "j6"), config)


def _j8(reads, result, config):
    from bench.judge import j8
    slo_from = _run_id(reads, "slo_from")
    return (lambda loadtests: j8.run(loadtests, config, slo_from=slo_from)), (sorted(reads["loadtests"]),)


# how each judgment's `run` is called again from what its result recorded
RECOMPUTE = {"J2": _j2, "J3": _j3, "J5": _j5, "J6": _j6, "J7": _j7, "J8": _j8}


def recompute(payload: Dict[str, Any], config: Dict[str, Any]) -> List[Path]:
    """The judgment run again on what the stored payload says it read: the files it returns, its
    result first (J5, J6 and J7 return their fact too)."""
    judgment = payload.get("judgment")
    if judgment not in RECOMPUTE:
        raise JudgmentError(f"{judgment} is not a judgment `bench verify` recomputes")
    try:
        run, arguments = RECOMPUTE[judgment](payload["reads"], payload["result"], config)
    except (KeyError, TypeError, AttributeError) as e:
        raise JudgmentError(f"its reads do not name what {judgment} needs to be recomputed ({e!r})") from e
    returned = run(*arguments)
    return [Path(path) for path in (returned if isinstance(returned, tuple) else (returned,))]


# ---------------------------------------------------------------- the check

def _label(path: Path) -> str:
    try:
        return relative(path)
    except ValueError:  # outside the repository
        return str(path)


def check(path: Path, config: Dict[str, Any]) -> Tuple[Optional[str], List[Path]]:
    """(why this stored result does not verify, or None; the results it built on)."""
    try:
        payload = read_result(path, read_json_judgment(path))
    except (OSError, ValueError, KeyError, TypeError) as e:  # missing, not a result, or not the content its directory names
        return str(e), []
    children = list(built_on(payload.get("reads")))
    directory = Path(paths.ROOT) / "judgments"
    before = set(directory.glob("*/*"))
    try:
        written = recompute(payload, config)
    except REFUSALS as e:
        return f"cannot be recomputed: {e}", children
    if written[0].resolve() == path.resolve():
        return None, children
    for file in written:  # leave judgments/ as it was: remove what this recomputation created
        if file.parent not in before and file.exists():
            file.unlink()
            if not any(file.parent.iterdir()):
                file.parent.rmdir()
    keys = ", ".join((payload.get("reads") or {}).get("config") or []) or "no key"
    return f"recomputes to {_label(written[0])}, not to the stored bytes (read {keys} of the configuration)", children


def verify(config: Dict[str, Any], plan: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(the results that verify, the divergences) over every result the plan or an arm's fact reads,
    and every result those built on."""
    reasons: Dict[Path, Optional[str]] = {}
    children: Dict[Path, List[Path]] = {}

    def closure(path: Path) -> Tuple[Dict[Path, str], List[Path]]:
        """(the results that do not verify, every result) among `path` and all it built on."""
        failed, visited, stack = {}, [], [path]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.append(current)
            if current not in reasons:
                reasons[current], children[current] = check(current, config)
            if reasons[current] is not None:
                failed[current] = reasons[current]
            stack += children[current]
        return failed, visited

    standing, listed = set(), {}
    for named in plan_results(plan):
        failed, visited = closure(_absolute(named))
        listed.update(failed)
        standing.update(set(visited) - set(failed))
    facts, divergences = fact_writers(config)
    for label, judgment, writers in facts:
        outcomes = [closure(writer) for writer in writers]
        if any(not failed for failed, _ in outcomes):
            for failed, visited in outcomes:
                if not failed:
                    standing.update(visited)
            continue
        why = "; ".join(f"{_label(path)}: {reason}" for failed, _ in outcomes for path, reason in sorted(failed.items()))
        divergences.append(f"{label}: no stored {judgment} result that wrote this fact verifies"
                           + (f" ({why})" if why else ""))
    divergences += [f"{_label(path)}: {reason}" for path, reason in sorted(listed.items())]
    return sorted(_label(path) for path in standing), divergences


def cli(args) -> int:
    import yaml
    from bench.contracts.config import load_config
    try:
        config = load_config(args.config)
        plan = yaml.safe_load(Path(args.plan).read_text())
        if not isinstance(plan, dict):
            raise ConfigError(f"{args.plan} is not a report plan (a mapping)")
    except (ConfigError, OSError, yaml.YAMLError) as e:
        print(f"bench verify: {e}", file=sys.stderr)
        return 2
    verified, divergences = verify(config, plan)
    for line in divergences:
        print(line)
    print(f"bench verify: {len(verified)} judgment(s) recomputed to the stored bytes, {len(divergences)} divergence(s)",
          file=sys.stderr)
    return 1 if divergences else 0
