"""`bench verify`: every stored judgment the report or an arm's fact reads is what today's code and
configuration compute.

A judgment is a pure function over facts, and its result names what it read (`reads`: the
executions, the results of other judgments, the configuration keys; design §6.2). So a stored
result needs no trust: it is recomputed from its `reads`, with the code and the configuration of
the working tree, and the canonical bytes are compared with the stored ones. A result computed with
another parameter or another version of the code, or edited afterwards, does not come out the same.

What is verified:
- the judgments the report plan names (`bench/report.py`): `j5`, `j6`, `j7`, `j8`, `per_call`,
  `teacher_train_cost`, the `cost` of every arm and every `format` entry;
- the judgments behind the facts `config.yaml › arms` points at: every stored J6 result that wrote
  that `choice`, J5 that `centroids`, J7 that `allocation` (a fact with no stored result behind it
  is a divergence). `adapters` is training's output (S5), not a judgment;
- every judgment one of those built on (a result named in `reads`), and so on.

J1 and J4 are not stored: the report computes them itself.

A divergence is one of: the stored file is not the content its directory names; its `reads` do not
name what the judgment needs; the judgment refuses to run on them today; or it runs and gives other
bytes. In the last case the recomputed result is written beside the stored one, as any judgment's
is, and its path is listed, so the two can be compared.
"""
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

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
REFUSALS = (JudgmentError, FactError, ConfigError, DataError)


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


def fact_results(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(the stored results that wrote the facts `arms` points at, the facts with none)."""
    results, divergences = [], []
    for arm, settings in sorted((config.get("arms") or {}).items()):
        for name, (judgment, written) in FACT_WRITERS.items():
            if not (settings or {}).get(name):
                continue
            try:
                _, sha = read_fact(str(_absolute(settings[name])), name)
            except (FactError, OSError) as e:
                divergences.append(f"arms.{arm}.{name} ({settings[name]}): {e}")
                continue
            writers = [str(path) for path in sorted((Path(paths.ROOT) / "judgments" / judgment).glob("*/result.json"))
                       if written(json.loads(path.read_bytes()).get("result") or {}) == sha]
            if not writers:
                divergences.append(f"arms.{arm}.{name} ({settings[name]}): no stored {judgment} result wrote this fact")
            results += writers
    return results, divergences


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
    return (lambda *args: j5.run(*args)[0]), (_run_id(reads, "curate"), _run_id(reads, "embed"),
                                               _run_id(reads, "embed_calib", False), config)


def _j6(reads, result, config):
    from bench.judge import j6
    candidates = [value for key, value in reads.items() if key != "config"]
    teacher_evals = {_run_id(candidate, "teacher_eval") for candidate in candidates}
    if len(teacher_evals) != 1:
        raise KeyError("teacher_eval")
    zeroshots = {_run_id(candidate, "replay"): _run_id(candidate, "replay_eval") for candidate in candidates}
    return (lambda *args: j6.run(*args)[0]), (zeroshots, teacher_evals.pop(), config)


def _j7(reads, result, config):
    from bench.judge import j7
    replays = {engine: (_run_id(reads[engine], "replay"), _run_id(reads[engine], "replay_eval")) for engine in ("cheap_alt", "slm")}
    return (lambda *args: j7.run(*args)[0]), (
        _fact_path("centroids", result["centroids"]), _fact_path("adapters", result["adapters"]), replays,
        _run_id(reads, "teacher_eval"), _built_on(reads, "j8"), _built_on(reads, "j6"), config)


def _j8(reads, result, config):
    from bench.judge import j8
    slo_from = _run_id(reads, "slo_from")
    return (lambda loadtests: j8.run(loadtests, config, slo_from=slo_from)), (sorted(reads["loadtests"]),)


# how each judgment's `run` is called again from what its result recorded
RECOMPUTE = {"J2": _j2, "J3": _j3, "J5": _j5, "J6": _j6, "J7": _j7, "J8": _j8}


def recompute(payload: Dict[str, Any], config: Dict[str, Any]) -> Path:
    """The result of running the judgment again on what the stored payload says it read."""
    judgment = payload["judgment"]
    if judgment not in RECOMPUTE:
        raise JudgmentError(f"{judgment} is not a judgment `bench verify` recomputes")
    try:
        run, arguments = RECOMPUTE[judgment](payload["reads"], payload["result"], config)
    except (KeyError, TypeError, AttributeError) as e:
        raise JudgmentError(f"its reads do not name what {judgment} needs to be recomputed ({e!r})") from e
    return Path(run(*arguments))


# ---------------------------------------------------------------- the check

def verify(config: Dict[str, Any], plan: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(the results verified, the divergences): every result the plan or an arm's fact reads, and
    every one those built on, recomputed and compared byte by byte."""
    from_facts, divergences = fact_results(config)
    queue = [_absolute(path) for path in plan_results(plan) + from_facts]
    seen, verified = set(), []
    while queue:
        path = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        try:
            label = relative(path)
        except ValueError:
            label = str(path)
        try:
            stored = path.read_bytes()
            payload = read_result(path, read_json_judgment(path))
        except (OSError, ValueError, KeyError) as e:  # missing, not JSON, or not the content its directory names
            divergences.append(f"{label}: {e}")
            continue
        queue += list(built_on(payload["reads"]))
        try:
            recomputed = recompute(payload, config)
        except REFUSALS as e:
            divergences.append(f"{label}: cannot be recomputed: {e}")
            continue
        if recomputed.read_bytes() != stored:
            divergences.append(f"{label}: recomputed as {relative(recomputed)} "
                               f"(read {', '.join(payload['reads'].get('config') or []) or 'no key'} of the configuration)")
        else:
            verified.append(label)
    return verified, divergences


def cli(args) -> int:
    import yaml
    from bench.contracts.config import load_config
    try:
        config = load_config(args.config)
        plan = yaml.safe_load(Path(args.plan).read_text())
    except (ConfigError, OSError, yaml.YAMLError) as e:
        print(f"bench verify: {e}", file=sys.stderr)
        return 2
    verified, divergences = verify(config, plan)
    for line in divergences:
        print(line)
    print(f"bench verify: {len(verified)} judgment(s) recomputed to the stored bytes, {len(divergences)} divergence(s)",
          file=sys.stderr)
    return 1 if divergences else 0
