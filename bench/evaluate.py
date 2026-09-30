"""`bench eval <run_id>`: the only source of execution accuracy (J1 reads what this writes).

Each prediction is paired with its gold by `question_id`, never by position. Gold and prediction
run back to back on the pinned SQLite file (sha256 checked against data/MANIFEST.json), read-only
and with a timeout. Correct means the same set of rows, BIRD's rule.

The current moment is the pre-registered `eval.fixed_date` (midnight UTC), in prediction and gold
alike, so arms evaluated on different days, on machines in different time zones, stay comparable.
"""
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from bench import barrier, data, paths
from bench.contracts.calls import read_calls, validate_calls
from bench.contracts.config import config_sha256
from bench.provenance import git_state

# Comments, string literals and quoted identifiers come first, so nothing inside them is rewritten
# (and an apostrophe in a comment opens no literal).
_STRING = r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\""
_QUOTED = r"--[^\n]*|/\*.*?(?:\*/|$)|" + _STRING + r"|`[^`]*`|\[[^\]]*\]"
_NOW = re.compile(_QUOTED + r"|\bCURRENT_(?:TIMESTAMP|DATE|TIME)\b"
                  r"|\b(?:date|time|datetime|julianday|unixepoch)\s*\(\s*\)"  # no argument: SQLite reads 'now'
                  r"|\bstrftime\s*\(\s*(?:" + _STRING + r")\s*\)",  # a format only: 'now' as well
                  re.IGNORECASE | re.DOTALL)
_LIMIT = re.compile(_QUOTED + r"|\bLIMIT\b", re.IGNORECASE | re.DOTALL)
_SKIPPED = ("'", '"', "`", "[", "--", "/*")


def fixed_date(config: Dict[str, Any]) -> str:
    """The pre-registered date, refused when unset: without it, EX would depend on the day of the eval."""
    value = config["eval"].get("fixed_date")
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        raise data.DataError(f"eval.fixed_date must be a pre-registered YYYY-MM-DD string, not {value!r}")


def fix_date(sql: str, day: str) -> Tuple[str, bool]:
    """`sql` with every reading of the current moment replaced by `day` at midnight, and whether
    anything was replaced: 'now' (any case, either quote), CURRENT_TIMESTAMP / CURRENT_DATE /
    CURRENT_TIME, and the date functions SQLite evaluates at 'now' when given no time value."""
    stamp = f"'{day} 00:00:00'"
    keywords = {"current_timestamp": stamp, "current_date": f"'{day}'", "current_time": "'00:00:00'"}
    replaced = False

    def replace(match: "re.Match[str]") -> str:
        nonlocal replaced
        token = match.group(0)
        if token.startswith(_SKIPPED):
            if token[0] in "'\"" and token[1:-1].lower() == "now":
                replaced = True
                return stamp
            return token
        replaced = True
        if token.lower() in keywords:
            return keywords[token.lower()]
        return f"{token[:-1].rstrip()}, {stamp})" if token.lower().startswith("strftime") else f"{token.split('(')[0]}({stamp})"
    return _NOW.sub(replace, sql), replaced


def gold_has_limit(sql: str) -> bool:
    """LIMIT outside string literals: ties at the cut can make a right answer look wrong, for every arm."""
    return any(not m.group(0).startswith(_SKIPPED) for m in _LIMIT.finditer(sql))


@contextmanager
def _utc() -> Iterator[None]:
    """SQLite's 'localtime' follows the process time zone; the evaluator runs in UTC."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = previous
        time.tzset()


def execute(db_path: Path, sql: str, timeout_s: float) -> Tuple[Optional[List[tuple]], Optional[str]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    deadline = time.monotonic() + timeout_s
    connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        with _utc():
            return connection.execute(sql).fetchall(), None
    except Exception as e:
        timed_out = time.monotonic() > deadline
        return None, "timeout" if timed_out else f"{type(e).__name__}: {e}"
    finally:
        connection.close()


def score_one(question: Dict[str, Any], predicted: Optional[str], path: Path, timeout_s: float,
              day: str) -> Dict[str, Any]:
    """One prediction against the gold of its own question, both at the fixed date."""
    executed_at = datetime.now(timezone.utc).isoformat()
    gold_sql, gold_substituted = fix_date(question["SQL"], day)
    gold_rows, gold_error = execute(path, gold_sql, timeout_s)
    pred_substituted = False
    if predicted is None:
        pred_rows, pred_error = None, "no prediction"
    else:
        pred_sql, pred_substituted = fix_date(predicted, day)
        pred_rows, pred_error = execute(path, pred_sql, timeout_s)
    return {
        "question_id": question["question_id"], "db_id": question["db_id"],
        "difficulty": question.get("difficulty"),
        "predicted_sql": predicted, "gold_sql": question["SQL"],
        "pred_error": pred_error, "gold_error": gold_error,
        "correct": gold_error is None and pred_error is None and set(pred_rows) == set(gold_rows),
        "pred_date_substituted": pred_substituted, "gold_date_substituted": gold_substituted,
        "gold_has_limit": gold_has_limit(question["SQL"]),
        "executed_at": executed_at,
    }


def score(predictions: Dict[str, Optional[str]], gold: Dict[str, dict], db_path: Callable[[str], Path],
          timeout_s: float, day: str) -> List[Dict[str, Any]]:
    """One result per prediction, each paired with the gold of the same question_id."""
    return [score_one({**gold[q], "question_id": q}, predictions[q], db_path(gold[q]["db_id"]), timeout_s, day)
            for q in sorted(predictions, key=int)]


def check_golds(questions: Dict[str, List[dict]], db_path: Callable[[str], Path], timeout_s: float,
                day: str) -> Dict[str, Any]:
    """Execute every gold of every split as the evaluator will (fixed date, read-only, timeout);
    the ones that fail are a fact of the data, recorded by `bench data` in data/MANIFEST.json."""
    failures = []
    for split, items in questions.items():
        for question in items:
            _, error = execute(db_path(question["db_id"]), fix_date(question["SQL"], day)[0], timeout_s)
            if error is not None:
                failures.append({"split": split, "question_id": question["question_id"],
                                 "db_id": question["db_id"], "error": error})
    return {"fixed_date": day, "timeout_s": timeout_s, "sqlite_version": sqlite3.sqlite_version,
            "checked": {split: len(items) for split, items in questions.items()}, "failures": failures}


def _source(source_run_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, dict]]:
    """The run to score, its configuration snapshot and the gold of its split, after every check
    both kinds of eval share: the snapshot is the one the run hashed, the barrier opens for that
    snapshot, a test run ran under the registration in force, and the run is `done`."""
    source = paths.RUNS / source_run_id
    run_manifest = json.loads((source / "manifest.json").read_text())
    config = json.loads((source / "config.json").read_text())  # the run's own snapshot, not today's config.yaml
    if config_sha256(config) != run_manifest["config_sha256"]:  # validated when the run loaded it
        raise data.DataError(f"the configuration snapshot of {source_run_id} does not match its manifest")
    split = run_manifest["split"]
    barrier.ensure_split_allowed(split, config)
    if split == "test":
        in_force = (paths.ROOT / barrier.PREREG_HASH).read_text().strip()
        if run_manifest.get("prereg_hash") != in_force:
            raise data.DataError(f"run {source_run_id} ran under pre-registration {run_manifest.get('prereg_hash')}, "
                                 f"not the one in force ({in_force}): it is reported, never scored")
    if run_manifest["status"] != "done":
        raise data.DataError(f"run {source_run_id} is {run_manifest['status']!r}, not 'done': "
                             f"an incomplete or failed run is never scored")
    return run_manifest, config, data.questions_for(config, split)


def _check_paired(question_ids: List[str], gold: Dict[str, dict], config: Dict[str, Any], split: str) -> List[str]:
    """Every scored question has a gold in the split and a database with the recorded hash."""
    unpaired = sorted(set(question_ids) - set(gold), key=int)
    if unpaired:
        raise data.DataError(f"predictions without a gold in the {split} split: {unpaired[:10]}")
    databases = sorted({gold[q]["db_id"] for q in question_ids})
    for db_id in databases:
        data.check_database(config, db_id)
    return databases


def _write(kind: str, source_run_id: str, run_manifest: Dict[str, Any], config: Dict[str, Any],
           databases: List[str], score_all: Callable[[str], List[Dict[str, Any]]], **extra: Any) -> Path:
    day = fixed_date(config)  # refused before the execution directory exists
    started = datetime.now(timezone.utc)
    run_id = f"{kind}-{source_run_id}-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
    out = paths.RUNS / run_id
    out.mkdir(parents=True)
    results = score_all(day)
    with open(out / "results.jsonl", "w") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    recorded = json.loads(paths.DATA_MANIFEST.read_text())["databases"]
    manifest = {
        "run_id": run_id, "type": "eval", **extra, "source_run_id": source_run_id,
        "source_type": run_manifest.get("type"), "arm": run_manifest.get("arm"), "engine": run_manifest.get("engine"),
        "mode": run_manifest.get("mode"), "split": run_manifest["split"],
        **git_state(), "source_commit": run_manifest["commit"], "config_sha256": run_manifest["config_sha256"],
        "prereg_hash": run_manifest.get("prereg_hash"),
        "fixed_date": day, "timeout_s": config["eval"]["timeout_s"], "sqlite_version": sqlite3.sqlite_version,
        "databases": {db: recorded[db]["sqlite"] for db in databases},
        "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        "n": len(results), "correct": sum(r["correct"] for r in results),
        "gold_errors": sum(r["gold_error"] is not None for r in results),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def evaluate(source_run_id: str) -> Path:
    """The eval execution of a run's final SQL per question (predictions.json)."""
    run_manifest, config, gold = _source(source_run_id)
    predictions: Dict[str, Optional[str]] = json.loads((paths.RUNS / source_run_id / "predictions.json").read_text())
    if set(predictions) != set(run_manifest["question_ids"]):
        raise data.DataError(f"the predictions of {source_run_id} are not exactly its question ids")
    databases = _check_paired(list(predictions), gold, config, run_manifest["split"])
    return _write("eval", source_run_id, run_manifest, config, databases, lambda day: score(
        predictions, gold, lambda db_id: paths.sqlite_path(config, db_id), config["eval"]["timeout_s"], day))


# The call sites whose output is a SQL query, and where their parser puts it (CHESS parsers.py).
PER_CALL_SQL = {"generate_candidate": "SQL", "revise": "refined_sql_query"}


def per_call_predictions(calls: List[dict]) -> List[Dict[str, Any]]:
    """One entry per generation or repair invocation, sorted by question, call site and key: the SQL
    of its last attempt that parsed, or None when no attempt did."""
    invocations: Dict[Tuple[str, str, str], Optional[dict]] = {}
    for record in calls:
        if record["call_site"] not in PER_CALL_SQL:
            continue
        key = (record["question_id"], record["call_site"], record["invocation_key"])
        best = invocations.setdefault(key, None)
        if record["parsed_ok"] and (best is None or record["attempt"] > best["attempt"]):
            invocations[key] = record
    entries = []
    for (question_id, call_site, invocation_key), record in sorted(
            invocations.items(), key=lambda item: (int(item[0][0]), item[0][1], item[0][2])):
        output = record["parsed_output"] if record else None
        sql = output.get(PER_CALL_SQL[call_site]) if isinstance(output, dict) else None
        entries.append({"question_id": question_id, "call_site": call_site, "invocation_key": invocation_key,
                        "call_id": record["call_id"] if record else None,
                        "attempt": record["attempt"] if record else None,
                        "sql": sql if isinstance(sql, str) else None})
    return entries


def evaluate_per_call(source_run_id: str) -> Path:
    """The eval execution of every generation and repair invocation in a run's calls.jsonl (an
    `agent` or a `replay` run), each against the gold of its question (design §5.1)."""
    run_manifest, config, gold = _source(source_run_id)
    calls = read_calls(paths.RUNS / source_run_id / "calls.jsonl")
    errors = validate_calls(calls)
    if errors:
        raise data.DataError(f"the calls.jsonl of {source_run_id} is not valid C1: {errors[:3]}")
    entries = per_call_predictions(calls)
    in_run = run_manifest.get("question_ids")
    if in_run is None:
        raise data.DataError(f"the manifest of {source_run_id} does not list its question ids")
    if not {e["question_id"] for e in entries} <= set(in_run):
        raise data.DataError(f"calls of {source_run_id} for questions outside its question ids")
    databases = _check_paired([e["question_id"] for e in entries], gold, config, run_manifest["split"])

    def score_calls(day: str) -> List[Dict[str, Any]]:
        results = []
        for entry in entries:
            question = {**gold[entry["question_id"]], "question_id": entry["question_id"]}
            result = score_one(question, entry["sql"], paths.sqlite_path(config, question["db_id"]),
                               config["eval"]["timeout_s"], day)
            if entry["call_id"] is None:
                result["pred_error"] = "no attempt parsed"
            results.append({"question_id": entry["question_id"], "call_site": entry["call_site"],
                            "invocation_key": entry["invocation_key"], "correct": result["correct"],
                            "call_id": entry["call_id"], "attempt": entry["attempt"],
                            **{k: v for k, v in result.items() if k not in ("question_id", "correct")}})
        return results
    return _write("eval-per-call", source_run_id, run_manifest, config, databases, score_calls, per_call=True)
