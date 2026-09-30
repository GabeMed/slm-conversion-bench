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
from bench.contracts.config import config_sha256
from bench.provenance import git_state

# String literals and quoted identifiers come first, so nothing inside them is ever rewritten.
_QUOTED = r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`|\[[^\]]*\]"
_NOW = re.compile(_QUOTED + r"|\bCURRENT_(?:TIMESTAMP|DATE|TIME)\b"
                  r"|\b(?:date|time|datetime|julianday|unixepoch)\s*\(\s*\)"  # no argument: SQLite reads 'now'
                  r"|\bstrftime\s*\(\s*'(?:[^']|'')*'\s*\)", re.IGNORECASE)  # a format only: 'now' as well
_LIMIT = re.compile(_QUOTED + r"|\bLIMIT\b", re.IGNORECASE)


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
        if token[0] in "'\"`[":
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
    return any(m.group(0)[0] not in "'\"`[" for m in _LIMIT.finditer(sql))


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


def evaluate(source_run_id: str) -> Path:
    source = paths.RUNS / source_run_id
    run_manifest = json.loads((source / "manifest.json").read_text())
    config = json.loads((source / "config.json").read_text())  # the run's own snapshot, not today's config.yaml
    if config_sha256(config) != run_manifest["config_sha256"]:  # validated when the run loaded it
        raise data.DataError(f"the configuration snapshot of {source_run_id} does not match its manifest")
    split = run_manifest["split"]
    barrier.ensure_split_allowed(split, config)
    if run_manifest["status"] != "done":
        raise data.DataError(f"run {source_run_id} is {run_manifest['status']!r}, not 'done': "
                             f"an incomplete or failed run is never scored")
    predictions: Dict[str, Optional[str]] = json.loads((source / "predictions.json").read_text())
    if set(predictions) != set(run_manifest["question_ids"]):
        raise data.DataError(f"the predictions of {source_run_id} are not exactly its question ids")
    gold = data.questions_for(config, split)
    unpaired = sorted(set(predictions) - set(gold), key=int)
    if unpaired:
        raise data.DataError(f"predictions without a gold in the {split} split: {unpaired[:10]}")
    databases = sorted({gold[q]["db_id"] for q in predictions})
    for db_id in databases:
        data.check_database(config, db_id)
    day = fixed_date(config)

    started = datetime.now(timezone.utc)
    run_id = f"eval-{source_run_id}-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
    out = paths.RUNS / run_id
    out.mkdir(parents=True)
    results = score(predictions, gold, lambda db_id: paths.sqlite_path(config, db_id),
                    config["eval"]["timeout_s"], day)
    with open(out / "results.jsonl", "w") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    correct = sum(r["correct"] for r in results)
    manifest = {
        "run_id": run_id, "type": "eval", "source_run_id": source_run_id, "split": split,
        **git_state(), "source_commit": run_manifest["commit"], "config_sha256": run_manifest["config_sha256"],
        "databases": {db: json.loads(paths.DATA_MANIFEST.read_text())["databases"][db]["sqlite"] for db in databases},
        "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        "n": len(results), "correct": correct, "gold_errors": sum(r["gold_error"] is not None for r in results),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


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
