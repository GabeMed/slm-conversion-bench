"""`bench eval <run_id>`: the only source of execution accuracy (J1 reads what this writes).

Each prediction is paired with its gold by `question_id`, never by position. Gold and prediction
run back to back on the pinned SQLite file (sha256 checked against data/MANIFEST.json), read-only
and with a timeout. Correct means the same set of rows, BIRD's rule.
"""
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from bench import barrier, data, paths
from bench.contracts.config import config_sha256, load_config
from bench.provenance import git_state


def execute(db_path: Path, sql: str, timeout_s: float) -> Tuple[Optional[List[tuple]], Optional[str]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    deadline = time.monotonic() + timeout_s
    connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        return connection.execute(sql).fetchall(), None
    except Exception as e:
        timed_out = time.monotonic() > deadline
        return None, "timeout" if timed_out else f"{type(e).__name__}: {e}"
    finally:
        connection.close()


def score(predictions: Dict[str, Optional[str]], gold: Dict[str, dict], db_path: Callable[[str], Path],
          timeout_s: float) -> List[Dict[str, Any]]:
    """One result per prediction, each paired with the gold of the same question_id."""
    results: List[Dict[str, Any]] = []
    for question_id in sorted(predictions, key=int):
        question = gold[question_id]
        path = db_path(question["db_id"])
        predicted = predictions[question_id]
        executed_at = datetime.now(timezone.utc).isoformat()
        gold_rows, gold_error = execute(path, question["SQL"], timeout_s)
        if predicted is None:
            pred_rows, pred_error = None, "no prediction"
        else:
            pred_rows, pred_error = execute(path, predicted, timeout_s)
        results.append({
            "question_id": question_id, "db_id": question["db_id"],
            "difficulty": question.get("difficulty"),
            "predicted_sql": predicted, "gold_sql": question["SQL"],
            "pred_error": pred_error, "gold_error": gold_error,
            "correct": gold_error is None and pred_error is None and set(pred_rows) == set(gold_rows),
            "executed_at": executed_at,
        })
    return results


def evaluate(source_run_id: str) -> Path:
    source = paths.RUNS / source_run_id
    run_manifest = json.loads((source / "manifest.json").read_text())
    config = load_config(paths.ROOT / run_manifest["config_path"])
    if config_sha256(config) != run_manifest["config_sha256"]:
        raise data.DataError(f"{run_manifest['config_path']} changed since run {source_run_id}")
    split = run_manifest["split"]
    barrier.ensure_split_allowed(split)
    predictions: Dict[str, Optional[str]] = json.loads((source / "predictions.json").read_text())
    gold = data.questions_for(config, split)
    unpaired = sorted(set(predictions) - set(gold), key=int)
    if unpaired:
        raise data.DataError(f"predictions without a gold in the {split} split: {unpaired[:10]}")
    databases = sorted({gold[q]["db_id"] for q in predictions})
    for db_id in databases:
        data.check_database(config, db_id)

    started = datetime.now(timezone.utc)
    run_id = f"eval-{source_run_id}-{started.strftime('%Y%m%dT%H%M%SZ')}"
    out = paths.RUNS / run_id
    out.mkdir(parents=True)
    results = score(predictions, gold, lambda db_id: paths.sqlite_path(config, db_id), config["eval"]["timeout_s"])
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
