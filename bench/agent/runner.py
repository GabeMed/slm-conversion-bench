"""`bench run` and `bench preprocess`: executions of the agent (agent environment only).

An `agent` run writes, under runs/<run_id>/:
- manifest.json: arm, split, question ids, commit, configuration hash, database hashes, status;
- calls.jsonl: one C1 line per LLM invocation;
- predictions.json: {question_id: final SQL or null}, by CHESS's final-SQL rule;
- chess/: CHESS's own outputs, kept for inspection and without authority.
"""
import json
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench import barrier, data, paths
from bench.provenance import git_state
from bench.contracts.calls import read_calls, validate_calls
from bench.contracts.config import config_sha256, engine_spec, load_config


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _prepare_chess(config: Dict[str, Any]):
    """Point CHESS at the pinned data and templates, and hand it the configuration."""
    os.environ["DB_ROOT_PATH"] = str(paths.bird_root(config))
    os.environ.setdefault("INDEX_SERVER_PORT", "0")  # read at import by CHESS; unused by IR -> SS -> CG
    os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Chroma would otherwise send usage telemetry
    src = str(paths.VENDOR_CHESS / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from bench.agent import hooks
    hooks.configure(config)
    import llm.prompts
    llm.prompts.TEMPLATES_ROOT_PATH = str(paths.VENDOR_CHESS / "templates")
    return hooks


def final_sql(state: Any) -> Optional[str]:
    """CHESS's final SQL (workflow/agents/evaluation.py): the first SQL under the last key of SQL_meta_infos."""
    infos = state.SQL_meta_infos
    if not infos:
        return None
    last = infos[list(infos)[-1]]
    if not last:
        return None
    first = last[0]
    return first["SQL"] if isinstance(first, dict) else first.SQL


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run_agent(config_path: str, arm: str, split: str, ids: Optional[List[str]] = None,
              limit: Optional[int] = None) -> Path:
    config = load_config(config_path)
    barrier.ensure_split_allowed(split)
    questions = data.questions_for(config, split)
    if ids:
        outside = [q for q in ids if q not in questions]
        if outside:
            raise data.DataError(f"ids not in the {split} split: {outside}")
        selected = list(ids)
    else:
        selected = sorted(questions, key=int)[:limit] if limit else sorted(questions, key=int)
    databases = sorted({questions[q]["db_id"] for q in selected})
    for db_id in databases:
        data.check_database(config, db_id)

    started = _now()
    run_id = f"agent-{arm}-{split}-{started.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = paths.RUNS / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id, "type": "agent", "arm": arm, "split": split, "question_ids": selected,
        **git_state(), "config_path": str(Path(config_path).resolve().relative_to(paths.ROOT)),
        "config_sha256": config_sha256(config),
        "splits_sha256": data.sha256_file(paths.SPLITS),
        "data_manifest_sha256": data.sha256_file(paths.DATA_MANIFEST),
        "databases": databases,
        "engines": {role: {k: engine_spec(config, role)[k] for k in ("model", "endpoint")}
                    for role in ("production_llm", "cheap_alt")},
        "embeddings": config["embeddings"], "allocation": None,
        "started_at": started.isoformat(), "finished_at": None, "status": "running",
    }
    _write_json(run_dir / "manifest.json", manifest)

    hooks = _prepare_chess(config)
    hooks.start_run(run_id, arm, run_dir / "calls.jsonl")
    from runner.run_manager import RunManager
    RunManager.RESULT_ROOT_PATH = str(run_dir / "chess")
    dataset = [{**questions[q], "question_id": int(q)} for q in selected]
    _write_json(run_dir / "questions.json", dataset)
    manager = RunManager(Namespace(
        data_mode="dev", data_path=str(run_dir / "questions.json"), config=config["agent"], num_workers=1,
        log_level="warning", pick_final_sql=False, run_start_time=started.isoformat()))
    manager.initialize_tasks(dataset)

    predictions: Dict[str, Optional[str]] = {}
    failures: Dict[str, str] = {}
    try:
        for task in manager.tasks:
            question_id = str(task.question_id)
            hooks.set_question(question_id)
            try:
                state, _, _ = manager.worker(task)
                predictions[question_id] = final_sql(state)
            except Exception as e:  # one question failing does not stop the run; it is reported
                predictions[question_id] = None
                failures[question_id] = f"{type(e).__name__}: {e}"
    finally:
        hooks.end_run()
        _write_json(run_dir / "predictions.json", predictions)
        calls = read_calls(run_dir / "calls.jsonl")
        manifest.update({
            "finished_at": _now().isoformat(),
            "status": "done" if len(predictions) == len(selected) else "interrupted",
            "n_calls": len(calls), "call_sites_seen": sorted({c["call_site"] for c in calls}),
            "c1_errors": len(validate_calls(calls)), "failures": failures,
        })
        _write_json(run_dir / "manifest.json", manifest)
    return run_dir


def preprocess(config_path: str, db_ids: List[str]) -> None:
    """CHESS preprocessing (value LSH and column-description vector DB) with the configured embeddings."""
    config = load_config(config_path)
    _prepare_chess(config)
    from database_utils.db_catalog.preprocess import make_db_context_vec_db
    from database_utils.db_values.preprocess import make_db_lsh
    settings = config["preprocess"]
    for db_id in db_ids:
        data.check_database(config, db_id)
        db_dir = paths.bird_root(config) / "dev_databases" / db_id
        if not (db_dir / "preprocessed" / f"{db_id}_lsh.pkl").exists():
            make_db_lsh(str(db_dir), signature_size=settings["signature_size"], n_gram=settings["n_gram"],
                        threshold=settings["threshold"], verbose=False)
        make_db_context_vec_db(str(db_dir), use_value_description=settings["use_value_description"])
