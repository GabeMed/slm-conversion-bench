"""`bench run` and `bench preprocess`: executions of the agent (agent environment only).

An `agent` run writes, under runs/<run_id>/:
- manifest.json: arm, split, question ids, commit, configuration hash, facts, status, and what
  went wrong per question;
- config.json: the merged configuration the run used (a fact: `bench eval` reads this snapshot,
  never the current config.yaml, and checks it against the manifest's hash);
- calls.jsonl: one C1 line per LLM invocation;
- predictions.json: {question_id: final SQL or null}, by CHESS's final-SQL rule;
- chess/: CHESS's own outputs, kept for inspection and without authority.

A run is `failed` when the harness failed, a question raised, a C1 line is invalid or a database
changed (it stops at the first question that fails: nothing after it could be `done`);
`interrupted` when it stopped early for any other reason; `done` otherwise.
"""
import json
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench import barrier, data, paths
from bench.contracts.calls import read_calls, validate_calls
from bench.contracts.config import chess_team_config, config_sha256, load_config
from bench.provenance import git_state, scrub


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _prepare_chess(config: Dict[str, Any], db_root: Path):
    """Point CHESS at the databases and templates, and hand it the configuration."""
    os.environ["DB_ROOT_PATH"] = str(db_root)
    os.environ.setdefault("INDEX_SERVER_PORT", "0")  # read at import by CHESS; unused by IR -> SS -> CG
    os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Chroma would otherwise send usage telemetry
    src = str(paths.VENDOR_CHESS / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from bench.agent import hooks
    hooks.configure(config)
    import llm.prompts
    import runner.database_manager
    llm.prompts.TEMPLATES_ROOT_PATH = str(paths.VENDOR_CHESS / "templates")
    runner.database_manager.DB_ROOT_PATH = Path(db_root)  # read at import; set again for this run
    runner.database_manager.DatabaseManager._instance = None  # the singleton keeps its paths otherwise
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


def new_outcome() -> Dict[str, Dict]:
    return {"predictions": {}, "failures": {}, "harness_errors": {}, "tool_errors": {}}


def _execute_questions(config: Dict[str, Any], arm: str, run_id: str, run_dir: Path,
                       dataset: List[Dict[str, Any]], db_root: Path, outcome: Dict[str, Dict]) -> None:
    """Run the patched CHESS on each question, one after another; C1 goes to run_dir/calls.jsonl.

    Internal: the caller has already applied the test barrier to `dataset`. Fills `outcome` as it
    goes (so an interruption keeps what finished): the predictions and, per question, `failures`
    (the question raised), `harness_errors` and `tool_errors` (CHESS tools that failed and that
    CHESS swallowed; they can be the model's doing, so they are reported without failing the run).
    """
    hooks = _prepare_chess(config, db_root)
    hooks.start_run(run_id, arm, run_dir / "calls.jsonl")
    try:
        from runner.run_manager import RunManager
        RunManager.RESULT_ROOT_PATH = str(run_dir / "chess")
        _write_json(run_dir / "questions.json", dataset)
        manager = RunManager(Namespace(
            data_mode="dev", data_path=str(run_dir / "questions.json"), config=chess_team_config(config), num_workers=1,
            log_level="warning", pick_final_sql=False, run_start_time=run_id))
        manager.initialize_tasks(dataset)
        for task in manager.tasks:
            question_id = str(task.question_id)
            hooks.set_question(question_id)
            try:
                state, _, _ = manager.worker(task)
                outcome["predictions"][question_id] = final_sql(state)
                if state.errors:
                    outcome["tool_errors"][question_id] = {k: scrub(str(v)) for k, v in state.errors.items()}
            except Exception as e:  # reported, and it fails the run: the agent swallows the model's errors itself
                outcome["predictions"][question_id] = None
                outcome["failures"][question_id] = scrub(f"{type(e).__name__}: {e}")
            harness_errors = hooks.take_harness_errors()
            if harness_errors:
                outcome["harness_errors"][question_id] = harness_errors
            if question_id in outcome["failures"] or harness_errors:
                break  # the run is failed: spend nothing more on it
    finally:
        hooks.end_run()


def run_status(selected: List[str], outcome: Dict[str, Dict], c1_errors: int, databases_changed: List[str]) -> str:
    if outcome["failures"] or outcome["harness_errors"] or c1_errors or databases_changed:
        return "failed"
    if set(outcome["predictions"]) != set(selected):
        return "interrupted"
    return "done"


def _check_arm(config: Dict[str, Any], arm: str) -> Dict[str, str]:
    """Before any work: every engine the arm can route to is buildable (endpoint and API key
    present), the retrieval embeddings are, and every fact the arm uses is set and consistent.
    Returns {fact: sha256}."""
    from bench.agent import hooks
    from bench.contracts.router import arm_facts, possible_engines
    engines = possible_engines(arm, config)
    hooks.configure(config)
    for engine in engines:
        hooks.chat_model(engine, 0.0)
    hooks.embeddings("entity")
    hooks.embeddings("context")
    return {name: sha for name, (_, sha) in arm_facts(arm, config).items()}


def _preprocess_stamps(config: Dict[str, Any], db_dir: Path) -> Dict[Path, Dict[str, Any]]:
    """What each preprocessing output was built with. `bench preprocess` writes these stamps last,
    so a stamp that matches means a complete build with the settings this run would query with."""
    from bench.agent import hooks
    hooks.configure(config)
    embeddings = config["embeddings"]
    return {
        db_dir / "preprocessed" / "STAMP.json": {"preprocess": config["preprocess"]},
        db_dir / hooks.vector_db_dirname() / "STAMP.json": {
            "provider": embeddings["provider"], "context_model": embeddings["context_model"],
            "fake_size": embeddings["fake_size"], "use_value_description": config["preprocess"]["use_value_description"]},
    }


def _check_preprocessed(config: Dict[str, Any], db_ids: List[str]) -> None:
    for db_id in db_ids:
        db_dir = paths.bird_root(config) / "dev_databases" / db_id
        for stamp, expected in _preprocess_stamps(config, db_dir).items():
            if not stamp.exists() or json.loads(stamp.read_text()) != expected:
                raise data.DataError(f"{db_id} is not preprocessed for this configuration "
                                     f"({stamp.parent.name}/{stamp.name} missing or different): "
                                     f"run `bench preprocess --db {db_id}` with the same --config")


def run_agent(config_path: str, arm: str, split: str, ids: Optional[List[str]] = None,
              limit: Optional[int] = None) -> Path:
    config = load_config(config_path)
    barrier.ensure_split_allowed(split, config)
    questions = data.questions_for(config, split)
    if ids:
        outside = [q for q in ids if q not in questions]
        if outside:
            raise data.DataError(f"ids not in the {split} split: {outside}")
        if len(set(ids)) != len(ids):
            raise data.DataError("an id appears more than once in --ids")
        selected = list(ids)
    else:
        selected = sorted(questions, key=int)[:limit] if limit else sorted(questions, key=int)
    facts = _check_arm(config, arm)
    databases = sorted({questions[q]["db_id"] for q in selected})
    for db_id in databases:
        data.check_database(config, db_id)
    _check_preprocessed(config, databases)

    started = _now()
    run_id = f"agent-{arm}-{split}-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
    run_dir = paths.RUNS / run_id
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "config.json", config)
    manifest = {
        "run_id": run_id, "type": "agent", "arm": arm, "split": split, "question_ids": selected,
        **git_state(), "config_path": str(Path(config_path).resolve().relative_to(paths.ROOT)),
        "config_sha256": config_sha256(config),
        "splits_sha256": data.sha256_file(paths.SPLITS),
        "data_manifest_sha256": data.sha256_file(paths.DATA_MANIFEST),
        "databases": databases, "facts": facts,
        "started_at": started.isoformat(), "finished_at": None, "status": "running",
    }
    _write_json(run_dir / "manifest.json", manifest)

    outcome = new_outcome()
    stopped_by = None
    try:
        dataset = [{**questions[q], "question_id": int(q)} for q in selected]
        _execute_questions(config, arm, run_id, run_dir, dataset, paths.bird_root(config), outcome)
    except BaseException as e:  # recorded in the manifest, then re-raised
        stopped_by = scrub(f"{type(e).__name__}: {e}")
        raise
    finally:
        _write_json(run_dir / "predictions.json", outcome["predictions"])
        _finish(manifest, run_dir, config, selected, databases, outcome, stopped_by)
    return run_dir


def _finish(manifest, run_dir, config, selected, databases, outcome, stopped_by) -> None:
    """Close the manifest whatever happened; a failure to read the evidence is itself recorded."""
    problems = []
    try:
        calls = read_calls(run_dir / "calls.jsonl") if (run_dir / "calls.jsonl").exists() else []
        c1_errors = len(validate_calls(calls))
    except Exception as e:
        calls, c1_errors = [], 1
        problems.append(scrub(f"calls.jsonl unreadable: {type(e).__name__}: {e}"))
    changed = []
    for db_id in databases:
        try:
            data.check_database(config, db_id)
        except Exception:
            changed.append(db_id)
    manifest.update({
        "finished_at": _now().isoformat(), "status": run_status(selected, outcome, c1_errors, changed),
        "stopped_by": stopped_by, "problems": problems,
        "n_calls": len(calls), "call_sites_seen": sorted({c["call_site"] for c in calls}),
        "c1_errors": c1_errors, "failures": outcome["failures"], "harness_errors": outcome["harness_errors"],
        "tool_errors": outcome["tool_errors"], "databases_changed": changed,
    })
    _write_json(run_dir / "manifest.json", manifest)


def preprocess(config_path: str, db_ids: List[str]) -> None:
    """CHESS preprocessing (value LSH and column-description vector DB) with the configured embeddings."""
    config = load_config(config_path)
    _prepare_chess(config, paths.bird_root(config))
    from database_utils.db_catalog.preprocess import make_db_context_vec_db
    from database_utils.db_values.preprocess import make_db_lsh
    settings = config["preprocess"]
    for db_id in db_ids:
        data.check_database(config, db_id)
        db_dir = paths.bird_root(config) / "dev_databases" / db_id
        lsh_stamp, vector_stamp = _preprocess_stamps(config, db_dir).items()
        if not lsh_stamp[0].exists() or json.loads(lsh_stamp[0].read_text()) != lsh_stamp[1]:
            lsh_stamp[0].unlink(missing_ok=True)
            make_db_lsh(str(db_dir), signature_size=settings["signature_size"], n_gram=settings["n_gram"],
                        threshold=settings["threshold"], verbose=False)
            _write_json(lsh_stamp[0], lsh_stamp[1])  # last: a stamp means a complete build
        vector_stamp[0].unlink(missing_ok=True)
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()  # a cached client would still point at the directory CHESS deletes
        make_db_context_vec_db(str(db_dir), use_value_description=settings["use_value_description"])
        _write_json(vector_stamp[0], vector_stamp[1])
