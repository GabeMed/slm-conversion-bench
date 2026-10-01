"""`bench run` and `bench preprocess`: executions of the agent (agent environment only).

An `agent` run writes, under runs/<run_id>/:
- manifest.json: arm, split, question ids, commit, configuration hash, facts, few-shot, status,
  and what went wrong per question;
- config.json: the merged configuration the run used (a fact: `bench eval` reads this snapshot,
  never the current config.yaml, and checks it against the manifest's hash);
- calls.jsonl: one C1 line per LLM invocation;
- predictions.json: {question_id: final SQL or null}, by CHESS's final-SQL rule;
- chess/: CHESS's own outputs, kept for inspection and without authority.

Arms B0, B1, B3, B4 and B5 run the patched CHESS, every call routed by C4; B2 is one call per
question with no agent (`--engine production_llm|cheap_alt`), written the same way. The gold SQL
never enters the agent's state (patch 13): CHESS receives each question without it.

A run is `failed` when the harness failed, a question raised, a C1 line is invalid or a database
changed (it stops at the first question that fails: nothing after it could be `done`);
`interrupted` when it stopped early for any other reason; `done` otherwise. On the test split, an
intent is committed and published before anything runs and the manifest is committed when the run
ends (`registry`), and a call site outside the one registered on train and calib aborts the run.

`--workers N` answers the questions of one run in N processes (CHESS's `DatabaseManager` is a
singleton per process that switches database, so questions cannot share a process): worker k takes
every N-th question from the k-th, writes its own C1 file, and the files are merged into the run's
`calls.jsonl` when all have ended. It is one run, one manifest (`workers`), and the first question
that fails stops every worker. The number of workers changes no output, so it is not part of the
configuration's identity.
"""
import json
import multiprocessing
import multiprocessing.connection
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from bench import barrier, data, paths
from bench.agent import TRACING_OFF, registry
from bench.contracts.calls import read_calls, validate_calls
from bench.contracts.config import chess_team_config, config_sha256, load_config
from bench.provenance import git_state, scrub

SINGLE_CALL_ENGINES = ("production_llm", "cheap_alt")
SINGLE_CALL_TEMPLATE = "generate_candidate_one"
SINGLE_CALL_KEY = "b2:0"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Variables that would change where a request goes, or send it elsewhere too: an OpenAI base URL
# redirects the retrieval embeddings (the chat models get theirs from C2 explicitly), and LangChain
# tracing ships every prompt to LangSmith. Unset or off, whatever the shell or a .env had.
REDIRECTING_ENV = ("OPENAI_BASE_URL", "OPENAI_API_BASE")


def _prepare_chess(config: Dict[str, Any], db_root: Path):
    """Point CHESS at the databases and templates, and hand it the configuration."""
    os.environ["DB_ROOT_PATH"] = str(db_root)
    os.environ.setdefault("INDEX_SERVER_PORT", "0")  # read at import by CHESS; unused by IR -> SS -> CG
    os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Chroma would otherwise send usage telemetry
    os.environ.update(TRACING_OFF)
    for name in REDIRECTING_ENV:
        os.environ.pop(name, None)
    for name in [n for n in os.environ if n.upper().startswith("CHROMA_")]:  # e.g. CHROMA_SERVER_HOST: a remote
        os.environ.pop(name)  # vector DB; Chroma's settings read their variables in any case
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


def _write_manifest(run_dir: Path, manifest: Dict[str, Any], config: Dict[str, Any]) -> None:
    """A run's manifest, redacted as a whole: provider text reaches it through every field that keeps an
    error (stopped_by, problems, failures, tool_errors, harness_errors), so no credential is written."""
    from bench.provenance import redact
    (run_dir / "manifest.json").write_text(redact(json.dumps(manifest, indent=2, ensure_ascii=False), config) + "\n")


def new_outcome() -> Dict[str, Dict]:
    return {"predictions": {}, "failures": {}, "harness_errors": {}, "tool_errors": {}}


AGENT_TASK_FIELDS = ("question_id", "db_id", "question", "evidence", "difficulty")


def agent_task(question: Dict[str, Any]) -> Dict[str, Any]:
    """What CHESS receives for a question (patch 13): only the fields of its `Task` that are not the
    gold, named here, so no gold-bearing field (`SQL`, Plat-SQL's `original_SQL`, ...) can pass."""
    return {key: question[key] for key in AGENT_TASK_FIELDS if key in question}


def _answer(question_id: str, hooks, outcome: Dict[str, Dict], answer: Callable[[], Tuple[Optional[str], Dict]]) -> bool:
    """One question: set it, answer it, record what went wrong. False when the run must stop."""
    hooks.set_question(question_id)
    try:
        outcome["predictions"][question_id], tool_errors = answer()
        if tool_errors:
            outcome["tool_errors"][question_id] = {k: scrub(str(v)) for k, v in tool_errors.items()}
    except Exception as e:  # reported, and it fails the run: the agent swallows the model's errors itself
        outcome["predictions"][question_id] = None
        outcome["failures"][question_id] = scrub(f"{type(e).__name__}: {e}")
    harness_errors = hooks.take_harness_errors()
    if harness_errors:
        outcome["harness_errors"][question_id] = harness_errors
    return not (question_id in outcome["failures"] or harness_errors)  # a failed run spends nothing more


def _answer_each(hooks, outcome: Dict[str, Dict], stop: Any,
                 answers: Iterable[Tuple[str, Callable[[], Tuple[Optional[str], Dict]]]]) -> None:
    """The questions of one process, one after another, until one fails. `stop` is the flag the
    workers of a run share: a question that fails here sets it, and no question starts once it is
    set. A question that was running when the flag was set from outside had its model calls refused
    (`hooks.RunAborted`, which CHESS swallows as a tool error): what it returned is not an answer,
    and it is left out, so the run cannot end `done` on it."""
    for question_id, answer in answers:
        if stop is not None and stop.is_set():
            break
        if not _answer(question_id, hooks, outcome, answer):
            if stop is not None:
                stop.set()
            break
        if stop is not None and stop.is_set():
            del outcome["predictions"][question_id]
            outcome["tool_errors"].pop(question_id, None)
            break


def _execute_questions(config: Dict[str, Any], arm: str, run_id: str, run_dir: Path,
                       dataset: List[Dict[str, Any]], db_root: Path, outcome: Dict[str, Dict],
                       few_shot: Optional[Dict[str, List]] = None,
                       allowed_call_sites: Optional[List[str]] = None, part: str = "", stop: Any = None) -> None:
    """Run the patched CHESS on each question, one after another; C1 goes to run_dir/calls<part>.jsonl
    (`part` is empty for a run in one process, `.w<k>` for worker k).

    Internal: the caller has already applied the test barrier to `dataset`. Fills `outcome` as it
    goes (so an interruption keeps what finished): the predictions and, per question, `failures`
    (the question raised), `harness_errors` and `tool_errors` (CHESS tools that failed and that
    CHESS swallowed; they can be the model's doing, so they are reported without failing the run).
    """
    hooks = _prepare_chess(config, db_root)
    hooks.start_run(run_id, arm, run_dir / f"calls{part}.jsonl", few_shot=few_shot,
                    allowed_call_sites=allowed_call_sites, stop=stop)
    try:
        from runner.run_manager import RunManager
        RunManager.RESULT_ROOT_PATH = str(run_dir / "chess")
        tasks = [agent_task(question) for question in dataset]
        questions_path = run_dir / f"questions{part}.json"  # CHESS names its own outputs after it: one per worker
        _write_json(questions_path, tasks)
        manager = RunManager(Namespace(
            data_mode="dev", data_path=str(questions_path), config=chess_team_config(config), num_workers=1,
            log_level="warning", pick_final_sql=False, run_start_time=run_id))
        manager.initialize_tasks(tasks)

        def answers():
            for task in manager.tasks:
                def answer(task=task):
                    state, _, _ = manager.worker(task)
                    return final_sql(state), state.errors
                yield str(task.question_id), answer
        _answer_each(hooks, outcome, stop, answers())
    finally:
        _close(hooks, outcome)


def _close(hooks, outcome: Dict[str, Dict]) -> None:
    outcome["unregistered_call_sites"] = hooks.unregistered_call_sites()
    hooks.end_run()


def single_call_generator(config: Dict[str, Any]) -> Dict[str, Any]:
    """B2 generates with CHESS's `generate_candidate_one` template and the parser the agent pairs it with."""
    tools = config["agent"]["team_agents"]["candidate_generator"]["tools"]
    for generator in tools["generate_candidate"]["generator_configs"]:
        if generator["template_name"] == SINGLE_CALL_TEMPLATE:
            return generator
    raise data.DataError(f"agent.team_agents.candidate_generator has no {SINGLE_CALL_TEMPLATE} generator")


def single_call_prompt(config: Dict[str, Any]):
    """B2's template and parser, resolved before a run exists (CHESS must be importable)."""
    from llm.parsers import get_parser
    from llm.prompts import get_prompt
    return get_prompt(template_name=SINGLE_CALL_TEMPLATE), get_parser(single_call_generator(config)["parser_name"])


def _execute_single_call(config: Dict[str, Any], engine: str, run_id: str, run_dir: Path,
                         dataset: List[Dict[str, Any]], db_root: Path, outcome: Dict[str, Dict],
                         allowed_call_sites: Optional[List[str]] = None, part: str = "", stop: Any = None) -> None:
    """B2: text to SQL in one call per question, no agent and no retrieval: the question, its
    evidence and the complete schema of the database (as CHESS writes it), on a fixed engine. A
    call the model fails (after the retries) leaves the question without a prediction, as a
    swallowed tool error leaves it in the agent; it is reported and does not fail the run."""
    hooks = _prepare_chess(config, db_root)
    hooks.start_run(run_id, None, run_dir / f"calls{part}.jsonl", engine=engine, few_shot={},
                    allowed_call_sites=allowed_call_sites, stop=stop)
    try:
        from runner.database_manager import DatabaseManager
        prompt, parser = single_call_prompt(config)

        def answers():
            for question in dataset:
                def answer(question=question):
                    manager = DatabaseManager(db_mode="dev", db_id=question["db_id"])
                    schema = manager.get_database_schema_string(manager.get_db_schema(), {}, {}, include_value_description=True)
                    messages = prompt.invoke({"DATABASE_SCHEMA": schema, "QUESTION": question["question"],
                                              "HINT": question["evidence"]}).to_messages()
                    try:
                        return hooks.invoke_tool_call("generate_candidate", SINGLE_CALL_KEY, messages, parser)["SQL"], {}
                    except hooks.HarnessError:
                        raise
                    except Exception as e:  # the model's failure, already in C1
                        return None, {"generate_candidate": f"{type(e).__name__}: {e}"}
                yield str(question["question_id"]), answer
        _answer_each(hooks, outcome, stop, answers())
    finally:
        _close(hooks, outcome)


# ---------------------------------------------------------------- one run in several processes

OUTCOME_KEYS = ("predictions", "failures", "harness_errors", "tool_errors")


def worker_parts(dataset: List[Any], workers: int) -> List[List[Any]]:
    """The questions of each worker: worker k takes every `workers`-th question from the k-th, so the
    parts are disjoint, cover the run and depend only on the order of the questions."""
    return [part for part in (dataset[k::workers] for k in range(workers)) if part]


def _worker(execute: Callable[..., None], kwargs: Dict[str, Any], k: int, stop: Any) -> None:
    """One worker process: answer its part, then leave what happened in run_dir/outcome.w<k>.json
    (the parent merges it), redacted like the manifest it ends up in: the file outlives a parent that
    dies before merging. Whatever stops it other than a question that failed stops every worker too
    and is reported in `stopped_by`."""
    from bench.provenance import redact
    outcome: Dict[str, Any] = new_outcome()
    try:
        execute(outcome=outcome, part=f".w{k}", stop=stop, **kwargs)
    except BaseException as e:  # a Ctrl-C included: the parent reports it, this process just ends
        stop.set()
        outcome["stopped_by"] = scrub(f"{type(e).__name__}: {e}")
    finally:
        (kwargs["run_dir"] / f"outcome.w{k}.json").write_text(redact(json.dumps(outcome, ensure_ascii=False), kwargs["config"]))


def _in_workers(execute: Callable[..., None], kwargs: Dict[str, Any], parts: List[List[Dict[str, Any]]],
                outcome: Dict[str, Dict]) -> None:
    """Run `execute` on each part of the questions in its own process (spawned: the agent's threads
    and clients do not survive a fork), wait for all of them, then merge what they left into
    `outcome` and their C1 files, in worker order, into run_dir/calls.jsonl. A worker that dies
    (a non-zero exit: it could not set the flag itself) stops the others, and fails the run. A worker
    that stopped for anything but a failed question raises here once everything is merged, as the
    same failure raises in a run of one process."""
    run_dir = kwargs["run_dir"]
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    processes = [context.Process(target=_worker, args=(execute, {**kwargs, "dataset": part}, k, stop))
                 for k, part in enumerate(parts)]
    started: List[Any] = []
    try:
        try:
            for process in processes:
                process.start()
                started.append(process)
            running = list(started)
            while running:
                ended = multiprocessing.connection.wait([process.sentinel for process in running])
                for process in [p for p in running if p.sentinel in ended]:
                    process.join()
                    running.remove(process)
                    if process.exitcode != 0:
                        stop.set()
        except BaseException:  # a Ctrl-C reaches every worker as well: wait for what they finished
            stop.set()
            for process in started:
                process.join()
            raise
    finally:
        stopped = _merge_workers(run_dir, processes, outcome)
    if stopped:
        from bench.agent.hooks import HarnessError
        raise HarnessError("; ".join(stopped))


def _merge_workers(run_dir: Path, processes: List[Any], outcome: Dict[str, Dict]) -> List[str]:
    """Merge every worker's outcome and C1 file into the run's. A worker that left no outcome died:
    that is the harness's failure, recorded under `harness_errors` (so the run is `failed`). Returns
    why workers stopped (`stopped_by`), if any did."""
    stopped, unregistered = [], set()
    with open(run_dir / "calls.jsonl", "a") as merged:
        for k, process in enumerate(processes):
            calls_path, outcome_path = run_dir / f"calls.w{k}.jsonl", run_dir / f"outcome.w{k}.json"
            if calls_path.exists():
                text = calls_path.read_text()
                merged.write(text if not text or text.endswith("\n") else text + "\n")  # a killed worker's cut-off
                calls_path.unlink()                                 # last line never runs into the next worker's first
            if not outcome_path.exists():
                outcome["harness_errors"][f"worker {k}"] = [
                    f"the worker ended without reporting (exit code {process.exitcode}): its questions have no outcome"]
                continue
            reported = json.loads(outcome_path.read_text())
            outcome_path.unlink()
            for key in OUTCOME_KEYS:
                outcome[key].update(reported[key])
            unregistered.update(reported.get("unregistered_call_sites", []))
            if reported.get("stopped_by"):
                stopped.append(f"worker {k} stopped: {reported['stopped_by']}")
    outcome["unregistered_call_sites"] = sorted(unregistered)
    return stopped


def run_status(selected: List[str], outcome: Dict[str, Dict], c1_errors: int, databases_changed: List[str]) -> str:
    if outcome["failures"] or outcome["harness_errors"] or c1_errors or databases_changed:
        return "failed"
    if set(outcome["predictions"]) != set(selected):
        return "interrupted"
    return "done"


def check_engines(config: Dict[str, Any], engines: List[str], embeddings: bool = True) -> None:
    """Every engine is buildable (endpoint and API key present), and the retrieval embeddings are."""
    from bench.agent import hooks
    hooks.configure(config)
    for engine in engines:
        hooks.chat_model(engine, 0.0)
    if embeddings:
        hooks.embeddings("entity")
        hooks.embeddings("context")


def _check_arm(config: Dict[str, Any], arm: str) -> Dict[str, str]:
    """Before any work: every engine the arm can route to is buildable, the retrieval embeddings
    are, and every fact the arm uses is set and consistent. Returns {fact: sha256}."""
    from bench.contracts.router import arm_facts, possible_engines
    check_engines(config, possible_engines(arm, config))
    return {name: sha for name, (_, sha) in arm_facts(arm, config).items()}


def few_shot_for(config: Dict[str, Any], engines: List[str]) -> Tuple[Optional[Dict], Dict[str, Any]]:
    """B1's prefix when `cheap_alt` can be reached, else None; and the manifest fields that record
    it (`few_shot`, `few_shot_k`, `few_shot_sha256`, all null when there is no prefix)."""
    if "cheap_alt" not in engines:
        return None, {"few_shot": None, "few_shot_k": None, "few_shot_sha256": None}
    from bench.agent import few_shot
    prefix, provenance = few_shot.build(config)
    return prefix, {"few_shot": provenance, "few_shot_k": provenance["k"], "few_shot_sha256": few_shot.digests(prefix)}


def _preprocess_stamps(config: Dict[str, Any], db_dir: Path) -> Dict[Path, Dict[str, Any]]:
    """What each preprocessing output was built with. `bench preprocess` writes these stamps last,
    so a stamp that matches means a complete build with the settings this run would query with."""
    from bench.agent import hooks
    hooks.configure(config)
    embeddings = config["embeddings"]
    vector = {"provider": embeddings["provider"], "context_model": embeddings["context_model"],
              "fake_size": embeddings["fake_size"], "use_value_description": config["preprocess"]["use_value_description"]}
    if embeddings["provider"] == "local":  # only then, so the stamps of the other providers are unchanged
        vector["local"] = embeddings["local"]
    return {
        db_dir / "preprocessed" / "STAMP.json": {"preprocess": config["preprocess"]},
        db_dir / hooks.vector_db_dirname() / "STAMP.json": vector,
    }


def _check_preprocessed(config: Dict[str, Any], db_ids: List[str]) -> None:
    for db_id in db_ids:
        db_dir = paths.bird_root(config) / "dev_databases" / db_id
        for stamp, expected in _preprocess_stamps(config, db_dir).items():
            if not stamp.exists() or json.loads(stamp.read_text()) != expected:
                raise data.DataError(f"{db_id} is not preprocessed for this configuration "
                                     f"({stamp.parent.name}/{stamp.name} missing or different): "
                                     f"run `bench preprocess --db {db_id}` with the same --config")


def select_questions(config: Dict[str, Any], split: str, ids: Optional[List[str]],
                     limit: Optional[int]) -> Tuple[Dict[str, Dict], List[str]]:
    barrier.ensure_split_allowed(split, config)
    questions = data.questions_for(config, split)
    if ids:
        outside = [q for q in ids if q not in questions]
        if outside:
            raise data.DataError(f"ids not in the {split} split: {outside}")
        if len(set(ids)) != len(ids):
            raise data.DataError("an id appears more than once in --ids")
        return questions, list(ids)
    return questions, sorted(questions, key=int)[:limit] if limit else sorted(questions, key=int)


def used_key_envs(config: Dict[str, Any], engines: List[str], retrieval: bool) -> List[str]:
    """The environment variables holding the credentials an execution uses: those of its engines'
    endpoints (API keys and `headers_env` values), and OPENAI_API_KEY when it retrieves with OpenAI
    embeddings."""
    from bench.contracts.config import endpoint_credential_envs, engine_spec
    names = [name for engine in engines for name in endpoint_credential_envs(engine_spec(config, engine)["endpoint"])]
    if retrieval and config["embeddings"]["provider"] == "openai":
        names.append("OPENAI_API_KEY")
    return [name for name in names if name]


def open_run(config: Dict[str, Any], config_path: str, run_type: str, label: str, split: str,
             fields: Dict[str, Any], key_envs: List[str]) -> Tuple[Path, Dict[str, Any], Optional[List[str]]]:
    """Create runs/<run_id>/ with its manifest and configuration snapshot, after every precondition.
    On the test split: the registered call sites are required, every key used is long enough to be
    redacted, and the intent is committed and published first (a refusal there leaves no run).
    Returns (run_dir, manifest, the call sites a test run may emit or None)."""
    if split == "test":
        registry.check_keys(key_envs)
    test = registry.registered_call_sites() if split == "test" else None
    started = _now()
    run_id = f"{run_type}-{label}-{split}-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
    manifest = {
        "run_id": run_id, "type": run_type, "split": split, **fields,
        **git_state(), "config_path": str(Path(config_path).resolve().relative_to(paths.ROOT)),
        "config_sha256": config_sha256(config),
        "splits_sha256": data.sha256_file(paths.SPLITS),
        "data_manifest_sha256": data.sha256_file(paths.DATA_MANIFEST),
        "started_at": started.isoformat(), "finished_at": None, "status": "running",
    }
    if test is not None:
        manifest.update({"prereg_hash": barrier.prereg_hash_in_force(config), "call_sites_registry_sha256": test["sha256"]})
        manifest["registry_intent"] = registry.commit_intent(manifest)
    run_dir = paths.RUNS / run_id
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "config.json", config)
    _write_manifest(run_dir, manifest, config)
    return run_dir, manifest, None if test is None else test["call_sites"]


def run_agent(config_path: str, arm: str, split: str, ids: Optional[List[str]] = None,
              limit: Optional[int] = None, engine: Optional[str] = None, workers: int = 1) -> Path:
    config = load_config(config_path)
    from bench.agent import hooks
    hooks.configure(config)
    if workers < 1:
        raise data.DataError("--workers must be at least 1")
    if arm == "B2":
        if engine not in SINGLE_CALL_ENGINES:
            raise data.DataError(f"B2 needs --engine, one of {SINGLE_CALL_ENGINES}")
    elif engine is not None:
        raise data.DataError(f"--engine is for B2; {arm} is routed by the router")
    questions, selected = select_questions(config, split, ids, limit)
    databases = sorted({questions[q]["db_id"] for q in selected})
    if arm == "B2":
        check_engines(config, [engine], embeddings=False)  # no retrieval: no embeddings, no preprocessing
        _prepare_chess(config, paths.bird_root(config))
        single_call_prompt(config)
        facts, few_shot = {}, {}
        recorded = {"few_shot": None, "few_shot_k": None, "few_shot_sha256": None}  # B2 gets no prefix
    else:
        from bench.contracts.router import possible_engines
        facts = _check_arm(config, arm)
        few_shot, recorded = few_shot_for(config, possible_engines(arm, config))
    for db_id in databases:
        data.check_database(config, db_id)
    if arm != "B2":
        _check_preprocessed(config, databases)

    fields = {"arm": arm, "question_ids": selected, "databases": databases, "facts": facts, **recorded,
              "workers": workers}
    if arm == "B2":
        fields.update({"mode": "single_call", "engine": engine})
    if arm == "B2":
        key_envs = used_key_envs(config, [engine], retrieval=False)
    else:
        key_envs = used_key_envs(config, possible_engines(arm, config), retrieval=True)
    run_dir, manifest, allowed = open_run(config, config_path, "agent", f"B2-{engine}" if arm == "B2" else arm,
                                          split, fields, key_envs)
    outcome = new_outcome()
    stopped_by = None
    try:
        dataset = [{**questions[q], "question_id": int(q)} for q in selected]
        kwargs = {"config": config, "run_id": manifest["run_id"], "run_dir": run_dir,
                  "db_root": paths.bird_root(config), "allowed_call_sites": allowed}
        if arm == "B2":
            execute, kwargs = _execute_single_call, {**kwargs, "engine": engine}
        else:
            execute, kwargs = _execute_questions, {**kwargs, "arm": arm, "few_shot": few_shot}
        if workers == 1:
            execute(dataset=dataset, outcome=outcome, **kwargs)
        else:
            _in_workers(execute, kwargs, worker_parts(dataset, workers), outcome)
    except BaseException as e:  # recorded in the manifest, then re-raised
        stopped_by = scrub(f"{type(e).__name__}: {e}")
        raise
    finally:
        _write_json(run_dir / "predictions.json", {q: outcome["predictions"][q] for q in selected
                                                   if q in outcome["predictions"]})  # in the run's order
        finish(manifest, run_dir, config, databases, outcome, stopped_by,
               lambda c1_errors, changed: run_status(selected, outcome, c1_errors, changed),
               {"failures": outcome["failures"], "tool_errors": outcome["tool_errors"]})
    return run_dir


def finish(manifest: Dict[str, Any], run_dir: Path, config: Dict[str, Any], databases: List[str],
           outcome: Dict[str, Any], stopped_by: Optional[str], status: Callable[[int, List[str]], str],
           fields: Dict[str, Any], problems: Optional[List[str]] = None) -> None:
    """Close the manifest whatever happened; a failure to read the evidence is itself recorded.
    On the test split, the manifest is then committed to the registry."""
    problems = list(problems or [])
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
    errors: Dict[str, Dict[str, int]] = {}
    for call in calls:  # a systematic rejection (a 400 on every call) is visible here, not only in C1
        if call.get("error"):
            kinds = errors.setdefault(call["call_site"], {})
            kind = call["error"].split(":")[0]
            kinds[kind] = kinds.get(kind, 0) + 1
    manifest.update({
        "finished_at": _now().isoformat(), "status": status(c1_errors, changed),
        "stopped_by": stopped_by, "problems": problems,
        "n_calls": len(calls), "call_sites_seen": sorted({c["call_site"] for c in calls}),
        "errors_by_call_site": errors,
        "c1_errors": c1_errors, **fields, "harness_errors": outcome["harness_errors"],
        "databases_changed": changed,
    })
    if manifest["split"] == "test":
        manifest["unregistered_call_sites"] = outcome.get("unregistered_call_sites", [])
    _write_manifest(run_dir, manifest, config)
    if manifest["split"] == "test":
        try:
            registry.commit_manifest(run_dir, config)
        except registry.RegistryError as e:  # never a `done` run the registry does not show
            manifest["status"] = "failed"
            manifest["problems"].append(scrub(f"the manifest could not be committed to the registry: {e}"))
            _write_manifest(run_dir, manifest, config)
            if stopped_by is None:  # otherwise the exception that stopped the run goes on
                raise


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
