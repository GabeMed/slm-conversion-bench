"""`bench replay`: another engine answering the teacher's own inputs, call by call (design §5.1).

A replay resends the **first attempt** of every invocation of a source run (a `done` B0 agent run)
to another engine: `--engine <name>` fixed, or `--arm <arm>` to route each call as that arm does
(C4, with its facts). Each resent invocation keeps the source's `question_id` and `invocation_key`,
so J2 pairs them, and runs under the same policy as in the agent: the call site's temperature and
parser (the agent's own rules for the agents' calls), the parse and transport retries, and B1's
few-shot prefix whenever the engine is `cheap_alt`. `--call-sites` keeps only those call sites.
`zeroshot` (S4) is a replay with `--engine slm:<candidate>`.

It writes runs/<run_id>/ with `calls.jsonl`, `manifest.json` (`type: replay`, `source_run_id`) and
the `config.json` snapshot every execution keeps; no predictions, since no SQL is chosen. The
barrier applies to the source run's split, and on `test` so do the registry and the call-site
assertion. A model failure on one invocation (after its retries) is recorded in C1 and the replay
goes on; a harness failure stops it (`failed`).
"""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from bench import barrier, data, paths
from bench.agent import hooks
from bench.agent.runner import _prepare_chess, check_engines, few_shot_for, finish, open_run
from bench.contracts.calls import AGENT_CALL_SITES, CALL_SITES, read_calls, validate_calls
from bench.contracts.config import config_sha256, engine_spec, load_config
from bench.provenance import scrub

POLICY_BLOCKS = ("call_sites", "retries", "agent")  # temperatures, retry budgets, parsers and templates


def differences(source: Any, replay: Any, path: str) -> List[str]:
    """Where two configuration blocks differ, as dotted paths with both values."""
    if isinstance(source, dict) and isinstance(replay, dict):
        found = []
        for key in sorted(set(source) | set(replay), key=str):
            where = f"{path}.{key}"
            if key not in source:
                found.append(f"{where}: absent in the source, {json.dumps(replay[key])} here")
            elif key not in replay:
                found.append(f"{where}: {json.dumps(source[key])} in the source, absent here")
            else:
                found += differences(source[key], replay[key], where)
        return found
    return [] if source == replay else [f"{path}: {json.dumps(source)} in the source, {json.dumps(replay)} here"]


def _same_policy(source_run_id: str, source: Dict[str, Any], config: Dict[str, Any]) -> None:
    """A replay answers the source's invocations under the source's policy: the same temperatures,
    parsers and templates, and the same retry budgets, or the pairing J2 makes compares two policies."""
    source_config = json.loads((paths.RUNS / source_run_id / "config.json").read_text())
    if config_sha256(source_config) != source["config_sha256"]:
        raise data.DataError(f"the configuration snapshot of {source_run_id} does not match its manifest")
    found = [d for block in POLICY_BLOCKS for d in differences(source_config.get(block), config.get(block), block)]
    if found:
        raise data.DataError(f"the replay's configuration differs from {source_run_id}'s in "
                             f"{', '.join(POLICY_BLOCKS)}: " + "; ".join(found[:10]))


TOOL_PARSERS = {"extract_keywords": ("information_retriever", "extract_keywords"),
                "filter_column": ("schema_selector", "filter_column"),
                "select_tables": ("schema_selector", "select_tables"),
                "select_columns": ("schema_selector", "select_columns"),
                "revise": ("candidate_generator", "revise")}


def _source(source_run_id: str) -> Dict[str, Any]:
    path = paths.RUNS / source_run_id / "manifest.json"
    if not path.exists():
        raise data.DataError(f"no run {source_run_id}")
    manifest = json.loads(path.read_text())
    shape = (manifest.get("type"), manifest.get("arm"), manifest.get("status"))
    if shape != ("agent", "B0", "done"):
        raise data.DataError(f"{source_run_id} is (type, arm, status) = {shape}: a replay resends a done B0 agent run")
    return manifest


def readers(config: Dict[str, Any]) -> Callable[[Dict[str, Any]], Callable[[], Any]]:
    """For a source record, the call that resends it: the tool's parser as the agent configures it,
    or the agent's own reading of its answer (`Agent.parse_action`). CHESS must be importable."""
    from llm.parsers import get_parser
    from workflow.agents.agent import AGENT_CALL_SITES as AGENT_NAMES
    from workflow.team_builder import AGENT_CLASSES
    team = config["agent"]["team_agents"]
    agents = {}
    for agent_name, agent_config in team.items():
        agent = AGENT_CLASSES[agent_name](config=agent_config)
        agents[AGENT_NAMES[agent.name]] = agent
    parsers = {site: team[agent]["tools"][tool]["parser_name"] for site, (agent, tool) in TOOL_PARSERS.items()}
    generators = {g["template_name"]: g["parser_name"] for g in team["candidate_generator"]["tools"]["generate_candidate"]["generator_configs"]}

    def reader(record: Dict[str, Any]) -> Callable[[], Any]:
        site, key, messages = record["call_site"], record["invocation_key"], record["prompt_messages"]
        if site in AGENT_CALL_SITES:
            if [m["role"] for m in messages] != ["user"]:
                raise data.DataError(f"{record['question_id']} {site} {key}: an agent call is one user message")
            parse = agents[site].parse_action
            return lambda: hooks.invoke_agent_call(site, key, messages[0]["content"], parse)
        if site == "generate_candidate":
            template = key.split(":")[0]
            if template not in generators:
                raise data.DataError(f"{record['question_id']} {site} {key}: no generator {template!r} in the configuration")
            parser_name = generators[template]
        else:
            parser_name = parsers[site]
        lc_messages, parser = hooks._lc_messages(messages), get_parser(parser_name)
        return lambda: hooks.invoke_tool_call(site, key, lc_messages, parser)
    return reader


def replay(config_path: str, source_run_id: str, engine: Optional[str] = None, arm: Optional[str] = None,
           call_sites: Optional[List[str]] = None) -> Path:
    config = load_config(config_path)
    hooks.configure(config)
    if (engine is None) == (arm is None):
        raise data.DataError("a replay needs --engine or --arm, one of the two")
    source = _source(source_run_id)
    split = source["split"]
    barrier.ensure_split_allowed(split, config)  # the source run's split
    _same_policy(source_run_id, source, config)
    unknown = set(call_sites or ()) - set(CALL_SITES)
    if unknown:
        raise data.DataError(f"unknown call sites: {sorted(unknown)}")
    calls_path = paths.RUNS / source_run_id / "calls.jsonl"
    calls = read_calls(calls_path)
    errors = validate_calls(calls)
    if errors:
        raise data.DataError(f"{source_run_id}: calls.jsonl is not valid C1: {errors[:3]}")
    records = [c for c in calls if c["attempt"] == 1 and (not call_sites or c["call_site"] in call_sites)]

    if engine is not None:
        engine_spec(config, engine)  # a ConfigError names what is wrong with the engine
        engines, facts = [engine], {}
    else:
        from bench.contracts.router import arm_facts, possible_engines
        engines = possible_engines(arm, config)
        facts = {name: sha for name, (_, sha) in arm_facts(arm, config).items()}
    check_engines(config, engines, embeddings=False)  # nothing is retrieved: the prompts are the source's
    # (building CHESS's agents below, to read their answers, touches no embedding: patch 10 builds them at first use)
    few_shot, recorded = few_shot_for(config, engines)
    _prepare_chess(config, paths.bird_root(config))
    reader = readers(config)
    resend = [(record, reader(record)) for record in records]  # every record readable before anything runs

    question_ids = sorted({r["question_id"] for r in records}, key=int)
    fields = {"source_run_id": source_run_id, "source_config_sha256": source["config_sha256"],
              "source_calls_sha256": hashlib.sha256(calls_path.read_bytes()).hexdigest(),
              "engine": engine, "arm": arm, "call_sites": sorted(call_sites) if call_sites else None,
              "question_ids": question_ids, "n_invocations": len(records), "facts": facts, **recorded}
    label = arm or re.sub(r"[^A-Za-z0-9_.-]+", "_", engine)
    run_dir, manifest, allowed = open_run(config, config_path, "replay", label, split, fields)
    outcome = {"harness_errors": {}, "model_failures": {}, "replayed": set()}
    stopped_by = None
    try:
        _execute(config, manifest["run_id"], run_dir, arm, engine, resend, question_ids, few_shot, allowed, outcome)
    except BaseException as e:  # recorded in the manifest, then re-raised
        stopped_by = scrub(f"{type(e).__name__}: {e}")
        raise
    finally:
        empty = [] if records else [f"no invocation to replay: {source_run_id} has none"
                                    + (f" at the call sites {sorted(call_sites)}" if call_sites else "")]

        def status(c1_errors, changed):
            if outcome["harness_errors"] or c1_errors or empty:
                return "failed"
            return "done" if outcome["replayed"] == set(question_ids) else "interrupted"
        finish(manifest, run_dir, config, [], outcome, stopped_by, status,
               {"n_questions_replayed": len(outcome["replayed"]), "model_failures": outcome["model_failures"]},
               problems=empty)
    return run_dir


def _execute(config, run_id, run_dir, arm, engine, resend, question_ids, few_shot, allowed, outcome) -> None:
    """Question by question (the question is process state in `hooks`); the invocations of one
    question in parallel, at most `agent.max_workers` at once, as the agent's own steps are."""
    by_question: Dict[str, List] = {q: [] for q in question_ids}
    for record, call in resend:
        by_question[record["question_id"]].append((record, call))
    hooks.start_run(run_id, arm, run_dir / "calls.jsonl", engine=engine, few_shot=few_shot, allowed_call_sites=allowed)
    try:
        for question_id, invocations in by_question.items():
            hooks.set_question(question_id)
            with ThreadPoolExecutor(max_workers=max(1, min(len(invocations), hooks.max_workers()))) as pool:
                results = list(pool.map(_attempt, [call for _, call in invocations]))
            for (record, _), failure in zip(invocations, results):
                if failure is not None:
                    site = record["call_site"]
                    outcome["model_failures"][site] = outcome["model_failures"].get(site, 0) + 1
            harness_errors = hooks.take_harness_errors()
            if harness_errors:
                outcome["harness_errors"][question_id] = harness_errors
                break  # a failed replay spends nothing more
            outcome["replayed"].add(question_id)
    finally:
        outcome["unregistered_call_sites"] = hooks.unregistered_call_sites()
        hooks.end_run()


def _attempt(call: Callable[[], Any]) -> Optional[str]:
    """The model's failure (already in C1), or None. A harness failure is not the model's: it has
    recorded itself, the question loop stops on it, and the calls after it are refused."""
    try:
        call()
        return None
    except (hooks.HarnessError, hooks.RunAborted):
        return None
    except Exception as e:
        return f"{type(e).__name__}"
