"""`bench run` end to end on a synthetic repository, driving the real patched CHESS with a scripted
model: every LLM call reaches C1 with the right call site, question and invocation key; the agents'
actions are read with the agents' own rules; generation sees only the selected schema (D13); and a
run is `done` only when nothing went wrong. Agent environment only; no server, no BIRD download."""
import json
import re
import shutil
import sqlite3

import pytest

pytest.importorskip("langchain_core")
from langchain_core.messages import AIMessage  # noqa: E402

from bench import paths  # noqa: E402
from bench.agent import hooks, runner  # noqa: E402
from bench.contracts.calls import read_calls, validate_calls  # noqa: E402
from bench.contracts.config import config_sha256  # noqa: E402
from bench.contracts.facts import FactError  # noqa: E402
from bench.data import DataError  # noqa: E402
from bench.evaluate import evaluate  # noqa: E402
from synthetic import GOLD, make_repo  # noqa: E402

PLANS = {"Information Retriever": ["extract_keywords", "retrieve_entity", "retrieve_context"],
         "schema_selector": ["filter_column", "select_tables", "select_columns"],
         "Candidate Generator": ["generate_candidate", "revise"]}


class ScriptedChess:
    """Answers each CHESS prompt the way a model that follows the protocol would; `overrides`
    replaces the answer for one kind of prompt, `on_generate` runs a side effect."""

    def __init__(self, overrides=None, on_generate=None):
        self.overrides, self.on_generate = overrides or {}, on_generate

    def invoke(self, messages):
        text = messages[-1].content
        return AIMessage(content=self.answer(text), response_metadata={
            "token_usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 5}})

    def answer(self, text):
        agent = re.search(r"You are the (.+?) agent, in a team", text)
        if agent:
            done = text.count("<agent>\n<tool_call>")
            plan = PLANS[agent.group(1)]
            return f"<tool_call>{plan[done]}</tool_call>" if done < len(plan) else "DONE"
        for kind, marker in (("filter_column", "is_column_information_relevant"), ("select_tables", '"table_names"'),
                             ("select_columns", '"table_name1"'), ("keywords", "relevant keywords")):
            if marker in text and kind in self.overrides:
                return self.overrides[kind]
        if "is_column_information_relevant" in text:
            return '{"chain_of_thought_reasoning": "x", "is_column_information_relevant": "Yes"}'
        if '"table_names"' in text:
            return '{"chain_of_thought_reasoning": "x", "table_names": ["gas_t"]}'
        if '"table_name1"' in text:
            return '{"chain_of_thought_reasoning": "x", "gas_t": ["country", "segment"]}'
        if text.startswith("You are an experienced database expert"):
            if self.on_generate:
                self.on_generate()
            return f"plan\n<FINAL_ANSWER>\n{GOLD}\n</FINAL_ANSWER>"
        if "relevant keywords" in text:
            return '["gas stations", "CZE", "Premium"]'
        raise AssertionError(f"unexpected prompt: {text[:200]}")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root, config_path, config = make_repo(tmp_path, monkeypatch)
    runner.preprocess(str(config_path), ["tiny"])
    return config_path, config


def run(monkeypatch, repo, model=None, ids=("1", "2"), arm="B0"):
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: model or ScriptedChess())
    run_dir = runner.run_agent(str(repo[0]), arm, "train", ids=list(ids))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return run_dir, manifest, read_calls(run_dir / "calls.jsonl")


def test_a_clean_run_is_done_and_scores(monkeypatch, repo):
    run_dir, manifest, calls = run(monkeypatch, repo)
    assert manifest["status"] == "done", manifest
    assert json.loads((run_dir / "predictions.json").read_text()) == {"1": f" {GOLD} ", "2": f" {GOLD} "}
    assert config_sha256(json.loads((run_dir / "config.json").read_text())) == manifest["config_sha256"]
    assert manifest["facts"] == {} and manifest["c1_errors"] == 0 == len(validate_calls(calls))
    assert {c["question_id"] for c in calls} == {"1", "2"}
    eval_manifest = json.loads((evaluate(run_dir.name) / "manifest.json").read_text())
    assert (eval_manifest["n"], eval_manifest["correct"]) == (2, 2)


def test_every_call_site_is_logged_under_its_own_name(monkeypatch, repo):
    _, _, calls = run(monkeypatch, repo)
    per_question = [c for c in calls if c["question_id"] == "1"]
    assert {c["call_site"] for c in per_question} == {
        "agent_ir", "agent_ss", "agent_cg", "extract_keywords", "filter_column", "select_tables",
        "select_columns", "generate_candidate"}
    names = {"agent_ir": "Information Retriever", "agent_ss": "schema_selector", "agent_cg": "Candidate Generator"}
    for call in per_question:
        if call["call_site"] in names:
            assert f"You are the {names[call['call_site']]} agent" in call["prompt_messages"][0]["content"]


def test_agent_actions_are_read_with_the_agents_rules(monkeypatch, repo):
    _, _, calls = run(monkeypatch, repo)
    ss = sorted((c for c in calls if c["question_id"] == "1" and c["call_site"] == "agent_ss"),
                key=lambda c: c["invocation_key"])
    assert [(c["invocation_key"], c["parsed_output"]) for c in ss] == [
        ("ss:0", {"tool": "filter_column"}), ("ss:1", {"tool": "select_tables"}),
        ("ss:2", {"tool": "select_columns"}), ("ss:3", {"done": True})]


def test_invocation_keys_identify_each_invocation(monkeypatch, repo):
    _, _, calls = run(monkeypatch, repo)
    per_question = [c for c in calls if c["question_id"] == "1"]
    filtered = {c["invocation_key"] for c in per_question if c["call_site"] == "filter_column"}
    assert filtered == {"gas_t.id", "gas_t.country", "gas_t.segment", "other_u.id", "other_u.note"}
    assert [c["invocation_key"] for c in per_question if c["call_site"] == "generate_candidate"] == \
        ["generate_candidate_one:0"]


def test_generation_sees_only_the_selected_schema(monkeypatch, repo):
    _, _, calls = run(monkeypatch, repo)
    (generation,) = [c for c in calls if c["question_id"] == "1" and c["call_site"] == "generate_candidate"]
    prompt = generation["prompt_messages"][0]["content"]
    assert "gas_t" in prompt and "other_u" not in prompt  # D13: the selector's output reaches generation


# ---------------------------------------------------------------- a run is done only when nothing went wrong

def test_a_harness_failure_inside_chess_fails_the_run(monkeypatch, repo):
    real_route = hooks.route

    def route(arm, call_site, messages, config):
        if call_site == "filter_column":
            raise FactError("arms.B5.allocation is not set in the configuration")
        return real_route(arm, call_site, messages, config)
    monkeypatch.setattr(hooks, "route", route)
    run_dir, manifest, _ = run(monkeypatch, repo)
    assert manifest["status"] == "failed" and "allocation" in manifest["harness_errors"]["1"][0]
    assert list(json.loads((run_dir / "predictions.json").read_text())) == ["1"]  # stopped: 2 never ran


def test_embedding_failures_fail_the_run(monkeypatch, repo):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    def unreachable(self, texts):
        raise ConnectionError("embeddings endpoint unreachable")
    monkeypatch.setattr(DeterministicFakeEmbedding, "embed_documents", unreachable)
    _, manifest, _ = run(monkeypatch, repo, ids=("1",))
    assert manifest["status"] == "failed" and "ConnectionError" in json.dumps(manifest["harness_errors"])


def test_a_database_changed_during_the_run_fails_it(monkeypatch, repo):
    def tamper():
        connection = sqlite3.connect(paths.sqlite_path(repo[1], "tiny"))
        connection.execute("INSERT INTO other_u VALUES (2, 'written during the run')")
        connection.commit()
        connection.close()
    _, manifest, _ = run(monkeypatch, repo, model=ScriptedChess(on_generate=tamper), ids=("1",))
    assert (manifest["status"], manifest["databases_changed"]) == ("failed", ["tiny"])


def test_tool_errors_are_reported_without_failing_the_run(monkeypatch, repo):
    model = ScriptedChess(overrides={"select_tables": "the tables are gas_t, I think"})  # never parses
    _, manifest, calls = run(monkeypatch, repo, model=model, ids=("1",))
    assert manifest["status"] == "done"  # the model's doing: the agent carries on, as published
    assert "select_tables" in manifest["tool_errors"]["1"]
    attempts = [c for c in calls if c["call_site"] == "select_tables"]
    assert [c["attempt"] for c in attempts] == [1, 2] and not any(c["parsed_ok"] for c in attempts)


def test_preconditions_are_checked_before_a_run_starts(monkeypatch, repo):
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    with pytest.raises(FactError, match="choice is not set"):
        runner.run_agent(str(repo[0]), "B3", "train", ids=["1"])
    vector_db = paths.bird_root(repo[1]) / "dev_databases" / "tiny" / "context_vector_db_fake"
    shutil.rmtree(vector_db)
    with pytest.raises(DataError, match="not preprocessed"):
        runner.run_agent(str(repo[0]), "B0", "train", ids=["1"])
    assert not paths.RUNS.exists() or not any(paths.RUNS.iterdir())


def test_the_agent_cannot_write_to_the_database(monkeypatch, repo):
    run(monkeypatch, repo, ids=("1",))
    from database_utils.execution import execute_sql
    db = paths.sqlite_path(repo[1], "tiny")
    with pytest.raises(Exception, match="readonly"):
        execute_sql(str(db), "DELETE FROM gas_t")
    assert execute_sql(str(db), "SELECT count(*) FROM gas_t") == [(3,)]


def test_an_unbuildable_engine_stops_the_run_before_it_starts(repo, monkeypatch):
    import yaml
    config = yaml.safe_load(repo[0].read_text())
    config["roles"]["production_llm"]["endpoint"]["api_key_env"] = "BENCH_TEST_MISSING_KEY"
    repo[0].write_text(yaml.safe_dump(config))
    monkeypatch.delenv("BENCH_TEST_MISSING_KEY", raising=False)
    with pytest.raises(hooks.HarnessError, match="BENCH_TEST_MISSING_KEY"):
        runner.run_agent(str(repo[0]), "B0", "train", ids=["1"])
    assert not paths.RUNS.exists() or not any(paths.RUNS.iterdir())


def test_each_run_points_chess_at_its_own_databases(repo, tmp_path):
    from synthetic import make_db
    other_root = tmp_path / "other"
    make_db(other_root / "dev_databases" / "tiny")
    runner._prepare_chess(repo[1], paths.bird_root(repo[1]))
    from runner.database_manager import DatabaseManager
    assert str(DatabaseManager("dev", "tiny").db_path).startswith(str(paths.bird_root(repo[1])))
    runner._prepare_chess(repo[1], other_root)
    assert str(DatabaseManager("dev", "tiny").db_path).startswith(str(other_root))


def test_a_question_that_raises_fails_the_run_and_stops_it(monkeypatch, repo):
    def broken(state):
        raise KeyError("SQL_meta_infos")
    monkeypatch.setattr(runner, "final_sql", broken)
    run_dir, manifest, _ = run(monkeypatch, repo)
    assert manifest["status"] == "failed" and "KeyError" in manifest["failures"]["1"]
    assert list(json.loads((run_dir / "predictions.json").read_text())) == ["1"]


def test_invalid_c1_lines_fail_the_run(monkeypatch, repo):
    monkeypatch.setattr(runner, "validate_calls", lambda calls: ["line 1: broken"])
    _, manifest, _ = run(monkeypatch, repo, ids=("1",))
    assert (manifest["status"], manifest["c1_errors"]) == ("failed", 1)


def test_an_interrupted_run_keeps_what_finished(monkeypatch, repo):
    class Interrupting(ScriptedChess):
        def invoke(self, messages):
            if messages[-1].content.startswith("<system>") and hooks._run.question_id == "2":
                raise KeyboardInterrupt  # on the agent's own call, in the main thread, as a Ctrl-C would
            return super().invoke(messages)
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: Interrupting())
    with pytest.raises(KeyboardInterrupt):
        runner.run_agent(str(repo[0]), "B0", "train", ids=["1", "2"])
    (run_dir,) = list(paths.RUNS.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert (manifest["status"], manifest["stopped_by"]) == ("interrupted", "KeyboardInterrupt: ")
    assert json.loads((run_dir / "predictions.json").read_text()) == {"1": f" {GOLD} "}


def test_openai_embeddings_without_a_key_stop_the_run_before_it_starts(repo, monkeypatch):
    import yaml
    config = yaml.safe_load(repo[0].read_text())
    config["embeddings"]["provider"] = "openai"
    repo[0].write_text(yaml.safe_dump(config))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(hooks.HarnessError, match="OPENAI_API_KEY"):
        runner.run_agent(str(repo[0]), "B0", "train", ids=["1"])


def test_preprocessing_must_match_the_configuration(repo, monkeypatch):
    import yaml
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    stamp = paths.bird_root(repo[1]) / "dev_databases" / "tiny" / "preprocessed" / "STAMP.json"
    stamp.unlink()
    with pytest.raises(DataError, match="preprocessed/STAMP.json"):
        runner.run_agent(str(repo[0]), "B0", "train", ids=["1"])
    runner.preprocess(str(repo[0]), ["tiny"])
    config = yaml.safe_load(repo[0].read_text())
    config["embeddings"]["fake_size"] = 32  # another embedding: the vector DB must be rebuilt
    repo[0].write_text(yaml.safe_dump(config))
    with pytest.raises(DataError, match="context_vector_db_fake/STAMP.json"):
        runner.run_agent(str(repo[0]), "B0", "train", ids=["1"])
    runner.preprocess(str(repo[0]), ["tiny"])
    _, manifest, _ = run(monkeypatch, repo, ids=("1",))
    assert manifest["status"] == "done"
