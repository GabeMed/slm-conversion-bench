"""The patched CHESS, end to end, on a synthetic database with a scripted model: every LLM call of
the real agent code reaches C1 with the right call site, question and invocation key; the agents'
actions are read with the agents' own rules; generation sees only the selected schema (D13).
Agent environment only; no server and no BIRD download needed."""
import re
import sqlite3

import pytest

pytest.importorskip("langchain_core")
from langchain_core.messages import AIMessage  # noqa: E402

from bench import paths  # noqa: E402
from bench.agent import hooks, runner  # noqa: E402
from bench.contracts.calls import read_calls, validate_calls  # noqa: E402
from bench.contracts.config import load_config  # noqa: E402

PLANS = {"Information Retriever": ["extract_keywords", "retrieve_entity", "retrieve_context"],
         "schema_selector": ["filter_column", "select_tables", "select_columns"],
         "Candidate Generator": ["generate_candidate", "revise"]}
SQL = "SELECT count(*) FROM gas_t WHERE country = 'CZE' AND segment = 'Premium'"


class ScriptedChess:
    """Answers each CHESS prompt the way a model that follows the protocol would."""

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
        if "is_column_information_relevant" in text:
            keep = "gas_t" in text.split("Column Info")[-1] if "Column Info" in text else "gas_t" in text
            return '{"chain_of_thought_reasoning": "x", "is_column_information_relevant": "%s"}' % ("Yes" if keep else "No")
        if '"table_names"' in text:
            return '{"chain_of_thought_reasoning": "x", "table_names": ["gas_t"]}'
        if '"table_name1"' in text:
            return '{"chain_of_thought_reasoning": "x", "gas_t": ["country", "segment"]}'
        if text.startswith("You are an experienced database expert"):
            return f"plan\n<FINAL_ANSWER>\n{SQL}\n</FINAL_ANSWER>"
        if "relevant keywords" in text:
            return '["gas stations", "CZE", "Premium"]'
        raise AssertionError(f"unexpected prompt: {text[:200]}")


@pytest.fixture
def chess_run(tmp_path, monkeypatch):
    db_root = tmp_path / "bird"
    db_dir = db_root / "dev_databases" / "tiny"
    (db_dir / "database_description").mkdir(parents=True)
    connection = sqlite3.connect(db_dir / "tiny.sqlite")
    connection.execute("CREATE TABLE gas_t (id INTEGER PRIMARY KEY, country TEXT, segment TEXT)")
    connection.execute("CREATE TABLE other_u (id INTEGER PRIMARY KEY, note TEXT)")
    connection.executemany("INSERT INTO gas_t VALUES (?, ?, ?)",
                           [(1, "CZE", "Premium"), (2, "CZE", "Value"), (3, "SVK", "Premium")])
    connection.execute("INSERT INTO other_u VALUES (1, 'unrelated')")
    connection.commit()
    connection.close()
    header = "original_column_name,column_name,column_description,data_format,value_description\n"
    (db_dir / "database_description" / "gas_t.csv").write_text(
        header + "id,id,station id,integer,\ncountry,country,country code,text,\nsegment,segment,price segment,text,\n")
    (db_dir / "database_description" / "other_u.csv").write_text(
        header + "id,id,note id,integer,\nnote,note,a note,text,\n")

    config = load_config(paths.ROOT / "configs" / "smoke-local.yaml")
    runner._prepare_chess(config, db_root)
    from database_utils.db_catalog.preprocess import make_db_context_vec_db
    from database_utils.db_values.preprocess import make_db_lsh
    make_db_lsh(str(db_dir), signature_size=20, n_gram=3, threshold=0.01, verbose=False)
    make_db_context_vec_db(str(db_dir), use_value_description=True)

    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset = [{"question_id": qid, "db_id": "tiny", "question": "How many gas stations in CZE are Premium?",
                "evidence": "", "SQL": SQL} for qid in (7, 8)]
    outcome = runner.execute_questions(config, "B0", "wiring", run_dir, dataset, db_root)
    return outcome, read_calls(run_dir / "calls.jsonl")


def test_the_agent_runs_end_to_end(chess_run):
    outcome, calls = chess_run
    assert outcome["predictions"] == {"7": f" {SQL} ", "8": f" {SQL} "}
    assert outcome["failures"] == {} and outcome["harness_errors"] == {}
    assert validate_calls(calls) == []
    assert {c["question_id"] for c in calls} == {"7", "8"}


def test_every_call_site_is_logged_under_its_own_name(chess_run):
    _, calls = chess_run
    per_question = [c for c in calls if c["question_id"] == "7"]
    assert {c["call_site"] for c in per_question} == {
        "agent_ir", "agent_ss", "agent_cg", "extract_keywords", "filter_column", "select_tables",
        "select_columns", "generate_candidate"}
    names = {"agent_ir": "Information Retriever", "agent_ss": "schema_selector", "agent_cg": "Candidate Generator"}
    for call in per_question:
        if call["call_site"] in names:
            assert f"You are the {names[call['call_site']]} agent" in call["prompt_messages"][0]["content"]


def test_agent_actions_are_read_with_the_agents_rules(chess_run):
    _, calls = chess_run
    ss = sorted((c for c in calls if c["question_id"] == "7" and c["call_site"] == "agent_ss"),
                key=lambda c: c["invocation_key"])
    assert [(c["invocation_key"], c["parsed_output"]) for c in ss] == [
        ("ss:0", {"tool": "filter_column"}), ("ss:1", {"tool": "select_tables"}),
        ("ss:2", {"tool": "select_columns"}), ("ss:3", {"done": True})]


def test_invocation_keys_identify_each_invocation(chess_run):
    _, calls = chess_run
    per_question = [c for c in calls if c["question_id"] == "7"]
    filtered = {c["invocation_key"] for c in per_question if c["call_site"] == "filter_column"}
    assert filtered == {"gas_t.id", "gas_t.country", "gas_t.segment", "other_u.id", "other_u.note"}
    assert [c["invocation_key"] for c in per_question if c["call_site"] == "generate_candidate"] == \
        ["generate_candidate_one:0"]
    assert {c["invocation_key"] for c in per_question if c["call_site"] == "extract_keywords"} == {"single"}


def test_generation_sees_only_the_selected_schema(chess_run):
    _, calls = chess_run
    (generation,) = [c for c in calls if c["question_id"] == "7" and c["call_site"] == "generate_candidate"]
    prompt = generation["prompt_messages"][0]["content"]
    assert "gas_t" in prompt and "other_u" not in prompt  # D13: the selector's output reaches generation


def test_the_agent_cannot_write_to_the_database(chess_run, tmp_path):
    from database_utils.execution import execute_sql
    db = tmp_path / "bird" / "dev_databases" / "tiny" / "tiny.sqlite"
    with pytest.raises(Exception, match="readonly"):
        execute_sql(str(db), "DELETE FROM gas_t")
    assert execute_sql(str(db), "SELECT count(*) FROM gas_t") == [(3,)]
