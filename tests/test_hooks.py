"""The patched call layer (hooks): one C1 line per attempt, bounded retry, no fallback. Agent env only."""
import json

import pytest

pytest.importorskip("langchain_core")
from langchain_core.exceptions import OutputParserException  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.output_parsers import JsonOutputParser  # noqa: E402

from bench import paths  # noqa: E402
from bench.agent import hooks  # noqa: E402
from bench.contracts.calls import read_calls, validate_calls  # noqa: E402
from bench.contracts.config import load_config  # noqa: E402

USAGE = {"prompt_tokens": 12, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 8}}


class ScriptedModel:
    """Returns the scripted outputs in order; an Exception in the script is raised."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return AIMessage(content=item, response_metadata={"token_usage": USAGE})


@pytest.fixture
def run(tmp_path, monkeypatch):
    config = load_config(paths.ROOT / "configs" / "smoke-local.yaml")
    hooks.configure(config)
    hooks.start_run("test-run", "B0", tmp_path / "calls.jsonl")
    hooks.set_question("1470")
    model = ScriptedModel([])
    monkeypatch.setattr(hooks, "_chat_model", lambda engine, temperature: model)
    yield model, tmp_path / "calls.jsonl"
    hooks.end_run()


def test_parse_retry_is_bounded_and_chained(run):
    model, calls_path = run
    model.script = ["not json", '{"table_names": ["t"]}']
    parsed = hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    assert parsed == {"table_names": ["t"]}
    first, second = read_calls(calls_path)
    assert validate_calls([first, second]) == []
    assert (first["parsed_ok"], second["parsed_ok"]) == (False, True)
    assert second["retry_of"] == first["call_id"] and second["attempt"] == 2
    assert first["usage"] == {"input": 12, "cached_input": 8, "output": 3, "source": "api"}
    assert first["temperature"] == 0.0 and first["endpoint"] == "llamacpp"


def test_exhausted_retries_raise_and_empty_output_is_not_a_fallback(run):
    model, calls_path = run
    model.script = ["", "still not json"]
    with pytest.raises(OutputParserException):
        hooks.invoke_tool_call("extract_keywords", "single", [HumanMessage(content="q")], JsonOutputParser())
    lines = read_calls(calls_path)
    assert [r["error"].split(":")[0] for r in lines] == ["empty output", "OutputParserException"]
    assert model.calls == 2 and validate_calls(lines) == []
    assert lines[0]["temperature"] == 0.2  # per call site, from the configuration


def test_invocation_error_is_logged_then_raised(run):
    model, calls_path = run
    model.script = [TimeoutError("read timed out")]
    with pytest.raises(TimeoutError):
        hooks.invoke_tool_call("filter_column", "schools.County", [HumanMessage(content="q")], JsonOutputParser())
    (line,) = read_calls(calls_path)
    assert line["response_text"] is None and line["usage"]["source"] == "missing"
    assert line["invocation_key"] == "schools.County" and validate_calls([line]) == []


def test_agent_call_records_the_action_it_selects(run):
    model, calls_path = run
    model.script = ["<tool_call>select_tables</tool_call>", "<tool_call>1. extract_keywords<tool_call>"]

    def parse(response):
        if "DONE" in response:
            return {"done": True}
        name = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        if name not in ("select_tables", "extract_keywords"):
            raise ValueError(f"Tool {name} not found")
        return {"tool": name}

    assert hooks.invoke_agent_call("agent_ss", "ss:0", "state", parse) == "<tool_call>select_tables</tool_call>"
    hooks.invoke_agent_call("agent_ir", "ir:0", "state", parse)  # returns the text; the agent raises itself
    ok, bad = read_calls(calls_path)
    assert ok["parsed_output"] == {"tool": "select_tables"}
    assert bad["parsed_ok"] is False and "not found" in bad["error"]
    assert validate_calls([ok, bad]) == []


def test_calls_need_a_run_and_a_question(tmp_path):
    hooks.configure(load_config(paths.ROOT / "configs" / "smoke-local.yaml"))
    with pytest.raises(RuntimeError, match="no run"):
        hooks.invoke_agent_call("agent_ir", "ir:0", "state", lambda r: {"done": True})
