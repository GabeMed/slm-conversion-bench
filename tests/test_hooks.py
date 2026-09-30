"""The patched call layer (hooks): what reaches the model, one C1 line per attempt, bounded retry,
no fallback, harness failures surfaced. Agent environment only."""
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
SMOKE = paths.ROOT / "configs" / "smoke-local.yaml"


class ScriptedModel:
    """Returns the scripted outputs in order; an Exception in the script is raised."""

    def __init__(self):
        self.script, self.calls = [], 0

    def invoke(self, messages):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return AIMessage(content=item, response_metadata={"token_usage": USAGE})


@pytest.fixture
def run(tmp_path, monkeypatch):
    hooks.configure(load_config(SMOKE))
    hooks.start_run("test-run", "B0", tmp_path / "calls.jsonl")
    hooks.set_question("1470")
    model = ScriptedModel()
    built = []
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: built.append((engine, temperature)) or model)
    yield model, tmp_path / "calls.jsonl", built
    hooks.end_run()


def test_parse_retry_is_bounded_and_chained(run):
    model, calls_path, built = run
    model.script = ["not json", '{"table_names": ["t"]}']
    parsed = hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    assert parsed == {"table_names": ["t"]}
    first, second = read_calls(calls_path)
    assert validate_calls([first, second]) == []
    assert (first["parsed_ok"], second["parsed_ok"]) == (False, True)
    assert second["retry_of"] == first["call_id"] and second["attempt"] == 2
    assert first["usage"] == {"input": 12, "cached_input": 8, "output": 3, "source": "api"}
    assert (first["engine"], first["endpoint"], first["cluster"]) == ("production_llm", "llamacpp", None)


def test_the_temperature_sent_is_the_call_sites(run):
    model, calls_path, built = run
    model.script = ['["a"]', '{"is_column_information_relevant": "Yes"}']
    hooks.invoke_tool_call("extract_keywords", "single", [HumanMessage(content="q")], JsonOutputParser())
    hooks.invoke_tool_call("filter_column", "t.c", [HumanMessage(content="q")], JsonOutputParser())
    assert built == [("production_llm", 0.2), ("production_llm", 0.0)]
    assert [r["temperature"] for r in read_calls(calls_path)] == [0.2, 0.0]


def test_exhausted_retries_raise_and_empty_output_is_not_a_fallback(run):
    model, calls_path, _ = run
    model.script = ["", "still not json"]
    with pytest.raises(OutputParserException):
        hooks.invoke_tool_call("extract_keywords", "single", [HumanMessage(content="q")], JsonOutputParser())
    lines = read_calls(calls_path)
    assert [r["error"].split(":")[0] for r in lines] == ["empty output", "OutputParserException"]
    assert model.calls == 2 and validate_calls(lines) == []


def test_invocation_error_is_logged_then_raised(run):
    model, calls_path, _ = run
    model.script = [TimeoutError("read timed out")]
    with pytest.raises(TimeoutError):
        hooks.invoke_tool_call("filter_column", "schools.County", [HumanMessage(content="q")], JsonOutputParser())
    (line,) = read_calls(calls_path)
    assert line["response_text"] is None and line["usage"]["source"] == "missing"
    assert line["invocation_key"] == "schools.County" and validate_calls([line]) == []
    assert hooks.take_harness_errors() == []  # the model's failure, not the harness's


def test_each_line_carries_the_current_question_and_repetitions_get_a_suffix(run):
    model, calls_path, _ = run
    model.script = ['{"table_names": ["t"]}'] * 3
    hooks.set_question("1474")
    hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    hooks.set_question("1475")
    hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    lines = read_calls(calls_path)
    assert [(r["question_id"], r["invocation_key"]) for r in lines] == [
        ("1474", "single"), ("1474", "single@2"), ("1475", "single")]
    assert validate_calls(lines) == []


def test_agent_call_records_the_action_it_selects(run):
    model, calls_path, _ = run
    model.script = ["<tool_call>select_tables</tool_call>", "<tool_call>1. extract_keywords<tool_call>"]

    def parse(response):
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


def test_harness_failures_are_raised_and_kept_for_the_runner(run):
    model, calls_path, _ = run
    hooks._run.arm = "B3"  # B3 needs a `choice` fact the smoke config does not set
    with pytest.raises(hooks.HarnessError):
        hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    assert model.calls == 0 and read_calls(calls_path) == []
    (error,) = hooks.take_harness_errors()
    assert "choice is not set" in error
    assert hooks.take_harness_errors() == []


def test_no_question_no_call(tmp_path):
    hooks.configure(load_config(SMOKE))
    hooks.start_run("r", "B0", tmp_path / "calls.jsonl")
    try:
        with pytest.raises(hooks.HarnessError, match="set_question"):
            hooks.invoke_agent_call("agent_ir", "ir:0", "state", lambda r: {"done": True})
        assert read_calls(tmp_path / "calls.jsonl") == []
    finally:
        hooks.end_run()
    with pytest.raises(hooks.HarnessError, match="no run"):
        hooks.invoke_agent_call("agent_ir", "ir:0", "state", lambda r: {"done": True})


# ---------------------------------------------------------------- the client itself

def test_the_client_is_built_as_configured_one_per_temperature():
    config = load_config(SMOKE)
    config["roles"]["production_llm"]["params"] = {"max_tokens": 77, "timeout_s": 123}
    hooks.configure(config)
    cold, warm = hooks.chat_model("production_llm", 0.0), hooks.chat_model("production_llm", 0.2)
    assert cold is not warm and cold is hooks.chat_model("production_llm", 0.0)
    assert (cold.temperature, warm.temperature) == (0.0, 0.2)
    assert cold.max_retries == 0  # no hidden client retry
    assert cold.model_name == config["roles"]["production_llm"]["model"]
    assert cold.openai_api_base == config["roles"]["production_llm"]["endpoint"]["base_url"]
    assert cold.request_timeout == config["roles"]["production_llm"]["params"]["timeout_s"]
    assert cold.max_tokens == config["roles"]["production_llm"]["params"]["max_tokens"]


def test_a_configured_key_that_is_missing_never_falls_back_to_openai(monkeypatch):
    config = load_config(SMOKE)
    config["roles"]["production_llm"]["endpoint"] = {"kind": "api", "base_url": "https://provider.example/v1",
                                                     "api_key_env": "PRODUCTION_LLM_API_KEY"}
    hooks.configure(config)
    monkeypatch.delenv("PRODUCTION_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-never-leave")
    with pytest.raises(hooks.HarnessError, match="PRODUCTION_LLM_API_KEY"):
        hooks.chat_model("production_llm", 0.0)
    monkeypatch.setenv("PRODUCTION_LLM_API_KEY", "provider-key")
    assert hooks.chat_model("production_llm", 0.0).openai_api_key.get_secret_value() == "provider-key"


# ---------------------------------------------------------------- harness failures are never lost

def test_a_harness_error_records_itself_even_when_swallowed(run):
    try:
        raise hooks.HarnessError(f"missing {paths.ROOT}/data/x")
    except Exception:
        pass  # what CHESS's handlers do
    assert hooks.take_harness_errors() == ["missing <repo>/data/x"]  # recorded, and scrubbed of the local path


def test_embedding_failures_are_harness_failures(run, monkeypatch):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    def broken(self, texts):
        raise ConnectionError("embeddings endpoint unreachable")
    monkeypatch.setattr(DeterministicFakeEmbedding, "embed_documents", broken)
    with pytest.raises(hooks.HarnessError):
        hooks.embeddings("entity").embed_documents(["a"])
    (error,) = hooks.take_harness_errors()
    assert "ConnectionError" in error


def test_openai_embeddings_without_a_key_are_refused(monkeypatch):
    config = load_config(SMOKE)
    config["embeddings"]["provider"] = "openai"
    hooks.configure(config)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(hooks.HarnessError, match="OPENAI_API_KEY"):
        hooks.embeddings("entity")


def test_model_error_text_is_scrubbed_of_local_paths(run):
    model, calls_path, _ = run
    model.script = [OSError(f"cannot open {paths.ROOT}/secret/file")]
    with pytest.raises(OSError):
        hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    (line,) = read_calls(calls_path)
    assert str(paths.ROOT) not in line["error"] and "<repo>/secret/file" in line["error"]
