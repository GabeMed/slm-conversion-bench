"""The patched call layer (hooks): what reaches the model, one C1 line per attempt, bounded retry,
no fallback, harness failures surfaced. Agent environment only."""
import os

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
    """Returns the scripted outputs in order (a text, or a whole message with its metadata); an
    Exception in the script is raised."""

    def __init__(self):
        self.script, self.calls = [], 0

    def invoke(self, messages):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, AIMessage):
            return item
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
    assert first["usage"] == {"input": 12, "cached_input": 8, "output": 3, "reasoning": None, "source": "api"}
    assert (first["engine"], first["endpoint"], first["cluster"]) == ("production_llm", "llamacpp", None)


def test_the_temperature_sent_is_the_call_sites(run):
    model, calls_path, built = run
    model.script = ['["a"]', '{"is_column_information_relevant": "Yes"}']
    hooks.invoke_tool_call("extract_keywords", "single", [HumanMessage(content="q")], JsonOutputParser())
    hooks.invoke_tool_call("filter_column", "t.c", [HumanMessage(content="q")], JsonOutputParser())
    assert built == [("production_llm", 0.2), ("production_llm", 0.0)]
    assert [r["temperature"] for r in read_calls(calls_path)] == [0.2, 0.0]


def test_a_forced_temperature_is_what_c1_records_while_the_call_sites_is_still_sent(run):
    """Where the configuration declares that the provider forces a temperature (a reasoning mode that
    ignores the one sent), C1 records the value in effect; the request is the call site's, unchanged."""
    model, calls_path, built = run
    hooks._config["roles"]["production_llm"]["reasoning"]["forced_temperature"] = 1.0
    model.script = ['["a"]', '{"is_column_information_relevant": "Yes"}']
    hooks.invoke_tool_call("extract_keywords", "single", [HumanMessage(content="q")], JsonOutputParser())
    hooks.invoke_tool_call("filter_column", "t.c", [HumanMessage(content="q")], JsonOutputParser())
    assert built == [("production_llm", 0.2), ("production_llm", 0.0)]
    lines = read_calls(calls_path)
    assert [r["temperature"] for r in lines] == [1.0, 1.0] and validate_calls(lines) == []


def test_c1_records_how_the_answer_ended_its_reasoning_the_model_that_served_it_and_the_pinned_provider(run):
    """A cut-off answer is visible: `finish_reason` and the reasoning tokens (a part of `output`) are on
    the line, with the model the response names and the provider the configuration pins. Each is null
    when the provider does not say, and when nothing answered."""
    import httpx
    import openai
    model, calls_path, _ = run
    hooks._config["roles"]["production_llm"]["provider"]["name"] = "a-provider"
    cut = AIMessage(content="", response_metadata={
        "token_usage": {"prompt_tokens": 12, "completion_tokens": 4096, "completion_tokens_details": {"reasoning_tokens": 4096}},
        "finish_reason": "length", "model_name": "served-model-2026-09"})
    rejected = openai.BadRequestError("rejected", body=None,
                                      response=httpx.Response(400, request=httpx.Request("POST", "http://x")))
    model.script = [cut, '{"table_names": ["t"]}', rejected]
    hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    with pytest.raises(openai.BadRequestError):
        hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    truncated, silent, failed = read_calls(calls_path)
    assert (truncated["finish_reason"], truncated["model_reported"], truncated["provider"]) == \
        ("length", "served-model-2026-09", "a-provider")
    assert truncated["usage"] == {"input": 12, "cached_input": None, "output": 4096, "reasoning": 4096, "source": "api"}
    assert truncated["error"] == "empty output"  # still the model's empty output, but now told apart by the line
    assert (silent["finish_reason"], silent["model_reported"], silent["usage"]["reasoning"]) == (None, None, None)
    assert (failed["finish_reason"], failed["model_reported"], failed["usage"]["reasoning"]) == (None, None, None)
    assert silent["provider"] == failed["provider"] == "a-provider"  # the configuration's, whatever answered
    assert validate_calls([truncated, silent, failed]) == []


def test_what_the_api_answers_reaches_c1_through_the_agents_own_client(tmp_path):
    """On the wire, through the pinned LangChain and OpenAI client (no scripted message): the stop
    reason, the reasoning tokens and the model named by the response are on the C1 line, the forced
    temperature is what it records, and the call site's temperature is what was sent."""
    pytest.importorskip("langchain_openai")
    from test_preflight import FakeEngine
    config = load_config(SMOKE)
    with FakeEngine(content='{"table_names": ["t"]}', finish_reason="length", reasoning=7) as server:
        config["roles"]["production_llm"]["endpoint"]["base_url"] = server.base_url
        config["roles"]["production_llm"]["reasoning"] = {"enabled": True, "forced_temperature": 1.0}
        config["roles"]["production_llm"]["provider"]["name"] = "a-provider"
        hooks.configure(config)
        hooks.start_run("test-run", "B0", tmp_path / "calls.jsonl")
        try:
            hooks.set_question("1470")
            hooks.invoke_tool_call("extract_keywords", "single", [HumanMessage(content="q")], JsonOutputParser())
        finally:
            hooks.end_run()
    (line,) = read_calls(tmp_path / "calls.jsonl")
    assert (line["finish_reason"], line["model_reported"], line["provider"]) == \
        ("length", f"{config['roles']['production_llm']['model']}-as-served", "a-provider")
    assert line["usage"] == {"input": 12, "cached_input": 0, "output": 8, "reasoning": 7, "source": "api"}
    assert line["temperature"] == 1.0 and server.requests[0]["body"]["temperature"] == 0.2
    assert validate_calls([line]) == []


def test_an_engine_with_no_pinned_provider_records_none(run, monkeypatch):
    model, calls_path, _ = run
    monkeypatch.setattr(hooks, "route", lambda arm, call_site, messages, config: hooks.Route("slm:qwen3-8b", None))
    model.script = ['{"table_names": ["t"]}']
    hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    (line,) = read_calls(calls_path)
    assert line["model_role"] == "slm" and line["provider"] is None and validate_calls([line]) == []


def test_exhausted_retries_raise_and_empty_output_is_not_a_fallback(run):
    model, calls_path, _ = run
    model.script = ["", "still not json"]
    with pytest.raises(OutputParserException):
        hooks.invoke_tool_call("extract_keywords", "single", [HumanMessage(content="q")], JsonOutputParser())
    lines = read_calls(calls_path)
    assert [r["error"].split(":")[0] for r in lines] == ["empty output", "OutputParserException"]
    assert model.calls == 2 and validate_calls(lines) == []


def test_invocation_error_is_logged_then_raised(run):
    import httpx
    import openai
    model, calls_path, _ = run
    rejected = openai.BadRequestError("context length exceeded", body=None,
                                      response=httpx.Response(400, request=httpx.Request("POST", "http://x")))
    model.script = [rejected]  # the engine rejects this request: the model's failure (a closed set, F1 5b)
    with pytest.raises(openai.BadRequestError):
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


def test_the_reasoning_setting_and_the_routing_pin_are_part_of_the_request():
    """`params.extra_body` (a provider's reasoning switch) and `provider.routing` (an aggregator's pin)
    go out as the request's extra body, the named parameters as themselves; what the harness is told
    about the engine (`reasoning`, `provider`) is never sent."""
    config = load_config(SMOKE)
    routing = {"only": ["a-provider"], "allow_fallbacks": False, "require_parameters": True, "quantizations": ["fp8"]}
    config["roles"]["production_llm"]["params"] = {"max_tokens": 77, "timeout_s": 123, "reasoning_effort": "medium",
                                                   "extra_body": {"thinking": {"type": "enabled"}}}
    config["roles"]["production_llm"]["provider"]["routing"] = routing
    hooks.configure(config)
    model = hooks.chat_model("production_llm", 0.0)
    assert model.extra_body == {"thinking": {"type": "enabled"}, "provider": routing}
    assert model.model_kwargs == {"reasoning_effort": "medium"}
    assert config["roles"]["production_llm"]["params"]["extra_body"] == {"thinking": {"type": "enabled"}}  # not mutated
    assert hooks.chat_model("cheap_alt", 0.0).extra_body is None  # nothing declared, nothing sent


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
    model.script = [OSError(f"cannot open {paths.ROOT}/secret/file")]  # unrecognised: the harness's failure
    with pytest.raises(hooks.HarnessError):
        hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    (line,) = read_calls(calls_path)
    assert str(paths.ROOT) not in line["error"] and "<repo>/secret/file" in line["error"]
    (recorded,) = hooks.take_harness_errors()
    assert str(paths.ROOT) not in recorded and "<repo>/secret/file" in recorded


# ---------------------------------------------------------------- integration: headers, content, environment

def test_headers_env_are_sent_and_a_missing_one_is_a_harness_error(monkeypatch):
    config = load_config(SMOKE)
    config["roles"]["production_llm"]["endpoint"]["headers_env"] = {"Modal-Key": "BENCH_TEST_MODAL_KEY"}
    hooks.configure(config)
    monkeypatch.delenv("BENCH_TEST_MODAL_KEY", raising=False)
    with pytest.raises(hooks.HarnessError, match="BENCH_TEST_MODAL_KEY"):
        hooks.chat_model("production_llm", 0.0)
    monkeypatch.setenv("BENCH_TEST_MODAL_KEY", "k-123")
    assert hooks.chat_model("production_llm", 0.0).default_headers == {"Modal-Key": "k-123"}


def test_headers_env_must_map_header_names_to_variables():
    from bench.contracts.config import validate_config
    config = load_config(SMOKE)
    for bad in ({"Modal-Key": ""}, ["Modal-Key"], {"": "VAR"}):
        config["roles"]["production_llm"]["endpoint"]["headers_env"] = bad
        assert any("headers_env" in e for e in validate_config(config))


def test_content_that_is_not_text_is_a_harness_failure_with_its_line(run):
    model, calls_path, _ = run
    model.script = []
    model.invoke = lambda messages: AIMessage(content=[{"type": "text", "text": "{}"}],
                                              response_metadata={"token_usage": USAGE})
    with pytest.raises(hooks.HarnessError, match="list content"):
        hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    (line,) = read_calls(calls_path)
    assert line["response_text"] is None and "not text" in line["error"] and validate_calls([line]) == []
    assert line["usage"]["source"] == "api" and line["usage"]["input"] == 12  # answered and billed


def test_a_providers_error_that_echoes_a_key_is_recorded_without_it(run, monkeypatch):
    import httpx
    import openai
    model, calls_path, _ = run
    monkeypatch.setenv("OPENAI_API_KEY", "embeddings-key-0001")  # a credential of every configuration (C2)

    def refuse(messages):
        raise openai.BadRequestError("bad request from embeddings-key-0001", body=None,
                                     response=httpx.Response(400, request=httpx.Request("POST", "http://x")))
    model.script = []
    model.invoke = refuse
    with pytest.raises(openai.BadRequestError):
        hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    (line,) = read_calls(calls_path)
    assert "embeddings-key-0001" not in line["error"] and "<redacted>" in line["error"]


def test_the_agent_package_switches_tracing_off_and_runs_drop_chroma_servers(monkeypatch, tmp_path):
    import importlib
    import bench.agent
    from bench.agent import runner
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    importlib.reload(bench.agent)
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    for name in ("DB_ROOT_PATH", "INDEX_SERVER_PORT", "ANONYMIZED_TELEMETRY", *runner.TRACING_OFF):
        value = os.environ.get(name)  # restored after the test, whatever _prepare_chess sets; unset stays unset
        monkeypatch.setenv(name, value if value is not None else "unset")
        if value is None:
            monkeypatch.delenv(name)
    monkeypatch.setattr(hooks, "_config", hooks._config)
    monkeypatch.setenv("CHROMA_SERVER_HOST", "vectors.example.com")
    monkeypatch.setenv("chroma_server_ssl_enabled", "true")  # Chroma reads its variables in any case
    runner._prepare_chess(load_config(SMOKE), tmp_path)
    assert "CHROMA_SERVER_HOST" not in os.environ and "chroma_server_ssl_enabled" not in os.environ


def test_a_run_stopped_by_another_worker_makes_no_further_model_call(tmp_path, monkeypatch):
    """`bench run --workers`: the workers of a run share one flag. Once it is set (a question failed
    in any of them), a call is refused before the model is asked, as after this process's own failure:
    no C1 line, and nothing recorded as this worker's harness error."""
    import threading
    stop = threading.Event()
    hooks.configure(load_config(SMOKE))
    hooks.start_run("test-run", "B0", tmp_path / "calls.w1.jsonl", stop=stop)
    hooks.set_question("1470")
    model = ScriptedModel()
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: model)
    try:
        model.script = ['{"table_names": ["t"]}'] * 2
        hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
        stop.set()
        with pytest.raises(hooks.RunAborted, match="failed in another worker"):
            hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
        assert model.calls == 1 and len(read_calls(tmp_path / "calls.w1.jsonl")) == 1
        assert hooks.take_harness_errors() == []
    finally:
        hooks.end_run()
