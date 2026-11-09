"""An API engine on Anthropic's Messages API (`endpoint.protocol: anthropic`): the request it sends, the
answer in the shape C1 reads, and its failures classified as every engine's. Agent environment only."""
import copy

import pytest

pytest.importorskip("langchain_core")
anthropic = pytest.importorskip("anthropic")
import httpx  # noqa: E402
from anthropic.types import Message  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_core.output_parsers import JsonOutputParser  # noqa: E402

from bench import paths  # noqa: E402
from bench.agent import hooks  # noqa: E402
from bench.agent.anthropic_engine import AnthropicChat  # noqa: E402
from bench.contracts.calls import read_calls, validate_calls  # noqa: E402
from bench.contracts.config import load_config, validate_config  # noqa: E402

CLAUDE = paths.ROOT / "configs" / "teacher-claude.yaml"
SMOKE = paths.ROOT / "configs" / "smoke-local.yaml"


def _message(text='{"keywords": ["k"]}', stop_reason="end_turn", thinking=40):
    return Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 60, "cache_read_input_tokens": 30,
                  "cache_creation_input_tokens": 5, "output_tokens_details": {"thinking_tokens": thinking}},
    })


class _Stream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


@pytest.fixture
def claude(monkeypatch):
    """The Claude configuration, with the SDK's stream answering `answer` and recording each request."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = load_config(CLAUDE)
    hooks.configure(config)
    sent, answer = [], {"message": _message()}

    def stream(self, **kwargs):
        sent.append(kwargs)
        return _Stream(answer["message"])
    monkeypatch.setattr(anthropic.resources.messages.Messages, "stream", stream)
    yield config, sent, answer
    hooks.configure(load_config(SMOKE))


def test_the_configuration_loads_and_the_protocol_is_only_for_api_endpoints():
    config = load_config(CLAUDE)
    assert config["roles"]["production_llm"]["endpoint"]["protocol"] == "anthropic"
    bad = copy.deepcopy(config)
    bad["roles"]["production_llm"]["endpoint"]["protocol"] = "grpc"
    assert any("protocol" in e for e in validate_config(bad))
    bad = copy.deepcopy(config)
    bad["roles"]["slm_candidates"][0]["endpoint"]["protocol"] = "anthropic"
    assert any("protocol" in e for e in validate_config(bad))


def test_the_request_carries_no_temperature_and_its_system_prompt_apart(claude):
    config, sent, _ = claude
    model = hooks.chat_model("production_llm", 0.2)
    assert isinstance(model, AnthropicChat) and model.max_retries == 0
    model.invoke([SystemMessage(content="be terse"), HumanMessage(content="q")])
    (request,) = sent
    assert "temperature" not in request
    assert request["system"] == "be terse"
    assert request["messages"] == [{"role": "user", "content": "q"}]
    assert (request["model"], request["max_tokens"]) == ("claude-opus-5-5", 32000)
    assert request["output_config"] == {"effort": "medium"}
    assert "timeout_s" not in request and "cache_breaks" not in request
    assert "cache_breaks" in config["roles"]["production_llm"]["params"]  # not mutated


def test_a_missing_key_is_an_error_not_a_fallback(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    hooks.configure(load_config(CLAUDE))
    try:
        with pytest.raises(hooks.HarnessError, match="ANTHROPIC_API_KEY"):
            hooks.chat_model("production_llm", 0.0)
    finally:
        hooks.configure(load_config(SMOKE))


def test_the_answer_reaches_c1_as_every_engines_does(claude, tmp_path):
    """Usage counts the cache in the input as OpenAI's does, the thinking tokens are the reasoning, the
    stop reason is `stop`, and the temperature recorded is the forced one."""
    hooks.start_run("test-run", "B0", tmp_path / "calls.jsonl")
    try:
        hooks.set_question("1470")
        hooks.invoke_tool_call("extract_keywords", "single", [HumanMessage(content="q")], JsonOutputParser())
    finally:
        hooks.end_run()
    (line,) = read_calls(tmp_path / "calls.jsonl")
    assert line["usage"] == {"input": 135, "cached_input": 30, "cache_write": 5, "output": 60, "reasoning": 40,
                             "source": "api"}
    assert (line["finish_reason"], line["model_reported"], line["provider"]) == ("stop", "claude-opus-5-5", "anthropic")
    assert line["response_text"] == '{"keywords": ["k"]}' and line["parsed_ok"]
    assert (line["temperature"], line["endpoint"]) == (1.0, "api")
    assert validate_calls([line]) == []


@pytest.mark.parametrize("stop_reason, recorded", [("max_tokens", "length"), ("refusal", "refusal")])
def test_a_cut_off_or_refused_answer_is_named_as_such(claude, stop_reason, recorded):
    _, _, answer = claude
    answer["message"] = _message(text="", stop_reason=stop_reason)
    output = hooks.chat_model("production_llm", 0.0).invoke([HumanMessage(content="q")])
    assert output.response_metadata["finish_reason"] == recorded


def test_the_sdks_failures_are_classified_as_every_engines():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def status(code):
        return anthropic.APIStatusError("x", response=httpx.Response(code, request=request), body=None)
    assert hooks.classify(status(529)) == "transport"  # overloaded
    assert hooks.classify(status(429)) == "transport"
    assert hooks.classify(status(400)) == "harness"  # a 400 not about the prompt's length
    assert hooks.classify(status(401)) == "harness"
    assert hooks.classify(anthropic.APIConnectionError(request=request)) == "transport"
    broke = anthropic.BadRequestError(
        "Error code: 400 - Your credit balance is too low to access the Anthropic API.",
        response=httpx.Response(400, request=request), body=None)
    assert hooks.classify(broke) == "harness"  # an empty balance stops the run, never scores the model


def _rejected(message, code=400):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError(message, response=httpx.Response(code, request=request), body=None)


@pytest.mark.parametrize("message, code, classified", [
    ("Error code: 400 - This API key is not scoped to a workspace, so this request must include the "
     "anthropic-workspace-id header", 400, "harness"),
    ("Error code: 400 - Your credit balance is too low to access the Anthropic API.", 400, "harness"),
    ("Error code: 400 - prompt is too long: 250000 tokens > 200000 maximum", 400, "model"),
    ("Error code: 400 - {'error': {'message': \"This model's maximum context length is 8192 tokens\", "
     "'code': 'context_length_exceeded'}}", 400, "model"),
    ("Error code: 400 - {'error': {'code': 'context_length_exceeded'}}", 400, "model"),
    ("Error code: 400 - the request exceeds the available context size", 400, "model"),
    ("Error code: 400 - invalid_request_error: temperature: unexpected", 400, "harness"),
    ("Error code: 400 - model: claude-nope is not supported", 400, "harness"),
    ("Error code: 413 - request too large", 413, "model"),
    ("Error code: 400 - max_tokens: 64000 > 32000, which is the maximum allowed number of output tokens",
     400, "harness"),
    ("Error code: 400 - input length and `max_tokens` exceed context limit: 190000 + 64000 > 200000", 400, "harness"),
    ("Error code: 400 - max_completion_tokens is too large: exceeds the context window", 400, "harness"),
    ("Error code: 400 - This model's maximum context length is 8192 tokens. However, you requested 9000 tokens "
     "(8000 in the messages, 1000 in the completion).", 400, "model"),
    ("Error code: 400 - too many input tokens", 400, "model"),
])
def test_only_a_400_about_the_prompts_length_is_the_models_failure(message, code, classified):
    """A 400 about the account, the key, a parameter or the model name stops the run: nothing is scored."""
    assert hooks.classify(_rejected(message, code)) == classified


def test_an_openai_400_with_code_context_length_exceeded_is_the_models_failure():
    import openai
    request = httpx.Request("POST", "http://x/v1/chat/completions")
    e = openai.BadRequestError("Error code: 400", response=httpx.Response(400, request=request),
                               body={"code": "context_length_exceeded", "message": "too many"})
    assert hooks.classify(e) == "model"


def test_the_workspace_header_comes_from_the_environment(monkeypatch):
    """A user key (sk-ant-usr-...) must name its workspace; unset, the client is built as before."""
    def built(**over):
        return AnthropicChat(model="m", base_url="https://api.anthropic.com", api_key="k", max_tokens=1,
                             timeout=1.0, **over).client.default_headers
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    assert "anthropic-workspace-id" not in built()
    assert "anthropic-workspace-id" not in built(default_headers={"x-a": "1"})
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "")
    assert "anthropic-workspace-id" not in built()
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_01abc")
    assert built()["anthropic-workspace-id"] == "wrkspc_01abc"
    headers = built(default_headers={"x-a": "1"})
    assert headers["anthropic-workspace-id"] == "wrkspc_01abc" and headers["x-a"] == "1"


def test_the_harness_built_client_sends_the_workspace_header(claude, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_01abc")
    hooks.configure(claude[0])  # a fresh client cache
    client = hooks.chat_model("production_llm", 0.0).client
    assert client.default_headers["anthropic-workspace-id"] == "wrkspc_01abc"


def test_the_workspace_id_is_redacted_from_records(monkeypatch):
    from bench.provenance import redact
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_01abcdefgh")
    assert "wrkspc_01abcdefgh" not in redact("workspace wrkspc_01abcdefgh is archived", load_config(CLAUDE))


# ---------------------------------------------------------------- prompt caching

FILTER = ("Instructions and examples.\nColumn information:\nTable name: `t`\n\nColumn information:\n"
          "Original column name: `c`\n\nQuestion:\nhow many?")


def test_the_long_prompt_is_split_after_the_last_marker_and_the_prefix_is_cached(claude):
    _, sent, _ = claude
    hooks.chat_model("production_llm", 0.0).invoke([HumanMessage(content=FILTER)])
    (message,) = sent[0]["messages"]
    first, second = message["content"]
    assert first == {"type": "text", "text": FILTER[:FILTER.rfind("Column information:\n") + 20],
                     "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    assert second == {"type": "text", "text": "Original column name: `c`\n\nQuestion:\nhow many?"}
    assert first["text"] + second["text"] == FILTER  # the same bytes, in two blocks


def test_a_prompt_with_no_marker_is_sent_as_it_is(claude):
    _, sent, _ = claude
    hooks.chat_model("production_llm", 0.0).invoke([HumanMessage(content="no marker here")])
    assert sent[0]["messages"] == [{"role": "user", "content": "no marker here"}]


def test_the_first_marker_in_the_list_wins_at_its_last_occurrence():
    from bench.agent.anthropic_engine import _cache_split
    text = "a【Table creation statements】\nQuestion: x【Table creation statements】\nschema\nQuestion: q"
    (turn,), prefix = _cache_split([{"role": "user", "content": text}],
                                   ["Column information:\n", "【Table creation statements】\n", "Question: "])
    assert turn["content"][1]["text"] == "schema\nQuestion: q" and prefix is not None


def test_the_first_call_of_a_prefix_goes_alone_and_the_others_wait_for_it(monkeypatch):
    """Twenty calls with one prefix sent at once: the first is in flight alone; the others start only after
    it ends, so they read the cache it wrote instead of each writing it."""
    import threading
    import time
    from bench.agent import anthropic_engine
    monkeypatch.setattr(anthropic_engine, "_WARMUP", anthropic_engine._Warmup())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    hooks.configure(load_config(CLAUDE))
    lock, state = threading.Lock(), {"in_flight": 0, "log": []}

    def stream(self, **kwargs):
        with lock:
            state["in_flight"] += 1
            state["log"].append(state["in_flight"])
        time.sleep(0.05)
        with lock:
            state["in_flight"] -= 1
        return _Stream(_message())
    monkeypatch.setattr(anthropic.resources.messages.Messages, "stream", stream)
    model = hooks.chat_model("production_llm", 0.0)
    try:
        threads = [threading.Thread(target=model.invoke, args=([HumanMessage(content=FILTER + str(i))],))
                   for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        hooks.configure(load_config(SMOKE))
    assert state["log"][:2] == [1, 1] and len(state["log"]) == 20  # the second starts after the first ended
    assert max(state["log"]) > 1  # after the first, the rest run in parallel again


def test_a_failed_first_call_releases_the_others(monkeypatch):
    from bench.agent import anthropic_engine
    warmup = anthropic_engine._Warmup()
    assert warmup.enter("p", 1.0) is True
    warmup.leave("p")  # what invoke's `finally` does when the call raised
    assert warmup.enter("p", 1.0) is False
