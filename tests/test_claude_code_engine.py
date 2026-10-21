"""K1 through Claude Code (`bench run --via-claude-code`): the CHESS prompt on stdin of `claude -p` with
everything switchable switched off, the API's key never passed on, the answer and its usage mapped as the
Messages API engine maps them, failures classified, K1 only. A fake `claude`: nothing is sent. Agent
environment only."""
import json
import stat
import sys

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("anthropic")
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_core.output_parsers import JsonOutputParser  # noqa: E402

from bench import data, paths  # noqa: E402
from bench.agent import claude_code_engine, hooks  # noqa: E402
from bench.agent.runner import check_batch  # noqa: E402
from bench.contracts.calls import read_calls, validate_calls  # noqa: E402
from bench.contracts.config import load_config  # noqa: E402

CONFIG = paths.ROOT / "config.yaml"
SMOKE = paths.ROOT / "configs" / "smoke-local.yaml"

FAKE = r'''#!{python}
"""A fake `claude -p`: records its argv, stdin and environment, prints the answer the test wrote."""
import json, os, sys
from pathlib import Path
here = Path(__file__).parent
(here / "seen.json").write_text(json.dumps({{"argv": sys.argv[1:], "stdin": sys.stdin.read(), "cwd": os.getcwd(),
    "env": {{k: os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "DISABLE_PROMPT_CACHING", "DISABLE_COMPACT", "CLAUDE_CODE_ATTRIBUTION_HEADER",
    "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS", "CLAUDE_CODE_DISABLE_CLAUDE_MDS")}}}}))
sys.stdout.write((here / "answer.txt").read_text())
'''


def _answer(text='{"keywords": ["k"]}', **over):
    answer = {"type": "result", "subtype": "success", "is_error": False, "result": text, "stop_reason": "end_turn",
              "api_error_status": None, "modelUsage": {"claude-sonnet-5-5": {}},
              "usage": {"input_tokens": 2, "cache_creation_input_tokens": 383, "cache_read_input_tokens": 6061,
                        "output_tokens": 123, "output_tokens_details": {"thinking_tokens": 7}}}
    answer.update(over)
    return json.dumps(answer)


@pytest.fixture
def claude(monkeypatch, tmp_path):
    """Via Claude Code on, a fake `claude` in tmp_path; yields (model, write the next answer, what it saw)."""
    fake = tmp_path / "claude"
    fake.write_text(FAKE.format(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(claude_code_engine.CLAUDE_BIN_ENV, str(fake))
    monkeypatch.setenv(claude_code_engine.VIA_ENV, "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(claude_code_engine, "WORKDIR", tmp_path / "cwd")

    def no_network(*args, **kwargs):
        raise AssertionError("a Claude Code test reached the Messages API")
    monkeypatch.setattr("anthropic.resources.messages.Messages.stream", no_network)
    hooks.configure(load_config(CONFIG))
    model = hooks.chat_model("production_llm", 1.0)
    (tmp_path / "answer.txt").write_text(_answer())
    yield (model, lambda text: (tmp_path / "answer.txt").write_text(text),
           lambda: json.loads((tmp_path / "seen.json").read_text()))
    hooks.configure(load_config(SMOKE))


def test_the_prompt_goes_on_stdin_with_everything_switchable_switched_off(claude, tmp_path):
    model, _, seen = claude
    prompt = "<system>\nHuman: extract the keywords\nColumn information:\nthe question"
    answer = model.invoke([HumanMessage(content=prompt)])
    got = seen()
    assert got["stdin"] == prompt  # the cache split joined back: the CHESS text exactly
    argv = got["argv"]
    assert argv[:4] == ["-p", "--model", "claude-sonnet-5-5", "--effort"] and argv[4] == "high"
    for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in argv
    for flag in ("--system-prompt", "--tools", "--setting-sources"):
        assert argv[argv.index(flag) + 1] == ""
    assert argv[argv.index("--output-format") + 1] == "json"
    assert got["cwd"] == str(tmp_path / "cwd")
    env = got["env"]
    assert env["ANTHROPIC_API_KEY"] is None and env["ANTHROPIC_BASE_URL"] is None  # the subscription, never the key
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32000"
    assert (env["DISABLE_COMPACT"], env["CLAUDE_CODE_ATTRIBUTION_HEADER"]) == ("1", "0")
    assert env["DISABLE_PROMPT_CACHING"] == "1"  # every CHESS prompt is unique: a cache write is never read
    assert env["CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS"] == env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert answer.content == '{"keywords": ["k"]}'
    usage = answer.response_metadata["token_usage"]
    assert usage["prompt_tokens"] == 2 + 383 + 6061 and usage["completion_tokens"] == 123
    assert usage["prompt_tokens_details"] == {"cached_tokens": 6061, "cache_write_tokens": 383}
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 7}
    assert answer.response_metadata["finish_reason"] == "stop"
    assert answer.response_metadata["model_name"] == "claude-sonnet-5-5"


def test_a_system_prompt_is_refused(claude):
    model, _, _ = claude
    with pytest.raises(claude_code_engine.ClaudeCodeError, match="no system prompt") as e:
        model.invoke([SystemMessage(content="s"), HumanMessage(content="u")])
    assert hooks.classify(e.value) == "harness"


@pytest.mark.parametrize("over, classified", [
    ({"is_error": True, "subtype": "error", "api_error_status": 429}, "transport"),
    ({"is_error": True, "subtype": "error", "api_error_status": 529}, "transport"),
    ({"is_error": True, "subtype": "error", "api_error_status": 400}, "model"),
    ({"is_error": True, "subtype": "error", "api_error_status": 401}, "harness"),
    ({"is_error": True, "subtype": "error", "api_error_status": None}, "transport"),  # the usage limit
])
def test_a_failed_answer_is_classified_as_the_apis_own_status(claude, over, classified):
    model, answer, _ = claude
    answer(_answer(**over))
    with pytest.raises(claude_code_engine.ClaudeCodeError) as e:
        model.invoke([HumanMessage(content="u")])
    assert hooks.classify(e.value) == classified


def test_the_subscriptions_limit_is_waited_out_and_the_call_sent_again(claude, tmp_path, monkeypatch, capsys):
    model, answer, _ = claude
    limit = _answer(is_error=True, subtype="error", api_error_status=429,
                    result="You've hit your session limit · resets 9:20pm (America/Sao_Paulo)")
    answer(limit)
    waits = []

    def sleep(seconds):  # the limit lifts after the second pause
        waits.append(seconds)
        if len(waits) == 2:
            answer(_answer())
    monkeypatch.setattr(claude_code_engine.time, "sleep", sleep)
    out = model.invoke([HumanMessage(content="u")])
    assert out.content == '{"keywords": ["k"]}' and waits == [claude_code_engine.LIMIT_WAIT_S] * 2
    assert "hit your session limit" in capsys.readouterr().err


def test_a_429_that_is_not_the_limit_is_an_ordinary_transport_failure(claude, monkeypatch):
    model, answer, _ = claude
    answer(_answer(is_error=True, subtype="error", api_error_status=429, result="rate_limit_error"))
    monkeypatch.setattr(claude_code_engine.time, "sleep", lambda s: pytest.fail("waited on an ordinary 429"))
    with pytest.raises(claude_code_engine.ClaudeCodeError) as e:
        model.invoke([HumanMessage(content="u")])
    assert hooks.classify(e.value) == "transport"


def test_no_json_on_stdout_is_a_transport_failure(claude):
    model, answer, _ = claude
    answer("not json")
    with pytest.raises(claude_code_engine.ClaudeCodeError) as e:
        model.invoke([HumanMessage(content="u")])
    assert hooks.classify(e.value) == "transport"


def test_a_call_through_claude_code_reaches_c1_as_every_call_does(claude, tmp_path):
    hooks.start_run("test-run", "B0", tmp_path / "calls.jsonl")
    try:
        hooks.set_question("1470")
        hooks.invoke_tool_call("extract_keywords", "single", [HumanMessage(content="q")], JsonOutputParser())
    finally:
        hooks.end_run()
    (line,) = read_calls(tmp_path / "calls.jsonl")
    assert line["parsed_ok"] and line["usage"]["cached_input"] == 6061 and line["usage"]["cache_write"] == 383
    assert line["prompt_messages"] == [{"role": "user", "content": "q"}]  # the CHESS prompt only
    assert validate_calls([line]) == []


@pytest.mark.parametrize("arm, split", [("B0", "calib"), ("B0", "test"), ("B1", "train")])
def test_via_claude_code_is_refused_outside_k1(arm, split):
    with pytest.raises(data.DataError, match="--via-claude-code is for K1 only"):
        check_batch(load_config(CONFIG), arm, split, "--via-claude-code")
