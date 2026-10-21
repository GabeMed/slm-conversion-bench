"""K1 through Claude Code (`bench run --via-claude-code`, K1 only): the teacher's calls sent by the
`claude` CLI in headless mode, on the author's Claude subscription, instead of the Messages API.

Anthropic allowed this for this project only (SPEC D25). Each call runs `claude -p` once, with the CHESS
prompt on stdin and everything Claude Code can switch off switched off: its system prompt (replaced by an
empty one), its tools, settings, MCP servers, skills, CLAUDE.md files, git instructions, attribution
header, compaction and prompt caching. `max_tokens` and the effort are the role's. What Claude Code
still adds, with no documented switch, was measured through a logging proxy before the first run (SPEC
D25): the line "You are a Claude agent, built on Anthropic's Claude Agent SDK." as the system prompt, an
environment block (the working directory, fixed here; the platform; the model; the date) and a reminder
with the account's email: 518 to 541 input tokens a call. C1 records the CHESS prompt only, which is what the SLMs are trained on.

- **No prompt caching:** Claude Code places its cache breakpoint at the end of the message, and every
  CHESS prompt is unique, so each call wrote its whole prompt to the cache (at twice the input price)
  and almost never read it: 13.2M tokens written, 0.2M read in the probes. That spent the subscription's
  session limit about twice as fast as no cache does.
- **The answer** is the JSON `claude -p --output-format json` prints: `result` is the text, `usage` the
  API's own (the same fields as a Messages API answer, mapped as `anthropic_engine` maps them),
  `stop_reason` the API's.
- **No credential of the API is passed on:** the child's environment drops `ANTHROPIC_API_KEY`,
  `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_BASE_URL`, so Claude Code uses the subscription's login, never
  the paid key.
- **Errors:** a failed answer with the API's status is classified as the Messages API's would be (408,
  429 and >= 500 transport; 400 and 413 model); without a status (the subscription's usage limit, a CLI
  that did not start) it is transport, retried after the backoff; a timeout is transport.
- **The subscription's usage limit** (a 429 saying "You've hit your ... limit") is not a failure of the
  call: it waits `LIMIT_WAIT_S` and sends it again, for as long as the limit lasts, so one K1 run spans as
  many usage windows as it needs. No attempt is recorded for a refused call (it reached no model); each
  pause is printed on stderr.
- **Concurrency:** at most `CLAUDE_CODE_PARALLEL_ENV` (default 8) `claude` processes at once per
  process: each is a Node runtime, and one CHESS step sends up to `agent.max_workers` calls together.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

VIA_ENV = "BENCH_VIA_CLAUDE_CODE"
CLAUDE_CODE_PARALLEL_ENV = "BENCH_CLAUDE_CODE_PARALLEL"
CLAUDE_BIN_ENV = "BENCH_CLAUDE_BIN"  # the `claude` executable (tests point it at a fake)
WORKDIR = Path.home() / ".cache" / "slm-bench" / "claude-code-cwd"  # empty, the same for every call

DROPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
SWITCHES_ENV = {"DISABLE_PROMPT_CACHING": "1", "DISABLE_COMPACT": "1", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1", "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1"}
FLAGS = ["--safe-mode", "--system-prompt", "", "--tools", "", "--setting-sources", "", "--strict-mcp-config",
         "--disable-slash-commands", "--no-session-persistence", "--output-format", "json"]
FINISH_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}
LIMIT = re.compile(r"hit your .*limit", re.IGNORECASE)
LIMIT_WAIT_S = 600.0  # between two tries while the subscription's limit lasts

_slots: Optional[threading.Semaphore] = None
_slots_lock = threading.Lock()


class ClaudeCodeError(Exception):
    """A call Claude Code did not answer, with the kind `hooks.classify` gives it."""

    def __init__(self, message: str, classified_as: str):
        super().__init__(message)
        self.classified_as = classified_as


def _semaphore() -> threading.Semaphore:
    global _slots
    with _slots_lock:
        if _slots is None:
            _slots = threading.Semaphore(int(os.environ.get(CLAUDE_CODE_PARALLEL_ENV) or 8))
        return _slots


def prompt_text(kwargs: Dict[str, Any]) -> str:
    """The one user message of a CHESS call; anything else (a system prompt, several turns) is refused:
    Claude Code would send it in another shape than the Messages API engine does."""
    turns = kwargs["messages"]
    if kwargs.get("system") is not None or len(turns) != 1 or turns[0]["role"] != "user":
        raise ClaudeCodeError("--via-claude-code sends one user message and no system prompt", "harness")
    content = turns[0]["content"]
    if isinstance(content, list):  # a cache split: the blocks joined back, Claude Code places its own cache
        content = "".join(block["text"] for block in content)
    return content


def command(kwargs: Dict[str, Any]) -> List[str]:
    effort = (kwargs.get("output_config") or {}).get("effort")
    return ([os.environ.get(CLAUDE_BIN_ENV) or "claude", "-p", "--model", kwargs["model"]]
            + (["--effort", effort] if effort else []) + FLAGS)


def child_env(kwargs: Dict[str, Any]) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in DROPPED_ENV}
    env.update(SWITCHES_ENV)
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(kwargs["max_tokens"])
    return env


def _classify_failure(answer: Dict[str, Any]) -> str:
    status = answer.get("api_error_status")
    if isinstance(status, int):
        if status in (408, 429) or status >= 500:
            return "transport"
        if status in (400, 413):
            return "model"
        return "harness"
    return "transport"  # no status: the subscription's usage limit, or the CLI's own failure; retried


def response_metadata(answer: Dict[str, Any], model: str) -> Dict[str, Any]:
    usage = answer["usage"]
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
    reported = list(answer.get("modelUsage") or {})
    return {
        "token_usage": {
            "prompt_tokens": usage["input_tokens"] + cache_read + cache_write,
            "completion_tokens": usage["output_tokens"],
            "prompt_tokens_details": {"cached_tokens": cache_read, "cache_write_tokens": cache_write},
            "completion_tokens_details": {"reasoning_tokens": thinking},
        },
        "finish_reason": FINISH_REASONS.get(answer.get("stop_reason"), answer.get("stop_reason")),
        "model_name": reported[0] if len(reported) == 1 else model,
    }


def at_limit(answer: Dict[str, Any]) -> bool:
    """The subscription's usage limit refused the call (a session's or a week's)."""
    return bool(answer.get("is_error")) and answer.get("api_error_status") == 429 and \
        bool(LIMIT.search(str(answer.get("result") or "")))


def call(kwargs: Dict[str, Any], timeout: float):
    """One teacher call through `claude -p`: (the answer's text, its response metadata), sent again
    after a pause while the subscription's usage limit refuses it."""
    while True:
        answer = _once(kwargs, timeout)
        if not at_limit(answer):
            break
        print(f"claude -p: {answer.get('result')}; trying again in {LIMIT_WAIT_S:.0f} s", file=sys.stderr, flush=True)
        time.sleep(LIMIT_WAIT_S)
    if answer.get("is_error") or answer.get("subtype") != "success":
        raise ClaudeCodeError(f"claude -p failed ({answer.get('api_error_status')}): "
                              f"{str(answer.get('result'))[:300]}", _classify_failure(answer))
    return answer["result"], response_metadata(answer, kwargs["model"])


def _once(kwargs: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """`claude -p` run once: the JSON it printed."""
    prompt = prompt_text(kwargs)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    with _semaphore():
        try:
            done = subprocess.run(command(kwargs), input=prompt, capture_output=True, text=True,
                                  cwd=WORKDIR, env=child_env(kwargs), timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise ClaudeCodeError(f"claude -p took more than {timeout:.0f} s", "transport") from e
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError:
        raise ClaudeCodeError(f"claude -p exited {done.returncode} with no answer: {done.stderr.strip()[:300]}",
                              "transport") from None
