"""What the patched CHESS calls into (vendor/chess/PATCHES.md). Runs only in the agent environment.

Every LLM invocation of the agent goes through `invoke_tool_call` (tools) or `invoke_agent_call`
(the agent's choice of the next tool). Both ask the router (C4) for the engine, invoke it with
no hidden retry, and write one C1 line per attempt, failures included.

Two kinds of failure, kept apart:
- the **model's** (an invocation error, an empty or unparseable output) is a C1 line, and the
  exception goes on to CHESS, which handles it as it always did;
- the **harness's** (no run or question set, routing, configuration, a missing API key) is
  recorded here as well, because CHESS swallows every exception, and the runner reads it with
  `take_harness_errors()` to fail the run instead of recording it as done.

State is process-global: one run and one question at a time per process. The tools of a question
call the model from several threads; the writer and the counters are shared under a lock.
"""
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from bench.contracts.config import engine_spec
from bench.contracts.router import Route, route

_config: Optional[Dict[str, Any]] = None
_run: Optional["_RunState"] = None
_models: Dict[tuple, Any] = {}
_models_lock = threading.Lock()


class HarnessError(RuntimeError):
    """A failure of the harness, not of the model: the run must not be recorded as done."""


@dataclass
class _RunState:
    run_id: str
    arm: str
    calls_path: Path
    question_id: Optional[str] = None
    occurrences: Dict[tuple, int] = field(default_factory=dict)
    harness_errors: List[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class RoutedEngine:
    """What `get_llm_chain` returns: a placeholder, since the engine is chosen per call."""

    def __init__(self, engine_name: str):
        self.engine_name = engine_name


def configure(config: Dict[str, Any]) -> None:
    global _config
    _config = config
    _models.clear()


def start_run(run_id: str, arm: str, calls_path: Path) -> None:
    global _run
    if _config is None:
        raise HarnessError("hooks.configure() must run before start_run()")
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    calls_path.touch()
    _run = _RunState(run_id=run_id, arm=arm, calls_path=calls_path)


def set_question(question_id: str) -> None:
    run = _require_run()
    with run.lock:
        run.question_id = str(question_id)
        run.occurrences.clear()


def take_harness_errors() -> List[str]:
    """The harness errors since the last call (the runner calls it after each question)."""
    run = _require_run()
    with run.lock:
        errors, run.harness_errors = run.harness_errors, []
    return errors


def end_run() -> None:
    global _run
    _run = None


def _require_run() -> _RunState:
    if _run is None:
        raise HarnessError("no run in progress: the harness runner must call hooks.start_run()")
    return _run


def _harness(what: str, fn: Callable[[], Any]) -> Any:
    """Run a harness step; a failure is recorded for the runner, then raised."""
    try:
        return fn()
    except Exception as e:
        message = f"{what}: {type(e).__name__}: {e}"
        if _run is not None:
            with _run.lock:
                _run.harness_errors.append(message)
        raise HarnessError(message) from e


def _begin(call_site: str, invocation_key: str, messages: List[Dict[str, str]]):
    """Checks and decisions made before the model is called: question, identity, route."""
    run = _require_run()
    if run.question_id is None:
        raise HarnessError("hooks.set_question() was not called for this question")
    with run.lock:
        n = run.occurrences.get((call_site, invocation_key), 0) + 1
        run.occurrences[(call_site, invocation_key)] = n
    key = invocation_key if n == 1 else f"{invocation_key}@{n}"
    chosen = route(run.arm, call_site, messages, _config)
    temperature = float(_config["call_sites"][call_site]["temperature"])
    return key, chosen, temperature, chat_model(chosen.engine, temperature)


# ---------------------------------------------------------------- models

def chat_model(engine: str, temperature: float):
    """The client for one engine at one temperature, built once. A configured API key that is not
    set is an error: an empty key would make the client fall back to OPENAI_API_KEY."""
    key = (engine, temperature)
    with _models_lock:
        if key not in _models:
            from langchain_openai import ChatOpenAI

            spec = engine_spec(_config, engine)
            params = dict(spec.get("params") or {})
            endpoint = spec["endpoint"]
            if not endpoint.get("base_url"):
                raise HarnessError(f"engine {engine} has no endpoint.base_url in the configuration")
            api_key_env = endpoint.get("api_key_env")
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if not api_key:
                    raise HarnessError(f"engine {engine}: environment variable {api_key_env} is not set")
            else:
                api_key = "EMPTY"  # local servers (llama.cpp, vLLM) accept any key
            _models[key] = ChatOpenAI(
                model=spec["model"],
                openai_api_base=endpoint["base_url"],
                openai_api_key=api_key,
                temperature=temperature,
                max_tokens=params.pop("max_tokens", None),
                timeout=params.pop("timeout_s"),
                max_retries=0,  # every attempt is visible in C1; no hidden client retry
                model_kwargs=params,
            )
        return _models[key]


def _message_dicts(messages: List[Any]) -> List[Dict[str, str]]:
    roles = {"human": "user", "ai": "assistant", "system": "system"}
    return [{"role": roles[m.type], "content": m.content} for m in messages]


def _usage(output: Any) -> Dict[str, Any]:
    token_usage = (getattr(output, "response_metadata", None) or {}).get("token_usage") or {}
    if token_usage.get("prompt_tokens") is None or token_usage.get("completion_tokens") is None:
        return {"input": None, "cached_input": None, "output": None, "source": "missing"}
    details = token_usage.get("prompt_tokens_details") or {}
    return {
        "input": token_usage["prompt_tokens"],
        "cached_input": details.get("cached_tokens"),
        "output": token_usage["completion_tokens"],
        "source": "api",
    }


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _invoke(model: Any, messages: List[Any]):
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    try:
        output, exception = model.invoke(messages), None
    except Exception as e:  # the model's failure: recorded, then re-raised by the caller
        output, exception = None, e
    return output, exception, started_at, int((time.perf_counter() - t0) * 1000)


def _record(*, call_id, retry_of, attempt, call_site, invocation_key, chosen: Route, messages,
            started_at, latency_ms, temperature, output, parsed, parsed_ok, error) -> None:
    run = _require_run()
    spec = engine_spec(_config, chosen.engine)
    line = json.dumps({
        "run_id": run.run_id, "call_id": call_id, "retry_of": retry_of, "attempt": attempt,
        "question_id": run.question_id, "call_site": call_site, "invocation_key": invocation_key,
        "cluster": chosen.cluster, "engine": chosen.engine, "model_role": spec["model_role"],
        "model": spec["model"], "endpoint": spec["endpoint"]["kind"],
        "prompt_messages": messages,
        "response_text": output.content if output is not None else None,
        "parsed_output": _jsonable(parsed) if parsed_ok else None, "parsed_ok": parsed_ok,
        "usage": _usage(output) if output is not None else
        {"input": None, "cached_input": None, "output": None, "source": "missing"},
        "latency_ms": latency_ms, "started_at": started_at, "temperature": temperature, "error": error,
    }, ensure_ascii=False)
    with run.lock, open(run.calls_path, "a") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------- calls

def invoke_tool_call(call_site: str, invocation_key: str, lc_messages: List[Any], parser: Any) -> Any:
    """One tool call: route, invoke, parse; retry an empty or unparseable output up to the configured cap."""
    from langchain_core.exceptions import OutputParserException

    messages = _message_dicts(lc_messages)
    key, chosen, temperature, model = _harness(call_site, lambda: _begin(call_site, invocation_key, messages))
    max_attempts = _config["retries"]["parse_max_attempts"]
    retry_of = None
    for attempt in range(1, max_attempts + 1):
        call_id = str(uuid.uuid4())
        output, exception, started_at, latency_ms = _invoke(model, lc_messages)
        parsed, parsed_ok, error, retryable = None, False, None, False
        if exception is not None:
            error = f"{type(exception).__name__}: {exception}"
        elif not output.content.strip():
            error, retryable = "empty output", True
        else:
            try:
                parsed, parsed_ok = parser.invoke(output), True
            except OutputParserException as e:
                error, retryable = f"OutputParserException: {e}", True
            except Exception as e:  # not retried, as in CHESS
                error, exception = f"{type(e).__name__}: {e}", e
        _harness(call_site, lambda: _record(
            call_id=call_id, retry_of=retry_of, attempt=attempt, call_site=call_site, invocation_key=key,
            chosen=chosen, messages=messages, started_at=started_at, latency_ms=latency_ms,
            temperature=temperature, output=output, parsed=parsed, parsed_ok=parsed_ok, error=error))
        if parsed_ok:
            return parsed
        if exception is not None:
            raise exception
        if not retryable or attempt == max_attempts:
            raise OutputParserException(error)
        retry_of = call_id


def invoke_agent_call(call_site: str, invocation_key: str, message: str, parse: Callable[[str], Any]) -> str:
    """The agent's next-tool choice: one attempt, as in CHESS. `parse` is the agent's own reading of the response."""
    from langchain_core.messages import HumanMessage

    lc_messages = [HumanMessage(content=message)]
    messages = _message_dicts(lc_messages)
    key, chosen, temperature, model = _harness(call_site, lambda: _begin(call_site, invocation_key, messages))
    call_id = str(uuid.uuid4())
    output, exception, started_at, latency_ms = _invoke(model, lc_messages)
    parsed, parsed_ok, error = None, False, None
    if exception is not None:
        error = f"{type(exception).__name__}: {exception}"
    else:
        try:
            parsed, parsed_ok = parse(output.content), True
        except Exception as e:  # the agent itself raises on the same response right after
            error = f"{type(e).__name__}: {e}"
    _harness(call_site, lambda: _record(
        call_id=call_id, retry_of=None, attempt=1, call_site=call_site, invocation_key=key, chosen=chosen,
        messages=messages, started_at=started_at, latency_ms=latency_ms, temperature=temperature,
        output=output, parsed=parsed, parsed_ok=parsed_ok, error=error))
    if exception is not None:
        raise exception
    return output.content


# ---------------------------------------------------------------- retrieval embeddings

def embeddings(purpose: str):
    """Embeddings for CHESS retrieval: `entity` (retrieve_entity) or `context` (column-description vector DB)."""
    settings = _config["embeddings"]
    provider = settings["provider"]
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        if not os.environ.get("OPENAI_API_KEY"):
            raise HarnessError("embeddings.provider is openai and OPENAI_API_KEY is not set")
        return OpenAIEmbeddings(model=settings[f"{purpose}_model"])
    if provider == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=settings["fake_size"])
    raise NotImplementedError(f"embeddings provider {provider!r} is built in F1")


def vector_db_dirname() -> str:
    """One vector DB per provider, so a DB built with one embedding is never queried with another."""
    return f"context_vector_db_{_config['embeddings']['provider']}"
