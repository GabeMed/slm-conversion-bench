"""What the patched CHESS calls into (vendor/chess/PATCHES.md). Runs only in the agent environment.

Every LLM invocation of the agent goes through `invoke_tool_call` (tools) or `invoke_agent_call`
(the agent's choice of the next tool). Both ask the router (C4) for the engine, invoke it with
no hidden retry, and write one C1 line per attempt, failures included.

State is process-global: one run and one question at a time per process. The tools of a question
call the model from several threads, and the writer is shared under a lock.
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
from bench.contracts.router import route

_config: Optional[Dict[str, Any]] = None
_run: Optional["_RunState"] = None
_models: Dict[tuple, Any] = {}
_models_lock = threading.Lock()


@dataclass
class _RunState:
    run_id: str
    arm: str
    calls_path: Path
    question_id: Optional[str] = None
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
        raise RuntimeError("hooks.configure() must run before start_run()")
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    calls_path.touch()
    _run = _RunState(run_id=run_id, arm=arm, calls_path=calls_path)


def set_question(question_id: str) -> None:
    _require_run().question_id = str(question_id)


def end_run() -> None:
    global _run
    _run = None


def _require_run() -> _RunState:
    if _run is None:
        raise RuntimeError("no run in progress: the harness runner must call hooks.start_run()")
    return _run


# ---------------------------------------------------------------- models

def _chat_model(engine: str, temperature: float):
    key = (engine, temperature)
    with _models_lock:
        if key not in _models:
            from langchain_openai import ChatOpenAI

            spec = engine_spec(_config, engine)
            params = dict(spec.get("params") or {})
            endpoint = spec["endpoint"]
            if not endpoint.get("base_url"):
                raise RuntimeError(f"engine {engine} has no endpoint.base_url in the configuration")
            api_key_env = endpoint.get("api_key_env")
            _models[key] = ChatOpenAI(
                model=spec["model"],
                openai_api_base=endpoint["base_url"],
                openai_api_key=os.environ.get(api_key_env, "") if api_key_env else "EMPTY",
                temperature=temperature,
                max_tokens=params.pop("max_tokens", None),
                timeout=params.pop("timeout_s", 600),
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


def _invoke(engine: str, temperature: float, messages: List[Any]):
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    try:
        output, exception = _chat_model(engine, temperature).invoke(messages), None
    except Exception as e:  # recorded, then re-raised by the caller
        output, exception = None, e
    return output, exception, started_at, int((time.perf_counter() - t0) * 1000)


def _write(record: Dict[str, Any]) -> None:
    run = _require_run()
    line = json.dumps(record, ensure_ascii=False)
    with run.lock, open(run.calls_path, "a") as fh:
        fh.write(line + "\n")


def _record(*, call_id, retry_of, attempt, call_site, invocation_key, engine, messages, started_at,
            latency_ms, temperature, output, parsed, parsed_ok, error) -> None:
    run = _require_run()
    if run.question_id is None:
        raise RuntimeError("hooks.set_question() was not called for this question")
    spec = engine_spec(_config, engine)
    _write({
        "run_id": run.run_id, "call_id": call_id, "retry_of": retry_of, "attempt": attempt,
        "question_id": run.question_id, "call_site": call_site, "invocation_key": invocation_key,
        "cluster": None, "engine": f"{engine}:{spec['model']}", "model_role": spec["model_role"],
        "model": spec["model"], "endpoint": spec["endpoint"]["kind"],
        "prompt_messages": messages,
        "response_text": output.content if output is not None else None,
        "parsed_output": _jsonable(parsed) if parsed_ok else None, "parsed_ok": parsed_ok,
        "usage": _usage(output) if output is not None else
        {"input": None, "cached_input": None, "output": None, "source": "missing"},
        "latency_ms": latency_ms, "started_at": started_at, "temperature": temperature, "error": error,
    })


# ---------------------------------------------------------------- calls

def invoke_tool_call(call_site: str, invocation_key: str, lc_messages: List[Any], parser: Any) -> Any:
    """One tool call: route, invoke, parse; retry an empty or unparseable output up to the configured cap."""
    from langchain_core.exceptions import OutputParserException

    run = _require_run()
    messages = _message_dicts(lc_messages)
    engine = route(run.arm, call_site, messages, _config)
    temperature = float(_config["call_sites"][call_site]["temperature"])
    max_attempts = _config["retries"]["parse_max_attempts"]
    retry_of = None
    for attempt in range(1, max_attempts + 1):
        call_id = str(uuid.uuid4())
        output, exception, started_at, latency_ms = _invoke(engine, temperature, lc_messages)
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
        _record(call_id=call_id, retry_of=retry_of, attempt=attempt, call_site=call_site,
                invocation_key=invocation_key, engine=engine, messages=messages, started_at=started_at,
                latency_ms=latency_ms, temperature=temperature, output=output, parsed=parsed,
                parsed_ok=parsed_ok, error=error)
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

    run = _require_run()
    lc_messages = [HumanMessage(content=message)]
    messages = _message_dicts(lc_messages)
    engine = route(run.arm, call_site, messages, _config)
    temperature = float(_config["call_sites"][call_site]["temperature"])
    call_id = str(uuid.uuid4())
    output, exception, started_at, latency_ms = _invoke(engine, temperature, lc_messages)
    parsed, parsed_ok, error = None, False, None
    if exception is not None:
        error = f"{type(exception).__name__}: {exception}"
    else:
        try:
            parsed, parsed_ok = parse(output.content), True
        except Exception as e:  # the agent itself raises on the same response right after
            error = f"{type(e).__name__}: {e}"
    _record(call_id=call_id, retry_of=None, attempt=1, call_site=call_site, invocation_key=invocation_key,
            engine=engine, messages=messages, started_at=started_at, latency_ms=latency_ms,
            temperature=temperature, output=output, parsed=parsed, parsed_ok=parsed_ok, error=error)
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
        return OpenAIEmbeddings(model=settings[f"{purpose}_model"])
    if provider == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=settings["fake_size"])
    raise NotImplementedError(f"embeddings provider {provider!r} is built in F1")


def vector_db_dirname() -> str:
    """One vector DB per provider, so a DB built with one embedding is never queried with another."""
    return f"context_vector_db_{_config['embeddings']['provider']}"
