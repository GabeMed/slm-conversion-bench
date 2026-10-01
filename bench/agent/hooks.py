"""What the patched CHESS calls into (vendor/chess/PATCHES.md). Runs only in the agent environment.

Every LLM invocation of the agent goes through `invoke_tool_call` (tools) or `invoke_agent_call`
(the agent's choice of the next tool). Both ask the router (C4) for the engine (or use the run's
fixed engine: B2, `replay --engine`), invoke it with no hidden client retry, and write one C1 line
per attempt, failures included. Transport errors (429, 5xx, timeouts, connection errors) are
retried with backoff and unparseable output is retried on the same engine, with one attempt
counter for both (patch 5b). When the engine is `cheap_alt`, B1's few-shot prefix is prepended to
what is sent; the router sees the prompt without it, and C1 records the messages as sent.

Two kinds of failure, kept apart:
- the **model's**, a closed set: the engine rejecting this request (HTTP 400, 413) and an empty or
  unparseable output. It is a C1 line, and the exception goes on to CHESS, which handles it as it
  always did;
- the **harness's**, everything else: no run or question set, routing, configuration, a missing API
  key, an engine that stays unreachable after every transport retry, any other API or client error
  (401, 402, 403, 404, 409, 422, any other status, a response that does not validate, anything
  unrecognised), a call site outside the registered set on `test`. It is recorded here as well, because CHESS swallows every exception, and the runner reads it with
  `take_harness_errors()` to fail the run instead of recording it as done.

State is process-global: one run and one question at a time per process. The tools of a question
call the model from several threads; the writer and the counters are shared under a lock. A run
split over worker processes (`bench run --workers`) gives each its own C1 file and one shared
`stop` flag: once any worker's question fails, every worker refuses its next model call.
"""
import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from bench.contracts.config import ConfigError, engine_spec
from bench.contracts.router import Route, route
from bench.provenance import redact, scrub

_config: Optional[Dict[str, Any]] = None
_run: Optional["_RunState"] = None
_models: Dict[tuple, Any] = {}
_models_lock = threading.Lock()
_embedders: Dict[tuple, Any] = {}
_embed_lock = threading.Lock()  # one encoder per model, shared by threads
_sleep = time.sleep  # the backoff between transport retries; tests replace it
_jitter = random.Random()  # timing only: spreads the retries of concurrent calls, never touches content


class HarnessError(RuntimeError):
    """A failure of the harness, not of the model: the run must not be recorded as done.

    It records itself in the current run when it is created, wherever it is raised (inside CHESS,
    whose handlers swallow exceptions, included), so no call site has to remember to report it."""

    def __init__(self, message: str):
        super().__init__(message)
        run = _run
        if run is not None:
            with run.lock:
                run.harness_errors.append(redact(scrub(message), _config) if _config is not None else scrub(message))


class RunAborted(RuntimeError):
    """A model call refused because its run has already failed (a harness error, recorded once, or
    a question that failed in another worker): a failed run spends nothing more, within the question
    as after it. Not recorded again, and CHESS swallows it like any tool error."""


@dataclass
class _RunState:
    run_id: str
    arm: Optional[str]
    calls_path: Path
    engine: Optional[str] = None  # a fixed engine (B2, `replay --engine`): the router is not asked
    few_shot: Optional[Dict[str, List[Any]]] = None  # {call site: messages}; None: cheap_alt is refused
    allowed_call_sites: Optional[frozenset] = None  # `test` runs: the registered set (REQ-001)
    question_id: Optional[str] = None
    occurrences: Dict[tuple, int] = field(default_factory=dict)
    harness_errors: List[str] = field(default_factory=list)
    unregistered_call_sites: set = field(default_factory=set)
    stop: Any = None  # an Event shared by the worker processes of one run: set when a question failed in any
    lock: threading.RLock = field(default_factory=threading.RLock)


class RoutedEngine:
    """What `get_llm_chain` returns: a placeholder, since the engine is chosen per call."""

    def __init__(self, engine_name: str):
        self.engine_name = engine_name


def check_settings(config: Dict[str, Any]) -> None:
    """The agent's settings, validated by C2 itself (bench.contracts.config.validate_config)."""
    from bench.contracts.config import agent_settings_errors
    errors = agent_settings_errors(config)
    if errors:
        raise ConfigError("; ".join(errors))


def configure(config: Dict[str, Any]) -> None:
    global _config
    check_settings(config)
    _config = config
    _models.clear()


def config() -> Dict[str, Any]:
    """The configuration of this process (read by the patched CHESS: seeds, concurrency)."""
    if _config is None:
        raise HarnessError("hooks.configure() was not called")
    return _config


def max_workers() -> int:
    return config()["agent"]["max_workers"]


def start_run(run_id: str, arm: Optional[str], calls_path: Path, *, engine: Optional[str] = None,
              few_shot: Optional[Dict[str, List[Dict[str, str]]]] = None,
              allowed_call_sites: Optional[List[str]] = None, stop: Any = None) -> None:
    """`engine` fixes the engine of every call (no routing); `few_shot` is the prefix per call site
    for `cheap_alt` ({} for none by design); `allowed_call_sites` aborts any other call site; `stop`
    is the flag the worker processes of one run share (set: no further model call)."""
    global _run
    if _config is None:
        raise HarnessError("hooks.configure() must run before start_run()")
    if (arm is None) == (engine is None):
        raise HarnessError("a run is routed by its arm or has a fixed engine, one of the two")
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    calls_path.touch()
    prefix = None if few_shot is None else {site: _lc_messages(messages) for site, messages in few_shot.items()}
    _run = _RunState(run_id=run_id, arm=arm, calls_path=calls_path, engine=engine, few_shot=prefix,
                     allowed_call_sites=None if allowed_call_sites is None else frozenset(allowed_call_sites),
                     stop=stop)


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


def unregistered_call_sites() -> List[str]:
    """Call sites this run tried to call outside the registered set (each aborted the run)."""
    run = _require_run()
    with run.lock:
        return sorted(run.unregistered_call_sites)


def end_run() -> None:
    global _run
    _run = None


def _require_run() -> _RunState:
    if _run is None:
        raise HarnessError("no run in progress: the harness runner must call hooks.start_run()")
    return _run


def _harness(what: str, fn: Callable[[], Any]) -> Any:
    """Run a harness step; any failure becomes a (self-recording) HarnessError."""
    try:
        return fn()
    except HarnessError:
        raise
    except Exception as e:
        raise HarnessError(f"{what}: {type(e).__name__}: {e}") from e


def _begin(call_site: str, invocation_key: str, messages: List[Dict[str, str]]):
    """Checks and decisions made before the model is called: question, call site, identity, route.
    The router sees the agent's prompt as CHESS wrote it, before any few-shot prefix."""
    run = _require_run()
    if run.question_id is None:
        raise HarnessError("hooks.set_question() was not called for this question")
    if run.allowed_call_sites is not None and call_site not in run.allowed_call_sites:
        with run.lock:
            run.unregistered_call_sites.add(call_site)
        raise HarnessError(f"call site {call_site} is not in the set registered on train and calib "
                           f"(registry/call_sites.json): the run is aborted (REQ-001)")
    with run.lock:
        n = run.occurrences.get((call_site, invocation_key), 0) + 1
        run.occurrences[(call_site, invocation_key)] = n
    key = invocation_key if n == 1 else f"{invocation_key}@{n}"
    chosen = Route(run.engine, None) if run.engine else route(run.arm, call_site, messages, _config)
    temperature = float(_config["call_sites"][call_site]["temperature"])
    return key, chosen, temperature, chat_model(chosen.engine, temperature)


def _prefixed(call_site: str, engine: str, lc_messages: List[Any]) -> List[Any]:
    """What is sent: B1's few-shot pairs before the prompt whenever the engine is `cheap_alt`."""
    if engine != "cheap_alt":
        return lc_messages
    run = _require_run()
    if run.few_shot is None:
        raise HarnessError("cheap_alt was reached without its few-shot prefix (arms.B1.few_shot): "
                           "the runner must load it before the run starts")
    return list(run.few_shot.get(call_site, [])) + list(lc_messages)


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
            headers = {}
            for header, variable in (endpoint.get("headers_env") or {}).items():
                headers[header] = os.environ.get(variable)
                if not headers[header]:
                    raise HarnessError(f"engine {engine}: environment variable {variable} ({header}) is not set")
            # the request's extra body: the role's own (a provider's reasoning switch) and, through an
            # aggregator, the routing pin of `provider.routing`
            extra_body = dict(params.pop("extra_body", None) or {})
            routing = (spec.get("provider") or {}).get("routing")
            if routing is not None:
                extra_body["provider"] = routing
            _models[key] = ChatOpenAI(
                model=spec["model"],
                openai_api_base=endpoint["base_url"],
                openai_api_key=api_key,
                temperature=temperature,
                max_tokens=params.pop("max_tokens", None),
                timeout=params.pop("timeout_s"),
                max_retries=0,  # every attempt is visible in C1; no hidden client retry
                default_headers=headers or None,
                extra_body=extra_body or None,
                model_kwargs=params,
            )
        return _models[key]


def _message_dicts(messages: List[Any]) -> List[Dict[str, str]]:
    roles = {"human": "user", "ai": "assistant", "system": "system"}
    return [{"role": roles[m.type], "content": m.content} for m in messages]


def _lc_messages(messages: List[Dict[str, str]]) -> List[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    kinds = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}
    return [kinds[m["role"]](content=m["content"]) for m in messages]


NO_USAGE = {"input": None, "cached_input": None, "output": None, "reasoning": None, "source": "missing"}


def _usage(output: Any) -> Dict[str, Any]:
    token_usage = (getattr(output, "response_metadata", None) or {}).get("token_usage") or {}
    if token_usage.get("prompt_tokens") is None or token_usage.get("completion_tokens") is None:
        return dict(NO_USAGE)
    details = token_usage.get("prompt_tokens_details") or {}
    return {
        "input": token_usage["prompt_tokens"],
        "cached_input": details.get("cached_tokens"),
        "output": token_usage["completion_tokens"],
        # the reasoning tokens, a part of `output`; null when the provider does not report them
        "reasoning": (token_usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "source": "api",
    }


def recorded_temperature(spec: Dict[str, Any], temperature: float) -> float:
    """The temperature C1 records for a call sent at `temperature` to the engine of `spec`: the one in
    effect. Where the configuration declares that the provider forces a value (`reasoning.forced_temperature`),
    it is that value, whatever was sent."""
    forced = (spec.get("reasoning") or {}).get("forced_temperature")
    return temperature if forced is None else float(forced)


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
    metadata = (getattr(output, "response_metadata", None) or {}) if output is not None else {}
    line = json.dumps({
        "run_id": run.run_id, "call_id": call_id, "retry_of": retry_of, "attempt": attempt,
        "question_id": run.question_id, "call_site": call_site, "invocation_key": invocation_key,
        "cluster": chosen.cluster, "engine": chosen.engine, "model_role": spec["model_role"],
        "model": spec["model"], "endpoint": spec["endpoint"]["kind"],
        "prompt_messages": messages,
        "response_text": output.content if output is not None and isinstance(output.content, str) else None,
        "parsed_output": _jsonable(parsed) if parsed_ok else None, "parsed_ok": parsed_ok,
        "usage": _usage(output) if output is not None else dict(NO_USAGE),
        "latency_ms": latency_ms, "started_at": started_at,
        "temperature": recorded_temperature(spec, temperature),
        "error": redact(scrub(error), _config) if error is not None else None,  # no local path, no credential
        # what the answer says about itself (null when the provider does not say, or nothing answered),
        # and the provider the configuration pins for this engine
        "finish_reason": metadata.get("finish_reason"), "model_reported": metadata.get("model_name"),
        "provider": (spec.get("provider") or {}).get("name"),
    }, ensure_ascii=False)
    with run.lock, open(run.calls_path, "a") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------- calls

MODEL_REJECTIONS = (400, 413)  # the request this engine rejects (a prompt longer than its context)


def classify(exception: BaseException) -> str:
    """What an exception of the model call is: `transport` (retried: 408, 429, >= 500, and any timeout
    or lost connection, whichever layer raised it: the OpenAI client's own, httpx's, Python's), `model`
    (the closed set of the model's failures: the engine rejects this request, 400 or 413), or
    `harness` (everything else: another status, a response that does not validate, anything
    unrecognised)."""
    import httpx
    import openai
    if isinstance(exception, (openai.APIConnectionError, httpx.TransportError, TimeoutError, ConnectionError)):
        return "transport"  # APIConnectionError includes APITimeoutError
    if isinstance(exception, openai.APIStatusError):
        code = exception.status_code
        if code in (408, 429) or code >= 500:
            return "transport"
        if code in MODEL_REJECTIONS:
            return "model"
    return "harness"


def backoff_s(transport_failures: int) -> float:
    """The longest wait after the n-th transport failure of an invocation: base * 2^(n-1), capped."""
    backoff = _config["retries"]["http_backoff_s"]
    return float(min(backoff["max"], backoff["base"] * 2 ** (transport_failures - 1)))


def _backoff(transport_failures: int) -> None:
    """Wait between half and all of `backoff_s`, at random: calls that failed together (a server
    whose shared context the concurrent calls of one step overflowed) must not retry together,
    or they fail together again, every time."""
    _sleep(backoff_s(transport_failures) * _jitter.uniform(0.5, 1.0))


def _invocation(call_site: str, invocation_key: str, lc_messages: List[Any], interpret: Callable,
                parse_max_attempts: int):
    """One invocation, every attempt a C1 line chained by `retry_of`, one attempt counter for both
    kinds of retry: a transport failure is retried after a backoff up to `retries.http_max_attempts`
    such failures (then the harness fails: the engine is unreachable); an empty or unparseable
    output is retried up to `parse_max_attempts` such failures (then OutputParserException).

    `interpret(output)` -> (parsed, parsed_ok, error, outcome, exception), outcome one of `ok`,
    `answer` (return the text although it did not parse), `parse` (retry), `failed` (raise).
    Returns (output, parsed)."""
    from langchain_core.exceptions import OutputParserException

    run = _require_run()
    _refuse_if_failed(run, call_site)
    messages = _message_dicts(lc_messages)
    key, chosen, temperature, model = _harness(call_site, lambda: _begin(call_site, invocation_key, messages))
    sent = _harness(call_site, lambda: _prefixed(call_site, chosen.engine, lc_messages))
    sent_dicts = _message_dicts(sent)
    http_max_attempts = _config["retries"]["http_max_attempts"]
    failures = {"transport": 0, "parse": 0}
    retry_of, attempt = None, 0
    while True:
        if attempt:  # every retry, too: another thread may have failed the run (an unreachable engine)
            _refuse_if_failed(run, call_site)
        attempt += 1
        call_id = str(uuid.uuid4())
        output, exception, started_at, latency_ms = _invoke(model, sent)
        if exception is not None:
            parsed, parsed_ok = None, False
            error = f"{type(exception).__name__}: {exception}"
            outcome = {"transport": "transport", "model": "failed", "harness": "harness"}[classify(exception)]
        elif not isinstance(output.content, str):  # content parts, not text: nothing here can read them
            parsed, parsed_ok, outcome, exception = None, False, "harness", None  # its usage is still recorded
            error = f"the engine answered {type(output.content).__name__} content, not text"
        else:
            parsed, parsed_ok, error, outcome, exception = interpret(output)
        _harness(call_site, lambda: _record(
            call_id=call_id, retry_of=retry_of, attempt=attempt, call_site=call_site, invocation_key=key,
            chosen=chosen, messages=sent_dicts, started_at=started_at, latency_ms=latency_ms,
            temperature=temperature, output=output, parsed=parsed, parsed_ok=parsed_ok, error=error))
        if outcome in ("ok", "answer"):
            return output, parsed
        if outcome == "transport":
            failures["transport"] += 1
            if failures["transport"] >= http_max_attempts:
                raise HarnessError(f"{call_site}: engine {chosen.engine} unreachable after "
                                   f"{failures['transport']} transport failures: {error}") from exception
            _backoff(failures["transport"])
        elif outcome == "parse":
            failures["parse"] += 1
            if failures["parse"] >= parse_max_attempts:
                raise OutputParserException(error)
        elif outcome == "harness":
            raise HarnessError(f"{call_site}: engine {chosen.engine}: {error}") from exception
        else:
            raise exception
        retry_of = call_id


def _refuse_if_failed(run: _RunState, call_site: str) -> None:
    with run.lock:
        if run.harness_errors:
            raise RunAborted(f"{call_site}: the run has failed ({run.harness_errors[0][:200]}): no further model call")
    if run.stop is not None and run.stop.is_set():
        raise RunAborted(f"{call_site}: the run has failed in another worker: no further model call")


def invoke_tool_call(call_site: str, invocation_key: str, lc_messages: List[Any], parser: Any) -> Any:
    """One tool call: route, invoke, parse; an empty or unparseable output is retried up to the configured cap."""
    from langchain_core.exceptions import OutputParserException

    def interpret(output):
        if not output.content.strip():
            return None, False, "empty output", "parse", None
        try:
            return parser.invoke(output), True, None, "ok", None
        except OutputParserException as e:
            return None, False, f"OutputParserException: {e}", "parse", None
        except Exception as e:  # not retried, as in CHESS
            return None, False, f"{type(e).__name__}: {e}", "failed", e
    return _invocation(call_site, invocation_key, lc_messages, interpret,
                       _config["retries"]["parse_max_attempts"])[1]


def invoke_agent_call(call_site: str, invocation_key: str, message: str, parse: Callable[[str], Any]) -> str:
    """The agent's next-tool choice: one answer, as in CHESS (transport failures are retried).
    `parse` is the agent's own reading of the response; the agent raises on it itself."""
    from langchain_core.messages import HumanMessage

    def interpret(output):
        try:
            return parse(output.content), True, None, "ok", None
        except Exception as e:  # the agent itself raises on the same response right after
            return None, False, f"{type(e).__name__}: {e}", "answer", None
    return _invocation(call_site, invocation_key, [HumanMessage(content=message)], interpret, 1)[0].content


# ---------------------------------------------------------------- retrieval embeddings

def _recording(inner: Any) -> Any:
    """Retrieval embeddings are the harness's infrastructure, not the model under test: any failure
    (a missing or rejected key, a network error) is a harness failure, even where CHESS swallows it."""
    from langchain_core.embeddings import Embeddings

    class Recording(Embeddings):
        def embed_documents(self, texts):
            return _harness("embeddings", lambda: inner.embed_documents(texts))

        def embed_query(self, text):
            return _harness("embeddings", lambda: inner.embed_query(text))

    return Recording()


def embeddings(purpose: str):
    """Embeddings for CHESS retrieval: `entity` (retrieve_entity) or `context` (column-description vector DB)."""
    settings = _config["embeddings"]
    provider = settings["provider"]
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise HarnessError("embeddings.provider is openai and OPENAI_API_KEY is not set")
        from langchain_openai import OpenAIEmbeddings
        return _recording(OpenAIEmbeddings(model=settings[f"{purpose}_model"]))
    if provider == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding
        return _recording(DeterministicFakeEmbedding(size=settings["fake_size"]))
    if provider == "local":
        local = settings["local"]
        return _recording(_harness("embeddings", lambda: _local_embeddings(local["model"], local["revision"])))
    raise HarnessError(f"unknown embeddings provider {provider!r}")


def _local_embeddings(model: str, revision: str):
    """Patch 10b: a sentence-transformers model from the agent environment, pinned by revision,
    on CPU, unit vectors (CHESS compares them by dot product). Loaded once per process."""
    from langchain_core.embeddings import Embeddings

    with _models_lock:
        if (model, revision) not in _embedders:
            from sentence_transformers import SentenceTransformer
            _embedders[(model, revision)] = SentenceTransformer(model, revision=revision, device="cpu")
        encoder = _embedders[(model, revision)]

    class Local(Embeddings):
        def embed_documents(self, texts):
            with _embed_lock:
                return encoder.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True).tolist()

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    return Local()


def chroma_settings(persist_directory: Path):
    """Patch 14: the settings of CHESS's vector DB, given explicitly. Chroma's own settings are a
    pydantic BaseSettings that reads `./.env` (`env_file=".env"`), which could point the vector DB
    at a remote server; here no `.env` is read, and nothing is sent as telemetry."""
    import chromadb.config
    return chromadb.config.Settings(_env_file=None, is_persistent=True, persist_directory=str(persist_directory),
                                    anonymized_telemetry=False)


def vector_db_dirname() -> str:
    """One vector DB per provider, so a DB built with one embedding is never queried with another."""
    return f"context_vector_db_{_config['embeddings']['provider']}"
