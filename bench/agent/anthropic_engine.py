"""An API engine that speaks Anthropic's Messages API (`endpoint.protocol: anthropic`), through the
official SDK.

Everything downstream of the client (hooks._usage and _record, the preflight probe) reads an answer in
the shape `langchain_openai` gives it: `.content`, and `.response_metadata` with `token_usage`,
`finish_reason` and `model_name`. This client answers in that shape, so C1 and the judgments do not
know which protocol served a call.

What differs from an OpenAI-compatible engine, and how it is mapped:
- **No temperature is sent.** Current Claude models reject sampling parameters; the role declares the
  temperature in effect as `reasoning.forced_temperature`, which C1 records (hooks.recorded_temperature).
- **System messages** before the first turn go in the request's `system` field; a later one stays in
  `messages` with the role `system`, which current models accept mid-conversation.
- **Usage:** Anthropic's `input_tokens` excludes the cache, OpenAI's `prompt_tokens` includes it, so
  `prompt_tokens` = input + cache reads + cache writes, `cached_tokens` = cache reads, and the writes,
  which Anthropic bills apart, are `cache_write_tokens` (C1's `usage.cache_write`, priced by J3).
  The thinking tokens are `output_tokens_details.thinking_tokens`, a part of `output_tokens`.
- **Prompt caching** (`params.cache_breaks`, a list of markers): CHESS's long prompts are fixed
  instructions and examples followed by the question's own content. The last user message is sent as two
  text blocks, split right after the last occurrence of the first marker it contains, with a cache
  breakpoint on the first block, so every call with that prefix reads it from the cache. The first call
  of a prefix in this process goes alone, and the calls that arrive while it runs wait for it: calls
  sent in parallel would all miss the cache and all pay to write it.
- **Stop reason:** `end_turn` and `stop_sequence` are `stop`, `max_tokens` is `length`; any other
  (`refusal`, ...) is recorded as Anthropic names it, so the preflight probe fails on it.
- **Streaming** carries every request, as the SDK requires for long answers; `max_retries` is 0, as for
  every engine: each attempt is a C1 line.
- **Batch mode** (`bench run --batch`, K1 only; the environment variable `BENCH_ANTHROPIC_BATCH` tells
  each process): the same requests go through the Message Batches API, billed at half price. The calls a
  process makes within `BATCH_GATHER_S` of each other (one CHESS step: the columns `filter_column` judges
  in parallel) form one batch, polled until it ends; each call then returns as a streamed one would. A
  batched call's latency is how long the batch took, not a measurement. A request the batch could not
  answer raises an error classified as the API's own status would be (`classified_as`); stopping the
  run, or losing the parent process, cancels the batches still open.
"""
import hashlib
import multiprocessing
import os
import threading
import time
import uuid
from concurrent.futures import Future, wait
from typing import Any, Dict, List, Optional

FINISH_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}


def _split_system(messages: List[Any]):
    """The leading system messages as the `system` field; the rest as Messages API turns."""
    roles = {"human": "user", "ai": "assistant", "system": "system"}
    system, turns = [], []
    for message in messages:
        role = roles[message.type]
        if role == "system" and not turns:
            system.append(message.content)
        else:
            turns.append({"role": role, "content": message.content})
    return "\n\n".join(system) or None, turns


def _cache_split(turns: List[Dict[str, Any]], markers: List[str], ttl: Optional[str] = None):
    """`turns` with the last user message split after the first marker it contains (its last occurrence),
    the first block marked as a cache breakpoint; and the hash of that prefix (None when nothing splits)."""
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["role"] == "user" and isinstance(turns[i]["content"], str):
            break
    else:
        return turns, None
    text = turns[i]["content"]
    for marker in markers:
        at = text.rfind(marker)
        if at >= 0:
            cut = at + len(marker)
            if cut >= len(text):
                return turns, None
            control = {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}
            blocks = [{"type": "text", "text": text[:cut], "cache_control": control},
                      {"type": "text", "text": text[cut:]}]
            split = turns[:i] + [{"role": "user", "content": blocks}] + turns[i + 1:]
            prefix = repr(turns[:i]) + text[:cut]
            return split, hashlib.sha256(prefix.encode()).hexdigest()
    return turns, None


class _Warmup:
    """Per cache prefix: the first call goes alone; the calls that arrive while it runs wait for it."""

    def __init__(self):
        self.lock, self.done = threading.Lock(), {}

    def enter(self, key: str, timeout: float) -> bool:
        """True for the call that writes the prefix (it must call `leave`); False once it is written."""
        with self.lock:
            event = self.done.get(key)
            if event is None:
                self.done[key] = threading.Event()
                return True
        event.wait(timeout)  # a leader that failed still sets it: the next call writes the cache itself
        return False

    def leave(self, key: str) -> None:
        with self.lock:
            self.done[key].set()


_WARMUP = _Warmup()  # one per process: the clients of one engine at several temperatures share a cache


def _response_metadata(message: Any) -> Dict[str, Any]:
    usage = message.usage
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    details = getattr(usage, "output_tokens_details", None)
    thinking = getattr(details, "thinking_tokens", None) if details is not None else None
    return {
        "token_usage": {
            "prompt_tokens": usage.input_tokens + cache_read + cache_write,
            "completion_tokens": usage.output_tokens,
            "prompt_tokens_details": {"cached_tokens": cache_read, "cache_write_tokens": cache_write},
            "completion_tokens_details": {"reasoning_tokens": thinking},
        },
        "finish_reason": FINISH_REASONS.get(message.stop_reason, message.stop_reason),
        "model_name": message.model,
    }


class AnthropicChat:
    """The client for one Anthropic engine: `.invoke(langchain messages)` returns an `AIMessage`."""

    def __init__(self, *, model: str, base_url: str, api_key: str, max_tokens: int, timeout: float,
                 default_headers: Optional[Dict[str, str]] = None, extra_body: Optional[Dict[str, Any]] = None,
                 model_kwargs: Optional[Dict[str, Any]] = None):
        import anthropic

        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout
        self.extra_body, self.model_kwargs = extra_body, dict(model_kwargs or {})
        self.cache_breaks = list(self.model_kwargs.pop("cache_breaks", None) or [])
        self.cache_ttl = self.model_kwargs.pop("cache_ttl", None)
        self.max_retries = 0
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, timeout=timeout,
                                          max_retries=0, default_headers=default_headers)

    def request(self, messages: List[Any]):
        """The keyword arguments of the SDK call for `messages`, and the hash of its cached prefix (or None)."""
        system, turns = _split_system(messages)
        turns, prefix = _cache_split(turns, self.cache_breaks, self.cache_ttl)
        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "messages": turns, **self.model_kwargs}
        if system is not None:
            kwargs["system"] = system
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return kwargs, (None if prefix is None else hashlib.sha256(f"{system}|{prefix}".encode()).hexdigest())

    def invoke(self, messages: List[Any]):
        from langchain_core.messages import AIMessage

        kwargs, prefix = self.request(messages)
        batched = bool(os.environ.get(BATCH_ENV))
        # the others wait for the first call of a prefix as long as it can take: a batch, up to a day
        leader = prefix is not None and _WARMUP.enter(prefix, BATCH_DEADLINE_S if batched else self.timeout)
        try:
            if batched:
                message = _batcher(self.client).call(_batch_params(kwargs))
            else:
                with self.client.messages.stream(**kwargs) as stream:
                    message = stream.get_final_message()
        finally:
            if leader:
                _WARMUP.leave(prefix)
        text = "".join(block.text for block in message.content if block.type == "text")
        return AIMessage(content=text, response_metadata=_response_metadata(message))


# ---------------------------------------------------------------- batch mode (K1)

BATCH_ENV = "BENCH_ANTHROPIC_BATCH"
BATCH_GATHER_S = 2.0          # a call arriving within this of the previous one joins its batch
BATCH_POLL_S = (5.0, 60.0)    # the first wait between polls, and the longest
BATCH_DEADLINE_S = 25 * 3600  # Anthropic ends a batch within 24 h; after this, the call is a transport failure

# what each error type of a batched request is, as `hooks.classify` reads the API's own statuses
BATCH_ERROR_KINDS = {"invalid_request_error": "model", "rate_limit_error": "transport", "api_error": "transport",
                     "overloaded_error": "transport", "timeout_error": "transport"}


class BatchCallError(Exception):
    """A batched request that did not succeed, with the kind `hooks.classify` gives it."""

    def __init__(self, message: str, classified_as: str):
        super().__init__(message)
        self.classified_as = classified_as


def _batch_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """A streamed call's keyword arguments as one batch request's `params`: the extra body merged in."""
    params = {k: v for k, v in kwargs.items() if k != "extra_body"}
    params.update(kwargs.get("extra_body") or {})
    return params


def _batch_error(result: Any) -> BatchCallError:
    if result.type == "errored":
        error = result.error.error
        kind = BATCH_ERROR_KINDS.get(error.type, "harness")  # billing, authentication, ...: the account's
        if kind == "model" and "credit balance" in (error.message or "").lower():
            kind = "harness"
        return BatchCallError(f"batched request {error.type}: {error.message}", kind)
    return BatchCallError(f"batched request {result.type}", "transport")  # expired or canceled: send again


class _Batcher:
    """The batches of one process: calls gathered, submitted, polled, answered by custom_id."""

    def __init__(self, client: Any):
        self.client, self.lock = client, threading.Lock()
        self.pending: List[tuple] = []
        self.last_arrival = 0.0
        self.open: set = set()
        self.flusher: Optional[threading.Thread] = None

    def call(self, params: Dict[str, Any]):
        future: Future = Future()
        with self.lock:
            self.pending.append((uuid.uuid4().hex, params, future))
            self.last_arrival = time.monotonic()
            if self.flusher is None or not self.flusher.is_alive():
                self.flusher = threading.Thread(target=self._flush_when_quiet, daemon=True)
                self.flusher.start()
        try:
            while not future.done():  # short waits: a Ctrl-C or SIGTERM is raised between them
                wait([future], timeout=1.0)
            return future.result()  # its own exception, whatever it is (a TimeoutError included)
        except BaseException:
            if not future.done():  # interrupted (Ctrl-C, SIGTERM): spend nothing more
                self.cancel_open()
            raise

    def _flush_when_quiet(self) -> None:
        while True:
            with self.lock:
                quiet = time.monotonic() - self.last_arrival
                if quiet >= BATCH_GATHER_S:
                    batch, self.pending = self.pending, []
                    self.flusher = None
                    break
            time.sleep(BATCH_GATHER_S - quiet)
        if batch:
            threading.Thread(target=self._run, args=(batch,), daemon=True).start()

    def _run(self, batch: List[tuple]) -> None:
        futures = {custom_id: future for custom_id, _, future in batch}
        created = None
        try:
            created = self.client.messages.batches.create(
                requests=[{"custom_id": custom_id, "params": params} for custom_id, params, _ in batch])
            with self.lock:
                self.open.add(created.id)
            self._wait(created.id)
            for item in self.client.messages.batches.results(created.id):
                future = futures.pop(item.custom_id, None)
                if future is None:
                    continue
                if item.result.type == "succeeded":
                    future.set_result(item.result.message)
                else:
                    future.set_exception(_batch_error(item.result))
            for future in futures.values():
                future.set_exception(BatchCallError("the batch returned no result for this request", "transport"))
        except BaseException as e:  # creating or polling failed: every call of the batch failed with it
            for future in futures.values():
                if not future.done():
                    future.set_exception(e)
        finally:
            if created is not None:
                with self.lock:
                    self.open.discard(created.id)

    def _wait(self, batch_id: str) -> None:
        delay, started = BATCH_POLL_S[0], time.monotonic()
        parent = multiprocessing.parent_process()
        while True:
            status = self.client.messages.batches.retrieve(batch_id)
            if status.processing_status == "ended":
                return
            if parent is not None and not parent.is_alive():  # nobody left to merge what it answers
                self.cancel_open()
                raise BatchCallError("the run's parent process is gone", "harness")
            if time.monotonic() - started > BATCH_DEADLINE_S:
                raise BatchCallError(f"batch {batch_id} did not end within {BATCH_DEADLINE_S} s", "transport")
            time.sleep(delay)
            delay = min(delay * 2, BATCH_POLL_S[1])

    def cancel_open(self) -> None:
        with self.lock:
            open_ids = list(self.open)
        for batch_id in open_ids:
            try:
                self.client.messages.batches.cancel(batch_id)
            except Exception:  # noqa: BLE001 (best effort: the run is ending anyway)
                pass


_BATCHERS: Dict[int, _Batcher] = {}
_BATCHERS_LOCK = threading.Lock()


def _batcher(client: Any) -> _Batcher:
    """One batcher per client (per engine and temperature), so a batch holds requests of one key."""
    with _BATCHERS_LOCK:
        if id(client) not in _BATCHERS:
            _BATCHERS[id(client)] = _Batcher(client)
        return _BATCHERS[id(client)]
