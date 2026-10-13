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
"""
import hashlib
import threading
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


def _cache_split(turns: List[Dict[str, Any]], markers: List[str]):
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
            blocks = [{"type": "text", "text": text[:cut], "cache_control": {"type": "ephemeral"}},
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
        self.max_retries = 0
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, timeout=timeout,
                                          max_retries=0, default_headers=default_headers)

    def request(self, messages: List[Any]):
        """The keyword arguments of the SDK call for `messages`, and the hash of its cached prefix (or None)."""
        system, turns = _split_system(messages)
        turns, prefix = _cache_split(turns, self.cache_breaks)
        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "messages": turns, **self.model_kwargs}
        if system is not None:
            kwargs["system"] = system
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return kwargs, (None if prefix is None else hashlib.sha256(f"{system}|{prefix}".encode()).hexdigest())

    def invoke(self, messages: List[Any]):
        from langchain_core.messages import AIMessage

        kwargs, prefix = self.request(messages)
        leader = prefix is not None and _WARMUP.enter(prefix, self.timeout)
        try:
            with self.client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
        finally:
            if leader:
                _WARMUP.leave(prefix)
        text = "".join(block.text for block in message.content if block.type == "text")
        return AIMessage(content=text, response_metadata=_response_metadata(message))
