"""Which rejection of a request is the model's failure (F1 5b): only a prompt longer than the engine's
context. HTTP 413 always is; a 400 only when its message says so (Anthropic's "prompt is too long",
OpenAI-compatible `context_length_exceeded`, llama.cpp's and vLLM's context-size messages). Every other
400 (a workspace, key, permission, billing or credit state, an invalid parameter, an unsupported
model) is the harness's: the run fails and nothing is scored.

A message that gives the context, the prompt and the output budget in numbers is decided by them: the
model's only when the prompt alone is proven longer than the context. A prompt that fits, rejected because
prompt + max_tokens does not, is the harness's (the output budget is the configuration's). vLLM 0.30's
message for that check (vllm/renderers/params.py, `_token_len_check`) reads the same whether the prompt
alone overflows or only prompt + max_tokens does: it tokenizes at most max_model_len - max_tokens + 1
tokens and reports "at least" that many, so with max_tokens > 0 it never proves the prompt alone too long,
and the run fails rather than score a rejection the model may not own."""
import re
from typing import Optional

# only the prompt (the input) longer than the context: the output budget is the configuration's
CONTEXT_OVERFLOW = re.compile(
    r"prompt is too long|context[ _]length[ _]exceeded|exceeds the (available )?context (size|window|length)"
    r"|too many input tokens|input .* exceeds .* context"
    r"|maximum context length is \d+ tokens\. however, (you requested|your (request|messages))"
    r"|prompt \(length \d+\) is longer than the maximum model length"  # vLLM 0.30's input processor
    r"|prompt length \d+ exceeds the max_model_len", re.IGNORECASE)  # vLLM 0.30's Responses API
OUTPUT_BUDGET = re.compile(r"max_(completion_)?tokens", re.IGNORECASE)  # a parameter error, never the model's

_CONTEXT = r"maximum context length is (?P<context>\d+) tokens\. however, "
# vLLM 0.30, TokenizeParams._token_len_check: "... you requested 4096 output tokens and your prompt contains
# at least 28673 input tokens, for a total of at least 32769 tokens. ..."
VLLM_TOKENS = re.compile(_CONTEXT + r"you requested (?P<output>\d+) output tokens and your prompt contains "
                         r"(?P<at_least>at least )?(?P<prompt>\d+) input tokens", re.IGNORECASE)
# vLLM 0.30, TokenizeParams._text_len_check: "... your prompt contains 200000 characters (more than 172038
# characters, which is the upper bound for 28673 input tokens). ..."
VLLM_CHARS = re.compile(_CONTEXT + r"you requested (?P<output>\d+) output tokens and your prompt contains "
                        r"(?P<chars>\d+) characters \(more than (?P<bound_chars>\d+) characters, which is the "
                        r"upper bound for (?P<bound_tokens>\d+) input tokens\)", re.IGNORECASE)
# OpenAI's: "... you requested 9000 tokens (8000 in the messages, 1000 in the completion)."
OPENAI_SPLIT = re.compile(_CONTEXT + r"you requested \d+ tokens \((?P<prompt>\d+) in the messages, "
                          r"\d+ in the completion\)", re.IGNORECASE)
# OpenRouter's: "... you requested about 140000 tokens (139000 of text input, 1000 in the output)."
SPLIT_INPUTS = re.compile(_CONTEXT + r"you requested about \d+ tokens \((?P<parts>[^)]*)\)", re.IGNORECASE)
INPUT_PART = re.compile(r"(\d+) of [a-z ]*input", re.IGNORECASE)
# vLLM 0.30's input processor, a prompt exactly as long as the context: no output budget fits it
VLLM_PROMPT_FILLS = re.compile(r"prompt \(length \d+\) plus the number of requested output tokens \(at least 1\) "
                               r"is longer than the maximum model length", re.IGNORECASE)


def prompt_alone_overflows(message: str) -> Optional[bool]:
    """From a message that gives the lengths in numbers: whether the prompt alone is proven longer than the
    context. None when the message does not give them."""
    found = VLLM_TOKENS.search(message)
    if found:  # "at least N" proves the prompt at least N long: it overflows only if N > context
        return int(found["prompt"]) > int(found["context"])
    found = VLLM_CHARS.search(message)
    if found:  # more characters than the context's tokens can hold proves more tokens than the context
        chars, bound_chars, bound_tokens = int(found["chars"]), int(found["bound_chars"]), int(found["bound_tokens"])
        return chars * bound_tokens > int(found["context"]) * bound_chars
    found = OPENAI_SPLIT.search(message)
    if found:
        return int(found["prompt"]) > int(found["context"])
    found = SPLIT_INPUTS.search(message)
    if found:
        parts = [int(n) for n in INPUT_PART.findall(found["parts"])]
        if parts:
            return sum(parts) > int(found["context"])
    if VLLM_PROMPT_FILLS.search(message):
        return True
    return None


def is_model_rejection(status: int, message: str) -> bool:
    """Whether a rejection with this HTTP status and message is the model's failure."""
    if status == 413:
        return True
    if status != 400:
        return False
    message = message or ""
    decided = prompt_alone_overflows(message)
    if decided is not None:
        return decided
    return not OUTPUT_BUDGET.search(message) and bool(CONTEXT_OVERFLOW.search(message))
