"""Which rejection of a request is the model's failure (F1 5b): only a prompt longer than the engine's
context. HTTP 413 always is; a 400 only when its message says so (Anthropic's "prompt is too long",
OpenAI-compatible `context_length_exceeded`, llama.cpp's and vLLM's context-size messages). Every other
400 (a workspace, key, permission, billing or credit state, an invalid parameter, an unsupported
model) is the harness's: the run fails and nothing is scored."""
import re

# only the prompt (the input) longer than the context: the output budget is the configuration's
CONTEXT_OVERFLOW = re.compile(
    r"prompt is too long|context[ _]length[ _]exceeded|exceeds the (available )?context (size|window|length)"
    r"|too many input tokens|input .* exceeds .* context"
    r"|maximum context length is \d+ tokens\. however, (you requested|your (request|messages))", re.IGNORECASE)
OUTPUT_BUDGET = re.compile(r"max_(completion_)?tokens", re.IGNORECASE)  # a parameter error, never the model's


def is_model_rejection(status: int, message: str) -> bool:
    """Whether a rejection with this HTTP status and message is the model's failure."""
    if status == 413:
        return True
    message = message or ""
    return status == 400 and not OUTPUT_BUDGET.search(message) and bool(CONTEXT_OVERFLOW.search(message))
