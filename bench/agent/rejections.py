"""Which rejection of a request is the model's failure (F1 5b): only a prompt longer than the engine's
context. HTTP 413 always is; a 400 only when its message says so (Anthropic's "prompt is too long",
OpenAI-compatible `context_length_exceeded`, llama.cpp's and vLLM's context-size messages). Every other
400 (a workspace, key, permission, billing or credit state, an invalid parameter, an unsupported
model) is the harness's: the run fails and nothing is scored."""
import re

CONTEXT_OVERFLOW = re.compile(
    r"prompt is too long|context[ _](length|window)|too many (input )?tokens|maximum context"
    r"|exceeds? .*(context|token)", re.IGNORECASE)


def is_model_rejection(status: int, message: str) -> bool:
    """Whether a rejection with this HTTP status and message is the model's failure."""
    if status == 413:
        return True
    return status == 400 and bool(CONTEXT_OVERFLOW.search(message or ""))
