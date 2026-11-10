"""The model's failure is a prompt longer than the context, proven by the message; a prompt that fits,
rejected because prompt + max_tokens does not, is the harness's. The vLLM strings are vLLM 0.30.0's
(the serving image vllm/vllm-openai:v0.30.0), built from its f-strings with the serving numbers:
max_model_len 32768, max_tokens 4096."""
import httpx
import openai
import pytest

from bench.agent import hooks
from bench.agent.rejections import is_model_rejection, prompt_alone_overflows

CONTEXT, OUTPUT = 32768, 4096
MAX_INPUT = CONTEXT - OUTPUT  # vLLM's max_input_tokens


def vllm_token_check(prompt_tokens: int, output: int = OUTPUT) -> str:
    """vllm/renderers/params.py `_token_len_check` (v0.30.0), as VLLMValidationError's str: vLLM tokenizes at
    most max_input_tokens + 1 tokens (`get_encode_kwargs`), so any longer prompt reads "at least" that."""
    max_input = CONTEXT - output
    count = min(prompt_tokens, max_input + 1)
    qualifier = "at least " if count == max_input + 1 else ""
    return (f"This model's maximum context length is {CONTEXT} tokens. However, you requested {output} output "
            f"tokens and your prompt contains {qualifier}{count} input tokens, for a total of "
            f"{qualifier}{count + output} tokens. Please reduce the length of the input prompt or the number of "
            f"requested output tokens. (parameter=input_tokens, value={count})")


def vllm_text_check(chars: int, output: int = OUTPUT, chars_per_token: int = 6) -> str:
    """vllm/renderers/params.py `_text_len_check` (v0.30.0)."""
    max_input = CONTEXT - output
    return (f"This model's maximum context length is {CONTEXT} tokens. However, you requested {output} output "
            f"tokens and your prompt contains {chars} characters (more than {max_input * chars_per_token} "
            f"characters, which is the upper bound for {max_input} input tokens). Please reduce the length of "
            f"the input prompt or the number of requested output tokens. (parameter=input_text, value={chars})")


def vllm_400(message: str) -> openai.BadRequestError:
    """What the OpenAI client raises on vLLM's 400 (ErrorResponse: message, type, param, code)."""
    body = {"message": message, "type": "BadRequestError", "param": "input_tokens", "code": 400}
    request = httpx.Request("POST", "http://pod:8000/v1/chat/completions")
    return openai.BadRequestError(f"Error code: 400 - {{'error': {body}}}", response=httpx.Response(400, request=request),
                                  body=body)


def test_vllm_prompt_plus_max_tokens_over_the_context_is_the_harness_s():
    """The prompt (30000 tokens) fits the context (32768); only prompt + max_tokens (4096) does not."""
    message = vllm_token_check(30000)
    assert "at least 28673 input tokens" in message  # what vLLM 0.30 reports: never the prompt's true length
    assert is_model_rejection(400, message) is False
    assert hooks.classify(vllm_400(message)) == "harness"


def test_vllm_prompt_alone_over_the_context_is_not_provable_with_max_tokens_set():
    """A 40000-token prompt reads exactly as the 30000-token one: vLLM stops counting at 28673, so the
    message does not prove the prompt alone too long and the run fails rather than blame the model."""
    assert vllm_token_check(40000) == vllm_token_check(30000)
    assert hooks.classify(vllm_400(vllm_token_check(40000))) == "harness"


def test_vllm_prompt_alone_over_the_context_with_no_output_budget_is_the_models():
    """With no max_tokens (0 output tokens), "at least 32769" input tokens proves the prompt alone too long."""
    message = vllm_token_check(40000, output=0)
    assert "at least 32769 input tokens" in message
    assert is_model_rejection(400, message) is True
    assert hooks.classify(vllm_400(message)) == "model"


@pytest.mark.parametrize("chars, overflows", [
    (200_000, True),    # 200000 / 6 > 32768 tokens: the prompt alone is too long
    (180_000, False),   # over max_input's 172038 characters, yet it may fit the context
])
def test_vllm_character_check(chars, overflows):
    assert is_model_rejection(400, vllm_text_check(chars)) is overflows


@pytest.mark.parametrize("message, classified", [
    # vllm/v1/engine/input_processor.py (v0.30.0): the prompt alone, longer than the context
    ("The decoder prompt (length 40000) is longer than the maximum model length of 32768. Make sure that "
     "`max_model_len` is no smaller than the number of text tokens.", "model"),
    # the prompt exactly as long as the context: no output budget fits it
    ("The decoder prompt (length 32768) plus the number of requested output tokens (at least 1) is longer than "
     "the maximum model length of 32768. Make sure that `max_model_len` is no smaller than the number of text "
     "tokens (prompt + requested output tokens).", "model"),
    # vllm/entrypoints/serve/utils/api_utils.py get_max_tokens (v0.30.0)
    ("Input length (40000) exceeds model's maximum context length (32768).", "model"),
    # vllm/entrypoints/openai/responses/serving.py (v0.30.0)
    ("The engine prompt length 40000 exceeds the max_model_len 32768. Please reduce prompt.", "model"),
    # vllm/renderers/params.py (v0.30.0): the output budget alone over the context, a parameter error
    ("max_tokens=40000 cannot be greater than max_model_len=max_total_tokens=32768. Please request fewer "
     "output tokens. (parameter=max_tokens, value=40000)", "harness"),
])
def test_other_vllm_length_messages(message, classified):
    assert hooks.classify(vllm_400(message)) == classified


@pytest.mark.parametrize("message, overflows", [
    ("This model's maximum context length is 8192 tokens. However, you requested 9000 tokens "
     "(8000 in the messages, 1000 in the completion).", False),
    ("This model's maximum context length is 8192 tokens. However, you requested 9500 tokens "
     "(9000 in the messages, 500 in the completion).", True),
    ("This endpoint's maximum context length is 131072 tokens. However, you requested about 140000 tokens "
     "(125000 of text input, 15000 in the output). Please reduce the length of either one.", False),
    ("This endpoint's maximum context length is 131072 tokens. However, you requested about 140000 tokens "
     "(135000 of text input, 5000 in the output). Please reduce the length of either one.", True),
])
def test_split_messages_are_decided_by_the_prompt_alone(message, overflows):
    """OpenAI's and OpenRouter's messages give the prompt's share: the model's only when it alone overflows."""
    assert prompt_alone_overflows(message) is overflows
    assert is_model_rejection(400, message) is overflows


def test_messages_without_lengths_keep_the_patterns():
    assert prompt_alone_overflows("prompt is too long") is None
    assert is_model_rejection(400, "prompt is too long: 250000 tokens > 200000 maximum") is True
    assert is_model_rejection(400, "Your credit balance is too low to access the Anthropic API.") is False
    assert is_model_rejection(413, "request too large") is True
