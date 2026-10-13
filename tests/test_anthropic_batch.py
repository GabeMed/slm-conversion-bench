"""Batch mode of the Anthropic engine (`bench run --batch`, K1 only): calls gathered into one batch and
answered by custom_id, failures classified as the API's own, an interrupted call cancels its batch, and
the runner allows it on B0 train only. A fake Batches API: nothing is sent. Agent environment only."""
import _thread
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("anthropic")
from anthropic.types import Message  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.output_parsers import JsonOutputParser  # noqa: E402

from bench import data, paths  # noqa: E402
from bench.agent import anthropic_engine, hooks  # noqa: E402
from bench.agent.runner import check_batch  # noqa: E402
from bench.contracts.calls import read_calls, validate_calls  # noqa: E402
from bench.contracts.config import load_config  # noqa: E402

CLAUDE = paths.ROOT / "configs" / "teacher-claude.yaml"
SMOKE = paths.ROOT / "configs" / "smoke-local.yaml"


def _message(text):
    return Message.model_validate({
        "id": "msg", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 90,
                  "cache_creation_input_tokens": 0, "output_tokens_details": {"thinking_tokens": 2}}})


def _errored(kind, message="no"):
    return SimpleNamespace(type="errored", error=SimpleNamespace(error=SimpleNamespace(type=kind, message=message)))


class FakeBatches:
    """Answers each request with its own last user text, unless `outcome` says otherwise; `polls` is how
    many retrieves say `in_progress` before `ended` (None: never ends)."""

    def __init__(self, outcome=None, polls=1):
        self.outcome, self.polls = outcome, polls
        self.created, self.canceled, self.retrieves = [], [], 0

    def create(self, requests):
        requests = list(requests)
        self.created.append(requests)
        return SimpleNamespace(id=f"batch_{len(self.created)}")

    def retrieve(self, batch_id):
        self.retrieves += 1
        ended = self.polls is not None and self.retrieves > self.polls
        return SimpleNamespace(processing_status="ended" if ended else "in_progress")

    def results(self, batch_id):
        requests = self.created[int(batch_id.split("_")[1]) - 1]
        for request in requests:
            content = request["params"]["messages"][-1]["content"]
            text = content if isinstance(content, str) else content[-1]["text"]
            result = (self.outcome(text) if self.outcome else None) or \
                SimpleNamespace(type="succeeded", message=_message(f"answer to {text}"))
            yield SimpleNamespace(custom_id=request["custom_id"], result=result)

    def cancel(self, batch_id):
        self.canceled.append(batch_id)


@pytest.fixture
def batch(monkeypatch):
    """Batch mode on, a fake Batches API on the teacher's client, short waits."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv(anthropic_engine.BATCH_ENV, "1")
    monkeypatch.setattr(anthropic_engine, "BATCH_GATHER_S", 0.2)
    monkeypatch.setattr(anthropic_engine, "BATCH_POLL_S", (0.01, 0.02))
    monkeypatch.setattr(anthropic_engine, "_WARMUP", anthropic_engine._Warmup())
    fake = FakeBatches()
    batcher = anthropic_engine._Batcher(SimpleNamespace(messages=SimpleNamespace(batches=fake)))
    monkeypatch.setattr(anthropic_engine, "_batcher", lambda client: batcher)  # every client: the fake only

    def no_network(*args, **kwargs):
        raise AssertionError("a batch-mode test reached the streaming API")
    monkeypatch.setattr("anthropic.resources.messages.Messages.stream", no_network)
    hooks.configure(load_config(CLAUDE))
    model = hooks.chat_model("production_llm", 0.0)
    yield model, fake
    hooks.configure(load_config(SMOKE))


def _in_parallel(model, texts):
    out, threads = {}, []
    for text in texts:
        def run(text=text):
            try:
                out[text] = model.invoke([HumanMessage(content=text)])
            except Exception as e:  # noqa: BLE001
                out[text] = e
        threads.append(threading.Thread(target=run))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def test_calls_made_together_are_one_batch_and_each_gets_its_own_answer(batch):
    model, fake = batch
    out = _in_parallel(model, [f"q{i}" for i in range(12)])
    assert len(fake.created) == 1 and len(fake.created[0]) == 12
    assert all(out[f"q{i}"].content == f"answer to q{i}" for i in range(12))
    meta = out["q0"].response_metadata
    assert meta["finish_reason"] == "stop" and meta["token_usage"]["prompt_tokens"] == 100
    params = fake.created[0][0]["params"]
    assert "temperature" not in params and "extra_body" not in params and "timeout" not in params
    assert params["output_config"] == {"effort": "medium"} and params["model"] == "claude-opus-5-5"


def test_calls_apart_in_time_are_separate_batches(batch):
    model, fake = batch
    model.invoke([HumanMessage(content="first")])
    model.invoke([HumanMessage(content="second")])
    assert [len(b) for b in fake.created] == [1, 1]


def test_the_first_call_of_a_cached_prefix_is_a_batch_of_its_own(batch):
    """It writes the cache; the calls that share its prefix go together after it, and read it."""
    model, fake = batch
    prefix = "Instructions.\nColumn information:\n"
    _in_parallel(model, [f"{prefix}column {i}" for i in range(6)])
    assert [len(b) for b in fake.created] == [1, 5]
    first = fake.created[0][0]["params"]["messages"][0]["content"][0]
    assert first["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.parametrize("kind, message, classified", [
    ("invalid_request_error", "prompt is too long", "model"),
    ("invalid_request_error", "Your credit balance is too low", "harness"),
    ("billing_error", "no", "harness"),
    ("overloaded_error", "no", "transport"),
    ("api_error", "no", "transport"),
])
def test_a_failed_request_is_classified_as_the_apis_own_status(batch, kind, message, classified):
    model, fake = batch
    fake.outcome = lambda text: _errored(kind, message)
    with pytest.raises(anthropic_engine.BatchCallError) as caught:
        model.invoke([HumanMessage(content="q")])
    assert hooks.classify(caught.value) == classified


def test_an_expired_request_is_sent_again_as_a_transport_failure(batch):
    model, fake = batch
    fake.outcome = lambda text: SimpleNamespace(type="expired")
    with pytest.raises(anthropic_engine.BatchCallError) as caught:
        model.invoke([HumanMessage(content="q")])
    assert hooks.classify(caught.value) == "transport"


def test_an_interrupted_call_cancels_its_open_batch(batch):
    model, fake = batch
    fake.polls = None  # the batch never ends
    threading.Timer(0.5, _thread.interrupt_main).start()
    with pytest.raises(KeyboardInterrupt):
        model.invoke([HumanMessage(content="q")])
    time.sleep(0.05)
    assert fake.canceled == ["batch_1"]


def test_a_batched_call_reaches_c1_as_every_call_does(batch, tmp_path):
    hooks.start_run("test-run", "B0", tmp_path / "calls.jsonl")
    try:
        hooks.set_question("1470")
        model, fake = batch
        fake.outcome = lambda text: SimpleNamespace(type="succeeded", message=_message('{"keywords": ["k"]}'))
        hooks.invoke_tool_call("extract_keywords", "single", [HumanMessage(content="q")], JsonOutputParser())
    finally:
        hooks.end_run()
    (line,) = read_calls(tmp_path / "calls.jsonl")
    assert line["parsed_ok"] and line["usage"]["cached_input"] == 90 and line["usage"]["cache_write"] == 0
    assert validate_calls([line]) == []


@pytest.mark.parametrize("arm, split", [("B0", "calib"), ("B0", "test"), ("B1", "train"), ("B3", "train")])
def test_batch_is_refused_outside_k1(arm, split):
    with pytest.raises(data.DataError, match="K1 only"):
        check_batch(load_config(CLAUDE), arm, split)


def test_batch_needs_an_anthropic_teacher():
    with pytest.raises(data.DataError, match="protocol: anthropic"):
        check_batch(load_config(SMOKE), "B0", "train")
    check_batch(load_config(CLAUDE), "B0", "train")  # K1 on Claude: allowed


def test_a_batch_that_fails_with_a_timeout_raises_it_and_never_hangs(batch):
    """A TimeoutError from the API is the call's failure (a transport one), not the wait's own clock."""
    model, fake = batch

    def create(requests):
        raise TimeoutError("the API did not answer")
    fake.create = create
    with pytest.raises(TimeoutError):
        model.invoke([HumanMessage(content="q")])
    assert hooks.classify(TimeoutError()) == "transport"
