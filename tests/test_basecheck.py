"""The base-identity check's local parts (bench/basecheck.py): which rows, the per-step token means, the verdict,
and the loss of a row on a real (tiny) model. The pod's side is in tests/test_runpod.py."""
import pytest

from bench.basecheck import FAIL, PASS, UNDECIDED, last_steps, parse_tinker_losses, summarise
from bench.preflight import PreflightError
from bench.train_tinker import schedule, step_rows

SFT = {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 16, "num_train_epochs": 2, "max_steps": None,
       "warmup_steps": 0.03}
MANIFEST = {"hyperparameters": {"seed": 20260930, "sft": SFT}}


def test_the_rows_are_those_of_the_last_steps_tinker_trained_back_until_there_are_enough():
    n = 833  # c2: 53 steps an epoch, the last of one row
    selected = last_steps(MANIFEST, n, min_rows=100)
    steps = step_rows(n, schedule(n, SFT), 20260930)
    assert [r["step"] for r in selected] == [99] * 16 + [s for s in range(100, 106) for _ in range(16)] + [106]
    assert [r["index"] for r in selected] == [i for step in steps[98:] for i in step]
    assert len(selected) == 113 and last_steps(MANIFEST, n, min_rows=97)[0]["step"] == 100


def test_a_recorded_schedule_that_is_not_the_recomputed_one_refuses():
    recorded = {**MANIFEST, "stats": {"run": {"schedule": {**schedule(833, SFT), "total_steps": 105}}}}
    with pytest.raises(PreflightError, match="schedule"):
        last_steps(recorded, 833)


def _scored(adapter, base, tokens):
    return {"completion_tokens": tokens, "adapter": [a * n for a, n in zip(adapter, tokens)],
            "base": [b * n for b, n in zip(base, tokens)]}


def test_steps_are_token_means_and_the_verdict_pools_the_logged_steps():
    selected = [{"step": 100, "index": 5}, {"step": 100, "index": 7}, {"step": 106, "index": 1}]
    scored = _scored([0.1, 0.2, 0.12], [0.5, 0.7, 0.6], [30, 10, 20])
    result = summarise(selected, scored, {100: 0.125, 106: 0.12, 50: 0.4})
    step100 = result["steps"][0]
    assert step100["adapter_loss"] == pytest.approx((3 + 2) / 40) and step100["tinker_loss"] == 0.125
    assert result["status"] == PASS and result["ratio_to_tinker"] == pytest.approx(1.0)
    assert result["ignored_tinker_steps"] == [50] and result["n_rows"] == 3 and result["completion_tokens"] == 60
    assert summarise(selected, scored, {100: 0.10})["status"] == FAIL  # 25% above Tinker's
    assert summarise(selected, scored, {})["status"] == UNDECIDED
    no_effect = summarise(selected, _scored([0.1, 0.2, 0.12], [0.11, 0.2, 0.13], [30, 10, 20]), {100: 0.125})
    assert no_effect["status"] == FAIL and "adapter loaded" in no_effect["action"]


def test_tinker_losses_are_step_equals_loss():
    assert parse_tinker_losses(["100=0.1453", "106=0.1232"]) == {100: 0.1453, 106: 0.1232}
    assert parse_tinker_losses(None) == {}
    for bad in ("100", "x=0.1", "100=y"):
        with pytest.raises(PreflightError, match="STEP=LOSS"):
            parse_tinker_losses([bad])


def test_a_rows_loss_is_the_summed_cross_entropy_of_its_completion_tokens_only():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from bench.basecheck import row_nll
    from bench.train_tinker import encode_rows
    from test_train_fixtures import TINY

    tokenizer = transformers.AutoTokenizer.from_pretrained(TINY["repo"], revision=TINY["revision"])
    model = transformers.AutoModelForCausalLM.from_pretrained(TINY["repo"], revision=TINY["revision"]).eval()
    row = {"prompt": [{"role": "user", "content": "Name a colour."}],
           "completion": [{"role": "assistant", "content": "Blue, like the sea."}]}
    _, (encoded,) = encode_rows(tokenizer, [row], {"enable_thinking": False}, 4096, 4096)
    with torch.no_grad():
        got = row_nll(model, encoded, "cpu")
        logits = model(input_ids=torch.tensor([encoded["inputs"]])).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    expected = -sum(float(logp[i, t]) for i, (t, m) in enumerate(zip(encoded["targets"], encoded["mask"])) if m)
    assert sum(encoded["mask"]) < len(encoded["mask"]) and got == pytest.approx(expected, rel=1e-5)
