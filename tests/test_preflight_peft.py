"""P-4's HF-PEFT side on a real (tiny) model: the reference generations, and the whole local P-4 path with a
fake vLLM that serves exactly what PEFT computes. Runs in env/train."""
import json

import pytest

pytest.importorskip("peft")

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from bench import paths  # noqa: E402
from bench.contracts import facts  # noqa: E402
from bench.preflight import PASS, lora_parity, peft_generate  # noqa: E402
from test_preflight import FakeVLLM  # noqa: E402
from test_train_fixtures import TINY, TINY_NAME, make_s5_repo, save  # noqa: E402

KWARGS = {"enable_thinking": False}


def _random_adapter(path):
    """A LoRA adapter with random (not zero) B matrices, so it certainly changes the base's output."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY["repo"], revision=TINY["revision"])
    config = LoraConfig(r=8, lora_alpha=64, target_modules=["q_proj", "v_proj", "down_proj"], init_lora_weights=False)
    get_peft_model(model, config).save_pretrained(str(path))
    return path


def test_the_reference_base_is_the_base_model_and_the_adapter_changes_it(tmp_path):
    adapter = _random_adapter(tmp_path / "adapter")
    prompts = [[{"role": "user", "content": f"Name a colour, number {i}."}] for i in range(2)]
    ref = peft_generate(TINY, str(adapter), prompts, KWARGS, max_new_tokens=6, top_k=3, device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(TINY["repo"], revision=TINY["revision"])
    plain = AutoModelForCausalLM.from_pretrained(TINY["repo"], revision=TINY["revision"]).eval()
    assert ref["stop_ids"]
    for i, messages in enumerate(prompts):
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", **KWARGS)
        expected = plain.generate(**inputs, max_new_tokens=6, do_sample=False)[0, inputs["input_ids"].shape[1]:].tolist()
        assert ref["base"][i]["tokens"] == expected
        for which in ("base", "adapter"):
            generation = ref[which][i]
            assert len(generation["top"]) == len(generation["tokens"]) and all(len(t) == 3 for t in generation["top"])
            assert all(tok == top[0] for tok, top in zip(generation["tokens"], generation["top"]))  # greedy = top-1
    assert any(ref["adapter"][i]["tokens"] != ref["base"][i]["tokens"] for i in range(len(prompts)))


def test_local_p4_passes_when_the_server_reproduces_peft(tmp_path, monkeypatch):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    adapter = _random_adapter(paths.ROOT / "train" / "adapters" / "c0" / "adapter")
    manifest = {"cluster": "c0", "slm": TINY_NAME, "served_name": "c0", "base": TINY, "chat_template_kwargs": KWARGS,
                "adapter_sha256": facts.sha256_dir(adapter)}
    (adapter.parent / "manifest.json").write_text(json.dumps(manifest))
    settings = config["preflight"]["lora_parity"]
    settings.update({"n_prompts": 2, "max_new_tokens": 5})
    prompts = [json.loads(line)["prompt"] for line in (paths.ROOT / "train/datasets/c0.jsonl").read_text().splitlines()][:2]
    ref = peft_generate(TINY, str(adapter), prompts, KWARGS, 5, settings["top_logprobs"], "cpu")
    server = FakeVLLM({TINY_NAME: ref["base"], "c0": ref["adapter"]}, prompts)  # a vLLM that matches HF exactly
    try:
        config["roles"]["slm_candidates"][-1]["endpoint"]["base_url"] = server.base_url
        verdict = lora_parity(save(config, config_path), "c0", "local")
    finally:
        server.close()
    assert verdict["status"] == PASS, verdict["diagnosis"]
    assert verdict["base_matches_hf"] == verdict["adapter_matches_peft"] == 2 and verdict["reference_on"] == "local"
