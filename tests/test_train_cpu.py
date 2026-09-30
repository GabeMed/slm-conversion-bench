"""Acceptance (design §3.3, F3): `bench train` trains on a CPU with a tiny model for a few steps, through the
same `train_lora` the Modal GPU function runs, writes the adapter and its manifest, and registers the
`adapters` fact once every cluster has one. Runs in env/train (torch, TRL, PEFT); downloads a ~21 MB
random-weights Qwen3 once into the HF cache."""
import json

import pytest

pytest.importorskip("trl")

from bench import cli, paths  # noqa: E402
from bench.contracts import facts, router  # noqa: E402
from bench.train import (TrainError, check_loss_tokens, check_rows, check_target_modules, precheck,  # noqa: E402
                         training_plan, train_lora)
from test_train_fixtures import TINY, TINY_NAME, make_s5_repo, rel, rows, save  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TINY["repo"], revision=TINY["revision"])


def test_bench_train_on_cpu_writes_the_adapter_its_manifest_and_then_the_fact(tmp_path, monkeypatch):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    assert cli.main(["train", "--config", str(config_path), "--cluster", "c0", "--on", "local"]) == 0
    adapter = paths.ROOT / "train" / "adapters" / "c0" / "adapter"
    assert sorted(p.name for p in adapter.iterdir()) == ["README.md", "adapter_config.json", "adapter_model.safetensors"]
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == TINY["repo"] and adapter_config["revision"] == TINY["revision"]
    assert sorted(adapter_config["target_modules"]) == sorted(config["train"]["lora"]["target_modules"])
    assert adapter_config["r"] == 8

    manifest = json.loads((adapter.parent / "manifest.json").read_text())
    assert manifest["adapter_sha256"] == facts.sha256_dir(adapter)
    assert manifest["base"] == TINY and manifest["slm"] == TINY_NAME and manifest["served_name"] == "c0"
    assert manifest["dataset"]["rows"] == 4 and len(manifest["dataset"]["sha256"]) == 64
    assert manifest["hyperparameters"]["sft"]["max_steps"] == 2
    assert manifest["where"] == "local" and manifest["gpu"] is None
    assert manifest["gpu_seconds"] == 0.0 and manifest["cost_usd"] == 0.0
    stats = manifest["stats"]
    assert stats["device"] == "cpu" and stats["dtype"] == "float32" and stats["global_step"] == 2
    assert stats["examples_seen"] == 4 and stats["train_seconds"] > 0
    assert stats["versions"]["trl"] == "1.14.1" and stats["versions"]["peft"] == "0.21.1"
    assert not list((paths.ROOT / "judgments").glob("S5/*"))  # c1 has no adapter yet

    assert cli.main(["train", "--config", str(config_path), "--cluster", "c1", "--on", "local"]) == 0
    fact_path = next((paths.ROOT / "judgments" / "S5").glob("*/adapters.json"))
    fact, _ = facts.read_fact(str(fact_path), "adapters")
    assert {c: a["sha256"] for c, a in fact["adapters"].items()} == {
        c: facts.sha256_dir(paths.ROOT / "train" / "adapters" / c / "adapter") for c in ("c0", "c1")}
    config["arms"]["B4"]["adapters"] = rel(fact_path)
    assert "adapters" in router.arm_facts("B4", save(config, config_path))


def test_a_failed_training_leaves_the_previous_adapter_in_place(tmp_path, monkeypatch):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    assert cli.main(["train", "--config", str(config_path), "--cluster", "c0", "--on", "local"]) == 0
    before = facts.sha256_dir(paths.ROOT / "train" / "adapters" / "c0" / "adapter")
    config["train"]["lora"]["target_modules"] = ["q_proj", "in_proj"]
    assert cli.main(["train", "--config", str(save(config, config_path) and config_path), "--cluster", "c0",
                     "--on", "local"]) == 2
    assert facts.sha256_dir(paths.ROOT / "train" / "adapters" / "c0" / "adapter") == before
    assert [p.name for p in (paths.ROOT / "train" / "adapters").iterdir()] == ["c0"]  # no scratch left behind


def test_training_renders_what_serving_asks_and_learns_only_the_answer(tokenizer):
    """Thinking off: the prompt, with the generation prompt, is a prefix of prompt + completion, and the
    tokens under loss are the answer and the end of turn, never a think block."""
    row = rows("c0", 1)[0]
    kwargs = {"enable_thinking": False}
    stats, completions = check_rows(tokenizer, [row], kwargs, max_length=512)
    prompt = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True, return_dict=True, **kwargs)["input_ids"]
    full = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True, return_dict=True, **kwargs)["input_ids"]
    completion = tokenizer.decode(full[len(prompt):])
    assert completion.strip() == row["completion"][0]["content"] + "<|im_end|>"
    assert "<think>" not in completion and tokenizer.decode(prompt).endswith("<think>\n\n</think>\n\n")
    assert stats["completion_tokens"] == len(full) - len(prompt) and completions == [full[len(prompt):]]


def test_rows_serving_would_not_render_the_same_way_or_too_long_are_refused(tokenizer):
    row = rows("c0", 1)[0]
    with pytest.raises(TrainError, match="over train.sft.max_length"):
        check_rows(tokenizer, [row], {"enable_thinking": False}, max_length=10)
    original = tokenizer.chat_template
    try:  # a template whose generation prompt is not how it renders an assistant turn
        tokenizer.chat_template = ("{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}"
                                   "{% if add_generation_prompt %}<gen>{% endif %}")
        with pytest.raises(TrainError, match="not render the prompt as a prefix"):
            check_rows(tokenizer, [row], {}, max_length=512)
    finally:
        tokenizer.chat_template = original


def test_a_target_module_the_base_lacks_is_refused():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(TINY["repo"], revision=TINY["revision"])
    check_target_modules(model, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    with pytest.raises(TrainError, match=r"no module named \['in_proj'\]"):
        check_target_modules(model, ["q_proj", "in_proj"])


def test_train_lora_is_seeded(tmp_path):
    plan = {"base": TINY, "chat_template_kwargs": {"enable_thinking": False},
            "hyperparameters": {"seed": 7, "precision": "bf16",
                                "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
                                "sft": {"learning_rate": 1e-3, "num_train_epochs": 1, "max_steps": 2,
                                        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
                                        "max_length": 512, "lr_scheduler_type": "constant", "warmup_steps": 0,
                                        "gradient_checkpointing": False, "logging_steps": 1}}}
    data = rows("c0", 4)
    train_lora(plan, data, tmp_path / "a", "cpu")
    train_lora(plan, data, tmp_path / "b", "cpu")
    assert (tmp_path / "a" / "adapter_model.safetensors").read_bytes() == (tmp_path / "b" / "adapter_model.safetensors").read_bytes()


def test_the_loss_must_fall_exactly_on_the_completion_tokens():
    completions = [[7, 8, 2]]
    check_loss_tokens([{"input_ids": [1, 2, 7, 8, 2], "labels": [-100, -100, 7, 8, 2]}], completions)
    with pytest.raises(TrainError, match="not the 3 completion tokens"):  # loss on the prompt too
        check_loss_tokens([{"input_ids": [1, 2, 7, 8, 2], "labels": [1, 2, 7, 8, 2]}], completions)
    with pytest.raises(TrainError, match="not the 3 completion tokens"):  # a think block under the loss
        check_loss_tokens([{"input_ids": [1, 9, 7, 8, 2], "labels": [-100, 9, 7, 8, 2]}], completions)
    with pytest.raises(TrainError, match="kept 0 of 1 rows"):
        check_loss_tokens([], completions)
    check_loss_tokens([{"input_ids": [1, 7, 8, 2], "completion_mask": [0, 1, 1, 1]}], completions)


def test_a_kwarg_the_template_never_reads_is_refused(tokenizer):
    row = rows("c0", 1)[0]
    with pytest.raises(TrainError, match=r"does not read \['reasoning_mode'\]"):
        check_rows(tokenizer, [row], {"enable_thinking": False, "reasoning_mode": "off"}, max_length=512)


def test_precheck_runs_the_template_checks_with_the_candidates_tokenizer(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    plan, _ = training_plan(config, "c0")
    assert precheck(plan, rows("c0", 2))["completion_tokens"] > 0
    with pytest.raises(TrainError, match="over train.sft.max_length"):
        precheck({**plan, "hyperparameters": {**plan["hyperparameters"], "sft": {"max_length": 5}}}, rows("c0", 1))
