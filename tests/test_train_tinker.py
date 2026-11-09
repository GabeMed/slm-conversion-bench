"""`bench train --on tinker` without Tinker: rows to datums with the loss on the completion only, the steps and
the cosine schedule as transformers' Trainer counts them, the refusals (no SDK, a base Tinker does not offer, a
hyper-parameter it cannot honour), and the whole command against a fake SDK and a fake export (no network, no
TINKER_API_KEY, no torch)."""
import json
import math
import sys
import types
from pathlib import Path

import pytest
import yaml

from bench import cli, paths
from bench.contracts import facts
from bench.train import TrainError, plan_id, training_plan
from bench import train_tinker as tt
from test_train_fixtures import TINY, make_s5_repo, rows

SHIPPED = paths.ROOT / "configs" / "train-tinker.yaml"
ASSISTANT, END = 900, 901  # the fake template's assistant header and end of message


class FakeTokenizer:
    """A chat template that renders each message as its role and characters, and the generation prompt as the
    assistant header, so the prompt with it is a prefix of prompt + completion (as Qwen3's with thinking off)."""
    chat_template = "{% if enable_thinking %}{% endif %}"

    def apply_chat_template(self, messages, tokenize, return_dict, add_generation_prompt, enable_thinking=True):
        ids = []
        for m in messages:
            ids += ([ASSISTANT] if m["role"] == "assistant" else [800 + ("system", "user").index(m["role"])])
            ids += [ord(c) % 500 for c in m["content"]] + [END]
        return {"input_ids": ids + ([ASSISTANT] if add_generation_prompt else [])}


def settings(tmp_path, **changes):
    data = yaml.safe_load(SHIPPED.read_text())
    data["models"] = {TINY["repo"]: {"tinker_id": "tiny/Tinker-Qwen3", "max_context": 4096}}
    data.update(changes)
    path = tmp_path / "train-tinker.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


def fake_sdk(calls, version="0.39.0", offered=("tiny/Tinker-Qwen3",), rank=None):
    """The parts of `tinker` bench/train_tinker.py uses, recording every call in `calls`."""

    class TrainingClient:
        def get_info(self):
            return types.SimpleNamespace(model_id="run-1", lora_rank=rank)

        def forward_backward(self, datums, loss_fn):
            calls["forward_backward"].append((datums, loss_fn))
            loss = -sum(w * -1.0 for d in datums for w in d.loss_fn_inputs["weights"])  # logprob -1 per token
            return Future(types.SimpleNamespace(metrics={"loss:sum": loss}))

        def optim_step(self, params):
            calls["optim_step"].append(params)
            return Future(None)

        def save_weights_external(self, name, ttl_seconds=None):
            calls["save_weights_external"].append((name, ttl_seconds))
            return Future(types.SimpleNamespace(path=f"tinker://run-1/external_weights/{name}"))

    class ServiceClient:
        def __init__(self, user_metadata=None):
            calls["ServiceClient"].append(user_metadata)

        def create_rest_client(self):
            models = [types.SimpleNamespace(model_name=m, trainable=True, max_context_length=32768) for m in offered]
            return types.SimpleNamespace(get_server_capabilities=lambda: Future(types.SimpleNamespace(supported_models=models)))

        def create_lora_training_client(self, base_model, **lora):
            calls["create_lora_training_client"].append({"base_model": base_model, **lora})
            return Future(TrainingClient())

        def close(self, status):
            calls["close"].append(status)
            return Future(None)

    class Datum:
        def __init__(self, model_input, loss_fn_inputs):
            self.model_input, self.loss_fn_inputs = model_input, loss_fn_inputs

    return types.SimpleNamespace(
        __version__=version, ServiceClient=ServiceClient, Datum=Datum,
        ModelInput=types.SimpleNamespace(from_ints=lambda tokens: list(tokens)),
        AdamParams=lambda **params: params)


def fake_export(calls, alpha=16, modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
    def export(external_weights, plan, out: Path):
        calls["export"].append(external_weights)
        out.mkdir()
        (out / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "base_model_name_or_path": "/hf-cache/snapshots/x", "r": plan["hyperparameters"]["lora"]["r"],
            "lora_alpha": alpha, "lora_dropout": 0.0, "target_modules": list(modules), "rank_pattern": {}, "alpha_pattern": {}}))
        (out / "adapter_model.safetensors").write_bytes(b"weights of " + external_weights.encode())
    return export


@pytest.fixture
def tinker(monkeypatch):
    """A fake SDK, tokenizer and export in place; returns the calls they record."""
    calls = {k: [] for k in ("ServiceClient", "create_lora_training_client", "forward_backward", "optim_step",
                             "save_weights_external", "close", "export")}
    monkeypatch.setattr(tt, "_sdk", lambda: fake_sdk(calls))
    monkeypatch.setattr(tt, "_tokenizer", lambda plan: FakeTokenizer())
    monkeypatch.setattr(tt, "export_adapter", fake_export(calls))
    return calls


def train(config_path, settings_path, cluster="c0"):
    return cli.main(["train", "--config", str(config_path), "--cluster", cluster, "--on", "tinker",
                     "--tinker-config", str(settings_path)])


# ---------------------------------------------------------------- rows to datums

def test_a_row_trains_on_its_completion_tokens_only():
    row = rows("c0", 1)[0]
    tok = FakeTokenizer()
    (stats, [e]) = tt.encode_rows(tok, [row], {"enable_thinking": False}, 512, 4096)
    full = tok.apply_chat_template(row["prompt"] + row["completion"], True, True, False)["input_ids"]
    prompt = tok.apply_chat_template(row["prompt"], True, True, True)["input_ids"]
    assert e["inputs"] == full[:-1] and e["targets"] == full[1:]  # next-token: input t predicts t + 1
    assert len(e["mask"]) == len(e["targets"])
    assert [t for t, m in zip(e["targets"], e["mask"]) if m] == full[len(prompt):]  # the completion, END included
    assert e["mask"].index(1) == len(prompt) - 1 and all(e["mask"][len(prompt) - 1:])
    assert stats == {"longest_row_tokens": len(full), "tokens": len(full), "completion_tokens": len(full) - len(prompt)}


def test_a_row_over_tinkers_context_is_refused_never_truncated():
    row = rows("c0", 1)[0]
    with pytest.raises(TrainError, match="over Tinker's context"):
        tt.encode_rows(FakeTokenizer(), [row], {"enable_thinking": False}, 512, 10)
    with pytest.raises(TrainError, match="over train.sft.max_length"):  # the TRL path's own check comes first
        tt.encode_rows(FakeTokenizer(), [row], {"enable_thinking": False}, 10, 4096)


def test_each_step_weighs_its_completion_tokens_to_a_mean():
    encoded = [{"inputs": [0] * 5, "targets": [0] * 5, "mask": [0, 0, 1, 1, 1]},
               {"inputs": [0] * 3, "targets": [0] * 3, "mask": [0, 1, 0]}]
    data = tt.step_data(encoded, [0, 1])
    assert data[0]["weights"] == [0, 0, 0.25, 0.25, 0.25] and data[1]["weights"] == [0, 0.25, 0]
    assert sum(w for d in data for w in d["weights"]) == pytest.approx(1)  # Tinker's sum = TRL's token mean


# ---------------------------------------------------------------- steps and schedule

SFT = {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 16, "num_train_epochs": 2, "max_steps": None,
       "warmup_steps": 0.03, "learning_rate": 2e-4}


def test_steps_are_counted_as_the_trainer_counts_them():
    assert tt.schedule(100, SFT) == {"rows_per_step": 16, "steps_per_epoch": 7, "total_steps": 14, "warmup_steps": 1}
    assert tt.schedule(1000, SFT)["warmup_steps"] == math.ceil(126 * 0.03) == 4
    assert tt.schedule(100, {**SFT, "max_steps": 3}) == {"rows_per_step": 16, "steps_per_epoch": 7, "total_steps": 3,
                                                          "warmup_steps": 1}
    assert tt.schedule(100, {**SFT, "warmup_steps": 5})["warmup_steps"] == 5
    with pytest.raises(TrainError, match="warmup_steps"):
        tt.schedule(100, {**SFT, "warmup_steps": 2.5})


def test_the_learning_rate_is_cosine_with_linear_warm_up():
    sched = {"warmup_steps": 4, "total_steps": 104}
    lr = [tt.learning_rate(s, sched, 2e-4) for s in range(104)]
    assert lr[:5] == pytest.approx([0, 0.5e-4, 1e-4, 1.5e-4, 2e-4])
    assert lr[54] == pytest.approx(1e-4)  # half way through the decay
    assert all(a >= b for a, b in zip(lr[4:], lr[5:])) and lr[-1] == pytest.approx(2e-4 * 0.5 * (1 + math.cos(math.pi * 99 / 100)))


def test_every_epoch_takes_every_row_once_in_a_seeded_order():
    sched = tt.schedule(5, {**SFT, "gradient_accumulation_steps": 2, "num_train_epochs": 3})
    steps = tt.step_rows(5, sched, 7)
    assert [len(s) for s in steps] == [2, 2, 1] * 3
    assert all(sorted(sum(steps[i:i + 3], [])) == list(range(5)) for i in (0, 3, 6))
    assert steps == tt.step_rows(5, sched, 7) and steps != tt.step_rows(5, sched, 8)
    assert len(tt.step_rows(5, {**sched, "total_steps": 10}, 7)) == 10  # max_steps runs on past the epochs


# ---------------------------------------------------------------- what Tinker can and cannot honour

def registered_plan(repo="Qwen/Qwen3-8B"):
    """A plan with config.yaml's own train section (what the registered run would hand Tinker)."""
    train = yaml.safe_load((paths.ROOT / "config.yaml").read_text())["train"]
    return {"slm": "candidate", "base": {"repo": repo, "revision": "0" * 40},
            "hyperparameters": {"seed": train["seed"], "precision": train["precision"],
                                "lora": dict(train["lora"]), "sft": dict(train["sft"])}}


def test_the_registered_hyperparameters_map_onto_tinker():
    run = tt.tinker_plan(registered_plan(), tt.load_settings(str(SHIPPED)))
    assert run["base_model"] == "Qwen/Qwen3-8B"
    assert run["lora"] == {"rank": 16, "seed": 20260930, "train_attn": True, "train_mlp": True, "train_unembed": False}
    assert run["optimizer"] == {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": 1.0}
    assert set(run["differences"]) == {"base.revision", "lora.alpha", "lora.dropout", "lora.init", "precision",
                                       "sample_order"}


@pytest.mark.parametrize("repo", ["ibm-granite/granite-4.2-8b", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"])
def test_a_base_tinker_does_not_offer_is_refused(repo):
    with pytest.raises(TrainError, match=f"Tinker does not offer {repo}"):
        tt.tinker_plan(registered_plan(repo), tt.load_settings(str(SHIPPED)))


@pytest.mark.parametrize("change, match", [
    (lambda hp: hp["sft"].update(neftune_noise_alpha=5), r"does not know how to honour \['train.sft.neftune"),
    (lambda hp: hp["lora"].update(use_rslora=True), r"train.lora.use_rslora"),
    (lambda hp: hp.update(packing=True), r"train.packing"),
    (lambda hp: hp["lora"].update(target_modules=["q_proj", "v_proj"]), "not a union of them"),
    (lambda hp: hp["lora"].update(target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]), "not a union"),
    (lambda hp: hp["sft"].update(lr_scheduler_type="linear"), "cosine only"),
])
def test_a_hyperparameter_tinker_cannot_honour_is_refused(change, match):
    plan = registered_plan()
    change(plan["hyperparameters"])
    with pytest.raises(TrainError, match=match):
        tt.tinker_plan(plan, tt.load_settings(str(SHIPPED)))


def test_a_difference_not_accepted_refuses_and_one_that_does_not_arise_is_not_asked(tmp_path):
    accepted = ["base.revision", "lora.alpha", "lora.init", "precision", "sample_order"]
    path = settings(tmp_path, models={"Qwen/Qwen3-8B": {"tinker_id": "Qwen/Qwen3-8B", "max_context": 32768}},
                    accept_differences=accepted)
    with pytest.raises(TrainError, match=r"lora.dropout \(Tinker trains without LoRA dropout\).*accept_differences"):
        tt.tinker_plan(registered_plan(), tt.load_settings(str(path)))
    plan = registered_plan()
    plan["hyperparameters"]["lora"]["dropout"] = 0.0
    assert "lora.dropout" not in tt.tinker_plan(plan, tt.load_settings(str(path)))["differences"]


@pytest.mark.parametrize("change, match", [
    ({"accept_differences": ["lora.dropuot"]}, "accept_differences"),
    ({"optimizer": {"beta1": 0.9}}, "optimizer must set exactly"),
    ({"lora_groups": {"train_attn": ["q_proj"]}}, "lora_groups"),
    ({"export_ttl_s": 60}, "export_ttl_s"),
])
def test_malformed_settings_are_refused(tmp_path, change, match):
    with pytest.raises(TrainError, match=match):
        tt.load_settings(str(settings(tmp_path, **change)))


# ---------------------------------------------------------------- the command, against a fake service

def test_bench_train_on_tinker_writes_the_adapter_and_manifest_as_modal_does(tmp_path, monkeypatch, tinker, capsys):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    assert train(config_path, settings(tmp_path)) == 0
    adapter = paths.ROOT / "train" / "adapters" / "c0" / "adapter"
    assert sorted(p.name for p in adapter.iterdir()) == ["adapter_config.json", "adapter_model.safetensors"]
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == TINY["repo"] and adapter_config["revision"] == TINY["revision"]

    # the service: rank and seed from config.yaml, no unembedding; 2 steps (max_steps) of 2 rows, the loss
    # weighted to the step's completion-token mean, AdamW with the TRL path's defaults at the scheduled rate
    assert tinker["create_lora_training_client"] == [{"base_model": "tiny/Tinker-Qwen3", "rank": 8,
                                                     "seed": config["train"]["seed"], "train_attn": True,
                                                     "train_mlp": True, "train_unembed": False}]
    assert [len(d) for d, _ in tinker["forward_backward"]] == [2, 2]
    assert {loss for _, loss in tinker["forward_backward"]} == {"cross_entropy"}
    for datums, _ in tinker["forward_backward"]:
        assert sum(w for d in datums for w in d.loss_fn_inputs["weights"]) == pytest.approx(1)
        assert all(len(d.model_input) == len(d.loss_fn_inputs["target_tokens"]) == len(d.loss_fn_inputs["weights"])
                   for d in datums)
    sched = tt.schedule(4, config["train"]["sft"])
    assert [p["learning_rate"] for p in tinker["optim_step"]] == [tt.learning_rate(s, sched, 2e-4) for s in (0, 1)]
    assert all(p["beta2"] == 0.999 and p["eps"] == 1e-8 and p["grad_clip_norm"] == 1.0 for p in tinker["optim_step"])
    assert tinker["close"] == ["success"] and tinker["export"] == ["tinker://run-1/external_weights/" +
                                                                    tinker["save_weights_external"][0][0]]

    manifest = json.loads((adapter.parent / "manifest.json").read_text())
    assert manifest["where"] == "tinker" and manifest["adapter_sha256"] == facts.sha256_dir(adapter)
    assert manifest["served_name"] == f"c0-{manifest['adapter_sha256'][:12]}"
    assert manifest["base"] == TINY and manifest["hyperparameters"]["lora"]["dropout"] == 0.05
    assert manifest["gpu"] is None and manifest["cost_usd"] is None and manifest["gpu_seconds"] is None  # billed per token
    assert len(manifest["tinker_key"]) == 64 and "commit" in manifest
    plan = training_plan(config, "c0")[0]
    assert set(plan) == set(tt.PLAN_KEYS)  # the whole plan is in every manifest, whatever the backend
    assert manifest["plan_sha256"] == tt.plan_sha256(manifest) == plan_id(plan)
    stats = manifest["stats"]
    assert stats["backend"] == "tinker" and stats["run"]["base_model"] == "tiny/Tinker-Qwen3"
    assert stats["external_weights"].startswith("tinker://run-1/external_weights/") and stats["training_run"] == "run-1"
    assert stats["global_step"] == 2 and stats["examples_seen"] == 4 and not stats["collected_from_earlier_run"]
    assert stats["train_loss"] == pytest.approx(1)  # the fake's log-probability is -1 on every token: mean NLL 1
    assert stats["tokens_seen"] == sum(len(d.model_input) for ds, _ in tinker["forward_backward"] for d in ds)
    assert stats["completion_tokens_seen"] == stats["completion_tokens"]  # 4 rows, once each
    assert stats["trained"] == {"r": 8, "lora_alpha": 16, "target_modules": sorted(config["train"]["lora"]["target_modules"]),
                                "rank_pattern": {}, "alpha_pattern": {}}
    assert "lora.dropout" in stats["run"]["differences"] and stats["versions"]["tinker"] == "0.39.0"
    assert "price" not in json.dumps(stats)
    assert not list((paths.ROOT / "train" / "adapters" / ".tinker").iterdir())  # the pending record is cleared
    out = capsys.readouterr().out
    assert "prereg/DEVIATIONS.md" in out and f"/{manifest['adapter_sha256']}" in out

    # again: nothing is trained, the adapter is kept, the set is registered once c1 has one too
    assert train(config_path, settings(tmp_path)) == 0
    assert len(tinker["ServiceClient"]) == 1 and "nothing was trained" in capsys.readouterr().out
    assert train(config_path, settings(tmp_path), "c1") == 0
    assert "adapters fact:" in capsys.readouterr().out


def test_a_tinker_run_and_a_modal_run_of_the_same_plan_pair(tmp_path, monkeypatch, tinker):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    assert train(config_path, settings(tmp_path)) == 0
    adapters = paths.ROOT / "train" / "adapters"
    tinker_manifest = json.loads((adapters / "c0" / "manifest.json").read_text())
    (adapters / "c0").rename(tmp_path / "tinker-c0")  # set aside: the Modal run replaces train/adapters/c0
    from test_train_fixtures import fake_adapter

    fake_adapter("c0", config)  # as `--on modal` leaves it: the plan's keys, copied by write_manifest
    modal_manifest = json.loads((adapters / "c0" / "manifest.json").read_text())
    assert tt.plan_sha256(modal_manifest) == tinker_manifest["plan_sha256"]
    assert modal_manifest["adapter_sha256"] != tinker_manifest["adapter_sha256"]


def test_an_adapter_from_another_plan_or_backend_is_never_replaced(tmp_path, monkeypatch, tinker, capsys):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    from test_train_fixtures import fake_adapter

    fake_adapter("c0", config)
    assert train(config_path, settings(tmp_path)) == 2
    assert "holds another adapter" in capsys.readouterr().err and not tinker["ServiceClient"]


def test_an_adapter_with_another_alpha_is_not_installed_and_the_trained_weights_are_collected_later(
        tmp_path, monkeypatch, tinker, capsys):
    _, config_path, _ = make_s5_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(tt, "export_adapter", fake_export(tinker, alpha=32))
    assert train(config_path, settings(tmp_path)) == 2
    assert "'lora_alpha': 32" in capsys.readouterr().err
    assert not (paths.ROOT / "train" / "adapters" / "c0").exists()
    [pending] = (paths.ROOT / "train" / "adapters" / ".tinker").iterdir()
    # the same command downloads the exported weights again instead of training twice
    monkeypatch.setattr(tt, "export_adapter", fake_export(tinker))
    assert train(config_path, settings(tmp_path)) == 0
    assert len(tinker["ServiceClient"]) == 1 and len(tinker["export"]) == 2 and not pending.exists()
    manifest = json.loads((paths.ROOT / "train" / "adapters" / "c0" / "manifest.json").read_text())
    assert manifest["stats"]["collected_from_earlier_run"]


@pytest.mark.parametrize("sdk, match", [
    ({"version": "0.38.0"}, "tinker 0.38.0 is installed, configs/train-tinker.yaml pins 0.39.0"),
    ({"offered": ("Qwen/Qwen3-8B",)}, "the Tinker service does not train tiny/Tinker-Qwen3 today"),
    ({"rank": 32}, "rank 32, not 8"),
])
def test_what_the_service_answers_is_checked(tmp_path, monkeypatch, tinker, capsys, sdk, match):
    _, config_path, _ = make_s5_repo(tmp_path, monkeypatch)
    calls = {k: [] for k in tinker}
    monkeypatch.setattr(tt, "_sdk", lambda: fake_sdk(calls, **sdk))
    assert train(config_path, settings(tmp_path)) == 2
    assert match in capsys.readouterr().err
    assert calls["close"] in ([], ["errored"]) and not calls["forward_backward"]


def test_without_the_sdk_tinker_is_refused_and_modal_does_not_need_it(tmp_path, monkeypatch, capsys):
    _, config_path, _ = make_s5_repo(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "tinker", None)  # `import tinker` raises ImportError
    assert train(config_path, settings(tmp_path)) == 2
    assert "needs the Tinker SDK, which is optional: install env/tinker" in capsys.readouterr().err
    import bench.train

    assert "tinker" not in Path(bench.train.__file__).read_text()  # the registered path never imports it


def test_a_base_tinker_does_not_offer_is_refused_before_any_call(tmp_path, monkeypatch, tinker, capsys):
    _, config_path, _ = make_s5_repo(tmp_path, monkeypatch)
    assert train(config_path, SHIPPED) == 2
    assert f"Tinker does not offer {TINY['repo']}" in capsys.readouterr().err and not tinker["ServiceClient"]


def test_the_settings_pin_the_sdk_version_env_tinker_installs():
    env = paths.ROOT / "env" / "tinker"
    pins = dict(line.split("==") for line in (env / "requirements.in").read_text().splitlines()
                if "==" in line and not line.startswith("#"))
    assert tt.load_settings(str(SHIPPED))["sdk_version"] == pins["tinker"]
    assert f"\ntinker=={pins['tinker']}\n" in (env / "requirements.lock").read_text()
