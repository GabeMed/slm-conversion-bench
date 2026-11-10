"""`bench train --on tinker` without Tinker: rows to datums with the loss on the completion only, the steps and
the cosine schedule as transformers' Trainer counts them, the refusals (no SDK, a base Tinker does not offer, a
hyper-parameter it cannot honour), and the whole command against a fake SDK and a fake export (no network, no
TINKER_API_KEY, no torch)."""
import json
import math
import re
import sys
import types
from pathlib import Path

import pytest
import yaml

from bench import cli, paths
from bench.contracts import facts
from bench.train import TrainError, plan_id, training_plan
from bench import train_tinker as tt
from test_train_fixtures import TINY, make_s5_repo, rows, save

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


def fake_sdk(calls, version="0.39.0", offered=("tiny/Tinker-Qwen3",), rank=None, close_error=None, step_error=None,
             fail_at=None, first_run=1):
    """The parts of `tinker` bench/train_tinker.py uses, recording every call in `calls`. `fail_at`: the n-th
    forward_backward (1-based) raises, as an interruption would; training runs are named run-<first_run>, ..."""
    runs = iter(range(first_run, first_run + 100))

    class TrainingClient:
        def __init__(self):
            self.model_id = f"run-{next(runs)}"

        def get_info(self):
            return types.SimpleNamespace(model_id=self.model_id, lora_rank=rank)

        def forward_backward(self, datums, loss_fn):
            calls["forward_backward"].append((datums, loss_fn))
            if step_error:
                raise step_error
            if fail_at is not None and len(calls["forward_backward"]) == fail_at:
                raise ConnectionError("interrupted")
            loss = -sum(w * -1.0 for d in datums for w in d.loss_fn_inputs["weights"])  # logprob -1 per token
            return Future(types.SimpleNamespace(metrics={"loss:sum": loss}))

        def optim_step(self, params):
            calls["optim_step"].append(params)
            return Future(None)

        def save_state(self, name, ttl_seconds=None):
            calls.setdefault("save_state", []).append((name, ttl_seconds))
            return Future(types.SimpleNamespace(path=f"tinker://{self.model_id}/weights/{name}"))

        def save_weights_external(self, name, ttl_seconds=None):
            calls["save_weights_external"].append((name, ttl_seconds))
            return Future(types.SimpleNamespace(path=f"tinker://{self.model_id}/external_weights/{name}"))

    class ServiceClient:
        def __init__(self, user_metadata=None):
            calls["ServiceClient"].append(user_metadata)

        def create_rest_client(self):
            models = [types.SimpleNamespace(model_name=m, trainable=True, max_context_length=32768) for m in offered]
            return types.SimpleNamespace(
                get_server_capabilities=lambda: Future(types.SimpleNamespace(supported_models=models)),
                delete_checkpoint_from_tinker_path=lambda path: calls.setdefault("delete", []).append(path) or Future(None))

        def create_lora_training_client(self, base_model, **lora):
            calls["create_lora_training_client"].append({"base_model": base_model, **lora})
            return Future(TrainingClient())

        def create_training_client_from_state_with_optimizer(self, path, user_metadata=None):
            calls.setdefault("from_state", []).append(path)
            return Future(TrainingClient())

        def close(self, status):
            calls["close"].append(status)
            if close_error:
                raise close_error
            return Future(None)

    class Datum:
        def __init__(self, model_input, loss_fn_inputs):
            self.model_input, self.loss_fn_inputs = model_input, loss_fn_inputs

    return types.SimpleNamespace(
        __version__=version, ServiceClient=ServiceClient, Datum=Datum,
        ModelInput=types.SimpleNamespace(from_ints=lambda tokens: list(tokens)),
        AdamParams=lambda **params: params)


def fake_export(calls, alpha=16, modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                error=None):
    def export(external_weights, base_dir, out: Path):
        calls["export"].append((external_weights, base_dir))
        if error:
            raise error
        out.mkdir()
        (out / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "base_model_name_or_path": base_dir, "r": 8,
            "lora_alpha": alpha, "lora_dropout": 0.0, "target_modules": list(modules), "rank_pattern": {}, "alpha_pattern": {}}))
        (out / "adapter_model.safetensors").write_bytes(b"weights of " + external_weights.encode())
    return export


@pytest.fixture
def tinker(monkeypatch):
    """A fake SDK, tokenizer and export in place; returns the calls they record."""
    calls = {k: [] for k in ("ServiceClient", "create_lora_training_client", "forward_backward", "optim_step",
                             "save_weights_external", "close", "export", "base_snapshot", "save_state",
                             "from_state", "delete")}
    monkeypatch.setattr(tt, "_sdk", lambda: fake_sdk(calls))
    monkeypatch.setattr(tt, "_tokenizer", lambda plan: FakeTokenizer())
    monkeypatch.setattr(tt, "base_snapshot", lambda plan: calls["base_snapshot"].append(plan["base"]) or "/hf/base")
    monkeypatch.setattr(tt, "export_adapter", fake_export(calls))
    return calls


def full_repo(tmp_path, monkeypatch):
    """The S5 repository with no train.sft.max_steps (a full run: 2 epochs of 4 rows, 2 rows a step, 4 steps)."""
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    config["train"]["sft"]["max_steps"] = None
    return config_path, save(config, config_path)


def train(config_path, settings_path, cluster="c0", *extra):
    return cli.main(["train", "--config", str(config_path), "--cluster", cluster, "--on", "tinker",
                     "--tinker-config", str(settings_path), *extra])


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
    config_path, config = full_repo(tmp_path, monkeypatch)
    assert train(config_path, settings(tmp_path)) == 0
    adapter = paths.ROOT / "train" / "adapters" / "c0" / "adapter"
    assert sorted(p.name for p in adapter.iterdir()) == ["adapter_config.json", "adapter_model.safetensors"]
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == TINY["repo"] and adapter_config["revision"] == TINY["revision"]
    assert tinker["base_snapshot"] == [TINY]  # what the converter reads, fetched before anything is paid for

    # the service: rank and seed from config.yaml, no unembedding; 4 steps (2 epochs) of 2 rows, the loss
    # weighted to the step's completion-token mean, AdamW with the TRL path's defaults at the scheduled rate
    assert tinker["create_lora_training_client"] == [{"base_model": "tiny/Tinker-Qwen3", "rank": 8,
                                                     "seed": config["train"]["seed"], "train_attn": True,
                                                     "train_mlp": True, "train_unembed": False}]
    assert [len(d) for d, _ in tinker["forward_backward"]] == [2, 2, 2, 2]
    assert {loss for _, loss in tinker["forward_backward"]} == {"cross_entropy"}
    for datums, _ in tinker["forward_backward"]:
        assert sum(w for d in datums for w in d.loss_fn_inputs["weights"]) == pytest.approx(1)
        assert all(len(d.model_input) == len(d.loss_fn_inputs["target_tokens"]) == len(d.loss_fn_inputs["weights"])
                   for d in datums)
    sched = tt.schedule(4, config["train"]["sft"])
    assert [p["learning_rate"] for p in tinker["optim_step"]] == [tt.learning_rate(s, sched, 2e-4) for s in range(4)]
    assert all(p["beta2"] == 0.999 and p["eps"] == 1e-8 and p["grad_clip_norm"] == 1.0 for p in tinker["optim_step"])
    assert tinker["close"] == ["success"] and tinker["export"] == [("tinker://run-1/external_weights/" +
                                                                     tinker["save_weights_external"][0][0], "/hf/base")]

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
    assert stats["global_step"] == 4 and stats["examples_seen"] == 8 and not stats["collected_from_earlier_run"]
    assert not stats["probe"]
    assert stats["train_loss"] == pytest.approx(1)  # the fake's log-probability is -1 on every token: mean NLL 1
    assert stats["tokens_seen"] == sum(len(d.model_input) for ds, _ in tinker["forward_backward"] for d in ds)
    assert stats["completion_tokens_seen"] == 2 * stats["completion_tokens"]  # 4 rows, once each epoch
    assert stats["trained"] == {"r": 8, "lora_alpha": 16, "target_modules": sorted(config["train"]["lora"]["target_modules"]),
                                "rank_pattern": {}, "alpha_pattern": {}}
    assert "lora.dropout" in stats["run"]["differences"] and stats["versions"]["tinker"] == "0.39.0"
    assert stats["run"]["base_snapshot"] == "headers-only"  # the converter read the base's headers, not its weights
    assert "price" not in json.dumps(stats)
    assert not list((paths.ROOT / "train" / "adapters" / ".tinker").iterdir())  # the pending record is cleared
    out = capsys.readouterr().out
    assert "prereg/DEVIATIONS.md" in out
    if config.get("runpod"):  # the shipped configuration serves on RunPod: up uploads the adapter
        assert f"bench serve --on runpod up --candidate {manifest['slm']}" in out
    else:  # on Modal, the adapters volume
        assert f"/{manifest['adapter_sha256']}" in out
    assert "bench.train_tinker.plan_sha256 gives the same" in out

    # again: nothing is trained, the adapter is kept, the set is registered once c1 has one too
    assert train(config_path, settings(tmp_path)) == 0
    assert len(tinker["ServiceClient"]) == 1 and "nothing was trained" in capsys.readouterr().out
    assert train(config_path, settings(tmp_path), "c1") == 0
    assert "adapters fact:" in capsys.readouterr().out


def test_a_tinker_run_and_a_modal_run_of_the_same_plan_pair(tmp_path, monkeypatch, tinker):
    config_path, config = full_repo(tmp_path, monkeypatch)
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
    config_path, config = full_repo(tmp_path, monkeypatch)
    from test_train_fixtures import fake_adapter

    fake_adapter("c0", config)
    assert train(config_path, settings(tmp_path)) == 2
    assert "holds another adapter" in capsys.readouterr().err and not tinker["ServiceClient"]


def test_an_adapter_with_another_alpha_is_not_installed_and_the_trained_weights_are_collected_later(
        tmp_path, monkeypatch, tinker, capsys):
    config_path, _ = full_repo(tmp_path, monkeypatch)
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


# ---------------------------------------------------------------- after the paid run: nothing lost, nothing paid twice

def pending_records():
    root = paths.ROOT / "train" / "adapters" / ".tinker"
    return sorted(root.iterdir()) if root.is_dir() else []


def test_a_failed_close_after_the_export_neither_loses_it_nor_fails_the_run(tmp_path, monkeypatch, tinker, capsys):
    config_path, _ = full_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(tt, "_sdk", lambda: fake_sdk(tinker, close_error=ConnectionError("gone")))
    assert train(config_path, settings(tmp_path)) == 0
    assert "closing the Tinker session (success) failed: ConnectionError: gone" in capsys.readouterr().err
    assert (paths.ROOT / "train" / "adapters" / "c0" / "manifest.json").is_file()


def test_a_failed_close_does_not_hide_the_error_that_ended_the_run(tmp_path, monkeypatch, tinker):
    config_path, _ = full_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(tt, "_sdk", lambda: fake_sdk(tinker, close_error=ConnectionError("gone"),
                                                     step_error=RuntimeError("the step failed")))
    with pytest.raises(RuntimeError, match="the step failed"):
        train(config_path, settings(tmp_path))
    assert tinker["close"] == ["errored"] and not pending_records()


def test_the_export_is_recorded_before_the_session_closes(tmp_path, monkeypatch, tinker):
    config_path, _ = full_repo(tmp_path, monkeypatch)
    seen = []
    sdk = fake_sdk(tinker)
    close = sdk.ServiceClient.close

    def close_and_look(self, status):
        seen.append([json.loads(p.read_text()) for p in pending_records()])
        return close(self, status)

    monkeypatch.setattr(sdk.ServiceClient, "close", close_and_look)
    monkeypatch.setattr(tt, "_sdk", lambda: sdk)
    assert train(config_path, settings(tmp_path)) == 0
    [[record]] = seen
    assert record["stats"]["external_weights"].startswith("tinker://run-1/external_weights/")
    assert record["expires_at"] > record["exported_at"]  # export_ttl_s after it


def test_a_failed_download_or_conversion_says_how_to_retry_and_a_retry_does_not_train_again(
        tmp_path, monkeypatch, tinker, capsys):
    config_path, _ = full_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(tt, "export_adapter", fake_export(tinker, error=OSError("No space left on device")))
    assert train(config_path, settings(tmp_path)) == 2
    err = capsys.readouterr().err
    [pending] = pending_records()
    assert "could not be downloaded or converted (OSError: No space left on device)" in err
    assert f"recorded in train/adapters/.tinker/{pending.name}" in err and "delete that file to train again" in err
    assert not (paths.ROOT / "train" / "adapters" / "c0").exists()
    monkeypatch.setattr(tt, "export_adapter", fake_export(tinker))
    assert train(config_path, settings(tmp_path)) == 0
    assert len(tinker["ServiceClient"]) == 1 and not pending_records()


def test_expired_weights_are_named_and_the_way_to_train_again_given(tmp_path, monkeypatch, tinker, capsys):
    config_path, _ = full_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(tt, "export_adapter", fake_export(tinker, error=OSError("x")))
    assert train(config_path, settings(tmp_path)) == 2
    [pending] = pending_records()
    record = json.loads(pending.read_text())
    pending.write_text(json.dumps({**record, "expires_at": "2026-01-01T00:00:00+00:00"}))
    capsys.readouterr()
    assert train(config_path, settings(tmp_path)) == 2
    err = capsys.readouterr().err
    assert "expired on Tinker at 2026-01-01T00:00:00+00:00 (export_ttl_s)" in err
    assert f"delete train/adapters/.tinker/{pending.name} to train again" in err
    assert len(tinker["export"]) == 1 and len(tinker["ServiceClient"]) == 1  # neither downloaded nor trained
    pending.unlink()  # as the message says: the next run trains again
    monkeypatch.setattr(tt, "export_adapter", fake_export(tinker))
    assert train(config_path, settings(tmp_path)) == 0 and len(tinker["ServiceClient"]) == 2


def test_what_the_export_needs_fails_before_anything_is_paid_for(tmp_path, monkeypatch, tinker, capsys):
    config_path, _ = full_repo(tmp_path, monkeypatch)

    def no_base(plan):
        raise TrainError("cannot fetch the base")

    monkeypatch.setattr(tt, "base_snapshot", no_base)
    assert train(config_path, settings(tmp_path)) == 2
    assert "cannot fetch the base" in capsys.readouterr().err and not tinker["ServiceClient"]


def test_a_missing_cookbook_is_refused_by_name(monkeypatch):
    monkeypatch.setitem(sys.modules, "tinker_cookbook", None)
    with pytest.raises(tt.TinkerMissing, match="install env/tinker"):
        tt.base_snapshot({"base": dict(TINY)})


# ---------------------------------------------------------------- the header-only base

class LocalHub:
    """A snapshot directory standing in for the HF Hub: its files, and byte ranges of them (as range requests)."""

    def __init__(self, directory, size_offset=0):
        self.directory, self.size_offset, self.reads = Path(directory), size_offset, []

    def files(self):
        return sorted(p.name for p in self.directory.iterdir())

    def download(self, name):
        return str(self.directory / name)

    def read(self, name, start, end):
        self.reads.append((name, start, end))
        with open(self.directory / name, "rb") as f:
            f.seek(start)
            return f.read(end - start + 1), (self.directory / name).stat().st_size + self.size_offset


def tiny_snapshot():
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(TINY["repo"], revision=TINY["revision"], allow_patterns=["*.json", "*.safetensors"]))


def test_a_header_only_base_reads_as_the_real_one_and_takes_no_disk(tmp_path):
    pytest.importorskip("torch")  # safe_open(framework="pt")
    real = tiny_snapshot()
    hub = LocalHub(real)
    base = tt.header_only_base(TINY["repo"], TINY["revision"], tmp_path, hub)
    assert base == tmp_path / TINY["repo"].replace("/", "--") / TINY["revision"]  # apart from the HF cache
    assert sorted(p.name for p in base.iterdir()) == ["HEADERS-ONLY.txt", "config.json", "model.safetensors"]
    assert tt.header_shapes(base) == tt.header_shapes(real) and tt.header_shapes(base)["model.safetensors"]
    shard = base / "model.safetensors"
    assert shard.stat().st_size == (real / "model.safetensors").stat().st_size
    assert shard.stat().st_blocks * 512 < 1 << 20  # sparse: the header on disk, not the tensors
    length = int.from_bytes(shard.read_bytes()[:8], "little")
    assert hub.reads == [("model.safetensors", 0, 7), ("model.safetensors", 8, 8 + length - 1)]
    assert (base / "config.json").read_bytes() == (real / "config.json").read_bytes()
    assert "NOT the weights" in (base / "HEADERS-ONLY.txt").read_text()
    assert not [p for p in base.parent.iterdir() if p != base]  # the build directory is gone

    class Offline(LocalHub):
        def files(self):
            raise AssertionError("built once, then reused")

    assert tt.header_only_base(TINY["repo"], TINY["revision"], tmp_path, Offline(real)) == base


def test_a_header_that_does_not_describe_the_served_file_is_refused(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(TrainError, match="its header describes .* bytes, the Hub serves"):
        tt.header_only_base(TINY["repo"], TINY["revision"], tmp_path, LocalHub(tiny_snapshot(), size_offset=1))
    assert not (tmp_path / TINY["repo"].replace("/", "--") / TINY["revision"]).exists()


def test_the_cookbook_converts_an_adapter_against_a_header_only_base_as_against_the_real_one(tmp_path):
    """tinker-cookbook's weights.build_lora_adapter, on a Tinker-format adapter for the tiny Qwen3: the same PEFT
    adapter from the real base and from its header-only copy (env/tinker)."""
    pytest.importorskip("tinker_cookbook")
    import torch
    from safetensors.torch import load_file, save_file
    from tinker_cookbook import weights

    real = tiny_snapshot()
    base = tt.header_only_base(TINY["repo"], TINY["revision"], tmp_path / "headers", LocalHub(real))
    rank, torch_gen = 4, torch.Generator().manual_seed(0)
    shapes = tt.header_shapes(real)["model.safetensors"]
    modules = [k for k in shapes if k.endswith("_proj.weight")]
    assert len(modules) == 2 * 7  # 2 layers x q k v o gate up down
    raw = tmp_path / "raw"
    raw.mkdir()
    tensors = {}
    for key in modules:
        (out_features, in_features), _ = shapes[key]
        name = "base_model.model." + key.removesuffix(".weight")
        tensors[f"{name}.lora_A.weight"] = torch.randn(rank, in_features, generator=torch_gen)
        tensors[f"{name}.lora_B.weight"] = torch.randn(out_features, rank, generator=torch_gen)
    save_file(tensors, str(raw / "adapter_model.safetensors"))
    (raw / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": 8}))

    outputs = {}
    for label, directory in (("real", real), ("headers", base)):
        out = tmp_path / f"peft-{label}"
        weights.build_lora_adapter(base_model=str(directory), adapter_path=str(raw), output_path=str(out))
        config = json.loads((out / "adapter_config.json").read_text())
        config.pop("base_model_name_or_path")
        outputs[label] = config, load_file(str(out / "adapter_model.safetensors"))
    (real_config, real_tensors), (config, converted) = outputs["real"], outputs["headers"]
    assert config == real_config and config["r"] == rank
    assert sorted(config["target_modules"]) == sorted({k.split(".")[-2] for k in modules})
    assert converted.keys() == real_tensors.keys() and len(converted) == 2 * len(modules)
    assert all(torch.equal(converted[k], real_tensors[k]) for k in converted)


def _hub_reachable():
    import socket

    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _hub_reachable(), reason="needs the HF Hub (a few hundred KB of headers)")
def test_the_header_only_qwen3_8b_reads_as_the_hub_describes_it(tmp_path):
    """The real base, by range requests at the revision config.yaml pins: each shard's header as the Hub's own
    safetensors metadata gives it, at the shard's size, in under a megabyte of disk."""
    pytest.importorskip("torch")
    from huggingface_hub import get_safetensors_metadata

    from bench.contracts.config import load_config

    qwen = next(c for c in load_config(str(paths.ROOT / "config.yaml"))["roles"]["slm_candidates"]
                if c["hf"]["repo"] == "Qwen/Qwen3-8B")["hf"]
    base = tt.header_only_base(qwen["repo"], qwen["revision"], tmp_path)
    hub = get_safetensors_metadata(qwen["repo"], revision=qwen["revision"])
    shapes = tt.header_shapes(base)
    assert shapes == {shard: {k: (tuple(t.shape), t.dtype) for k, t in meta.tensors.items()}
                      for shard, meta in hub.files_metadata.items()}
    assert sum(len(t) for t in shapes.values()) == 399  # 36 layers x 11, the embeddings, the norm, lm_head
    sizes = {p.name: p.stat().st_size for p in base.glob("*.safetensors")}
    tensor_bytes = json.loads((base / "model.safetensors.index.json").read_text())["metadata"]["total_size"]
    assert 0 < sum(sizes.values()) - tensor_bytes < 1 << 20  # the shards' real sizes: their tensors and headers
    assert sum(p.stat().st_blocks * 512 for p in base.iterdir()) < 1 << 20


# ---------------------------------------------------------------- the probe

def test_a_run_cut_short_is_a_probe_kept_apart_and_never_registered(tmp_path, monkeypatch, tinker, capsys):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)  # train.sft.max_steps: 2
    assert train(config_path, settings(tmp_path)) == 0
    assert train(config_path, settings(tmp_path), "c1") == 0
    out = capsys.readouterr().out
    adapters = paths.ROOT / "train" / "adapters"
    assert "a probe (train.sft.max_steps)" in out and "adapters fact" not in out
    assert not (adapters / "c0").exists() and not (adapters / "c1").exists()
    manifest = json.loads((adapters / ".probe" / "c0" / "manifest.json").read_text())
    assert manifest["stats"]["probe"] and manifest["stats"]["global_step"] == 2
    # its first step runs at learning rate 0 (warm-up), which the probe shows the service accepts or not
    assert tinker["optim_step"][0]["learning_rate"] == 0
    from bench.train import register_adapters

    assert register_adapters(config) == (None, ["c0", "c1"])
    # the full run is not in its way, and the probe again only collects
    config["train"]["sft"]["max_steps"] = None
    full = tmp_path / "full.yaml"
    save(config, full)
    assert train(full, settings(tmp_path)) == 0 and (adapters / "c0" / "manifest.json").is_file()
    assert train(config_path, settings(tmp_path)) == 0 and "nothing was trained" in capsys.readouterr().out


# ---------------------------------------------------------------- resume: periodic states, --resume, refusals

def resume_records():
    root = paths.ROOT / "train" / "adapters" / ".tinker"
    return sorted(root.glob("*.resume.json")) if root.is_dir() else []


def new_calls():
    return {k: [] for k in ("ServiceClient", "create_lora_training_client", "forward_backward", "optim_step",
                            "save_weights_external", "close", "export", "save_state", "from_state", "delete")}


def interrupt(tmp_path, monkeypatch, tinker, capsys, save_every="2", fail_at=4):
    """A full run (4 steps) that saves a state every `save_every` steps and is interrupted at the `fail_at`-th step."""
    config_path, config = full_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(tt, "_sdk", lambda: fake_sdk(tinker, fail_at=fail_at))
    with pytest.raises(ConnectionError, match="interrupted"):
        train(config_path, settings(tmp_path), "c0", "--save-every", save_every)
    capsys.readouterr()
    return config_path, config


@pytest.mark.parametrize("every, saved", [("1", [1, 2, 3]), ("2", [2]), ("3", [3]), ("4", []), ("0", [])])
def test_states_are_saved_every_n_steps_and_only_the_latest_is_kept(tmp_path, monkeypatch, tinker, every, saved):
    config_path, _ = full_repo(tmp_path, monkeypatch)
    seen = []  # the resume record as each state was deleted: the newer state is on disk first
    sdk = fake_sdk(tinker)
    rest = sdk.ServiceClient.create_rest_client

    def watching(self):
        client = rest(self)
        delete = client.delete_checkpoint_from_tinker_path
        client.delete_checkpoint_from_tinker_path = lambda path: (
            seen.append((path, [s["path"] for s in json.loads(resume_records()[0].read_text())["states"]])), delete(path))[1]
        return client

    monkeypatch.setattr(sdk.ServiceClient, "create_rest_client", watching)
    monkeypatch.setattr(tt, "_sdk", lambda: sdk)
    tinker.update(save_state=[], delete=[])
    assert train(config_path, settings(tmp_path), "c0", "--save-every", every) == 0
    names = [f"c0-{json.loads((paths.ROOT / 'train/adapters/c0/manifest.json').read_text())['tinker_key'][:12]}-step{n:06d}"
             for n in saved]
    assert tinker["save_state"] == [(n, tt.STATE_TTL_S) for n in names]  # never after the last step: the export is
    paths_saved = [f"tinker://run-1/weights/{n}" for n in names]
    assert tinker["delete"] == paths_saved  # each replaced by the next, the last once the adapter is exported
    for i, (path, on_disk) in enumerate(seen[:-1] if saved else []):
        assert path == paths_saved[i] and paths_saved[i + 1] in on_disk
    manifest = json.loads((paths.ROOT / "train" / "adapters" / "c0" / "manifest.json").read_text())
    resume = manifest["stats"]["resume"]
    assert resume["save_every"] == int(every) and not resume["resumed"] and resume["steps_replayed"] == 0
    assert [s["step"] for s in resume["states"]] == saved and resume["caveats"] == {}
    assert [s["from_step"] for s in resume["segments"]] == [0] and resume["segments"][0]["training_run"] == "run-1"
    assert manifest["stats"]["global_step"] == 4 and not resume_records() and not pending_records()


def test_the_default_cadence_is_every_50_steps(tmp_path, monkeypatch, tinker):
    config_path, _ = full_repo(tmp_path, monkeypatch)
    assert train(config_path, settings(tmp_path)) == 0
    manifest = json.loads((paths.ROOT / "train" / "adapters" / "c0" / "manifest.json").read_text())
    assert tt.SAVE_EVERY == 50 and manifest["stats"]["resume"]["save_every"] == 50


def test_an_interrupted_run_resumes_after_its_state_at_the_same_step_order_and_learning_rate(
        tmp_path, monkeypatch, tinker, capsys):
    config_path, config = interrupt(tmp_path, monkeypatch, tinker, capsys)  # a state at step 2, cut at step 4
    first = {k: list(v) for k, v in tinker.items()}
    [record_path] = resume_records()
    record = json.loads(record_path.read_text())
    assert [s["step"] for s in record["states"]] == [2] and record["reached_step"] == 3 and len(record["losses"]) == 3
    state = record["states"][0]["path"]

    # without --resume, the run refuses to pay again for the steps saved
    again = new_calls()
    monkeypatch.setattr(tt, "_sdk", lambda: fake_sdk(again, first_run=2))
    assert train(config_path, settings(tmp_path)) == 2
    err = capsys.readouterr().err
    assert "saved its state at step 2" in err and "pass --resume" in err and not again["ServiceClient"]

    assert train(config_path, settings(tmp_path), "c0", "--resume", "--save-every", "2") == 0
    out = capsys.readouterr()
    assert again["from_state"] == [state] and not again["create_lora_training_client"]
    assert "resuming at step 3/4" in out.err and "(1 steps after it are trained again)" in out.err
    # the steps after the state, as the interrupted run took them (rows and tokens), at their global learning rate
    sched = tt.schedule(4, config["train"]["sft"])
    assert [p["learning_rate"] for p in again["optim_step"]] == [tt.learning_rate(s, sched, 2e-4) for s in (2, 3)]
    assert [[d.model_input for d in ds] for ds, _ in again["forward_backward"]] == \
        [[d.model_input for d in ds] for ds, _ in first["forward_backward"][2:4]]
    assert again["save_state"] == []  # step 4 is the last: the export follows
    assert again["delete"] == [state]  # deleted once the adapter is exported

    manifest = json.loads((paths.ROOT / "train" / "adapters" / "c0" / "manifest.json").read_text())
    stats, resume = manifest["stats"], manifest["stats"]["resume"]
    assert stats["global_step"] == 4 and stats["examples_seen"] == 8 and stats["train_loss"] == pytest.approx(1)
    assert stats["tokens_seen"] == sum(len(d.model_input) for ds, _ in first["forward_backward"][:2] for d in ds) + \
        sum(len(d.model_input) for ds, _ in again["forward_backward"] for d in ds)
    assert stats["training_run"] == "run-2" and stats["external_weights"].startswith("tinker://run-2/")
    assert resume["resumed"] and resume["resumed_from_steps"] == [2] and resume["steps_replayed"] == 1
    assert [(s["from_step"], s["training_run"], s["state"]) for s in resume["segments"]] == \
        [(0, "run-1", None), (2, "run-2", state)]
    assert all("commit" in s for s in resume["segments"])
    assert [(s["step"], s["path"]) for s in resume["states"]] == [(2, state)]
    assert set(resume["caveats"]) == set(tt.RESUME_CAVEATS)
    assert manifest["started_at"] == record["started_at"]  # the run began with the interrupted segment
    assert "resumed at steps [2] (1 steps trained twice)" in out.out
    assert not resume_records() and not pending_records()


def resume_with(tmp_path, monkeypatch, config_path, settings_path=None):
    calls = new_calls()
    monkeypatch.setattr(tt, "_sdk", lambda: fake_sdk(calls, first_run=2))
    code = train(config_path, settings_path or settings(tmp_path), "c0", "--resume")
    return code, calls


@pytest.mark.parametrize("change, match", [
    ("dataset", "dataset_sha256"),
    ("learning_rate", "plan_sha256"),
    ("optimizer", r"the Tinker run \(optimizer\)"),
    ("config", "config_sha256"),
    ("tokens", "encoded_sha256"),
])
def test_resume_refuses_anything_that_differs_from_the_interrupted_run(tmp_path, monkeypatch, tinker, capsys,
                                                                        change, match):
    config_path, config = interrupt(tmp_path, monkeypatch, tinker, capsys)
    settings_path = None
    if change == "dataset":
        from test_train_fixtures import write_dataset

        write_dataset("c0", list(reversed(rows("c0"))))
    elif change == "learning_rate":
        config["train"]["sft"]["learning_rate"] = 1e-4
        save(config, config_path)
    elif change == "optimizer":
        settings_path = settings(tmp_path, optimizer={"beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.0,
                                                      "grad_clip_norm": 1.0})
    elif change == "config":  # outside the plan: config.yaml as a whole must be the interrupted run's
        config["modal"]["volumes"]["adapters"] = "another-volume"
        save(config, config_path)
    else:  # the same plan tokenized otherwise (another tokenizer or code)
        class Other(FakeTokenizer):
            def apply_chat_template(self, messages, *a, **k):
                ids = super().apply_chat_template(messages, *a, **k)
                return {"input_ids": [7] + ids["input_ids"]}

        monkeypatch.setattr(tt, "_tokenizer", lambda plan: Other())
    code, calls = resume_with(tmp_path, monkeypatch, config_path, settings_path)
    err = capsys.readouterr().err
    assert code == 2 and re.search(match, err) and "nothing is resumed" in err, err
    assert not calls["ServiceClient"] and len(resume_records()) == 1  # nothing paid, the record kept


def test_resume_refuses_a_run_with_nothing_to_resume(tmp_path, monkeypatch, tinker, capsys):
    config_path, _ = full_repo(tmp_path, monkeypatch)
    code, calls = resume_with(tmp_path, monkeypatch, config_path)
    assert code == 2 and "no interrupted run of c0 is recorded" in capsys.readouterr().err
    assert not calls["ServiceClient"]


def test_resume_refuses_an_expired_state(tmp_path, monkeypatch, tinker, capsys):
    config_path, _ = interrupt(tmp_path, monkeypatch, tinker, capsys)
    [record_path] = resume_records()
    record = json.loads(record_path.read_text())
    record["states"][-1]["expires_at"] = "2026-01-01T00:00:00+00:00"
    record_path.write_text(json.dumps(record))
    code, calls = resume_with(tmp_path, monkeypatch, config_path)
    err = capsys.readouterr().err
    assert code == 2 and "expired on Tinker at 2026-01-01" in err and "delete" in err and not calls["ServiceClient"]


def test_a_state_saved_before_the_first_step_failed_is_not_needed_and_nothing_is_recorded(
        tmp_path, monkeypatch, tinker, capsys):
    interrupt(tmp_path, monkeypatch, tinker, capsys, save_every="2", fail_at=2)  # cut before the first state
    assert not resume_records()


def test_resume_and_save_every_are_for_tinker_only(tmp_path, monkeypatch, capsys):
    _, config_path, _ = make_s5_repo(tmp_path, monkeypatch)
    for extra in (["--resume"], ["--save-every", "5"]):
        assert cli.main(["train", "--config", str(config_path), "--cluster", "c0", "--on", "modal", *extra]) == 2
        assert "for --on tinker only" in capsys.readouterr().err


def test_a_negative_cadence_is_refused(tmp_path, monkeypatch, tinker, capsys):
    config_path, _ = full_repo(tmp_path, monkeypatch)
    assert train(config_path, settings(tmp_path), "c0", "--save-every", "-1") == 2
    assert "--save-every must be a whole number" in capsys.readouterr().err and not tinker["ServiceClient"]
