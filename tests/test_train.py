"""`bench train` without training: the dataset contract, what a plan is decided on, the refusals, and the
registration of the `adapters` fact as the router reads it (no torch needed)."""
import copy
import hashlib
import json

import pytest

from bench import paths
from bench.contracts import facts, router
from bench.train import (TrainError, gpu_cost, parse_dataset, register_adapters, row_problems, training_plan)
from test_train_fixtures import TINY, TINY_NAME, fake_adapter, fake_modal_app, make_s5_repo, rel, rows, save

GOOD = rows("c0", 1)[0]


def test_a_prompt_completion_row_is_accepted():
    assert row_problems(GOOD) == []
    assert row_problems({**GOOD, "source_call_id": "x"}) == []  # extra fields are ignored, never trained on


@pytest.mark.parametrize("row, problem", [
    ([], "must be an object"),
    ({"completion": GOOD["completion"]}, "prompt must be a non-empty list"),
    ({**GOOD, "prompt": []}, "prompt must be a non-empty list"),
    ({**GOOD, "completion": GOOD["completion"] * 2}, "exactly one message"),
    ({**GOOD, "completion": [{"role": "user", "content": "x"}]}, "must be an assistant message"),
    ({**GOOD, "prompt": GOOD["prompt"] + [{"role": "assistant", "content": "x"}]}, "must not end with an assistant"),
    ({**GOOD, "prompt": [{"role": "tool", "content": "x"}]}, "role must be one of"),
    ({**GOOD, "prompt": [{"role": "user", "content": 3}]}, "content must be a string"),
    ({**GOOD, "prompt": [{"role": "user", "content": "x", "name": "y"}]}, "must be {role, content}"),
])
def test_malformed_rows_are_refused(row, problem):
    assert any(problem in p for p in row_problems(row))


def test_one_bad_line_refuses_the_whole_dataset_and_says_which():
    raw = (json.dumps(GOOD) + "\n\n" + json.dumps({**GOOD, "completion": []}) + "\n").encode()
    with pytest.raises(TrainError, match=r"c0.jsonl:3: completion"):
        parse_dataset(raw, "c0.jsonl")
    with pytest.raises(TrainError, match="not JSON"):
        parse_dataset(b"{nope\n", "c0.jsonl")
    with pytest.raises(TrainError, match="no rows"):
        parse_dataset(b"\n", "c0.jsonl")
    assert parse_dataset((json.dumps({**GOOD, "extra": 1}) + "\n").encode(), "c0.jsonl") == [GOOD]


def test_the_plan_trains_the_chosen_base_on_the_pinned_revision_with_serving_kwargs(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    plan, raw = training_plan(config, "c0")
    assert plan["slm"] == TINY_NAME and plan["base"] == TINY and plan["served_name"] == "c0"
    assert plan["chat_template_kwargs"] == {"enable_thinking": False}
    assert plan["dataset"] == {"path": "train/datasets/c0.jsonl", "sha256": hashlib.sha256(raw).hexdigest(), "rows": 4}
    b4 = config["arms"]["B4"]
    assert plan["facts"] == {"choice": facts.read_fact(str(paths.ROOT / b4["choice"]), "choice")[1],
                             "centroids": facts.read_fact(str(paths.ROOT / b4["centroids"]), "centroids")[1]}
    assert plan["hyperparameters"]["lora"]["target_modules"] == config["train"]["lora"]["target_modules"]
    assert plan["hyperparameters"]["seed"] == config["train"]["seed"]


def test_training_refuses_while_the_teachers_terms_are_unrecorded(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch, terms=False)
    with pytest.raises(TrainError, match="D6.*weights_license, provider_terms, checked_on"):
        training_plan(config, "c0")


def test_training_needs_the_facts_b4_uses(tmp_path, monkeypatch):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    config["arms"]["B4"]["centroids"] = None
    with pytest.raises(TrainError, match="arms.B4.centroids is not set"):
        training_plan(save(config, config_path), "c0")


def test_only_clusters_of_the_centroids_with_a_dataset_are_trained(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    with pytest.raises(TrainError, match="'c9' is not a cluster of the centroids"):
        training_plan(config, "c9")
    (paths.ROOT / "train" / "datasets" / "c1.jsonl").unlink()
    with pytest.raises(TrainError, match="no dataset for cluster c1"):
        training_plan(config, "c1")


def test_a_candidate_without_pinned_weights_is_refused(tmp_path, monkeypatch):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    config["roles"]["slm_candidates"][-1]["hf"]["revision"] = "main"
    with pytest.raises(TrainError, match="40-hex revision"):
        training_plan(save(config, config_path), "c0")


def test_a_cluster_name_that_cannot_be_served_is_refused(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch, clusters=("c 0", "c1"))
    with pytest.raises(TrainError, match="cannot be a served adapter name"):
        training_plan(config, "c 0")


def test_the_shipped_candidates_pin_their_weights():
    from bench.contracts.config import load_config
    from bench.train import candidate

    config = load_config(paths.ROOT / "config.yaml")
    for entry in config["roles"]["slm_candidates"]:
        assert candidate(config, entry["name"])["hf"]["repo"]
        assert entry["chat_template_kwargs"] == {"enable_thinking": False}
    for name, pin in config["roles"]["slm_reserve_hf"].items():
        assert name in config["roles"]["slm_reserve"] and len(pin["revision"]) == 40


# ---------------------------------------------------------------- the adapters fact

def test_the_fact_is_written_once_every_cluster_has_an_adapter_and_the_router_accepts_it(tmp_path, monkeypatch):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    fake_adapter("c0", config)
    assert register_adapters(config) == (None, ["c1"])
    fake_adapter("c1", config)
    fact_path, missing = register_adapters(config)
    assert missing == [] and fact_path.parent.parent.name == "S5"
    fact, _ = facts.read_fact(str(fact_path), "adapters")
    assert fact["slm"] == TINY_NAME and set(fact["adapters"]) == {"c0", "c1"}
    assert fact["adapters"]["c1"] == {"served_name": "c1",
                                      "sha256": facts.sha256_dir(paths.ROOT / "train/adapters/c1/adapter")}
    # the router's own consistency checks (choice, centroids, one adapter per cluster) accept it
    config["arms"]["B4"]["adapters"] = rel(fact_path)
    config = save(config, config_path)
    assert set(router.arm_facts("B4", config)) == {"choice", "centroids", "adapters"}
    assert router.possible_engines("B4", config) == [f"slm:{TINY_NAME}+lora:c0", f"slm:{TINY_NAME}+lora:c1"]


def test_an_adapter_trained_on_other_facts_does_not_count(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    fake_adapter("c0", config, facts_override={"choice": "0" * 64, "centroids": "1" * 64})
    fake_adapter("c1", config)
    assert register_adapters(config) == (None, ["c0"])


@pytest.mark.parametrize("change", ["revision", "kwargs"])
def test_an_adapter_trained_on_other_weights_or_kwargs_than_the_candidates_does_not_count(tmp_path, monkeypatch, change):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    fake_adapter("c0", config)
    fake_adapter("c1", config)
    entry = config["roles"]["slm_candidates"][-1]
    if change == "revision":  # e.g. after P-4 said to fix the pinned revision
        entry["hf"]["revision"] = "b" * 40
    else:
        entry["chat_template_kwargs"] = {"enable_thinking": True}
    assert register_adapters(save(config, config_path)) == (None, ["c0", "c1"])


def test_an_adapter_changed_after_training_is_refused(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    adapter = fake_adapter("c0", config)
    fake_adapter("c1", config)
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(TrainError, match="changed after training"):
        register_adapters(config)


def test_cost_is_gpu_seconds_times_the_dated_price():
    config = {"modal": {"gpu_prices": {"as_of": "2026-09-30", "usd_per_s": {"L40S": 0.0005}}}}
    assert gpu_cost(config, "L40S", 3600) == {"gpu": "L40S", "gpu_seconds": 3600, "cost_usd": 1.8,
                                              "price_usd_per_s": 0.0005, "price_as_of": "2026-09-30"}
    assert gpu_cost(config, None, 10)["cost_usd"] == 0.0
    with pytest.raises(TrainError, match="no price for GPU H200"):
        gpu_cost(copy.deepcopy(config), "H200", 1)


def test_bench_train_reports_a_refusal_as_exit_code_2(tmp_path, monkeypatch, capsys):
    from bench import cli

    _, config_path, _ = make_s5_repo(tmp_path, monkeypatch, terms=False)
    assert cli.main(["train", "--config", str(config_path), "--cluster", "c0", "--on", "local"]) == 2
    assert "D6" in capsys.readouterr().err
    assert not (paths.ROOT / "train" / "adapters" / "c0").exists()


def _adapter_files(tmp_path):
    files = {"adapter_config.json": b'{"r": 8}', "adapter_model.safetensors": b"weights"}
    probe = tmp_path / "probe"
    probe.mkdir()
    for name, content in files.items():
        (probe / name).write_bytes(content)
    return files, facts.sha256_dir(probe)


def test_training_on_modal_keeps_what_modal_trained_and_costs_its_gpu_seconds(tmp_path, monkeypatch):
    from bench import train
    from bench.train import train_cluster

    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    prechecked = []
    monkeypatch.setattr(train, "precheck", lambda plan, rows: prechecked.append(len(rows)))
    files, sha = _adapter_files(tmp_path)
    stats = {"device": "cuda", "train_seconds": 90.0, "examples_seen": 8}
    calls = fake_modal_app(monkeypatch, "train", train_adapter=lambda plan, raw: {
        "sha256": sha, "files": files, "stats": stats, "function_seconds": 100.0})
    result = train_cluster(str(config_path), "c0", "modal")
    (plan, raw), = calls["train_adapter"]
    assert plan["dataset"]["sha256"] == hashlib.sha256(raw).hexdigest() and plan["base"] == TINY
    assert prechecked == [4]  # the template checks ran here, before any GPU
    assert calls["app.run"] == [{"detach": True}]  # the training survives this machine sleeping
    adapter = paths.ROOT / "train" / "adapters" / "c0" / "adapter"
    assert {p.name: p.read_bytes() for p in adapter.iterdir()} == files
    manifest = json.loads(result["manifest"].read_text())
    price = config["modal"]["gpu_prices"]["usd_per_s"][config["train"]["gpu"]]
    assert (manifest["where"], manifest["gpu"], manifest["gpu_seconds"]) == ("modal", config["train"]["gpu"], 100.0)
    assert manifest["cost_usd"] == round(100.0 * price, 4) and manifest["adapter_sha256"] == sha
    assert manifest["stats"] == {**stats, "collected_from_earlier_run": False} and result["missing"] == ["c1"]


def test_an_adapter_that_is_not_the_one_modal_trained_is_refused(tmp_path, monkeypatch):
    from bench import train
    from bench.train import train_cluster

    _, config_path, _ = make_s5_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(train, "precheck", lambda plan, rows: None)
    files, _ = _adapter_files(tmp_path)
    fake_modal_app(monkeypatch, "train", train_adapter=lambda plan, raw: {
        "sha256": "0" * 64, "files": files, "stats": {}, "function_seconds": 1.0})
    with pytest.raises(TrainError, match="not the one trained there"):
        train_cluster(str(config_path), "c0", "modal")
    assert [p.name for p in (paths.ROOT / "train" / "adapters").iterdir()] == []  # nothing installed
