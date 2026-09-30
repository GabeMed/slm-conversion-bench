"""A synthetic repository ready for S5 (no tests here): the synthetic repo of `synthetic.make_repo`, plus an
SLM candidate pinned to a tiny random-weights Qwen3 (the smallest causal LM TRL accepts, ~21 MB), a
`choice` fact naming it, a `centroids` fact, B4 pointing at both, the teacher's terms recorded, and one
dataset per cluster in TRL's conversational prompt-completion format."""
import copy
import json
import math
from pathlib import Path
from typing import Dict, List

import yaml

from bench import paths
from bench.contracts import facts
from bench.contracts.config import load_config
from synthetic import make_repo

TINY = {"repo": "trl-internal-testing/tiny-Qwen3ForCausalLM", "revision": "25bb9b8963688237e94750f6f3e48f1947d3c7a3"}
TINY_NAME = "tiny-qwen3"
EMBEDDING = {"model": "m", "revision": "a" * 40, "max_seq_length": 512, "truncation": "tail", "text": "prompt"}
TERMS = {"weights_license": "https://example.org/weights-license", "provider_terms": "https://example.org/terms",
         "checked_on": "2026-09-30"}


def rows(cluster: str, n: int = 4) -> List[Dict]:
    return [{"prompt": [{"role": "system", "content": f"You answer for {cluster}."},
                        {"role": "user", "content": f"Is column c{i} of table t relevant to question {i}?"}],
             "completion": [{"role": "assistant", "content": '{"is_column_information_relevant": "Yes"}'}]}
            for i in range(n)]


def write_dataset(cluster: str, items: List[Dict]) -> Path:
    path = paths.ROOT / "train" / "datasets" / f"{cluster}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in items))
    return path


def rel(path: Path) -> str:
    return str(path.relative_to(paths.ROOT))


def save(config: Dict, config_path: Path) -> Dict:
    config_path.write_text(yaml.safe_dump(config))
    return load_config(config_path)


def make_s5_repo(tmp_path, monkeypatch, clusters=("c0", "c1"), terms=True):
    """Returns (root, config path, config)."""
    root, config_path, config = make_repo(tmp_path, monkeypatch)
    facts._read_fact.cache_clear()
    config = copy.deepcopy(config)
    config["roles"]["slm_candidates"].append({
        "name": TINY_NAME, "base": "tiny random Qwen3 (tests)", "hf": dict(TINY),
        "chat_template_kwargs": {"enable_thinking": False},
        "endpoint": {"kind": "vllm", "base_url": None, "api_key_env": None}, "params": {"max_tokens": 64, "timeout_s": 60}})
    if terms:
        config["roles"]["production_llm"]["terms"] = dict(TERMS)
    choice = facts.write_fact("J6", "choice", {"slm": TINY_NAME})
    vectors = {c: [1.0 if i == j else 0.0 for j in range(len(clusters))] for i, c in enumerate(clusters)}
    assert all(abs(math.sqrt(sum(x * x for x in v)) - 1) < 1e-9 for v in vectors.values())
    centroids = facts.write_fact("J5", "centroids", {"embedding": EMBEDDING, "clusters": vectors})
    config["arms"]["B4"] = {"choice": rel(choice), "centroids": rel(centroids), "adapters": None}
    config["train"]["sft"].update({"max_steps": 2, "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
                                   "max_length": 512, "logging_steps": 1, "gradient_checkpointing": False})
    config["train"]["lora"].update({"r": 8, "alpha": 16})
    for cluster in clusters:
        write_dataset(cluster, rows(cluster))
    return root, config_path, save(config, config_path)


def fake_adapter(cluster, config, slm=TINY_NAME, facts_override=None, served_name=None):
    """An adapter directory and manifest as bench train leaves them (weights are placeholder bytes)."""
    adapter = paths.ROOT / "train" / "adapters" / cluster / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "cluster": cluster}))
    (adapter / "adapter_model.safetensors").write_bytes(cluster.encode() * 10)
    from bench.train import training_plan

    plan, _ = training_plan(config, cluster)
    manifest = {**plan, "slm": slm, "served_name": served_name or cluster,
                "facts": facts_override or plan["facts"], "adapter_sha256": facts.sha256_dir(adapter)}
    (adapter.parent / "manifest.json").write_text(json.dumps(manifest))
    return adapter


def fake_modal_app(monkeypatch, module: str, **functions):
    """Stand in for `modal_apps.<module>` (an ephemeral app with `.remote` functions), so the local side of
    `--on modal` runs without Modal. Returns {name: [args of each call]}."""
    import contextlib
    import sys
    import types

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import modal_apps

    calls = {name: [] for name in functions}

    def remote(name, fn):
        def call(*args):
            calls[name].append(args)
            return fn(*args)
        return types.SimpleNamespace(remote=call)

    fake = types.SimpleNamespace(app=types.SimpleNamespace(run=contextlib.nullcontext),
                                 **{name: remote(name, fn) for name, fn in functions.items()})
    monkeypatch.setitem(sys.modules, f"modal_apps.{module}", fake)
    monkeypatch.setattr(modal_apps, module, fake, raising=False)
    return calls
