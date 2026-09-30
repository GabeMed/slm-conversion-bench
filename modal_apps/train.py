"""S5 on a Modal GPU: `train_adapter` runs `bench.train.train_lora`, the same code the local CPU run uses, in
the image of env/train/requirements.lock; and `peft_reference`, P-4's HF-PEFT side
(`bench.preflight.peft_generate`). Both are called by `bench train --on modal` and
`bench preflight --parity <cluster> --on modal`, which run this app ephemerally.

A trained adapter is kept on the adapters volume at `/adapters/<sha256>` (its `facts.sha256_dir`), the
path the serving app loads it from, and is returned to the local side, which checks the sha256 again.
The local side runs the app detached, so a training goes on if the operator's machine sleeps; its result is
stored on the volume, keyed by the plan, the GPU and the code (bench.train.modal_key), with the run's own
GPU, price and commit; running the same command again collects it instead of training twice.
"""
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import modal

from modal_apps import common

SETTINGS = common.load("train")

hf_cache = modal.Volume.from_name(SETTINGS["volumes"]["hf_cache"], create_if_missing=True)
adapters = modal.Volume.from_name(SETTINGS["volumes"]["adapters"], create_if_missing=True)
hf_secrets = [modal.Secret.from_name(SETTINGS["secrets"]["hf_token"])] if SETTINGS["secrets"].get("hf_token") else []

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(requirements=[str(common.TRAIN_LOCK)])
    .env({common.ENV: json.dumps(SETTINGS, sort_keys=True), "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("bench", "modal_apps")
)

app = modal.App(SETTINGS["apps"]["train"])
volumes = {common.HF_CACHE: hf_cache, common.ADAPTERS: adapters}


@app.function(image=image, gpu=SETTINGS["train"]["gpu"], timeout=SETTINGS["train"]["timeout_s"], volumes=volumes,
              secrets=hf_secrets, single_use_containers=True)
def train_adapter(plan: dict, dataset: bytes, identity: dict) -> dict:
    """`identity` (bench.train.run_identity): the GPU with its price, the code and image, the commit. The
    stored result is keyed by the plan, the GPU and the code, and records the identity it ran with."""
    started = time.monotonic()
    from bench.contracts.facts import sha256_dir
    from bench.train import modal_key, parse_dataset, sha256_bytes, train_lora

    if sha256_bytes(dataset) != plan["dataset"]["sha256"]:
        raise RuntimeError("the dataset received is not the one planned (sha256 differs)")
    if identity["gpu"] != SETTINGS["train"]["gpu"]:
        raise RuntimeError(f"this function runs on {SETTINGS['train']['gpu']}, the plan asks {identity['gpu']}")
    rows = parse_dataset(dataset, plan["dataset"]["path"])
    key = modal_key(plan, identity)
    adapters.reload()
    stored = common.stored_training(common.ADAPTERS, key, time.time(), SETTINGS["train"]["timeout_s"])
    if stored is not None:
        return stored
    common.mark_started(common.ADAPTERS, key, time.time())
    adapters.commit()
    try:
        with tempfile.TemporaryDirectory() as work:
            out = Path(work) / "adapter"
            stats = train_lora(plan, rows, out, "cuda")
            sha = sha256_dir(out)
            destination = Path(common.ADAPTERS) / sha
            if not destination.exists():
                shutil.copytree(out, destination)
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
            run = {**identity, "observed_gpus": [line.strip() for line in smi.stdout.splitlines() if line.strip()]}
            result = {"sha256": sha, "stats": stats, "function_seconds": round(time.monotonic() - started, 3), "run": run}
            common.store_training(common.ADAPTERS, key, result)
            files = {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    finally:  # a failed training releases its marker: the next attempt is not refused as "still running"
        common.release_started(common.ADAPTERS, key)
        adapters.commit()
        hf_cache.commit()
    return {**result, "files": files, "reused": False}


@app.function(image=image, gpu=SETTINGS["train"]["gpu"], timeout=SETTINGS["train"]["reference_timeout_s"],
              volumes=volumes, secrets=hf_secrets)
def peft_reference(adapter_sha256: str, base: dict, prompts: list, template_kwargs: dict, max_new_tokens: int,
                   top_k: int) -> dict:
    started = time.monotonic()
    from bench.contracts.facts import sha256_dir
    from bench.preflight import peft_generate

    adapter = Path(common.ADAPTERS) / adapter_sha256
    if not adapter.is_dir() or sha256_dir(adapter) != adapter_sha256:
        raise RuntimeError(f"adapter {adapter_sha256} is not on the adapters volume")
    reference = peft_generate(base, str(adapter), prompts, template_kwargs, max_new_tokens, top_k, "cuda")
    return {**reference, "function_seconds": round(time.monotonic() - started, 3)}
