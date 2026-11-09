"""P-4's HF-PEFT reference on a RunPod GPU (`bench preflight --parity <cluster> --on runpod`), as
modal_apps/train.py:peft_reference runs it on Modal: bench.preflight.peft_generate, in the training environment
(env/train/requirements.lock), on train.gpu, with the adapter read from the volume at its sha256 (uploaded from
train/adapters/<cluster>/adapter when missing, and checked byte for byte in the pod).

The image: runpod.images.train when set (a custom image built from env/runpod/Dockerfile.train, the lock
already installed), otherwise runpod.images.gpu_job with the lock installed at the first job onto the volume
(a venv keyed by the lock's sha256, reused by every later job and by the load client)."""
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from runpod_apps import pods as rp
from runpod_apps import settings as rs
from runpod_apps.settings import WORK, RunPodError


def peft_reference(config: Dict[str, Any], cluster: str, adapter_sha256: str, base: Dict[str, str],
                   prompts: List[List[dict]], template_kwargs: Dict[str, Any], max_new_tokens: int,
                   top_k: int) -> Dict[str, Any]:
    provider = rp.make_provider(config)
    settings = provider.settings
    image = settings["images"].get("train") or settings["images"]["gpu_job"]
    timeout = config["preflight"]["lora_parity"]["reference_timeout_s"]
    gpu = rs.gpu_id(settings, config["train"]["gpu"])
    env = {"HF_TOKEN": rs.secret_ref(settings["secrets"]["hf_token"])} if settings["secrets"].get("hf_token") else {}
    with provider.job_pod("parity", image, gpu=gpu, env=env) as ready:
        remote = ready["remote"]
        provider.ensure_adapters(remote, [{"cluster": cluster, "sha256": adapter_sha256}])
        spec = {"adapter_sha256": adapter_sha256, "base": base, "prompts": prompts, "template_kwargs": template_kwargs,
                "max_new_tokens": max_new_tokens, "top_k": top_k}
        with tempfile.TemporaryDirectory() as work:
            local = Path(work) / "parity.json"
            local.write_text(json.dumps(spec))
            remote.shell.put(local, f"{WORK}/parity.json")
        remote.start_job("parity", rp.TRAIN_ENV + f'"$PY" -m runpod_apps.job parity {WORK}/parity.json {WORK}/parity.out.json')
        remote.wait_job("parity", timeout + 3600, provider.poll_s)  # the first job also builds the environment
        out = remote.read(f"{WORK}/parity.out.json")
        if out is None:
            raise RunPodError("the P-4 reference ended without its output")
        return {**json.loads(out), "pod_id": ready["pod"]["id"], "usd_per_hour": ready["pod"].get("cost")}
