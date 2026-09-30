"""The AIPerf client of the load test, in a Modal CPU container beside the deployed vLLM server, so the
network between client and server is the cloud's, not the operator's. Called by
`bench loadtest --on modal`, once per concurrency level; the image is the training image
(env/train/requirements.lock pins AIPerf v0.13.0). Returns AIPerf's artifacts as they come out, and the
state the server recorded about itself (the GPUs it got), read from the vLLM-cache volume.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import modal

from modal_apps import common
from modal_apps.train import hf_cache, image

SETTINGS = common.load("loadtest")
api_key = modal.Secret.from_name(SETTINGS["secrets"]["vllm_api_key"])
vllm_cache = modal.Volume.from_name(SETTINGS["volumes"]["vllm_cache"], create_if_missing=True)

app = modal.App(SETTINGS["apps"]["loadtest"])


@app.function(image=image, cpu=SETTINGS["loadtest"]["cpu"], timeout=SETTINGS["loadtest"]["timeout_s"],
              volumes={common.HF_CACHE: hf_cache, common.VLLM_CACHE: vllm_cache}, secrets=[api_key])
def run_aiperf(payloads: bytes, args: dict) -> dict:
    from bench.loadtest import PAYLOADS, aiperf_command, run_aiperf as run, wait_ready, warm_up

    key = os.environ.get("VLLM_API_KEY")
    state = common.state_path(common.VLLM_CACHE, args["candidate"]) if args.get("candidate") else None
    with tempfile.TemporaryDirectory() as work:
        run_dir = Path(work)
        (run_dir / PAYLOADS).write_bytes(payloads)
        waited, card = wait_ready(args["base_url"], key, args["model"], args["ready_timeout_s"])
        warm_up(args["base_url"], key, args["warmup"], args["timeout_s"])
        cmd = aiperf_command(shutil.which("aiperf"), args["url"], args["model"], args["concurrency"],
                             args["request_count"], args["tokenizer"], args["timeout_s"], args["stream"], key)
        returncode = run(run_dir, cmd)
        files = {p.relative_to(run_dir).as_posix(): p.read_bytes() for p in sorted(run_dir.rglob("*"))
                 if p.is_file() and p.name != PAYLOADS}
        vllm_cache.reload()  # the server wrote its state when it started, before it answered /models
        server_state = json.loads(state.read_text()) if state and state.is_file() else None
    return {"returncode": returncode, "ready_after_s": waited, "served_model": card, "server_state": server_state,
            "files": files}
