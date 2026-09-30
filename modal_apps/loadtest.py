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
proxy_auth = modal.Secret.from_name(SETTINGS["secrets"]["proxy_auth"])  # the variables headers_env names
vllm_cache = modal.Volume.from_name(SETTINGS["volumes"]["vllm_cache"], create_if_missing=True)

app = modal.App(SETTINGS["apps"]["loadtest"])


@app.function(image=image, cpu=SETTINGS["loadtest"]["cpu"], timeout=SETTINGS["loadtest"]["timeout_s"],
              volumes={common.HF_CACHE: hf_cache, common.VLLM_CACHE: vllm_cache}, secrets=[api_key, proxy_auth])
def run_aiperf(payloads: bytes, args: dict) -> dict:
    """bench.loadtest.run_level, as locally, with this container's credentials: vLLM's key from its secret,
    the headers from the proxy_auth secret by the names the endpoint's headers_env gives."""
    from bench.loadtest import PAYLOADS, env_headers, run_level

    key = os.environ.get("VLLM_API_KEY")
    headers = env_headers(args["headers_env"])
    state = common.state_path(common.VLLM_CACHE, args["candidate"]) if args.get("candidate") else None
    with tempfile.TemporaryDirectory() as work:
        run_dir = Path(work)
        (run_dir / PAYLOADS).write_bytes(payloads)
        result = run_level(run_dir, args, key, headers, "VLLM_API_KEY", shutil.which("aiperf"))
        files = {p.relative_to(run_dir).as_posix(): p.read_bytes() for p in sorted(run_dir.rglob("*"))
                 if p.is_file() and p.name != PAYLOADS}
        vllm_cache.reload()  # the server wrote its state when it started, before it answered /models
        server_state = json.loads(state.read_text()) if state and state.is_file() else None
    return {**result, "server_state": server_state, "files": files}
