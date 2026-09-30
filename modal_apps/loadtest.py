"""The AIPerf client of the load test, in a Modal CPU container beside the deployed vLLM server, so the
network between client and server is the cloud's, not the operator's. Called by
`bench loadtest --on modal`, once per concurrency level; the image is the training image
(env/train/requirements.lock pins AIPerf v0.13.0). Returns AIPerf's artifacts as they come out.
"""
import os
import shutil
import tempfile
from pathlib import Path

import modal

from modal_apps import common
from modal_apps.train import hf_cache, image

SETTINGS = common.load("loadtest")
api_key = modal.Secret.from_name(SETTINGS["secrets"]["vllm_api_key"])

app = modal.App(SETTINGS["apps"]["loadtest"])


@app.function(image=image, cpu=SETTINGS["loadtest"]["cpu"], timeout=SETTINGS["loadtest"]["timeout_s"],
              volumes={common.HF_CACHE: hf_cache}, secrets=[api_key])
def run_aiperf(payloads: bytes, args: dict) -> dict:
    from bench.loadtest import PAYLOADS, aiperf_command, run_aiperf as run, wait_ready

    key = os.environ.get("VLLM_API_KEY")
    with tempfile.TemporaryDirectory() as work:
        run_dir = Path(work)
        (run_dir / PAYLOADS).write_bytes(payloads)
        waited = wait_ready(args["base_url"], key, args["model"], args["ready_timeout_s"])
        cmd = aiperf_command(shutil.which("aiperf"), args["url"], args["model"], args["concurrency"],
                             args["request_count"], args["warmup"], args["tokenizer"], args["timeout_s"],
                             args["stream"], key)
        returncode = run(run_dir, cmd)
        files = {p.relative_to(run_dir).as_posix(): p.read_bytes() for p in sorted(run_dir.rglob("*"))
                 if p.is_file() and p.name != PAYLOADS}
    return {"returncode": returncode, "ready_after_s": waited, "files": files}
