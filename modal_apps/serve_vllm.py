"""The vLLM server of one SLM candidate and its adapters (design §5.1, "Serving").

OpenAI-compatible; the candidate is served under `--served-model-name <name>` and each adapter under
`--lora-modules <served_name>=/adapters/<sha256>`, which is how the router's engines `slm:<name>` and
`slm:<name>+lora:<served_name>` reach it (`bench/contracts/config.py:engine_spec`). Prefix caching on,
`usage` with cached tokens, thinking off by the candidate's template kwargs, only the request's sampling
parameters (`serving.generation_config: vllm`). Weights come from the HF-cache volume at the pinned
revision, offline: `download` puts them there once. One GPU container at most, so a load test measures one
GPU; it answers only requests carrying the key of the Modal secret `modal.secrets.vllm_api_key`.

    modal run -m modal_apps.serve_vllm::download           (BENCH_CANDIDATE=<name> for both)
    modal deploy -m modal_apps.serve_vllm

then set the candidate's `endpoint.base_url` to the printed URL + `/v1`, and its `endpoint.api_key_env` to
a local variable holding the same key. Before any latency measurement the container warms itself up (a
request to the base and to each adapter) and AIPerf warms up again.
"""
import json
import os
import subprocess

import modal

from modal_apps import common

SETTINGS = common.load("serve")
PLAN = SETTINGS["plan"]

hf_cache = modal.Volume.from_name(SETTINGS["volumes"]["hf_cache"], create_if_missing=True)
vllm_cache = modal.Volume.from_name(SETTINGS["volumes"]["vllm_cache"], create_if_missing=True)
adapters = modal.Volume.from_name(SETTINGS["volumes"]["adapters"], create_if_missing=True)
api_key = modal.Secret.from_name(SETTINGS["secrets"]["vllm_api_key"])
hf_secrets = [modal.Secret.from_name(SETTINGS["secrets"]["hf_token"])] if SETTINGS["secrets"].get("hf_token") else []

image = (
    modal.Image.from_registry(PLAN["base_image"], add_python="3.12")
    .entrypoint([])
    .uv_pip_install(requirements=[str(common.SERVE_LOCK)])
    .env({common.ENV: json.dumps(SETTINGS, sort_keys=True), "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("bench", "modal_apps")
)

app = modal.App(f"{SETTINGS['apps']['serve']}-{PLAN['name']}")


@app.function(image=image, volumes={common.HF_CACHE: hf_cache}, secrets=hf_secrets, timeout=3600)
def download() -> str:
    """The candidate's weights at the pinned revision, into the HF-cache volume (once, before serving)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(PLAN["repo"], revision=PLAN["revision"])
    hf_cache.commit()
    return path


@app.server(image=image, gpu=PLAN["gpu"], cpu=PLAN["cpu"], port=common.PORT, secrets=[api_key],
            volumes={common.HF_CACHE: hf_cache, common.VLLM_CACHE: vllm_cache, common.ADAPTERS: adapters},
            env={"HF_HUB_OFFLINE": "1"}, min_containers=0, max_containers=1,
            max_concurrency=PLAN["max_concurrent_requests"], scaledown_window=PLAN["scaledown_window_s"],
            startup_timeout=PLAN["startup_timeout_s"], unauthenticated=PLAN["unauthenticated"])
class Server:
    @modal.enter()
    def start(self):
        common.check_adapters(PLAN)
        self.process = subprocess.Popen(common.vllm_command(PLAN))
        base = f"http://127.0.0.1:{common.PORT}"
        common.wait_healthy(base, self.process, PLAN["startup_timeout_s"])
        common.warm_up(base, PLAN, os.environ.get("VLLM_API_KEY"))

    @modal.exit()
    def stop(self):
        self.process.terminate()
        self.process.wait(timeout=60)
