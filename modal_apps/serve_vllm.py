"""The vLLM server of one SLM candidate and its adapters (design §5.1, "Serving").

OpenAI-compatible; the candidate is served under `--served-model-name <name>` and each adapter under
`--lora-modules <served_name>=/adapters/<sha256>`, which is how the router's engines `slm:<name>` and
`slm:<name>+lora:<served_name>` reach it (`bench/contracts/config.py:engine_spec`). Prefix caching on,
`usage` with cached tokens, thinking off by the candidate's template kwargs, only the request's sampling
parameters (`serving.generation_config: vllm`). Weights come from the HF-cache volume at the pinned
revision, offline: `download` puts them there once. One GPU container at most, so a load test measures one
GPU. Auth, fail closed: Modal proxy auth (`serving.unauthenticated: false`; clients send the headers their
endpoint's `headers_env` names) and vLLM's own key from the Modal secret `modal.secrets.vllm_api_key` as a
second layer; the container refuses to start with neither. Knobs that change no output (min_containers,
scaledown_window_s) come from modal_apps/deploy.yaml and BENCH_MIN_CONTAINERS / BENCH_SCALEDOWN_WINDOW_S.

    modal run -m modal_apps.serve_vllm::download           (BENCH_CANDIDATE=<name> for both)
    modal deploy -m modal_apps.serve_vllm

then set the candidate's `endpoint.base_url` to the printed URL + `/v1`, and export the variables its
endpoint names: `api_key_env` (SLM_VLLM_API_KEY, the key stored as VLLM_API_KEY in the Modal secret) and
`headers_env` (SLM_MODAL_KEY, SLM_MODAL_SECRET: a Modal proxy auth token). At start the container records the GPUs it got and the command it
runs on the vLLM-cache volume (`bench-serving/<name>.json`), which the load test reads back. Before any latency measurement the container warms itself up (a
request to the base and to each adapter) and AIPerf warms up again.
"""
import json
import os

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


@app.function(image=image, volumes={common.HF_CACHE: hf_cache}, secrets=hf_secrets, timeout=PLAN["download_timeout_s"])
def download() -> str:
    """The candidate's weights at the pinned revision, into the HF-cache volume (once, before serving)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(PLAN["repo"], revision=PLAN["revision"])
    hf_cache.commit()
    return path


@app.server(image=image, gpu=PLAN["gpu"], cpu=PLAN["cpu"], memory=PLAN["memory_gib"] * 1024,  # MiB: what J8 prices
            port=common.PORT, secrets=[api_key],
            volumes={common.HF_CACHE: hf_cache, common.VLLM_CACHE: vllm_cache, common.ADAPTERS: adapters},
            env={"HF_HUB_OFFLINE": "1"}, min_containers=PLAN["min_containers"], max_containers=1,
            max_concurrency=PLAN["max_concurrent_requests"], scaledown_window=PLAN["scaledown_window_s"],
            startup_timeout=PLAN["startup_timeout_s"], unauthenticated=PLAN["unauthenticated"])
class Server:
    @modal.enter()
    def start(self):
        # common.start_serving: fail closed on auth first, then adapters, vLLM, health, warm-up, own state
        self.process = common.start_serving(PLAN, dict(os.environ), commit=vllm_cache.commit)

    @modal.exit()
    def stop(self):
        self.process.terminate()
        self.process.wait(timeout=60)
