"""What runs inside a pod, started over SSH as `python -m runpod_apps.job <job> ...` from /bench/code (the
checkout's code, uploaded). Each mirrors a Modal function, through the same shared code:

    serve <plan.json>              modal_apps/serve_vllm.py:Server (common.start_serving), then waits on vLLM
    download <in.json>             modal_apps/serve_vllm.py:download (the weights at the pinned revision)
    parity <in.json> <out.json>    modal_apps/train.py:peft_reference (bench.preflight.peft_generate on the GPU)
    aiperf <args.json> <run dir>   modal_apps/loadtest.py:run_aiperf (bench.loadtest.run_level)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HF_COMPLETE = "/root/.cache/huggingface/bench-complete"  # one marker per snapshot downloaded whole
READY = "/bench/jobs/serve.ready"


def complete_marker(repo: str, revision: str) -> str:
    return f"{HF_COMPLETE}/{repo.replace('/', '--')}@{revision}"


def serve(plan_path: str) -> int:
    """common.start_serving as the Modal server runs it (auth fails closed first, the adapters' bytes, vLLM,
    health, warm-up, the server's own state), with the weights offline; the vLLM in the image must be the
    version the configuration pins. Writes READY once warm, then lives as long as vLLM."""
    from modal_apps import common

    plan = json.loads(Path(plan_path).read_text())
    import vllm

    if vllm.__version__ != plan["vllm_version"]:
        raise common.SettingsError(f"the image serves vLLM {vllm.__version__}, serving.vllm_version is "
                                   f"{plan['vllm_version']}: fix runpod.images.serve")
    if not Path(complete_marker(plan["repo"], plan["revision"])).is_file():
        raise common.SettingsError(f"{plan['repo']}@{plan['revision']} is not on the volume: "
                                   f"bench serve --on runpod download --candidate {plan['name']}")
    environ = dict(os.environ)

    def launch(cmd):
        return subprocess.Popen(cmd, env={**environ, "HF_HUB_OFFLINE": "1"})

    process = common.start_serving(plan, environ, launch=launch)
    state = common.state_path(common.VLLM_CACHE, plan["name"])
    recorded = json.loads(state.read_text())
    recorded.update({"provider": "runpod", "image": environ.get("BENCH_IMAGE"), "pod_id": environ.get("RUNPOD_POD_ID"),
                     "vllm_version": vllm.__version__})
    state.write_text(json.dumps(recorded, indent=2))
    Path(READY).write_text(str(state))
    return process.wait()


def download(in_path: str) -> int:
    from huggingface_hub import snapshot_download

    spec = json.loads(Path(in_path).read_text())
    path = snapshot_download(spec["repo"], revision=spec["revision"])
    marker = Path(complete_marker(spec["repo"], spec["revision"]))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(path)
    print(path)
    return 0


def parity(in_path: str, out_path: str) -> int:
    started = time.monotonic()
    from bench.contracts.facts import sha256_dir
    from bench.preflight import peft_generate
    from modal_apps import common

    spec = json.loads(Path(in_path).read_text())
    adapter = Path(common.ADAPTERS) / spec["adapter_sha256"]
    if not adapter.is_dir() or sha256_dir(adapter) != spec["adapter_sha256"]:
        raise RuntimeError(f"adapter {spec['adapter_sha256']} is not on the volume intact")
    reference = peft_generate(spec["base"], str(adapter), spec["prompts"], spec["template_kwargs"],
                              spec["max_new_tokens"], spec["top_k"], "cuda")
    out = {**reference, "function_seconds": round(time.monotonic() - started, 3), "observed_gpus": common.gpu_names()}
    Path(out_path).write_text(json.dumps(out))
    return 0


def aiperf(args_path: str, run_dir: str) -> int:
    """bench.loadtest.run_level with this pod's credentials: vLLM's key from the RunPod secret (VLLM_API_KEY),
    the headers from the variables headers_env names (none on RunPod)."""
    from bench.loadtest import env_headers, local_aiperf, run_level

    args = json.loads(Path(args_path).read_text())
    key = os.environ.get("VLLM_API_KEY")
    headers = env_headers(args["headers_env"])
    result = run_level(Path(run_dir), args, key, headers, "VLLM_API_KEY", local_aiperf())
    Path(run_dir, "result.json").write_text(json.dumps(result))
    return 0


def main(argv) -> int:
    jobs = {"serve": serve, "download": download, "parity": parity, "aiperf": aiperf}
    if not argv or argv[0] not in jobs:
        print(f"usage: python -m runpod_apps.job {{{','.join(jobs)}}} ...", file=sys.stderr)
        return 2
    return jobs[argv[0]](*argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
