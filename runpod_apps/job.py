"""What runs inside a pod, started over SSH as `python -m runpod_apps.job <job> ...` from /bench/code (the
checkout's code, uploaded). Each mirrors a Modal function, through the same shared code:

    serve <plan.json>              modal_apps/serve_vllm.py:Server (common.start_serving), then waits on vLLM
    download <in.json>             modal_apps/serve_vllm.py:download (the weights at the pinned revision)
    parity <in.json> <out.json>    modal_apps/train.py:peft_reference (bench.preflight.peft_generate on the GPU)
    basecheck <in.json> <out.json> bench.basecheck.teacher_forced on the GPU (`bench preflight --base-check`)
    aiperf <args.json> <run dir>   modal_apps/loadtest.py:run_aiperf (bench.loadtest.run_level)
    calibrate <in.json> <out.json> a few short requests at fixed concurrencies, timed (the tunnel's overhead)
    cpuset <out.json>              the serving pod's CPUs, the server's affinity, the cores of a co-located
                                   AIPerf client (runpod_apps/colocated.py)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HF_COMPLETE = "/root/.cache/huggingface/bench-complete"  # one marker per snapshot downloaded whole
READY = "/bench/jobs/serve.ready"
CODE_RECORD = "/bench/code.json"  # runpod_apps.pods.CODE_RECORD: the record of the code uploaded to this pod
SERVING_PACKAGES = ("vllm", "torch", "transformers", "flashinfer-python")
REFERENCE_PACKAGES = ("torch", "transformers", "peft", "accelerate")


def versions(packages) -> dict:
    """The installed version of each distribution, None for one that is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in packages:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def complete_marker(repo: str, revision: str) -> str:
    return f"{HF_COMPLETE}/{repo.replace('/', '--')}@{revision}"


def code_record():
    """The record of the code on this pod (CODE_RECORD: its sha256, commit and dirtiness), None without one."""
    try:
        record = json.loads(Path(CODE_RECORD).read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def serve(plan_path: str) -> int:
    """common.start_serving as the Modal server runs it (auth fails closed first, the adapters' bytes, vLLM,
    health, warm-up, the server's own state), with the weights offline; the vLLM in the image must be the
    version the configuration pins. Writes READY once warm, then lives as long as vLLM. The state records the
    code this server was started from (`code`: the pod's code record as it was when the job began, None
    without one), which a later upload to the pod does not change."""
    code = code_record()
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
                     "vllm_version": vllm.__version__, "versions": versions(SERVING_PACKAGES), "code": code})
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
    out = {**reference, "function_seconds": round(time.monotonic() - started, 3), "observed_gpus": common.gpu_names(),
           "versions": versions(REFERENCE_PACKAGES)}
    serving = Path(spec.get("serving_state") or "/nonexistent")
    if serving.is_file():  # the server it is compared with: its image's versions, beside the reference's
        out["serving_state"] = json.loads(serving.read_text())
    Path(out_path).write_text(json.dumps(out))
    return 0


def basecheck(in_path: str, out_path: str) -> int:
    """The training rows' teacher-forced losses on the pinned base, with the adapter and without (bench/basecheck.py)."""
    started = time.monotonic()
    from bench.basecheck import teacher_forced
    from bench.contracts.facts import sha256_dir
    from modal_apps import common

    spec = json.loads(Path(in_path).read_text())
    adapter = Path(common.ADAPTERS) / spec["adapter_sha256"]
    if not adapter.is_dir() or sha256_dir(adapter) != spec["adapter_sha256"]:
        raise RuntimeError(f"adapter {spec['adapter_sha256']} is not on the volume intact")
    scored = teacher_forced(spec["base"], str(adapter), spec["rows"], spec["template_kwargs"], spec["max_length"],
                            spec["max_context"], "cuda")
    out = {**scored, "function_seconds": round(time.monotonic() - started, 3), "observed_gpus": common.gpu_names(),
           "versions": versions(REFERENCE_PACKAGES)}
    Path(out_path).write_text(json.dumps(out))
    return 0


def aiperf(args_path: str, run_dir: str) -> int:
    """bench.loadtest.run_level with this pod's credentials: vLLM's key from the RunPod secret (VLLM_API_KEY),
    the headers from the variables headers_env names (none on RunPod). With `cpu_sample` in the args (the
    client co-located on the serving pod, pinned to those cores), the CPU each side used over AIPerf's run."""
    from bench.loadtest import env_headers, local_aiperf, run_level

    args = json.loads(Path(args_path).read_text())
    key = os.environ.get("VLLM_API_KEY")
    headers = env_headers(args["headers_env"])
    measure = None
    if args.get("cpu_sample"):
        from runpod_apps.colocated import Sampler

        cores = args["cpu_sample"]["cores"]
        measure = lambda: Sampler(cores)  # noqa: E731
    result = run_level(Path(run_dir), args, key, headers, "VLLM_API_KEY", local_aiperf(), measure=measure)
    Path(run_dir, "result.json").write_text(json.dumps(result))
    return 0


def calibrate(in_path: str, out_path: str) -> int:
    """`requests[c]` short greedy requests at each concurrency c to `base_url`, each timed whole: run on the
    serving pod against localhost and on the load client through its tunnel, the difference is the tunnel's
    (and the network's) share of a request's latency. Standard library only (http.client sets TCP_NODELAY)."""
    import statistics
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    spec = json.loads(Path(in_path).read_text())
    key = os.environ.get("VLLM_API_KEY")
    body = json.dumps({"model": spec["model"], "messages": [{"role": "user", "content": "Reply with the word OK."}],
                       "max_tokens": spec.get("max_tokens", 8), "temperature": 0.0}).encode()
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}

    def one(_):
        started = time.perf_counter()
        request = urllib.request.Request(f"{spec['base_url'].rstrip('/')}/chat/completions", data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=spec.get("timeout_s", 120)) as response:
                response.read()
        except Exception:  # noqa: BLE001 (counted, never fatal: a calibration, not a measurement)
            return None
        return (time.perf_counter() - started) * 1000

    levels = []
    for concurrency in spec["concurrency"]:
        n = int(spec["requests"][str(concurrency)])
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            timed = list(pool.map(one, range(n)))
        ok = [t for t in timed if t is not None]
        levels.append({"concurrency": concurrency, "requests": n, "errors": n - len(ok),
                       "p50_ms": round(statistics.median(ok), 2) if ok else None,
                       "mean_ms": round(statistics.fmean(ok), 2) if ok else None})
    Path(out_path).write_text(json.dumps({"base_url": spec["base_url"], "levels": levels}))
    return 0


def cpuset(out_path: str) -> int:
    from runpod_apps.colocated import write_probe

    return write_probe(out_path)


def main(argv) -> int:
    jobs = {"serve": serve, "download": download, "parity": parity, "basecheck": basecheck, "aiperf": aiperf,
            "calibrate": calibrate, "cpuset": cpuset}
    if not argv or argv[0] not in jobs:
        print(f"usage: python -m runpod_apps.job {{{','.join(jobs)}}} ...", file=sys.stderr)
        return 2
    return jobs[argv[0]](*argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
