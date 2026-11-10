"""P-4's HF-PEFT reference on RunPod (`bench preflight --parity <cluster> --on runpod`), as
modal_apps/train.py:peft_reference runs it on Modal: bench.preflight.peft_generate, in the training environment
(env/train/requirements.lock), with the adapter read from the volume at its sha256 (uploaded from
train/adapters/<cluster>/adapter when missing, and checked byte for byte in the pod).

It runs in the pod that serves the adapter's SLM, never in a pod of its own (A100s are scarce, and a fresh pod
may never answer SSH): bench.preflight.lora_parity has already collected the served generations when it is
called, so the server is stopped, its VRAM waited free, the reference run on the same GPU, and the server
started again as `bench serve --on runpod up` starts it, whether the reference succeeded or not (unless the stop
touched nothing: a pod without nvidia-smi keeps its one server). A restart that fails is reported beside the
reference's result or error, never in place of it.

The serving image (runpod.images.serve) has no training environment: the train lock is installed at the first
job onto the volume (a venv keyed by the lock's sha256, Python managed by uv, reused by every later job and by
the load client), or taken from /opt/bench-train when the image has it. The reference runs the code the
serving pod was given at `up`.

`bench preflight --base-check <cluster> --on runpod` (bench/basecheck.py) runs the same way, its job
`runpod_apps.job basecheck` in place of `parity`; with `--no-restart` the server is left stopped (the pod goes
down next)."""
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench import paths
from runpod_apps import pods as rp
from runpod_apps import serve as rserve
from runpod_apps.settings import VOLUME, WORK, RunPodError
from runpod_apps.ssh import Remote, open_tunnel

STOP_TIMEOUT_S = 300   # the server's processes gone and its VRAM free (SIGKILL past half of it)
FREE_MIB = 1024        # what the GPU may still hold, in MiB, once the server is gone
NO_GPU_VIEW = 3        # the stop's exit code when the pod has no nvidia-smi: it stopped nothing

# Stops `runpod_apps.job serve` (its process group: the job's shell, the job, vLLM and vLLM's workers), then
# waits until nothing of it runs and the GPU holds at most FREE_MIB. /proc only (the image may lack procps);
# the patterns are split so that they never match this script's own command line. Every process it signals is
# one it found in this container's /proc (the server's, or a member of its process groups): never a PID that
# nvidia-smi reports, which inside a container may be the host's, naming another process here or none.
# Exit 3 (NO_GPU_VIEW) comes before anything is touched: the server is left serving.
STOP_SERVE = r"""set -u
T={timeout}; FREE={free}; JOB="runpod_apps.job"; VLLM="vllm"
command -v nvidia-smi >/dev/null 2>&1 || {{ echo "no nvidia-smi: the GPU's memory cannot be seen; nothing was stopped"; exit {no_gpu_view}; }}
rm -f {work}/jobs/serve.ready
server() {{
  for d in /proc/[0-9]*; do
    p=${{d#/proc/}}; [ "$p" = "$$" ] && continue
    c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$c" in *"$JOB serve"*|*"$VLLM serve"*) echo "$p";; esac
  done
}}
pgid() {{ local s; s=$(cat "/proc/$1/stat" 2>/dev/null) || return 1; s=${{s##*\) }}; set -- $s; echo "$3"; }}
mine=$(pgid $$)
groups=""
for p in $(server); do
  g=$(pgid "$p") || continue
  [ "$g" -gt 1 ] && [ "$g" != "$mine" ] && groups="$groups $g"
done
members() {{
  for d in /proc/[0-9]*; do
    p=${{d#/proc/}}; [ "$p" = "$$" ] && continue
    g=$(pgid "$p") || continue
    for w in $groups; do [ "$g" = "$w" ] && echo "$p"; done
  done
}}
for g in $groups; do kill -TERM -- "-$g" 2>/dev/null || true; done
kill -TERM $(server) 2>/dev/null || true
start=$(date +%s); killed=""
while :; do
  alive=$( {{ server; members; }} | tr '\n' ' ')
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{{s+=$1}} END {{print s+0}}')
  if [ -z "${{alive// /}}" ] && [ "$used" -le "$FREE" ]; then echo "server stopped; GPU memory used: $used MiB"; exit 0; fi
  waited=$(( $(date +%s) - start ))
  if [ "$waited" -ge $(( T / 2 )) ] && [ -z "$killed" ]; then
    killed=1
    echo "still running after $waited s (processes: ${{alive:-none}}; $used MiB used): SIGKILL"
    for g in $groups; do kill -KILL -- "-$g" 2>/dev/null || true; done
    kill -KILL $alive 2>/dev/null || true
  fi
  if [ "$waited" -ge "$T" ]; then echo "the GPU is not free after $T s (processes: ${{alive:-none}}; $used MiB used)"; exit 1; fi
  sleep 2
done
"""


def peft_reference(config: Dict[str, Any], cluster: str, adapter_sha256: str, base: Dict[str, str],
                   prompts: List[List[dict]], template_kwargs: Dict[str, Any], max_new_tokens: int,
                   top_k: int) -> Dict[str, Any]:
    spec = {"base": base, "prompts": prompts, "template_kwargs": template_kwargs,
            "max_new_tokens": max_new_tokens, "top_k": top_k}
    return run_with_server_stopped(config, cluster, adapter_sha256, "parity", spec, "P-4's reference")


def base_identity(config: Dict[str, Any], cluster: str, adapter_sha256: str, base: Dict[str, str], rows: List[dict],
                  template_kwargs: Dict[str, Any], max_length: int, max_context: int,
                  restart: bool = True) -> Dict[str, Any]:
    """bench.basecheck.teacher_forced in the serving pod, as P-4's reference runs there (`--base-check`)."""
    spec = {"base": base, "rows": rows, "template_kwargs": template_kwargs, "max_length": max_length,
            "max_context": max_context}
    return run_with_server_stopped(config, cluster, adapter_sha256, "basecheck", spec, "the base check", restart,
                                   needs="bench/basecheck.py")


def run_with_server_stopped(config: Dict[str, Any], cluster: str, adapter_sha256: str, job: str, spec: Dict[str, Any],
                            label: str, restart: bool = True, needs: Optional[str] = None) -> Dict[str, Any]:
    """`python -m runpod_apps.job <job> <spec> <out>` in the train environment of the pod serving the cluster's
    SLM, with its server stopped, then started again (unless `restart` is False: the pod is about to go down).
    `needs`: a file of the code the job runs; when the pod's code (uploaded at `up`) lacks it, this checkout's
    code is uploaded first (a superset of what serves)."""
    provider = rp.make_provider(config)
    slm = json.loads((paths.ROOT / "train" / "adapters" / cluster / "manifest.json").read_text())["slm"]
    record = provider.state.serving(slm)
    pod = provider.api.get_pod(record["id"]) if record else None
    if pod is None or pod.get("status") != "RUNNING":
        raise RunPodError(f"no running pod serves {slm} ({label} runs in it, after the served generations): "
                          f"bench serve --on runpod up --candidate {slm}")
    remote = provider.remote(pod)
    provider.ensure_adapters(remote, [{"cluster": cluster, "sha256": adapter_sha256}])
    provider.clear_stale_train_lock(remote)
    if needs and not remote.exists(f"{WORK}/code/{needs}"):  # a pod brought `up` by code without this job
        provider.upload_code(remote)
    spec = {"adapter_sha256": adapter_sha256, **spec,
            "serving_state": f"{VOLUME}/vllm-cache/bench-serving/{slm}.json"}  # the server compared with
    with tempfile.TemporaryDirectory() as work:
        local = Path(work) / f"{job}.json"
        local.write_text(json.dumps(spec))
        remote.shell.put(local, f"{WORK}/{job}.json")
    remote.check(f"rm -f {WORK}/{job}.out.json", f"clearing the last {job} output")
    result, error = None, None
    try:
        try:
            remote.start_job("serve-stop", STOP_SERVE.format(timeout=STOP_TIMEOUT_S, free=FREE_MIB, work=WORK,
                                                             no_gpu_view=NO_GPU_VIEW))
            remote.wait_job("serve-stop", STOP_TIMEOUT_S + 120, provider.poll_s)
        except RunPodError as e:
            if remote.job_rc("serve-stop") == NO_GPU_VIEW:  # it stopped nothing: the server still serves
                raise ServerNotStopped(f"{label} did not run: the pod serving {slm} has no nvidia-smi, so the "
                                       f"server could not be stopped and its VRAM seen free; it was left serving, "
                                       f"untouched ({e})") from e
            raise
        timeout = config["preflight"]["lora_parity"]["reference_timeout_s"]
        remote.start_job(job, rp.TRAIN_ENV + f'"$PY" -m runpod_apps.job {job} {WORK}/{job}.json {WORK}/{job}.out.json')
        remote.wait_job(job, timeout + 3600, provider.poll_s)  # the first job also builds the environment
        out = remote.read(f"{WORK}/{job}.out.json")
        if out is None:
            raise RunPodError(f"{label} ended without its output")
        result = {**json.loads(out), "pod_id": pod["id"], "usd_per_hour": pod.get("cost")}
    except ServerNotStopped:
        raise  # a second server would be started beside the first
    except BaseException as e:  # noqa: B902 -- an interrupted job serves again too, then is re-raised
        error = e
    if not restart:  # the pod goes down next: no server to wait for
        if error is not None:
            raise error
        print(f"{label} finished; the server of {slm} was left stopped: bench serve --on runpod down "
              f"--candidate {slm}", file=sys.stderr)
        return {**result, "server_left_stopped": True}
    # the server is started again whatever the job did; its failure never replaces the job's result or
    # error, and is reported beside them
    try:
        restart_serving(provider, remote, slm, pod["id"], label)
    except RunPodError as restart_error:
        if error is not None:
            error.add_note(f"and then: {restart_error}")
            raise error
        print(f"{label} finished, but {restart_error}", file=sys.stderr)
        return {**result, "restart_error": str(restart_error)}
    if error is not None:
        raise error
    return result


class ServerNotStopped(RunPodError):
    """The server could not be stopped and was not touched: it is not to be started again."""


def restart_serving(provider: rp.Provider, remote: Remote, slm: str, pod_id: str, label: str = "P-4's reference") -> None:
    """The server started again as `up` starts it (its plan, already in the pod, written anew only if gone),
    warm, and healthy through the local tunnel."""
    from modal_apps import common

    config = provider.config
    try:
        if not remote.exists(rserve.PLAN):
            plan = common.serve_plan(config, slm)
            remote.check(f"echo {rp.quote(json.dumps(plan))} > {rserve.PLAN}", "writing the plan")
        remote.check(f"rm -f {WORK}/jobs/serve.ready", "clearing the server's ready marker")
        remote.start_job("serve", f"python3 -m runpod_apps.job serve {rserve.PLAN}")
        remote.wait_job("serve", config["serving"]["startup_timeout_s"] + config["serving"]["warmup_timeout_s"],
                        provider.poll_s, ready=f"{WORK}/jobs/serve.ready")
        observed = json.loads(remote.read(f"{VOLUME}/vllm-cache/bench-serving/{slm}.json") or "{}")
        provider.state.set_pod(pod_id, observed_gpus=observed.get("gpus"))
        rserve.ensure_tunnel(provider, slm, open_tunnel)
    except Exception as e:
        raise RunPodError(f"the server of {slm} did not come back after {label} ({type(e).__name__}: {e}); "
                          f"pod {pod_id} is still billing: bench serve --on runpod down --candidate {slm}, then up") from e
