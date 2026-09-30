"""What the Modal apps take from config.yaml, and the helpers their containers run. No `modal` import here, so
all of it is testable without the SDK or an account.

The settings are computed on the local side (from `BENCH_CONFIG`, default config.yaml, and for the serving
app `BENCH_CANDIDATE`) and carried into the image as one environment variable, `BENCH_MODAL`: a container
never reads config.yaml, and the settings it runs with are part of the image's identity. Serving knobs that
change no output (`DEPLOY_KNOBS`) come from modal_apps/deploy.yaml and environment overrides instead, so they
can change after `bench prereg` without touching the registered config.yaml.
"""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ENV = "BENCH_MODAL"
ROOT = Path(__file__).resolve().parent.parent
TRAIN_LOCK = ROOT / "env" / "train" / "requirements.lock"
SERVE_LOCK = ROOT / "env" / "train" / "serve-vllm.lock"  # vLLM and everything it pulls, pinned
HF_CACHE = "/root/.cache/huggingface"
VLLM_CACHE = "/root/.cache/vllm"
ADAPTERS = "/adapters"  # the adapters volume: one directory per adapter, named by its sha256_dir
RESULTS = "results"     # beside them: <plan id>.json once a training finished, <plan id>.started while it runs
STATE_DIR = "bench-serving"  # on the vLLM-cache volume: what each running server observed about itself
PORT = 8000
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)  # vllm/config/lora.py:MaxLoRARanks @ v0.30.0
SERVING_KEYS = ("vllm_version", "base_image", "gpu", "cpu", "max_model_len", "gpu_memory_utilization",
                "prefix_caching", "generation_config", "max_concurrent_requests", "startup_timeout_s",
                "download_timeout_s", "warmup_timeout_s", "unauthenticated")
DEPLOY_FILE = ROOT / "modal_apps" / "deploy.yaml"
DEPLOY_KNOBS = {"min_containers": "BENCH_MIN_CONTAINERS", "scaledown_window_s": "BENCH_SCALEDOWN_WINDOW_S"}


class SettingsError(RuntimeError):
    pass


def common_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    modal = config["modal"]
    return {"apps": dict(modal["apps"]), "volumes": dict(modal["volumes"]), "secrets": dict(modal["secrets"]),
            "train": {"gpu": config["train"]["gpu"], "timeout_s": config["train"]["timeout_s"],
                      "reference_timeout_s": config["preflight"]["lora_parity"]["reference_timeout_s"]},
            "loadtest": dict(config["loadtest"]["client"])}


def deploy_knobs(environ: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    """min_containers and scaledown_window_s: modal_apps/deploy.yaml, each overridden by its variable."""
    import yaml

    environ = os.environ if environ is None else environ
    knobs = yaml.safe_load(DEPLOY_FILE.read_text())
    if set(knobs) != set(DEPLOY_KNOBS):
        raise SettingsError(f"{DEPLOY_FILE.name} must set exactly {sorted(DEPLOY_KNOBS)}")
    for knob, variable in DEPLOY_KNOBS.items():
        value = environ.get(variable, knobs[knob])
        try:
            knobs[knob] = int(value)
        except (TypeError, ValueError):
            raise SettingsError(f"{variable} must be an integer, not {value!r}") from None
    return knobs


def check_serving_template(entry: Dict[str, Any]) -> None:
    """The candidate's template reads every kwarg it is served with (thinking off), with or without
    training: bench.train.check_template_reads on its pinned tokenizer."""
    from transformers import AutoTokenizer

    from bench.train import check_template_reads

    tokenizer = AutoTokenizer.from_pretrained(entry["hf"]["repo"], revision=entry["hf"]["revision"])
    check_template_reads(tokenizer, dict(entry.get("chat_template_kwargs") or {}))


def lora_rank(r: int) -> int:
    """vLLM's --max-lora-rank: the smallest rank it accepts that holds an adapter of rank r."""
    for rank in VLLM_LORA_RANKS:
        if rank >= r:
            return rank
    raise SettingsError(f"LoRA rank {r} is above vLLM's largest ({VLLM_LORA_RANKS[-1]})")


def adapters_to_serve(config: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    """The adapters served beside candidate `name`: the `adapters` fact B4 points at when it is this
    candidate's; otherwise every adapter trained on it so far (the first one, for P-4, before the set is
    complete). Each must be in train/adapters/ with the sha256 it is known by."""
    from bench import paths
    from bench.contracts.facts import read_fact, sha256_dir

    root = paths.ROOT / "train" / "adapters"
    fact_path = ((config.get("arms") or {}).get("B4") or {}).get("adapters")
    entries = None
    if fact_path:
        fact, _ = read_fact(str(paths.ROOT / fact_path), "adapters")
        if fact["slm"] == name:
            entries = [(c, a["served_name"], a["sha256"]) for c, a in sorted(fact["adapters"].items())]
    if entries is None:
        entries = []
        for manifest_path in sorted(root.glob("*/manifest.json")) if root.is_dir() else []:
            manifest = json.loads(manifest_path.read_text())
            if manifest["slm"] == name:
                entries.append((manifest["cluster"], manifest["served_name"], manifest["adapter_sha256"]))
    from bench.train import candidate, served_name as content_name

    entry = candidate(config, name)
    trained_for = {"base": dict(entry["hf"]), "chat_template_kwargs": dict(entry.get("chat_template_kwargs") or {})}
    adapters = []
    for cluster, served_name, sha in entries:
        adapter = root / cluster / "adapter"
        if not adapter.is_dir() or sha256_dir(adapter) != sha:
            raise SettingsError(f"train/adapters/{cluster}/adapter is not the adapter {sha} to be served")
        if served_name != content_name(cluster, sha):
            raise SettingsError(f"{cluster}: served name {served_name!r} is not {content_name(cluster, sha)!r}, "
                                "the one its bytes give")
        manifest = json.loads((root / cluster / "manifest.json").read_text())
        if {k: manifest.get(k) for k in trained_for} != trained_for:
            raise SettingsError(f"the adapter of {cluster} was trained on other weights or template kwargs than "
                                f"{name}'s pinned ones: retrain it")
        rank = json.loads((adapter / "adapter_config.json").read_text())["r"]
        adapters.append({"cluster": cluster, "served_name": served_name, "sha256": sha, "r": rank})
    return adapters


def serve_plan(config: Dict[str, Any], name: str, environ: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    from bench.train import TrainError, candidate

    entry = candidate(config, name)
    try:
        check_serving_template(entry)
    except TrainError as e:
        raise SettingsError(f"{name}: {e}") from None
    return {"name": name, "repo": entry["hf"]["repo"], "revision": entry["hf"]["revision"],
            "chat_template_kwargs": dict(entry.get("chat_template_kwargs") or {}),
            "adapters": adapters_to_serve(config, name),
            **{k: config["serving"][k] for k in SERVING_KEYS}, **deploy_knobs(environ)}


def check_auth(plan: Dict[str, Any], environ: Dict[str, str]) -> str:
    """In the serving container, before vLLM starts: fail closed. Modal's proxy auth (unauthenticated false)
    and vLLM's own key are two layers; at least one must hold. Returns which."""
    key = bool(environ.get("VLLM_API_KEY"))
    if plan["unauthenticated"] and not key:
        raise SettingsError("refusing to serve with neither Modal proxy auth nor a VLLM_API_KEY")
    return "+".join([*([] if plan["unauthenticated"] else ["proxy"]), *(["vllm-key"] if key else [])])


def load(kind: str) -> Dict[str, Any]:
    """The settings of one app (`serve`, `train` or `loadtest`): in a container, what the local side
    computed; locally, computed now."""
    if os.environ.get(ENV):
        return json.loads(os.environ[ENV])
    from bench.contracts.config import load_config

    config = load_config(os.environ.get("BENCH_CONFIG") or ROOT / "config.yaml")
    settings = common_settings(config)
    if kind == "serve":
        name = os.environ.get("BENCH_CANDIDATE")
        if not name:
            raise SettingsError("set BENCH_CANDIDATE to the candidate to serve (roles.slm_candidates[].name)")
        settings["plan"] = serve_plan(config, name)
    return settings


def vllm_command(plan: Dict[str, Any]) -> List[str]:
    """`vllm serve` for one candidate and its adapters (design §5.1, "Serving")."""
    cmd = ["vllm", "serve", plan["repo"], "--revision", plan["revision"], "--tokenizer-revision", plan["revision"],
           "--served-model-name", plan["name"], "--host", "0.0.0.0", "--port", str(PORT),
           "--max-model-len", str(plan["max_model_len"]),
           "--gpu-memory-utilization", str(plan["gpu_memory_utilization"]),
           "--enable-prefix-caching" if plan["prefix_caching"] else "--no-enable-prefix-caching",
           "--enable-prompt-tokens-details",  # usage.prompt_tokens_details.cached_tokens (P-2)
           "--generation-config", plan["generation_config"]]
    if plan["chat_template_kwargs"]:
        cmd += ["--default-chat-template-kwargs", json.dumps(plan["chat_template_kwargs"], sort_keys=True)]
    adapters = plan["adapters"]
    if adapters:
        n = str(len(adapters))
        cmd += ["--enable-lora", "--max-loras", n, "--max-cpu-loras", n,
                "--max-lora-rank", str(lora_rank(max(a["r"] for a in adapters))),
                "--lora-modules", *[f"{a['served_name']}={ADAPTERS}/{a['sha256']}" for a in adapters]]
    return cmd


def check_adapters(plan: Dict[str, Any], root: str = ADAPTERS) -> None:
    """In the serving container: each adapter on the volume is the one the plan names, byte for byte."""
    from bench.contracts.facts import sha256_dir

    for adapter in plan["adapters"]:
        path = Path(root) / adapter["sha256"]
        if not path.is_dir() or sha256_dir(path) != adapter["sha256"]:
            raise SettingsError(
                f"adapter {adapter['served_name']} ({adapter['sha256']}) is not on the adapters volume: train it with "
                f"`bench train --on modal`, or upload train/adapters/{adapter['cluster']}/adapter to /{adapter['sha256']}")


def _post(url: str, body: Dict[str, Any], api_key: Optional[str], timeout_s: float) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


def wait_healthy(base: str, process: Optional[subprocess.Popen], timeout_s: float, poll_s: float = 5.0) -> None:
    """Until vLLM answers /health; a server process that exits first is an error, not a wait."""
    started = time.monotonic()
    while True:
        if process is not None and process.poll() is not None:
            raise SettingsError(f"vllm exited with {process.returncode} before serving")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=10) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        if time.monotonic() - started > timeout_s:
            raise SettingsError(f"vllm not healthy after {timeout_s} s")
        time.sleep(poll_s)


def warm_up(base: str, plan: Dict[str, Any], api_key: Optional[str], timeout_s: float) -> List[str]:
    """One short greedy request to the base and to each adapter, so the first measured request finds the
    adapters loaded and the kernels compiled."""
    models = [plan["name"], *[a["served_name"] for a in plan["adapters"]]]
    for model in models:
        _post(f"{base}/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Hello"}],
                                              "max_tokens": 8, "temperature": 0.0}, api_key, timeout_s)
    return models


def stored_training(root: str, plan_id: str, now: float, stale_after_s: float) -> Optional[Dict[str, Any]]:
    """In the training container: the stored result of a finished training of this plan, with its files
    (so running the same command after a disconnect collects it); None when there is none. A training of
    the plan started less than stale_after_s ago is still running: an error, never a second training."""
    from bench.contracts.facts import sha256_dir

    results = Path(root) / RESULTS
    done = results / f"{plan_id}.json"
    if done.is_file():
        result = json.loads(done.read_text())
        adapter = Path(root) / result["sha256"]
        if not adapter.is_dir() or sha256_dir(adapter) != result["sha256"]:
            raise SettingsError(f"the stored adapter {result['sha256']} is not on the volume intact")
        files = {p.relative_to(adapter).as_posix(): p.read_bytes() for p in sorted(adapter.rglob("*")) if p.is_file()}
        return {**result, "files": files, "reused": True}
    started = results / f"{plan_id}.started"
    if started.is_file() and now - float(started.read_text()) < stale_after_s:
        raise SettingsError(f"a training of this plan started {now - float(started.read_text()):.0f} s ago is still "
                            "running: wait for it, then run the same command to collect it")
    return None


def mark_started(root: str, plan_id: str, now: float) -> None:
    results = Path(root) / RESULTS
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{plan_id}.started").write_text(str(now))


def store_training(root: str, plan_id: str, result: Dict[str, Any]) -> None:
    results = Path(root) / RESULTS
    (results / f"{plan_id}.json").write_text(json.dumps(result, sort_keys=True))
    release_started(root, plan_id)


def release_started(root: str, plan_id: str) -> None:
    (Path(root) / RESULTS / f"{plan_id}.started").unlink(missing_ok=True)


def serving_state(plan: Dict[str, Any], nvidia_smi: str) -> Dict[str, Any]:
    """What a serving container records about itself at start, for the executions that measure it: the
    GPUs it actually got (Modal may substitute a type), the command it runs, the adapters it serves."""
    return {"gpus": [line.strip() for line in nvidia_smi.splitlines() if line.strip()],
            "vllm_command": vllm_command(plan), "revision": plan["revision"],
            "adapters": {a["served_name"]: a["sha256"] for a in plan["adapters"]}}


def state_path(root: str, name: str) -> Path:
    return Path(root) / STATE_DIR / f"{name}.json"


def gpu_names() -> List[str]:
    """The GPUs this container got (nvidia-smi), or none where there is no GPU."""
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    except FileNotFoundError:
        return []
    return [line.strip() for line in smi.stdout.splitlines() if line.strip()]


def start_serving(plan: Dict[str, Any], environ: Dict[str, str], launch: Callable = subprocess.Popen,
                  adapters_root: str = ADAPTERS, state_root: str = VLLM_CACHE, commit: Callable[[], Any] = lambda: None,
                  base: str = f"http://127.0.0.1:{PORT}", gpus: Callable[[], List[str]] = gpu_names) -> Any:
    """The serving container's start, in order: refuse without any auth (before anything runs), check the
    adapters' bytes, launch vLLM, wait until healthy, warm up the base and every adapter, record what this
    server observed about itself (for the load test). Returns the vLLM process."""
    auth = check_auth(plan, environ)
    check_adapters(plan, adapters_root)
    process = launch(vllm_command(plan))
    wait_healthy(base, process, plan["startup_timeout_s"])
    warm_up(base, plan, environ.get("VLLM_API_KEY"), plan["warmup_timeout_s"])
    state = state_path(state_root, plan["name"])
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({**serving_state(plan, "\n".join(gpus())), "auth": auth}, indent=2))
    commit()
    return process


def run_training(root: str, key: str, now: float, train: Callable[[], Dict[str, Any]],
                 commit: Callable[[], Any]) -> Dict[str, Any]:
    """The training container's bookkeeping around `train()`: the plan is marked started (and committed, so a
    second attempt sees it), the result is stored under `key` without its files; the marker is released and
    the volume committed whatever happens, so a failed training never blocks the next attempt."""
    mark_started(root, key, now)
    commit()
    try:
        result = train()
        store_training(root, key, {k: v for k, v in result.items() if k != "files"})
        return result
    finally:
        release_started(root, key)
        commit()
