"""`bench loadtest --engine <engine> --source <run_id>`: the execution `loadtest` (SPEC 6.6; J8's input).

Replays the real calls of an execution: every line of the source run's `calls.jsonl` (every invocation and
attempt the agent sent, retries included, in file order) becomes one AIPerf `raw_payload` line, the request
body the agent's client sends (`bench/agent/hooks.py:chat_model`: model, messages, temperature, max_tokens
and the engine's other params, no streaming), with `model` set to the engine's served name. AIPerf v0.13.0
cycles over the lines at a fixed concurrency after a warm-up, once per concurrency level, and every level is
its own run:

    runs/loadtest-<engine>-c<concurrency>-<timestamp>/
        payloads.jsonl                 the lines replayed
        profile_export_aiperf.json     AIPerf's export as it comes out (and the rest of its artifacts)
        manifest.json                  engine, GPU, concurrency, prefix cache, source run, tokenizer, status

`--on local` runs AIPerf here against the engine's endpoint (the llama.cpp smoke server);
`--on modal` runs it in a Modal CPU container (`modal_apps/loadtest.py`) against the deployed vLLM server,
so the client sits in the same cloud as the server. Either way it first waits for the server to list the
model (a cold vLLM loads its weights first) and warms it up before measuring.
"""
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench import paths

EXPORT = "profile_export_aiperf.json"
PAYLOADS = "payloads.jsonl"
# params the agent's client does not send as body fields (hooks.chat_model pops them)
CLIENT_ONLY_PARAMS = ("max_tokens", "timeout_s")


class LoadtestError(RuntimeError):
    pass


def build_payloads(calls: List[Dict[str, Any]], model: str, params: Dict[str, Any], stream: bool) -> List[Dict[str, Any]]:
    """One request body per C1 line, as the agent's client sends it, addressed to `model`."""
    extra = {k: v for k, v in params.items() if k not in CLIENT_ONLY_PARAMS}
    payloads = []
    for call in calls:
        body = {"model": model, "messages": call["prompt_messages"], "temperature": call["temperature"]}
        if params.get("max_tokens") is not None:
            body["max_tokens"] = params["max_tokens"]
        body.update(extra)
        if stream:
            body.update({"stream": True, "stream_options": {"include_usage": True}})
        payloads.append(body)
    return payloads


def server_root(base_url: str) -> str:
    """AIPerf's --url is the server root; it appends /v1/chat/completions itself."""
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        raise LoadtestError(f"endpoint.base_url {base_url!r} does not end in /v1 (an OpenAI-compatible base)")
    return base[:-len("/v1")]


def aiperf_command(aiperf: str, url: str, model: str, concurrency: int, request_count: int, warmup: int,
                   tokenizer: Dict[str, str], timeout_s: float, stream: bool, api_key: Optional[str]) -> List[str]:
    """The AIPerf v0.13.0 invocation, run from inside the run directory (relative paths)."""
    cmd = [aiperf, "profile", "--model", model, "--url", url, "--endpoint-type", "chat",
           "--input-file", PAYLOADS, "--custom-dataset-type", "raw_payload",
           "--concurrency", str(concurrency), "--request-count", str(request_count),
           "--warmup-request-count", str(warmup), "--request-timeout-seconds", str(timeout_s),
           "--tokenizer", tokenizer["repo"], "--tokenizer-revision", tokenizer["revision"],
           "--use-server-token-count", "--output-artifact-dir", ".", "--ui-type", "none"]
    if stream:
        cmd.append("--streaming")
    if api_key:
        cmd += ["--api-key", api_key]
    return cmd


def wait_ready(base_url: str, api_key: Optional[str], model: str, timeout_s: float, poll_s: float = 5.0) -> float:
    """Wait until the server lists `model` (vLLM lists every adapter too); seconds waited."""
    started = time.monotonic()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last = "no answer"
    while True:
        try:
            request = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                listed = {m.get("id") for m in json.loads(response.read()).get("data") or []}
            if model in listed:
                return time.monotonic() - started
            last = f"listed {sorted(listed)}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
        if time.monotonic() - started > timeout_s:
            raise LoadtestError(f"{model} not served at {base_url} after {timeout_s:.0f} s ({last})")
        time.sleep(poll_s)


def run_aiperf(run_dir: Path, cmd: List[str]) -> int:
    """Run AIPerf inside run_dir; its console output goes to aiperf.out beside its own log."""
    with open(run_dir / "aiperf.out", "w") as out:
        return subprocess.run(cmd, cwd=run_dir, stdout=out, stderr=subprocess.STDOUT).returncode


# ---------------------------------------------------------------- the execution

def _slug(engine: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", engine)


def _source_calls(config: Dict[str, Any], source: str) -> Dict[str, Any]:
    from bench import barrier
    from bench.contracts.calls import read_calls, validate_calls

    run_dir = paths.RUNS / source
    if not (run_dir / "manifest.json").is_file() or not (run_dir / "calls.jsonl").is_file():
        raise LoadtestError(f"runs/{source} has no manifest.json and calls.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    barrier.ensure_split_allowed(manifest.get("split"), config)
    calls = read_calls(run_dir / "calls.jsonl")
    errors = validate_calls(calls)
    if errors:
        raise LoadtestError(f"runs/{source}/calls.jsonl is not valid C1: {errors[:3]}")
    if not calls:
        raise LoadtestError(f"runs/{source}/calls.jsonl has no calls")
    return {"calls": calls, "split": manifest.get("split"), "status": manifest.get("status")}


def _tokenizer(config: Dict[str, Any], engine: str, override: Optional[Dict[str, str]]) -> Dict[str, str]:
    """The engine's tokenizer, for AIPerf's token counts: the candidate's pinned weights for an SLM."""
    from bench.train import candidate

    if override:
        return override
    if engine.startswith("slm:"):
        name = engine[len("slm:"):].partition("+lora:")[0]
        return dict(candidate(config, name)["hf"])
    raise LoadtestError(f"engine {engine} is not an SLM: give its tokenizer (--tokenizer REPO --tokenizer-revision REV)")


def loadtest(config_path: str, engine: str, source: str, on: str, concurrency: Optional[List[int]] = None,
             tokenizer: Optional[Dict[str, str]] = None) -> List[Path]:
    import os

    from bench.contracts.config import config_sha256, engine_spec, load_config
    from bench.provenance import git_state

    config = load_config(config_path)
    settings = config["loadtest"]
    spec = engine_spec(config, engine)
    endpoint = spec["endpoint"]
    if not endpoint.get("base_url"):
        raise LoadtestError(f"engine {engine} has no endpoint.base_url: serve it first")
    url = server_root(endpoint["base_url"])
    api_key = None
    if endpoint.get("api_key_env") and on == "local":
        api_key = os.environ.get(endpoint["api_key_env"])
        if not api_key:
            raise LoadtestError(f"environment variable {endpoint['api_key_env']} is not set")
    source_run = _source_calls(config, source)
    payloads = build_payloads(source_run["calls"], spec["model"], spec.get("params") or {}, settings["stream"])
    raw = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in payloads).encode()
    tok = _tokenizer(config, engine, tokenizer)
    vllm = endpoint["kind"] == "vllm"
    levels = concurrency or settings["concurrency"]

    remote = None
    if on == "modal":
        sys.path.insert(0, str(paths.ROOT))
        from modal_apps import loadtest as modal_loadtest

        remote = modal_loadtest
    elif on != "local":
        raise LoadtestError(f"--on must be local or modal, not {on!r}")

    run_dirs = []
    context = remote.app.run() if remote else nullcontext()
    with context:
        for level in levels:
            started = datetime.now(timezone.utc)
            run_id = f"loadtest-{_slug(engine)}-c{level}-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
            run_dir = paths.RUNS / run_id
            run_dir.mkdir(parents=True)
            (run_dir / PAYLOADS).write_bytes(raw)
            manifest = {
                "run_id": run_id, "type": "loadtest", "engine": engine, "model": spec["model"],
                "endpoint_kind": endpoint["kind"], "where": on,
                "gpu": config["serving"]["gpu"] if vllm else None,
                "prefix_cache": config["serving"]["prefix_caching"] if vllm else None,
                "server": f"vllm {config['serving']['vllm_version']}" if vllm else endpoint["kind"],
                "source_run_id": source, "source_split": source_run["split"], "source_status": source_run["status"],
                "n_payloads": len(payloads), "n_questions": len({c["question_id"] for c in source_run["calls"]}),
                "payloads_sha256": hashlib.sha256(raw).hexdigest(), "concurrency": level,
                "request_count": settings["request_count"], "warmup_request_count": settings["warmup_request_count"],
                "stream": settings["stream"], "tokenizer": tok, "aiperf_version": settings["aiperf_version"],
                "config_sha256": config_sha256(config), **git_state(),
                "started_at": started.isoformat(), "finished_at": None, "status": "running",
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            args = {"url": url, "model": spec["model"], "concurrency": level, "request_count": settings["request_count"],
                    "warmup": settings["warmup_request_count"], "tokenizer": tok,
                    "timeout_s": settings["request_timeout_s"], "stream": settings["stream"],
                    "base_url": endpoint["base_url"], "ready_timeout_s": settings["ready_timeout_s"]}
            status = "failed"
            try:
                if remote:
                    result = remote.run_aiperf.remote(raw, args)
                    for rel, content in result["files"].items():
                        (run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                        (run_dir / rel).write_bytes(content)
                    returncode, waited = result["returncode"], result["ready_after_s"]
                else:
                    waited = wait_ready(endpoint["base_url"], api_key, spec["model"], settings["ready_timeout_s"])
                    cmd = aiperf_command(str(Path(sys.executable).parent / "aiperf"), url, spec["model"], level,
                                         settings["request_count"], settings["warmup_request_count"], tok,
                                         settings["request_timeout_s"], settings["stream"], api_key)
                    returncode = run_aiperf(run_dir, cmd)
                status = "done" if returncode == 0 and (run_dir / EXPORT).is_file() else "failed"
                manifest.update({"aiperf_returncode": returncode, "ready_after_s": round(waited, 1)})
            except Exception as e:
                manifest["stopped_by"] = f"{type(e).__name__}: {e}"
                raise
            finally:
                manifest.update({"status": status, "finished_at": datetime.now(timezone.utc).isoformat()})
                (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            run_dirs.append(run_dir)
            if status != "done":
                raise LoadtestError(f"AIPerf failed at concurrency {level} (runs/{run_id}/aiperf.out)")
    return run_dirs


def cli(args) -> int:
    tokenizer = None
    if args.tokenizer or args.tokenizer_revision:
        if not (args.tokenizer and args.tokenizer_revision and re.match(r"^[0-9a-f]{40}$", args.tokenizer_revision)):
            print("bench loadtest: --tokenizer needs --tokenizer-revision, a 40-hex commit", file=sys.stderr)
            return 2
        tokenizer = {"repo": args.tokenizer, "revision": args.tokenizer_revision}
    try:
        run_dirs = loadtest(args.config, args.engine, args.source, args.on, args.concurrency, tokenizer)
    except LoadtestError as e:
        print(f"bench loadtest: {e}", file=sys.stderr)
        return 2
    for run_dir in run_dirs:
        print(run_dir.relative_to(paths.ROOT))
    return 0
