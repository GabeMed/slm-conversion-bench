"""`bench loadtest --engine <engine> --source <run_id>`: the execution `loadtest` (SPEC 6.6; J8's input).

Replays the real calls of an execution: every line of the source run's `calls.jsonl` (every invocation and
attempt the agent sent, retries included, in file order) becomes one request body, the one the agent's
client sends (`bench/agent/hooks.py:chat_model`: model, messages, temperature, max_tokens and the engine's
other params, no streaming), with `model` set to the engine's served name. The first
`warmup_request_count` calls warm the server up; then each concurrency level replays its own next
`request_count` calls, once each, as AIPerf v0.13.0 `raw_payload` lines: no prompt is sent twice, so the
prefix cache holds only what real traffic shares, never an exact repeat carried over from a previous level
(a source too small for that wraps around, and the manifest says so). Every level is its own run:

    runs/loadtest-<engine>-c<concurrency>-<timestamp>/
        payloads.jsonl                 the lines replayed
        profile_export_aiperf.json     AIPerf's export as it comes out (and the rest of its artifacts)
        manifest.json                  engine, GPU, concurrency, prefix cache, source run, tokenizer, status

`--on local` runs AIPerf here against the engine's endpoint (the llama.cpp smoke server);
`--on modal` runs it in a Modal CPU container (`modal_apps/loadtest.py`) against the deployed vLLM server,
so the client sits in the same cloud as the server. Either way it first waits for the server to list the
model (a cold vLLM loads its weights first), records the model card the server lists (for an adapter, the
path it was loaded from), and warms it up before measuring. The manifest keeps what was configured
(`gpu`, `prefix_cache`) apart from what was observed (`observed`): the GPUs the Modal server reports it
got, and the share of prompt tokens the server says it read from its cache (AIPerf's
`overall_usage_prompt_cache_read_pct`, from the servers' own `usage`). The levels of one invocation share a
`sweep_id` and run in increasing order on one warm server, so later levels meet a fuller prefix cache.

Credentials: the endpoint's API key (`api_key_env`) and its `headers_env` (each header sent with the value
of the environment variable it names; Modal proxy auth). AIPerf receives them through a YAML config that
holds only `${VAR}` references (`aiperf.yaml`, kept in the run), never on its command line; and every
credential value is scrubbed from AIPerf's artifacts after the run (AIPerf 0.13.0 writes a header such as
Modal-Key verbatim into its export).
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
from typing import Any, Dict, List, Optional, Tuple

from bench import paths

EXPORT = "profile_export_aiperf.json"
PAYLOADS = "payloads.jsonl"
AIPERF_CONFIG = "aiperf.yaml"
REDACTED = b"<redacted>"
# params the agent's client does not send as body fields (hooks.chat_model pops them)
CLIENT_ONLY_PARAMS = ("max_tokens", "timeout_s")


class LoadtestError(RuntimeError):
    pass


def env_headers(headers_env: Optional[Dict[str, str]], environ: Optional[Dict[str, str]] = None,
                error: type = LoadtestError) -> Dict[str, str]:
    """`endpoint.headers_env` ({header: variable}) as headers, each with its variable's value; a configured
    variable that is missing is an error (the contract of `headers_env`)."""
    import os

    environ = os.environ if environ is None else environ
    missing = [variable for variable in (headers_env or {}).values() if not environ.get(variable)]
    if missing:
        raise error(f"environment variable(s) {missing} named by endpoint.headers_env are not set")
    return {header: environ[variable] for header, variable in (headers_env or {}).items()}


def auth_headers(api_key: Optional[str], headers: Dict[str, str]) -> Dict[str, str]:
    return {**headers, **({"Authorization": f"Bearer {api_key}"} if api_key else {})}


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


def level_slice(payloads: List[dict], warmup: int, level: int, count: int) -> Dict[str, Any]:
    """The calls of one concurrency level: the `count` after the warm-up and the previous levels, wrapping
    around (and saying so) only when the source is too small."""
    n = len(payloads)
    start = warmup + level * count
    indices = [(start + i) % n for i in range(count)]
    return {"payloads": [payloads[i] for i in indices], "offset": start % n,
            "repeats": start + count > n or count > n}


def warm_up(base_url: str, headers: Dict[str, str], payloads: List[dict], timeout_s: float) -> int:
    """Send the warm-up calls one by one (outside every measured slice); returns how many."""
    headers = {"Content-Type": "application/json", **headers}
    for body in payloads:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                         headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                response.read()
        except urllib.error.HTTPError as e:
            raise LoadtestError(f"warm-up refused by {base_url}: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            raise LoadtestError(f"warm-up failed at {base_url}: {type(e).__name__}: {e}") from None
    return len(payloads)


def aiperf_config(url: str, model: str, concurrency: int, request_count: int, timeout_s: float, stream: bool,
                  api_key_var: Optional[str], headers_env: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """AIPerf v0.13.0's YAML config for one level: each line of payloads.jsonl once, in order (no AIPerf
    warm-up: the warm-up is done before, on other calls). Credentials only as ${VAR} references, which
    AIPerf resolves from its environment (verified: its CLI flags take no such reference)."""
    endpoint: Dict[str, Any] = {"url": url, "type": "chat", "timeout": timeout_s, "useServerTokenCount": True}
    if stream:
        endpoint["streaming"] = True
    if api_key_var:
        endpoint["apiKey"] = "${" + api_key_var + "}"
    if headers_env:
        endpoint["headers"] = {header: "${" + variable + "}" for header, variable in headers_env.items()}
    return {"schemaVersion": "2.0", "benchmark": {
        "model": model, "endpoint": endpoint,
        "dataset": {"type": "file", "name": "payloads", "path": PAYLOADS, "format": "raw_payload", "sampling": "sequential"},
        "phases": {"type": "concurrency", "name": "profiling", "concurrency": concurrency, "requests": request_count}}}


def aiperf_command(aiperf: Optional[str], tokenizer: Dict[str, str]) -> List[str]:
    """The AIPerf invocation, run from inside the run directory, where aiperf.yaml sits."""
    if not aiperf or not Path(aiperf).is_file():
        raise LoadtestError(f"AIPerf is not installed here ({aiperf}): use env/train")
    return [aiperf, "profile", "--config", AIPERF_CONFIG, "--tokenizer", tokenizer["repo"],
            "--tokenizer-revision", tokenizer["revision"], "--output-artifact-dir", ".", "--ui-type", "none"]


def write_aiperf_config(run_dir: Path, config: Dict[str, Any]) -> None:
    import yaml

    (run_dir / AIPERF_CONFIG).write_text(yaml.safe_dump(config, sort_keys=False))


def scrub_secrets(run_dir: Path, secrets: List[str]) -> List[str]:
    """Replace every credential value in the run's files; the files changed, relative to the run."""
    values = [v.encode() for v in secrets if v]
    changed = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        content = path.read_bytes()
        scrubbed = content
        for value in values:
            scrubbed = scrubbed.replace(value, REDACTED)
        if scrubbed != content:
            path.write_bytes(scrubbed)
            changed.append(path.relative_to(run_dir).as_posix())
    return changed


def wait_ready(base_url: str, headers: Dict[str, str], model: str, timeout_s: float,
               poll_s: float = 5.0) -> Tuple[float, Dict[str, Any]]:
    """Wait until the server lists `model` (vLLM lists every adapter too); seconds waited and its card."""
    started = time.monotonic()
    last = "no answer"
    while True:
        try:
            request = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                cards = {m.get("id"): m for m in json.loads(response.read()).get("data") or []}
            listed = set(cards)
            if model in cards:
                return time.monotonic() - started, cards[model]
            last = f"listed {sorted(listed)}"
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):  # the credentials are refused: waiting will not change that
                raise LoadtestError(f"{base_url} refused the credentials (HTTP {e.code}): check api_key_env and "
                                    "headers_env") from None
            last = f"HTTP {e.code}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
        if time.monotonic() - started > timeout_s:
            raise LoadtestError(f"{model} not served at {base_url} after {timeout_s:.0f} s ({last})")
        time.sleep(poll_s)


def local_aiperf() -> Optional[str]:
    """The AIPerf of this environment (env/train), or on the PATH."""
    import shutil

    beside = Path(sys.executable).parent / "aiperf"
    return str(beside) if beside.is_file() else shutil.which("aiperf")


def run_aiperf(run_dir: Path, cmd: List[str]) -> int:
    """Run AIPerf inside run_dir; its console output goes to aiperf.out beside its own log."""
    with open(run_dir / "aiperf.out", "w") as out:
        return subprocess.run(cmd, cwd=run_dir, stdout=out, stderr=subprocess.STDOUT).returncode


# ---------------------------------------------------------------- one level, here or in the Modal load client

def run_level(run_dir: Path, args: Dict[str, Any], api_key: Optional[str], headers: Dict[str, str],
              api_key_var: Optional[str], aiperf: Optional[str]) -> Dict[str, Any]:
    """One concurrency level, the same sequence locally and in modal_apps/loadtest.py: wait for the model,
    warm up on calls outside the measured slice, write aiperf.yaml (credentials by reference only), run
    AIPerf, scrub every credential value from what it wrote. `api_key_var` is the variable holding
    `api_key` in AIPerf's environment."""
    cmd = aiperf_command(aiperf, args["tokenizer"])
    sent = auth_headers(api_key, headers)
    waited, card = wait_ready(args["base_url"], sent, args["model"], args["ready_timeout_s"])
    warm_up(args["base_url"], sent, args["warmup"], args["timeout_s"])
    write_aiperf_config(run_dir, aiperf_config(args["url"], args["model"], args["concurrency"], args["request_count"],
                                               args["timeout_s"], args["stream"], api_key_var if api_key else None,
                                               args["headers_env"]))
    redacted: List[str] = []
    try:
        returncode = run_aiperf(run_dir, cmd)
    finally:  # an interrupted AIPerf may have written credentials too
        redacted = scrub_secrets(run_dir, [api_key or "", *headers.values()])
    return {"returncode": returncode, "ready_after_s": waited, "served_model": card, "secrets_redacted_in": redacted}


# ---------------------------------------------------------------- the execution

def observed(run_dir: Path, server_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What the run saw, not what was configured: the server's own GPUs (Modal only) and its cache reads."""
    export = json.loads((run_dir / EXPORT).read_text()) if (run_dir / EXPORT).is_file() else {}
    cache = export.get("overall_usage_prompt_cache_read_pct")
    return {"gpus": (server_state or {}).get("gpus"), "vllm_command": (server_state or {}).get("vllm_command"),
            "prompt_cache_read_pct": cache.get("avg") if isinstance(cache, dict) else cache}


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
    return {"calls": calls, "split": manifest.get("split"), "status": manifest.get("status"), "arm": manifest.get("arm")}


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
    from bench.provenance import git_state, scrub

    config = load_config(config_path)
    settings = config["loadtest"]
    spec = engine_spec(config, engine)
    endpoint = spec["endpoint"]
    if not endpoint.get("base_url"):
        raise LoadtestError(f"engine {engine} has no endpoint.base_url: serve it first")
    url = server_root(endpoint["base_url"])
    api_key, headers = None, {}
    if on == "local":  # on Modal, the credentials are the load client's secrets, never sent from here
        if endpoint.get("api_key_env"):
            api_key = os.environ.get(endpoint["api_key_env"])
            if not api_key:
                raise LoadtestError(f"environment variable {endpoint['api_key_env']} is not set")
        headers = env_headers(endpoint.get("headers_env"))
    source_run = _source_calls(config, source)
    payloads = build_payloads(source_run["calls"], spec["model"], spec.get("params") or {}, settings["stream"])
    warmup = payloads[:min(settings["warmup_request_count"], len(payloads))]
    tok = _tokenizer(config, engine, tokenizer)
    vllm = endpoint["kind"] == "vllm"
    levels = concurrency or settings["concurrency"]

    remote, aiperf = None, None
    if on == "local":
        aiperf = local_aiperf()
        aiperf_command(aiperf, tok)  # no AIPerf: refused before any run directory or request
    if on == "modal":
        sys.path.insert(0, str(paths.ROOT))
        os.environ["BENCH_CONFIG"] = str(Path(config_path).resolve())  # the Modal apps read this configuration
        from modal_apps import loadtest as modal_loadtest

        remote = modal_loadtest
    elif on != "local":
        raise LoadtestError(f"--on must be local or modal, not {on!r}")

    run_dirs = []
    sweep_id = f"sweep-{_slug(engine)}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
    context = remote.app.run() if remote else nullcontext()
    with context:
        for index, level in enumerate(levels):
            chosen = level_slice(payloads, len(warmup), index, settings["request_count"])
            raw = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in chosen["payloads"]).encode()
            started = datetime.now(timezone.utc)
            run_id = f"loadtest-{_slug(engine)}-c{level}-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
            run_dir = paths.RUNS / run_id
            run_dir.mkdir(parents=True)
            (run_dir / PAYLOADS).write_bytes(raw)
            manifest = {
                "run_id": run_id, "type": "loadtest", "sweep_id": sweep_id, "level_index": index,
                "levels": list(levels), "engine": engine, "model": spec["model"],
                "endpoint_kind": endpoint["kind"], "where": on,
                "gpu": config["serving"]["gpu"] if vllm else None,
                "prefix_cache": config["serving"]["prefix_caching"] if vllm else None,
                "server": f"vllm {config['serving']['vllm_version']}" if vllm else endpoint["kind"],
                "source_run_id": source, "source_arm": source_run["arm"], "source_split": source_run["split"],
                "source_status": source_run["status"],
                "source_calls": len(payloads), "n_questions": len({c["question_id"] for c in source_run["calls"]}),
                "payload_offset": chosen["offset"], "payloads_repeat": chosen["repeats"],
                "payloads_sha256": hashlib.sha256(raw).hexdigest(), "concurrency": level,
                "request_count": settings["request_count"], "warmup_request_count": len(warmup),
                "stream": settings["stream"], "tokenizer": tok, "aiperf_version": settings["aiperf_version"],
                "config_sha256": config_sha256(config), **git_state(),
                "started_at": started.isoformat(), "finished_at": None, "status": "running",
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            args = {"url": url, "model": spec["model"], "concurrency": level, "request_count": settings["request_count"],
                    "warmup": warmup, "tokenizer": tok, "timeout_s": settings["request_timeout_s"],
                    "stream": settings["stream"], "base_url": endpoint["base_url"],
                    "ready_timeout_s": settings["ready_timeout_s"], "headers_env": endpoint.get("headers_env") or {},
                    "candidate": engine[len("slm:"):].partition("+lora:")[0] if engine.startswith("slm:") else None}
            status = "failed"
            try:
                if remote:
                    result = remote.run_aiperf.remote(raw, args)
                    for rel, content in result["files"].items():
                        (run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                        (run_dir / rel).write_bytes(content)
                    returncode, waited, card = result["returncode"], result["ready_after_s"], result["served_model"]
                    server_state, redacted = result.get("server_state"), result["secrets_redacted_in"]
                else:
                    result = run_level(run_dir, args, api_key, headers, endpoint.get("api_key_env"), aiperf)
                    returncode, waited, card = result["returncode"], result["ready_after_s"], result["served_model"]
                    server_state, redacted = None, result["secrets_redacted_in"]
                status = "done" if returncode == 0 and (run_dir / EXPORT).is_file() else "failed"
                manifest.update({"aiperf_returncode": returncode, "ready_after_s": round(waited, 1),
                                 "served_model": {k: card.get(k) for k in ("id", "root", "parent")},
                                 "observed": observed(run_dir, server_state), "secrets_redacted_in": redacted})
            except Exception as e:
                manifest["stopped_by"] = scrub(f"{type(e).__name__}: {e}")
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
