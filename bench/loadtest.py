"""`bench loadtest --engine <engine> --source <run_id> [--as-deployed]`: the execution `loadtest` (SPEC 6.6;
J8's input).

Replays the real calls of an execution: every line of the source run's `calls.jsonl` that the engine under test
served (every invocation and attempt the agent sent, retries included) becomes one request body, the one the
agent's client sends (`bench/agent/hooks.py:chat_model`: model, messages, temperature, max_tokens and the
engine's other params, no streaming), with `model` set to the engine's served name.

**Each engine on its own traffic** (`select_traffic`). An SLM engine is load-tested on the calls its own
end-to-end calibration execution sent it, never on another arm's: the base (`slm:<candidate>`) on every call of
B3 on calib; an adapter (`slm:<candidate>+lora:<name>`) on the calls B4 on calib sent it, which are one cluster's
(the router sends a cluster to its adapter, C1's `cluster`), the per-engine sweep J7 and B5 read. With
`--as-deployed`, B4 as it is served: one server with every adapter, the payload all of B4's calls on calib in the
proportions B4 sent them, each line's `model` the adapter the router chose for it (AIPerf's `raw_payload` sends
each line as it is); its engine is `slm:<candidate>+as-deployed`. The manifest records the sweep's traffic
(`traffic`: the cluster, or `all`; `traffic_kind`: base, adapter or as_deployed) and the composition of its pool by
cluster, call site and engine: J8 reweights the teacher's latencies to that call-site mix for the sweep's limit,
and adds the router's time to a sweep the router feeds. A non-SLM
engine (the local smoke server) replays every call of its source.

**Each level draws its own calls from the engine's whole pool**, seeded (`seeds.loadtest_shuffle` and the level's
index): `request_count` calls, or `request_count_by_level[<concurrency>]` where the configuration gives one, none
twice within the level, **stratified by cluster** (proportional allocation, largest remainder: every level holds
each cluster in the pool's proportion), **or by call site for the base**: B3 does not route, so its calls carry
no cluster, and the call site is the cluster's analogue (`stratified_by`). Levels draw
independently, so a small pool's calls come back in later levels; that is why **every level starts from an
empty prefix cache** (`cold_prefix_cache: always`, the same for every sweep): no level meets a cache that a
previous one filled with its own prompts. A pool smaller than a level's count is replayed whole, once, and the
manifest says so (`short_pool`). The first `warmup_request_count` calls of the seeded order warm the server up
before each level, outside the measured calls.

**The cache reset**: after the warm-up and before AIPerf, the server's prefix cache is emptied (vLLM's
`POST /reset_prefix_cache`, which vLLM 0.30 registers only with VLLM_SERVER_DEV_MODE=1; `bench serve --on
runpod` sets it) and the answer must say `success: true`, retried while blocks are still held. AIPerf's own
`resetKvCache` sends the same POST but accepts `{"success": false}` (vLLM answers 200 either way), so the load
test does it itself. A server without the route is an error: start it in dev mode, or restart vLLM between
levels. Every level is its own run:

    runs/loadtest-<engine>-c<concurrency>-<timestamp>/
        payloads.jsonl                 the lines replayed
        profile_export_aiperf.json     AIPerf's export as it comes out (and the rest of its artifacts)
        manifest.json                  engine, GPU, concurrency, prefix cache, source run, tokenizer, status

`--on local` runs AIPerf here against the engine's endpoint (the llama.cpp smoke server);
`--on modal` runs it in a Modal CPU container (`modal_apps/loadtest.py`) against the deployed vLLM server,
so the client sits in the same cloud as the server; `--on runpod` in a RunPod CPU pod in the serving pod's
data center (`runpod_apps/loadtest.py`), through its own SSH tunnel to the server, or, with `--client
colocated` (an opt-in fallback for a data center with no CPU pod), on the serving pod itself, pinned to 4
cores, on the loopback. Each co-located level records `contention_check` (runpod_apps/colocated.py's
no-contention criterion); a sweep with a level that fails it, or records none (not measured fails), is still
completed and written, and `bench loadtest` then exits 3, printing `CONTENTION: level N ... failed <which>`.
On RunPod each level records the code that ran AIPerf (`pod_code_sha256`, `pod_code_commit`, `pod_code_dirty`):
`commit` is this checkout's, which the pod's need not be.

**One client kind per base and configuration** (`check_one_client`): the manifest records the client
(`client.kind`: local, modal, pod or colocated, and what it ran on), and a sweep whose kind differs from a done
load test of the same base candidate under the same configuration is refused before it starts, so J8's sweeps
all share one client. Either way it first waits
for the server to list the model (a cold vLLM loads its weights first), records the model card the server
lists (for an adapter, the path it was loaded from), and warms it up before measuring. The manifest keeps
what was configured (`gpu`, `prefix_cache`) apart from what was observed (`observed`): the GPUs the Modal or
RunPod server reports it got, and the share of prompt tokens the server says it read from its cache (AIPerf's
`overall_usage_prompt_cache_read_pct`, from the servers' own `usage`). The levels of one invocation share a
`sweep_id` and run in increasing order on one warm server (its prefix cache emptied before each, above).

Credentials: the endpoint's API key (`api_key_env`) and its `headers_env` (each header sent with the value
of the environment variable it names; Modal proxy auth). AIPerf receives them through a YAML config that
holds only `${VAR}` references (`aiperf.yaml`, kept in the run), never on its command line; and every
credential value is scrubbed from AIPerf's artifacts after the run (AIPerf 0.13.0 writes a header such as
Modal-Key verbatim into its export).
"""
import hashlib
import json
from fractions import Fraction
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from bench import paths

EXPORT = "profile_export_aiperf.json"
PAYLOADS = "payloads.jsonl"
AIPERF_CONFIG = "aiperf.yaml"
REDACTED = b"<redacted>"
# params the agent's client does not send as body fields (hooks.chat_model pops them)
CLIENT_ONLY_PARAMS = ("max_tokens", "timeout_s")
AS_DEPLOYED = "+as-deployed"  # the engine label of B4 served as deployed: every adapter on one server
COLD_PREFIX_CACHE = ("never", "always")
RESET_PREFIX_CACHE = "/reset_prefix_cache"  # vLLM's, registered with VLLM_SERVER_DEV_MODE=1


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


def build_payloads(calls: List[Dict[str, Any]], model: str, params: Dict[str, Any], stream: bool,
                   call_sites: Dict[str, Dict[str, Any]], models: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """One request body per C1 line, as the agent's client sends it, addressed to `model`, or with `models`
    ({engine as C1 records it: served name}) to the one that served the call (B4 as deployed). The
    temperature is the call site's (`call_sites`, the configuration's), as the client sends: what C1
    records is the effective one, which a provider that forces its own may have changed."""
    extra = {k: v for k, v in params.items() if k not in CLIENT_ONLY_PARAMS and v is not None}  # null: unset
    payloads = []
    for call in calls:
        body = {"model": models[call["engine"]] if models else model, "messages": call["prompt_messages"],
                "temperature": float(call_sites[call["call_site"]]["temperature"])}
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


def shuffled_order(n: int, seed: Optional[int]) -> List[int]:
    """The indices of a pool of `n` in the order it is replayed: shuffled with `seed`, the same seed giving the
    same order; no seed, the file order."""
    order = list(range(n))
    if seed is not None:
        random.Random(seed).shuffle(order)
    return order


def shuffle_pool(payloads: List[dict], seed: Optional[int]) -> List[dict]:
    """The pool in the seed's order (the warm-up's calls are its first)."""
    return [payloads[i] for i in shuffled_order(len(payloads), seed)]


def level_counts(levels: List[int], settings: Dict[str, Any]) -> List[int]:
    """The calls each level replays, as configured: `request_count_by_level[<concurrency>]`, else `request_count`."""
    by_level = {int(k): int(v) for k, v in (settings.get("request_count_by_level") or {}).items()}
    return [by_level.get(level, int(settings["request_count"])) for level in levels]


def largest_remainder(sizes: Dict[str, int], count: int) -> Dict[str, int]:
    """`count` split over the strata in proportion to their `sizes` (Hamilton's method): each its quota's
    integer part, then one more to the largest remainders, ties to the smaller stratum name. Never more than a
    stratum holds."""
    total = sum(sizes.values())
    quotas = {k: Fraction(count * n, total) for k, n in sizes.items()}
    out = {k: int(q) for k, q in quotas.items()}
    left = count - sum(out.values())
    for k in sorted(sizes, key=lambda k: (-(quotas[k] - out[k]), str(k)))[:left]:
        out[k] += 1
    return {k: min(n, sizes[k]) for k, n in out.items()}


def draw_level(strata: List[str], count: int, seed: int, index: int) -> Dict[str, Any]:
    """The pool positions one level replays: `count` of them (all, if the pool holds fewer: `short_pool`), none
    twice, stratified (`strata[i]` is position i's stratum: its cluster, or its call site for the base) with proportional allocation by largest
    remainder, each stratum's drawn at random with the level's own seed (`seed` and the level's index). Levels
    draw independently: a call can come back in a later level, which starts from an empty prefix cache."""
    rng = random.Random(f"{seed}:{index}")
    by_stratum: Dict[str, List[int]] = {}
    for position, stratum in enumerate(strata):
        by_stratum.setdefault(stratum, []).append(position)
    taken = largest_remainder({k: len(v) for k, v in by_stratum.items()}, min(count, len(strata)))
    chosen = [i for k in sorted(by_stratum) for i in rng.sample(by_stratum[k], taken[k])]
    rng.shuffle(chosen)
    return {"positions": chosen, "short_pool": count > len(strata), "strata": dict(sorted(taken.items()))}


def cold_cache(mode: str) -> bool:
    """Whether every level starts from an empty prefix cache (`loadtest.cold_prefix_cache`)."""
    if mode not in COLD_PREFIX_CACHE:
        raise LoadtestError(f"loadtest.cold_prefix_cache must be one of {COLD_PREFIX_CACHE}, not {mode!r}")
    return mode == "always"


def _require_source(source: Dict[str, Any], arm: str, engine: str) -> None:
    if (source.get("arm"), source.get("split")) != (arm, "calib"):
        raise LoadtestError(f"{engine} is load-tested on {arm}'s own end-to-end execution on calib, not on "
                            f"runs/{source['run_id']} ({source.get('arm')} on {source.get('split')})")


def select_traffic(calls: List[Dict[str, Any]], engine: str, source: Dict[str, Any],
                   as_deployed: bool = False) -> Dict[str, Any]:
    """The calls of the source an engine is load-tested on, and the sweep's traffic (what J8 compares with the
    teacher's latency on the same calls): {"calls", "traffic" (a cluster, or "all"), "kind"}. `source` holds the
    source's run_id, arm and split. An SLM engine only on the calls its own arm sent it on calib (C1's `engine`,
    the router's choice): the base on all of B3's; an adapter on the calls B4 sent it, one cluster's; as
    deployed, on all of B4's, each to its own adapter."""
    sent = sorted({c["engine"] for c in calls})
    if as_deployed:
        if not engine.startswith("slm:") or "+lora:" in engine:
            raise LoadtestError(f"--as-deployed serves an SLM candidate's adapters together: give the base "
                                f"(slm:<candidate>), not {engine}")
        _require_source(source, "B4", engine + AS_DEPLOYED)
        strays = [e for e in sent if not e.startswith(engine + "+lora:")]
        if strays:
            raise LoadtestError(f"runs/{source['run_id']} sent calls to {strays}, not adapters of {engine}")
        return {"calls": list(calls), "traffic": "all", "kind": "as_deployed"}
    if engine.startswith("slm:") and "+lora:" in engine:
        _require_source(source, "B4", engine)
        mine = [c for c in calls if c["engine"] == engine]
        if not mine:
            raise LoadtestError(f"runs/{source['run_id']} sent no call to {engine} (it sent to {sent})")
        clusters = sorted({str(c.get("cluster")) for c in mine})
        if len(clusters) != 1 or clusters[0] == "None":
            raise LoadtestError(f"runs/{source['run_id']} sent {engine} the calls of clusters {clusters}: an adapter "
                                f"serves one cluster, as the router assigned it")
        return {"calls": mine, "traffic": clusters[0], "kind": "adapter"}
    if engine.startswith("slm:"):
        _require_source(source, "B3", engine)
        if sent != [engine]:
            raise LoadtestError(f"runs/{source['run_id']} sent calls to {sent}, not all to {engine}")
        return {"calls": list(calls), "traffic": "all", "kind": "base"}
    return {"calls": list(calls), "traffic": "all", "kind": "replay"}  # the local smoke server: every call, to it


def composition(calls: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """How many of `calls` each cluster, call site and engine has (a call with no cluster under "none")."""
    out: Dict[str, Dict[str, int]] = {"clusters": {}, "call_sites": {}, "engines": {}}
    for call in calls:
        for key, value in (("clusters", call.get("cluster") or "none"), ("call_sites", call["call_site"]),
                           ("engines", call["engine"])):
            out[key][value] = out[key].get(value, 0) + 1
    return {key: dict(sorted(counts.items())) for key, counts in out.items()}


def reset_prefix_cache(base_url: str, headers: Dict[str, str], timeout_s: float, attempts: int = 5,
                       wait_s: float = 2.0) -> Dict[str, Any]:
    """Empty the server's prefix cache (vLLM's POST /reset_prefix_cache, at the server root): the answer must be
    `success: true`; vLLM says false while blocks are still held, so that is retried. Returns the attempts."""
    url = server_root(base_url) + RESET_PREFIX_CACHE
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, data=b"", headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                answer = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code in (404, 405):
                raise LoadtestError(f"{url} is not served (HTTP {e.code}): vLLM registers it only with "
                                    f"VLLM_SERVER_DEV_MODE=1; serve with it, or restart vLLM between levels") from None
            raise LoadtestError(f"the prefix cache reset was refused by {url}: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise LoadtestError(f"the prefix cache reset failed at {url}: {type(e).__name__}: {e}") from None
        if answer.get("success") is True:
            return {"attempts": attempt}
        time.sleep(wait_s)
    raise LoadtestError(f"{url} did not empty the prefix cache in {attempts} attempts (success: false)")


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
              api_key_var: Optional[str], aiperf: Optional[str], measure: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
    """One concurrency level, the same sequence locally and in modal_apps/loadtest.py: wait for the model (and,
    as deployed, every adapter in `models`), warm up on calls outside the measured slice, empty the prefix cache
    when `reset_prefix_cache`, write aiperf.yaml (credentials by reference only), run
    AIPerf, scrub every credential value from what it wrote. `api_key_var` is the variable holding
    `api_key` in AIPerf's environment. `measure`: a context manager around AIPerf's run alone, whose `result`
    is returned as `cpu_sample` (the co-located client's CPU sample, runpod_apps/colocated.py)."""
    cmd = aiperf_command(aiperf, args["tokenizer"])
    sent = auth_headers(api_key, headers)
    waited, card = wait_ready(args["base_url"], sent, args["model"], args["ready_timeout_s"])
    for model in args.get("models") or []:  # as deployed: every adapter the payload names is listed too
        waited += wait_ready(args["base_url"], sent, model, args["ready_timeout_s"])[0]
    warm_up(args["base_url"], sent, args["warmup"], args["timeout_s"])
    reset = reset_prefix_cache(args["base_url"], sent, args["timeout_s"]) if args.get("reset_prefix_cache") else None
    write_aiperf_config(run_dir, aiperf_config(args["url"], args["model"], args["concurrency"], args["request_count"],
                                               args["timeout_s"], args["stream"], api_key_var if api_key else None,
                                               args["headers_env"]))
    redacted: List[str] = []
    sampler = measure() if measure else None
    try:
        with sampler or nullcontext():
            returncode = run_aiperf(run_dir, cmd)
    finally:  # an interrupted AIPerf may have written credentials too
        redacted = scrub_secrets(run_dir, [api_key or "", *headers.values()])
    out = {"returncode": returncode, "ready_after_s": waited, "served_model": card, "secrets_redacted_in": redacted,
           "prefix_cache_reset": reset}
    if sampler is not None:
        out["cpu_sample"] = sampler.result
    return out


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


CLIENTS = ("pod", "colocated")  # --on runpod's load client: a CPU pod (the default), or the serving pod itself


def client_kind(on: str, client: str = "pod") -> str:
    """The kind of load client a sweep runs on: local, modal, or, on RunPod, pod or colocated."""
    if client not in CLIENTS:
        raise LoadtestError(f"--client must be one of {CLIENTS}, not {client!r}")
    if client != "pod" and on != "runpod":
        raise LoadtestError(f"--client {client} is RunPod's (--on runpod), not --on {on}'s")
    return client if on == "runpod" else on


def recorded_client(manifest: Dict[str, Any]) -> str:
    """The client kind a load test's manifest records; one from before the record, its `where` (a RunPod one ran
    on a CPU pod, the only client then)."""
    kind = (manifest.get("client") or {}).get("kind")
    return kind or ("pod" if manifest.get("where") == "runpod" else manifest.get("where"))


def check_one_client(base: str, config_hash: str, kind: str) -> None:
    """Refuse a sweep whose client kind differs from a done load test's of the same base engine under the same
    configuration (J8 takes the sweeps of one base: they must share one client)."""
    others: Dict[str, set] = {}
    for manifest_path in sorted(paths.RUNS.glob("loadtest-*/manifest.json")):
        try:
            found = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if (found.get("type"), found.get("status"), found.get("config_sha256")) != ("loadtest", "done", config_hash):
            continue
        if str(found.get("engine", "")).split("+", 1)[0] != base:
            continue
        other = recorded_client(found)
        if other != kind:
            others.setdefault(other, set()).add(found.get("sweep_id") or manifest_path.parent.name)
    if others:
        listed = "; ".join(f"{k}: {', '.join(sorted(v))}" for k, v in sorted(others.items(), key=lambda kv: str(kv[0])))
        raise LoadtestError(f"the load tests of {base} under this configuration ran on another client ({listed}), "
                            f"not {kind}: every sweep J8 reads must share one client kind; run with that client, "
                            f"or move those runs out of runs/")


def loadtest(config_path: str, engine: str, source: str, on: str, concurrency: Optional[List[int]] = None,
             tokenizer: Optional[Dict[str, str]] = None, as_deployed: bool = False, client: str = "pod") -> List[Path]:
    import os

    from bench.contracts.config import config_sha256, engine_spec, load_config
    from bench.provenance import git_state, redact, scrub

    config = load_config(config_path)
    settings = config["loadtest"]
    kind = client_kind(on, client)
    spec = engine_spec(config, engine)
    endpoint = spec["endpoint"]
    if not endpoint.get("base_url"):
        raise LoadtestError(f"engine {engine} has no endpoint.base_url: serve it first")
    url = server_root(endpoint["base_url"])
    api_key, headers = None, {}
    if on == "local":  # on Modal or RunPod, the credentials are the load client's secrets, never sent from here
        if endpoint.get("api_key_env"):
            api_key = os.environ.get(endpoint["api_key_env"])
            if not api_key:
                raise LoadtestError(f"environment variable {endpoint['api_key_env']} is not set")
        headers = env_headers(endpoint.get("headers_env"))
    source_run = _source_calls(config, source)
    selected = select_traffic(source_run["calls"], engine, {**source_run, "run_id": source}, as_deployed)
    label = engine + AS_DEPLOYED if as_deployed else engine
    models = ({e: engine_spec(config, e)["model"] for e in sorted({c["engine"] for c in selected["calls"]})}
              if as_deployed else None)
    shuffle_seed = config["seeds"].get("loadtest_shuffle")
    if shuffle_seed is None:
        raise LoadtestError("seeds.loadtest_shuffle is not set: each level's draw is seeded")
    pool = selected["calls"]
    payloads = build_payloads(pool, spec["model"], spec.get("params") or {}, settings["stream"], config["call_sites"],
                              models)
    # the router's clusters; B3 does not route (no cluster), so the base is stratified by call site, their analogue
    stratified_by = "cluster" if selected["kind"] in ("adapter", "as_deployed") else "call_site"
    strata = [c.get("cluster") or "none" if stratified_by == "cluster" else c["call_site"] for c in pool]
    warmup = shuffle_pool(payloads, shuffle_seed)[:min(settings["warmup_request_count"], len(payloads))]
    tok = _tokenizer(config, engine, tokenizer)
    vllm = endpoint["kind"] == "vllm"
    levels = list(concurrency or settings["concurrency"])
    counts = level_counts(levels, settings)
    cold = cold_cache(settings.get("cold_prefix_cache", "never"))
    check_one_client(engine.split("+", 1)[0], config_sha256(config), kind)

    remote, aiperf = None, None
    if on == "local":
        aiperf = local_aiperf()
        aiperf_command(aiperf, tok)  # no AIPerf: refused before any run directory or request
    if on == "modal":
        sys.path.insert(0, str(paths.ROOT))
        os.environ["BENCH_CONFIG"] = str(Path(config_path).resolve())  # the Modal apps read this configuration
        from modal_apps import loadtest as modal_loadtest

        remote = modal_loadtest
    elif on == "runpod":
        if not (vllm and engine.startswith("slm:")):
            raise LoadtestError("--on runpod load-tests an SLM candidate's vLLM server (slm:<candidate>[+lora:...])")
        sys.path.insert(0, str(paths.ROOT))
        from runpod_apps import loadtest as runpod_loadtest

        remote = runpod_loadtest.connect(config, engine[len("slm:"):].partition("+lora:")[0], client)
    elif on != "local":
        raise LoadtestError(f"--on must be local, modal or runpod, not {on!r}")

    run_dirs = []
    sweep_id = f"sweep-{_slug(label)}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
    context = remote.app.run() if remote else nullcontext()
    with context:
        for index, (level, configured) in enumerate(zip(levels, counts)):
            drawn = draw_level(strata, configured, shuffle_seed, index)
            chosen = [payloads[i] for i in drawn["positions"]]
            count = len(chosen)
            raw = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in chosen).encode()
            started = datetime.now(timezone.utc)
            run_id = f"loadtest-{_slug(label)}-c{level}-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
            run_dir = paths.RUNS / run_id
            run_dir.mkdir(parents=True)
            (run_dir / PAYLOADS).write_bytes(raw)
            manifest = {
                "run_id": run_id, "type": "loadtest", "sweep_id": sweep_id, "level_index": index,
                "levels": levels,
                "engine": label, "model": spec["model"], "models": sorted(models.values()) if models else None,
                "traffic": selected["traffic"], "traffic_kind": selected["kind"],
                "cluster": selected["traffic"] if selected["kind"] == "adapter" else None,
                "endpoint_kind": endpoint["kind"], "where": on, "client": {"kind": kind},
                "gpu": config["serving"]["gpu"] if vllm else None,
                "prefix_cache": config["serving"]["prefix_caching"] if vllm else None,
                "server": f"vllm {config['serving']['vllm_version']}" if vllm else endpoint["kind"],
                "source_run_id": source, "source_arm": source_run["arm"], "source_split": source_run["split"],
                "source_status": source_run["status"],
                "source_calls": len(payloads), "source_calls_all": len(source_run["calls"]),
                "n_questions": len({c["question_id"] for c in pool}),
                "composition": {"source": composition(pool),
                                "level": composition([pool[i] for i in drawn["positions"]])},
                "stratified_by": stratified_by,
                "strata": drawn["strata"],  # the level's calls per stratum (proportional, largest remainder)
                "payload_shuffle_seed": shuffle_seed, "level_seed": f"{shuffle_seed}:{index}",
                "short_pool": drawn["short_pool"],  # the pool holds fewer calls than the level: all of them, once
                "payloads_sha256": hashlib.sha256(raw).hexdigest(), "concurrency": level,
                "request_count": count, "request_count_configured": configured, "cold_prefix_cache": cold,
                "warmup_request_count": len(warmup),
                "stream": settings["stream"], "tokenizer": tok, "aiperf_version": settings["aiperf_version"],
                "config_sha256": config_sha256(config), **git_state(),
                "started_at": started.isoformat(), "finished_at": None, "status": "running",
            }
            (run_dir / "manifest.json").write_text(redact(json.dumps(manifest, indent=2), config) + "\n")
            args = {"url": url, "model": spec["model"], "concurrency": level, "request_count": count,
                    "models": sorted(models.values()) if models else None, "reset_prefix_cache": cold,
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
                                 "prefix_cache_reset": result.get("prefix_cache_reset"),
                                 "served_model": {k: card.get(k) for k in ("id", "root", "parent")},
                                 "observed": observed(run_dir, server_state), "secrets_redacted_in": redacted})
                if (server_state or {}).get("versions"):  # the serving image's vllm, torch, transformers, flashinfer
                    manifest["server_versions"] = server_state["versions"]
                if remote and result.get("tunnel_calibration"):  # RunPod: what the client's SSH tunnel adds
                    manifest["tunnel_calibration"] = result["tunnel_calibration"]
                if remote and result.get("loopback_calibration"):  # co-located: the loopback's, pinned
                    manifest["loopback_calibration"] = result["loopback_calibration"]
                if remote and result.get("client"):  # RunPod: what the client ran on
                    manifest["client"] = result["client"]
                if remote and result.get("pod_code"):  # RunPod: the code AIPerf ran (`commit` is this checkout's)
                    code = result["pod_code"]
                    manifest.update({"pod_code_sha256": code.get("sha256"), "pod_code_commit": code.get("commit"),
                                     "pod_code_dirty": code.get("dirty"), "pod_code_uploaded": code.get("uploaded")})
                if result.get("cpu_sample") is not None:  # co-located: AIPerf's and the server's CPU over the level
                    sample = dict(result["cpu_sample"])
                    manifest["contention_check"] = sample.pop("contention_check", None)
                    manifest["cpu_sample"] = sample
            except Exception as e:
                manifest["stopped_by"] = scrub(f"{type(e).__name__}: {e}")
                raise
            finally:
                manifest.update({"status": status, "finished_at": datetime.now(timezone.utc).isoformat()})
                (run_dir / "manifest.json").write_text(redact(json.dumps(manifest, indent=2), config) + "\n")
            run_dirs.append(run_dir)
            if status != "done":
                raise LoadtestError(f"AIPerf failed at concurrency {level} (runs/{run_id}/aiperf.out)")
    return run_dirs


CONTENTION_EXIT = 3  # `bench loadtest`'s exit code when a co-located level failed the no-contention criterion


def contention_failures(run_dirs: List[Path]) -> List[str]:
    """One line per level whose manifest records a failed no-contention check (the co-located client's), or, on
    the co-located client, none at all: a quantity that is not measured fails the level. Other clients (a CPU
    pod, Modal, local) share no cores with the server: the check does not apply to them."""
    lines = []
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        check = manifest.get("contention_check")
        level = (f"CONTENTION: level {manifest['level_index'] + 1} (concurrency {manifest['concurrency']}, "
                 f"runs/{manifest['run_id']})")
        if check is not None and not check.get("pass"):
            lines.append(f"{level} failed {'; '.join(check.get('failed') or ['?'])}")
        elif check is None and recorded_client(manifest) == "colocated":
            lines.append(f"{level} failed: not measured (no contention_check recorded)")
    return lines


def cli(args) -> int:
    tokenizer = None
    if args.tokenizer or args.tokenizer_revision:
        if not (args.tokenizer and args.tokenizer_revision and re.match(r"^[0-9a-f]{40}$", args.tokenizer_revision)):
            print("bench loadtest: --tokenizer needs --tokenizer-revision, a 40-hex commit", file=sys.stderr)
            return 2
        tokenizer = {"repo": args.tokenizer, "revision": args.tokenizer_revision}
    errors: Tuple[type, ...] = (LoadtestError,)
    if args.on == "runpod":  # a pod or RunPod call that failed is reported like a load test error
        sys.path.insert(0, str(paths.ROOT))
        from runpod_apps.settings import RunPodError

        errors += (RunPodError,)
    try:
        run_dirs = loadtest(args.config, args.engine, args.source, args.on, args.concurrency, tokenizer,
                            bool(getattr(args, "as_deployed", False)), getattr(args, "client", None) or "pod")
    except errors as e:
        print(f"bench loadtest: {e}", file=sys.stderr)
        return 2
    for run_dir in run_dirs:
        print(run_dir.relative_to(paths.ROOT))
    failures = contention_failures(run_dirs)  # the sweep is complete and written; the exit says it is not clean
    for line in failures:
        print(line, file=sys.stderr)
    return CONTENTION_EXIT if failures else 0
