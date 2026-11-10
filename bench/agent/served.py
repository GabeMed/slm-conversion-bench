"""The server behind an execution's SLM engines, read and checked before the execution exists (no intent, no call).

`bench run` and `bench replay` with an engine `slm:<candidate>[+lora:<name>]` record, per candidate, what
serves it, under `served` in the manifest:

- on RunPod (the candidate's endpoint is the local end of its SSH tunnel, `runpod.local_ports`): the pod this
  machine recorded as serving it (`.runpod/state.json`) and its hourly price; the plan the server was started
  with (`/bench/plan.json` on the pod, written by `bench serve --on runpod up`); the state the server wrote
  about itself on the volume when it started (`bench-serving/<candidate>.json`: the pod, the vLLM version and
  packages, the command, the adapters); `/v1/models` and vLLM's `/version`, asked through the tunnel.
- elsewhere (a Modal endpoint, a local server): `{"checked": false, "why": ...}`. The test split refuses it:
  there the server must be one whose state can be read.

The execution is refused when the served plan is not `modal_apps.common.serve_plan(config, candidate)` (the
same repo, revision, template kwargs, adapters and their sha256, serving settings), leaving out the Modal-only
deploy knobs (`modal_apps.common.DEPLOY_KNOBS`, which change no output); when the server's state was written by
another pod; when the vLLM the server reports is not `serving.vllm_version`; or when `/v1/models` lacks a model
the execution can route to. A pod brought up by other code (another plan) is replaced with `bench serve --on
runpod down` and `up`.
"""
import hashlib
import json
import os
import sys
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from bench import data, paths
from bench.contracts.config import engine_spec
from bench.contracts.facts import canonical

Observe = Callable[[Dict[str, Any], str], Dict[str, Any]]
Get = Callable[[str, Optional[str]], Any]


def slm_models(engines: List[str]) -> Dict[str, List[str]]:
    """{candidate: the model names its engines ask the server for}: the candidate's own name for the base,
    the adapter's served name for `+lora:<name>`."""
    found: Dict[str, set] = {}
    for engine in engines:
        if engine.startswith("slm:"):
            candidate, _, adapter = engine[len("slm:"):].partition("+lora:")
            found.setdefault(candidate, set()).add(adapter or candidate)
    return {candidate: sorted(models) for candidate, models in found.items()}


def runpod_base_url(config: Dict[str, Any], candidate: str) -> Optional[str]:
    """The candidate's base_url when it is the local end of its RunPod tunnel, else None."""
    settings = config.get("runpod")
    port = (settings.get("local_ports") or {}).get(candidate) if isinstance(settings, dict) else None
    if not isinstance(port, int) or isinstance(port, bool):
        return None
    tunnel = f"http://127.0.0.1:{port}/v1"
    return tunnel if engine_spec(config, f"slm:{candidate}")["endpoint"].get("base_url") == tunnel else None


def _repo_imports() -> None:
    if str(paths.ROOT) not in sys.path:  # runpod_apps and modal_apps are not installed packages
        sys.path.insert(0, str(paths.ROOT))


def observe_runpod(config: Dict[str, Any], candidate: str) -> Dict[str, Any]:
    """The pod serving `candidate` (this machine's record, running per the API), its plan.json and the state
    its server wrote on the volume (None for a file that is not there)."""
    _repo_imports()
    from runpod_apps import pods as rp
    from runpod_apps import serve as rserve
    from runpod_apps.settings import VOLUME

    provider = rp.make_provider(config)
    record = provider.state.serving(candidate)
    if record is None:
        raise data.DataError(f"no pod serves {candidate} on this machine's record: bench serve --on runpod up "
                             f"--candidate {candidate}")
    pod = provider.api.get_pod(record["id"])
    if pod is None or pod.get("status") != "RUNNING":
        raise data.DataError(f"the pod {record['id']} of {candidate} is {pod.get('status') if pod else 'gone'}: "
                             f"bench serve --on runpod down --candidate {candidate}, then up")
    remote = provider.remote(pod)
    plan = remote.read(rserve.PLAN)
    state = remote.read(f"{VOLUME}/vllm-cache/bench-serving/{candidate}.json")
    return {"pod_id": pod["id"], "usd_per_hour": pod.get("cost"),
            "plan": json.loads(plan) if plan else None, "server_state": json.loads(state) if state else None}


def http_get(url: str, api_key: Optional[str]) -> Any:
    """The JSON of a GET, with vLLM's key."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        return json.loads(response.read())


def expected_plan(config: Dict[str, Any], candidate: str) -> Dict[str, Any]:
    _repo_imports()
    from modal_apps import common
    return common.serve_plan(config, candidate)


def deploy_knobs() -> tuple:
    _repo_imports()
    from modal_apps import common
    return tuple(common.DEPLOY_KNOBS)


def check_served(config: Dict[str, Any], engines: List[str], split: str, observe: Observe = observe_runpod,
                 get: Get = http_get, expected: Callable[[Dict[str, Any], str], Dict[str, Any]] = expected_plan,
                 knobs: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
    """{candidate: what serves it} for every SLM candidate `engines` reach (None when they reach none), after
    the checks of the module's docstring; a server that fails one refuses the execution (DataError)."""
    wanted = slm_models(engines)
    if not wanted:
        return None
    out: Dict[str, Any] = {}
    for candidate, models in sorted(wanted.items()):
        base = runpod_base_url(config, candidate)
        if base is None:
            why = "its endpoint is not the local end of a RunPod tunnel (runpod.local_ports): its server cannot be read"
            if split == "test":
                raise data.DataError(f"refusing the test split: {candidate}'s server cannot be checked ({why})")
            out[candidate] = {"checked": False, "why": why}
            continue
        found = observe(config, candidate)
        endpoint = engine_spec(config, f"slm:{candidate}")["endpoint"]
        key = os.environ.get(endpoint["api_key_env"]) if endpoint.get("api_key_env") else None
        listed = sorted(m.get("id") for m in (get(f"{base}/models", key) or {}).get("data") or [])
        version = (get(base[:-len("/v1")] + "/version", key) or {}).get("version")
        plan, state = found.get("plan"), found.get("server_state") or {}
        ignored = knobs if knobs is not None else deploy_knobs()
        want = {k: v for k, v in expected(config, candidate).items() if k not in ignored}
        problems = []
        if plan is None:
            problems.append("the pod holds no plan.json")
        else:
            served = {k: v for k, v in plan.items() if k not in ignored}
            differ = sorted(k for k in set(served) | set(want) if served.get(k) != want.get(k))
            if differ:
                problems.append(f"the served plan differs from serve_plan(config) in {', '.join(differ)}")
        if not state:
            problems.append("its server left no state on the volume (bench-serving)")
        elif state.get("pod_id") != found["pod_id"]:
            problems.append(f"the server state on the volume was written by pod {state.get('pod_id')}, not by "
                            f"{found['pod_id']}")
        if version != config["serving"]["vllm_version"]:
            problems.append(f"the server reports vLLM {version}, serving.vllm_version is {config['serving']['vllm_version']}")
        missing = [m for m in models if m not in listed]
        if missing:
            problems.append(f"/v1/models lacks {', '.join(missing)}")
        if problems:
            raise data.DataError(f"refusing to run on the server of {candidate} (pod {found['pod_id']}): "
                                 f"{'; '.join(problems)}. Bring up a fresh server with this code: bench serve --on "
                                 f"runpod down --candidate {candidate}, then up")
        out[candidate] = {"checked": True, "provider": "runpod", "pod_id": found["pod_id"],
                          "usd_per_hour": found.get("usd_per_hour"), "plan": plan,
                          "plan_sha256": hashlib.sha256(canonical(plan)).hexdigest(), "models": listed,
                          "vllm_version": version, "server_state": state}
    return out
