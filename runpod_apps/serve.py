"""`bench serve --on runpod {setup,download,up,down,status,tunnel}`: the vLLM server of one SLM candidate on a
RunPod GPU pod, the same server modal_apps/serve_vllm.py deploys (`common.serve_plan` and
`common.vllm_command`, unchanged), reached at http://127.0.0.1:<runpod.local_ports[candidate]>/v1 through an
SSH tunnel.

    setup      the network volume, in the data center that has the serving GPU now and holds network volumes
               (or runpod.datacenter);
               once: a volume lives in one data center, and every pod is created there
    download   the candidate's weights at the pinned revision onto the volume, from a CPU pod (once)
    up         a GPU pod: the code, the adapters missing from the volume, vLLM; warm; then the local tunnel.
               Prints the base_url
    down       terminate the candidate's pod (--all: every pod this machine recorded, and every pod the API
               lists under runpod.name_prefix) and close its tunnel
    status     the pods up, their hourly price and what they have cost so far; the tunnels
    tunnel     reopen a serving pod's tunnel (after the laptop slept)

A pod bills by the second until it is terminated: RunPod has no scale-to-zero. `down` when done; each pod also
terminates itself past runpod.max_pod_hours (`--hours` overrides it for one `up`).
"""
import json
import sys
from typing import Any, Dict, List, Optional

from runpod_apps import pods as rp
from runpod_apps import settings as rs
from runpod_apps.settings import VOLUME, WORK, RunPodError
from runpod_apps.ssh import close_tunnel, is_tunnel, open_tunnel, port_open
from runpod_apps.state import now, state_dir

PLAN = f"{WORK}/plan.json"


def complete_marker(repo: str, revision: str) -> str:
    return f"{VOLUME}/hf/bench-complete/{repo.replace('/', '--')}@{revision}"


def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


def check_endpoint(config: Dict[str, Any], settings: Dict[str, Any], candidate: str) -> int:
    """The candidate's endpoint is the tunnel's: base_url on runpod.local_ports, no proxy headers."""
    from bench.train import candidate as entry_of

    port = rs.local_port(settings, candidate)
    endpoint = entry_of(config, candidate)["endpoint"]
    if endpoint.get("base_url") not in (None, base_url(port)):
        raise RunPodError(f"{candidate}'s endpoint.base_url is {endpoint['base_url']!r}, the tunnel serves "
                          f"{base_url(port)}: apply configs/runpod.patch.yaml")
    if endpoint.get("headers_env"):
        raise RunPodError(f"{candidate}'s endpoint.headers_env names Modal proxy headers; RunPod serves behind "
                          "SSH and vLLM's key: apply configs/runpod.patch.yaml")
    return port


# ---------------------------------------------------------------- setup and download

def setup(provider: rp.Provider) -> Dict[str, Any]:
    from runpod_apps.api import best_datacenters

    settings, api = provider.settings, provider.api
    found = provider.state.read()["volume"]
    if found:
        volume = api.get_volume(found["id"])
        if volume is None:
            raise RunPodError(f"the recorded volume {found['id']} no longer exists: delete .runpod/state.json's "
                              "volume and run setup again")
        return {**found, "existing": True}
    gpu = rs.gpu_id(settings, provider.config["serving"]["gpu"])
    datacenter = settings["datacenter"]
    choices = best_datacenters(api.gpu_availability(settings["cloud"], settings["min_cuda_version"]), gpu)
    skipped = []  # where the GPU is but a network volume cannot be
    if datacenter is None:
        if not choices:
            raise RunPodError(f"no data center has {gpu} available now ({settings['cloud']}, CUDA >= "
                              f"{settings['min_cuda_version']}): try later, or set runpod.datacenter to one that "
                              "offers it")
        holding = set(api.volume_datacenters())
        skipped = [c["id"] for c in choices if c["id"] not in holding]
        usable = [c for c in choices if c["id"] in holding]
        if not usable:
            raise RunPodError(f"{gpu} is available now only in data centers without network volumes "
                              f"({', '.join(skipped)}): try later, or set runpod.datacenter to one with both "
                              f"(volumes: {', '.join(sorted(holding)) or 'none listed'})")
        datacenter = usable[0]["id"]
    name = settings["volume"]["name"]
    same = [v for v in api.list_volumes() if v.get("name") == name and v.get("dataCenter") == datacenter]
    volume = same[0] if same else api.create_volume(name, settings["volume"]["size_gb"], datacenter)
    record = {"id": volume["id"], "name": volume["name"], "datacenter": volume["dataCenter"],
              "size_gb": volume["size"], "created_at": now(),
              "gpu_availability_then": {c["id"]: c["availability"] for c in choices},
              "skipped_without_volumes": skipped}
    provider.state.update(lambda d: d.__setitem__("volume", record))
    return {**record, "existing": bool(same)}


def check_secret(provider: rp.Provider) -> None:
    name = provider.settings["secrets"]["vllm_api_key"]
    if name not in provider.api.secret_names():
        raise RunPodError(f"no RunPod secret {name!r}: create it (console > Secrets) with the value the candidates' "
                          "endpoint.api_key_env variable holds (RUNBOOK, RunPod setup)")


def download(provider: rp.Provider, candidate: str, force: bool = False) -> Dict[str, Any]:
    """The candidate's weights onto the volume (Modal: serve_vllm::download), from a CPU pod, once."""
    from bench.train import candidate as entry_of

    hf = entry_of(provider.config, candidate)["hf"]
    key = f"{hf['repo']}@{hf['revision']}"
    if key in provider.state.read()["downloaded"] and not force:
        return {"repo": hf["repo"], "revision": hf["revision"], "skipped": True}
    settings = provider.settings
    hub = _pins(("huggingface-hub", "hf-xet"))
    env = _hf_token_env(settings)
    timeout = provider.config["serving"]["download_timeout_s"]
    with provider.job_pod("download", settings["images"]["cpu_job"], cpu_vcpus=provider.config["loadtest"]["client"]["cpu"],
                          env=env, candidate=candidate) as ready:
        remote = ready["remote"]
        remote.check(f"echo {rp.quote(json.dumps(hf))} > {WORK}/download.json", "writing the download spec")
        remote.start_job("download", f"pip install -q huggingface_hub=={hub['huggingface-hub']} hf_xet=={hub['hf-xet']} "
                                     f"&& python -m runpod_apps.job download {WORK}/download.json")
        remote.wait_job("download", timeout, provider.poll_s)
        if not remote.exists(complete_marker(hf["repo"], hf["revision"])):
            raise RunPodError(f"the download of {key} ended without its completion marker")
    provider.state.update(lambda d: d["downloaded"].__setitem__(key, now()))
    return {"repo": hf["repo"], "revision": hf["revision"], "skipped": False}


def _pins(names) -> Dict[str, str]:
    pins = {}
    for line in (rp.REPO / "env" / "train" / "serve-vllm.lock").read_text().splitlines():
        name, sep, version = line.partition("==")
        if sep and name.strip() in names:
            pins[name.strip()] = version.split(";")[0].strip()
    missing = [n for n in names if n not in pins]
    if missing:
        raise RunPodError(f"env/train/serve-vllm.lock pins no {missing}")
    return pins


def _hf_token_env(settings: Dict[str, Any]) -> Dict[str, str]:
    name = settings["secrets"].get("hf_token")
    return {"HF_TOKEN": rs.secret_ref(name)} if name else {}


# ---------------------------------------------------------------- up, down, status, tunnel

def up(provider: rp.Provider, candidate: str, hours: Optional[float] = None,
       plan: Optional[Dict[str, Any]] = None, tunnel=open_tunnel) -> Dict[str, Any]:
    """The candidate's server. With a pod already up for it, only its tunnel is checked (and reopened)."""
    from modal_apps import common

    import math

    config, settings = provider.config, provider.settings
    if hours is not None and not (math.isfinite(hours) and hours > 0):
        raise RunPodError(f"--hours must be a finite number of hours above 0, not {hours}")
    port = check_endpoint(config, settings, candidate)
    existing = provider.state.serving(candidate)
    if existing:
        live = provider.api.get_pod(existing["id"])
        if live is not None and live.get("status") not in ("TERMINATED", "EXITED", "ERROR"):
            return {**ensure_tunnel(provider, candidate, tunnel), "reused": True}
        provider.terminate(existing["id"])  # gone (its deadline, the console): forget it, serve anew
    plan = plan or common.serve_plan(config, candidate)  # locally, as `modal deploy` computes it
    if not config["serving"]["unauthenticated"]:
        raise RunPodError("serving.unauthenticated is false (Modal proxy auth), which RunPod does not have: apply "
                          "configs/runpod.patch.yaml (vLLM's key behind SSH is the auth)")
    check_secret(provider)
    unrecorded = serving_pods(provider, candidate)  # not only this machine's record: the API's listing too
    if unrecorded:
        raise RunPodError(f"pod(s) {', '.join(p['id'] for p in unrecorded)} already serve {candidate} but this machine "
                          f"did not record them: bench serve --on runpod down --all, then up")
    gpu = rs.gpu_id(settings, config["serving"]["gpu"])
    env = {"VLLM_API_KEY": rs.secret_ref(settings["secrets"]["vllm_api_key"]), **_hf_token_env(settings)}
    body = rp.pod_body(settings, provider.volume(), "serve", settings["images"]["serve"],
                       rp.deadline(settings, "serve", hours), rs.public_key(settings), candidate=candidate,
                       gpu=gpu, env=env)
    pod = provider.create("serve", body, candidate)
    try:
        ready = provider.wait_ssh(pod["id"], config["serving"]["startup_timeout_s"])
        remote = ready["remote"]
        code = provider.upload_code(remote)
        if not remote.exists(complete_marker(plan["repo"], plan["revision"])):
            raise RunPodError(f"{plan['repo']}@{plan['revision']} is not on the volume: "
                              f"bench serve --on runpod download --candidate {candidate}")
        uploaded = provider.ensure_adapters(remote, plan["adapters"])
        remote.check(f"echo {rp.quote(json.dumps(plan))} > {PLAN}", "writing the plan")
        remote.start_job("serve", f"python3 -m runpod_apps.job serve {PLAN}")
        remote.wait_job("serve", config["serving"]["startup_timeout_s"] + config["serving"]["warmup_timeout_s"],
                        provider.poll_s, ready=f"{WORK}/jobs/serve.ready")
        observed = json.loads(remote.read(f"{VOLUME}/vllm-cache/bench-serving/{candidate}.json") or "{}")
        provider.state.set_pod(pod["id"], local_port=port, code_sha256=code, observed_gpus=observed.get("gpus"),
                               adapters={a["served_name"]: a["sha256"] for a in plan["adapters"]})
    except BaseException:
        provider.terminate(pod["id"])  # a server that did not come up is not left billing
        raise
    try:
        opened = ensure_tunnel(provider, candidate, tunnel)
    except RunPodError as e:  # the server is warm: keep it, and say it bills
        raise RunPodError(f"{e}. The server is up (pod {pod['id']}, billing): fix that, then bench serve --on runpod "
                          f"tunnel --candidate {candidate}; or down --candidate {candidate}") from None
    return {**opened, "reused": False, "pod_id": pod["id"], "uploaded_adapters": uploaded,
            "observed_gpus": observed.get("gpus"), "cost_per_hour": pod.get("cost")}


def serving_pods(provider: rp.Provider, candidate: str) -> List[Dict[str, Any]]:
    """The live pods the API lists as serving `candidate` (by their BENCH_ROLE and BENCH_CANDIDATE)."""
    return [p for p in provider.api.list_pods()
            if (p.get("env") or {}).get("BENCH_ROLE") == "serve" and (p.get("env") or {}).get("BENCH_CANDIDATE") == candidate
            and str(p.get("name", "")).startswith(provider.settings["name_prefix"] + "-")
            and p.get("status") not in ("EXITED", "TERMINATED", "ERROR")]


def ensure_tunnel(provider: rp.Provider, candidate: str, tunnel=open_tunnel) -> Dict[str, Any]:
    from modal_apps import common

    record = provider.state.serving(candidate)
    if record is None:
        raise RunPodError(f"no pod serves {candidate}: bench serve --on runpod up --candidate {candidate}")
    port = rs.local_port(provider.settings, candidate)
    if not (is_tunnel(record.get("tunnel_pid")) and port_open(port)):
        pod = provider.api.get_pod(record["id"])
        if pod is None or pod.get("status") != "RUNNING":
            raise RunPodError(f"the pod {record['id']} of {candidate} is {pod.get('status') if pod else 'gone'}: "
                              f"bench serve --on runpod down --candidate {candidate}, then up")
        endpoint = rp.ssh_endpoint(pod)
        pid = tunnel(endpoint["host"], endpoint["port"], rs.private_key(provider.settings),
                     state_dir() / "known_hosts" / pod["id"], port, state_dir() / f"tunnel-{candidate}.log")
        provider.state.set_pod(record["id"], tunnel_pid=pid)
    common.wait_healthy(f"http://127.0.0.1:{port}", None, 60, poll_s=1.0)
    return {"candidate": candidate, "base_url": base_url(port), "pod_id": record["id"]}


def down(provider: rp.Provider, candidate: Optional[str] = None, everything: bool = False) -> Dict[str, Any]:
    """Terminate, and say what each pod cost so far by its hourly price. The pods this machine recorded go
    first; then, with --all, every pod the API lists under runpod.name_prefix (a listing that fails is a
    warning, never a reason to stop). Every pod is tried whatever fails before it: {"done", "failed",
    "warnings"}, and the command exits non-zero when anything failed."""
    if not everything and not candidate:
        raise RunPodError("down needs --candidate <name> or --all")
    done: List[Dict[str, Any]] = []
    failed: List[str] = []
    warnings: List[str] = []

    def terminate(pod_id: str, record: Dict[str, Any]) -> None:
        try:
            close_tunnel(record.get("tunnel_pid"))
            live = None
            try:
                live = provider.api.get_pod(pod_id)
            except Exception as e:  # noqa: BLE001 (the price is only reported: terminate anyway)
                warnings.append(f"{pod_id}: could not read it before terminating ({type(e).__name__}: {e})")
            provider.terminate(pod_id)
            done.append({"id": pod_id, "name": record.get("name"), "was": live.get("status") if live else "gone",
                         "spent_usd_estimate": _spent(live)})
        except Exception as e:  # noqa: BLE001 (one pod's failure never leaves the others running)
            failed.append(f"{pod_id}: {type(e).__name__}: {e}")

    recorded = provider.state.read()["pods"]
    for pod_id, record in list(recorded.items()):
        if everything or (record["role"] == "serve" and record["candidate"] == candidate):
            terminate(pod_id, record)
    if everything:
        prefix = provider.settings["name_prefix"] + "-"
        try:
            listed = provider.api.list_pods()
        except Exception as e:  # noqa: BLE001
            listed = []
            warnings.append(f"the pods could not be listed ({type(e).__name__}: {e}): pods this machine did not "
                            "record may still run: check the console")
        tried = {d["id"] for d in done} | {f.split(":")[0] for f in failed}
        for pod in listed:
            if str(pod.get("name", "")).startswith(prefix) and pod["id"] not in tried \
                    and pod.get("status") != "TERMINATED":
                terminate(pod["id"], {"id": pod["id"], "name": pod.get("name"), "role": "unrecorded"})
    return {"done": done, "failed": failed, "warnings": warnings}


def _spent(pod: Optional[Dict[str, Any]]) -> Optional[float]:
    if not pod or pod.get("cost") is None or not (pod.get("runtime") or {}).get("uptime"):
        return None
    return round(pod["cost"] * pod["runtime"]["uptime"] / 3600, 4)


def status(provider: rp.Provider) -> Dict[str, Any]:
    data = provider.state.read()
    prefix = provider.settings["name_prefix"] + "-"
    live = {p["id"]: p for p in provider.api.list_pods() if str(p.get("name", "")).startswith(prefix)}
    rows = []
    for pod_id in sorted(set(live) | set(data["pods"])):
        record = data["pods"].get(pod_id, {})
        pod = provider.api.get_pod(pod_id)
        rows.append({"id": pod_id, "name": (pod or {}).get("name") or record.get("name"), "role": record.get("role", "unrecorded"),
                     "status": pod.get("status") if pod else "gone", "usd_per_hour": (pod or {}).get("cost"),
                     "spent_usd_estimate": _spent(pod), "tunnel": is_tunnel(record.get("tunnel_pid")),
                     "local_port": record.get("local_port")})
    return {"volume": data["volume"], "pods": rows}


# ---------------------------------------------------------------- the command

def cli(args) -> int:
    from bench.contracts.config import load_config
    from bench.train import TrainError
    from modal_apps.common import SettingsError

    try:
        provider = rp.make_provider(load_config(args.config))
        needs_candidate = args.action in ("download", "up", "tunnel")
        if needs_candidate and not args.candidate:
            raise RunPodError(f"{args.action} needs --candidate <name>")
        if args.action == "setup":
            volume = setup(provider)
            print(f"volume {volume['id']} ({volume['name']}, {volume['size_gb']} GB) in {volume['datacenter']}"
                  f"{' (existing)' if volume['existing'] else ''}; every pod is created there")
            try:
                check_secret(provider)
            except RunPodError as e:
                print(f"before up: {e}", file=sys.stderr)
        elif args.action == "download":
            done = download(provider, args.candidate, args.force)
            print(f"{done['repo']}@{done['revision']} {'already on the volume' if done['skipped'] else 'on the volume'}")
        elif args.action == "up":
            served = up(provider, args.candidate, args.hours)
            if not served["reused"]:
                print(f"pod {served['pod_id']} serves {args.candidate} on {served['observed_gpus']} at "
                      f"${served['cost_per_hour']}/h until `bench serve --on runpod down --candidate {args.candidate}`")
            print(served["base_url"])
        elif args.action == "tunnel":
            print(ensure_tunnel(provider, args.candidate)["base_url"])
        elif args.action == "down":
            result = down(provider, args.candidate, args.all)
            for pod in result["done"]:
                print(f"terminated {pod['id']} {pod['name']} (was {pod['was']}; ~${pod['spent_usd_estimate']})")
            for warning in result["warnings"]:
                print(f"warning: {warning}", file=sys.stderr)
            if result["failed"]:
                raise RunPodError("could not terminate every pod (check the console): " + "; ".join(result["failed"]))
        elif args.action == "status":
            found = status(provider)
            volume = found["volume"]
            print(f"volume: {volume['id']} in {volume['datacenter']}" if volume else "volume: none (setup)")
            for row in found["pods"]:
                print(f"{row['id']} {row['name']} {row['role']} {row['status']} ${row['usd_per_hour']}/h "
                      f"~${row['spent_usd_estimate']} tunnel={'up' if row['tunnel'] else 'down'}")
            if not found["pods"]:
                print("no pod up")
    except (RunPodError, SettingsError, TrainError) as e:
        print(f"bench serve: {e}", file=sys.stderr)
        return 2
    return 0
