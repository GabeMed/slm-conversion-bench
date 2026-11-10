"""The AIPerf client of `bench loadtest --on runpod`: a CPU pod in the serving pod's data center (the volume's),
so the network between client and server is the cloud's, not the operator's (modal_apps/loadtest.py's
reason). The same object shape as that Modal app (`app.run()`, `run_aiperf.remote(payloads, args)`), so
bench.loadtest drives both the same way, and the same code runs each level (bench.loadtest.run_level).

The client reaches vLLM as the operator does, through an SSH tunnel to the serving pod (the pod publishes SSH
only, and RunPod's HTTPS proxy cuts requests at 100 s): the client pod makes its own key pair, whose public
half the serving pod authorizes for the load test only (removed after it, tagged with the client's pod id);
the private half never leaves the client. The tunnel listens on the endpoint's own local port, so the
base_url in config.yaml is the client's as well. The client's credentials are its own: vLLM's key from the
RunPod secret, never sent from here. The training environment (AIPerf v0.13.0) is the lock's, on the volume.

`--client colocated` (opt-in; the default is the CPU pod above) is the fallback for a data center with no CPU
pod to give: AIPerf runs on the serving pod itself, in the same training environment from the volume, pinned
with `taskset` to 4 cores the server is not pinned to (the server's threads that could run on them are then
kept off them, so the two are disjoint), against vLLM on the loopback (no tunnel, no key).
Its record (cores, nproc, the cgroup's CPU limits) goes in every level's manifest (`client`), with a CPU
sample of AIPerf and of the server over the level (`cpu_sample`) and the loopback's calibration
(`loopback_calibration`, the tunnel's counterpart); see runpod_apps/colocated.py. The serving pod's code is
made this checkout's first (Provider.sync_code: uploaded unless its sha256 is already the pod's), so a pod
brought `up` by older code never runs the load test with it.

Every level records the code that ran AIPerf (`pod_code_sha256`, and `pod_code_commit`/`pod_code_dirty`, the
checkout it was uploaded from); the manifest's own `commit` is this checkout's.
"""
import io
import json
import tarfile
import tempfile
import types
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from runpod_apps import pods as rp
from runpod_apps import settings as rs
from runpod_apps.colocated import format_cpu_list
from runpod_apps.settings import VOLUME, WORK, RunPodError
from runpod_apps.ssh import TUNNEL_OPTIONS

CLIENT_KEY = "/root/.ssh/bench_client"
CLIENTS = ("pod", "colocated")  # `bench loadtest --client`: the default, and the opt-in fallback
LOOPBACK = f"http://127.0.0.1:{rp.POD_PORT}"


def connect(config: Dict[str, Any], candidate: str, client_kind: str = "pod") -> Any:
    """The load client of `candidate`'s server: `pod`, a CPU pod in its data center through an SSH tunnel;
    `colocated`, the serving pod itself, AIPerf pinned to 4 of its cores, on the loopback
    (runpod_apps/colocated.py). Each level's result carries the client's record (`client`)."""
    if client_kind not in CLIENTS:
        raise RunPodError(f"the load client is one of {CLIENTS}, not {client_kind!r}")
    session: Dict[str, Any] = {}

    @contextmanager
    def run(**_options):
        provider = rp.make_provider(config)
        record = provider.state.serving(candidate)
        if record is None:
            raise RunPodError(f"no pod serves {candidate}: bench serve --on runpod up --candidate {candidate}")
        server_pod = provider.api.get_pod(record["id"])
        if server_pod is None or server_pod.get("status") != "RUNNING":
            raise RunPodError(f"the pod {record['id']} serving {candidate} is not running")
        server = provider.remote(server_pod)
        if client_kind == "colocated":
            with colocated(provider, server, server_pod):
                yield
            return
        endpoint = rp.ssh_endpoint(server_pod)
        from bench.train import candidate as entry_of

        port = urllib.parse.urlsplit(entry_of(config, candidate)["endpoint"]["base_url"]).port
        env = {"VLLM_API_KEY": rs.secret_ref(provider.settings["secrets"]["vllm_api_key"])}
        vcpus = config["loadtest"]["client"]["cpu"]
        with provider.job_pod("loadtest", provider.settings["images"]["cpu_job"],
                              cpu_vcpus=vcpus, env=env, candidate=candidate) as ready:
            client, tag = ready["remote"], f"bench-loadtest-{ready['pod']['id']}"
            client.check(f"rm -f {CLIENT_KEY} {CLIENT_KEY}.pub && ssh-keygen -q -t ed25519 -N '' -C {tag} -f {CLIENT_KEY}",
                         "making the client's key")
            public = client.check(f"cat {CLIENT_KEY}.pub", "reading the client's public key").strip()
            try:
                server.check(f"echo {rp.quote(public)} >> /root/.ssh/authorized_keys", "authorizing the load client")
                client.check(f"ssh -f -N -i {CLIENT_KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
                             f"-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ForwardAgent=no "
                             f"{' '.join(TUNNEL_OPTIONS)} "
                             f"-L 127.0.0.1:{port}:localhost:{rp.POD_PORT} -p {endpoint['port']} root@{endpoint['host']} "
                             f"> {WORK}/jobs/tunnel.log 2>&1 < /dev/null",
                             "opening the client's tunnel to the server")
                # the training environment, once per lock on the volume (AIPerf), before any level is timed
                provider.clear_stale_train_lock(client)
                client.start_job("train-env", rp.TRAIN_ENV + '"$PY" -c "import aiperf"')
                client.wait_job("train-env", 3600, provider.poll_s)
                calibration = calibrate(config, candidate, port, server, client, provider.poll_s)
                flavor = provider.settings["cpu_flavor"]
                session.update(provider=provider, client=client, levels=0, calibration=calibration,
                               code={**ready["code"], "uploaded": True},
                               calibration_key="tunnel_calibration", pin="", rewrite={},
                               record={"kind": "pod", "description": f"cpu pod {flavor} {vcpus}",
                                       "pod_id": ready["pod"]["id"], "flavor": flavor, "vcpus": vcpus,
                                       "server_pod_id": server_pod["id"], "reaches_server": "ssh tunnel"})
                yield
            finally:
                session.clear()
                keys = "/root/.ssh/authorized_keys"
                server.shell.run(f"grep -v {tag} {keys} > {keys}.tmp; mv {keys}.tmp {keys}")

    @contextmanager
    def colocated(provider, server, server_pod):
        """AIPerf on the serving pod: the training environment (AIPerf) from the volume, the CPUs read and 4
        cores chosen (runpod_apps.job cpuset), every AIPerf job pinned to them with taskset, against vLLM on
        the loopback; the calibration on the loopback, unpinned (as the server's own side of the tunnel's is)
        and pinned."""
        code = provider.sync_code(server)  # a pod brought `up` by other code (older, or without this client) gets this
        provider.clear_stale_train_lock(server)
        server.start_job("train-env", rp.TRAIN_ENV + '"$PY" -c "import aiperf"')
        server.wait_job("train-env", 3600, provider.poll_s)
        server.check(f"mkdir -p {WORK}/loadtest && rm -f {WORK}/loadtest/cpuset.json", "clearing the last CPU probe")
        server.start_job("cpuset", f"python3 -m runpod_apps.job cpuset {WORK}/loadtest/cpuset.json")
        server.wait_job("cpuset", 300, provider.poll_s)
        probe = json.loads(server.read(f"{WORK}/loadtest/cpuset.json") or "{}")
        if not probe.get("taskset"):
            raise RunPodError("the serving pod has no taskset (util-linux): the co-located client cannot be pinned")
        cores = probe["choice"]["cores"]
        pin = f"taskset -c {format_cpu_list(cores)} "
        calibration = calibrate(config, candidate, None, server, server, provider.poll_s, pin=pin)
        record = {"kind": "colocated", "description": "colocated on the serving pod", "pod_id": server_pod["id"],
                  "server_pod_id": server_pod["id"], "hostname": probe.get("hostname"),
                  "reaches_server": f"loopback {LOOPBACK}", "pinned_cores": cores,
                  "pinned_cores_list": format_cpu_list(cores), "cores_rule": probe["choice"]["rule"],
                  "server_pinned": probe["choice"]["server_pinned"], "server_cores": probe["choice"]["server_cores"],
                  "server_affinity": probe.get("server_affinity"), "server_pids": probe.get("server_pids"),
                  "server_isolated": probe.get("server_isolated"),
                  "server_affinity_after": probe.get("server_affinity_after"),
                  "nproc": probe.get("nproc"), "allowed_cpus": probe.get("allowed_cpus"),
                  "allowed_count": probe.get("allowed_count"), "cgroup": probe.get("cgroup")}
        session.update(provider=provider, client=server, levels=0, calibration=calibration, code=code,
                       calibration_key="loopback_calibration", pin=pin, record=record,
                       rewrite={"base_url": f"{LOOPBACK}/v1", "url": LOOPBACK, "cpu_sample": {"cores": cores}})
        try:
            yield
        finally:
            session.clear()

    def run_aiperf(payloads: bytes, args: Dict[str, Any]) -> Dict[str, Any]:
        from bench.loadtest import PAYLOADS

        if not session:
            raise RunPodError("run_aiperf outside app.run()")
        client, provider = session["client"], session["provider"]
        session["levels"] += 1
        name = f"aiperf-{session['levels']}-c{args['concurrency']}"
        run_dir = f"{WORK}/loadtest/{name}"
        with tempfile.TemporaryDirectory() as work:
            local = Path(work)
            (local / PAYLOADS).write_bytes(payloads)
            (local / "args.json").write_text(json.dumps({**args, **session["rewrite"]}))
            client.check(f"rm -rf {run_dir} {run_dir}.tgz && mkdir -p {run_dir}", "mkdir")
            client.shell.put(local / PAYLOADS, f"{run_dir}/{PAYLOADS}")
            client.shell.put(local / "args.json", f"{run_dir}.args.json")
        client.start_job(name, rp.TRAIN_ENV + f'{session["pin"]}"$PY" -m runpod_apps.job aiperf {run_dir}.args.json {run_dir}')
        client.wait_job(name, config["loadtest"]["client"]["timeout_s"], provider.poll_s)
        client.check(f"tar -czf {run_dir}.tgz --exclude {PAYLOADS} -C {run_dir} .", "packing the artifacts")
        with tempfile.TemporaryDirectory() as work:
            archive = Path(work) / "artifacts.tgz"
            client.shell.get(f"{run_dir}.tgz", archive)
            files = unpack(archive.read_bytes())
        result = json.loads(files.pop("result.json"))
        state = client.read(f"{VOLUME}/vllm-cache/bench-serving/{candidate}.json")
        return {**result, "server_state": json.loads(state) if state else None, "files": files,
                session["calibration_key"]: session["calibration"], "client": session["record"],
                "pod_code": session["code"]}

    return types.SimpleNamespace(app=types.SimpleNamespace(run=run), run_aiperf=types.SimpleNamespace(remote=run_aiperf))


def calibration_spec(config: Dict[str, Any], candidate: str, base_url: str) -> Dict[str, Any]:
    """Concurrency 1 (8 requests) and the sweep's highest (one request per slot), short and greedy."""
    top = max(config["loadtest"]["concurrency"])
    return {"base_url": base_url, "model": candidate, "concurrency": [1, top], "requests": {"1": 8, str(top): top},
            "max_tokens": 8, "timeout_s": 120}


def calibrate(config: Dict[str, Any], candidate: str, port: Optional[int], server, client, poll_s: float,
              pin: str = "") -> Dict[str, Any]:
    """Once per sweep, before it: the same small sample timed on the serving pod against vLLM's own port and on
    the load client, through its tunnel (`port`); `overhead_ms` (client minus server, per concurrency, p50) is
    what the tunnel and the network add to each measured request, which the report can state. Co-located
    (`port` None: the client is the server), the client's side is the loopback again, pinned (`pin`): the
    overhead is the loopback's and the pinning's."""
    out = {}
    client_side = ("client_tunnel", client, f"http://127.0.0.1:{port}/v1", "") if port is not None else \
        ("client_loopback", client, f"{LOOPBACK}/v1", pin)
    for where, remote, base, prefix in (("server_localhost", server, f"{LOOPBACK}/v1", ""), client_side):
        remote.check(f"rm -f {WORK}/calibrate.out.json && "
                     f"echo {rp.quote(json.dumps(calibration_spec(config, candidate, base)))} > {WORK}/calibrate.json",
                     "writing the calibration")
        remote.start_job("calibrate", f"{prefix}python3 -m runpod_apps.job calibrate {WORK}/calibrate.json {WORK}/calibrate.out.json")
        remote.wait_job("calibrate", 1800, poll_s)
        out[where] = json.loads(remote.read(f"{WORK}/calibrate.out.json") or "{}").get("levels")
    by_level = {lv["concurrency"]: lv for lv in out["server_localhost"] or []}
    out["overhead_ms"] = {str(lv["concurrency"]): round(lv["p50_ms"] - by_level[lv["concurrency"]]["p50_ms"], 2)
                          for lv in out[client_side[0]] or []
                          if lv.get("p50_ms") is not None and (by_level.get(lv["concurrency"]) or {}).get("p50_ms") is not None}
    return out


def unpack(archive: bytes) -> Dict[str, bytes]:
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).as_posix()
            if not member.isfile() or name.startswith("/") or ".." in Path(name).parts:
                continue
            files[name[2:] if name.startswith("./") else name] = tar.extractfile(member).read()
    return files
