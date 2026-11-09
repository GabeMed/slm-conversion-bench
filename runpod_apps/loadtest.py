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
"""
import io
import json
import tarfile
import tempfile
import types
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from runpod_apps import pods as rp
from runpod_apps import settings as rs
from runpod_apps.settings import VOLUME, WORK, RunPodError

CLIENT_KEY = "/root/.ssh/bench_client"


def connect(config: Dict[str, Any], candidate: str) -> Any:
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
        endpoint = rp.ssh_endpoint(server_pod)
        server = provider.remote(server_pod)
        from bench.train import candidate as entry_of

        port = urllib.parse.urlsplit(entry_of(config, candidate)["endpoint"]["base_url"]).port
        env = {"VLLM_API_KEY": rs.secret_ref(provider.settings["secrets"]["vllm_api_key"])}
        with provider.job_pod("loadtest", provider.settings["images"]["cpu_job"],
                              cpu_vcpus=config["loadtest"]["client"]["cpu"], env=env, candidate=candidate) as ready:
            client, tag = ready["remote"], f"bench-loadtest-{ready['pod']['id']}"
            client.check(f"rm -f {CLIENT_KEY} {CLIENT_KEY}.pub && ssh-keygen -q -t ed25519 -N '' -C {tag} -f {CLIENT_KEY}",
                         "making the client's key")
            public = client.check(f"cat {CLIENT_KEY}.pub", "reading the client's public key").strip()
            try:
                server.check(f"echo {rp.quote(public)} >> /root/.ssh/authorized_keys", "authorizing the load client")
                client.check(f"ssh -f -N -i {CLIENT_KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
                             f"-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 "
                             f"-L 127.0.0.1:{port}:localhost:{rp.POD_PORT} -p {endpoint['port']} root@{endpoint['host']} "
                             f"> {WORK}/jobs/tunnel.log 2>&1 < /dev/null",
                             "opening the client's tunnel to the server")
                # the training environment, once per lock on the volume (AIPerf), before any level is timed
                client.start_job("train-env", rp.TRAIN_ENV + '"$PY" -c "import aiperf"')
                client.wait_job("train-env", 3600, provider.poll_s)
                session.update(provider=provider, client=client, levels=0)
                yield
            finally:
                session.clear()
                keys = "/root/.ssh/authorized_keys"
                server.shell.run(f"grep -v {tag} {keys} > {keys}.tmp; mv {keys}.tmp {keys}")

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
            (local / "args.json").write_text(json.dumps(args))
            client.check(f"mkdir -p {run_dir}", "mkdir")
            client.shell.put(local / PAYLOADS, f"{run_dir}/{PAYLOADS}")
            client.shell.put(local / "args.json", f"{run_dir}.args.json")
        client.start_job(name, rp.TRAIN_ENV + f'"$PY" -m runpod_apps.job aiperf {run_dir}.args.json {run_dir}')
        client.wait_job(name, config["loadtest"]["client"]["timeout_s"], provider.poll_s)
        client.check(f"tar -czf {run_dir}.tgz --exclude {PAYLOADS} -C {run_dir} .", "packing the artifacts")
        with tempfile.TemporaryDirectory() as work:
            archive = Path(work) / "artifacts.tgz"
            client.shell.get(f"{run_dir}.tgz", archive)
            files = unpack(archive.read_bytes())
        result = json.loads(files.pop("result.json"))
        state = client.read(f"{VOLUME}/vllm-cache/bench-serving/{candidate}.json")
        return {**result, "server_state": json.loads(state) if state else None, "files": files}

    return types.SimpleNamespace(app=types.SimpleNamespace(run=run), run_aiperf=types.SimpleNamespace(remote=run_aiperf))


def unpack(archive: bytes) -> Dict[str, bytes]:
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).as_posix()
            if not member.isfile() or name.startswith("/") or ".." in Path(name).parts:
                continue
            files[name[2:] if name.startswith("./") else name] = tar.extractfile(member).read()
    return files
