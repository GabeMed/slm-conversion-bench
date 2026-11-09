"""RunPod without RunPod: a fake of the REST API v2 (the endpoints runpod_apps uses, with the shapes of
https://api.runpod.io/v2/openapi.json) on a local port, and `LocalShell`, a pod's shell that runs each command
with bash on this machine, the pod's paths mapped into a directory (the volume shared between pods, the rest
per pod). Jobs and in-pod `ssh` are intercepted and handed to the test. Nothing touches the network."""
import json
import re
import shutil
import subprocess
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from runpod_apps.ssh import JOB_HEADER, Result, Shell

OPENAPI = Path(__file__).resolve().parent / "runpod_openapi.json"  # https://api.runpod.io/v2/openapi.json, 2026-11-09


def openapi_validator(schema: str):
    """A JSON Schema (2020-12) validator of one of the spec's component schemas, its $refs resolved in the spec."""
    from jsonschema import Draft202012Validator

    components = json.loads(OPENAPI.read_text())["components"]
    return Draft202012Validator({"$ref": f"#/components/schemas/{schema}", "components": components})


REQUEST_SCHEMAS = {"/v2/pods": "CreatePodRequest", "/v2/network-volumes": "CreateNetworkVolumeRequest"}


class FakeRunPod:
    """Pods start PROVISIONING and are RUNNING, with a direct SSH endpoint, after `ready_after` GETs each.
    `fail` holds faults to inject, each {"method", "path", "status", "times", "apply", "headers", "raw"}: `apply`
    performs the request before answering the error (a create that succeeded although the answer was lost);
    `raw` answers those bytes with that status instead of a problem (an HTML page; with `length`, a body cut
    short). Every POST body is validated against the OpenAPI spec (a 400 listing the errors). `status`
    overrides a pod's status ({pod id: "EXITED"})."""

    def __init__(self, ready_after: int = 1, gpus: Optional[List[Dict[str, Any]]] = None,
                 secrets=("slm-bench-vllm-api-key",)):
        self.pods: Dict[str, Dict[str, Any]] = {}
        self.terminated: List[str] = []
        self.volumes: Dict[str, Dict[str, Any]] = {}
        self.requests: List[Dict[str, Any]] = []
        self.fail: List[Dict[str, Any]] = []
        self.ready_after, self.secrets = ready_after, list(secrets)
        self.gpus = gpus if gpus is not None else [
            {"id": "NVIDIA L40S", "name": "L40S", "memory": 48, "secure": True, "community": True,
             "price": {"secure": 1.09, "community": 0.79}, "maxCount": {"secure": 8, "community": 8},
             "availability": "HIGH", "dataCenters": [{"id": "US-TX-3", "name": "Texas", "availability": "LOW"},
                                                     {"id": "EU-RO-1", "name": "Romania", "availability": "HIGH"},
                                                     {"id": "CA-MTL-1", "name": "Montreal", "availability": "HIGH"},
                                                     {"id": "EU-SE-1", "name": "Sweden", "availability": "NONE"}]},
            {"id": "NVIDIA H100 80GB HBM3", "name": "H100", "memory": 80, "secure": True, "community": False,
             "price": {"secure": 2.69, "community": 0}, "maxCount": {"secure": 8, "community": 0},
             "availability": "HIGH", "dataCenters": [{"id": "US-KS-2", "name": "Kansas", "availability": "HIGH"}]}]
        # the catalog of data centers: networkVolumeTypes empty where a network volume cannot be made (RunPod 400s)
        self.datacenters = [{"id": d, "name": d, "region": "UNKNOWN", "compliance": [], "globalNetwork": False,
                             "networkVolumeTypes": ["STANDARD"]}
                            for d in ("CA-MTL-1", "EU-RO-1", "EU-SE-1", "US-KS-2", "US-TX-3")]
        self.polls: Dict[str, int] = {}
        self.made = 0
        self.status: Dict[str, str] = {}
        self.validators = {path: openapi_validator(schema) for path, schema in REQUEST_SCHEMAS.items()}
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status: int, payload: Any = None, headers: Optional[Dict[str, str]] = None):
                body = b"" if payload is None else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json" if status < 400 else "application/problem+json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle(self, method: str):
                url = urllib.parse.urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                query = dict(urllib.parse.parse_qsl(url.query))
                with fake.lock:
                    fake.requests.append({"method": method, "path": url.path, "query": query, "body": body,
                                          "authorization": self.headers.get("Authorization")})
                    fault = next((f for f in fake.fail if f["method"] == method and url.path == f["path"]
                                  and f.get("times", 1) > 0), None)
                    if fault:
                        fault["times"] = fault.get("times", 1) - 1
                        if fault.get("apply"):
                            fake.route(method, url.path, query, body)
                        if "raw" in fault:
                            self.send_response(fault["status"])
                            self.send_header("Content-Length", str(fault.get("length", len(fault["raw"]))))
                            self.end_headers()
                            self.wfile.write(fault["raw"])
                            self.close_connection = True
                            return None
                        return self._send(fault["status"], {"title": "Error", "status": fault["status"],
                                                            "detail": fault.get("detail", "injected")},
                                          fault.get("headers"))
                    errors = [e.message for e in fake.validators[url.path].iter_errors(body)] \
                        if method == "POST" and url.path in fake.validators else []
                    if errors:
                        return self._send(400, {"title": "Bad Request", "status": 400, "detail": "invalid body",
                                                "errors": errors})
                    status, payload = fake.route(method, url.path, query, body)
                self._send(status, payload)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_DELETE(self):
                self._handle("DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()

    # ---------------------------------------------------------------- the API

    def _pod_view(self, pod: Dict[str, Any]) -> Dict[str, Any]:
        n = self.polls.get(pod["id"], 0)
        running = n >= self.ready_after
        view = dict(pod)
        view.update(status=self.status.get(pod["id"]) or ("RUNNING" if running else "PROVISIONING"), startedAt="2026-11-09T12:00:00Z" if running else None,
                    ssh={"proxy": {"host": "ssh.runpod.io", "port": 22, "username": f"{pod['id']}-x",
                                   "command": f"ssh {pod['id']}-x@ssh.runpod.io"},
                         "direct": {"host": "203.0.113.7", "port": pod["_ssh_port"], "username": "root",
                                    "command": f"ssh root@203.0.113.7 -p {pod['_ssh_port']}"} if running else None},
                    runtime={"uptime": 1800, "ports": [{"private": 22, "public": pod["_ssh_port"], "type": "tcp",
                                                        "ip": "203.0.113.7"}]} if running else None)
        return {k: v for k, v in view.items() if not k.startswith("_")}

    def route(self, method: str, path: str, query: Dict[str, str], body: Any):
        parts = path.strip("/").split("/")
        if parts[:2] == ["v2", "pods"]:
            if len(parts) == 2 and method == "GET":
                return 200, {"pods": [self._pod_view(p) for p in self.pods.values()],
                             "pagination": {"nextCursor": None, "hasNextPage": False}}
            if len(parts) == 2 and method == "POST":
                self.made += 1
                pod_id = f"pod{self.made:04d}"
                pod = {"id": pod_id, "name": body["name"], "image": body["image"], "args": "", "disk": body.get("disk"),
                       "ports": body.get("ports"), "env": body.get("env"), "registry": None, "actions": ["terminate"],
                       "mounts": body.get("mounts"), "cloud": body.get("cloud", "SECURE"),
                       "dataCenterId": (body.get("dataCenterIds") or ["US-TX-3"])[0], "cudaVersion": "13.0",
                       "template": None, "cost": 1.09 if "gpu" in body else 0.12, "locked": False,
                       "globalNetworking": {"enabled": False}, "createdAt": "2026-11-09T12:00:00Z",
                       "_body": body, "_ssh_port": 40000 + self.made - 1}
                if "gpu" in body:
                    pod["gpu"] = {"id": body["gpu"]["id"], "count": 1, "vcpuCount": 16, "memory": 94}
                else:
                    pod["cpu"] = dict(body["cpu"])
                self.pods[pod_id] = pod
                self.polls[pod_id] = 0
                return 201, self._pod_view(pod)
            pod = self.pods.get(parts[2])
            if pod is None:
                return 404, {"title": "Not Found", "status": 404, "detail": "pod not found"}
            if method == "GET":
                self.polls[pod["id"]] += 1
                return 200, self._pod_view(pod)
            if method == "DELETE":
                self.terminated.append(pod["id"])
                del self.pods[pod["id"]]
                return 204, None
        if parts[:2] == ["v2", "network-volumes"]:
            if len(parts) == 2 and method == "GET":
                return 200, {"networkVolumes": list(self.volumes.values()),
                             "pagination": {"nextCursor": None, "hasNextPage": False}}
            if len(parts) == 2 and method == "POST":
                if not any(d["id"] == body["dataCenter"] and d["networkVolumeTypes"] for d in self.datacenters):
                    return 400, {"title": "Bad Request", "status": 400,
                                 "detail": f'Data center "{body["dataCenter"]}" not found or does not support network '
                                           "volumes"}
                volume = {"id": f"vol{len(self.volumes) + 1}", "name": body["name"], "size": body["size"],
                          "dataCenter": body["dataCenter"], "type": "STANDARD"}
                self.volumes[volume["id"]] = volume
                return 201, volume
            volume = self.volumes.get(parts[2])
            return (200, volume) if volume else (404, {"title": "Not Found", "status": 404, "detail": "no volume"})
        if path == "/v2/catalog/gpus":
            return 200, {"gpus": self.gpus}
        if path == "/v2/catalog/datacenters":
            return 200, {"dataCenters": self.datacenters}
        if path == "/v2/account/secrets":
            return 200, {"secrets": [{"id": f"s{i}", "name": n, "createdAt": "2026-11-01T00:00:00Z"}
                                     for i, n in enumerate(self.secrets)]}
        if path == "/v2/billing/pods":
            return 200, {"records": [], "metadata": {"query": query, "recordCount": 0, "uniquePodCount": 0,
                                                     "totals": {"totalAmount": 0.5, "gpuAmount": 0.5, "cpuAmount": 0,
                                                                "diskAmount": 0}}}
        return 404, {"title": "Not Found", "status": 404, "detail": path}

    def created(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """The bodies of the pods created (live or terminated), in order."""
        return [r["body"] for r in self.requests if r["method"] == "POST" and r["path"] == "/v2/pods"
                and (role is None or f"-{role}" in r["body"]["name"])]


# ---------------------------------------------------------------- a pod's shell, on this machine

_MAPPED = ("/runpod-volume", "/bench", "/root")
_PATH = re.compile(r"(?<![\w.\-/])(" + "|".join(re.escape(p) for p in _MAPPED) + r")(?=/|\b)")


class LocalShell(Shell):
    """One pod: /runpod-volume is `volume` (shared by every pod), /bench and /root are the pod's own
    directory. A job (Remote.start_job) is handed to `jobs[<its name without the level suffix>]`, which
    returns its exit code; an `ssh` command to `on_ssh`. Everything else runs in bash."""

    def __init__(self, pod_dir: Path, volume: Path, jobs: Optional[Dict[str, Callable[..., Optional[int]]]] = None,
                 on_ssh: Optional[Callable[[str], int]] = None):
        """`jobs` None: a job runs in bash like any command (nohup, detached), as in a pod."""
        self.pod_dir, self.volume, self.jobs, self.on_ssh = pod_dir, volume, jobs, on_ssh
        self.commands: List[str] = []
        self.started: List[str] = []
        for d in ("bench/jobs", "root/.ssh"):
            (pod_dir / d).mkdir(parents=True, exist_ok=True)
        (pod_dir / "bench" / "env").write_text("declare -x VLLM_API_KEY=\"from-the-secret\"\n")  # what BOOTSTRAP saves
        (pod_dir / "root" / ".ssh" / "authorized_keys").write_text("ssh-ed25519 AAAA operator\n")
        volume.mkdir(parents=True, exist_ok=True)

    def local(self, path: str) -> Path:
        return Path(self.map(path))

    def map(self, text: str) -> str:
        def to_local(match):
            prefix = match.group(1)
            return str(self.volume) if prefix == "/runpod-volume" else str(self.pod_dir / prefix.lstrip("/"))
        return _PATH.sub(to_local, text)

    def run(self, command: str, timeout_s: float = 600) -> Result:
        self.commands.append(command)
        if command.startswith(JOB_HEADER) and self.jobs is not None:
            name = command.splitlines()[0][len(JOB_HEADER):]
            self.started.append(name)
            handler = self.jobs[re.sub(r"-\d+-c\d+$", "", name)]
            rc = handler(self, name, command)
            if rc is not None:  # None: the job keeps running (a server)
                (self.pod_dir / "bench" / "jobs" / f"{name}.rc").write_text(f"{rc}\n")
            return Result(0, "", "")
        if command.startswith("ssh "):
            return Result(self.on_ssh(command) if self.on_ssh else 0, "", "")
        done = subprocess.run(["bash", "-c", self.map(command)], capture_output=True, text=True, timeout=timeout_s)
        return Result(done.returncode, done.stdout, done.stderr)

    def put(self, local: Path, remote: str) -> None:
        target = self.local(remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(local).is_dir():
            shutil.copytree(local, target)
        else:
            shutil.copy2(local, target)

    def get(self, remote: str, local: Path) -> None:
        source = self.local(remote)
        if source.is_dir():
            shutil.copytree(source, local)
        else:
            shutil.copy2(source, local)
