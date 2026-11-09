"""Pods: what each one is created with, the script it boots, and the steps every use of a pod shares (wait
until SSH answers, upload the code, put adapters on the volume, terminate).

Every pod boots the same script (`BOOTSTRAP`): the network volume where every tool looks for it
(/root/.cache/huggingface, /root/.cache/vllm and /adapters are links into /runpod-volume, the paths the Modal
apps mount, so `modal_apps.common` runs unchanged), sshd authorizing the operator's public key only, the
secrets RunPod substituted saved for jobs started over SSH (/bench/env, mode 600), and a deadline past which
the pod terminates itself: the backstop of `bench serve --on runpod down`, never the plan. Then it idles; the
local side drives it over SSH. No custom image is needed: an image without sshd or curl gets them from apt.

The pods run the code of this checkout, uploaded at start (bench/, modal_apps/, runpod_apps/ and the locks).
"""
import gzip
import hashlib
import io
import shlex
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from bench import paths
from runpod_apps import settings as rs
from runpod_apps.settings import VOLUME, WORK, RunPodError
from runpod_apps.ssh import Remote, Shell, SshShell
from runpod_apps.state import State, state_dir

REPO = Path(__file__).resolve().parent.parent  # the code the pods run: this checkout's, wherever paths.ROOT points
CODE = ("bench", "modal_apps", "runpod_apps")
LOCKS = ("env/train/requirements.lock", "env/train/serve-vllm.lock")
POD_PORT = 8000  # modal_apps.common.PORT: vLLM inside the serving pod, reached only through SSH
SSH_PORT = "22/tcp"

BOOTSTRAP = r"""set -u
VOL=/runpod-volume
W=/bench
log() { echo "[bench] $*"; }
mkdir -p "$W/jobs"
# the volume where every tool looks for it: the paths the Modal apps mount
mkdir -p "$VOL/hf" "$VOL/vllm-cache" "$VOL/adapters" /root/.cache
rm -rf /root/.cache/huggingface /root/.cache/vllm /adapters
ln -s "$VOL/hf" /root/.cache/huggingface
ln -s "$VOL/vllm-cache" /root/.cache/vllm
ln -s "$VOL/adapters" /adapters
# sshd, from apt where the image lacks it (and ssh, curl)
need=""
[ -x /usr/sbin/sshd ] || need="openssh-server"
command -v ssh >/dev/null 2>&1 || need="$need openssh-client"
command -v curl >/dev/null 2>&1 || need="$need curl ca-certificates"
if [ -n "$need" ]; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $need >/dev/null \
    || log "apt-get install $need failed"
fi
# past its deadline the pod terminates itself (RunPod gives every pod RUNPOD_POD_ID and a pod-scoped RUNPOD_API_KEY)
if [ -n "${BENCH_DEADLINE_S:-}" ]; then
  ( sleep "$BENCH_DEADLINE_S"
    log "deadline of $BENCH_DEADLINE_S s reached: terminating this pod"
    command -v runpodctl >/dev/null 2>&1 && runpodctl remove pod "${RUNPOD_POD_ID:-}"
    curl -fsS -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
      "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID:-}" >/dev/null 2>&1
    touch "$W/stop" ) &
fi
# the secrets RunPod substituted, for the jobs started over SSH (an SSH session does not inherit them)
names=$(env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' | grep -E '^(VLLM_API_KEY|HF_TOKEN|RUNPOD_POD_ID|BENCH_[A-Z_]+|HF_[A-Z_]+)$')
( umask 077; : > "$W/env"; [ -z "$names" ] || declare -px $names > "$W/env" 2>/dev/null )
mkdir -p /root/.ssh /run/sshd && chmod 700 /root/.ssh
printf '%s\n' "${PUBLIC_KEY:-}" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A >/dev/null 2>&1
/usr/sbin/sshd -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password -o AllowTcpForwarding=yes \
  && log "sshd ready" || log "sshd did not start"
while [ ! -f "$W/stop" ]; do sleep 5; done
"""

# The training environment (env/train/requirements.lock, Python 3.12), for P-4's reference and the AIPerf
# client, as the Modal training image has it: a custom image's /opt/bench-train when it was built from this
# lock (env/runpod/Dockerfile.train), otherwise a venv on the volume keyed by the lock's sha256, built once
# by uv (Python managed by uv too, so the venv runs on any image). Leaves its interpreter in $PY.
TRAIN_ENV = r"""set -e
LOCK="$PWD/env/train/requirements.lock"
SHA=$(sha256sum "$LOCK" | cut -c1-16)
if [ -x /opt/bench-train/bin/python ] && [ "$(cut -c1-16 /opt/bench-train/LOCK_SHA256 2>/dev/null)" = "$SHA" ]; then
  PY=/opt/bench-train/bin/python
else
  T=/runpod-volume/tools
  UV="$T/uv-$BENCH_UV_VERSION/uv"
  if [ ! -x "$UV" ]; then
    mkdir -p "$T/uv-$BENCH_UV_VERSION"
    curl -LsSf "https://astral.sh/uv/$BENCH_UV_VERSION/install.sh" | env UV_INSTALL_DIR="$T/uv-$BENCH_UV_VERSION" UV_NO_MODIFY_PATH=1 sh
  fi
  V="/runpod-volume/venvs/train-$SHA"
  L="$V.lock"
  STALE="${BENCH_LOCK_STALE_S:-1200}"
  mkdir -p /runpod-volume/venvs
  # one builder at a time: the lock holds its pod and a heartbeat; a lock whose heartbeat is older than STALE
  # is a builder that died (the local side also removes one whose pod no longer runs)
  until mkdir "$L" 2>/dev/null; do
    beat=""; owner="?"
    [ -f "$L/owner" ] && read -r owner beat < "$L/owner" || true
    [ -n "$beat" ] || beat=$(stat -c %Y "$L" 2>/dev/null || date +%s)
    age=$(( $(date +%s) - beat ))
    if [ "$age" -gt "$STALE" ]; then echo "breaking $L: no heartbeat from $owner for $age s"; rm -rf "$L"; continue; fi
    sleep 5
  done
  echo "${RUNPOD_POD_ID:-unknown} $(date +%s)" > "$L/owner"
  ( while [ -d "$L" ]; do echo "${RUNPOD_POD_ID:-unknown} $(date +%s)" > "$L/owner.tmp" 2>/dev/null \
      && mv "$L/owner.tmp" "$L/owner" 2>/dev/null; sleep 60; done ) &
  HEARTBEAT=$!
  trap 'kill $HEARTBEAT 2>/dev/null; rm -rf "$L"' EXIT
  if [ ! -f "$V/.complete" ]; then
    rm -rf "$V"
    UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_INSTALL_DIR="$T/python" "$UV" venv --python 3.12 "$V"
    UV_CACHE_DIR="$T/uv-cache" UV_LINK_MODE=copy "$UV" pip install --python "$V/bin/python" -r "$LOCK"
    touch "$V/.complete"
  fi
  kill $HEARTBEAT 2>/dev/null; rm -rf "$L"; trap - EXIT
  PY="$V/bin/python"
fi
export PATH="$(dirname "$PY"):$PATH"
set +e
"""


def pod_name(settings: Dict[str, Any], role: str, candidate: Optional[str] = None, token: str = "") -> str:
    """<name_prefix>-<role>[-<candidate>]-<token>: the token, unique to one create, is how a create whose
    answer was lost recognises its pod (api.RunPodClient.create_pod)."""
    return "-".join(p for p in (settings["name_prefix"], role, candidate, token) if p)


def role_prefix(settings: Dict[str, Any], role: str, candidate: Optional[str] = None) -> str:
    return pod_name(settings, role, candidate) + "-"


def pod_body(settings: Dict[str, Any], volume: Dict[str, Any], role: str, image: str, deadline_s: float,
             public_key: str, candidate: Optional[str] = None, gpu: Optional[str] = None,
             cpu_vcpus: Optional[int] = None, env: Optional[Dict[str, str]] = None,
             token: Optional[str] = None) -> Dict[str, Any]:
    """POST /v2/pods: one GPU (`gpu`, RunPod's id) or a CPU pod (`cpu_vcpus`), in the volume's data center with
    the volume mounted, publishing SSH only, booting BOOTSTRAP. Named and tagged (env BENCH_POD_TOKEN, BENCH_ROLE,
    BENCH_CANDIDATE) so the pod is recognised from the API's listing alone."""
    import secrets

    if (gpu is None) == (cpu_vcpus is None):
        raise RunPodError("a pod has a GPU or a CPU count, not both")
    token = token or secrets.token_hex(4)
    body: Dict[str, Any] = {
        "name": pod_name(settings, role, candidate, token), "image": image, "cloud": settings["cloud"],
        "dataCenterIds": [volume["datacenter"]], "disk": settings["disk_gb"], "ports": [SSH_PORT],
        "env": {"PUBLIC_KEY": public_key, "BENCH_DEADLINE_S": str(int(deadline_s)), "BENCH_IMAGE": image,
                "BENCH_POD_TOKEN": token, "BENCH_ROLE": role, "BENCH_CANDIDATE": candidate or "",
                "BENCH_LOCK_STALE_S": str(int(float(settings.get("lock_stale_minutes", 20)) * 60)),
                "BENCH_UV_VERSION": str(settings["uv_version"]), "HF_XET_HIGH_PERFORMANCE": "1",
                "HF_HOME": "/root/.cache/huggingface", **(env or {})},  # the volume's link, whatever the image sets
        "mounts": {"network": [{"volumeId": volume["id"], "path": VOLUME}]},
        "entrypoint": ["bash", "-c"], "cmd": [BOOTSTRAP],
    }
    if gpu is not None:
        body["gpu"] = {"id": gpu, "count": 1, "minCudaVersion": str(settings["min_cuda_version"])}
    else:
        body["cpu"] = {"id": settings["cpu_flavor"], "vcpuCount": cpu_vcpus}
    return body


def code_archive() -> bytes:
    """This checkout's code and locks, as the pods run them (no tests, data or caches)."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        files = [p for d in CODE for p in sorted((REPO / d).rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".yaml", ".sh")]
        files += [REPO / lock for lock in LOCKS]
        for path in files:
            info = tar.gettarinfo(str(path), arcname=path.relative_to(REPO).as_posix())
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(path, "rb") as f:
                tar.addfile(info, f)
    return buffer.getvalue()


def ssh_endpoint(pod: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The pod's direct SSH (public IP and the port mapped to 22), once it has one."""
    direct = (pod.get("ssh") or {}).get("direct")
    if direct and direct.get("host") and direct.get("port"):
        return {"host": direct["host"], "port": int(direct["port"])}
    for port in ((pod.get("runtime") or {}).get("ports") or []):
        if port.get("private") == 22 and port.get("public") and port.get("ip"):
            return {"host": port["ip"], "port": int(port["public"])}
    return None


class Provider:
    """RunPod for this configuration: the API, the local state, and a Shell per pod (SSH unless a test gives
    another)."""

    def __init__(self, config: Dict[str, Any], api=None, state: Optional[State] = None,
                 shell_for: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Shell]] = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 poll_s: float = 10.0):
        from runpod_apps.api import RunPodClient

        self.config, self.settings = config, rs.runpod_settings(config)
        self.api = api or RunPodClient(rs.api_key(self.settings))
        self.state = state or State()
        self.shell_for = shell_for or self._ssh
        self.sleep, self.clock, self.poll_s = sleep, clock, poll_s

    def _ssh(self, pod: Dict[str, Any], endpoint: Dict[str, Any]) -> Shell:
        return SshShell(endpoint["host"], endpoint["port"], rs.private_key(self.settings),
                        state_dir() / "known_hosts" / pod["id"])

    def remote(self, pod: Dict[str, Any]) -> Remote:
        endpoint = ssh_endpoint(pod)
        if endpoint is None:
            raise RunPodError(f"pod {pod['id']} has no SSH endpoint (status {pod.get('status')})")
        return Remote(self.shell_for(pod, endpoint), self.sleep, self.clock)

    def volume(self) -> Dict[str, Any]:
        volume = self.state.read()["volume"]
        if not volume:
            raise RunPodError("no network volume yet: bench serve --on runpod setup")
        return volume

    # ---------------------------------------------------------------- one pod's life

    def create(self, role: str, body: Dict[str, Any], candidate: Optional[str] = None) -> Dict[str, Any]:
        pod = self.api.create_pod(body)
        self.state.add_pod(pod, role, candidate)  # before anything waits on it: `down --all` can always find it
        return pod

    def wait_ssh(self, pod_id: str, timeout_s: float) -> Dict[str, Any]:
        """Until the pod runs, publishes SSH and its sshd answers (an image may first install it); the pod
        and its Remote."""
        started = self.clock()
        last = "not seen yet"
        while True:
            pod = self.api.get_pod(pod_id)
            if pod is None or pod.get("status") in ("ERROR", "EXITED", "TERMINATED"):
                raise RunPodError(f"pod {pod_id} is {pod.get('status') if pod else 'gone'} before SSH answered")
            last = pod.get("status")
            if pod.get("status") == "RUNNING" and ssh_endpoint(pod):
                remote = self.remote(pod)
                answer = remote.shell.run(f"test -f {WORK}/env && echo ready", timeout_s=60)
                if answer.returncode == 0 and "ready" in answer.stdout:
                    self.state.set_pod(pod_id, ssh=ssh_endpoint(pod), datacenter=pod.get("dataCenterId"),
                                       cost_per_hour=pod.get("cost"))
                    return {"pod": pod, "remote": remote}
                last = f"RUNNING, sshd not answering ({answer.returncode})"
            if self.clock() - started > timeout_s:
                raise RunPodError(f"pod {pod_id} not reachable over SSH after {timeout_s:.0f} s ({last})")
            self.sleep(self.poll_s)

    def terminate(self, pod_id: str) -> bool:
        gone = self.api.terminate_pod(pod_id)
        self.state.remove_pod(pod_id)
        return gone

    def upload_code(self, remote: Remote) -> str:
        """This checkout's code into /bench/code; the archive's sha256."""
        import tempfile

        archive = code_archive()
        with tempfile.TemporaryDirectory() as work:
            local = Path(work) / "code.tar.gz"
            local.write_bytes(archive)
            remote.check(f"mkdir -p {WORK}", "mkdir")
            remote.shell.put(local, f"{WORK}/code.tar.gz")
        remote.check(f"rm -rf {WORK}/code && mkdir -p {WORK}/code && tar -xzf {WORK}/code.tar.gz -C {WORK}/code",
                     "unpacking the code")
        return hashlib.sha256(archive).hexdigest()

    def ensure_adapters(self, remote: Remote, adapters: List[Dict[str, Any]]) -> List[str]:
        """Each adapter (train/adapters/<cluster>/adapter) at /runpod-volume/adapters/<sha256>, uploaded when
        missing, through a temporary name so an interrupted upload is never taken for an adapter (its bytes are
        checked again where it is used). The sha256s uploaded now."""
        uploaded = []
        for adapter in adapters:
            target = f"{VOLUME}/adapters/{adapter['sha256']}"
            if remote.exists(f"{target}/adapter_config.json"):
                continue
            local = paths.ROOT / "train" / "adapters" / adapter["cluster"] / "adapter"
            if not local.is_dir():
                raise RunPodError(f"train/adapters/{adapter['cluster']}/adapter is missing: nothing to upload")
            partial = f"{VOLUME}/adapters/.upload-{adapter['sha256']}"
            remote.check(f"rm -rf {partial} && mkdir -p {VOLUME}/adapters", "preparing the adapter upload")
            remote.shell.put(local, partial)
            remote.check(f"rm -rf {target} && mv {partial} {target}", "installing the adapter")
            uploaded.append(adapter["sha256"])
        return uploaded

    def clear_stale_train_lock(self, remote: Remote) -> Optional[str]:
        """The training environment's build lock on the volume, removed when the pod that holds it no longer
        runs (a builder that died with its pod); the pod id it named, if removed."""
        sha = hashlib.sha256((REPO / LOCKS[0]).read_bytes()).hexdigest()[:16]
        lock = f"{VOLUME}/venvs/train-{sha}.lock"
        owner = (remote.read(f"{lock}/owner") or "").split()
        if not owner or owner[0] == "unknown":
            return None
        pod = self.api.get_pod(owner[0])
        if pod is not None and pod.get("status") not in ("EXITED", "TERMINATED", "ERROR"):
            return None
        remote.shell.run(f"rm -rf {lock}")
        return owner[0]

    @contextmanager
    def job_pod(self, role: str, image: str, gpu: Optional[str] = None, cpu_vcpus: Optional[int] = None,
                env: Optional[Dict[str, str]] = None, candidate: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """A pod for one job: created, reached, given the code; terminated whatever happens (and past
        runpod.max_pod_hours.job by itself)."""
        body = pod_body(self.settings, self.volume(), role, image, deadline(self.settings, "job"),
                        rs.public_key(self.settings), candidate=candidate,
                        gpu=gpu, cpu_vcpus=cpu_vcpus, env=env)
        pod = self.create(role, body, candidate)
        try:
            ready = self.wait_ssh(pod["id"], self.config["serving"]["startup_timeout_s"])
            ready["code_sha256"] = self.upload_code(ready["remote"])
            yield ready
        finally:
            self.terminate(pod["id"])


def make_provider(config: Dict[str, Any]) -> Provider:
    """The Provider every command uses (the tests replace it with one on a fake API and fake shells)."""
    return Provider(config)


def deadline(settings: Dict[str, Any], kind: str, hours: Optional[float] = None) -> float:
    return float(hours if hours is not None else settings["max_pod_hours"][kind]) * 3600


def quote(value: str) -> str:
    return shlex.quote(value)
