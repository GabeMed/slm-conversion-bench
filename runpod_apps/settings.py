"""config.yaml's `runpod` section, checked once: only runpod_apps/ and `bench preflight` (the training backend)
read it, never a judgment. configs/runpod.patch.yaml documents and adds it."""
import os
from pathlib import Path
from typing import Any, Dict

VOLUME = "/runpod-volume"   # where every pod mounts the network volume
WORK = "/bench"             # each pod's own scratch: the code, the jobs' logs and markers (container disk)
KEYS = ("api_key_env", "cloud", "datacenter", "gpu_ids", "min_cuda_version", "volume", "images", "cpu_flavor",
        "disk_gb", "secrets", "ssh_key", "local_ports", "name_prefix", "max_pod_hours", "uv_version",
        "train_backend")


class RunPodError(RuntimeError):
    """A RunPod call, a pod or a job that failed; its text never carries a credential."""


def runpod_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    settings = config.get("runpod")
    if not isinstance(settings, dict):
        raise RunPodError("config.yaml has no `runpod` section: apply configs/runpod.patch.yaml "
                          "(python scripts/apply_config_patch.py configs/runpod.patch.yaml --write)")
    missing = [k for k in KEYS if k not in settings]
    if missing:
        raise RunPodError(f"runpod section lacks {missing} (configs/runpod.patch.yaml has every key)")
    if settings["cloud"] not in ("SECURE", "COMMUNITY"):
        raise RunPodError("runpod.cloud must be SECURE or COMMUNITY")
    for key in ("name", "size_gb"):
        if not settings["volume"].get(key):
            raise RunPodError(f"runpod.volume.{key} is not set")
    for key in ("serve", "gpu_job", "cpu_job"):
        if not settings["images"].get(key):
            raise RunPodError(f"runpod.images.{key} is not set")
    if not isinstance(settings["local_ports"], dict):
        raise RunPodError("runpod.local_ports must map each candidate to the local port of its tunnel")
    return settings


def gpu_id(settings: Dict[str, Any], gpu: str) -> str:
    """RunPod's GPU type id of a GPU the configuration names (serving.gpu, train.gpu)."""
    if gpu not in settings["gpu_ids"]:
        raise RunPodError(f"runpod.gpu_ids has no RunPod id for {gpu!r}")
    return settings["gpu_ids"][gpu]


def api_key(settings: Dict[str, Any], environ=None) -> str:
    environ = os.environ if environ is None else environ
    key = environ.get(settings["api_key_env"])
    if not key:
        raise RunPodError(f"environment variable {settings['api_key_env']} is not set (a RunPod API key)")
    return key


def public_key(settings: Dict[str, Any]) -> str:
    """The operator's SSH public key, which every pod authorizes (sent as PUBLIC_KEY: not a secret)."""
    path = Path(os.path.expanduser(settings["ssh_key"]) + ".pub")
    if not path.is_file():
        raise RunPodError(f"no SSH public key at {path} (runpod.ssh_key names the private key; ssh-keygen -t ed25519)")
    return path.read_text().strip()


def private_key(settings: Dict[str, Any]) -> str:
    return os.path.expanduser(settings["ssh_key"])


def local_port(settings: Dict[str, Any], candidate: str) -> int:
    port = settings["local_ports"].get(candidate)
    if not isinstance(port, int):
        raise RunPodError(f"runpod.local_ports has no port for {candidate}")
    return port


def secret_ref(name: str) -> str:
    """How a pod's environment names an account secret: RunPod substitutes the value at boot."""
    return "{{ RUNPOD_SECRET_" + name + " }}"
