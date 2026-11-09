"""The local record of what this machine created on RunPod (`.runpod/state.json`, git-ignored): the network
volume, every pod still up (recorded the moment the API returns it, before anything waits on it) and each
serving pod's local SSH tunnel. It is what `bench serve --on runpod down --all` terminates; `down --all`
also terminates every pod the API lists under `runpod.name_prefix`, so a pod this file missed is not left
billing. Written whole and atomically (a temporary file of its own renamed over it), each change under an
exclusive lock."""
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from bench import paths


def state_dir() -> Path:
    return paths.ROOT / ".runpod"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class State:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or state_dir() / "state.json"

    def read(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return {"volume": None, "pods": {}, "terminated": [], "downloaded": {}}
        data = json.loads(self.path.read_text())
        for key, empty in (("volume", None), ("pods", {}), ("terminated", []), ("downloaded", {})):
            data.setdefault(key, empty)
        return data

    def write(self, data: Dict[str, Any]) -> None:
        """Whole, through a temporary file of its own renamed over the state."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @contextmanager
    def locked(self) -> Iterator[None]:
        """An exclusive lock (flock on a file beside the state) across one read-modify-write: two commands of
        this machine (an `up` and a `down`) never lose each other's record."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def update(self, change) -> Dict[str, Any]:
        with self.locked():
            data = self.read()
            change(data)
            self.write(data)
        return data

    # ---------------------------------------------------------------- pods

    def add_pod(self, pod: Dict[str, Any], role: str, candidate: Optional[str] = None) -> None:
        record = {"id": pod["id"], "name": pod.get("name"), "role": role, "candidate": candidate,
                  "created_at": now(), "cost_per_hour": pod.get("cost"), "datacenter": pod.get("dataCenterId")}
        self.update(lambda d: d["pods"].__setitem__(pod["id"], record))

    def set_pod(self, pod_id: str, **fields: Any) -> None:
        def change(data):
            if pod_id in data["pods"]:
                data["pods"][pod_id].update(fields)
        self.update(change)

    def remove_pod(self, pod_id: str) -> None:
        def change(data):
            record = data["pods"].pop(pod_id, None)
            if record is not None:
                data["terminated"].append({**record, "terminated_at": now()})
                data["terminated"] = data["terminated"][-50:]
        self.update(change)

    def serving(self, candidate: str) -> Optional[Dict[str, Any]]:
        for record in self.read()["pods"].values():
            if record["role"] == "serve" and record["candidate"] == candidate:
                return record
        return None
