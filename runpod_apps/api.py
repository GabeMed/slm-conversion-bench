"""A small typed client of RunPod's REST API v2 (https://api.runpod.io/v2/openapi.json, read 2026-11-09):
pods, network volumes, the GPU catalog and secrets (names only). Standard library only.

Retries: a 429 waits for its Retry-After, a 5xx or a connection failure backs off exponentially, up to
`max_attempts`. Creating a pod is not idempotent, so a create that failed in transit is retried only after
listing the pods by name: one that was created after all is adopted, never created twice (a pod bills). The
API key is sent in the Authorization header and never appears in an error's text.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from runpod_apps.settings import RunPodError

BASE = "https://api.runpod.io"
RETRYABLE = {429, 500, 502, 503, 504}
AVAILABILITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}


class ApiError(RunPodError):
    def __init__(self, status: Optional[int], message: str):
        super().__init__(message)
        self.status = status


class RunPodClient:
    def __init__(self, api_key: str, base: str = BASE, max_attempts: int = 5, backoff_s: float = 2.0,
                 timeout_s: float = 60.0, sleep: Callable[[float], None] = time.sleep):
        self._key, self.base, self.max_attempts = api_key, base.rstrip("/"), max_attempts
        self.backoff_s, self.timeout_s, self._sleep = backoff_s, timeout_s, sleep

    # ---------------------------------------------------------------- transport

    def _clean(self, text: str) -> str:
        return text.replace(self._key, "<redacted>") if self._key else text

    def _once(self, method: str, path: str, query: Optional[Dict[str, Any]], body: Any) -> Any:
        url = self.base + path
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        data = None if body is None else json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {self._key}", "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            raw = response.read()
        return json.loads(raw) if raw else None

    def request(self, method: str, path: str, query: Optional[Dict[str, Any]] = None, body: Any = None,
                retry: bool = True, on_retry: Optional[Callable[[], Any]] = None) -> Any:
        """One API call; `on_retry` runs before a retry and may return the answer instead (the pod a failed
        create made after all)."""
        attempt = 0
        while True:
            attempt += 1
            wait, failure = None, None
            try:
                return self._once(method, path, query, body)
            except urllib.error.HTTPError as e:
                detail = _problem(e)
                failure = ApiError(e.code, self._clean(f"RunPod {method} {path}: HTTP {e.code} {detail}"))
                if e.code not in RETRYABLE or not retry:
                    raise failure from None
                if e.code == 429:
                    try:
                        wait = float(e.headers.get("Retry-After") or "")
                    except ValueError:
                        wait = None
            except (urllib.error.URLError, OSError) as e:  # no answer: connection refused, reset, timeout
                failure = ApiError(None, self._clean(f"RunPod {method} {path}: {type(e).__name__}: {e}"))
                if not retry:
                    raise failure from None
            if attempt >= self.max_attempts:
                raise failure
            self._sleep(wait if wait is not None else self.backoff_s * 2 ** (attempt - 1))
            if on_retry is not None:
                found = on_retry()
                if found is not None:
                    return found

    # ---------------------------------------------------------------- pods

    def create_pod(self, body: Dict[str, Any]) -> Dict[str, Any]:
        def created_after_all():
            pods = self.list_pods(name=body["name"])
            return pods[0] if pods else None
        return self.request("POST", "/v2/pods", body=body, on_retry=created_after_all)

    def get_pod(self, pod_id: str) -> Optional[Dict[str, Any]]:
        """The pod, or None once it no longer exists (terminated)."""
        try:
            return self.request("GET", f"/v2/pods/{pod_id}")
        except ApiError as e:
            if e.status == 404:
                return None
            raise

    def list_pods(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        pods, cursor = [], None
        while True:
            page = self.request("GET", "/v2/pods", query={"cursor": cursor}) or {}
            pods += page.get("pods") or []
            pagination = page.get("pagination") or {}
            cursor = pagination.get("nextCursor")
            if not pagination.get("hasNextPage") or not cursor:
                break
        return [p for p in pods if name is None or p.get("name") == name]

    def terminate_pod(self, pod_id: str) -> bool:
        """DELETE /v2/pods/{id}; True when it was terminated now, False when it was already gone."""
        try:
            self.request("DELETE", f"/v2/pods/{pod_id}")
            return True
        except ApiError as e:
            if e.status == 404:
                return False
            raise

    # ---------------------------------------------------------------- volumes, catalog, secrets

    def list_volumes(self) -> List[Dict[str, Any]]:
        return (self.request("GET", "/v2/network-volumes") or {}).get("networkVolumes") or []

    def get_volume(self, volume_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.request("GET", f"/v2/network-volumes/{volume_id}")
        except ApiError as e:
            if e.status == 404:
                return None
            raise

    def create_volume(self, name: str, size_gb: int, datacenter: str) -> Dict[str, Any]:
        def created_after_all():
            found = [v for v in self.list_volumes() if v.get("name") == name and v.get("dataCenter") == datacenter]
            return found[0] if found else None
        return self.request("POST", "/v2/network-volumes", body={"name": name, "size": size_gb, "dataCenter": datacenter},
                            on_retry=created_after_all)

    def gpu_availability(self, cloud: str, min_cuda_version: Optional[str]) -> List[Dict[str, Any]]:
        query = {"include": "AVAILABILITY", "product": "POD", "cloud": cloud, "minCudaVersion": min_cuda_version}
        return (self.request("GET", "/v2/catalog/gpus", query=query) or {}).get("gpus") or []

    def secret_names(self) -> List[str]:
        return [s["name"] for s in (self.request("GET", "/v2/account/secrets") or {}).get("secrets") or []]


def _problem(error: urllib.error.HTTPError) -> str:
    """RunPod's problem+json body as one line (title, detail, validation errors)."""
    try:
        problem = json.loads(error.read() or b"{}")
    except (ValueError, OSError):
        return ""
    if not isinstance(problem, dict):
        return ""
    parts = [problem.get("title"), problem.get("detail"), "; ".join(problem.get("errors") or [])]
    return ": ".join(str(p) for p in parts if p)


def best_datacenters(gpus: List[Dict[str, Any]], gpu: str) -> List[Dict[str, str]]:
    """The data centers offering `gpu` (RunPod's id) now, most available first (ties by id, so the choice is
    reproducible); none when it is unavailable everywhere."""
    for entry in gpus:
        if entry.get("id") == gpu:
            found = [d for d in entry.get("dataCenters") or [] if AVAILABILITY_RANK.get(d.get("availability"), 0) > 0]
            return sorted(found, key=lambda d: (-AVAILABILITY_RANK[d["availability"]], d["id"]))
    return []
