"""A small typed client of RunPod's REST API v2 (https://api.runpod.io/v2/openapi.json, read 2026-11-09):
pods, network volumes, the GPU and data center catalogs and secrets (names only). Standard library only.

Retries: a 429 waits for its Retry-After; a 5xx, a connection failure or an answer that is not the JSON
asked for (cut short, an HTML page) backs off exponentially, up to `max_attempts`. Creating a pod is not
idempotent, so a create that failed in transit is retried only after looking for the pod it may have made:
by its name, which carries a token unique to that create, and the same token in its environment
(BENCH_POD_TOKEN), never a pod that already exited. The same look is made once more after the last attempt;
when it cannot be made, the error says a pod may exist and how to find it. The API key is sent in the
Authorization header and never appears in an error's text.
"""
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from runpod_apps.settings import RunPodError

BASE = "https://api.runpod.io"
RETRYABLE = {429, 500, 502, 503, 504}
MAX_PAGES = 100
DEAD = ("EXITED", "TERMINATED", "ERROR")
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
        """One API call; `on_retry` runs before each retry and once after the last failed attempt, and may
        return the answer instead (the pod a failed create made after all)."""
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
            except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as e:
                # no usable answer: refused, reset, timed out, cut short (IncompleteRead), not JSON
                failure = ApiError(None, self._clean(f"RunPod {method} {path}: no usable answer: "
                                                     f"{type(e).__name__}: {str(e)[:200]}"))
                if not retry:
                    raise failure from None
            if attempt >= self.max_attempts:
                if on_retry is not None:
                    found = on_retry()
                    if found is not None:
                        return found
                raise failure
            self._sleep(wait if wait is not None else self.backoff_s * 2 ** (attempt - 1))
            if on_retry is not None:
                found = on_retry()
                if found is not None:
                    return found

    # ---------------------------------------------------------------- pods

    def create_pod(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /v2/pods. `body` names the pod uniquely and carries the same token as env BENCH_POD_TOKEN
        (pods.pod_body), which is how a create whose answer was lost is recognised."""
        name, token = body["name"], (body.get("env") or {}).get("BENCH_POD_TOKEN")
        if not token or not name.endswith(token):
            raise RunPodError(f"pod {name}: no unique token in its name and BENCH_POD_TOKEN: it could not be "
                              "recognised if its answer were lost")

        def created_after_all():
            found = [p for p in self.list_pods(name=name) if (p.get("env") or {}).get("BENCH_POD_TOKEN") == token
                     and p.get("status") not in DEAD]
            return found[0] if found else None

        def look():
            try:
                return created_after_all()
            except ApiError as e:
                raise ApiError(e.status, f"creating pod {name} failed and the pods could not be listed to see whether "
                                         f"it was made ({e}): a pod named {name} may exist: run `bench serve --on "
                                         f"runpod down --all`") from None
        try:
            return self.request("POST", "/v2/pods", body=body, on_retry=look)
        except ApiError as e:
            if e.status is None or e.status in RETRYABLE:
                raise ApiError(e.status, f"{e}; a pod named {name} may exist: run `bench serve --on runpod down "
                                         "--all`") from None
            raise

    def get_pod(self, pod_id: str) -> Optional[Dict[str, Any]]:
        """The pod, or None once it no longer exists (terminated)."""
        try:
            return self.request("GET", f"/v2/pods/{pod_id}")
        except ApiError as e:
            if e.status == 404:
                return None
            raise

    def list_pods(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every page, at most MAX_PAGES, stopping where the cursor does not advance."""
        pods, cursor, seen = [], None, set()
        for _ in range(MAX_PAGES):
            page = self.request("GET", "/v2/pods", query={"cursor": cursor}) or {}
            pods += page.get("pods") or []
            pagination = page.get("pagination") or {}
            cursor = pagination.get("nextCursor")
            if not pagination.get("hasNextPage") or not cursor or cursor in seen:
                break
            seen.add(cursor)
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

    def volume_datacenters(self) -> List[str]:
        """The data centers that can hold a network volume: those listing a volume tier (networkVolumeTypes; empty
        means none, and creating a volume there is a 400)."""
        found = (self.request("GET", "/v2/catalog/datacenters") or {}).get("dataCenters") or []
        return sorted(d["id"] for d in found if d.get("networkVolumeTypes"))

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
