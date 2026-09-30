"""Judgments that configure later executions become facts when used (design §1.2).

A fact is written once, at `judgments/<J>/<sha256>/<name>.json`, where the sha256 is that of
the file's canonical JSON. The manifest of every execution that uses it records the sha, so
recomputing a judgment produces a new fact and never rewrites the past.

The four facts that cross fronts, and their shapes:

| Fact | Written by | Read by | Shape |
|---|---|---|---|
| `centroids` | J5 (S3) | `assign` (C4) | `{"embedding": {"model": str, "revision": str}, "clusters": {"<cluster>": [float, ...]}}`, unit vectors in the space of `clusters.embed(prompt_text(...))` |
| `choice` | J6 (S4) | B3, B4 | `{"slm": "<roles.slm_candidates[].name>"}` |
| `adapters` | S5 (training) | B4, B5 | `{"slm": "<candidate name>", "centroids": "<sha>", "adapters": {"<cluster>": {"served_name": str, "sha256": str}}}` |
| `allocation` | J7 (S6) | B5 | `{"centroids": "<sha>", "adapters": "<sha>", "allocation": {"<cluster>": "production_llm" \\| "cheap_alt" \\| "slm"}}` |

`configure` points each arm at the facts it uses (`arms` in config.yaml).
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

from bench import paths

FACTS = ("centroids", "choice", "adapters", "allocation")
ENGINES_IN_ALLOCATION = ("production_llm", "cheap_alt", "slm")


class FactError(ValueError):
    pass


def canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write_fact(judgment: str, name: str, payload: Dict[str, Any], root: Path = paths.ROOT) -> Path:
    if name not in FACTS:
        raise FactError(f"unknown fact {name!r}")
    validate_fact(name, payload)
    sha = hashlib.sha256(canonical(payload)).hexdigest()
    path = root / "judgments" / judgment / sha / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(canonical(payload))
    return path


@lru_cache(maxsize=None)
def read_fact(path: str) -> Tuple[Dict[str, Any], str]:
    """The payload and its sha256; the sha must be the directory the file lives in."""
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if Path(path).parent.name != sha:
        raise FactError(f"{path}: content sha256 {sha} is not its directory name")
    payload = json.loads(raw)
    validate_fact(Path(path).stem, payload)
    return payload, sha


def validate_fact(name: str, payload: Dict[str, Any]) -> None:
    if name == "centroids":
        embedding, clusters = payload.get("embedding") or {}, payload.get("clusters") or {}
        if not isinstance(embedding.get("model"), str) or not isinstance(embedding.get("revision"), str):
            raise FactError("centroids.embedding needs model and revision")
        if not clusters or len({len(v) for v in clusters.values()}) != 1:
            raise FactError("centroids.clusters must be non-empty vectors of one dimension")
    elif name == "choice":
        if not isinstance(payload.get("slm"), str):
            raise FactError("choice.slm must be a candidate name")
    elif name == "adapters":
        if not isinstance(payload.get("slm"), str) or not isinstance(payload.get("centroids"), str):
            raise FactError("adapters needs slm and centroids")
        for cluster, adapter in (payload.get("adapters") or {}).items():
            if not isinstance(adapter.get("served_name"), str) or not isinstance(adapter.get("sha256"), str):
                raise FactError(f"adapters.{cluster} needs served_name and sha256")
    elif name == "allocation":
        if not isinstance(payload.get("centroids"), str) or not isinstance(payload.get("adapters"), str):
            raise FactError("allocation needs the centroids and adapters it was decided on")
        bad = {c: e for c, e in (payload.get("allocation") or {}).items() if e not in ENGINES_IN_ALLOCATION}
        if bad:
            raise FactError(f"allocation engines must be in {ENGINES_IN_ALLOCATION}: {bad}")
    else:
        raise FactError(f"unknown fact {name!r}")
