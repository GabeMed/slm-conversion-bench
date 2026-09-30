"""Judgments that configure later executions become facts when used (design §1.2).

A fact is written once, at `judgments/<J>/<sha256>/<name>.json`, where the sha256 is that of the
file's canonical JSON. Each arm points at the facts it uses in `config.yaml › arms`, and the
manifest of every execution records their sha, so recomputing a judgment produces a new fact and
never rewrites the past. The router (`bench/contracts/router.py`) checks that an arm's facts were
decided on each other.

The four facts that cross fronts, and their shapes:

**`centroids`** (J5, S3) → `assign` in B4 and B5:
    {"embedding": {"model": <hf id>, "revision": <40-hex commit>, "max_seq_length": <int>,
                   "truncation": "head" | "tail", "text": "prompt", "trust_remote_code": <bool>},
     "clusters": {"<cluster>": [float, ...]}}
Vectors are unit-norm and live in the **prompt-only** space that `clusters.embed` produces with that
`embedding` block, whatever features formed the clusters (the SPEC forms clusters on prompt and
action, and assigns on the prompt only). `truncation` says which end of an over-long prompt is
kept (`head` keeps the start, `tail` the end). `trust_remote_code` (default false) runs the model
repository's own code, pinned by the revision; many long-context embedding models need it.

**`choice`** (J6, S4) → B3, B4, B5:  `{"slm": "<roles.slm_candidates[].name>"}`

**`adapters`** (S5) → B4, B5:
    {"slm": <candidate name>, "choice": <sha of the choice fact>, "centroids": <sha of the centroids fact>,
     "adapters": {"<cluster>": {"served_name": <non-empty, unique, not the base's name>, "sha256": <sha256_dir of the adapter>}}}
One adapter per centroid cluster; each served by vLLM under `served_name`.

**`allocation`** (J7, S6) → B5:
    {"centroids": <sha>, "adapters": <sha>, "allocation": {"<cluster>": "production_llm" | "cheap_alt" | "slm"}}
A cluster left out stays with `production_llm` (SPEC S6).
"""
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from bench import paths

FACTS = ("centroids", "choice", "adapters", "allocation")
ENGINES_IN_ALLOCATION = ("production_llm", "cheap_alt", "slm")
TRUNCATION = ("head", "tail")
EMBEDDED_TEXT = ("prompt",)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FactError(ValueError):
    pass


def canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_dir(path: Path) -> str:
    """The identity of a directory (an adapter): sha256 over its sorted (relative path, file sha256) pairs."""
    files = sorted(p for p in Path(path).rglob("*") if p.is_file())
    if not files:
        raise FactError(f"{path} has no files: not an adapter")
    digest = hashlib.sha256()
    for file in files:
        digest.update(f"{file.relative_to(path).as_posix()}\0{hashlib.sha256(file.read_bytes()).hexdigest()}\n".encode())
    return digest.hexdigest()


def write_fact(judgment: str, name: str, payload: Dict[str, Any], root: Optional[Path] = None) -> Path:
    validate_fact(name, payload)
    sha = hashlib.sha256(canonical(payload)).hexdigest()
    path = (root or paths.ROOT) / "judgments" / judgment / sha / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(canonical(payload))
    return path


def read_fact(path: str, name: str) -> Tuple[Dict[str, Any], str]:
    """The payload and its sha256. The file must be the fact `name`, and its content sha256 must be
    the directory it lives in. Checked on every read: the cache below is keyed by the file's
    modification time and size, so a file changed on disk is read and verified again."""
    stat = Path(path).stat()
    return _read_fact(path, name, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=None)
def _read_fact(path: str, name: str, mtime_ns: int, size: int) -> Tuple[Dict[str, Any], str]:
    if Path(path).name != f"{name}.json":
        raise FactError(f"{path} is not a {name} fact")
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if Path(path).parent.name != sha:
        raise FactError(f"{path}: content sha256 {sha} is not its directory name")
    payload = json.loads(raw)
    validate_fact(name, payload)
    return payload, sha


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FactError(message)


def validate_fact(name: str, payload: Dict[str, Any]) -> None:
    if name == "centroids":
        embedding, clusters = payload.get("embedding") or {}, payload.get("clusters") or {}
        _require(isinstance(embedding.get("model"), str) and embedding["model"], "centroids.embedding.model is required")
        _require(isinstance(embedding.get("revision"), str) and bool(_COMMIT.match(embedding["revision"])),
                 "centroids.embedding.revision must be a 40-hex commit")
        _require(isinstance(embedding.get("max_seq_length"), int) and embedding["max_seq_length"] > 0,
                 "centroids.embedding.max_seq_length must be a positive integer")
        _require(embedding.get("truncation") in TRUNCATION, f"centroids.embedding.truncation must be in {TRUNCATION}")
        _require(embedding.get("text") in EMBEDDED_TEXT, f"centroids.embedding.text must be in {EMBEDDED_TEXT}")
        _require(isinstance(embedding.get("trust_remote_code", False), bool), "centroids.embedding.trust_remote_code must be a boolean")
        _require(bool(clusters), "centroids.clusters must not be empty")
        _require(len({len(v) for v in clusters.values()}) == 1, "centroids.clusters must share one dimension")
        for cluster, vector in clusters.items():
            _require(abs(math.sqrt(sum(x * x for x in vector)) - 1.0) < 1e-3, f"centroid {cluster} is not unit-norm")
    elif name == "choice":
        _require(isinstance(payload.get("slm"), str) and payload["slm"], "choice.slm must be a candidate name")
    elif name == "adapters":
        for key in ("slm", "choice", "centroids"):
            _require(isinstance(payload.get(key), str) and payload[key], f"adapters.{key} is required")
        adapters = payload.get("adapters") or {}
        _require(bool(adapters), "adapters.adapters must not be empty")
        served = [a.get("served_name") for a in adapters.values()]
        _require(all(isinstance(s, str) and s for s in served), "every adapter needs a non-empty served_name")
        _require(len(set(served)) == len(served), "served names must be unique")
        _require(payload["slm"] not in served, "an adapter cannot be served under the base's name (B4 would silently become B3)")
        _require(all(isinstance(a.get("sha256"), str) and _SHA256.match(a["sha256"]) for a in adapters.values()),
                 "every adapter needs its sha256 (bench.contracts.facts.sha256_dir)")
        # what the set was trained on, which the router checks against the candidate it is served on
        _require(isinstance(payload.get("base_revision"), str) and bool(_COMMIT.match(payload["base_revision"])),
                 "adapters.base_revision must be the 40-hex commit of the base weights they were trained on")
        _require(isinstance(payload.get("chat_template_kwargs"), dict),
                 "adapters.chat_template_kwargs must be the mapping the chat template was rendered with in training")
    elif name == "allocation":
        _require(isinstance(payload.get("centroids"), str) and isinstance(payload.get("adapters"), str),
                 "allocation needs the centroids and adapters it was decided on")
        bad = {c: e for c, e in (payload.get("allocation") or {}).items() if e not in ENGINES_IN_ALLOCATION}
        _require(not bad, f"allocation engines must be in {ENGINES_IN_ALLOCATION}: {bad}")
    else:
        raise FactError(f"unknown fact {name!r}")
