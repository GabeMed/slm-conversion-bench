"""What the judgments of F4 share: executions read as facts, invocations out of C1, and results
written by content.

A judgment is a pure function over facts (design §1.2): it reads executions under `runs/` and
earlier judgments, never a model, a GPU or a database, and re-running it gives the same bytes. Its
result is written once, at `judgments/<J>/<sha256>/result.json`, with the sha256 of its canonical
JSON as the directory, and records under `reads` every execution it read (with the sha256 of the
manifest it saw) and every judgment it built on. A judgment that configures a later execution also
writes its fact with `bench.contracts.facts.write_fact` (J5 centroids, J6 choice, J7 allocation).
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bench import paths
from bench.contracts.calls import read_calls, validate_calls
from bench.judge import JudgeError

Identity = Tuple[str, str, str]  # (question_id, call_site, invocation_key): one invocation, across runs
JudgmentError = JudgeError  # one error type for every judgment, F2's and F4's


def canonical(payload: Any) -> bytes:
    """Sorted keys, no whitespace, and no NaN: a result that cannot be written exactly is refused."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def read_jsonl(path: Path) -> List[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- executions

def run_dir(run_id: str) -> Path:
    path = paths.RUNS / run_id
    if not (path / "manifest.json").is_file():
        raise JudgmentError(f"no execution {run_id} (runs/{run_id}/manifest.json is missing)")
    return path


def manifest(run_id: str) -> Dict[str, Any]:
    return json.loads((run_dir(run_id) / "manifest.json").read_text())


def reference(run_id: str) -> Dict[str, str]:
    """How a result names an execution it read: its id and the sha256 of the manifest it saw."""
    raw = (run_dir(run_id) / "manifest.json").read_bytes()
    return {"run_id": run_id, "manifest_sha256": hashlib.sha256(raw).hexdigest()}


def require_done(run_id: str, **expected: Any) -> Dict[str, Any]:
    """The manifest of a finished execution, checked against what the caller needs of it
    (`type`, `arm`, `split`, ...). An incomplete or failed execution is never judged. An `eval`
    execution records no status: it writes its manifest last, after every result, so a manifest
    means it finished."""
    found = manifest(run_id)
    finished = found.get("status") == "done" or (found.get("type") == "eval" and "status" not in found)
    if not finished:
        raise JudgmentError(f"{run_id} is {found.get('status')!r}, not 'done': an incomplete execution is never judged")
    for key, value in expected.items():
        allowed = value if isinstance(value, (tuple, list, set)) else (value,)
        if found.get(key) not in allowed:
            raise JudgmentError(f"{run_id}: {key} is {found.get(key)!r}, expected {value!r}")
    return found


def calls_of(run_id: str) -> List[dict]:
    """The C1 lines of an execution, validated: a judgment never reads an invalid log."""
    calls = read_calls(run_dir(run_id) / "calls.jsonl")
    errors = validate_calls(calls)
    if errors:
        raise JudgmentError(f"{run_id}/calls.jsonl breaks C1: {errors[:3]}")
    return calls


# ---------------------------------------------------------------- invocations

def invocations(calls: Iterable[dict]) -> Dict[Identity, List[dict]]:
    """Every invocation and its attempts, in attempt order."""
    grouped: Dict[Identity, List[dict]] = {}
    for call in calls:
        grouped.setdefault((call["question_id"], call["call_site"], call["invocation_key"]), []).append(call)
    for attempts in grouped.values():
        attempts.sort(key=lambda c: c["attempt"])
    return grouped


def final(attempts: List[dict]) -> dict:
    """The attempt whose output the agent used, or failed on: the last one."""
    return attempts[-1]


def question_order(identity: Identity) -> Tuple[int, str, str]:
    question_id, call_site, key = identity
    return (int(question_id) if question_id.isdigit() else 0, call_site, key)


# ---------------------------------------------------------------- results

def write_result(judgment: str, reads: Dict[str, Any], result: Dict[str, Any], root: Optional[Path] = None) -> Path:
    payload = {"judgment": judgment, "reads": reads, "result": result}
    raw = canonical(payload)
    path = (root or paths.ROOT) / "judgments" / judgment / hashlib.sha256(raw).hexdigest() / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    return path


def read_result(path, judgment: str) -> Dict[str, Any]:
    """A judgment result, checked to be the judgment named and to hash to its directory."""
    path = Path(path)
    if not path.is_absolute():
        path = paths.ROOT / path
    raw = path.read_bytes()
    if path.parent.name != hashlib.sha256(raw).hexdigest():
        raise JudgmentError(f"{path}: content sha256 is not its directory name")
    payload = json.loads(raw)
    if payload.get("judgment") != judgment:
        raise JudgmentError(f"{path} is a {payload.get('judgment')} result, not {judgment}")
    return payload


def result_reference(path) -> Dict[str, str]:
    """How a result names another judgment's result it built on: the sha256 of its content."""
    return {"judgment": read_json_judgment(path), "sha256": Path(path).parent.name}


def read_json_judgment(path) -> str:
    path = Path(path)
    return json.loads((path if path.is_absolute() else paths.ROOT / path).read_bytes())["judgment"]


def n_boot(config: Dict[str, Any]) -> int:
    """J4's bootstrap resamples: F2's `stats.n_boot`."""
    value = (config.get("stats") or {}).get("n_boot")
    if not isinstance(value, int) or value < 1:
        raise JudgmentError("stats.n_boot (J4's bootstrap resamples, F2) is not set")
    return value


def relative(path: Path) -> str:
    """A path as a result records it: relative to the repository, never a local absolute path."""
    return Path(path).resolve().relative_to(Path(paths.ROOT).resolve()).as_posix()
