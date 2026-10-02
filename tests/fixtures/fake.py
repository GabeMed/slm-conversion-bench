"""Synthetic executions for F4's tests: valid C1 lines with CHESS-shaped prompts (a long fixed
template, the part that varies at the end), and runs written where `bench.paths` points."""
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench import paths

# Each call site's template: many fixed lines, like CHESS's few-shot templates, and one shared
# preamble line, so what tells two call sites apart is not a single line.
PREAMBLE = "You are an expert and very smart data analyst."
TEMPLATES = {site: "\n".join([PREAMBLE] + [f"{site} example {i}: follow the instructions of step {i} carefully."
                                           for i in range(30)])
             for site in ("agent_ir", "agent_ss", "agent_cg", "extract_keywords", "filter_column",
                          "select_tables", "select_columns", "generate_candidate", "revise")}


def call_id(*parts: Any) -> str:
    return str(uuid.UUID(bytes=hashlib.md5("|".join(map(str, parts)).encode()).digest(), version=4))


def prompt(site: str, question: str, detail: str = "") -> List[Dict[str, str]]:
    return [{"role": "user", "content": f"{TEMPLATES[site]}\nQuestion: {question}\nDetail: {detail}"}]


def usage(input_tokens: int = 1000, cached: int = 0, output: int = 50, source: str = "api") -> Dict[str, Any]:
    if source == "missing":
        return {"input": None, "cached_input": None, "output": None, "source": "missing"}
    return {"input": input_tokens, "cached_input": cached, "output": output, "source": source}


def call(run_id: str, question_id: str, site: str, key: str = "single", *, response: Optional[str] = "ok",
         parsed: Any = None, parsed_ok: bool = True, attempt: int = 1, retry_of: Optional[str] = None,
         messages: Optional[List[Dict[str, str]]] = None, role: str = "production_llm",
         engine: str = "production_llm", model: str = "teacher-model", cluster: Optional[str] = None,
         use: Optional[Dict[str, Any]] = None, latency_ms: int = 100,
         provider: Optional[str] = "provider-x") -> Dict[str, Any]:
    return {
        "run_id": run_id, "call_id": call_id(run_id, question_id, site, key, attempt), "retry_of": retry_of,
        "attempt": attempt, "question_id": question_id, "call_site": site, "invocation_key": key,
        "cluster": cluster, "engine": engine, "model_role": role, "model": model, "endpoint": "fake",
        "prompt_messages": messages or prompt(site, f"question {question_id}"),
        "response_text": response, "parsed_output": parsed if parsed_ok else None, "parsed_ok": parsed_ok,
        "usage": use or usage(), "latency_ms": latency_ms, "started_at": "2026-09-30T12:00:00+00:00",
        "temperature": 0.0, "error": None if parsed_ok else "unparsed",
        "provider": None if role == "slm" else provider,  # the provider the configuration pins for an API role
    }


def write_run(run_id: str, manifest: Dict[str, Any], calls: Optional[List[dict]] = None,
              config: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, Any]] = None) -> Path:
    run_dir = paths.RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": run_id, "status": "done", **manifest}
    if manifest["status"] is None:  # an eval execution records no status (it writes its manifest last)
        del manifest["status"]
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if calls is not None:
        (run_dir / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    if config is not None:
        (run_dir / "config.json").write_text(json.dumps(config))
    for name, content in (files or {}).items():
        path = run_dir / name
        if isinstance(content, list):
            path.write_text("".join(json.dumps(row) + "\n" for row in content))
        else:
            path.write_text(content if isinstance(content, str) else json.dumps(content))
    return run_dir


# ---------------------------------------------------------------- embeddings

DIMENSION = 64


def fake_embed(texts, embedding):
    """A deterministic stand-in for `clusters.embed`: unit bag-of-hashed-words vectors. Texts that
    share a template land together, as they would with a real model."""
    vectors = []
    for text in texts:
        vector = [0.0] * DIMENSION
        for word in text.split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSION] += 1.0
        norm = sum(x * x for x in vector) ** 0.5 or 1.0
        vectors.append([x / norm for x in vector])
    return vectors


def fake_tokens(texts):
    return [len(text.split()) for text in texts]


def example(question_id: str, site: str, key: str = "single", detail: str = "", answer: str = "ok") -> Dict[str, Any]:
    return {"call_id": call_id("curated", question_id, site, key), "question_id": question_id, "call_site": site,
            "invocation_key": key, "prompt": prompt(site, f"question {question_id} about topic {question_id}", detail),
            "completion": [{"role": "assistant", "content": f"{site} answer {answer} {question_id}"}]}


def curate_run(run_id: str, examples: List[Dict[str, Any]], masked: Optional[Dict[str, str]] = None,
               originals_of: Optional[List[Dict[str, Any]]] = None) -> Path:
    """A curate execution and the teacher execution it curated, whose calls are the examples'
    originals (same call id; `originals_of` when the examples' own labels are not the calls').
    `masked` replaces text in the examples only, as masking would."""
    source = f"{run_id}-source"
    originals = [{**call(source, e["question_id"], e["call_site"], e["invocation_key"], messages=e["prompt"],
                         response=e["completion"][0]["content"], parsed={}), "call_id": e["call_id"]}
                 for e in (originals_of or examples)]
    write_run(source, {"type": "agent", "arm": "B0", "split": "train"}, originals)

    def mask(text: str) -> str:
        for old, new in (masked or {}).items():
            text = text.replace(old, new)
        return text
    shown = [{**e, "prompt": [{**m, "content": mask(m["content"])} for m in e["prompt"]],
              "completion": [{**m, "content": mask(m["content"])} for m in e["completion"]]} for e in examples]
    return write_run(run_id, {"type": "curate", "split": "train", "n": len(examples), "sources": [{"run_id": source}],
                              "counts": {"total": {"kept": len(examples)}}}, files={"examples.jsonl": shown})


# ---------------------------------------------------------------- a repository

def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return merged


def repo(tmp_path: Path, monkeypatch, overrides: Optional[Dict[str, Any]] = None):
    """`bench.paths` pointed at tmp_path, and the shipped config.yaml with `overrides` merged in,
    written there. Returns (config path, config)."""
    import yaml
    from bench.contracts.config import load_config
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, value in {"ROOT": tmp_path, "RUNS": tmp_path / "runs"}.items():
        monkeypatch.setattr(paths, name, value)
    config = _merge(load_config(Path(__file__).resolve().parents[2] / "config.yaml"), overrides or {})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path, load_config(config_path)
