"""`bench embed --source <run_id>`: the `embed` execution (design §1.2), the vectors S3 works on.

It runs in the agent environment because it must embed with `bench.contracts.clusters.embed`, the
function the router's `assign` uses, specified whole by `config.yaml › clustering.embedding` (which
J5 then writes into the centroids fact). For each invocation of the source it embeds, separately:
- **the prompt**, `prompt_text(prompt_messages)`: the only thing the router sees (SPEC S3);
- **the prompt and the action**, the same text with the teacher's response appended as the
  assistant's turn: what S3 forms clusters on. Only an invocation whose last attempt parsed has an
  action.

The source is a `curate` execution (the curated training examples) or an `agent` / `replay`
execution (e.g. the teacher on `calib`, for the assignment rate and the allocation). Texts are
embedded `clustering.batch_size` at a time; the default, 1, is how the router embeds (one prompt per
call), so both get the same vectors, with no padding in between. Every text's length in the
model's tokens is recorded, and whether it was longer than `max_seq_length` and so cut
(`truncation` says which end is kept): the report gives the fraction cut.

Writes `runs/embed-<ts>/`: `prompt.npy` and `prompt_action.npy` (float32, one unit row per text),
`index.jsonl` (per invocation: `call_id`, `question_id`, its rows, its token counts; never the call
site, which S3 must not see) and the manifest.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from bench import barrier, paths
from bench.contracts import clusters
from bench.contracts.config import config_sha256, load_config
from bench.judge.base import (JudgmentError, calls_of, final, invocations, manifest as run_manifest, question_order,
                              read_jsonl, reference, require_done, write_jsonl)
from bench.provenance import git_state

Texts = List[Dict[str, Any]]


def texts_of(source_run_id: str) -> Texts:
    """One row per invocation: `call_id` (the example's, or the first attempt's), `question_id`,
    `prompt` and `action` (None when no attempt parsed)."""
    found = run_manifest(source_run_id)
    if found.get("type") == "curate":
        require_done(source_run_id)
        rows = []
        for example in read_jsonl(paths.RUNS / source_run_id / "examples.jsonl"):
            rows.append({"call_id": example["call_id"], "question_id": example["question_id"],
                         "prompt": clusters.prompt_text(example["prompt"]),
                         "action": clusters.prompt_text(example["prompt"] + example["completion"])})
        return rows
    require_done(source_run_id, type=("agent", "replay"))
    rows = []
    for identity, attempts in sorted(invocations(calls_of(source_run_id)).items(), key=lambda kv: question_order(kv[0])):
        last = final(attempts)
        messages = attempts[0]["prompt_messages"]
        rows.append({"call_id": attempts[0]["call_id"], "question_id": identity[0],
                     "prompt": clusters.prompt_text(messages),
                     "action": clusters.prompt_text(messages + [{"role": "assistant", "content": last["response_text"]}])
                     if last["parsed_ok"] else None})
    return rows


def token_counter(embedding: Dict[str, Any]) -> Callable[[Sequence[str]], List[int]]:
    """The length of each text in the embedding model's own tokens, special tokens included."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding["model"], revision=embedding["revision"],
                                              trust_remote_code=embedding.get("trust_remote_code", False))
    return lambda texts: [len(ids) for ids in tokenizer(list(texts), add_special_tokens=True, truncation=False)["input_ids"]]


def _embed_all(texts: List[str], embedding: Dict[str, Any], batch_size: int, embed_fn) -> List[List[float]]:
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors += embed_fn(texts[start:start + batch_size], embedding)
    return vectors


def _lengths(tokens: List[int], limit: int) -> Dict[str, Any]:
    cut = sum(n > limit for n in tokens)
    return {"n": len(tokens), "max": max(tokens, default=0), "mean": sum(tokens) / len(tokens) if tokens else 0.0,
            "truncated": cut, "truncated_fraction": cut / len(tokens) if tokens else 0.0}


def run_embed(source_run_id: str, config_path: str = "config.yaml", embed_fn=None,
              count_tokens: Optional[Callable[[Sequence[str]], List[int]]] = None) -> Path:
    import numpy as np
    config = load_config(config_path)
    settings = config["clustering"]
    embedding = settings["embedding"]
    source = run_manifest(source_run_id)
    snapshot = paths.RUNS / source_run_id / "config.json"
    barrier.ensure_split_allowed(source.get("split"), json.loads(snapshot.read_text()) if snapshot.exists() else config)
    rows = texts_of(source_run_id)
    if not rows:
        raise JudgmentError(f"{source_run_id} has no invocations to embed")
    embed_fn = embed_fn or clusters.embed
    count_tokens = count_tokens or token_counter(embedding)

    started = datetime.now(timezone.utc)
    prompts = [r["prompt"] for r in rows]
    actions = [r["action"] for r in rows if r["action"] is not None]
    prompt_vectors = _embed_all(prompts, embedding, settings["batch_size"], embed_fn)
    action_vectors = _embed_all(actions, embedding, settings["batch_size"], embed_fn)
    prompt_tokens, action_tokens = count_tokens(prompts), count_tokens(actions)

    run_id = f"embed-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
    out = paths.RUNS / run_id
    out.mkdir(parents=True)
    np.save(out / "prompt.npy", np.asarray(prompt_vectors, dtype=np.float32))
    np.save(out / "prompt_action.npy", np.asarray(action_vectors, dtype=np.float32).reshape(len(actions), -1))
    index, action_row = [], 0
    for row_number, (row, tokens) in enumerate(zip(rows, prompt_tokens)):
        entry = {"call_id": row["call_id"], "question_id": row["question_id"], "prompt_row": row_number,
                 "prompt_tokens": tokens, "action_row": None, "action_tokens": None}
        if row["action"] is not None:
            entry.update(action_row=action_row, action_tokens=action_tokens[action_row])
            action_row += 1
        index.append(entry)
    write_jsonl(out / "index.jsonl", index)
    limit = embedding["max_seq_length"]
    manifest = {
        "run_id": run_id, "type": "embed", "status": "done", "source": reference(source_run_id),
        "source_type": source.get("type"), "split": source.get("split"), **git_state(),
        "config_sha256": config_sha256(config), "embedding": embedding, "batch_size": settings["batch_size"],
        "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        "n": len(rows), "dimension": len(prompt_vectors[0]),
        "tokens": {"prompt": _lengths(prompt_tokens, limit), "prompt_action": _lengths(action_tokens, limit)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out
