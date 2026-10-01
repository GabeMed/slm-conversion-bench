"""S2 · curation and filtering (SPEC §4 S2): the teacher's logs on `train` become training examples.

`bench curate --source <run_id> ...` is an **execution**, not a judgment: its success filter runs
the teacher's SQL on the pinned database. Per invocation of the teacher (its last attempt, which
must have parsed), in this order, with the count of each step, overall and per call site:

1. **Success filter, with a production signal, never the gold.** SQL generation and repair: the SQL
   the agent took from the output runs without error on the question's database (read-only, with a
   timeout, through the evaluator's own `execute`) and returns at least one row. The date it runs at
   is the pre-registered `eval.fixed_date` of the source's configuration, the one the evaluator
   substitutes for 'now', so a query about "today" means the same day in curation and in scoring.
   Every other call site: the output parsed, which is the agent's own format check (C1 `parsed_ok`).
2. **Masking of sensitive data** by the regular expressions of `config.yaml › curation.mask`, in the
   prompt and in the completion alike, with the number of detections per pattern. Applied after the
   filter, so the SQL that is executed is the one the agent executed. A SQL completion that masking
   changes is dropped (`masked_sql`): it is no longer the SQL that passed the filter, and training on
   it would teach a query that never ran.
3. **Exact duplicates**: the same masked prompt and completion.
4. **Near duplicates** by MinHash (`curation.near_duplicate`). CHESS prompts are mostly template (a
   column-filter prompt is ~17 kB of fixed examples around a few lines that vary), so the Jaccard
   of two whole prompts is near 1 for any two calls of a call site: comparing them whole would
   delete almost every example. Within a call site, a prompt line present in more than
   `template_share` of its prompts is template and is left out of the comparison; two examples
   are near duplicates when what is left of their prompts **and** their completions both reach the
   threshold. Of each group, the first (by question id) is kept.

**Paraphrase** of entities and numbers is declared **not applied** (SPEC S2, deviation): in
text-to-SQL the values are part of the right answer.

Only the teacher's outputs are curated: every call of the source must be `production_llm`, and the
source's configuration must record that the teacher's weights license and provider terms allow
training on its outputs (SPEC §3, the do-not-train rule). The output is `runs/curate-<ts>/`:
`examples.jsonl` and a manifest with the counts. `bench datasets` then splits the examples by the
S3 clusters (J5) into `train/datasets/<cluster>.jsonl`, in TRL's conversational prompt/completion
shape (design §5.1, "Dados de treino").
"""
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bench import barrier, data, paths
from bench.contracts.calls import CALL_SITES
from bench.contracts.clusters import prompt_text
from bench.contracts.config import config_sha256, load_config
from bench.judge.base import (JudgmentError, calls_of, canonical, final, invocations, question_order, read_jsonl,
                              read_result, reference, require_done, write_jsonl)
from bench.provenance import git_state

SQL_OUTPUT_KEY = {"generate_candidate": "SQL", "revise": "refined_sql_query"}  # CHESS's parsers (llm/parsers.py)
STEPS = ("invocations", "unparsed", "sql_error", "sql_empty", "passed_filter", "masked_sql", "exact_duplicates",
         "near_duplicates", "kept")


class CurationError(JudgmentError):
    pass


# ---------------------------------------------------------------- masking

def _luhn(digits: str) -> bool:
    numbers = [int(d) for d in digits if d.isdigit()]
    checksum = sum(numbers[-1::-2]) + sum(sum(divmod(2 * d, 10)) for d in numbers[-2::-2])
    return checksum % 10 == 0


class Masker:
    """The patterns of `curation.mask`: {name: {pattern, placeholder, check?}}; `check: luhn` keeps
    only digit runs that pass the card checksum, so ids and dates are not masked as cards."""

    def __init__(self, spec: Dict[str, Dict[str, str]]):
        self.patterns = [(name, re.compile(p["pattern"]), p["placeholder"], p.get("check")) for name, p in sorted(spec.items())]
        self.detections: Counter = Counter()

    def __call__(self, text: str) -> str:
        for name, pattern, placeholder, check in self.patterns:
            def replace(match: re.Match) -> str:
                if check == "luhn" and not _luhn(match.group(0)):
                    return match.group(0)
                self.detections[name] += 1
                return placeholder
            text = pattern.sub(replace, text)
        return text


# ---------------------------------------------------------------- near duplicates

def _shingles(text: str, n: int) -> set:
    words = text.split()
    if len(words) <= n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _minhash(shingles: set, num_perm: int):
    from datasketch import MinHash
    signature = MinHash(num_perm=num_perm, seed=1)
    signature.update_batch([s.encode() for s in sorted(shingles)])
    return signature


def template_lines(prompts: Sequence[str], share: float) -> set:
    """Lines present in more than `share` of the prompts: the call site's template."""
    counts = Counter(line for prompt in prompts for line in set(prompt.splitlines()))
    return {line for line, n in counts.items() if n > share * len(prompts)}


def near_duplicates(examples: List[dict], settings: Dict[str, Any]) -> List[int]:
    """Indices (into `examples`, all of one call site, in keep-priority order) that repeat an
    earlier example: variable prompt text and completion both at or above the threshold."""
    from datasketch import MinHashLSH
    n, num_perm, threshold = settings["shingle_words"], settings["num_perm"], settings["threshold"]
    prompts = [prompt_text(e["prompt"]) for e in examples]
    template = template_lines(prompts, settings["template_share"])
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: Dict[str, Tuple[Any, Any]] = {}
    dropped = []
    for i, (example, prompt) in enumerate(zip(examples, prompts)):
        variable = "\n".join(line for line in prompt.splitlines() if line not in template)
        prompt_sig = _minhash(_shingles(variable, n), num_perm)
        completion_sig = _minhash(_shingles(example["completion"][0]["content"], n), num_perm)
        duplicate = any(prompt_sig.jaccard(kept[k][0]) >= threshold and completion_sig.jaccard(kept[k][1]) >= threshold
                        for k in sorted(lsh.query(prompt_sig)))
        if duplicate:
            dropped.append(i)
        else:
            kept[str(i)] = (prompt_sig, completion_sig)
            lsh.insert(str(i), prompt_sig)
    return dropped


# ---------------------------------------------------------------- the success filter

def sql_signal(sql: Optional[str], run_sql: Callable[[str], Tuple[Optional[list], Optional[str]]]) -> Optional[str]:
    """None when the SQL passes the production signal; otherwise the step that drops it."""
    if not sql or not sql.strip():
        return "sql_error"
    rows, error = run_sql(sql)
    if error is not None:
        return "sql_error"
    return "sql_empty" if not rows else None


def curate(calls: List[dict], settings: Dict[str, Any], run_sql: Callable[[str, str], Tuple[Optional[list], Optional[str]]]
           ) -> Tuple[List[dict], Dict[str, Any]]:
    """The curated examples and the counts of every step. `run_sql(question_id, sql)` executes on
    the question's database. Pure but for `run_sql`."""
    counts = {site: Counter() for site in CALL_SITES}
    candidates: Dict[str, List[dict]] = {site: [] for site in CALL_SITES}
    for identity, attempts in sorted(invocations(calls).items(), key=lambda kv: question_order(kv[0])):
        question_id, call_site, invocation_key = identity
        counts[call_site]["invocations"] += 1
        chosen = final(attempts)
        if not chosen["parsed_ok"]:
            counts[call_site]["unparsed"] += 1
            continue
        if call_site in SQL_OUTPUT_KEY:
            dropped_by = sql_signal((chosen["parsed_output"] or {}).get(SQL_OUTPUT_KEY[call_site]),
                                    lambda sql: run_sql(question_id, sql))
            if dropped_by:
                counts[call_site][dropped_by] += 1
                continue
        counts[call_site]["passed_filter"] += 1
        candidates[call_site].append({
            "call_id": chosen["call_id"], "question_id": question_id, "call_site": call_site,
            "invocation_key": invocation_key, "prompt": chosen["prompt_messages"],
            "completion": [{"role": "assistant", "content": chosen["response_text"]}]})

    masker = Masker(settings["mask"])
    examples: List[dict] = []
    for call_site in CALL_SITES:
        seen, unique = set(), []
        for example in candidates[call_site]:
            example["prompt"] = [{"role": m["role"], "content": masker(m["content"])} for m in example["prompt"]]
            completion = example["completion"][0]["content"]
            if call_site in SQL_OUTPUT_KEY and masker(completion) != completion:
                counts[call_site]["masked_sql"] += 1
                continue
            example["completion"] = [{"role": "assistant", "content": masker(completion)}]
            key = hashlib.sha256(canonical([example["prompt"], example["completion"]])).hexdigest()
            if key in seen:
                counts[call_site]["exact_duplicates"] += 1
                continue
            seen.add(key)
            unique.append(example)
        dropped = set(near_duplicates(unique, settings["near_duplicate"])) if len(unique) > 1 else set()
        counts[call_site]["near_duplicates"] += len(dropped)
        kept = [e for i, e in enumerate(unique) if i not in dropped]
        counts[call_site]["kept"] += len(kept)
        examples += kept
    per_site = {site: {step: counts[site][step] for step in STEPS} for site in CALL_SITES if counts[site]["invocations"]}
    total = {step: sum(c[step] for c in per_site.values()) for step in STEPS}
    return examples, {"total": total, "per_call_site": per_site,
                      "mask_detections": dict(sorted(masker.detections.items())),
                      "paraphrase": "not applied (SPEC §4 S2: in text-to-SQL the values are part of the answer)"}


# ---------------------------------------------------------------- the execution

def _teacher_may_train(config: Dict[str, Any]) -> None:
    terms = config["roles"]["production_llm"].get("terms") or {}
    missing = [k for k in ("weights_license", "provider_terms", "checked_on") if not terms.get(k)]
    if missing:
        raise CurationError(f"roles.production_llm.terms.{', '.join(missing)} not recorded: the teacher's outputs "
                            f"may not become training data until its terms are checked (SPEC §3, do-not-train rule)")


def run_curate(source_run_ids: List[str], config_path: str = "config.yaml") -> Path:
    config = load_config(config_path)
    settings = config["curation"]
    calls: List[dict] = []
    databases: Dict[str, Path] = {}  # question id -> the pinned SQLite file of its database
    days = set()  # the pre-registered date the SQL runs at
    seen_questions: set = set()
    for run_id in source_run_ids:
        found = require_done(run_id, type="agent", arm="B0", split="train")
        snapshot = json.loads((paths.RUNS / run_id / "config.json").read_text())
        if config_sha256(snapshot) != found["config_sha256"]:
            raise CurationError(f"the configuration snapshot of {run_id} does not match its manifest")
        _teacher_may_train(snapshot)
        barrier.ensure_split_allowed("train", snapshot)
        from bench.evaluate import fixed_date
        try:  # the day eval scores at, checked as eval checks it
            days.add(fixed_date(snapshot))
        except data.DataError as e:
            raise CurationError(f"{run_id}'s configuration: {e}: the SQL has no pre-registered day to run at") from e
        overlap = seen_questions & set(found["question_ids"])
        if overlap:
            raise CurationError(f"{run_id} repeats questions of another source: {sorted(overlap, key=int)[:5]}")
        seen_questions |= set(found["question_ids"])
        run_calls = calls_of(run_id)
        others = {c["model_role"] for c in run_calls} - {"production_llm"}
        if others:
            raise CurationError(f"{run_id} has calls of {sorted(others)}: only the teacher's outputs are training data")
        calls += run_calls
        questions = data.questions_for(snapshot, "train")
        for db_id in sorted({questions[q]["db_id"] for q in found["question_ids"]}):
            data.check_database(snapshot, db_id)
        databases.update({q: paths.sqlite_path(snapshot, questions[q]["db_id"]) for q in found["question_ids"]})

    if len(days) > 1:
        raise CurationError(f"the sources were run under different fixed dates {sorted(days)}")
    day = days.pop()
    from bench.evaluate import execute

    def run_sql(question_id: str, sql: str):
        rows, error, _ = execute(databases[question_id], sql, settings["sql_timeout_s"], day)
        return rows, error

    started = datetime.now(timezone.utc)
    examples, counts = curate(calls, settings, run_sql)
    run_id = f"curate-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
    out = paths.RUNS / run_id
    out.mkdir(parents=True)
    write_jsonl(out / "examples.jsonl", examples)
    manifest = {
        "run_id": run_id, "type": "curate", "split": "train", "status": "done",
        "sources": [reference(r) for r in source_run_ids], **git_state(),
        "config_sha256": config_sha256(config), "settings": settings,
        "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        "n": len(examples), "counts": counts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


# ---------------------------------------------------------------- datasets per cluster

def write_datasets(curate_run_id: str, j5_result: str, config_path: str = "config.yaml",
                   out: Optional[Path] = None) -> Path:
    """`train/datasets/<cluster>.jsonl` from the curated examples and the S3 clusters (J5), one
    TRL conversational line `{"prompt": [...], "completion": [{"role": "assistant", ...}]}` each,
    plus a manifest with the volume of each cluster against the paper's 10k–100k rule of thumb."""
    config = load_config(config_path)
    require_done(curate_run_id, type="curate")
    j5 = read_result(j5_result, "J5")
    if j5["reads"]["curate"]["run_id"] != curate_run_id:
        raise CurationError(f"{j5_result} clustered {j5['reads']['curate']['run_id']}, not {curate_run_id}")
    members: Dict[str, str] = j5["result"]["members"]
    examples = read_jsonl(paths.RUNS / curate_run_id / "examples.jsonl")
    if {e["call_id"] for e in examples} != set(members):
        raise CurationError("the J5 clusters are not over exactly the curated examples")
    out = out or paths.ROOT / "train" / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    by_cluster: Dict[str, List[dict]] = {c: [] for c in j5["result"]["clusters"]}
    for example in examples:
        by_cluster[members[example["call_id"]]].append({"prompt": example["prompt"], "completion": example["completion"]})
    low, high = config["curation"]["rule_of_thumb"]
    for stale in out.glob("*.jsonl"):  # the files of earlier clusters: never left for S5 to train on
        stale.unlink()
    for cluster, rows in by_cluster.items():
        write_jsonl(out / f"{cluster}.jsonl", rows)
    manifest = {
        "curate": reference(curate_run_id), "j5": Path(j5_result).parent.name,
        "centroids": j5["result"]["centroids"]["sha256"],
        "clusters": {c: {"n": len(rows), "rule_of_thumb": "below" if len(rows) < low else "above" if len(rows) > high else "within"}
                     for c, rows in by_cluster.items()},
        "rule_of_thumb": [low, high],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out
