"""b1k · B1's few-shot, chosen on the pilot (design §6.3, T18).

With k = 3 examples B1 sends about four times the input tokens; with k = 0 it may lose quality.
Fixed blind, either can decide the cost verdicts, so k ∈ {0, 3} is chosen by a rule, on the pilot
questions, before `bench prereg`:

- two executions of the agent as arm B1 on calib, on exactly the pilot questions
  (`bench.data.pilot_ids`), every call on `cheap_alt`: one with k = 0, one with k = 3 (an agent
  execution records the k it ran with, `few_shot_k`), each with its end-to-end evaluation
  (`bench eval <run>`, not `--per-call`);
- each pilot question of one execution is paired with the same question of the other, and each is
  right or wrong end to end, by its evaluation (a question the agent answered with no SQL is wrong);
- **k = 3 only if the one-sided 95% lower bound of EX_k3 − EX_k0 is above 0** (`seeds.bootstrap`,
  `stats.n_boot`). Otherwise k = 0, a tie included: the examples must earn their tokens. The bound is
  J4's paired bootstrap, the question as the unit (`bench.judge.j4.noninferiority` at a margin of 0,
  its `ci_low`).

Until 2026-11-10 the unit was the call, on the gold call sites of two replays of the teacher's
execution (prereg/DEVIATIONS.md): a replay feeds every call the teacher's context, so it cannot see
what the examples do to the agent's own path (which tools it calls, whether it reaches SQL at all).

Writes `judgments/b1k/<sha256>/choice.json`, the sha256 that of its canonical JSON. The author
copies `k` into `arms.B1.few_shot.k` before `bench prereg`.
"""
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bench import paths
from bench.judge.base import JudgmentError, calls_of, canonical, n_boot, read_jsonl, reference, require_done, run_dir

JUDGMENT = "b1k"
K = {"k0": 0, "k3": 3}
# seeds.calib_split, stats.pilot_mix and stats.pilot_size are what bench.data.pilot_ids reads
CONFIG_KEYS = ("seeds.bootstrap", "seeds.calib_split", "stats.n_boot", "stats.pilot_mix", "stats.pilot_size")
RULE = ("k = 3 only if the one-sided 95% lower bound of EX_k3 - EX_k0 end to end on the pilot questions, paired and "
        "resampled by question, is above 0; else k = 0")


def choose(k0: Dict[str, bool], k3: Dict[str, bool], seed: int, resamples: int) -> Dict[str, Any]:
    """The rule, on the end-to-end correctness of the same questions with k = 0 and with k = 3."""
    from bench.judge.j4 import noninferiority  # F2's paired bootstrap
    if set(k0) != set(k3) or not k0:
        raise JudgmentError("the two executions must be evaluated on the same questions, and on at least one: "
                            f"{len(set(k0) ^ set(k3))} are in one only")
    paired = noninferiority(k3, k0, seed, resamples, margin=0.0)
    n = paired["n"]
    return {"k": 3 if paired["ci_low"] > 0 else 0, "rule": RULE, "unit": "question, end to end", "n": n,
            "ex_k0": sum(k0.values()) / n, "ex_k3": sum(k3.values()) / n, "diff": paired["diff"],
            "ci_low": paired["ci_low"], "k3_only": paired["a_only"], "k0_only": paired["b_only"],
            "ids": sorted(k0, key=int), "seed": seed, "n_boot": resamples}


def end_to_end(name: str, run_id: str, eval_run_id: str, pilot: List[str]) -> Tuple[Dict[str, Any], Dict[str, bool], int]:
    """(what was read, the correctness of each pilot question, how many it answered with no SQL) for
    one of the two executions, checked to be B1 on the pilot of calib, on `cheap_alt`, with that k."""
    found = require_done(run_id, type="agent", arm="B1", split="calib")
    if found.get("few_shot_k") != K[name]:
        raise JudgmentError(f"{run_id} ran with few_shot_k {found.get('few_shot_k')!r}, not {K[name]}")
    if sorted(found.get("question_ids") or [], key=int) != sorted(pilot, key=int):
        raise JudgmentError(f"{run_id} did not run exactly the pilot questions (bench.data.pilot_ids)")
    engines = {c["engine"] for c in calls_of(run_id)}
    if engines != {"cheap_alt"}:
        raise JudgmentError(f"{run_id} is not an execution on cheap_alt: its calls ran on {sorted(engines)}")
    evaluated = require_done(eval_run_id, type="eval")
    if evaluated.get("per_call"):
        raise JudgmentError(f"{eval_run_id} is a per-call evaluation: the choice reads the end-to-end one")
    if evaluated.get("source_run_id") != run_id:
        raise JudgmentError(f"{eval_run_id} evaluated {evaluated.get('source_run_id')}, not {run_id}")
    rows = read_jsonl(run_dir(eval_run_id) / "results.jsonl")
    correct = {r["question_id"]: bool(r["correct"]) for r in rows}
    if len(rows) != len(correct) or set(correct) != set(pilot):
        raise JudgmentError(f"{eval_run_id} does not evaluate each pilot question exactly once")
    without_sql = sum(1 for r in rows if not r.get("predicted_sql"))
    return {"run": reference(run_id), "eval": reference(eval_run_id)}, correct, without_sql


def run(k0: Tuple[str, str], k3: Tuple[str, str], config: Dict[str, Any]) -> Path:
    """`k0` and `k3` are (B1's agent execution on the pilot, its end-to-end eval run). Returns the
    path of choice.json."""
    from bench import data
    pilot = data.pilot_ids(config)
    reads_k0, correct_k0, without_sql_k0 = end_to_end("k0", *k0, pilot)
    reads_k3, correct_k3, without_sql_k3 = end_to_end("k3", *k3, pilot)
    result = choose(correct_k0, correct_k3, config["seeds"]["bootstrap"], n_boot(config))
    result["without_sql"] = {"k0": without_sql_k0, "k3": without_sql_k3}
    result["pilot_ids"] = sorted(pilot, key=int)
    payload = {"judgment": JUDGMENT, "reads": {"k0": reads_k0, "k3": reads_k3, "config": list(CONFIG_KEYS)}, "result": result}
    raw = canonical(payload)
    path = Path(paths.ROOT) / "judgments" / JUDGMENT / hashlib.sha256(raw).hexdigest() / "choice.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    return path
