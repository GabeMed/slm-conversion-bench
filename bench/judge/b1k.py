"""b1k · B1's few-shot, chosen on the pilot (design §6.3, T18).

With k = 3 examples B1 sends about four times the input tokens; with k = 0 it may lose quality.
Fixed blind, either can decide the cost verdicts, so k ∈ {0, 3} is chosen by a rule, on the pilot
questions, before `bench prereg`:

- two replays of the teacher's B0 execution on calib, restricted to the pilot questions
  (`bench.data.pilot_ids`), both on `cheap_alt`: one with k = 0, one with k = 3 (a replay records
  the k it ran with, `few_shot_k`);
- on the call sites with gold (SQL generation and repair), each invocation of one replay is paired
  with the same invocation of the other, and each is right or wrong by its per-call evaluation
  (`bench eval <replay> --per-call`); an invocation whose final attempt did not parse is wrong;
- **k = 3 only if the one-sided 95% lower bound of EX_k3 − EX_k0 is above 0** (`seeds.bootstrap`,
  `stats.n_boot`). Otherwise k = 0, a tie included: the examples must earn their tokens. EX is per
  call, and the bound is J4's paired bootstrap with **the question as the unit**: the gold calls
  of one question share its context, so they are resampled together, as J2 and J4 do. With one
  gold call per question it is J4's bootstrap exactly.

Writes `judgments/b1k/<sha256>/choice.json`, the sha256 that of its canonical JSON. The author
copies `k` into `arms.B1.few_shot.k` before `bench prereg`.
"""
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bench import paths
from bench.contracts.concordance import GOLD_CALL_SITES
from bench.judge import j2
from bench.judge.base import Identity, JudgmentError, canonical, invocations, n_boot

JUDGMENT = "b1k"
K = {"k0": 0, "k3": 3}
CONFIG_KEYS = ("seeds.bootstrap", "stats.n_boot")  # and the pilot questions, by `bench.data.pilot_ids`
RULE = ("k = 3 only if the one-sided 95% lower bound of EX_k3 - EX_k0 per call, paired by invocation and "
        "resampled by question, is above 0; else k = 0")


def lower_bound(questions: List[Tuple[int, int]], seed: int, resamples: int) -> float:
    """The one-sided 95% lower bound of the mean paired difference per call. `questions` is, per
    question, (the sum of its calls' differences, its number of calls): J4's bootstrap and quantile,
    the question as the unit."""
    from bench.judge.j4 import lower_quantile  # F2's
    rng = random.Random(seed)
    n = len(questions)
    means = []
    for _ in range(resamples):
        drawn = [questions[int(rng.random() * n)] for _ in range(n)]
        means.append(sum(total for total, _ in drawn) / sum(calls for _, calls in drawn))
    return lower_quantile(means)


def choose(k0: Dict[Identity, bool], k3: Dict[Identity, bool], seed: int, resamples: int) -> Dict[str, Any]:
    """The rule, on the per-call correctness of the same invocations with k = 0 and with k = 3."""
    if set(k0) != set(k3) or not k0:
        raise JudgmentError("the two replays must answer the same invocations of the gold call sites, and at least one: "
                            f"{len(set(k0) ^ set(k3))} are in one only")
    by_question: Dict[str, List[int]] = {}
    for identity in k0:
        by_question.setdefault(identity[0], []).append(int(k3[identity]) - int(k0[identity]))
    questions = [(sum(by_question[q]), len(by_question[q])) for q in sorted(by_question)]
    ci_low = lower_bound(questions, seed, resamples)
    n = len(k0)
    return {"k": 3 if ci_low > 0 else 0, "rule": RULE, "n": n, "n_questions": len(questions),
            "ex_k0": sum(k0.values()) / n, "ex_k3": sum(k3.values()) / n,
            "diff": sum(total for total, _ in questions) / n, "ci_low": ci_low, "seed": seed, "n_boot": resamples}


def gold_calls(name: str, replay_run_id: str, eval_run_id: str, teacher_eval_run_id: str, pilot: List[str]):
    """(what was read, per-call correctness of the replay, and of the teacher) on the gold call sites
    of one of the two replays, checked to be the pilot on `cheap_alt` with that k."""
    reads, teacher, replay, replay_eval, teacher_eval, found = j2.replay_inputs(replay_run_id, eval_run_id, teacher_eval_run_id)
    if found.get("engine") != "cheap_alt":
        raise JudgmentError(f"{replay_run_id} is not a replay on cheap_alt")
    if found.get("few_shot_k") != K[name]:
        raise JudgmentError(f"{replay_run_id} ran with few_shot_k {found.get('few_shot_k')!r}, not {K[name]}")
    if set(found.get("question_ids") or []) != set(pilot):
        raise JudgmentError(f"{replay_run_id} did not replay exactly the pilot questions (bench.data.pilot_ids)")
    mine = {identity: j2.call_correct(identity, attempts, replay_eval, "replay")
            for identity, attempts in invocations(replay).items() if identity[1] in GOLD_CALL_SITES}
    source = invocations(teacher)
    if set(mine) - set(source):
        raise JudgmentError(f"{replay_run_id} has invocations its source does not: {sorted(set(mine) - set(source))[:3]}")
    return reads, mine, {identity: j2.call_correct(identity, source[identity], teacher_eval, "teacher") for identity in mine}


def run(k0: Tuple[str, str], k3: Tuple[str, str], teacher_eval_run_id: str, config: Dict[str, Any]) -> Path:
    """`k0` and `k3` are (replay run, its per-call eval run). Returns the path of choice.json."""
    from bench import data
    pilot = data.pilot_ids(config)
    reads_k0, correct_k0, teacher = gold_calls("k0", *k0, teacher_eval_run_id, pilot)
    reads_k3, correct_k3, _ = gold_calls("k3", *k3, teacher_eval_run_id, pilot)
    result = choose(correct_k0, correct_k3, config["seeds"]["bootstrap"], n_boot(config))
    result["ex_teacher"] = sum(teacher.values()) / len(teacher) if teacher else None
    result["pilot_ids"] = sorted(pilot, key=int)
    payload = {"judgment": JUDGMENT, "reads": {"k0": reads_k0, "k3": reads_k3, "config": list(CONFIG_KEYS)}, "result": result}
    raw = canonical(payload)
    path = Path(paths.ROOT) / "judgments" / JUDGMENT / hashlib.sha256(raw).hexdigest() / "choice.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    return path
