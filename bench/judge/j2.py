"""J2 · per call site, with the teacher's context (SPEC §6.2; contract C3).

A `replay` execution sends the first attempt of every invocation of a teacher execution (B0) to
another engine, with the same retry policy (design §5.1). J2 pairs the two by invocation
(question id, call site, invocation key) and judges the engine's final output:
- **SQL generation and repair** (the call sites with gold): execution accuracy of the replayed SQL,
  from the per-call evaluation of the replay (`bench eval <replay> --per-call`, F2), next to the
  teacher's own on the same invocations. For the non-inferiority test (J4, which takes one boolean
  per question), each question is one unit: it counts as correct when every one of its paired
  invocations in the group is. Both are reported: EX per call (`ex_replay`, `ex_teacher`) and EX
  per question (`ex_by_question`), the quantity J4 tests; they differ only where a question has
  more than one gold call in the group (several repair rounds, or generation and repair together).
- **Every other call site**: agreement with the teacher's decision (`bench.contracts.concordance.agree`),
  over the invocations whose teacher output parsed. This is fidelity, not accuracy (SPEC §6.2).
- **Format validity** for every call site: the share of the engine's invocations whose final attempt
  parsed (SPEC §6.3). An invocation whose last attempt did not parse has no output: it disagrees,
  and its SQL is wrong.
- **Truncation** for every call site of one execution: of the calls that returned an answer (every
  attempt is a call), the share that was cut off, so a cut-off answer is not read as the model's
  failure. A call is truncated when the API stopped it at the token limit (`finish_reason` is
  `length`) or its content is empty. A C1 line without `finish_reason` (written before the field
  existed) is unknown, and so is one whose provider reported none; the rate is over the calls that
  are known.

Invocations are grouped by call site, or by any other key (J7 groups them by cluster). A replay
that declares its `call_sites` (F1's `replay --call-site`) is judged on those only: the teacher's
invocations of other call sites are out of its scope, not wrong. Within its scope, an invocation of
the teacher that the replay lacks is a refusal: a missing answer is not evidence either way. With a
single execution and no replay, J2 gives the format validity of that execution alone (A5).
"""
from typing import Any, Callable, Dict, Iterable, List, Optional

from bench.contracts.concordance import GOLD_CALL_SITES, agree
from bench.judge.base import (Identity, JudgmentError, calls_of, final, invocations, manifest as run_manifest,
                              read_jsonl, reference, require_done, run_dir, write_result)

JUDGMENT = "J2"
CONFIG_KEYS = ()  # J2 reads no configuration key; its result's `reads` says so (design §6.2)
Group = Callable[[Identity, List[dict]], Optional[str]]


def by_call_site(identity: Identity, attempts: List[dict]) -> str:
    return identity[1]


def eval_index(rows: Iterable[dict]) -> Dict[Identity, bool]:
    """Per-call evaluation rows (F2's results.jsonl) by invocation."""
    index: Dict[Identity, bool] = {}
    for row in rows:
        identity = (str(row["question_id"]), row["call_site"], row["invocation_key"])
        if identity in index:
            raise JudgmentError(f"two per-call results for {identity}")
        index[identity] = bool(row["correct"])
    return index


def _correct(identity: Identity, attempts: Optional[List[dict]], results: Dict[Identity, bool], who: str) -> bool:
    if attempts is None or not final(attempts)["parsed_ok"]:
        return False  # no output, no SQL: wrong
    if identity not in results:
        raise JudgmentError(f"the per-call evaluation of the {who} has no result for {identity}")
    return results[identity]


def format_validity(calls: List[dict], group: Group = by_call_site) -> Dict[str, Dict[str, Any]]:
    """Per group: invocations and the share whose final attempt parsed."""
    out: Dict[str, Dict[str, Any]] = {}
    for identity, attempts in invocations(calls).items():
        key = group(identity, attempts)
        if key is None:
            continue
        entry = out.setdefault(key, {"n": 0, "valid": 0, "attempts": 0})
        entry["n"] += 1
        entry["valid"] += final(attempts)["parsed_ok"]
        entry["attempts"] += len(attempts)
    for entry in out.values():
        entry["rate"] = entry["valid"] / entry["n"]
    return dict(sorted(out.items()))


def truncation(calls: Iterable[dict]) -> Dict[str, Dict[str, Any]]:
    """Per call site: answers, how many were truncated and how many are unknown, and the rate over
    the known ones (see the module docstring). A call that failed before any answer is not counted."""
    out: Dict[str, Dict[str, Any]] = {}
    for call in calls:
        if call["response_text"] is None:
            continue
        entry = out.setdefault(call["call_site"], {"answers": 0, "truncated": 0, "unknown": 0})
        entry["answers"] += 1
        if "finish_reason" not in call:
            entry["unknown"] += 1
        elif call["finish_reason"] == "length" or not call["response_text"].strip():
            entry["truncated"] += 1
        elif call["finish_reason"] is None:
            entry["unknown"] += 1
    for entry in out.values():
        known = entry["answers"] - entry["unknown"]
        entry["rate"] = entry["truncated"] / known if known else None
    return dict(sorted(out.items()))


def compare(teacher_calls: List[dict], replay_calls: List[dict], replay_eval: Optional[Dict[Identity, bool]] = None,
            teacher_eval: Optional[Dict[Identity, bool]] = None, group: Group = by_call_site,
            call_sites: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Per group: the engine against the teacher on the same invocations (see the module docstring);
    `call_sites`, the replay's declared scope (None: every call site)."""
    teacher, replay = invocations(teacher_calls), invocations(replay_calls)
    if call_sites is not None:
        scope = set(call_sites)
        teacher = {identity: attempts for identity, attempts in teacher.items() if identity[1] in scope}
    extra = set(replay) - set(teacher)
    if extra:
        raise JudgmentError(f"the replay has invocations its source does not (or out of its call sites): {sorted(extra)[:3]}")
    missing = sorted(set(teacher) - set(replay))
    if missing:
        raise JudgmentError(f"the replay lacks {len(missing)} of its source's invocations within its call sites, "
                            f"e.g. {missing[:2]}: a missing answer is not evidence")
    out: Dict[str, Dict[str, Any]] = {}
    for identity in sorted(teacher):
        key = group(identity, teacher[identity])
        if key is None:
            continue
        question_id, call_site, _ = identity
        entry = out.setdefault(key, {"n": 0, "missing_in_replay": 0, "format_valid": 0, "gold": None, "agreement": None})
        entry["n"] += 1
        mine = replay.get(identity)
        entry["missing_in_replay"] += mine is None
        entry["format_valid"] += mine is not None and final(mine)["parsed_ok"]
        if call_site in GOLD_CALL_SITES:
            if replay_eval is None or teacher_eval is None:
                raise JudgmentError(f"{call_site} is judged by execution: pass the per-call evaluations of both runs")
            gold = entry["gold"] = entry["gold"] or {"n": 0, "correct_replay": 0, "correct_teacher": 0,
                                                     "by_question": {"replay": {}, "teacher": {}}}
            engine_ok = _correct(identity, mine, replay_eval, "replay")
            teacher_ok = _correct(identity, teacher[identity], teacher_eval, "teacher")
            gold["n"] += 1
            gold["correct_replay"] += engine_ok
            gold["correct_teacher"] += teacher_ok
            for who, ok in (("replay", engine_ok), ("teacher", teacher_ok)):
                gold["by_question"][who][question_id] = gold["by_question"][who].get(question_id, True) and ok
        else:
            agreement = entry["agreement"] = entry["agreement"] or {"n": 0, "agree": 0, "teacher_unparsed": 0}
            reference_output = final(teacher[identity])
            if not reference_output["parsed_ok"]:
                agreement["teacher_unparsed"] += 1  # no teacher decision to agree with
                continue
            engine_output = final(mine)["parsed_output"] if mine is not None and final(mine)["parsed_ok"] else None
            agreement["n"] += 1
            agreement["agree"] += agree(call_site, reference_output["parsed_output"], engine_output)
    for entry in out.values():
        entry["format_valid_rate"] = entry["format_valid"] / entry["n"]
        if entry["gold"]:
            entry["gold"]["ex_replay"] = entry["gold"]["correct_replay"] / entry["gold"]["n"]
            entry["gold"]["ex_teacher"] = entry["gold"]["correct_teacher"] / entry["gold"]["n"]
            by_question = entry["gold"]["by_question"]
            entry["gold"]["ex_by_question"] = {who: sum(v.values()) / len(v) for who, v in by_question.items()}
        if entry["agreement"]:
            n = entry["agreement"]["n"]
            entry["agreement"]["rate"] = entry["agreement"]["agree"] / n if n else None
    return dict(sorted(out.items()))


def site_score(entry: Dict[str, Any]) -> Optional[float]:
    """One number per call site, the per-call-site score of S4: EX where there is gold, agreement
    elsewhere."""
    if entry["gold"]:
        return entry["gold"]["ex_replay"]
    return entry["agreement"]["rate"] if entry["agreement"] else None


# ---------------------------------------------------------------- reading the executions

def per_call_eval(eval_run_id: str, source_run_id: str) -> Dict[Identity, bool]:
    found = require_done(eval_run_id, type="eval")
    if not found.get("per_call"):
        raise JudgmentError(f"{eval_run_id} is an end-to-end evaluation: J2 needs the per-call one (bench eval --per-call)")
    if found.get("source_run_id") != source_run_id:
        raise JudgmentError(f"{eval_run_id} evaluated {found.get('source_run_id')}, not {source_run_id}")
    return eval_index(read_jsonl(run_dir(eval_run_id) / "results.jsonl"))


def replay_inputs(replay_run_id: str, replay_eval_run_id: Optional[str], teacher_eval_run_id: Optional[str]):
    """(what was read, teacher calls, replay calls, replay per-call eval, teacher per-call eval, replay manifest)."""
    replayed = require_done(replay_run_id, type="replay")
    teacher_run_id = replayed["source_run_id"]
    require_done(teacher_run_id, type="agent", arm="B0", split=replayed.get("split"))
    reads = {"teacher": reference(teacher_run_id), "replay": reference(replay_run_id)}
    replay_eval = teacher_eval = None
    if replay_eval_run_id:
        replay_eval = per_call_eval(replay_eval_run_id, replay_run_id)
        reads["replay_eval"] = reference(replay_eval_run_id)
    if teacher_eval_run_id:
        teacher_eval = per_call_eval(teacher_eval_run_id, teacher_run_id)
        reads["teacher_eval"] = reference(teacher_eval_run_id)
    return reads, calls_of(teacher_run_id), calls_of(replay_run_id), replay_eval, teacher_eval, replayed


def judge_replay(replay_run_id: str, replay_eval_run_id: Optional[str], teacher_eval_run_id: Optional[str]):
    reads, teacher, replay, replay_eval, teacher_eval, replayed = replay_inputs(
        replay_run_id, replay_eval_run_id, teacher_eval_run_id)
    return reads, {"mode": "replay", "split": replayed.get("split"),
                   "engine": replayed.get("engine"), "arm": replayed.get("arm"), "call_sites": replayed.get("call_sites"),
                   "per_call_site": compare(teacher, replay, replay_eval, teacher_eval,
                                            call_sites=replayed.get("call_sites"))}


def judge_run(run_id: str):
    """Format validity and truncation of one execution (an agent arm, or a replay) by call site."""
    found = require_done(run_id, type=("agent", "replay"))
    calls = calls_of(run_id)
    return {"run": reference(run_id)}, {"mode": "run", "split": found.get("split"), "arm": found.get("arm"),
                                         "engine": found.get("engine"),
                                         "format_validity": format_validity(calls), "truncation": truncation(calls)}


def run(run_id: Optional[str] = None, replay_run_id: Optional[str] = None, replay_eval_run_id: Optional[str] = None,
        teacher_eval_run_id: Optional[str] = None):
    """The J2 result of one execution (`run_id`), or of a replay against its teacher. Returns its path."""
    reads, result = (judge_run(run_id) if run_id else
                     judge_replay(replay_run_id, replay_eval_run_id, teacher_eval_run_id))
    return write_result(JUDGMENT, {**reads, "config": list(CONFIG_KEYS)}, result)


def engine_of(run_id: str) -> str:
    found = run_manifest(run_id)
    if not found.get("engine"):
        raise JudgmentError(f"{run_id} was not a replay on one fixed engine")
    return found["engine"]
