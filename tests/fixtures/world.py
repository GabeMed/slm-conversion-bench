"""A synthetic benchmark world: the teacher (B0) on a split, replays of its invocations on other
engines with a chosen quality, per-call evaluations, and a J4 stand-in. Deterministic."""
import hashlib
import math
from typing import Any, Callable, Dict, List, Optional

from fixtures.fake import call, prompt, usage, write_run

GOLD_SITES = ("generate_candidate", "revise")


def unit(*parts: Any) -> float:
    """A deterministic number in [0, 1) for these parts."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16 ** 12


def invocations_of(question_id: str) -> List[tuple]:
    """(call_site, invocation_key, teacher's parsed output) of one question, CHESS-shaped."""
    rows = [("agent_ir", "ir:0", {"tool": "extract_keywords"}),
            ("extract_keywords", "single", ["alpha", "beta"]),
            ("filter_column", "t.a", {"chain_of_thought_reasoning": "x", "is_column_information_relevant": "Yes"}),
            ("filter_column", "t.b", {"chain_of_thought_reasoning": "x", "is_column_information_relevant": "No"}),
            ("select_tables", "single", {"chain_of_thought_reasoning": "x", "table_names": ["t"]}),
            ("generate_candidate", "generate_candidate_one:0", {"SQL": f"SELECT {question_id}", "plan": ""})]
    if int(question_id) % 3 == 0:
        rows.append(("revise", "revise_1:0", {"refined_sql_query": f"SELECT {question_id} -- fixed"}))
    return rows


def other(site: str, output: Any) -> Any:
    """An output of the same call site with a different decision."""
    if site == "extract_keywords":
        return output + ["gamma"]
    if site == "filter_column":
        flip = "No" if output["is_column_information_relevant"] == "Yes" else "Yes"
        return {**output, "is_column_information_relevant": flip}
    if site == "select_tables":
        return {**output, "table_names": ["u"]}
    if site.startswith("agent_"):
        return {"done": True}
    return output


def teacher(run_id: str, split: str, questions: List[str], model: str = "teacher-model") -> List[dict]:
    return [call(run_id, q, site, key, messages=prompt(site, f"question {q}", key), response=str(output),
                 parsed=output, model=model, use=usage(1000 + 100 * i, 200, 40))
            for q in questions for i, (site, key, output) in enumerate(invocations_of(q))]


def replay(source: List[dict], run_id: str, engine: str, quality: float, *, role: Optional[str] = None,
           model: str = "engine-model", cluster_of: Optional[Callable[[dict], str]] = None,
           fail_rate: float = 0.0) -> List[dict]:
    """The source's invocations answered by `engine`: the teacher's decision with probability
    `quality`, another one otherwise, and no parse with probability `fail_rate`."""
    role = role or ("slm" if engine.startswith("slm:") else engine)
    out = []
    for c in source:
        u = unit(engine, c["question_id"], c["call_site"], c["invocation_key"])
        failed = unit("fail", engine, c["question_id"], c["call_site"], c["invocation_key"]) < fail_rate
        output = c["parsed_output"] if u < quality else other(c["call_site"], c["parsed_output"])
        out.append(call(run_id, c["question_id"], c["call_site"], c["invocation_key"], messages=c["prompt_messages"],
                        response=None if failed else str(output), parsed=None if failed else output,
                        parsed_ok=not failed, role=role, engine=engine, model=model,
                        cluster=cluster_of(c) if cluster_of else None,
                        use=usage(source="missing") if failed else usage(900, 0, 30)))
    return out


def gold_correct(run_id_seed: str, quality: float) -> Callable[[dict], bool]:
    return lambda c: unit("gold", run_id_seed, c["question_id"], c["call_site"], c["invocation_key"]) < quality


def per_call_eval(eval_run_id: str, source_run_id: str, calls: List[dict], correct: Callable[[dict], bool]) -> str:
    rows = [{"question_id": c["question_id"], "call_site": c["call_site"], "invocation_key": c["invocation_key"],
             "correct": bool(c["parsed_ok"] and correct(c))}
            for c in calls if c["call_site"] in GOLD_SITES and c["parsed_ok"]]
    write_run(eval_run_id, {"type": "eval", "source_run_id": source_run_id, "per_call": True, "status": None},
              files={"results.jsonl": rows})
    return eval_run_id


def fake_noninferiority(correct_a: Dict[str, bool], correct_b: Dict[str, bool], delta_cap_pp: float, seed: int,
                        n_boot: int, *, d_pilot: Optional[float] = None, margin: Optional[float] = None) -> Dict[str, Any]:
    """A stand-in for F2's J4 with its contract (bench/judge/j4.py on F2's branch): A, the
    candidate, against B, the reference, on the same questions; diff = EX_A − EX_B; Δ from
    `d_pilot` (or from these pairs, with no verdict); non-inferior when the lower bound is above −Δ.
    A normal approximation stands in for the bootstrap."""
    from bench.judge import JudgeError
    if set(correct_a) != set(correct_b) or not correct_a:
        raise JudgeError("the two arms must be scored on the same questions, and on at least one")
    if d_pilot is not None and margin is not None:
        raise JudgeError("give the pilot discordance or the margin, not both")
    if margin is not None and not 0 <= margin <= delta_cap_pp / 100:
        raise JudgeError(f"margin must be within [0, {delta_cap_pp / 100}]")
    ids = sorted(correct_a)
    diffs = [int(correct_a[q]) - int(correct_b[q]) for q in ids]
    n = len(ids)
    d = sum(x != 0 for x in diffs) / n
    if margin is not None:  # used as given, testable by construction (F2's J4)
        delta, testable, margin_from = margin, True, "given"
    else:
        delta = (1.6449 + 0.8416) * math.sqrt((d if d_pilot is None else d_pilot) / n)
        testable, margin_from = delta * 100 <= delta_cap_pp, "pairs" if d_pilot is None else "pilot"
    ci_low = sum(diffs) / n - 1.6449 * math.sqrt(d / n)
    return {"n": n, "d": d, "d_pilot": d_pilot, "margin_from": margin_from,
            "delta": delta, "testable": testable, "diff": sum(diffs) / n, "ci_low": ci_low,
            "noninferior": ci_low > -delta if testable and margin_from != "pairs" else None, "power": 0.8}


def fake_margin(d: float, n: int, delta_cap_pp: float) -> Dict[str, Any]:
    """A stand-in for F2's J4 `margin`: Δ for a discordance d over n paired questions."""
    delta = (1.6449 + 0.8416) * math.sqrt(d / n)
    return {"delta": delta, "testable": delta * 100 <= delta_cap_pp}


def fake_ex_table(eval_run_dirs) -> List[Dict[str, Any]]:
    """A stand-in for F2's J1 `ex_table`: one row per question of each end-to-end eval execution."""
    import json
    from pathlib import Path
    rows = []
    for run_dir in map(Path, eval_run_dirs):
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["type"] == "eval" and not manifest.get("per_call")
        for line in (run_dir / "results.jsonl").read_text().splitlines():
            r = json.loads(line)
            rows.append({"arm": manifest["arm"], "engine": manifest.get("engine"), "split": manifest["split"],
                         "question_id": r["question_id"], "difficulty": r["difficulty"], "correct": bool(r["correct"]),
                         "source_run_id": manifest["source_run_id"]})
    return rows


def fake_ex_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A stand-in for F2's J1 `ex_summary`: EX per (arm, engine, split), with its difficulties."""
    def ex(group):
        return {"n": len(group), "correct": sum(r["correct"] for r in group),
                "ex": sum(r["correct"] for r in group) / len(group) if group else None}
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["arm"], row["engine"] or "", row["split"]), []).append(row)
    return [{"arm": arm, "engine": engine or None, "split": split, **ex(group),
             "by_difficulty": {d: ex([r for r in group if r["difficulty"] == d]) for d in sorted({r["difficulty"] for r in group})}}
            for (arm, engine, split), group in sorted(groups.items())]
