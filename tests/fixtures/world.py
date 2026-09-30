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
    write_run(eval_run_id, {"type": "eval", "source_run_id": source_run_id, "per_call": True},
              files={"results.jsonl": rows})
    return eval_run_id


def fake_noninferiority(correct_a: Dict[str, bool], correct_b: Dict[str, bool], delta_cap_pp: float, seed: int,
                        n_boot: int) -> Dict[str, Any]:
    """A stand-in for F2's J4 with its signature: b against a, paired by question id, a normal
    approximation instead of the bootstrap. Proportions, with Δ compared to the cap in pp."""
    shared = sorted(set(correct_a) & set(correct_b))
    n = len(shared)
    diff = sum(correct_b[q] - correct_a[q] for q in shared) / n
    d = sum(correct_a[q] != correct_b[q] for q in shared) / n
    delta = (1.6449 + 0.8416) * math.sqrt(d / n)
    ci_low = diff - 1.6449 * math.sqrt(d / n)
    return {"d": d, "delta": delta, "diff": diff, "ci_low": ci_low, "noninferior": ci_low >= -delta,
            "power": 0.8, "testable": delta * 100 <= delta_cap_pp}
