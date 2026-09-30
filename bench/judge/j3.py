"""J3 · cost (SPEC §6.6): `calls.jsonl` × the dated price table, per call, per question and per
correct query.

- **API engines** (`production_llm`, `cheap_alt`): each call's `usage` × `prices.table[<model>]`,
  the model as C1 records it: uncached input, cached input and output tokens, each at its price per
  million. Variants are recomputed from the same `usage`, never by running again: `standard`,
  `no_cache` (cached tokens at the input price) and `batch` (standard less the entry's
  `batch_discount`, when the table has one). Retries are calls and are priced (SPEC §2).
- **`usage.source`**: `missing` on a call that returned a response is refused (the cost would be
  invented). A call that failed before any response (a timeout, an HTTP error) has no usage by
  construction (C1): it is counted as `failed_unbilled` and not priced, and the result is marked a
  `lower_bound`, since a provider may bill such a call; refusing it would leave the arm with no cost
  at all. `estimated` usage is priced and the result is labelled `estimated`.
- **The SLM** has no token price: each SLM call costs the load test's cost per request (J8) at each
  utilization of `cost.utilizations`, so an arm with SLM calls has one total per API variant and
  utilization (`standard@50%`).
- **Per correct query**: the total over the execution divided by its correct questions, read from
  its `eval` execution (J1's source).
- **Replaceable fraction** (SPEC §5), for an execution with SLM calls: the share of calls and of
  tokens the SLM served, and of cost, as the share of the production LLM's price for all the
  execution's tokens that fell on SLM calls.
"""
from typing import Any, Dict, List, Optional, Tuple

from bench.judge.base import (JudgmentError, calls_of, read_jsonl, read_result, reference, require_done,
                              result_reference, run_dir, write_result)

JUDGMENT = "J3"
API_VARIANTS = ("standard", "no_cache", "batch")


def utilization_label(u: float) -> str:
    return f"{round(u * 100)}%"


def tokens(call: dict) -> Tuple[int, int, int]:
    """(uncached input, cached input, output)."""
    use = call["usage"]
    cached = use["cached_input"] or 0
    return use["input"] - cached, cached, use["output"]


def api_cost(call: dict, entry: Dict[str, Any]) -> Dict[str, Optional[float]]:
    uncached, cached, output = tokens(call)
    per = 1_000_000
    standard = (uncached * entry["input_per_mtok"] + cached * entry["cached_input_per_mtok"]
                + output * entry["output_per_mtok"]) / per
    discount = entry.get("batch_discount")
    return {"standard": standard,
            "no_cache": ((uncached + cached) * entry["input_per_mtok"] + output * entry["output_per_mtok"]) / per,
            "batch": standard * (1 - discount) if discount is not None else None}


def price_calls(calls: List[dict], prices: Dict[str, Any], slm_per_request: Optional[Dict[str, float]] = None
                ) -> Dict[str, Any]:
    """Totals over `calls`: counts, tokens, API cost per variant, SLM cost per utilization."""
    table, api = prices.get("table") or {}, {v: 0.0 for v in API_VARIANTS}
    slm: Dict[str, float] = {u: 0.0 for u in (slm_per_request or {})}
    counts = {"calls": 0, "slm_calls": 0, "failed_unbilled": 0, "estimated": 0}
    token_totals = {"input": 0, "cached_input": 0, "output": 0}
    batch_known = True
    for call in calls:
        counts["calls"] += 1
        use = call["usage"]
        if use["source"] == "missing":
            if call["response_text"] is not None:
                raise JudgmentError(f"call {call['call_id']} has a response but no usage: it cannot be priced")
            counts["failed_unbilled"] += 1
            continue
        counts["estimated"] += use["source"] == "estimated"
        uncached, cached, output = tokens(call)
        token_totals["input"] += uncached + cached
        token_totals["cached_input"] += cached
        token_totals["output"] += output
        if call["model_role"] == "slm":
            if slm_per_request is None:
                raise JudgmentError("the calls include SLM calls: pass the load test's cost per request (J8)")
            counts["slm_calls"] += 1
            for u, cost in slm_per_request.items():
                slm[u] += cost
            continue
        if not prices.get("as_of"):
            raise JudgmentError("prices.as_of is not set: an undated price table prices nothing")
        entry = table.get(call["model"])
        if entry is None:
            raise JudgmentError(f"prices.table has no entry for model {call['model']!r}")
        for variant, cost in api_cost(call, entry).items():
            if cost is None:
                batch_known = False
            else:
                api[variant] += cost
    if not batch_known:
        api["batch"] = None
    return {**counts, "tokens": token_totals, "api": api, "slm": slm}


def scenarios(priced: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Total cost per scenario: an API variant, and with SLM calls, a utilization too."""
    out = {}
    for variant, api in priced["api"].items():
        if not priced["slm"]:
            out[variant] = api
        for u, slm in priced["slm"].items():
            out[f"{variant}@{u}"] = None if api is None else api + slm
    return out


def replaceable_fraction(calls: List[dict], prices: Dict[str, Any], production_model: str) -> Optional[Dict[str, Any]]:
    billed = [c for c in calls if c["usage"]["source"] != "missing"]
    slm = [c for c in billed if c["model_role"] == "slm"]
    if not slm:
        return None
    entry = (prices.get("table") or {}).get(production_model)

    def at_production(subset: List[dict]) -> Optional[float]:
        return sum(api_cost(c, entry)["standard"] for c in subset) if entry else None

    def total_tokens(subset: List[dict]) -> int:
        return sum(c["usage"]["input"] + c["usage"]["output"] for c in subset)
    whole, part = at_production(billed), at_production(slm)
    return {"calls": len(slm) / len(calls), "tokens": total_tokens(slm) / total_tokens(billed),
            "cost_at_production_price": part / whole if whole else None}


def judge(calls: List[dict], question_ids: List[str], correct: Optional[Dict[str, bool]], prices: Dict[str, Any],
          production_model: str, slm_per_request: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    priced = price_calls(calls, prices, slm_per_request)
    totals = scenarios(priced)
    n_correct = sum(correct.values()) if correct is not None else None
    by_site: Dict[str, Dict[str, Any]] = {}
    for call in calls:
        by_site.setdefault(call["call_site"], []).append(call)
    return {
        "prices_as_of": prices.get("as_of"), "label": "estimated" if priced["estimated"] else "measured",
        "lower_bound": priced["failed_unbilled"] > 0,
        "n_questions": len(question_ids), "n_correct": n_correct, **{k: priced[k] for k in
                                                                    ("calls", "slm_calls", "failed_unbilled", "estimated", "tokens")},
        "total": totals,
        "per_call": {k: v / priced["calls"] if v is not None and priced["calls"] else None for k, v in totals.items()},
        "per_question": {k: v / len(question_ids) if v is not None and question_ids else None for k, v in totals.items()},
        "per_correct": {k: v / n_correct if v is not None and n_correct else None for k, v in totals.items()},
        "by_call_site": {site: price_calls(site_calls, prices, slm_per_request) for site, site_calls in sorted(by_site.items())},
        "replaceable_fraction": replaceable_fraction(calls, prices, production_model),
    }


# ---------------------------------------------------------------- reading the executions

def slm_cost_per_request(j8_path: Optional[str], calls: List[dict]) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
    engines = {c["engine"].split("+lora:")[0] for c in calls if c["model_role"] == "slm"}
    if not engines:
        return None, {}
    if not j8_path:
        raise JudgmentError(f"the execution has SLM calls ({sorted(engines)}): pass the J8 result of their load test")
    j8 = read_result(j8_path, "J8")
    if engines != {j8["result"]["engine"].split("+lora:")[0]}:
        raise JudgmentError(f"the load test measured {j8['result']['engine']!r}, the calls used {sorted(engines)}")
    return j8["result"]["cost_per_request"], {"j8": result_reference(j8_path)}


def run(run_id: str, eval_run_id: Optional[str], j8_path: Optional[str], config: Dict[str, Any]):
    found = require_done(run_id, type=("agent", "replay"))
    calls = calls_of(run_id)
    reads: Dict[str, Any] = {"run": reference(run_id)}
    correct = None
    if eval_run_id:
        evaluated = require_done(eval_run_id, type="eval")
        if evaluated.get("per_call"):
            raise JudgmentError(f"{eval_run_id} is a per-call evaluation: the cost per correct query needs the end-to-end one")
        if evaluated.get("source_run_id") != run_id:
            raise JudgmentError(f"{eval_run_id} evaluated {evaluated.get('source_run_id')}, not {run_id}")
        correct = {r["question_id"]: bool(r["correct"]) for r in read_jsonl(run_dir(eval_run_id) / "results.jsonl")}
        reads["eval"] = reference(eval_run_id)
    slm, j8_read = slm_cost_per_request(j8_path, calls)
    reads.update(j8_read)
    question_ids = found.get("question_ids") or sorted({c["question_id"] for c in calls})
    result = {"arm": found.get("arm"), "engine": found.get("engine"), "split": found.get("split"),
              **judge(calls, question_ids, correct, config["prices"], config["roles"]["production_llm"]["model"], slm)}
    return write_result(JUDGMENT, reads, result)
