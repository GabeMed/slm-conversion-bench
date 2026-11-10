"""J3 · cost (SPEC §6.6): `calls.jsonl` × the dated price table, per call, per question and per
correct query.

- **API engines** (`production_llm`, `cheap_alt`): each call's `usage` × `prices.table[<model>]`,
  the model as C1 records it: uncached input, cached input and output tokens, each at its price per
  million. A price is one provider's: the entry carries `provider`, and a call whose C1 `provider`
  differs from it is refused, never priced at another provider's price (design §6.3, T6). An entry
  that names no provider prices nothing.
  Variants are recomputed from the same `usage`, never by running again: `standard`,
  `no_cache` (cached tokens at the input price) and `batch` (standard less the entry's
  `batch_discount`, when the table has one). Retries are calls and are priced (SPEC §2).
- **`usage.source`**: `missing` on a call that returned a response is refused (the cost would be
  invented). A call that failed before any response (a timeout, an HTTP error) has no usage by
  construction (C1): it is counted as `failed_unbilled` and not priced, and the result is marked a
  `lower_bound`, since a provider may bill such a call; refusing it would leave the arm with no cost
  at all. `estimated` usage is priced and the result is labelled `estimated`. A call with
  `source: api` and `cached_input: null` had its cache **not reported**: it is priced with no cache
  discount, so the arm's cost is an `upper_bound` (`cache_not_reported` counts them).
- **The SLM** has no token price: each SLM call costs a load test's cost per request (J8) at each
  utilization J8 gives, so an arm with SLM calls has one total per API variant and utilization
  (`standard@50%`). Which load test, by arm: **B4** at its as-deployed sweep (one server, every adapter, B4's
  own mix: `measured`); **B3** at the base's (`measured`); **B5** (and any other arm) each call at its own
  engine's sweep, the adapter of its cluster (`extrapolated from per-engine load tests`: B5 serves several
  adapters on one server, which a sweep of one adapter does not measure). J8 must have measured every engine
  needed. An engine with no load level within its latency limit (J8's `outside_slo`) has no cost: every total
  its calls fall in is null, and `slm_cost_basis.outside_slo` names it. Where J8 gives the cost at the
  sensitivity price too (Modal's), every SLM scenario has its twin at that price (`standard@50%@sensitivity`).
- **Per correct query**: the total over the execution divided by its correct questions, read from
  its `eval` execution (J1's source).
- **Per question** (`by_question`: {question id: {scenario: cost}}): the cost of each question's calls,
  in every scenario of `total`; a question without calls costs 0. The report resamples it, paired by
  question, for the confidence interval of a cost ratio (design §6.2, T9).
- **Replaceable fraction** (SPEC §5), for an execution with SLM calls: the share of calls the SLM
  served (failed calls counted on both sides), of tokens (billed calls on both sides, a failed call
  has none), and of cost, as the share of the production LLM's price for all the execution's tokens
  that fell on SLM calls. The result records the price table it used (`as_of` and sha256).
"""
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from bench.judge.base import (JudgmentError, calls_of, canonical, read_jsonl, read_result, reference, require_done,
                              result_reference, run_dir, write_result)

JUDGMENT = "J3"
CONFIG_KEYS = ["prices", "roles.production_llm.model"]  # the keys of config.yaml this judgment reads (design §6.2)
API_VARIANTS = ("standard", "no_cache", "batch")


def utilization_label(u: float) -> str:
    return f"{round(u * 100)}%"


def tokens(call: dict) -> Tuple[int, int, int, int]:
    """(uncached input, cached input, input written to the cache, output). `cache_write` is reported by a
    provider that bills writing the cache apart (Anthropic); absent, there is none."""
    use = call["usage"]
    cached, written = use["cached_input"] or 0, use.get("cache_write") or 0
    return use["input"] - cached - written, cached, written, use["output"]


def api_cost(call: dict, entry: Dict[str, Any]) -> Dict[str, Optional[float]]:
    uncached, cached, written, output = tokens(call)
    if written and entry.get("cache_write_per_mtok") is None:
        raise JudgmentError(f"call {call['call_id']} wrote {written} tokens to the cache and the price entry of "
                            f"{call['model']!r} has no cache_write_per_mtok: a cache write is billed at its own price")
    per = 1_000_000
    standard = (uncached * entry["input_per_mtok"] + cached * entry["cached_input_per_mtok"]
                + written * (entry.get("cache_write_per_mtok") or 0) + output * entry["output_per_mtok"]) / per
    discount = entry.get("batch_discount")
    return {"standard": standard,
            "no_cache": ((uncached + cached + written) * entry["input_per_mtok"] + output * entry["output_per_mtok"]) / per,
            "batch": standard * (1 - discount) if discount is not None else None}


def price_calls(calls: List[dict], prices: Dict[str, Any],
                slm_per_request: Optional[Dict[str, Dict[str, Optional[float]]]] = None) -> Dict[str, Any]:
    """Totals over `calls`: counts, tokens, API cost per variant, SLM cost per utilization. `slm_per_request`
    is {engine: {label: cost per request}}, each SLM call priced at its engine's; a null cost (an engine
    outside its latency limit) makes every total it falls in null."""
    table, api = prices.get("table") or {}, {v: 0.0 for v in API_VARIANTS}
    labels = sorted({u for costs in (slm_per_request or {}).values() for u in costs})
    slm: Dict[str, Optional[float]] = {u: 0.0 for u in labels}
    counts = {"calls": 0, "slm_calls": 0, "failed_unbilled": 0, "estimated": 0, "cache_not_reported": 0}
    token_totals = {"input": 0, "cached_input": 0, "cache_write": 0, "output": 0}
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
        uncached, cached, written, output = tokens(call)
        token_totals["input"] += uncached + cached + written
        token_totals["cached_input"] += cached
        token_totals["cache_write"] += written
        token_totals["output"] += output
        if call["model_role"] == "slm":
            if slm_per_request is None:
                raise JudgmentError("the calls include SLM calls: pass the load test's cost per request (J8)")
            if call["engine"] not in slm_per_request:
                raise JudgmentError(f"call {call['call_id']} ran on {call['engine']}, which the load test did not price")
            counts["slm_calls"] += 1
            for u, cost in slm_per_request[call["engine"]].items():
                slm[u] = None if slm[u] is None or cost is None else slm[u] + cost
            continue
        if not prices.get("as_of"):
            raise JudgmentError("prices.as_of is not set: an undated price table prices nothing")
        entry = table.get(call["model"])
        if entry is None:
            raise JudgmentError(f"prices.table has no entry for model {call['model']!r}")
        if entry.get("provider") is None or call.get("provider") != entry["provider"]:
            raise JudgmentError(f"call {call['call_id']} was served by provider {call.get('provider')!r}, and the price "
                                f"entry of {call['model']!r} is of provider {entry.get('provider')!r}: a price is one provider's")
        counts["cache_not_reported"] += use["source"] == "api" and use["cached_input"] is None
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
            out[f"{variant}@{u}"] = None if api is None or slm is None else api + slm
    return out


def replaceable_fraction(calls: List[dict], prices: Dict[str, Any], production_model: str) -> Optional[Dict[str, Any]]:
    billed = [c for c in calls if c["usage"]["source"] != "missing"]
    slm_calls = [c for c in calls if c["model_role"] == "slm"]
    slm = [c for c in billed if c["model_role"] == "slm"]
    if not slm_calls:
        return None
    entry = (prices.get("table") or {}).get(production_model)

    def at_production(subset: List[dict]) -> Optional[float]:
        return sum(api_cost(c, entry)["standard"] for c in subset) if entry else None

    def total_tokens(subset: List[dict]) -> int:
        return sum(c["usage"]["input"] + c["usage"]["output"] for c in subset)
    whole, part = at_production(billed), at_production(slm)
    return {"calls": len(slm_calls) / len(calls), "tokens": total_tokens(slm) / total_tokens(billed) if billed else None,
            "cost_at_production_price": part / whole if whole else None}


def judge(calls: List[dict], question_ids: List[str], correct: Optional[Dict[str, bool]], prices: Dict[str, Any],
          production_model: str, slm_per_request: Optional[Dict[str, Dict[str, Optional[float]]]] = None) -> Dict[str, Any]:
    priced = price_calls(calls, prices, slm_per_request)
    totals = scenarios(priced)
    n_correct = sum(correct.values()) if correct is not None else None
    by_site: Dict[str, Dict[str, Any]] = {}
    of_question: Dict[str, List[dict]] = {q: [] for q in question_ids}
    for call in calls:
        by_site.setdefault(call["call_site"], []).append(call)
        if call["question_id"] not in of_question:
            raise JudgmentError(f"call {call['call_id']} is of question {call['question_id']}, which the execution does not list")
        of_question[call["question_id"]].append(call)
    return {
        "prices_as_of": prices.get("as_of"), "label": "estimated" if priced["estimated"] else "measured",
        "prices": {"as_of": prices.get("as_of"), "sha256": hashlib.sha256(canonical(prices.get("table") or {})).hexdigest()},
        "lower_bound": priced["failed_unbilled"] > 0,
        "upper_bound": priced["cache_not_reported"] > 0, "cache_not_reported": priced["cache_not_reported"],
        "n_questions": len(question_ids), "n_correct": n_correct, **{k: priced[k] for k in
                                                                    ("calls", "slm_calls", "failed_unbilled", "estimated", "tokens")},
        "total": totals,
        "per_call": {k: v / priced["calls"] if v is not None and priced["calls"] else None for k, v in totals.items()},
        "per_question": {k: v / len(question_ids) if v is not None and question_ids else None for k, v in totals.items()},
        "per_correct": {k: v / n_correct if v is not None and n_correct else None for k, v in totals.items()},
        "by_call_site": {site: price_calls(site_calls, prices, slm_per_request) for site, site_calls in sorted(by_site.items())},
        "by_question": {q: scenarios(price_calls(mine, prices, slm_per_request)) for q, mine in of_question.items()},
        "replaceable_fraction": replaceable_fraction(calls, prices, production_model),
    }


# ---------------------------------------------------------------- reading the executions

def engines_used(calls: List[dict]) -> set:
    return {c["engine"] for c in calls if c["model_role"] == "slm"}


def engine_costs(entry: Dict[str, Any], labels: List[str]) -> Dict[str, Optional[float]]:
    """One J8 engine's cost per request at each label, and at the sensitivity price (`<label>@sensitivity`)
    where J8 gives it; all null for an engine outside its latency limit."""
    out = {u: (entry.get("cost_per_request") or {}).get(u) for u in labels}
    if "cost_per_request_sensitivity" in entry:
        out.update({f"{u}@sensitivity": (entry["cost_per_request_sensitivity"] or {}).get(u) for u in labels})
    return out


def slm_cost_per_request(j8_path: Optional[str], calls: List[dict], arm: Optional[str]
                         ) -> Tuple[Optional[Dict[str, Dict[str, Optional[float]]]], Dict[str, Any]]:
    """{engine as C1 records it: {label: cost per request}} for the SLM engines the calls used, by the arm's
    rule (module docstring), and what was read."""
    engines = {c["engine"].split("+lora:")[0] for c in calls if c["model_role"] == "slm"}
    if not engines:
        return None, {}
    if not j8_path:
        raise JudgmentError(f"the execution has SLM calls ({sorted(engines)}): pass the J8 result of their load test")
    j8 = read_result(j8_path, "J8")["result"]
    if engines != {j8["engine"].split("+", 1)[0]}:
        raise JudgmentError(f"the load test measured {j8['engine']!r}, the calls used {sorted(engines)}")
    labels = j8.get("utilizations") or []
    used = sorted(engines_used(calls))
    if arm == "B4":
        deployed = (j8.get("as_deployed") or {}).get(f"{j8['engine']}+as-deployed")
        if deployed is None:
            raise JudgmentError("B4 is priced at its as-deployed sweep, which the load test does not have")
        cost = {engine: engine_costs(deployed, labels) for engine in used}
        basis, rule = "measured", "every SLM call at B4's as-deployed sweep (one server, every adapter, B4's mix)"
        outside = [f"{j8['engine']}+as-deployed"] if deployed["outside_slo"] else []
    else:
        per_engine = j8.get("engines") or {}
        uncovered = [engine for engine in used if engine not in per_engine]
        if uncovered:
            raise JudgmentError(f"the load test did not measure every SLM engine the calls used: {uncovered}")
        cost = {engine: engine_costs(per_engine[engine], labels) for engine in used}
        measured = len(used) == 1 and "+lora:" not in used[0]
        basis = "measured" if measured else "extrapolated from per-engine load tests"
        rule = "each SLM call at its own engine's sweep"
        outside = [engine for engine in used if per_engine[engine]["outside_slo"]]
    return cost, {"j8": result_reference(j8_path), "slm_cost_basis": {
        "basis": basis, "rule": rule, "call_engines": used, "outside_slo": outside,
        "sensitivity": (j8.get("sensitivity") or {}).get("provider"), "combined": j8.get("combined")}}


def run(run_id: str, eval_run_id: Optional[str], j8_path: Optional[str], config: Dict[str, Any]):
    found = require_done(run_id, type=("agent", "replay"))
    calls = calls_of(run_id)
    reads: Dict[str, Any] = {"run": reference(run_id), "config": CONFIG_KEYS}
    correct = None
    if eval_run_id:
        evaluated = require_done(eval_run_id, type="eval")
        if evaluated.get("per_call"):
            raise JudgmentError(f"{eval_run_id} is a per-call evaluation: the cost per correct query needs the end-to-end one")
        if evaluated.get("source_run_id") != run_id:
            raise JudgmentError(f"{eval_run_id} evaluated {evaluated.get('source_run_id')}, not {run_id}")
        correct = {r["question_id"]: bool(r["correct"]) for r in read_jsonl(run_dir(eval_run_id) / "results.jsonl")}
        reads["eval"] = reference(eval_run_id)
    slm, j8_read = slm_cost_per_request(j8_path, calls, found.get("arm"))
    basis = j8_read.pop("slm_cost_basis", None)
    reads.update(j8_read)
    question_ids = found.get("question_ids") or sorted({c["question_id"] for c in calls})
    result = {"arm": found.get("arm"), "engine": found.get("engine"), "split": found.get("split"), "slm_cost_basis": basis,
              **judge(calls, question_ids, correct, config["prices"], config["roles"]["production_llm"]["model"], slm)}
    return write_result(JUDGMENT, reads, result)
