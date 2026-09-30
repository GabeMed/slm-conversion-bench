"""R · the report (SPEC §8), from the judgments J1–J8 and the test registry only.

`bench report --plan <plan.yaml>` reads a plan that names, and only names, what the report is made
of (no number goes in it):

    split: test
    arms:       {<arm>: {eval: <eval execution>, cost: <its J3 result>}}      # B0, B1, B2-production, B2-cheap, B3, B4, B5
    pilot:      {B0: <eval execution on calib>, B3: <eval execution on calib>}   # the same pilot questions
    format:     {B0: <J2 result, format validity of its run>, B4: <...>}
    per_call:   <J2 result of the test inputs replayed as B4>
    j5: <J5 result>   j6: <J6 result>   j7: <J7 result>   j8: <J8 result>
    teacher_train_cost: <J3 result of the teacher on train>

Every figure is read from a judgment or computed by one: J1's per-question table and summary
(`ex_table`, `ex_summary`) and J4's tests (`noninferiority`, the candidate first, the reference
second), run here through F2's functions. **Every J4 margin comes from the pilot** (SPEC §6.4,
§7.3), measured on calib before training, between the SLM and the production LLM: `d_pilot` is the
discordance J4 measures between B3 (the zero-shot SLM) and B0 on the pre-registered pilot questions
(`bench.data.pilot_sample`), one number for every end-to-end comparison; for repair, between the
zero-shot candidate S4 chose and the teacher on the pilot questions' repair calls (J6's result).
J4 then takes Δ at n = the test's pairs. A comparison with no pilot gets no verdict (J4 returns
`noninferior: None`), never a pass. The test
registry is read from git as F1 records it: `registry/test/<run_id>.intent.json` before a test
execution starts and `<run_id>.manifest.json` when it ends, as committed at HEAD; an intent with no
manifest is an interrupted execution.

Writes `reports/<sha256>/`: `report.md`, `report.json` (every number, with the judgments it came
from; its sha256 names the directory) and `ex_cost.svg`, the main chart.

**How each row of the SPEC §5 map is decided** (the SPEC fixes the criteria; where it leaves a
term open, the reading is stated here and in the row). A J4 comparison has one of five outcomes,
shared by every row: no pilot (no verdict), not testable (Δ above the cap), non-inferior, worse
(the difference is below −Δ) or inconclusive (neither).
- V1/A1: confirms if B4 or B5 is non-inferior to B0; refutes if both are worse; otherwise the
  outcomes say why not (no pilot, not testable, inconclusive).
- A4/A11: the replaceable fraction of B5 by call, token and cost; the SPEC says "high" without a
  number, so the row is descriptive.
- Appendix B: on the per-call-site evaluation of B4 on the test inputs. The routine (the call sites
  the paper assigns to SLMs: keywords, column filter, table and column selection) passes when each
  agrees with the teacher at `thresholds.concordance_min` or more: the agreement proxy, which D15
  says supports no per-cluster claim, so the verdict says it rests on it. Refutes when the
  routine fails, or when repair is non-inferior (a tie: the partition is conservative); confirms
  when repair is worse and the routine passes; otherwise repair's outcome says why not.
- A5: format validity per call site (SPEC §6.3), B4 against B0: confirms when B4 is at least B0's
  on every call site both have, refutes when it is below on any.
- A6, V3/A2, AV2: per utilization of the SLM. "The best arm without training" is the cheapest per
  correct query among B0, B1 and B2 that is non-inferior to B0 (B0 always is). A6 refutes if B5 is
  not cheaper than it and confirms if B5 is cheaper and non-inferior to B0. V3 compares it with the
  cheapest of B4 and B5 that are non-inferior to B0: confirms at `claims.v3_min_ratio` times cheaper
  or more, refutes below. AV2 wins (the paper is refuted) when B1 costs no more per correct query
  than the cheaper of B4 and B5; the fixed cost and its payback (SPEC §6.6) are not a J1–J8 output
  and are left out, which the row says.
- A7, A2/A3, B2: not measured by J1–J8 in the core; stated as such.
"""
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from bench import paths
from bench.provenance import scrub
from bench.judge.base import (JudgmentError, canonical, manifest, n_boot, read_result, reference, result_reference,
                              run_dir)

UNTRAINED = ("B0", "B1", "B2-production", "B2-cheap")
TRAINED = ("B4", "B5")
SLM_ARMS = ("B3", "B4", "B5")
ARMS = ("B0", "B1", "B2-production", "B2-cheap", "B3", "B4", "B5")
# how J1 names an arm of the plan: (arm, engine); B2 is one arm with two single-call engines
J1_KEY = {arm: (arm, None) for arm in ARMS} | {"B2-production": ("B2", "production_llm"), "B2-cheap": ("B2", "cheap_alt")}
ROUTINE = ("extract_keywords", "filter_column", "select_tables", "select_columns")
PAIRS = (("B4", "B0"), ("B5", "B0"), ("B4", "B3"), ("B5", "B4"), ("B1", "B0"), ("B2-production", "B0"),
         ("B2-cheap", "B0"), ("B3", "B0"))  # (candidate, reference)
REGISTRY_DIR = "registry/test"


# ---------------------------------------------------------------- reading

def _result(plan: Dict[str, Any], key: str, judgment: str) -> Optional[Dict[str, Any]]:
    path = plan.get(key)
    return read_result(path, judgment) if path else None


def costs_of(j3: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Cost per correct query at standard prices: {"": x} for an API arm, {"20%": x, ...} with an SLM."""
    per_correct = j3["per_correct"]
    if "standard" in per_correct:
        return {"": per_correct["standard"]}
    return {k.split("@", 1)[1]: v for k, v in per_correct.items() if k.startswith("standard@")}


def per_arm(evals: Dict[str, str], split: str, ex_table: Callable) -> Dict[str, List[Dict[str, Any]]]:
    """J1's rows of each arm's eval execution, checked to be that arm, on that split. One call to
    `ex_table` for all of them, so J1 checks they share the instrument."""
    rows = ex_table([run_dir(e) for e in evals.values()])
    out = {}
    for arm, eval_run_id in evals.items():
        if arm not in ARMS:
            raise JudgmentError(f"unknown arm {arm!r} in the plan (one of {ARMS})")
        source = manifest(eval_run_id)["source_run_id"]
        mine = [r for r in rows if r["source_run_id"] == source]
        wrong = {(r["arm"], r["engine"], r["split"]) for r in mine} - {(*J1_KEY[arm], split)}
        if not mine or wrong:
            raise JudgmentError(f"{eval_run_id} is not {arm} on {split}: J1 reads {sorted(map(str, wrong)) or 'nothing'}")
        out[arm] = mine
    return out


def correct_of(rows: List[Dict[str, Any]]) -> Dict[str, bool]:
    return {str(r["question_id"]): bool(r["correct"]) for r in rows}


def pilot_d(pilot_evals: Dict[str, str], pilot_ids: List[str], ex_table: Callable, noninferiority: Callable,
            settings: tuple) -> Optional[float]:
    """The pilot's discordance: J4's d between B3 (the zero-shot SLM) and B0 on the pilot questions."""
    if not pilot_evals:
        return None
    if set(pilot_evals) != {"B0", "B3"}:
        raise JudgmentError("the pilot is the zero-shot SLM (B3) against the production LLM (B0) on calib: name those two")
    rows = per_arm(pilot_evals, "calib", ex_table)
    b0, b3 = correct_of(rows["B0"]), correct_of(rows["B3"])
    missing = sorted(set(pilot_ids) - (set(b0) & set(b3)), key=int)
    if missing:
        raise JudgmentError(f"the pilot's B0 and B3 executions did not answer every pilot question: {missing[:5]}")
    return plain(noninferiority({q: b3[q] for q in pilot_ids}, {q: b0[q] for q in pilot_ids}, *settings))["d"]


def gather(plan: Dict[str, Any], config: Dict[str, Any], ex_table: Callable, ex_summary: Callable,
           noninferiority: Callable, pilot_ids: List[str]) -> Dict[str, Any]:
    """Every number of the report, each from a judgment."""
    split = plan["split"]
    specs = plan.get("arms") or {}
    rows = per_arm({arm: spec["eval"] for arm, spec in specs.items()}, split, ex_table)
    pilot_plan = plan.get("pilot") or {}
    sources: Dict[str, Any] = {"arms": {}, "pilot": {arm: reference(e) for arm, e in pilot_plan.items()}}
    arms, correct = {}, {}
    for arm, spec in specs.items():
        j3 = read_result(spec["cost"], "J3")
        if j3["reads"].get("eval", {}).get("run_id") != spec["eval"]:
            raise JudgmentError(f"the J3 result of {arm} was not computed with {spec['eval']}")
        (summary,) = ex_summary(rows[arm])
        correct[arm] = correct_of(rows[arm])
        arms[arm] = {"n": summary["n"], "ex": summary["ex"],
                     "by_difficulty": {d: v["ex"] for d, v in summary["by_difficulty"].items()},
                     "run_id": j3["reads"]["run"]["run_id"], "cost_per_correct": costs_of(j3["result"]),
                     "cost_label": j3["result"]["label"], "lower_bound": j3["result"].get("lower_bound", False),
                     "slm_cost_basis": (j3["result"].get("slm_cost_basis") or {}).get("basis"),
                     "failed_unbilled": j3["result"]["failed_unbilled"], "prices_as_of": j3["result"]["prices_as_of"],
                     "calls": j3["result"]["calls"], "replaceable_fraction": j3["result"]["replaceable_fraction"]}
        sources["arms"][arm] = {"eval": reference(spec["eval"]), "j3": result_reference(spec["cost"])}
    settings = (config["thresholds"]["delta_cap_pp"], config["seeds"]["bootstrap"], n_boot(config))
    d_pilot = pilot_d(pilot_plan, pilot_ids, ex_table, noninferiority, settings)
    tests = {}
    for candidate, reference_arm in PAIRS:
        if candidate in correct and reference_arm in correct:
            tests[f"{reference_arm}|{candidate}"] = plain(noninferiority(correct[candidate], correct[reference_arm],
                                                                         *settings, d_pilot=d_pilot))

    judged = {}
    for key, judgment in (("j5", "J5"), ("j6", "J6"), ("j7", "J7"), ("j8", "J8"), ("per_call", "J2"),
                          ("teacher_train_cost", "J3")):
        found = _result(plan, key, judgment)
        judged[key] = found["result"] if found else None
        if found:
            sources[key] = result_reference(plan[key])
    if judged["per_call"] and (judged["per_call"].get("arm"), judged["per_call"].get("split")) != ("B4", split):
        raise JudgmentError(f"the per-call evaluation must be a replay routed as B4 on {split}")
    formats = {}
    for arm, path in (plan.get("format") or {}).items():
        found = read_result(path, "J2")
        if arm not in arms or found["reads"]["run"]["run_id"] != arms[arm]["run_id"]:
            raise JudgmentError(f"the format result of {arm} is not of {arm}'s execution")
        formats[arm] = found["result"]["format_validity"]
        sources.setdefault("format", {})[arm] = result_reference(path)
    repair = None
    revise = (judged["per_call"] or {}).get("per_call_site", {}).get("revise")
    if revise and revise["gold"]:
        d = None  # the repair pilot: J6's chosen candidate, zero-shot, against the teacher on calib's repair calls
        zeroshot = ((judged["j6"] or {}).get("per_call_site") or {}).get((judged["j6"] or {}).get("choice"), {})
        gold = (zeroshot.get("revise") or {}).get("gold")
        ids = sorted(set(pilot_ids) & set(gold["by_question"]["teacher"])) if gold else []
        if ids:
            d = plain(noninferiority({q: gold["by_question"]["replay"][q] for q in ids},
                                     {q: gold["by_question"]["teacher"][q] for q in ids}, *settings))["d"]
        repair = plain(noninferiority(revise["gold"]["by_question"]["replay"], revise["gold"]["by_question"]["teacher"],
                                      *settings, d_pilot=d))
    registry = test_registry()
    if split == "test" and not registry["available"]:
        raise JudgmentError(f"a test report needs the test registry, which git could not give: {registry['reason']}")
    return {"split": split, "arms": arms, "tests": tests, "d_pilot": d_pilot, "pilot_ids": sorted(pilot_ids, key=int), "repair_test": repair, "formats": formats,
            "judgments": judged, "concordance_min": config["thresholds"]["concordance_min"],
            "v3_min_ratio": config["claims"]["v3_min_ratio"], "registry": registry, "sources": sources}


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value.item() if hasattr(value, "item") else value


# ---------------------------------------------------------------- the SPEC §5 map

NO_PILOT = "no verdict (no pilot d)"


def outcome(test: Optional[Dict[str, Any]]) -> str:
    """The one reading of a J4 comparison every row shares."""
    if not test:
        return "no data"
    if test.get("d_pilot") is None:
        return "no pilot"
    if not test["testable"]:
        return "not testable"
    if test["noninferior"]:
        return "non-inferior"
    return "worse" if test["diff"] < -test["delta"] else "inconclusive"


def _cost(data, arm, u) -> Optional[float]:
    costs = data["arms"].get(arm, {}).get("cost_per_correct") or {}
    return costs.get("") if "" in costs else costs.get(u)


def _passes_v1(data, arm) -> bool:
    if arm == "B0":
        return "B0" in data["arms"]
    return outcome(data["tests"].get(f"B0|{arm}")) == "non-inferior"


def utilizations(data) -> List[str]:
    found = {u for arm in SLM_ARMS for u in (data["arms"].get(arm, {}).get("cost_per_correct") or {}) if u}
    return sorted(found, key=lambda u: float(u.rstrip("%")))


def best_untrained(data, u) -> Tuple[Optional[str], Optional[float]]:
    options = [(c, a) for a in UNTRAINED if a in data["arms"] and _passes_v1(data, a)
               for c in [_cost(data, a, u)] if c is not None]
    return (min(options)[1], min(options)[0]) if options else (None, None)


def _power(*tests) -> str:
    return ", ".join(f"{t['power']:.2f}" for t in tests if t and t.get("power") is not None) or "—"


def _why_not(outcomes: List[str]) -> str:
    """The verdict when a row neither confirms nor refutes, from its comparisons' outcomes."""
    if outcomes and all(o == "no pilot" for o in outcomes):
        return NO_PILOT
    if outcomes and all(o == "not testable" for o in outcomes):
        return "not testable"
    return "inconclusive"


def v1_verdict(t4, t5) -> str:
    outcomes = [outcome(t) for t in (t4, t5) if t]
    if not outcomes:
        return "no data"
    if "non-inferior" in outcomes:
        return "confirms"
    if outcomes == ["worse", "worse"]:
        return "refutes"
    return _why_not(outcomes)


def claims_map(data: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = []
    t4, t5 = data["tests"].get("B0|B4"), data["tests"].get("B0|B5")
    rows.append({"claim": "V1 / A1: SLMs suffice for agent calls (p.3–4)",
                 "result": "; ".join(f"{a} − B0: {_pp(t['diff'])} (Δ {_margin(t['delta'])} from the {t['margin_from']}, "
                                     f"CI low {_pp(t['ci_low'])})" for a, t in (("B4", t4), ("B5", t5)) if t) or "—",
                 "verdict": v1_verdict(t4, t5), "power": _power(t4, t5)})

    fraction = data["arms"].get("B5", {}).get("replaceable_fraction")
    rows.append({"claim": "A4 / A11: calls are narrow, subtasks simple (p.5, p.7)",
                 "result": (f"B5 replaceable: {_pct(fraction['calls'])} of calls, {_pct(fraction['tokens'])} of tokens, "
                            f"{_pct(fraction['cost_at_production_price'])} of cost") if fraction else "—",
                 "verdict": "descriptive (the SPEC fixes no number for 'high')" if fraction else "no data", "power": "—"})

    rows.append(_appendix_b(data))
    rows.append(_a5(data))

    a6, v3, av2 = [], [], []
    for u in utilizations(data) or [""]:
        base, base_cost = best_untrained(data, u)
        b5_cost = _cost(data, "B5", u)
        if b5_cost is None or base_cost is None:
            a6.append((u, "no data"))
        elif b5_cost >= base_cost:
            a6.append((u, "refutes"))
        else:
            a6.append((u, "confirms" if _passes_v1(data, "B5") else _why_not([outcome(t5)])))
        trained = [c for a in TRAINED if _passes_v1(data, a) for c in [_cost(data, a, u)] if c is not None]
        if base_cost is None:
            v3.append((u, "no data"))
        elif not trained:
            v3.append((u, f"{_why_not([outcome(t) for t in (t4, t5) if t])} (no trained arm passes V1)"))
        else:
            ratio = base_cost / min(trained)
            v3.append((u, f"{'confirms' if ratio >= data['v3_min_ratio'] else 'refutes'} ({ratio:.1f}× vs {base})"))
        slm_costs = [c for a in TRAINED for c in [_cost(data, a, u)] if c is not None]
        b1 = _cost(data, "B1", u)
        av2.append((u, "no data" if b1 is None or not slm_costs else
                    "refutes the paper (AV2 wins)" if b1 <= min(slm_costs) else "does not refute"))
    per_u = lambda items: " · ".join(f"{u}: {v}" if u else v for u, v in items)  # noqa: E731
    rows.append({"claim": "A6: heterogeneous systems (p.6)", "result": "B5 vs the best arm without training, per utilization",
                 "verdict": per_u(a6), "power": _power(t5)})
    rows.append({"claim": "A7: agent logs become data (p.6)", "result": "needs extension 2 (teacher × gold)",
                 "verdict": "not testable in the core", "power": "—"})
    rows.append({"claim": "V3 / A2: a 7B SLM is 10–30× cheaper (p.4)",
                 "result": f"cost per correct query, best trained vs best untrained arm (confirms at {data['v3_min_ratio']}×)",
                 "verdict": per_u(v3), "power": _power(t4, t5)})
    rows.append({"claim": "AV2 / CA3–CA4: centralized scale can be cheaper (p.7–8)",
                 "result": "B1 vs the cheaper of B4, B5, marginal cost only: the fixed cost and its payback (SPEC §6.6) are not a J1–J8 output",
                 "verdict": per_u(av2), "power": "—"})
    rows.append({"claim": "A2 / A3: adapting is fast and cheap (p.5)", "result": "time and cost per adapter: S5 training manifests",
                 "verdict": "descriptive (not a J1–J8 output)", "power": "—"})
    rows.append({"claim": "B2: generalist benchmarks guide selection poorly (p.8)", "result": "needs extension 10",
                 "verdict": "not testable in the core", "power": "—"})
    j5 = data["judgments"]["j5"]
    rows.append({"claim": "S3: clustering discovers the tasks (p.9)",
                 "result": (f"ARI {j5['ari_call_sites']:.3f} over {j5['k']} clusters; "
                            f"{_pct(j5['truncation']['prompt_action']['truncated_fraction'])} of texts cut") if j5 else "—",
                 "verdict": "descriptive" if j5 else "no data", "power": "—"})
    rows.append({"claim": "AV1: a same-generation LLM always wins (p.7)",
                 "result": f"B4 − B0: {_pp(t4['diff'])}" if t4 else "—", "verdict": "descriptive (not a direct test)" if t4 else "no data",
                 "power": "—"})
    return rows


def _a5(data) -> Dict[str, str]:
    row = {"claim": "A5: one format with a trained SLM is preferable (p.6)", "power": "—"}
    b0f, b4f = data["formats"].get("B0"), data["formats"].get("B4")
    sites = sorted(set(b0f or {}) & set(b4f or {}))
    if not sites:
        return {**row, "result": "—", "verdict": "no data"}
    worse = [s for s in sites if b4f[s]["rate"] < b0f[s]["rate"]]
    result = ("B4 below B0 on " + ", ".join(f"{s} ({_pct(b4f[s]['rate'])} vs {_pct(b0f[s]['rate'])})" for s in worse)
              if worse else f"B4 at least B0 on all {len(sites)} call sites")
    return {**row, "result": result, "verdict": "refutes" if worse else "confirms"}


def _appendix_b(data) -> Dict[str, str]:
    row = {"claim": "Appendix B: the LLM keeps unstructured error resolution (p.16)", "power": _power(data["repair_test"])}
    per_call, repair = data["judgments"]["per_call"], data["repair_test"]
    if not per_call or not repair:
        return {**row, "result": "—", "verdict": "no data"}
    routine = {site: e["agreement"]["rate"] for site, e in per_call["per_call_site"].items()
               if site in ROUTINE and e["agreement"] and e["agreement"]["rate"] is not None}
    passes_routine = bool(routine) and all(rate >= data["concordance_min"] for rate in routine.values())
    repaired = outcome(repair)
    result = (f"repair: SLM − teacher {_pp(repair['diff'])} (Δ {_margin(repair['delta'])} from the {repair['margin_from']}), "
              f"{repaired}; routine agreement: " + ", ".join(f"{s} {_pct(r)}" for s, r in sorted(routine.items())))
    proxy = "the routine by the agreement proxy, which supports no per-cluster claim (D15)"
    if not passes_routine:
        verdict = f"refutes (the SLM loses on the routine; {proxy})"
    elif repaired == "non-inferior":
        verdict = "refutes (the SLM ties on repair)"
    elif repaired == "worse":
        verdict = f"confirms ({proxy})"
    else:
        verdict = _why_not([repaired])
    return {**row, "result": result, "verdict": verdict}


# ---------------------------------------------------------------- S1–S6 and S4

def steps(data: Dict[str, Any]) -> List[Dict[str, str]]:
    j = data["judgments"]
    train, j5, j6, j7 = j["teacher_train_cost"], j["j5"], j["j6"], j["j7"]
    unmeasured = "not measured by J1–J8"
    rows = [{"step": "S1 · collection", "did": (f"{train['calls']} teacher calls logged on train (C1)" if train else "—"),
             "cost": (_usd(train["total"].get("standard")) if train else "—"), "changed": "the training data exists"}]
    if j5:
        c = j5["curation"]["total"]
        rows.append({"step": "S2 · curation",
                     "did": (f"{c['invocations']} invocations; {c['passed_filter']} passed the production signal; "
                             f"{c.get('masked_sql', 0)} SQL completions dropped because masking changed them; "
                             f"{c['exact_duplicates']} exact and {c['near_duplicates']} near duplicates removed; "
                             f"{sum(j5['curation']['mask_detections'].values())} sensitive-data detections masked; paraphrase not applied"),
                     "cost": unmeasured, "changed": f"{c['kept']} training examples"})
        calib = j5["assignment"]["calib"]
        rows.append({"step": "S3 · clustering",
                     "did": f"{j5['k']} clusters on prompt+action, ARI {j5['ari_call_sites']:.3f} against call sites",
                     "cost": unmeasured, "changed": (f"assignment by prompt: {_pct(j5['assignment']['train_in_sample'])} in sample, "
                                                     f"{_pct(calib['rate']) if calib else '—'} on calib")})
    else:
        rows += [{"step": "S2 · curation", "did": "—", "cost": "—", "changed": "—"},
                 {"step": "S3 · clustering", "did": "—", "cost": "—", "changed": "—"}]
    rows.append({"step": "S4 · selection", "did": (f"zero-shot on calib: " + ", ".join(
        f"{n} {s:.3f}" for n, s in sorted(j6["score"].items())) if j6 else "—"), "cost": unmeasured,
        "changed": f"base {j6['choice']} (by {j6['decided_by']})" if j6 else "—"})
    rows.append({"step": "S5 · specialization", "did": f"one adapter per cluster (adapters fact {j7['adapters'][:12]})" if j7 else "—",
                 "cost": "S5 training manifests (not a J1–J8 output)", "changed": _test_line(data, "B3|B4")})
    rested = sorted(c for c, d in (j7 or {}).get("clusters", {}).items() if d.get("cost_dependent"))
    rows.append({"step": "S6 · router", "did": ("allocation: " + ", ".join(f"{c} → {e}" for c, e in sorted(j7["allocation"].items()))
                                                 + (f"; {len(rested)} chosen on the SLM cost extrapolated from per-adapter "
                                                    f"load tests alone ({', '.join(rested)})" if rested else "")
                                                 if j7 else "—"),
                 "cost": unmeasured, "changed": _test_line(data, "B4|B5")})
    return rows


def _test_line(data, key) -> str:
    test = data["tests"].get(key)
    if not test:
        return "—"
    reference_arm, candidate = key.split("|")
    return f"{candidate} − {reference_arm}: {_pp(test['diff'])} (CI low {_pp(test['ci_low'])}), {outcome(test)}"


# ---------------------------------------------------------------- the test registry

def test_registry(root: Optional[Path] = None) -> Dict[str, Any]:
    """Test-split executions as F1 commits them (`registry/test/`), read at HEAD: each intent, and
    its manifest if the execution ended; an intent without one is an interrupted execution."""
    root = root or paths.ROOT

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    listed = git("ls-tree", "-r", "--name-only", "HEAD", REGISTRY_DIR)
    if listed.returncode != 0:
        return {"available": False, "reason": scrub(listed.stderr.strip() or "git failed")[:200], "runs": []}
    files = {Path(line).name for line in listed.stdout.splitlines()}
    run_ids = sorted({name.rsplit(".", 2)[0] for name in files if name.endswith((".intent.json", ".manifest.json"))})
    runs = []
    for run_id in run_ids:
        name = f"{run_id}.manifest.json" if f"{run_id}.manifest.json" in files else f"{run_id}.intent.json"
        shown = git("show", f"HEAD:{REGISTRY_DIR}/{name}")
        if shown.returncode != 0:
            raise JudgmentError(f"cannot read {REGISTRY_DIR}/{name} at HEAD")
        found = json.loads(shown.stdout)
        ended = name.endswith(".manifest.json")
        runs.append({"run_id": run_id, "type": found.get("type"), "arm": found.get("arm"), "engine": found.get("engine"),
                     "status": found.get("status") if ended else "interrupted (intent, no manifest)",
                     "commit": found.get("commit"), "prereg_hash": found.get("prereg_hash")})
    return {"available": True, "runs": runs}


# ---------------------------------------------------------------- rendering

def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _pp(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:+.1f} pp"


def _margin(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:.1f} pp"


def _usd(x: Optional[float]) -> str:
    return "—" if x is None else f"${x:,.4g}"


def _table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows])


def chart_svg(data: Dict[str, Any]) -> str:
    """EX × cost per correct query (log scale), one point per arm; an SLM arm has one point per
    utilization, joined. Blue: no SLM; orange: SLM. Every point carries its label and a tooltip."""
    points = []  # (arm, utilization, cost, ex)
    for arm in ARMS:
        found = data["arms"].get(arm)
        if not found or found["ex"] is None:
            continue
        for u, cost in sorted(found["cost_per_correct"].items(), key=lambda kv: -float(kv[0].rstrip("%") or 0)):
            if cost:
                points.append((arm, u, cost, found["ex"]))
    width, height, left, right, top, bottom = 640, 420, 64, 24, 40, 48
    if not points:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="20" y="40">no data</text></svg>'
    xs = [math.log10(p[2]) for p in points]
    lo, hi = min(xs), max(xs)
    lo, hi = (lo - 0.5, hi + 0.5) if hi - lo < 1e-9 else (lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    x = lambda cost: left + (math.log10(cost) - lo) / (hi - lo) * (width - left - right)  # noqa: E731
    y = lambda ex: top + (1 - ex) * (height - top - bottom)  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
           f'font-family="system-ui, sans-serif" font-size="12" role="img" aria-label="EX against cost per correct query">',
           "<style>:root{--surface:#fcfcfb;--ink:#1f1f1e;--muted:#6b6a64;--grid:#e4e3dd;--no-slm:#2a78d6;--slm:#eb6834}"
           "@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#ffffff;--muted:#c3c2b7;--grid:#34342f;"
           "--no-slm:#3987e5;--slm:#d95926}}"
           "text{fill:var(--muted)} .label{fill:var(--ink)} .grid{stroke:var(--grid);stroke-width:1}"
           ".no-slm{fill:var(--no-slm)} .slm{fill:var(--slm)} .line{stroke:var(--slm);stroke-width:2;fill:none}"
           "circle{stroke:var(--surface);stroke-width:2}</style>",
           f'<rect width="{width}" height="{height}" style="fill:var(--surface)"/>']
    for tick in range(0, 101, 20):
        out.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(tick / 100):.1f}" y2="{y(tick / 100):.1f}"/>'
                   f'<text x="{left - 8}" y="{y(tick / 100) + 4:.1f}" text-anchor="end">{tick}%</text>')
    for exponent in range(math.floor(lo), math.ceil(hi) + 1):
        if lo <= exponent <= hi:
            px = x(10 ** exponent)
            out.append(f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" y1="{top}" y2="{height - bottom}"/>'
                       f'<text x="{px:.1f}" y="{height - bottom + 16}" text-anchor="middle">${10 ** exponent:g}</text>')
    out.append(f'<text x="{(left + width - right) / 2}" y="{height - 8}" text-anchor="middle">cost per correct query (USD, log scale)</text>'
               f'<text x="14" y="{(top + height - bottom) / 2}" text-anchor="middle" transform="rotate(-90 14 {(top + height - bottom) / 2})">EX</text>')
    for arm in SLM_ARMS:
        line = [p for p in points if p[0] == arm]
        if len(line) > 1:
            out.append('<polyline class="line" points="' + " ".join(f"{x(p[2]):.1f},{y(p[3]):.1f}" for p in line) + '"/>')
    for arm, u, cost, ex in points:
        kind = "slm" if arm in SLM_ARMS else "no-slm"
        tip = f"{arm}{' at ' + u + ' utilization' if u else ''}: EX {_pct(ex)}, {_usd(cost)} per correct query"
        out.append(f'<circle class="{kind}" cx="{x(cost):.1f}" cy="{y(ex):.1f}" r="5"><title>{tip}</title></circle>')
    placed: List[Tuple[float, float, float]] = []  # (x, y, width) of labels already drawn
    for arm in sorted(ARMS, key=lambda a: -max((p[3] for p in points if p[0] == a), default=0)):
        line = [p for p in points if p[0] == arm]
        if not line:
            continue
        anchor = line[len(line) // 2]
        lx, ly, lw = x(anchor[2]) + 8, y(anchor[3]) - 8, 7.5 * len(arm)
        while any(abs(ly - py) < 14 and lx < px + pw and px < lx + lw for px, py, pw in placed):
            ly += 14  # below the label it would overlap
        placed.append((lx, ly, lw))
        out.append(f'<text class="label" x="{lx:.1f}" y="{ly:.1f}">{arm}</text>')
    out.append(f'<circle class="no-slm" cx="{left + 6}" cy="12" r="5"/><text class="label" x="{left + 16}" y="16">no SLM</text>'
               f'<circle class="slm" cx="{left + 86}" cy="12" r="5"/>'
               f'<text class="label" x="{left + 96}" y="16">SLM: one point per utilization, 20% (right) to 100% (left)</text>')
    out.append("</svg>")
    return "\n".join(out)


def render(data: Dict[str, Any]) -> str:
    j6, per_call = data["judgments"]["j6"], data["judgments"]["per_call"]
    parts = [f"# slm-conversion-bench · report ({data['split']})", "",
             "Generated by `bench report` from the judgments J1–J8 and the test registry; every number below is in "
             "`report.json` with the judgment it came from. The texts that interpret it (error analysis, limitations, "
             "external references) are the author's (SPEC §8).", "",
             "## EX and cost per correct query", "", "![EX against cost per correct query](ex_cost.svg)", ""]
    difficulties = sorted({d for a in data["arms"].values() for d in a["by_difficulty"]})
    rows = []
    for arm in ARMS:
        a = data["arms"].get(arm)
        if a:
            costs = ", ".join(f"{u + ': ' if u else ''}{_usd(c)}" for u, c in
                              sorted(a["cost_per_correct"].items(), key=lambda kv: float(kv[0].rstrip("%") or 0)))
            usage = a["cost_label"] + (f", lower bound ({a['failed_unbilled']} failed calls unpriced)" if a["lower_bound"] else "")
            usage += f"; SLM cost {a['slm_cost_basis']}" if a.get("slm_cost_basis") else ""
            rows.append([arm, a["n"], _pct(a["ex"])] + [_pct(a["by_difficulty"].get(d)) for d in difficulties]
                        + [costs, usage])
    parts += [_table(["arm", "n", "EX"] + [f"EX {d}" for d in difficulties] + ["cost per correct query (standard prices)", "usage"], rows), ""]
    parts += ["## The SPEC §5 map", "", _table(["claim", "result", "verdict", "power"],
                                               [[r["claim"], r["result"], r["verdict"], r["power"]] for r in data["map"]]), ""]
    parts += ["## S1–S6: what each step did, cost and changed", "",
              _table(["step", "what it did", "cost", "what it changed"], [[s["step"], s["did"], s["cost"], s["changed"]] for s in data["steps"]]), ""]
    parts += ["## S4 desk triage", ""]
    triage = (j6 or {}).get("triage") or []
    if triage:
        keys = [k for k in triage[0] if k != "candidate"]
        parts.append(_table(["candidate"] + keys + ["zero-shot score (calib)"],
                            [[t.get("candidate")] + [t.get(k, "—") for k in keys] + [
                                f"{j6['score'][t['candidate']]:.3f}" if t.get("candidate") in j6["score"] else "—"] for t in triage]))
    else:
        parts.append("The desk triage is not recorded in `config.yaml › selection.triage`.")
    parts.append("")
    fraction = data["arms"].get("B5", {}).get("replaceable_fraction")
    parts += ["## Replaceable fraction (B5)", "",
              _table(["by calls", "by tokens", "by cost (at the production LLM's price)"],
                     [[_pct(fraction["calls"]), _pct(fraction["tokens"]), _pct(fraction["cost_at_production_price"])]])
              if fraction else "No B5 execution with SLM calls.", ""]
    if per_call:
        rows = []
        for site, e in per_call["per_call_site"].items():
            rows.append([site, e["n"], _pct(e["format_valid_rate"]),
                         (f"{_pct(e['gold']['ex_replay'])} vs teacher {_pct(e['gold']['ex_teacher'])} per call; "
                          f"{_pct(e['gold']['ex_by_question']['replay'])} vs {_pct(e['gold']['ex_by_question']['teacher'])} "
                          f"per question") if e["gold"] else "—",
                         _pct(e["agreement"]["rate"]) if e["agreement"] else "—"])
        parts += [f"## Per-call-site evaluation ({per_call.get('arm') or per_call.get('engine')}, teacher's context)", "",
                  _table(["call site", "n", "format valid", "EX (gold)", "agreement with the teacher (fidelity)"], rows), ""]
    registry = data["registry"]
    parts += ["## Test registry", ""]
    if not registry["available"]:
        parts.append(f"The registry could not be read from git: {registry['reason']}.")
    elif registry["runs"]:
        parts.append(_table(["execution", "type", "arm", "status", "commit", "pre-registration"],
                            [[r["run_id"], r["type"] or "—", r["arm"] or "—", r["status"], (r["commit"] or "—")[:12],
                              (r["prereg_hash"] or "—")[:12]] for r in registry["runs"]]))
    else:
        parts.append("No test-split execution is committed.")
    parts += ["", "## Error analysis, limitations, external references", "",
              "Written by the author (SPEC §8); external results are labelled as not comparable.", ""]
    return "\n".join(parts)


def run(plan_path: str, config: Dict[str, Any], ex_table: Optional[Callable] = None, ex_summary: Optional[Callable] = None,
        noninferiority: Optional[Callable] = None, out_root: Optional[Path] = None,
        pilot_ids: Optional[List[str]] = None) -> Path:
    import yaml
    plan = yaml.safe_load(Path(plan_path).read_text())
    if ex_table is None or ex_summary is None:
        from bench.judge import j1  # F2's J1
        ex_table, ex_summary = ex_table or j1.ex_table, ex_summary or j1.ex_summary
    if noninferiority is None:
        from bench.judge.j4 import noninferiority  # F2's J4
    if pilot_ids is None:
        from bench.judge.j7 import registered_pilot
        pilot_ids = registered_pilot(config)
    data = gather(plan, config, ex_table, ex_summary, noninferiority, pilot_ids)
    data["map"], data["steps"] = claims_map(data), steps(data)
    raw = canonical(data)
    out = (out_root or paths.ROOT / "reports") / hashlib.sha256(raw).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_bytes(raw)
    (out / "ex_cost.svg").write_text(chart_svg(data))
    (out / "report.md").write_text(render(data))
    return out
