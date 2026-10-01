"""R · the report (SPEC §8), from the judgments J1–J8 and the test registry only.

`bench report --plan <plan.yaml>` reads a plan that names, and only names, what the report is made
of (no number goes in it):

    split: test
    arms:       {<arm>: {eval: <eval execution>, cost: <its J3 result>}}      # B0, B1, B2-production, B2-cheap, B3, B4, B5
    pilot:      {B0: <eval execution on calib>, B3: <eval execution on calib>}   # the same pilot questions
    format:     {B0: <J2 result, format validity of its run>, B4: <...>}
    per_call:   <J2 result of the test inputs replayed as B4>
    j5: <J5 result>   j6: <J6 result>   j7: <J7 result>   j8: <J8 result>
    adapters: <the adapters fact, when the plan has no J7; with J7, it must be J7's>
    teacher_train_cost: <J3 result of the teacher on train>

Every figure is read from a judgment or computed by one: J1's per-question table and summary
(`ex_table`, `ex_summary`) and J4's tests (`noninferiority`, the candidate first, the reference
second). **Every J4 margin is the fixed `thresholds.delta_pp`** (design §6.3 T3), end to end and per
call site, never derived from a discordance. **The pilot gives only the planned power** (SPEC §6.4),
measured on calib before training: `d_pilot` is the discordance J4 measures between B3 (the zero-shot
SLM) and B0 on the pilot questions (`bench.data.pilot_ids`), and the planned power of a comparison is
J4's `power` at that d and the comparison's n (the test's 498 questions); for repair, d is between the
zero-shot candidate S4 chose and the teacher on the pilot questions' repair calls (J6's result). A
comparison with no pilot still gets its verdict, with no planned power. **A test report is read only under the
registration in force**, before anything is read: the barrier finds it published, intact and
matching the configuration given (every threshold, seed and the pilot come from it), and the analysis
code is the registered one. **A test report is bound to the test registry**: every run it binds ran
under that registration; every arm's run is a `done` entry; a configuration (type, arm, engine, pre-registration)
with more than one **completed** test run gets no verdict, and every run is listed (SPEC §6.1): one
that did not finish as "not completed", which never blocks its re-run, and one under an earlier
registration as "superseded by re-registration"; `prereg/DEVIATIONS.md`, one line per new registration,
is printed when it exists (T13); the facts the B3–B5 runs recorded are the ones the plan's J5, J6 and J7 name; the per-call evaluation
replays the plan's B0 run; and the pilot finished before the first test execution started. The test
registry is read from git as F1 records it: `registry/test/<run_id>.intent.json` before a test
execution starts and `<run_id>.manifest.json` when it ends, as committed at HEAD; an intent with no
manifest is an interrupted execution.

Writes `reports/<sha256>/`: `report.md`, `report.json` (every number, with the judgments it came
from; its sha256 names the directory) and `ex_cost.svg`, the main chart.

**How each row of the SPEC §5 map is decided** (the SPEC fixes the criteria; where it leaves a
term open, the reading is stated here and in the row). A J4 comparison has one outcome, shared by
every row: several runs (no verdict), non-inferior (the one-sided 95% lower bound above −Δ), worse
(the one-sided 95% **upper** bound below −Δ: refuting takes the same standard as confirming) or
inconclusive (neither), which a row states with the planned power. A cost verdict
carries the labels of the costs it rests on (estimated, lower bound, upper bound, extrapolated), and
one that rests on an upper-bound cost is inconclusive; arms priced from different price tables (by
date or by content) are refused.
Every verdict holds for this workload only: the map opens with a scope line, and every claim says
"(this workload)" (T15).
- V1/A1: judged on B4 alone (T7): confirms if B4 is non-inferior to B0, refutes if it is worse,
  otherwise inconclusive with the planned power. B5 against B0 belongs to A6: B5 can be non-inferior
  by keeping its calls on the LLM.
- A4/A11: the replaceable fraction of B5 by call, token and cost, and whether B5 met
  non-inferiority; the SPEC says "high" without a number, so the row is descriptive. Not testable
  when B5's allocation (J7) gave no cluster to an SLM.
- Appendix B: decided on repair alone, on the per-call-site evaluation of B4 on the test inputs (T4):
  confirms when the SLM is worse on repair, refutes when it is non-inferior, otherwise inconclusive
  with the planned power. The routine call sites' agreement with the teacher is shown beside it as a
  proxy (D15: it supports no per-cluster claim) and has no effect on the verdict.
- A5: format validity per call site (SPEC §6.3), B4 against B0 (T11): refutes only if B4 is more than
  `thresholds.format_tolerance_pp` below B0 on a call site with at least `allocation.min_calls`
  invocations in both arms; confirms otherwise; call sites with fewer are listed and do not count.
- A6, V3/A2, AV2: cost per correct query. "The best arm without training" is the cheapest among B0,
  B1 and B2 that is non-inferior to B0 (B0 always is). An SLM arm's cost falls as its utilization
  rises, so each claim is decided where it is hardest to win (T10): it confirms only if it holds at
  the lowest configured utilization (`cost.utilizations`), refutes only if it fails at the highest,
  and is otherwise utilization-dependent, with the tipping point u* at which the SLM arm breaks
  even (interpolated: its cost is A + G/u, fitted through the lowest and the highest utilization).
  - V3 (T9): the ratio of cost per correct query, the best arm without training ÷ the cheapest of the
    SLM arms (B4, B5 with SLM calls) that are non-inferior to B0, with a 95% interval from a paired
    bootstrap over the questions (`seeds.bootstrap`, `stats.n_boot`; each question's cost is J3's
    `by_question`). Each resample takes the cheapest arm of each set again, so the interval covers the
    choice of the arms too (which arms are non-inferior is J4's verdict, and stays). At one utilization it reads "meets the paper's bar" if the lower bound is at or
    above `claims.v3_min_ratio`, "cheaper, below the bar" if the lower bound is above 1, "refutes V3"
    if the upper bound is at or below 1, and inconclusive otherwise. Over the two ends: the reading
    at the lowest utilization when it is cheaper there (the bar is met only if met there); refutes when
    not cheaper at the highest; utilization-dependent, with u*, when not cheaper at the lowest and
    cheaper at the highest; and when an end's interval holds 1 and neither rule decides, inconclusive,
    saying how each end reads (no u* is printed for an end the data does not decide). Not testable
    when every SLM arm is worse than B0: there is no arm to price.
  - A6: confirms if B5 is cheaper than the best arm without training at the lowest utilization and
    non-inferior to B0; refutes if B5 is worse than B0, or not cheaper at the highest. Not testable
    when B5's allocation gave no cluster to an SLM, or its execution made no SLM call.
  - AV2 wins (the paper is refuted) only if B1 costs no more than the cheaper SLM arm at the highest
    utilization too; it does not refute if the SLM arm is cheaper at the lowest. The fixed cost and
    its payback (SPEC §6.6) are not a J1–J8 output and are left out, which the row says.
- A7, A2/A3, B2: not measured by J1–J8 in the core; stated as such.
"""
import hashlib
import json
import math
import random
import subprocess
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from bench import barrier, paths
from bench.contracts.concordance import GOLD_CALL_SITES as GOLD_SITES
from bench.contracts.facts import read_fact
from bench.contracts.router import ARM_FACTS
from bench.provenance import scrub
from bench.judge import j4
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
DEVIATIONS = "prereg/DEVIATIONS.md"


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


def pilot_d(pilot_evals: Dict[str, str], pilot_ids: List[str], ex_table: Callable) -> Optional[float]:
    """The pilot's discordance: between B3 (the zero-shot SLM) and B0 on the pilot questions. It gives the
    planned power only, never a margin."""
    if not pilot_evals:
        return None
    if set(pilot_evals) != {"B0", "B3"}:
        raise JudgmentError("the pilot is the zero-shot SLM (B3) against the production LLM (B0) on calib: name those two")
    rows = per_arm(pilot_evals, "calib", ex_table)
    b0, b3 = correct_of(rows["B0"]), correct_of(rows["B3"])
    missing = sorted(set(pilot_ids) - (set(b0) & set(b3)), key=int)
    if missing:
        raise JudgmentError(f"the pilot's B0 and B3 executions did not answer every pilot question: {missing[:5]}")
    return j4.discordance({q: b3[q] for q in pilot_ids}, {q: b0[q] for q in pilot_ids})


def planned(test: Dict[str, Any], d_pilot: Optional[float]) -> Dict[str, Any]:
    """A J4 test with its pilot discordance and planned power: J4's `power` at the pilot's d, this
    comparison's n and its fixed margin (None without a pilot)."""
    power = None if d_pilot is None else j4.power(d_pilot, test["n"], test["delta"])
    return {**test, "d_pilot": d_pilot, "planned_power": power}


NEEDS = ARM_FACTS  # the facts each trained arm runs on (C4)
FACT_SOURCE = {"choice": "j6", "centroids": "j5", "adapters": "j7 or plan.adapters", "allocation": "j7"}


def _expected_facts(plan: Dict[str, Any], judged: Dict[str, Any]) -> Dict[str, str]:
    """The fact shas the plan's judgments name."""
    expected = {}
    if judged.get("j6"):
        expected["choice"] = judged["j6"]["choice_fact"]["sha256"]
    if judged.get("j5"):
        expected["centroids"] = judged["j5"]["centroids"]["sha256"]
    adapters = [judged["j7"]["adapters"]] if judged.get("j7") else []
    if plan.get("adapters"):
        adapters.append(read_fact(str(paths.ROOT / plan["adapters"]), "adapters")[1])
    if len(set(adapters)) > 1:
        raise JudgmentError(f"the plan's adapters fact {adapters[-1]!r} is not the one its J7 allocated on {adapters[0]!r}")
    if adapters:
        expected["adapters"] = adapters[0]
    if judged.get("j7"):
        expected["allocation"] = judged["j7"]["allocation_fact"]["sha256"]
    return expected


def registration_in_force(split: str, config: Dict[str, Any]) -> Optional[str]:
    """For a test report, the pre-registration in force: the barrier's (published, intact, and matching
    `config`, whose thresholds, seeds and pilot every verdict reads, and the splits, SPEC.md and the data
    manifest), with the analysis code the registered one (bench.prereg.check_registered_analysis_code).
    The readings of SPEC §5 cannot change after the test without changing the registration."""
    if split != "test":
        return None
    from bench.prereg import PreregError, check_registered_analysis_code
    try:
        in_force = barrier.prereg_hash_in_force(config)
        check_registered_analysis_code(paths.ROOT)
    except (barrier.TestSplitLocked, PreregError, OSError) as e:
        raise JudgmentError(f"a test report is read only under the registration in force: {e}") from e
    return in_force


def _registry_bindings(split: str, arms: Dict[str, Any], registry: Dict[str, Any], judged: Dict[str, Any],
                       reads: Dict[str, Any], pilot_plan: Dict[str, str], expected: Dict[str, str],
                       in_force: Optional[str]) -> Optional[List[str]]:
    """A test report is bound to what the test registry recorded (SPEC 6.1, REQ-013):
    - every arm's run, and the per-call evaluation's replay, is a `done` entry made under the
      pre-registration in force (`in_force`), and each configuration's **completed** runs under it are
      all known (several `done` runs of one configuration: no verdict). A run that did not finish is
      listed as not completed and never blocks its re-run (T13);
    - every trained arm's run (and the replay routed as B4) recorded the facts the plan's judgments
      name, and a plan without them is refused, never unchecked;
    - the per-call evaluation replays the plan's B0 run, and that teacher run is bound like an arm's
      (its several runs count with the replay's), even when the plan has no B0 arm;
    - every pilot the report uses (the pilot's evals, J6's zero-shot per-call evals) finished
      before the first test execution started.
    Marks `several_runs` on the arms; returns the replay's several runs, if any; raises otherwise."""
    if split != "test":
        return None
    if not registry["available"]:
        raise JudgmentError(f"a test report needs the test registry, which git could not give: {registry['reason']}")
    undated = [r["run_id"] for r in registry["runs"] if not r.get("started_at")]
    if undated:
        raise JudgmentError(f"registry entries without started_at (no intent): {undated}")
    entries = {r["run_id"]: r for r in registry["runs"]}

    def bind(label: str, run_id: str) -> Tuple[Dict[str, Any], Optional[List[str]]]:
        entry = entries.get(run_id)
        if entry is None or entry["status"] != "done":
            raise JudgmentError(f"{label}'s run {run_id} is not a done entry of the test registry "
                                f"({'not registered' if entry is None else entry['status']})")
        if entry["prereg_hash"] != in_force:
            raise JudgmentError(f"{label}'s run {run_id} ran under pre-registration {entry['prereg_hash']}, not the one "
                                f"in force ({in_force}): it is reported, never read for a verdict")
        key = (entry["type"], entry["arm"], entry["engine"], entry["prereg_hash"])
        same = sorted(r["run_id"] for r in registry["runs"] if r["status"] == "done"
                      and (r["type"], r["arm"], r["engine"], r["prereg_hash"]) == key)
        return entry, (same if len(same) > 1 else None)

    def facts_of(label: str, entry: Dict[str, Any], names: Tuple[str, ...]) -> None:
        missing = [n for n in names if n not in expected]
        if missing:
            raise JudgmentError(f"{label}'s facts cannot be checked: the plan names no "
                                f"{', '.join(FACT_SOURCE[n] for n in missing)}")
        recorded = entry.get("facts") or {}
        for name in names:
            if recorded.get(name) != expected[name]:
                raise JudgmentError(f"{label}'s test run recorded the {name} fact {recorded.get(name)!r}; "
                                    f"the plan's judgments name {expected[name]!r}")

    for arm, found in arms.items():
        entry, found["several_runs"] = bind(arm, found["run_id"])
        if arm in NEEDS:
            facts_of(arm, entry, NEEDS[arm])
    replay_several = None
    if reads.get("per_call"):
        entry, replay_several = bind("the per-call evaluation", reads["per_call"]["replay"]["run_id"])
        facts_of("the per-call evaluation's replay (routed as B4)", entry, NEEDS["B4"])
        teacher = reads["per_call"]["teacher"]["run_id"]
        if "B0" in arms and teacher != arms["B0"]["run_id"]:
            raise JudgmentError("the per-call evaluation replays another teacher run than the plan's B0")
        _, teacher_several = bind("the per-call evaluation's teacher", teacher)  # bound even with no B0 arm
        replay_several = sorted(set((replay_several or []) + (teacher_several or []))) or None
    pilots = dict(pilot_plan)
    if judged.get("j6") and reads.get("j6"):
        zeroshot = reads["j6"].get(judged["j6"]["choice"]) or {}
        for who in ("replay_eval", "teacher_eval"):
            if zeroshot.get(who):
                pilots[f"J6's zero-shot {who}"] = zeroshot[who]["run_id"]
    first = min(datetime.fromisoformat(r["started_at"]) for r in registry["runs"]) if registry["runs"] else None
    for label, eval_run_id in pilots.items():
        finished = manifest(eval_run_id).get("finished_at")
        if first is not None and (not finished or datetime.fromisoformat(finished) >= first):
            raise JudgmentError(f"the pilot's {label} evaluation {eval_run_id} did not finish before the first test "
                                f"execution started ({first.isoformat()}): the pilot is read before the test is touched")
    return replay_several


REQUIRED_COST = ("upper_bound", "cache_not_reported", "lower_bound", "failed_unbilled", "prices", "by_question")


def gather(plan: Dict[str, Any], config: Dict[str, Any], ex_table: Callable, ex_summary: Callable,
           noninferiority: Callable, pilot_ids: List[str]) -> Dict[str, Any]:
    """Every number of the report, each from a judgment."""
    split = plan["split"]
    registry, in_force, remote = None, None, None
    if split == "test":  # before anything is read: the registry as published, then the registration in force
        remote, why_not = barrier.fetch_origin_main(paths.ROOT)
        if remote is None:
            raise JudgmentError(f"a test report needs the published test registry: {why_not}")
        registry = test_registry(ref=remote)  # a record only a local clone holds is no record
        if not registry["available"]:
            raise JudgmentError(f"a test report needs the test registry, which git could not give: {registry['reason']}")
        in_force = registration_in_force(split, config)
    else:
        registry = test_registry()
    specs = plan.get("arms") or {}
    rows = per_arm({arm: spec["eval"] for arm, spec in specs.items()}, split, ex_table)
    pilot_plan = plan.get("pilot") or {}
    sources: Dict[str, Any] = {"arms": {}, "pilot": {arm: reference(e) for arm, e in pilot_plan.items()}}

    judged, reads = {}, {}
    for key, judgment in (("j5", "J5"), ("j6", "J6"), ("j7", "J7"), ("j8", "J8"), ("per_call", "J2"),
                          ("teacher_train_cost", "J3")):
        found = _result(plan, key, judgment)
        judged[key] = found["result"] if found else None
        reads[key] = found["reads"] if found else None
        if found:
            sources[key] = result_reference(plan[key])
    if judged["j8"]:  # the configuration's GPU prices (for a test report, the registered ones)
        gpu = hashlib.sha256(canonical((config.get("modal") or {}).get("gpu_prices") or {})).hexdigest()
        if judged["j8"].get("gpu_prices_sha256") != gpu:
            raise JudgmentError("the plan's J8 priced the GPUs with another table than the configuration's modal.gpu_prices")

    arms, correct, by_question = {}, {}, {}
    for arm, spec in specs.items():
        j3 = read_result(spec["cost"], "J3")
        if j3["reads"].get("eval", {}).get("run_id") != spec["eval"]:
            raise JudgmentError(f"the J3 result of {arm} was not computed with {spec['eval']}")
        cost = j3["result"]
        missing = [k for k in REQUIRED_COST if k not in cost]
        if missing:
            raise JudgmentError(f"the J3 result of {arm} records no {', '.join(missing)}: it cannot say what its cost is")
        basis = (cost.get("slm_cost_basis") or {}).get("basis")
        if basis is not None and (j3["reads"].get("j8") or {}).get("sha256") != (sources.get("j8") or {}).get("sha256"):
            raise JudgmentError(f"the J3 result of {arm} priced its SLM calls with another load test than the plan's j8")
        (summary,) = ex_summary(rows[arm])
        correct[arm], by_question[arm] = correct_of(rows[arm]), cost["by_question"]
        if set(by_question[arm]) != set(correct[arm]):
            raise JudgmentError(f"the J3 result of {arm} does not price the questions its evaluation scored")
        arms[arm] = {"n": summary["n"], "ex": summary["ex"],
                     "by_difficulty": {d: v["ex"] for d, v in summary["by_difficulty"].items()},
                     "run_id": j3["reads"]["run"]["run_id"], "cost_per_correct": costs_of(cost),
                     "cost_label": cost["label"], "lower_bound": cost["lower_bound"], "upper_bound": cost["upper_bound"],
                     "cache_not_reported": cost["cache_not_reported"], "slm_cost_basis": basis,
                     "failed_unbilled": cost["failed_unbilled"], "prices_as_of": cost["prices_as_of"],
                     "prices_sha256": cost["prices"]["sha256"],
                     "calls": cost["calls"], "replaceable_fraction": cost["replaceable_fraction"]}
        sources["arms"][arm] = {"eval": reference(spec["eval"]), "j3": result_reference(spec["cost"])}
    tables = {(a["prices_as_of"], a["prices_sha256"]) for a in arms.values()}
    if len(tables) > 1:
        raise JudgmentError(f"the arms were priced with different price tables {sorted(map(str, tables))}")
    prices = config["prices"]  # the configuration's (for a test report, the registered one)
    configured = (prices.get("as_of"), hashlib.sha256(canonical(prices.get("table") or {})).hexdigest())
    if tables and tables != {configured}:
        raise JudgmentError(f"the arms were priced with another price table than the configuration's {configured}")

    per_call = judged["per_call"]
    if per_call and (per_call.get("mode"), per_call.get("arm"), per_call.get("split")) != ("replay", "B4", split):
        raise JudgmentError(f"the per-call evaluation must be a replay routed as B4 on {split}")
    if judged["j7"]:
        if (reads["j7"].get("j8") or {}).get("sha256") != (sources.get("j8") or {}).get("sha256"):
            raise JudgmentError("the plan's J7 ordered engines by another load test than the plan's j8")
        j7_prices = judged["j7"].get("prices") or {}
        if arms and (j7_prices.get("as_of"), j7_prices.get("sha256")) not in tables:
            raise JudgmentError("the plan's J7 ordered engines by another price table than the arms were priced with")
    replay_several = _registry_bindings(split, arms, registry, judged, reads, pilot_plan, _expected_facts(plan, judged),
                                        in_force)
    if split == "test":  # every evaluation read was scored under the registration in force (eval records it)
        evals = {f"{arm}'s": spec["eval"] for arm, spec in specs.items()}
        for who in ("replay_eval", "teacher_eval"):
            if (reads.get("per_call") or {}).get(who):
                evals[f"the per-call {who.replace('_', ' ')}'s"] = reads["per_call"][who]["run_id"]
        for label, eval_run_id in evals.items():
            recorded = manifest(eval_run_id).get("prereg_hash")
            if recorded != in_force:
                raise JudgmentError(f"{label} evaluation {eval_run_id} was scored under pre-registration {recorded}, "
                                    f"not the one in force ({in_force})")
    if per_call and "call_sites" not in per_call:
        raise JudgmentError("the per-call J2 result does not record which call sites the replay covered")
    coverage = None if not per_call else per_call["call_sites"]
    # K4 and Appendix B rest on the calls with gold alone (T4): the routine is a proxy beside them
    uncovered = sorted(set(GOLD_SITES) - set(coverage)) if coverage is not None else []

    margin = config["thresholds"]["delta_pp"] / 100  # the one fixed margin of every non-inferiority (T3)
    test_of = lambda a, b: plain(noninferiority(a, b, config["seeds"]["bootstrap"], n_boot(config), margin=margin))  # noqa: E731
    d_pilot = pilot_d(pilot_plan, pilot_ids, ex_table)
    tests = {}
    for candidate, reference_arm in PAIRS:
        if candidate in correct and reference_arm in correct:
            several = sorted(set((arms[candidate].get("several_runs") or []) + (arms[reference_arm].get("several_runs") or [])))
            tests[f"{reference_arm}|{candidate}"] = {**planned(test_of(correct[candidate], correct[reference_arm]), d_pilot),
                                                     "several_runs": several or None}
    formats = {}
    for arm, path in (plan.get("format") or {}).items():
        found = read_result(path, "J2")
        if arm not in arms or found["reads"]["run"]["run_id"] != arms[arm]["run_id"]:
            raise JudgmentError(f"the format result of {arm} is not of {arm}'s execution")
        formats[arm] = found["result"]["format_validity"]
        sources.setdefault("format", {})[arm] = result_reference(path)
    gold_tests = {}  # SPEC 7.2 K4: n and Δ per cluster with gold, here each gold call site
    zeroshot = ((judged["j6"] or {}).get("per_call_site") or {}).get((judged["j6"] or {}).get("choice"), {})
    for site in (GOLD_SITES if not uncovered else ()):
        entry = (per_call or {}).get("per_call_site", {}).get(site)
        if not entry or not entry["gold"]:
            continue
        d = None  # its pilot: J6's chosen candidate, zero-shot, against the teacher on the pilot questions' calls
        gold = (zeroshot.get(site) or {}).get("gold")
        ids = sorted(set(pilot_ids) & set(gold["by_question"]["teacher"])) if gold else []
        if ids:
            d = j4.discordance({q: gold["by_question"]["replay"][q] for q in ids},
                               {q: gold["by_question"]["teacher"][q] for q in ids})
        # the replay's several runs, and its teacher's (B0's run), as the registry binding found them
        gold_tests[site] = {**planned(test_of(entry["gold"]["by_question"]["replay"], entry["gold"]["by_question"]["teacher"]), d),
                            "several_runs": replay_several}
    thresholds = config["thresholds"]
    data = {"split": split, "scope": SCOPE, "arms": arms, "tests": tests, "d_pilot": d_pilot,
            "pilot_ids": sorted(pilot_ids, key=int), "gold_tests": gold_tests, "repair_test": gold_tests.get("revise"),
            "per_call_uncovered": uncovered, "formats": formats, "judgments": judged,
            "format_tolerance_pp": thresholds["format_tolerance_pp"], "min_calls": config["allocation"]["min_calls"],
            "v3_min_ratio": config["claims"]["v3_min_ratio"],
            "utilizations": [f"{round(u * 100)}%" for u in sorted(config["cost"]["utilizations"])],
            "registry": registry, "prereg_in_force": in_force,
            "deviations": deviations(ref=remote if split == "test" else "HEAD"), "sources": sources}
    data["cost_ratios"] = cost_ratios(data, by_question, correct, config["seeds"]["bootstrap"], n_boot(config))
    return data


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value.item() if hasattr(value, "item") else value


# ---------------------------------------------------------------- the SPEC §5 map

SCOPE = ("Scope: every verdict below holds for this workload only: one agent (CHESS IR→SS→CG), one domain "
         "(text-to-SQL), code agency, and the same 11 databases in train and test. None is a verdict on agentic "
         "systems in general.")
NO_SLM = "not testable (B5 allocated nothing to an SLM)"
RATIO_ALPHA = Fraction(1, 40)  # each side of the cost ratio's two-sided 95% interval


def outcome(test: Optional[Dict[str, Any]]) -> str:
    """The one reading of a J4 comparison every row shares, against the fixed margin Δ: non-inferior when
    the one-sided 95% lower bound of the difference is above −Δ, worse when the upper bound is below −Δ
    (the same standard, from the other side), inconclusive otherwise."""
    if not test:
        return "no data"
    if test.get("several_runs"):
        return "several runs"
    if test["noninferior"]:
        return "non-inferior"
    return "worse" if test["ci_high"] < -test["delta"] else "inconclusive"


def _several(data, *tests) -> Optional[str]:
    runs = sorted({r for t in tests if t for r in (t.get("several_runs") or [])})
    return f"no verdict (several test runs of one configuration: {', '.join(runs)})" if runs else None


def several_of(data, arms: List[str]) -> Optional[str]:
    """No verdict, naming them, when any of these arms' configurations has several completed test runs
    under the registration in force (SPEC 6.1): the one reading every row that rests on an arm's test
    run uses."""
    named = [a for a in dict.fromkeys(arms) if a and (data["arms"].get(a) or {}).get("several_runs")]
    if not named:
        return None
    runs = sorted({r for a in named for r in data["arms"][a]["several_runs"]})
    return f"no verdict (several test runs of one configuration of {', '.join(named)}: {', '.join(runs)})"


def _cost(data, arm, u) -> Optional[float]:
    costs = data["arms"].get(arm, {}).get("cost_per_correct") or {}
    return costs.get("") if "" in costs else costs.get(u)


def _passes_v1(data, arm) -> bool:
    if arm == "B0":
        return "B0" in data["arms"]
    return outcome(data["tests"].get(f"B0|{arm}")) == "non-inferior"


def _fraction(label: str) -> float:
    return float(label.rstrip("%")) / 100


def present(data, arms) -> List[str]:
    return [a for a in arms if a in data["arms"]]


def slm_arms(data) -> List[str]:
    """The trained arms whose execution had SLM calls: their cost moves with the utilization. A B5 that
    kept every call on an LLM is not one."""
    return [a for a in present(data, TRAINED) if "" not in (data["arms"][a].get("cost_per_correct") or {})]


def b5_without_slm(data) -> bool:
    """B5's allocation (J7) gave no cluster to an SLM: what B5 measures is then an LLM system (T7)."""
    j7 = data["judgments"].get("j7")
    return bool(j7) and "slm" not in j7["allocation"].values()


def best_untrained(data) -> Tuple[Optional[str], Optional[float]]:
    """The cheapest per correct query among the arms without training that are non-inferior to B0 (B0
    always is; B2 counts, design §6.3 T9). None of them has SLM calls, so the cost is one number. An
    arm with several completed test runs is never left out here: `cost_verdict` gives every verdict
    over this set no verdict, naming it."""
    options = [(c, a) for a in present(data, UNTRAINED) if _passes_v1(data, a)
               for c in [_cost(data, a, "")] if c is not None]
    return (min(options)[1], min(options)[0]) if options else (None, None)


def best_slm(data, u) -> Tuple[Optional[str], Optional[float]]:
    """The cheapest per correct query, at utilization u, among the SLM arms that are non-inferior to B0."""
    options = [(c, a) for a in slm_arms(data) if _passes_v1(data, a) for c in [_cost(data, a, u)] if c is not None]
    return (min(options)[1], min(options)[0]) if options else (None, None)


def cost_ratio_ci(bases: Dict[str, Dict[str, Tuple[float, bool]]], slms: Dict[str, Dict[str, Tuple[float, bool]]],
                  seed: int, n_boot: int) -> Dict[str, float]:
    """The ratio of cost per correct query, the cheapest of `bases` ÷ the cheapest of `slms` (each {arm:
    {question: (cost, correct)}}, every arm over the same questions), and its 95% interval from a paired
    bootstrap with the question as the unit: a resample draws questions with replacement, keeps every
    arm's cost and answer of each together, and takes the cheapest arm of each set again, so the interval
    covers the choice of the arms and not only the ratio of the pair the full sample chose. The bounds
    are inverted-CDF quantiles, as J4's: the ⌈n_boot/40⌉-th smallest and largest resampled ratio. An arm
    with no correct answer in a resample costs infinitely per correct query: the ratio is 0 when no SLM
    arm has one."""
    arms = [*bases.values(), *slms.values()]
    if not bases or not slms or not arms[0] or any(set(arm) != set(arms[0]) for arm in arms):
        raise JudgmentError("a cost ratio pairs its arms on the same questions, and on at least one")
    rows = [tuple(x for arm in arms for x in (arm[q][0], int(bool(arm[q][1])))) for q in sorted(arms[0])]

    def ratio(sample: List[tuple]) -> float:
        sums = [sum(column) for column in zip(*sample)]  # per arm: its cost, its correct answers
        per_correct = [sums[i] / sums[i + 1] if sums[i + 1] else math.inf for i in range(0, len(sums), 2)]
        base, slm = min(per_correct[:len(bases)]), min(per_correct[len(bases):])
        if base == math.inf or not slm:
            raise JudgmentError("a cost ratio is undefined when no reference arm has a correct answer or the SLM arm "
                                "has no cost")
        return base / slm
    rng, n = random.Random(seed), len(rows)
    resampled = sorted(ratio([rows[int(rng.random() * n)] for _ in range(n)]) for _ in range(n_boot))
    k = -(-n_boot * RATIO_ALPHA.numerator // RATIO_ALPHA.denominator)
    return {"ratio": ratio(rows), "ci_low": resampled[k - 1], "ci_high": resampled[n_boot - k]}


def cost_ratios(data, by_question: Dict[str, Dict[str, Dict[str, float]]], correct: Dict[str, Dict[str, bool]],
                seed: int, n_boot: int) -> Dict[str, Dict[str, Any]]:
    """Per configured utilization, V3's cost ratio with its interval: the best arm without training ÷ the
    cheapest SLM arm (T9), each read from J3's `by_question` and J1's answers. `base` and `slm` name the
    arms the full sample chose; the interval is over every arm non-inferior to B0, chosen again in each
    resample."""
    out = {}
    base, _ = best_untrained(data)
    pairs = lambda arm, scenario: {q: (by_question[arm][q][scenario], correct[arm][q]) for q in correct[arm]}  # noqa: E731
    untrained = {a: pairs(a, "standard") for a in present(data, UNTRAINED) if _passes_v1(data, a)}
    for u in data["utilizations"]:
        slm, _ = best_slm(data, u)
        if base is None or slm is None:
            continue
        trained = {a: pairs(a, f"standard@{u}") for a in slm_arms(data) if _passes_v1(data, a)}
        out[u] = {"base": base, "slm": slm, **cost_ratio_ci(untrained, trained, seed, n_boot)}
    return out


def cost_reading(found: Dict[str, Any], bar: float) -> str:
    """How one cost ratio reads (T9), by its 95% interval: the SLM arm `meets` the paper's bar (the lower
    bound at or above it), is `cheaper` below the bar (the lower bound above 1), `refutes` V3 (the
    upper bound at or below 1: not cheaper), or is `inconclusive`."""
    if found["ci_low"] >= bar:
        return "meets"
    if found["ci_low"] > 1:
        return "cheaper"
    return "refutes" if found["ci_high"] <= 1 else "inconclusive"


def break_even(costs: Dict[str, Optional[float]], target: float) -> Optional[float]:
    """u*: the utilization at which an SLM arm's cost per correct query reaches `target`. The SLM's share
    of the cost scales with 1/u and the API's does not, so cost(u) = A + G/u, fitted through the lowest
    and the highest utilization of `costs` ({label: cost}). None when the arm never gets there."""
    points = sorted((_fraction(label), cost) for label, cost in costs.items() if label and cost is not None)
    if len(points) < 2:
        return None
    (u_low, c_low), (u_high, c_high) = points[0], points[-1]
    g = (c_low - c_high) / (1 / u_low - 1 / u_high)
    a = c_high - g / u_high
    return g / (target - a) if g > 0 and target > a else None


def tipping_point(data, arms: List[str], target: float) -> str:
    """The lowest u* among these SLM arms: where the cheapest of them starts to cost less than `target`."""
    found = [u for u in (break_even(data["arms"][a].get("cost_per_correct") or {}, target) for a in arms) if u is not None]
    return f"tipping point u* ≈ {100 * min(found):.0f}%" if found else "no break-even utilization"


def cost_labels(data, used: List[str]) -> List[str]:
    """The labels of the costs a verdict rests on."""
    labels = set()
    for arm in used:
        found = data["arms"].get(arm) or {}
        if found.get("cost_label") == "estimated":
            labels.add(f"estimated ({arm})")
        if found.get("lower_bound"):
            labels.add(f"lower bound ({found.get('failed_unbilled')} failed calls of {arm} unpriced)")
        if found.get("upper_bound"):
            labels.add(f"upper bound (cache not reported for {found['cache_not_reported']} calls of {arm})")
        if found.get("slm_cost_basis") and found["slm_cost_basis"] != "measured":
            labels.add(f"{found['slm_cost_basis']} ({arm})")
    return sorted(labels)


def cost_verdict(data, verdict: str, arms: List[str]) -> str:
    """A cost verdict over a set of arms (the minimum is taken over it, so every arm of the set counts,
    not only the winner): no verdict when any of them has several completed test runs, naming it; the
    labels of every arm's cost; inconclusive when any rests on an upper-bound cost, unless the verdict
    is already no data or no verdict."""
    arms = [a for a in dict.fromkeys(arms) if a]
    several = several_of(data, arms)
    if several:
        return several
    labels = cost_labels(data, arms)
    if any(label.startswith("upper bound") for label in labels) and not verdict.startswith(("no data", "no verdict")):
        verdict = "inconclusive (rests on an upper-bound cost)"
    return f"{verdict} [costs: {'; '.join(labels)}]" if labels else verdict


def _power(*tests) -> str:
    """The planned power of these comparisons: from the pilot's d, at the fixed margin (T3)."""
    return ", ".join(f"{t['planned_power']:.2f}" for t in tests if t and t.get("planned_power") is not None) or "—"


def _inconclusive(*tests) -> str:
    """A row that neither confirms nor refutes is inconclusive, with its planned power (SPEC §6.4): a
    low power never makes a comparison untestable."""
    power = _power(*tests)
    return f"inconclusive (planned power {power})" if power != "—" else "inconclusive (no pilot: planned power unknown)"


def v1_verdict(t4) -> str:
    """V1 is judged on B4 alone (T7): B5 can be non-inferior by keeping its calls on the LLM, so B5 against
    B0 belongs to A6."""
    several = _several(None, t4)
    if several:
        return several
    return {"no data": "no data", "non-inferior": "confirms", "worse": "refutes"}.get(outcome(t4)) or _inconclusive(t4)


def _ci(t) -> str:
    return f"CI [{_pp(t['ci_low'])}, {_pp(t.get('ci_high'))}]"


def _against_b0(arm: str, t) -> str:
    return f"{arm} − B0: {_pp(t['diff'])} (Δ {_margin(t['delta'])}, {_ci(t)})" if t else f"{arm} − B0: —"


def claims_map(data: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = []
    t4, t5 = data["tests"].get("B0|B4"), data["tests"].get("B0|B5")
    rows.append({"claim": "V1 / A1 (this workload): SLMs suffice for agent calls (p.3–4)",
                 "result": _against_b0("B4", t4) if t4 else "—", "verdict": v1_verdict(t4), "power": _power(t4)})

    fraction = data["arms"].get("B5", {}).get("replaceable_fraction")
    a4 = {"claim": "A4 / A11 (this workload): calls are narrow, subtasks simple (p.5, p.7)", "power": "—"}
    if b5_without_slm(data):
        rows.append({**a4, "result": "—", "verdict": NO_SLM})
    elif not fraction:
        rows.append({**a4, "result": "—", "verdict": "no data"})
    else:
        rows.append({**a4, "result": (f"B5 replaceable: {_pct(fraction['calls'])} of calls, {_pct(fraction['tokens'])} of tokens, "
                                      f"{_pct(fraction['cost_at_production_price'])} of cost; B5 against B0: {outcome(t5)}"),
                     "verdict": several_of(data, ["B5", "B0"]) or "descriptive (the SPEC fixes no number for 'high')"})

    rows.append(_appendix_b(data))
    rows.append(_a5(data))
    rows.append(_a6(data, t5))
    rows.append({"claim": "A7 (this workload): agent logs become data (p.6)", "result": "needs extension 2 (teacher × gold)",
                 "verdict": "not testable in the core", "power": "—"})
    rows.append(_v3(data, t4, t5))
    rows.append(_av2(data))
    rows.append({"claim": "A2 / A3 (this workload): adapting is fast and cheap (p.5)",
                 "result": "time and cost per adapter: S5 training manifests",
                 "verdict": "descriptive (not a J1–J8 output)", "power": "—"})
    rows.append({"claim": "B2 (this workload): generalist benchmarks guide selection poorly (p.8)", "result": "needs extension 10",
                 "verdict": "not testable in the core", "power": "—"})
    j5 = data["judgments"]["j5"]
    rows.append({"claim": "S3 (this workload): clustering discovers the tasks (p.9)",
                 "result": (f"ARI {j5['ari_call_sites']:.3f} over {j5['k']} clusters; "
                            f"{_pct(j5['truncation']['prompt']['truncated_fraction'])} of prompts cut (what the router embeds)")
                 if j5 else "—",
                 "verdict": "descriptive" if j5 else "no data", "power": "—"})
    rows.append({"claim": "AV1 (this workload): a same-generation LLM always wins (p.7)",
                 "result": f"B4 − B0: {_pp(t4['diff'])}" if t4 else "—",
                 "verdict": (several_of(data, ["B4", "B0"]) or "descriptive (not a direct test)") if t4 else "no data",
                 "power": "—"})
    return rows


def _bracket(data) -> Tuple[str, str]:
    """The lowest and the highest configured utilization: the SLM's worst and best case (T10). A cost
    claim confirms only if it holds at the lowest and refutes only if it fails at the highest."""
    ordered = sorted(data["utilizations"], key=_fraction)
    return ordered[0], ordered[-1]


def _a6(data, t5) -> Dict[str, str]:
    row = {"claim": "A6 (this workload): heterogeneous systems (p.6)", "power": _power(t5)}
    if b5_without_slm(data):
        return {**row, "result": "—", "verdict": NO_SLM}
    low, high = _bracket(data)
    arms = ["B5"] + present(data, UNTRAINED)
    base, base_cost = best_untrained(data)
    at_low, at_high = _cost(data, "B5", low), _cost(data, "B5", high)
    quality = f"{_against_b0('B5', t5)}, {outcome(t5)}"
    if at_low is None or at_high is None or base_cost is None:
        return {**row, "result": quality, "verdict": cost_verdict(data, "no data", arms)}
    if "B5" not in slm_arms(data):  # allocated to an SLM, and no call reached one: an LLM system all the same
        return {**row, "result": quality, "verdict": several_of(data, arms) or "not testable (B5's execution made no SLM call)"}
    result = (f"{quality}; cost per correct query: B5 {_usd(at_low)} at {low} utilization and {_usd(at_high)} at {high}, "
              f"against {_usd(base_cost)} of the best arm without training ({base})")
    if outcome(t5) == "worse":
        verdict = "refutes (B5 is worse than B0)"
    elif at_high >= base_cost:
        verdict = f"refutes (B5 is not cheaper even at {high} utilization)"
    elif at_low >= base_cost:
        verdict = f"utilization-dependent ({tipping_point(data, ['B5'], base_cost)})"
    elif _passes_v1(data, "B5"):
        verdict = f"confirms (B5 is cheaper even at {low} utilization, and non-inferior to B0)"
    else:
        verdict = _inconclusive(t5)
    return {**row, "result": result, "verdict": cost_verdict(data, verdict, arms)}


def _v3(data, t4, t5) -> Dict[str, str]:
    bar = data["v3_min_ratio"]
    row = {"claim": "V3 / A2 (this workload): a 7B SLM is 10–30× cheaper (p.4)", "power": _power(t4, t5)}
    said = {"meets": f"meets the paper's {bar:g}× bar", "cheaper": f"cheaper, below the {bar:g}× bar",
            "refutes": "refutes V3", "inconclusive": "inconclusive"}
    end = {"meets": f"meets the {bar:g}× bar", "cheaper": "cheaper", "refutes": "not cheaper",
           "inconclusive": "the interval holds 1"}  # how one end of the bracket reads inside a verdict
    low, high = _bracket(data)
    arms = present(data, TRAINED) + present(data, UNTRAINED)
    base, base_cost = best_untrained(data)
    ratios = data.get("cost_ratios") or {}
    readings = {u: cost_reading(found, bar) for u, found in ratios.items()}
    result = " · ".join(f"{u}: {found['ratio']:.1f}× (95% CI [{found['ci_low']:.1f}, {found['ci_high']:.1f}]) "
                        f"{found['base']} ÷ {found['slm']}, {said[readings[u]]}"
                        for u, found in sorted(ratios.items(), key=lambda kv: _fraction(kv[0])))
    result = (f"cost per correct query, the best arm without training ÷ the cheapest SLM arm non-inferior to B0, per "
              f"utilization: {result or '—'}")
    passing = [a for a in slm_arms(data) if _passes_v1(data, a)]
    if base is None or not slm_arms(data):
        verdict = "no data"
    elif not passing:
        worse = all(outcome(data["tests"].get(f"B0|{a}")) == "worse" for a in slm_arms(data))
        verdict = ("not testable (every SLM arm is worse than B0)" if worse
                   else f"{_inconclusive(t4, t5)} (no SLM arm is non-inferior to B0)")
    elif low not in ratios or high not in ratios:
        verdict = "no data"
    elif readings[low] == "meets":
        verdict = f"meets the paper's {bar:g}× bar (even at {low} utilization)"
    elif readings[low] == "cheaper":
        verdict = f"cheaper even at {low} utilization, below the {bar:g}× bar there"
    elif readings[high] == "refutes":
        verdict = f"refutes V3 (not cheaper even at {high} utilization)"
    elif readings[low] == "refutes" and readings[high] != "inconclusive":  # decided at both ends, in opposite ways
        verdict = f"utilization-dependent ({tipping_point(data, passing, base_cost)})"
    else:  # an end the interval does not decide: no tipping point is read off the point costs
        verdict = f"inconclusive (at {low} utilization: {end[readings[low]]}; at {high}: {end[readings[high]]})"
    return {**row, "result": result, "verdict": cost_verdict(data, verdict, arms)}


def _av2(data) -> Dict[str, str]:
    row = {"claim": "AV2 / CA3–CA4 (this workload): centralized scale can be cheaper (p.7–8)", "power": "—",
           "result": "B1 vs the cheaper of B4, B5, marginal cost only: the fixed cost and its payback (SPEC §6.6) are not a J1–J8 output"}
    low, high = _bracket(data)
    arms = ["B1"] + present(data, TRAINED)
    b1 = _cost(data, "B1", "")
    cheapest = lambda u: min((c for a in slm_arms(data) for c in [_cost(data, a, u)] if c is not None), default=None)  # noqa: E731
    at_low, at_high = cheapest(low), cheapest(high)
    if b1 is None or at_low is None or at_high is None:
        verdict = "no data"
    elif b1 <= at_high:
        verdict = f"refutes the paper (AV2 wins: B1 costs no more even at {high} utilization)"
    elif at_low < b1:
        verdict = f"does not refute (the SLM arm is cheaper even at {low} utilization)"
    else:
        verdict = f"utilization-dependent ({tipping_point(data, slm_arms(data), b1)})"
    return {**row, "verdict": cost_verdict(data, verdict, arms)}


def _a5(data) -> Dict[str, str]:
    """A5 (T11): B4's format validity against B0's, per call site with at least `allocation.min_calls`
    invocations in both arms. Refutes only when B4 is more than `thresholds.format_tolerance_pp` below
    B0 on one of them; the rates are compared as exact fractions."""
    row = {"claim": "A5 (this workload): one format with a trained SLM is preferable (p.6)", "power": "—"}
    b0f, b4f = data["formats"].get("B0"), data["formats"].get("B4")
    sites = sorted(set(b0f or {}) & set(b4f or {}))
    if not sites:
        return {**row, "result": "—", "verdict": "no data"}
    floor, tolerance = data["min_calls"], Fraction(str(data["format_tolerance_pp"])) / 100
    counted = [s for s in sites if min(b0f[s]["n"], b4f[s]["n"]) >= floor]
    thin = [s for s in sites if s not in counted]
    rate = lambda found: Fraction(found["valid"], found["n"])  # noqa: E731
    below = [s for s in counted if rate(b0f[s]) - rate(b4f[s]) > tolerance]
    shown = lambda s: f"{s} ({_pct(b4f[s]['rate'])} vs {_pct(b0f[s]['rate'])})"  # noqa: E731
    result = (f"B4 more than {data['format_tolerance_pp']:g} pp below B0 on " + ", ".join(shown(s) for s in below) if below
              else f"B4 within {data['format_tolerance_pp']:g} pp of B0, or above, on all {len(counted)} call sites counted")
    if thin:
        result += f"; not counted (fewer than {floor} invocations in an arm): " + ", ".join(shown(s) for s in thin)
    if not counted:
        return {**row, "result": result, "verdict": f"no data (no call site with {floor} invocations in both arms)"}
    return {**row, "result": result, "verdict": several_of(data, ["B4", "B0"]) or ("refutes" if below else "confirms")}


def _appendix_b(data) -> Dict[str, str]:
    """Appendix B is decided on repair alone (T4): the routine's agreement with the teacher is a proxy
    that supports no claim (D15), shown beside it with no effect on the verdict."""
    row = {"claim": "Appendix B (this workload): the LLM keeps unstructured error resolution (p.16)",
           "power": _power(data["repair_test"])}
    per_call, repair = data["judgments"]["per_call"], data["repair_test"]
    if data.get("per_call_uncovered"):
        return {**row, "result": "—", "verdict": f"no data (the replay covers only {', '.join(per_call['call_sites'])})"}
    if not per_call or not repair:
        return {**row, "result": "—", "verdict": "no data"}
    routine = {site: e["agreement"]["rate"] for site, e in per_call["per_call_site"].items()
               if site in ROUTINE and e["agreement"] and e["agreement"]["rate"] is not None}
    repaired = outcome(repair)
    result = (f"repair: SLM − teacher {_pp(repair['diff'])} (Δ {_margin(repair['delta'])}, {_ci(repair)}), {repaired}; "
              f"routine agreement with the teacher, a proxy with no effect on the verdict (D15): "
              + (", ".join(f"{s} {_pct(r)}" for s, r in sorted(routine.items())) or "none measured"))
    verdict = _several(data, repair) or {"worse": "confirms (the SLM is worse on repair)",
                                         "non-inferior": "refutes (the SLM is non-inferior on repair)"}.get(repaired) \
        or _inconclusive(repair)
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

def test_registry(root: Optional[Path] = None, ref: str = "HEAD") -> Dict[str, Any]:
    """Test-split executions as F1 commits them (`registry/test/`), read at `ref` (for a test report,
    the commit origin's main is at, as a fetch just found it): each run's intent (when it started)
    merged with its manifest (how it ended, the facts it used); an intent without a manifest is an
    interrupted execution."""
    root = root or paths.ROOT

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    listed = git("ls-tree", "-r", "--name-only", ref, REGISTRY_DIR)
    if listed.returncode != 0:
        return {"available": False, "reason": scrub(listed.stderr.strip() or "git failed")[:200], "runs": []}
    files = {Path(line).name for line in listed.stdout.splitlines()}
    run_ids = sorted({name.rsplit(".", 2)[0] for name in files if name.endswith((".intent.json", ".manifest.json"))})

    def shown(name: str) -> Optional[Dict[str, Any]]:
        if name not in files:
            return None
        out = git("show", f"{ref}:{REGISTRY_DIR}/{name}")
        if out.returncode != 0:
            raise JudgmentError(f"cannot read {REGISTRY_DIR}/{name} at {ref}")
        return json.loads(out.stdout)
    runs = []
    for run_id in run_ids:
        intent, ended = shown(f"{run_id}.intent.json") or {}, shown(f"{run_id}.manifest.json")
        found = ended or intent
        runs.append({"run_id": run_id, "type": found.get("type"), "arm": found.get("arm"), "engine": found.get("engine"),
                     "status": ended.get("status") if ended else "interrupted (intent, no manifest)",
                     "commit": found.get("commit"), "prereg_hash": found.get("prereg_hash"),
                     "started_at": intent.get("started_at"), "facts": (ended or {}).get("facts")})
    return {"available": True, "runs": runs}


def deviations(root: Optional[Path] = None, ref: str = "HEAD") -> Optional[str]:
    """`prereg/DEVIATIONS.md` as committed at `ref` (for a test report, where origin's main is): one line
    per new registration, with its date and the defect (`bench prereg --replace --reason`). None when
    there is none."""
    shown = subprocess.run(["git", "-C", str(root or paths.ROOT), "show", f"{ref}:{DEVIATIONS}"], capture_output=True, text=True)
    return shown.stdout if shown.returncode == 0 else None


def run_reading(run: Dict[str, Any], in_force: Optional[str]) -> str:
    """How the registry table reads an entry (T13): a run under an earlier registration was superseded;
    one that did not finish is not completed, and neither ever counts toward 'several runs'."""
    if in_force is not None and run["prereg_hash"] != in_force:
        return "superseded by re-registration"
    return "done" if run["status"] == "done" else f"not completed ({run['status']})"


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
               f'<text class="label" x="{left + 96}" y="16">SLM: one point per utilization, {data["utilizations"][0]} (right) to '
               f'{data["utilizations"][-1]} (left)</text>')
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
            labels = [label for label in cost_labels(data, [arm]) if not label.startswith("estimated")]
            usage = a["cost_label"] + (f"; {'; '.join(labels)}" if labels else "")
            rows.append([arm, a["n"], _pct(a["ex"])] + [_pct(a["by_difficulty"].get(d)) for d in difficulties]
                        + [costs, usage])
    parts += [_table(["arm", "n", "EX"] + [f"EX {d}" for d in difficulties] + ["cost per correct query (standard prices)", "usage"], rows), ""]
    parts += ["## The SPEC §5 map", "", data["scope"], "", _table(["claim", "result", "verdict", "planned power"],
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
              NO_SLM[0].upper() + NO_SLM[1:] + "." if b5_without_slm(data) else
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
    if data["per_call_uncovered"]:
        parts += ["## Clusters with gold: per-call non-inferiority on the test inputs (SPEC §7.2 K4)", "",
                  f"No data: the per-call replay covers only {', '.join(per_call['call_sites'])}.", ""]
    elif data["gold_tests"]:
        parts += ["## Clusters with gold: per-call non-inferiority on the test inputs (SPEC §7.2 K4)", "",
                  _table(["call site", "n (questions)", "pilot d", "Δ", "SLM − teacher", "CI", "outcome", "planned power"],
                         [[site, t["n"], "—" if t.get("d_pilot") is None else f"{t['d_pilot']:.3f}", _margin(t["delta"]),
                           _pp(t["diff"]), _ci(t), outcome(t), _power(t)] for site, t in sorted(data["gold_tests"].items())]), ""]
    registry = data["registry"]
    parts += ["## Test registry", ""]
    if not registry["available"]:
        parts.append(f"The registry could not be read from git: {registry['reason']}.")
    elif registry["runs"]:
        parts.append(_table(["execution", "type", "arm", "status", "commit", "pre-registration"],
                            [[r["run_id"], r["type"] or "—", r["arm"] or "—", run_reading(r, data["prereg_in_force"]),
                              (r["commit"] or "—")[:12], (r["prereg_hash"] or "—")[:12]] for r in registry["runs"]]))
    else:
        parts.append("No test-split execution is committed.")
    if data["deviations"]:
        parts += ["", f"## Deviations from the pre-registration (`{DEVIATIONS}`)", "", data["deviations"].strip()]
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
    noninferiority = noninferiority or j4.noninferiority
    if pilot_ids is None:
        from bench.data import pilot_ids as registered_pilot
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
