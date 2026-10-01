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
second), run here through F2's functions. **Every J4 margin comes from the pilot** (SPEC §6.4,
§7.3), measured on calib before training, between the SLM and the production LLM: `d_pilot` is the
discordance J4 measures between B3 (the zero-shot SLM) and B0 on the pre-registered pilot questions
(`bench.data.pilot_sample`), one number for every end-to-end comparison; for repair, between the
zero-shot candidate S4 chose and the teacher on the pilot questions' repair calls (J6's result).
J4 then takes Δ at n = the test's pairs. A comparison with no pilot gets no verdict (J4 returns
`noninferior: None`, `margin_from: "pairs"`), never a pass. **A test report is read only under the
registration in force**, before anything is read: the barrier finds it published, intact and
matching the configuration given (every threshold, seed and the pilot come from it), and the analysis
code is the registered one. **A test report is bound to the test registry**: every run it binds ran
under that registration; every arm's run is a `done` entry; a configuration (type, arm, engine, pre-registration)
with more than one registered test run gets no verdict, and every run is listed (SPEC §6.1); the
facts the B3–B5 runs recorded are the ones the plan's J5, J6 and J7 name; the per-call evaluation
replays the plan's B0 run; and the pilot finished before the first test execution started. The test
registry is read from git as F1 records it: `registry/test/<run_id>.intent.json` before a test
execution starts and `<run_id>.manifest.json` when it ends, as committed at HEAD; an intent with no
manifest is an interrupted execution.

Writes `reports/<sha256>/`: `report.md`, `report.json` (every number, with the judgments it came
from; its sha256 names the directory) and `ex_cost.svg`, the main chart.

**How each row of the SPEC §5 map is decided** (the SPEC fixes the criteria; where it leaves a
term open, the reading is stated here and in the row). A J4 comparison has one outcome, shared by
every row: several runs (no verdict), no pilot (no verdict), not testable (Δ above the cap),
non-inferior (the one-sided 95% lower bound above −Δ), worse (the one-sided 95% **upper** bound
below −Δ: refuting takes the same standard as confirming) or inconclusive (neither). A cost verdict
carries the labels of the costs it rests on (estimated, lower bound, upper bound, extrapolated), and
one that rests on an upper-bound cost is inconclusive; arms priced from different price tables (by
date or by content) are refused.
- V1/A1: confirms if B4 or B5 is non-inferior to B0; refutes if both are worse; otherwise the
  outcomes say why not (no pilot, not testable, inconclusive).
- A4/A11: the replaceable fraction of B5 by call, token and cost, and whether B5 met
  non-inferiority; the SPEC says "high" without a number, so the row is descriptive.
- Appendix B: on the per-call-site evaluation of B4 on the test inputs. The routine (the call sites
  the paper assigns to SLMs: keywords, column filter, table and column selection) passes when each
  agrees with the teacher at `thresholds.concordance_min` or more: the agreement proxy, which D15
  says supports no per-cluster claim, so the verdict says it rests on it. No routine call site
  measured is no data. Refutes when the routine fails, or when repair is non-inferior (a tie: the
  partition is conservative); confirms when repair is worse and the routine passes; otherwise
  repair's outcome says why not.
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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from bench import barrier, paths
from bench.contracts.concordance import GOLD_CALL_SITES as GOLD_SITES
from bench.contracts.facts import read_fact
from bench.contracts.router import ARM_FACTS
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
      pre-registration in force (`in_force`), and each configuration's registered runs are all
      known (several of one configuration: no verdict);
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
        same = sorted(r["run_id"] for r in registry["runs"] if (r["type"], r["arm"], r["engine"], r["prereg_hash"]) == key)
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
                                f"execution started ({first.isoformat()}): its margin would not be pre-registered")
    return replay_several


REQUIRED_COST = ("upper_bound", "cache_not_reported", "lower_bound", "failed_unbilled", "prices")


def gather(plan: Dict[str, Any], config: Dict[str, Any], ex_table: Callable, ex_summary: Callable,
           noninferiority: Callable, pilot_ids: List[str]) -> Dict[str, Any]:
    """Every number of the report, each from a judgment."""
    split = plan["split"]
    registry, in_force = None, None
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

    arms, correct = {}, {}
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
        correct[arm] = correct_of(rows[arm])
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
    uncovered = sorted(set(ROUTINE + GOLD_SITES) - set(coverage)) if coverage is not None else []

    settings = (config["thresholds"]["delta_cap_pp"], config["seeds"]["bootstrap"], n_boot(config))
    d_pilot = pilot_d(pilot_plan, pilot_ids, ex_table, noninferiority, settings)
    tests = {}
    for candidate, reference_arm in PAIRS:
        if candidate in correct and reference_arm in correct:
            test = plain(noninferiority(correct[candidate], correct[reference_arm], *settings, d_pilot=d_pilot))
            several = sorted(set((arms[candidate].get("several_runs") or []) + (arms[reference_arm].get("several_runs") or [])))
            tests[f"{reference_arm}|{candidate}"] = {**test, "several_runs": several or None}
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
            d = plain(noninferiority({q: gold["by_question"]["replay"][q] for q in ids},
                                     {q: gold["by_question"]["teacher"][q] for q in ids}, *settings))["d"]
        # the replay's several runs, and its teacher's (B0's run), as the registry binding found them
        gold_tests[site] = {**plain(noninferiority(entry["gold"]["by_question"]["replay"], entry["gold"]["by_question"]["teacher"],
                                                   *settings, d_pilot=d)), "several_runs": replay_several}
    utilizations_cfg = [f"{round(u * 100)}%" for u in config["cost"]["utilizations"]]
    return {"split": split, "arms": arms, "tests": tests, "d_pilot": d_pilot, "pilot_ids": sorted(pilot_ids, key=int),
            "gold_tests": gold_tests, "repair_test": gold_tests.get("revise"), "per_call_uncovered": uncovered,
            "formats": formats, "judgments": judged, "concordance_min": config["thresholds"]["concordance_min"],
            "v3_min_ratio": config["claims"]["v3_min_ratio"], "utilizations": utilizations_cfg,
            "registry": registry, "sources": sources}


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value.item() if hasattr(value, "item") else value


# ---------------------------------------------------------------- the SPEC §5 map

NO_PILOT = "no verdict (no pilot d)"


def outcome(test: Optional[Dict[str, Any]]) -> str:
    """The one reading of a J4 comparison every row shares. No verdict is read from J4 itself
    (`margin_from == "pairs"` or `noninferior is None`); worse means the one-sided 95% upper bound
    of the difference is below −Δ (the same standard as non-inferior, from the other side)."""
    if not test:
        return "no data"
    if test.get("several_runs"):
        return "several runs"
    if test.get("margin_from") == "pairs":
        return "no pilot"
    if not test["testable"]:
        return "not testable"
    if test["noninferior"] is None:
        return "no pilot"
    if test["noninferior"]:
        return "non-inferior"
    ci_high = test.get("ci_high")
    return "worse" if ci_high is not None and ci_high < -test["delta"] else "inconclusive"


def _several(data, *tests) -> Optional[str]:
    runs = sorted({r for t in tests if t for r in (t.get("several_runs") or [])})
    return f"no verdict (several test runs of one configuration: {', '.join(runs)})" if runs else None


def several_of(data, arms: List[str]) -> Optional[str]:
    """No verdict, naming them, when any of these arms' configurations has several registered test
    runs (SPEC 6.1): the one reading every row that rests on an arm's test run uses."""
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


def utilizations(data) -> List[str]:
    found = {u for arm in SLM_ARMS for u in (data["arms"].get(arm, {}).get("cost_per_correct") or {}) if u}
    return sorted(found, key=lambda u: float(u.rstrip("%")))


def present(data, arms) -> List[str]:
    return [a for a in arms if a in data["arms"]]


def best_untrained(data, u) -> Tuple[Optional[str], Optional[float]]:
    """The cheapest per correct query among the arms without training that are non-inferior to B0.
    An arm with several registered test runs is never left out here: `cost_verdict` gives every
    verdict over this set no verdict, naming it."""
    options = [(c, a) for a in present(data, UNTRAINED) if _passes_v1(data, a)
               for c in [_cost(data, a, u)] if c is not None]
    return (min(options)[1], min(options)[0]) if options else (None, None)


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
    not only the winner): no verdict when any of them has several registered test runs, naming it;
    the labels of every arm's cost; inconclusive when any rests on an upper-bound cost, unless the
    verdict is already no data or no verdict."""
    arms = [a for a in dict.fromkeys(arms) if a]
    several = several_of(data, arms)
    if several:
        return several
    labels = cost_labels(data, arms)
    if any(label.startswith("upper bound") for label in labels) and not verdict.startswith(("no data", "no verdict")):
        verdict = "inconclusive (rests on an upper-bound cost)"
    return f"{verdict} [costs: {'; '.join(labels)}]" if labels else verdict


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
    several = _several(None, t4, t5)
    if several:
        return several
    outcomes = [outcome(t) for t in (t4, t5) if t]
    if not outcomes:
        return "no data"
    if "non-inferior" in outcomes:
        return "confirms"
    if outcomes == ["worse", "worse"]:
        return "refutes"
    return _why_not(outcomes)


def _ci(t) -> str:
    return f"CI [{_pp(t['ci_low'])}, {_pp(t.get('ci_high'))}]"


def claims_map(data: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = []
    t4, t5 = data["tests"].get("B0|B4"), data["tests"].get("B0|B5")
    rows.append({"claim": "V1 / A1: SLMs suffice for agent calls (p.3–4)",
                 "result": "; ".join(f"{a} − B0: {_pp(t['diff'])} (Δ {_margin(t['delta'])} from the {t['margin_from']}, "
                                     f"{_ci(t)})" for a, t in (("B4", t4), ("B5", t5)) if t) or "—",
                 "verdict": v1_verdict(t4, t5), "power": _power(t4, t5)})

    fraction = data["arms"].get("B5", {}).get("replaceable_fraction")
    rows.append({"claim": "A4 / A11: calls are narrow, subtasks simple (p.5, p.7)",
                 "result": (f"B5 replaceable: {_pct(fraction['calls'])} of calls, {_pct(fraction['tokens'])} of tokens, "
                            f"{_pct(fraction['cost_at_production_price'])} of cost; B5 against B0: {outcome(t5)}")
                 if fraction else "—",
                 "verdict": (several_of(data, ["B5", "B0"]) or "descriptive (the SPEC fixes no number for 'high')")
                 if fraction else "no data", "power": "—"})

    rows.append(_appendix_b(data))
    rows.append(_a5(data))

    untrained, trained_arms = present(data, UNTRAINED), present(data, TRAINED)
    a6, v3, av2 = [], [], []
    for u in utilizations(data) or [""]:
        base, base_cost = best_untrained(data, u)
        b5_cost = _cost(data, "B5", u)
        if b5_cost is None or base_cost is None:
            a6.append((u, cost_verdict(data, "no data", ["B5"] + untrained)))
        elif b5_cost >= base_cost:
            a6.append((u, cost_verdict(data, "refutes", ["B5"] + untrained)))
        else:
            a6.append((u, cost_verdict(data, "confirms" if _passes_v1(data, "B5") else _why_not([outcome(t5)]),
                                       ["B5"] + untrained)))
        trained = [(c, a) for a in trained_arms if _passes_v1(data, a) for c in [_cost(data, a, u)] if c is not None]
        if base_cost is None:
            v3.append((u, cost_verdict(data, "no data", trained_arms + untrained)))
        elif not trained:
            v3.append((u, cost_verdict(data, f"{_why_not([outcome(t) for t in (t4, t5) if t])} (no trained arm passes V1)",
                                       trained_arms + untrained)))
        else:
            cheapest, arm = min(trained)
            ratio = base_cost / cheapest
            v3.append((u, cost_verdict(data, f"{'confirms' if ratio >= data['v3_min_ratio'] else 'refutes'} "
                                             f"({ratio:.1f}× vs {base})", trained_arms + untrained)))
        slm_costs = [(c, a) for a in trained_arms for c in [_cost(data, a, u)] if c is not None]
        b1 = _cost(data, "B1", u)
        if b1 is None or not slm_costs:
            av2.append((u, cost_verdict(data, "no data", ["B1"] + trained_arms)))
        else:
            cheapest, arm = min(slm_costs)
            av2.append((u, cost_verdict(data, "refutes the paper (AV2 wins)" if b1 <= cheapest else "does not refute",
                                        ["B1"] + trained_arms)))
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
                            f"{_pct(j5['truncation']['prompt']['truncated_fraction'])} of prompts cut (what the router embeds)")
                 if j5 else "—",
                 "verdict": "descriptive" if j5 else "no data", "power": "—"})
    rows.append({"claim": "AV1: a same-generation LLM always wins (p.7)",
                 "result": f"B4 − B0: {_pp(t4['diff'])}" if t4 else "—",
                 "verdict": (several_of(data, ["B4", "B0"]) or "descriptive (not a direct test)") if t4 else "no data",
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
    return {**row, "result": result, "verdict": several_of(data, ["B4", "B0"]) or ("refutes" if worse else "confirms")}


def _appendix_b(data) -> Dict[str, str]:
    row = {"claim": "Appendix B: the LLM keeps unstructured error resolution (p.16)", "power": _power(data["repair_test"])}
    per_call, repair = data["judgments"]["per_call"], data["repair_test"]
    if data.get("per_call_uncovered"):
        return {**row, "result": "—", "verdict": f"no data (the replay covers only {', '.join(per_call['call_sites'])})"}
    if not per_call or not repair:
        return {**row, "result": "—", "verdict": "no data"}
    routine = {site: e["agreement"]["rate"] for site, e in per_call["per_call_site"].items()
               if site in ROUTINE and e["agreement"] and e["agreement"]["rate"] is not None}
    repaired = outcome(repair)
    result = (f"repair: SLM − teacher {_pp(repair['diff'])} (Δ {_margin(repair['delta'])} from the {repair['margin_from']}, "
              f"{_ci(repair)}), {repaired}; routine agreement: "
              + (", ".join(f"{s} {_pct(r)}" for s, r in sorted(routine.items())) or "none measured"))
    several = _several(data, repair)
    if several:
        return {**row, "result": result, "verdict": several}
    if not routine:
        return {**row, "result": result, "verdict": "no data (no routine call site measured)"}
    passes_routine = all(rate >= data["concordance_min"] for rate in routine.values())
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
    if data["per_call_uncovered"]:
        parts += ["## Clusters with gold: per-call non-inferiority on the test inputs (SPEC §7.2 K4)", "",
                  f"No data: the per-call replay covers only {', '.join(per_call['call_sites'])}.", ""]
    elif data["gold_tests"]:
        parts += ["## Clusters with gold: per-call non-inferiority on the test inputs (SPEC §7.2 K4)", "",
                  _table(["call site", "n (questions)", "pilot d", "Δ", "SLM − teacher", "CI", "outcome", "power"],
                         [[site, t["n"], "—" if t.get("d_pilot") is None else f"{t['d_pilot']:.3f}", _margin(t["delta"]),
                           _pp(t["diff"]), _ci(t), outcome(t), _power(t)] for site, t in sorted(data["gold_tests"].items())]), ""]
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
