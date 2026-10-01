"""R · the report: the SPEC §5 map rules, the test registry, and the whole pipeline end to end on a
fake execution (S2 → S3 → S4 → S6, the costs, the per-call evaluation, the report)."""
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from bench import paths, report
from bench.contracts import clusters, facts
from bench.judge.base import JudgmentError, read_jsonl, read_result, relative, write_result
from bench.judge.j1 import ex_summary, ex_table
from bench.judge.j4 import noninferiority
from fixtures.world import unit

VOCABULARY = ("confirms", "refutes", "inconclusive", "not testable", "descriptive", "no data", "does not refute",
              "no verdict", "meets the paper's", "cheaper even at", "utilization-dependent")


def j4(ok=True, diff=-0.01, pilot=0.1, ci_low=None, ci_high=None):
    """A J4 result as the report holds it: J4's test at the fixed margin, with the pilot's d and the
    planned power beside it (None without a pilot)."""
    return {"n": 498, "d": 0.1, "delta": 0.05, "diff": diff, "ci_low": ci_low if ci_low is not None else (0.0 if ok else -0.09),
            "ci_high": ci_high if ci_high is not None else diff + 0.05, "noninferior": ok, "power": 0.97,
            "d_pilot": pilot, "planned_power": 0.8 if pilot is not None else None}


def j4_results(ok4=True, ok5=True, diff=-0.01, pilot=0.1):
    return {"B0|B4": j4(ok4, diff, pilot), "B0|B5": j4(ok5, diff, pilot)}


def data_with(tests, costs, formats=None, repair=None, per_call=None, ratios=None, j7=None):
    arms = {arm: {"cost_per_correct": c, "replaceable_fraction": None} for arm, c in costs.items()}
    return {"arms": arms, "tests": tests, "formats": formats or {}, "repair_test": repair,
            "judgments": {"j5": None, "per_call": per_call, "j7": j7}, "v3_min_ratio": 3,
            "utilizations": ["20%", "50%", "100%"], "min_calls": 100, "format_tolerance_pp": 1, "cost_ratios": ratios or {}}


def ratio(point, low, high, base="B0", slm="B4"):
    """V3's cost ratio at one utilization, with its 95% interval, as `report.cost_ratios` gives it."""
    return {"base": base, "slm": slm, "ratio": point, "ci_low": low, "ci_high": high}


def row(rows, prefix):
    return next(r for r in rows if r["claim"].startswith(prefix))


def test_the_outcome_of_a_comparison():
    assert report.outcome(None) == "no data"
    assert report.outcome(j4(True)) == "non-inferior"
    assert report.outcome(j4(False, diff=-0.2)) == "worse"
    assert report.outcome(j4(False, diff=-0.02)) == "inconclusive"
    assert report.outcome(j4(True, pilot=None)) == "non-inferior"  # the margin is fixed: no pilot, still a verdict
    assert report.outcome({**j4(True), "several_runs": ["a", "b"]}) == "several runs"


def test_the_verdict_on_each_side_of_minus_5_pp():
    """Against the fixed Δ = 5 p.p. (T3): non-inferior with the lower bound above −Δ, worse with the upper
    bound below it, inconclusive in between. No comparison is 'not testable'."""
    assert report.outcome(j4(True, ci_low=-0.0499)) == "non-inferior"
    assert report.outcome(j4(False, diff=-0.06, ci_low=-0.09, ci_high=-0.0501)) == "worse"
    assert report.outcome(j4(False, diff=-0.06, ci_low=-0.09, ci_high=-0.0499)) == "inconclusive"  # the point is below −Δ, not the bound
    assert report.outcome(j4(False, diff=-0.06, ci_low=-0.09, ci_high=-0.05)) == "inconclusive"    # a bound at −Δ is not below it
    assert report.outcome(j4(False, ci_low=-0.0501, ci_high=0.01)) == "inconclusive"


def test_planned_power_is_j4s_at_the_pilots_d_and_the_tests_n():
    """The pilot's d gives the planned power and nothing else: by hand, at d = 0.20, n = 498, Δ = 0.05,
    Φ(0.05 / sqrt(0.20 / 498) − 1.6448536) = Φ(2.494995 − 1.6448536) = Φ(0.850141) = 0.8024."""
    test = {"n": 498, "delta": 0.05, "noninferior": False, "ci_high": 0.0}
    with_pilot = report.planned(test, 0.20)
    assert with_pilot["d_pilot"] == 0.20 and with_pilot["planned_power"] == pytest.approx(0.8024, abs=1e-4)
    assert with_pilot["delta"] == 0.05 and report.outcome(with_pilot) == "inconclusive"  # the margin did not move
    assert report.planned(test, None)["planned_power"] is None


def test_v1_is_judged_on_b4_alone():
    """T7: B5 can be non-inferior by keeping its calls on the LLM, so it neither confirms nor refutes V1."""
    v1 = lambda tests: row(report.claims_map(data_with(tests, {"B0": {"": 1.0}})), "V1")  # noqa: E731
    assert v1(j4_results())["verdict"] == "confirms"
    assert v1(j4_results(ok4=False, ok5=True))["verdict"] == "inconclusive (planned power 0.80)"  # B5 passing confirms nothing
    assert v1({"B0|B4": j4(False, diff=-0.2), "B0|B5": j4(True)})["verdict"] == "refutes"         # B4 worse, whatever B5 does
    assert v1({"B0|B4": j4(True), "B0|B5": j4(False, diff=-0.2)})["verdict"] == "confirms"
    assert v1({"B0|B5": j4(True)})["verdict"] == "no data"                                        # B5 alone is no evidence
    shown = v1(j4_results())
    assert "B5" not in shown["result"] and shown["result"].startswith("B4 − B0: -1.0 pp (Δ 5.0 pp, CI [") and shown["power"] == "0.80"
    # without a pilot the margin is still the fixed one: a verdict, with no planned power
    no_pilot = v1(j4_results(pilot=None))
    assert no_pilot["verdict"] == "confirms" and no_pilot["power"] == "—"
    assert v1(j4_results(False, False, pilot=None))["verdict"] == "inconclusive (no pilot: planned power unknown)"


def test_every_claim_is_scoped_to_this_workload():
    """T15: a scope line opens the map, and each claim says '(this workload)'."""
    rows = report.claims_map(data_with(j4_results(), {"B0": {"": 1.0}}))
    assert len(rows) == 12 and all("(this workload): " in r["claim"] for r in rows)
    for words in ("this workload only", "one agent", "one domain", "code agency", "the same 11 databases in train and test"):
        assert words in report.SCOPE


# ---------------------------------------------------------------- the cost rules (T9, T10)

def test_the_cost_ratio_and_its_interval_by_hand():
    """Two questions, both arms right on both: the reference pays $1 and $3, the SLM arm $1 each. A
    resample of two is (q1, q1), (q1, q2), (q2, q1) or (q2, q2): the ratio of cost per correct query is
    1 (probability 1/4), 2 (1/2) or 3 (1/4). So the point is 2 and the 95% interval, [1, 3]."""
    base, slm = {"B0": {"q1": (1.0, True), "q2": (3.0, True)}}, {"B4": {"q1": (1.0, True), "q2": (1.0, True)}}
    assert report.cost_ratio_ci(base, slm, seed=7, n_boot=4000) == {"ratio": 2.0, "ci_low": 1.0, "ci_high": 3.0}
    assert report.cost_ratio_ci(base, slm, seed=7, n_boot=4000) == report.cost_ratio_ci(base, slm, seed=7, n_boot=4000)
    # the answers count: with the SLM arm wrong on q2, its cost per correct query doubles and the ratio halves;
    # a resample of q2 alone (probability 1/4) leaves it no correct answer, an infinite cost per correct query: ratio 0
    wrong = report.cost_ratio_ci(base, {"B4": {"q1": (1.0, True), "q2": (1.0, False)}}, seed=7, n_boot=400)
    assert wrong["ratio"] == 1.0 and wrong["ci_low"] == 0.0
    with pytest.raises(JudgmentError, match="same questions"):
        report.cost_ratio_ci(base, {"B4": {"q1": (1.0, True)}}, seed=7, n_boot=10)
    with pytest.raises(JudgmentError, match="same questions"):
        report.cost_ratio_ci(base, {}, seed=7, n_boot=10)
    with pytest.raises(JudgmentError, match="undefined"):
        report.cost_ratio_ci({"B0": {"q1": (1.0, False), "q2": (3.0, False)}}, slm, seed=7, n_boot=10)


def test_the_cost_ratios_interval_covers_the_choice_of_the_arms():
    """Two reference arms that swap places: X pays $1 and $3, Y pays $3 and $1, the SLM arm $1 each. Each
    resample takes the cheaper of X and Y again: (q1, q1) gives X at $1, (q2, q2) gives Y at $1, and a
    mixed resample gives both at $2. So the ratio is 1 or 2, half the time each: the interval is [1, 2].
    Bootstrapping only the pair the full sample chose (X, on the tie) would give [1, 3]."""
    bases = {"X": {"q1": (1.0, True), "q2": (3.0, True)}, "Y": {"q1": (3.0, True), "q2": (1.0, True)}}
    slm = {"B4": {"q1": (1.0, True), "q2": (1.0, True)}}
    assert report.cost_ratio_ci(bases, slm, seed=7, n_boot=4000) == {"ratio": 2.0, "ci_low": 1.0, "ci_high": 2.0}
    # and on the SLM side: the cheaper of two SLM arms in each resample
    slms = {"B4": {"q1": (1.0, True), "q2": (3.0, True)}, "B5": {"q1": (3.0, True), "q2": (1.0, True)}}
    flat = {"B0": {"q1": (6.0, True), "q2": (6.0, True)}}
    assert report.cost_ratio_ci(flat, slms, seed=7, n_boot=4000) == {"ratio": 3.0, "ci_low": 3.0, "ci_high": 6.0}
    # a reference arm with no correct answer in a resample is not the cheapest: the other one is taken
    lame = {"X": {"q1": (1.0, True), "q2": (1.0, False)}, "Y": {"q1": (2.0, True), "q2": (2.0, True)}}
    found = report.cost_ratio_ci(lame, slm, seed=7, n_boot=400)
    assert found["ratio"] == 2.0 and found["ci_low"] == 1.0 and found["ci_high"] == 2.0


def test_the_cost_ratios_interval_is_the_two_sided_95_percent_one():
    """Three questions, the reference paying $1, $2 and $6 and the SLM arm $1 each, all right: a resample's
    ratio is the mean of three draws from {1, 2, 6}. It is 1 only when all three draws are the first
    question, and 6 only when all are the third: probability 1/27 = 3.7% each, more than the 2.5% a 95%
    interval leaves on each side and less than the 5% of a 90% one. So the 95% interval is [1, 6]; a 90%
    one would stop at the next values, 4/3 and 14/3."""
    base = {"q1": (1.0, True), "q2": (2.0, True), "q3": (6.0, True)}
    slm = {q: (1.0, True) for q in base}
    assert report.cost_ratio_ci({"B0": base}, {"B4": slm}, seed=7, n_boot=4000) == {"ratio": 3.0, "ci_low": 1.0, "ci_high": 6.0}


def test_the_cost_ratio_is_resampled_paired_by_question():
    """Arms with the same cost and answer on every question: the ratio is 1 in every resample, which
    resampling the two arms apart would not give. An interval that holds the bar exactly meets it."""
    arm = {str(q): (1.0 + q % 3, q % 4 != 0) for q in range(40)}
    assert report.cost_ratio_ci({"B0": arm}, {"B4": dict(arm)}, seed=3, n_boot=200) == {"ratio": 1.0, "ci_low": 1.0, "ci_high": 1.0}
    thrice = {q: (3 * cost, ok) for q, (cost, ok) in arm.items()}
    found = report.cost_ratio_ci({"B0": thrice}, {"B4": arm}, seed=3, n_boot=200)
    assert found["ratio"] == pytest.approx(3.0) and found["ci_low"] == pytest.approx(3.0) == found["ci_high"]


def test_the_ratio_is_the_best_arm_without_training_over_the_cheapest_slm_arm():
    """T9: per configured utilization, from J3's by_question; B2 stays in the comparator set."""
    tests = {**j4_results(), "B0|B1": j4(False), "B0|B2-cheap": j4(True)}
    costs = {"B0": {"": 1.0}, "B1": {"": 0.1}, "B2-cheap": {"": 0.6}, "B4": {"20%": 0.4, "50%": 0.2, "100%": 0.1},
             "B5": {"20%": 0.3, "50%": 0.25, "100%": 0.2}}
    data = data_with(tests, costs)
    questions = ["1", "2"]
    correct = {arm: {q: True for q in questions} for arm in costs}
    priced = lambda arm: {q: {(f"standard@{u}" if u else "standard"): c / 2 for u, c in costs[arm].items()} for q in questions}  # noqa: E731
    found = report.cost_ratios(data, {arm: priced(arm) for arm in costs}, correct, seed=1, n_boot=50)
    # B1 is cheaper but not non-inferior to B0; B2-cheap is: it is the best arm without training
    assert {u: (r["base"], r["slm"]) for u, r in found.items()} == {"20%": ("B2-cheap", "B5"), "50%": ("B2-cheap", "B4"),
                                                                    "100%": ("B2-cheap", "B4")}
    assert [found[u]["ratio"] for u in ("20%", "50%", "100%")] == [pytest.approx(2.0), pytest.approx(3.0), pytest.approx(6.0)]
    # an SLM arm that is not non-inferior to B0 is not compared
    failing = report.cost_ratios(data_with({**tests, **j4_results(False, False)}, costs), {arm: priced(arm) for arm in costs},
                                 correct, seed=1, n_boot=50)
    assert failing == {}


def test_how_a_cost_ratio_reads():
    """T9: by the 95% interval, against 1 (cheaper at all) and the paper's bar (claims.v3_min_ratio)."""
    assert report.cost_reading(ratio(4.0, 3.0, 5.0), 3) == "meets"         # the lower bound at the bar
    assert report.cost_reading(ratio(4.0, 2.99, 5.0), 3) == "cheaper"      # 4× by the point, but the bar is not met
    assert report.cost_reading(ratio(2.0, 1.01, 3.5), 3) == "cheaper"      # 2× cheaper is no refutation
    assert report.cost_reading(ratio(1.5, 1.0, 2.5), 3) == "inconclusive"  # cheaper only with the lower bound above 1
    assert report.cost_reading(ratio(0.8, 0.5, 1.0), 3) == "refutes"       # the upper bound at 1: not cheaper
    assert report.cost_reading(ratio(0.9, 0.5, 1.01), 3) == "inconclusive"


def test_the_tipping_point_is_interpolated_on_a_cost_that_falls_with_the_utilization():
    """cost(u) = A + G/u through the lowest and highest utilization. By hand, through (20%, $2.0) and
    (100%, $0.4): G = (2.0 − 0.4) / (1/0.2 − 1/1.0) = 0.4 and A = 0.4 − 0.4/1.0 = 0, so the arm costs $1
    at u* = 0.4 / (1 − 0) = 40%. With an API share, (20%, $1.3) and (100%, $0.5): G = 0.8 / 4 = 0.2,
    A = 0.3, and $1 is reached at u* = 0.2 / (1 − 0.3) = 28.6%."""
    assert report.break_even({"20%": 2.0, "50%": 0.8, "100%": 0.4}, 1.0) == pytest.approx(0.4)
    assert report.break_even({"20%": 1.3, "100%": 0.5}, 1.0) == pytest.approx(0.2 / 0.7)
    assert report.break_even({"20%": 1.3, "100%": 0.5}, 0.25) is None     # its API share alone costs more than that
    assert report.break_even({"": 0.5}, 1.0) is None                      # no SLM share: nothing moves
    data = data_with({}, {"B4": {"20%": 2.0, "100%": 0.4}, "B5": {"20%": 1.3, "100%": 0.5}})
    assert report.tipping_point(data, ["B4", "B5"], 1.0) == "tipping point u* ≈ 29%"  # the first arm to get there
    assert report.tipping_point(data, ["B5"], 0.25) == "no break-even utilization"


def test_v3_confirms_at_the_lowest_utilization_and_refutes_at_the_highest():
    """T10: the SLM arm's cost falls as its utilization rises, so V3 holds only if it holds at the lowest
    configured utilization and fails only if it fails at the highest."""
    def v3(low, mid, high, costs=None, tests=None):
        costs = costs or {"B0": {"": 1.0}, "B4": {"20%": 0.25, "50%": 0.1, "100%": 0.05}}
        ratios = {u: found for u, found in (("20%", low), ("50%", mid), ("100%", high)) if found}
        return row(report.claims_map(data_with(tests or j4_results(), costs, ratios=ratios)), "V3")
    met = v3(ratio(4, 3.2, 5), ratio(10, 8, 12), ratio(20, 16, 24))
    assert met["verdict"] == "meets the paper's 3× bar (even at 20% utilization)"
    assert "20%: 4.0× (95% CI [3.2, 5.0]) B0 ÷ B4, meets the paper's 3× bar · 50%: 10.0× (95% CI [8.0, 12.0])" in met["result"]
    # 2× cheaper at the worst case: the old rule read 'refutes' (below 3×); it supports 'more economical'.
    # The bar is met only if met at the lowest utilization: here it is met at 100%, and the verdict is the 20% one
    assert v3(ratio(2, 1.5, 2.5), ratio(5, 4, 6), ratio(10, 8, 12))["verdict"] == \
        "cheaper even at 20% utilization, below the 3× bar there"
    assert v3(ratio(0.5, 0.4, 0.6), ratio(0.7, 0.6, 0.8), ratio(0.9, 0.8, 1.0))["verdict"] == \
        "refutes V3 (not cheaper even at 100% utilization)"
    # decided at both ends, in opposite ways: not cheaper at 20%, cheaper at 100%. B4's cost is $0.4/u,
    # which meets B0's $1 at u* = 40%, inside the configured range
    dear = {"B0": {"": 1.0}, "B4": {"20%": 2.0, "50%": 0.8, "100%": 0.4}}
    assert v3(ratio(0.5, 0.4, 0.6), ratio(1.25, 1.1, 1.4), ratio(2.5, 2.2, 2.8), dear)["verdict"] == \
        "utilization-dependent (tipping point u* ≈ 40%)"
    steep = {"B0": {"": 1.0}, "B4": {"20%": 2.0, "50%": 0.5, "100%": 0.25}}
    assert v3(ratio(0.5, 0.4, 0.6), ratio(2, 1.8, 2.2), ratio(4, 3.5, 4.5), steep)["verdict"].startswith("utilization-dependent")
    # an end the interval does not decide gets no tipping point: the point costs would put it outside the
    # configured range (here B4's $1.05 at 100% never reaches B0's $1), or below the lowest utilization
    never = {"B0": {"": 1.0}, "B4": {"20%": 2.0, "50%": 1.3, "100%": 1.05}}
    assert v3(ratio(0.5, 0.4, 0.6), ratio(0.77, 0.7, 0.85), ratio(0.95, 0.85, 1.06), never)["verdict"] == \
        "inconclusive (at 20% utilization: not cheaper; at 100%: the interval holds 1)"
    close = {"B0": {"": 1.0}, "B4": {"20%": 0.9, "50%": 0.55, "100%": 0.4}}
    assert v3(ratio(1.11, 0.8, 1.4), ratio(1.8, 1.5, 2.1), ratio(2.5, 2.2, 2.8), close)["verdict"] == \
        "inconclusive (at 20% utilization: the interval holds 1; at 100%: cheaper)"
    assert v3(ratio(1.11, 0.8, 1.4), ratio(1.8, 1.5, 2.1), ratio(4, 3.5, 4.5), close)["verdict"] == \
        "inconclusive (at 20% utilization: the interval holds 1; at 100%: meets the 3× bar)"
    assert v3(ratio(1.1, 0.8, 1.4), ratio(1.2, 0.9, 1.5), ratio(1.3, 0.95, 1.6))["verdict"] == \
        "inconclusive (at 20% utilization: the interval holds 1; at 100%: the interval holds 1)"
    # no SLM arm non-inferior to B0: no ratio, and the comparisons say why
    failing = v3(None, None, None, tests=j4_results(False, False))
    assert failing["verdict"] == "inconclusive (planned power 0.80, 0.80) (no SLM arm is non-inferior to B0)"
    both_worse = {"B0|B4": j4(False, diff=-0.2), "B0|B5": j4(False, diff=-0.2)}  # decided, not inconclusive: V1 and A6 refute
    assert v3(None, None, None, tests=both_worse)["verdict"] == "not testable (every SLM arm is worse than B0)"
    capped = data_with(both_worse, {"B0": {"": 1.0}, "B4": {"20%": 0.25, "100%": 0.05}})
    capped["arms"]["B0"].update(upper_bound=True, cache_not_reported=7)  # nor does an upper-bound cost make it inconclusive
    assert row(report.claims_map(capped), "V3")["verdict"] == "not testable (every SLM arm is worse than B0)"
    assert row(report.claims_map(data_with(j4_results(), {"B0": {"": 1.0}})), "V3")["verdict"] == "no data"


def test_a_b5_without_slm_calls_is_no_slm_arm():
    """A B5 that kept every call on an LLM has one cost at every utilization: V3 and AV2 do not read it as
    the SLM arm."""
    costs = {"B0": {"": 1.0}, "B1": {"": 0.5}, "B5": {"": 0.9}}
    data = data_with(j4_results(), costs)
    assert report.slm_arms(data) == [] and report.best_slm(data, "20%") == (None, None)
    assert row(report.claims_map(data), "V3")["verdict"] == "no data"
    assert row(report.claims_map(data), "AV2")["verdict"] == "no data"
    costs["B4"] = {"20%": 0.4, "50%": 0.2, "100%": 0.1}
    assert report.slm_arms(data_with(j4_results(), costs)) == ["B4"]


def test_a6_takes_b5_against_b0_under_the_utilization_bracket():
    """T7 and T10: B5 against B0 is A6's; it confirms only if B5 is cheaper at the lowest utilization (and
    non-inferior), and refutes only if B5 is not cheaper at the highest."""
    tests = {**j4_results(), "B0|B1": j4(False)}  # B1 is not non-inferior to B0: the best arm without training is B0, $1

    def a6(b5, tests=tests, **more):
        return row(report.claims_map(data_with(tests, {"B0": {"": 1.0}, "B1": {"": 0.2}, "B5": b5}, **more)), "A6")
    cheap = a6({"20%": 0.4, "50%": 0.2, "100%": 0.05})
    assert cheap["verdict"] == "confirms (B5 is cheaper even at 20% utilization, and non-inferior to B0)"
    assert cheap["result"] == ("B5 − B0: -1.0 pp (Δ 5.0 pp, CI [+0.0 pp, +4.0 pp]), non-inferior; cost per correct query: B5 $0.4 "
                               "at 20% utilization and $0.05 at 100%, against $1 of the best arm without training (B0)")
    assert a6({"20%": 2.0, "50%": 1.6, "100%": 1.5})["verdict"] == "refutes (B5 is not cheaper even at 100% utilization)"
    assert a6({"20%": 2.0, "50%": 1.6, "100%": 1.0})["verdict"].startswith("refutes")  # the same cost is not cheaper
    # cheaper at 100% only: $0.4/u meets $1 at u* = 40% (the old rule read 'refutes' at 20% and 'confirms' at 100%)
    assert a6({"20%": 2.0, "50%": 0.8, "100%": 0.4})["verdict"] == "utilization-dependent (tipping point u* ≈ 40%)"
    assert a6({"20%": 1.0, "50%": 0.4, "100%": 0.2})["verdict"] == "utilization-dependent (tipping point u* ≈ 20%)"  # the same cost at 20%
    # cheaper everywhere, but not shown non-inferior to B0: no confirmation
    assert a6({"20%": 0.4, "100%": 0.05}, tests=j4_results(ok5=False))["verdict"] == "inconclusive (planned power 0.80)"
    # B5 against B0 is this row's (T7): a B5 that is worse than B0 refutes it, however cheap
    worse = {"B0|B4": j4(True), "B0|B5": j4(False, diff=-0.2)}
    assert a6({"20%": 0.4, "100%": 0.05}, tests=worse)["verdict"] == "refutes (B5 is worse than B0)"
    capped = data_with(worse, {"B0": {"": 1.0}, "B5": {"20%": 0.4, "100%": 0.05}})
    capped["arms"]["B0"].update(upper_bound=True, cache_not_reported=7)  # a verdict on quality rests on no cost
    assert row(report.claims_map(capped), "A6")["verdict"] == "refutes (B5 is worse than B0)"
    capped["arms"]["B0"]["several_runs"] = ["b0-a", "b0-b"]
    assert row(report.claims_map(capped), "A6")["verdict"].startswith("no verdict (several test runs")
    # allocated to an SLM, and no call of the execution reached one: an LLM system, which says nothing of A6
    llm_only = a6({"": 0.4}, j7={"allocation": {"c0": "slm"}})
    assert llm_only["verdict"] == "not testable (B5's execution made no SLM call)"
    assert a6({"20%": 0.4, "100%": 0.05}, tests=j4_results(ok4=False))["verdict"].startswith("confirms")  # B4 is V1's, not A6's
    assert row(report.claims_map(data_with(tests, {"B0": {"": 1.0}})), "A6")["verdict"] == "no data"


def test_a_b5_that_allocated_nothing_to_an_slm_is_not_testable():
    """T7: A6 and the replaceable fraction read 'not testable' when J7 gave no cluster to an SLM."""
    j7 = {"allocation": {"c0": "production_llm", "c1": "cheap_alt"}}
    data = data_with(j4_results(), {"B0": {"": 1.0}, "B5": {"": 0.9}}, j7=j7)
    data["arms"]["B5"]["replaceable_fraction"] = None
    rows = report.claims_map(data)
    assert row(rows, "A6")["verdict"] == row(rows, "A4")["verdict"] == "not testable (B5 allocated nothing to an SLM)"
    assert row(rows, "V1")["verdict"] == "confirms"  # B4's own verdict is untouched
    with_slm = data_with(j4_results(), {"B0": {"": 1.0}, "B5": {"20%": 0.4, "100%": 0.2}}, j7={"allocation": {"c0": "slm"}})
    assert row(report.claims_map(with_slm), "A6")["verdict"].startswith("confirms")


def test_av2_wins_only_if_b1_is_no_dearer_at_the_highest_utilization():
    """T10: the SLM arm costs $0.1/u here. B1 refutes the paper only if it costs no more at 100% too."""
    b4 = {"20%": 0.5, "50%": 0.2, "100%": 0.1}
    av2 = lambda b1: row(report.claims_map(data_with(j4_results(), {"B0": {"": 1.0}, "B1": {"": b1}, "B4": b4})), "AV2")["verdict"]  # noqa: E731
    assert av2(0.05) == "refutes the paper (AV2 wins: B1 costs no more even at 100% utilization)"
    assert av2(0.1) == "refutes the paper (AV2 wins: B1 costs no more even at 100% utilization)"
    assert av2(0.6) == "does not refute (the SLM arm is cheaper even at 20% utilization)"
    # B1 at $0.2: cheaper than the SLM arm at 20% ($0.5), dearer at 100% ($0.1); the old rule read 'AV2 wins' at 20%
    assert av2(0.2) == "utilization-dependent (tipping point u* ≈ 50%)"
    assert av2(0.5) == "utilization-dependent (tipping point u* ≈ 20%)"  # the same cost at 20% is not cheaper
    assert row(report.claims_map(data_with(j4_results(), {"B0": {"": 1.0}, "B4": b4})), "AV2")["verdict"] == "no data"


# ---------------------------------------------------------------- A5, A4, Appendix B

def validity(valid, n):
    return {"n": n, "valid": valid, "attempts": n, "rate": valid / n}


def test_a5_refutes_only_beyond_the_tolerance_on_a_call_site_with_enough_calls():
    """T11: B4 more than thresholds.format_tolerance_pp (1) below B0, on a call site with at least
    allocation.min_calls (100) invocations in both arms."""
    a5 = lambda b0, b4: row(report.claims_map(data_with({}, {}, {"B0": b0, "B4": b4})), "A5")  # noqa: E731
    assert a5({"a": validity(900, 1000)}, {"a": validity(1000, 1000)})["verdict"] == "confirms"
    assert a5({"a": validity(1000, 1000)}, {"a": validity(999, 1000)})["verdict"] == "confirms"  # 0.1 pp below: the old rule refuted
    assert a5({"a": validity(1000, 1000)}, {"a": validity(990, 1000)})["verdict"] == "confirms"  # exactly 1 pp: not more than it
    beyond = a5({"a": validity(1000, 1000), "b": validity(500, 500)}, {"a": validity(989, 1000), "b": validity(500, 500)})
    assert beyond["verdict"] == "refutes" and beyond["result"] == "B4 more than 1 pp below B0 on a (98.9% vs 100.0%)"
    # a call site with fewer than min_calls invocations in either arm is listed and does not count
    thin = a5({"a": validity(1000, 1000), "b": validity(100, 100)}, {"a": validity(1000, 1000), "b": validity(50, 99)})
    assert thin["verdict"] == "confirms" and thin["result"] == (
        "B4 within 1 pp of B0, or above, on all 1 call sites counted; not counted (fewer than 100 invocations in an arm): "
        "b (50.5% vs 100.0%)")
    enough = a5({"b": validity(100, 100)}, {"b": validity(50, 100)})
    assert enough["verdict"] == "refutes"  # the floor itself counts
    assert a5({"b": validity(100, 100)}, {"b": validity(50, 99)})["verdict"] == "no data (no call site with 100 invocations in both arms)"
    assert a5({"a": validity(10, 10)}, {"z": validity(10, 10)})["verdict"] == "no data"


def test_a4_states_b5s_own_outcome_and_a5_honours_several_runs():
    fraction = {"calls": 0.5, "tokens": 0.4, "cost_at_production_price": 0.3}
    for ok5, diff, said in ((True, -0.01, "non-inferior"), (False, -0.2, "worse"), (False, -0.02, "inconclusive")):
        data = data_with(j4_results(ok5=ok5, diff=diff), {"B0": {"": 1.0}})
        data["arms"]["B5"] = {"replaceable_fraction": fraction}
        assert row(report.claims_map(data), "A4")["result"].endswith(f"B5 against B0: {said}")
    formats = {"B0": {"a": validity(900, 1000)}, "B4": {"a": validity(1000, 1000)}}
    data = data_with({}, {}, formats)
    data["arms"]["B4"] = {"several_runs": ["b4-a", "b4-b"]}
    assert row(report.claims_map(data), "A5")["verdict"] == \
        "no verdict (several test runs of one configuration of B4: b4-a, b4-b)"


def per_call(rates):
    return {"per_call_site": {site: {"agreement": {"rate": rate}} for site, rate in rates.items()}}


def test_appendix_b_is_decided_on_repair_alone():
    """T4: confirms when the SLM is worse on repair, refutes when it is non-inferior. The routine's
    agreement is a proxy (D15) shown beside it: low, or not measured, it changes no verdict."""
    def appendix_b(repair, calls=per_call({"filter_column": 0.97, "select_tables": 0.99})):
        return row(report.claims_map(data_with({}, {}, repair=repair, per_call=calls)), "Appendix B")
    assert appendix_b(j4(False, diff=-0.2))["verdict"] == "confirms (the SLM is worse on repair)"
    assert appendix_b(j4(True))["verdict"] == "refutes (the SLM is non-inferior on repair)"
    assert appendix_b(j4(False, diff=-0.02))["verdict"] == "inconclusive (planned power 0.80)"
    low = appendix_b(j4(False, diff=-0.2), per_call({"filter_column": 0.50}))  # the old rule refuted on the proxy here
    assert low["verdict"] == "confirms (the SLM is worse on repair)"
    assert low["result"].endswith("a proxy with no effect on the verdict (D15): filter_column 50.0%")
    assert appendix_b(j4(True), per_call({"filter_column": 0.50}))["verdict"] == "refutes (the SLM is non-inferior on repair)"
    unmeasured = appendix_b(j4(False, diff=-0.2), per_call({}))  # the old rule read 'no data'
    assert unmeasured["verdict"] == "confirms (the SLM is worse on repair)" and unmeasured["result"].endswith("none measured")
    assert appendix_b(None)["verdict"] == "no data"
    several = appendix_b({**j4(True), "several_runs": ["r-a", "r-b"]})
    assert several["verdict"] == "no verdict (several test runs of one configuration: r-a, r-b)"


def test_the_steps_table_reads_the_judgments():
    curation = {"total": {"invocations": 10, "passed_filter": 8, "masked_sql": 1, "exact_duplicates": 2,
                          "near_duplicates": 3, "kept": 2}, "mask_detections": {"email": 4}}
    data = data_with({"B3|B4": j4(True, diff=0.1)}, {})
    data["judgments"].update(teacher_train_cost={"calls": 10, "total": {"standard": 1.5}}, j6=None, j7=None, j5={
        "curation": curation, "k": 3, "ari_call_sites": 0.5, "assignment": {"train_in_sample": 1.0, "calib": {"rate": 0.75}}})
    rows = {r["step"].split(" ")[0]: r for r in report.steps(data)}
    assert rows["S1"]["did"].startswith("10 teacher calls") and rows["S1"]["cost"] == "$1.5"
    assert ("8 passed the production signal; 1 SQL completions dropped because masking changed them; "
            "2 exact and 3 near duplicates removed; 4 sensitive-data detections masked") in rows["S2"]["did"]
    assert rows["S2"]["changed"] == "2 training examples" and "75.0% on calib" in rows["S3"]["changed"]
    curation["total"]["over_cap"] = 5  # the cap on the examples per question and call site (T1)
    assert report.steps(data)[1]["changed"] == "2 training examples (5 above the cap per question and call site left out)"
    assert rows["S5"]["changed"] == "B4 − B3: +10.0 pp (CI low +0.0 pp), non-inferior"
    data["judgments"]["j7"] = {"adapters": "a" * 64, "allocation": {"c0": "slm", "c1": "production_llm"},
                               "clusters": {"c0": {"cost_dependent": True}, "c1": {"cost_dependent": False}}}
    s6 = {r["step"].split(" ")[0]: r for r in report.steps(data)}["S6"]["did"]
    assert s6 == ("allocation: c0 → slm, c1 → production_llm; 1 chosen on the SLM cost extrapolated from "
                  "per-adapter load tests alone (c0)")


RATIOS = {"20%": ratio(10, 8, 12), "50%": ratio(25, 20, 30), "100%": ratio(50, 40, 60)}


def test_cost_verdicts_carry_their_labels_and_an_upper_bound_is_inconclusive():
    costs = {"B0": {"": 1.0}, "B4": {"20%": 0.1, "100%": 0.02}, "B5": {"20%": 0.05, "100%": 0.01}}
    data = data_with(j4_results(), costs, ratios=RATIOS)
    data["arms"]["B5"]["slm_cost_basis"] = "extrapolated from per-adapter load tests"
    assert row(report.claims_map(data), "V3")["verdict"] == \
        "meets the paper's 3× bar (even at 20% utilization) [costs: extrapolated from per-adapter load tests (B5)]"
    assert "delta_pp" not in data  # every test carries its own margin
    data["arms"]["B0"].update(upper_bound=True, cache_not_reported=7)
    v3 = row(report.claims_map(data), "V3")["verdict"]
    assert v3.startswith("inconclusive (rests on an upper-bound cost) [costs: ") and \
        "upper bound (cache not reported for 7 calls of B0)" in v3
    assert row(report.claims_map(data), "A6")["verdict"].startswith("inconclusive (rests on an upper-bound cost)")
    data["arms"]["B5"].update(lower_bound=True, failed_unbilled=3)
    assert "lower bound (3 failed calls of B5 unpriced)" in row(report.claims_map(data), "A6")["verdict"]


def test_several_test_runs_of_one_configuration_get_no_verdict():
    tests = {"B0|B4": {**j4(True), "several_runs": ["agent-B4-a", "agent-B4-b"]}, "B0|B5": j4(True)}
    data = data_with(tests, {"B0": {"": 1.0}, "B4": {"20%": 0.1, "100%": 0.02}})
    data["arms"]["B4"]["several_runs"] = ["agent-B4-a", "agent-B4-b"]
    rows = report.claims_map(data)
    assert row(rows, "V1")["verdict"] == "no verdict (several test runs of one configuration: agent-B4-a, agent-B4-b)"
    assert row(rows, "V3")["verdict"] == "no verdict (several test runs of one configuration of B4: agent-B4-a, agent-B4-b)"


def test_every_arm_of_a_set_counts_not_only_its_cheapest():
    """V3 takes the cheapest untrained arm: B0 here. B1 is dearer (it loses) but its cost is an upper
    bound, and B2-cheap has several registered runs; neither may be left out of the verdict."""
    tests = {**j4_results(), "B0|B1": j4(True), "B0|B2-cheap": j4(True)}
    costs = {"B0": {"": 1.0}, "B1": {"": 3.0}, "B4": {"20%": 0.1, "100%": 0.02}, "B5": {"20%": 0.05, "100%": 0.01}}
    data = data_with(tests, costs, ratios=RATIOS)
    data["arms"]["B1"].update(upper_bound=True, cache_not_reported=4)
    v3 = row(report.claims_map(data), "V3")["verdict"]
    assert v3.startswith("inconclusive (rests on an upper-bound cost)") and "calls of B1" in v3
    data["arms"]["B2-cheap"] = {"cost_per_correct": {"": 5.0}, "several_runs": ["b2-a", "b2-b"]}
    for claim in ("V3", "A6"):
        assert row(report.claims_map(data), claim)["verdict"] == \
            "no verdict (several test runs of one configuration of B2-cheap: b2-a, b2-b)"


def registry_of(*runs):
    return {"available": True, "runs": [{"run_id": r, "type": t, "arm": a, "engine": None, "status": st,
                                          "commit": "c", "prereg_hash": "h", "started_at": "2026-10-01T09:00:00+00:00",
                                          "facts": f} for r, t, a, st, f in runs]}


TRAINED_FACTS = {"choice": "C", "centroids": "K", "adapters": "A"}


def test_a_test_report_is_bound_to_the_registry(tmp_path, monkeypatch):
    from fixtures.fake import repo, write_run
    repo(tmp_path, monkeypatch)
    arms = {"B0": {"run_id": "b0"}, "B4": {"run_id": "b4"}, "B5": {"run_id": "b5"}}
    judged = {"j6": {"choice_fact": {"sha256": "C"}, "choice": "qwen"}, "j5": {"centroids": {"sha256": "K"}},
              "j7": {"adapters": "A", "allocation_fact": {"sha256": "L"}}}
    expected = report._expected_facts({}, judged)
    rows = [("b0", "agent", "B0", "done", {}), ("b4", "agent", "B4", "done", TRAINED_FACTS),
            ("b5", "agent", "B5", "done", {**TRAINED_FACTS, "allocation": "L"}), ("r4", "replay", "B4", "done", TRAINED_FACTS)]
    per_call = {"per_call": {"teacher": {"run_id": "b0"}, "replay": {"run_id": "r4"}}}

    def bind(runs=rows, reads=per_call, pilot={}, split="test", expected=expected, arms=arms):
        marked = {a: dict(v) for a, v in arms.items()}
        several = report._registry_bindings(split, marked, registry_of(*runs), judged, reads, pilot, expected, "h")
        return marked, several
    bind()
    older = registry_of(*rows)
    older["runs"][1]["prereg_hash"] = "an-earlier-registration"
    with pytest.raises(JudgmentError, match="B4's run b4 ran under pre-registration an-earlier-registration"):
        report._registry_bindings("test", {a: dict(v) for a, v in arms.items()}, older, judged, per_call, {}, expected, "h")
    with pytest.raises(JudgmentError, match="not a done entry"):
        bind([("b0", "agent", "B0", "failed", {})] + rows[1:])
    with pytest.raises(JudgmentError, match="not registered"):
        bind(rows[1:])
    for name, wrong in (("choice", "X"), ("centroids", "X"), ("adapters", "X")):
        with pytest.raises(JudgmentError, match=f"recorded the {name} fact"):
            bind([rows[0], ("b4", "agent", "B4", "done", {**TRAINED_FACTS, name: wrong})] + rows[2:])
    with pytest.raises(JudgmentError, match="recorded the allocation fact"):
        bind(rows[:2] + [("b5", "agent", "B5", "done", {**TRAINED_FACTS, "allocation": "X"})] + rows[3:])
    for missing in ("choice", "adapters", "allocation"):  # a plan without them is refused, never unchecked
        with pytest.raises(JudgmentError, match="cannot be checked"):
            bind(expected={k: v for k, v in expected.items() if k != missing})
    with pytest.raises(JudgmentError, match="per-call evaluation's replay"):
        bind(rows[:3] + [("r4", "replay", "B4", "done", {**TRAINED_FACTS, "centroids": "X"})])
    with pytest.raises(JudgmentError, match="per-call evaluation's run r4 is not a done entry"):
        bind(rows[:3] + [("r4", "replay", "B4", "failed", TRAINED_FACTS)])
    with pytest.raises(JudgmentError, match="another teacher run"):
        bind(reads={"per_call": {"teacher": {"run_id": "other-b0"}, "replay": {"run_id": "r4"}}})
    with pytest.raises(JudgmentError, match="test registry"):
        report._registry_bindings("test", {}, {"available": False, "reason": "not a git checkout", "runs": []}, judged, {}, {},
                                  expected, "h")
    report._registry_bindings("calib", {}, {"available": False, "reason": "x", "runs": []}, judged, {}, {}, expected, None)
    undated = registry_of(*rows)
    undated["runs"][0]["started_at"] = None
    with pytest.raises(JudgmentError, match="without started_at"):
        report._registry_bindings("test", {a: dict(v) for a, v in arms.items()}, undated, judged, per_call, {}, expected, "h")
    marked, several = bind(rows + [("b4-again", "agent", "B4", "done", {}), ("r4-again", "replay", "B4", "done", {})])
    assert marked["B4"]["several_runs"] == ["b4", "b4-again"] and marked["B0"]["several_runs"] is None
    assert several == ["r4", "r4-again"]
    # a run that did not finish, or that ran under an earlier registration, never counts as a second run (T13)
    unfinished = [("b4-again", "agent", "B4", "failed", {}), ("b4-third", "agent", "B4", "interrupted (intent, no manifest)", {}),
                  ("r4-again", "replay", "B4", "failed", {}), ("b0-again", "agent", "B0", "running", {})]
    marked, several = bind(rows + unfinished)
    assert several is None and all(found["several_runs"] is None for found in marked.values())
    superseded = registry_of(*rows, ("b4-before", "agent", "B4", "done", {}))
    superseded["runs"][-1]["prereg_hash"] = "an-earlier-registration"
    marked = {a: dict(v) for a, v in arms.items()}
    assert report._registry_bindings("test", marked, superseded, judged, per_call, {}, expected, "h") is None
    assert marked["B4"]["several_runs"] is None
    write_run("eval-pilot-late", {"type": "eval", "status": None, "finished_at": "2026-10-01T10:00:00+00:00"})
    write_run("eval-pilot-early", {"type": "eval", "status": None, "finished_at": "2026-09-30T10:00:00+00:00"})
    bind(pilot={"B0": "eval-pilot-early"})
    with pytest.raises(JudgmentError, match="did not finish before the first test"):
        bind(pilot={"B0": "eval-pilot-early", "B3": "eval-pilot-late"})
    staggered = registry_of(*rows)
    staggered["runs"][1]["started_at"] = "2026-10-01T12:00:00+00:00"
    staggered["runs"][0]["started_at"] = "2026-09-30T08:00:00+00:00"  # the first test run started before the pilot ended
    with pytest.raises(JudgmentError, match=r"did not finish before the first test execution started \(2026-09-30T08"):
        report._registry_bindings("test", {a: dict(v) for a, v in arms.items()}, staggered, judged, per_call,
                                  {"B0": "eval-pilot-early"}, expected, "h")
    no_b0 = {a: v for a, v in arms.items() if a != "B0"}
    with pytest.raises(JudgmentError, match="per-call evaluation's teacher's run b0 is not a done entry"):
        bind(rows[1:], arms=no_b0)  # the teacher is bound even when the plan has no B0 arm
    _, several_teacher = bind(rows + [("b0-again", "agent", "B0", "done", {})], arms=no_b0)
    assert several_teacher == ["b0", "b0-again"]
    late_zeroshot = {**per_call, "j6": {"qwen": {"replay_eval": {"run_id": "eval-pilot-late"},
                                                  "teacher_eval": {"run_id": "eval-pilot-early"}}}}
    with pytest.raises(JudgmentError, match="J6's zero-shot replay_eval"):  # the K4 and repair pilots too
        bind(reads=late_zeroshot)


def test_the_s3_row_reports_what_the_router_embeds_the_prompts():
    data = data_with({}, {})
    data["judgments"]["j5"] = {"ari_call_sites": 0.9, "k": 4, "truncation": {"prompt": {"truncated_fraction": 0.25},
                                                                            "prompt_action": {"truncated_fraction": 0.75}}}
    assert row(report.claims_map(data), "S3")["result"] == "ARI 0.900 over 4 clusters; 25.0% of prompts cut (what the router embeds)"


def test_the_a4_row_states_b5s_own_outcome():
    tests = {"B0|B5": j4(False, diff=-0.02)}  # inconclusive
    data = data_with(tests, {})
    data["arms"]["B5"] = {"replaceable_fraction": {"calls": 0.5, "tokens": 0.4, "cost_at_production_price": 0.3}}
    assert row(report.claims_map(data), "A4")["result"].endswith("B5 against B0: inconclusive")
    data["tests"]["B0|B5"] = j4(False, diff=-0.2, ci_high=-0.1)
    assert row(report.claims_map(data), "A4")["result"].endswith("B5 against B0: worse")


def test_av2_counts_an_upper_bound_on_the_losing_trained_arm():
    costs = {"B0": {"": 1.0}, "B1": {"": 0.5}, "B4": {"20%": 0.1, "100%": 0.05}, "B5": {"20%": 0.3, "100%": 0.2}}
    data = data_with(j4_results(), costs)
    assert row(report.claims_map(data), "AV2")["verdict"] == "does not refute (the SLM arm is cheaper even at 20% utilization)"
    data["arms"]["B5"].update(upper_bound=True, cache_not_reported=2)  # B4 is the cheaper: B5 loses
    assert row(report.claims_map(data), "AV2")["verdict"].startswith("inconclusive (rests on an upper-bound cost)")


def test_rows_on_an_arm_with_several_runs_give_no_verdict():
    formats = {"B0": {"a": validity(900, 1000)}, "B4": {"a": validity(1000, 1000)}}
    data = data_with({"B0|B4": j4(True)}, {"B0": {"": 1.0}, "B4": {"20%": 0.1, "100%": 0.05}}, formats)
    assert row(report.claims_map(data), "A5")["verdict"] == "confirms"
    data["arms"]["B0"]["several_runs"] = ["b0-a", "b0-b"]
    for claim in ("A5", "AV1"):
        assert row(report.claims_map(data), claim)["verdict"] == \
            "no verdict (several test runs of one configuration of B0: b0-a, b0-b)"


def test_a_plan_adapters_fact_must_be_the_one_j7_allocated_on(tmp_path, monkeypatch):
    from fixtures.fake import repo
    repo(tmp_path, monkeypatch)
    fact = facts.write_fact("S5", "adapters", {"slm": "q", "choice": "c" * 64, "centroids": "k" * 64,
                                               "base_revision": "0" * 40, "chat_template_kwargs": {},
                                               "adapters": {"c0": {"served_name": "q-c0", "sha256": "a" * 64}}})
    plan = {"adapters": relative(fact)}
    assert report._expected_facts(plan, {})["adapters"] == fact.parent.name
    assert report._expected_facts(plan, {"j7": {"adapters": fact.parent.name, "allocation_fact": {"sha256": "L"}}})["adapters"] \
        == fact.parent.name
    with pytest.raises(JudgmentError, match="is not the one its J7 allocated on"):
        report._expected_facts(plan, {"j7": {"adapters": "other", "allocation_fact": {"sha256": "L"}}})


def test_the_chart_legend_reads_the_configured_utilizations():
    data = {"arms": {"B4": {"ex": 0.8, "cost_per_correct": {"30%": 0.2, "90%": 0.1}}}, "utilizations": ["30%", "90%"]}
    assert "one point per utilization, 30% (right) to 90% (left)" in report.chart_svg(data)


def test_b3s_choice_is_checked_like_every_trained_arms_facts(tmp_path, monkeypatch):
    from fixtures.fake import repo
    repo(tmp_path, monkeypatch)
    arms, judged = {"B3": {"run_id": "b3"}}, {"j6": {"choice_fact": {"sha256": "C"}, "choice": "qwen"}}
    bind = lambda choice, judged=judged: report._registry_bindings(  # noqa: E731
        "test", {a: dict(v) for a, v in arms.items()}, registry_of(("b3", "agent", "B3", "done", {"choice": choice})),
        judged, {}, {}, report._expected_facts({}, judged), "h")
    bind("C")
    with pytest.raises(JudgmentError, match="recorded the choice fact"):
        bind("X")
    with pytest.raises(JudgmentError, match="cannot be checked: the plan names no j6"):
        bind("C", {})


def test_registry_reads_f1s_committed_intents_and_manifests(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.org")
    git("config", "user.name", "t")
    registry = tmp_path / "registry" / "test"
    registry.mkdir(parents=True)
    for name, content in {"agent-B0-test-1.intent.json": {"arm": "B0", "type": "agent"},
                          "agent-B0-test-1.manifest.json": {"type": "agent", "arm": "B0", "status": "done",
                                                            "commit": "a" * 40, "prereg_hash": "b" * 64},
                          "agent-B4-test-2.intent.json": {"arm": "B4", "type": "agent"}}.items():
        (registry / name).write_text(json.dumps(content))
    git("add", "registry")
    git("commit", "-q", "-m", "registry")
    (registry / "agent-B0-test-1.manifest.json").write_text(json.dumps({"status": "edited after the commit"}))
    found = report.test_registry(tmp_path)
    assert found["available"] and [r["run_id"] for r in found["runs"]] == ["agent-B0-test-1", "agent-B4-test-2"]
    assert found["runs"][0]["status"] == "done" and found["runs"][0]["prereg_hash"] == "b" * 64  # as committed
    assert found["runs"][1]["status"].startswith("interrupted")
    assert not report.test_registry(tmp_path / "not-a-repo")["available"]
    # prereg/DEVIATIONS.md is read as committed too (T13); None when the repository has none
    assert report.deviations(tmp_path) is None
    (tmp_path / "prereg").mkdir()
    (tmp_path / "prereg" / "DEVIATIONS.md").write_text("- 2026-10-02 · the pilot's draw was not stratified\n")
    git("add", "prereg")
    git("commit", "-q", "-m", "a deviation")
    (tmp_path / "prereg" / "DEVIATIONS.md").write_text("edited after the commit\n")
    assert report.deviations(tmp_path) == "- 2026-10-02 · the pilot's draw was not stratified\n"


def test_how_the_registry_table_reads_a_run():
    """T13: a run under an earlier registration is superseded; one that did not finish is not completed."""
    run = lambda status, prereg="h": {"run_id": "r", "status": status, "prereg_hash": prereg}  # noqa: E731
    assert report.run_reading(run("done"), "h") == "done"
    assert report.run_reading(run("failed"), "h") == "not completed (failed)"
    assert report.run_reading(run("interrupted (intent, no manifest)"), "h") == "not completed (interrupted (intent, no manifest))"
    assert report.run_reading(run("done", "an-earlier-registration"), "h") == "superseded by re-registration"
    assert report.run_reading(run("failed", "an-earlier-registration"), "h") == "superseded by re-registration"
    assert report.run_reading(run("done", None), None) == "done"  # a calib report has no registration in force


def test_the_report_opens_the_map_with_its_scope_and_prints_the_deviations():
    """T15 and T13, on the rendered page: the scope line sits above the map; a B5 that allocated nothing to
    an SLM is said where the replaceable fraction would be; DEVIATIONS.md is printed when there is one."""
    data = {"split": "test", "scope": report.SCOPE, "arms": {}, "map": [], "steps": [], "per_call_uncovered": [],
            "gold_tests": {}, "prereg_in_force": "h", "deviations": "- 2026-10-02 · re-registered: the pilot was not stratified\n",
            "judgments": {"j6": None, "per_call": None, "j7": {"allocation": {"c0": "production_llm"}}},
            "registry": registry_of(("b4", "agent", "B4", "done", {}), ("b4-again", "agent", "B4", "failed", {}))}
    data["registry"]["runs"][0]["prereg_hash"] = "an-earlier-registration"
    page = report.render(data)
    assert page.index("## The SPEC §5 map") < page.index(report.SCOPE) < page.index("| claim | result | verdict | planned power |")
    assert "## Replaceable fraction (B5)\n\nNot testable (B5 allocated nothing to an SLM).\n" in page
    assert "| b4 | agent | B4 | superseded by re-registration |" in page
    assert "| b4-again | agent | B4 | not completed (failed) |" in page
    assert ("## Deviations from the pre-registration (`prereg/DEVIATIONS.md`)\n\n"
            "- 2026-10-02 · re-registered: the pilot was not stratified\n") in page
    assert "## Deviations" not in report.render({**data, "deviations": None})


# ---------------------------------------------------------------- end to end

TEST_IDS = [str(q) for q in range(5000, 5400)]
CALIB_IDS = [str(q) for q in range(2000, 2060)]
# the registered draw; 30 of 60 calib ids, so the pilot leaves out some repair questions (with 50, this
# draw holds all 20 of them, and restricting to the pilot would change nothing the tests could see)
PILOT = __import__("bench.data", fromlist=["pilot_sample"]).pilot_sample(CALIB_IDS, 30, 20260930)
QUALITY = {"B0": 1.0, "B1": 0.7, "B2-production": 0.6, "B2-cheap": 0.5, "B3": 0.5, "B4": 0.99, "B5": 0.995}


CODE = Path(__file__).resolve().parent.parent  # this repository, whatever bench.paths points at


def publish(root):
    subprocess.run(["git", "-C", str(root), "push", "-q", "origin", "HEAD:main"], check=True, capture_output=True)


def register_analysis_code(root, config):
    """A pre-registration of `config` as `bench prereg` writes it: SPEC.md, the configuration, the splits,
    the data manifest and a copy of the analysis code this repository tracks, committed in `root` and
    pushed to a bare origin, which the barrier fetches. Returns the hash in force."""
    from bench.contracts.config import config_sha256
    from bench.prereg import ANALYSIS_CODE, analysis_code

    def git(*args, cwd=root):
        subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)

    def sha(rel):
        return hashlib.sha256((root / rel).read_bytes()).hexdigest()
    tracked = subprocess.run(["git", "-C", str(CODE), "ls-files", "--", *ANALYSIS_CODE], check=True,
                             capture_output=True, text=True).stdout.split()
    for rel in tracked:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((CODE / rel).read_bytes())
    (root / "SPEC.md").write_text("protocol\n")
    git("add", "bench", "SPEC.md")
    git("commit", "-q", "-m", "analysis code")
    (root / "prereg").mkdir()
    manifest = json.dumps({"spec_sha256": sha("SPEC.md"), "config_sha256": config_sha256(config),
                           "splits_sha256": sha("data/splits.json"), "data_manifest_sha256": sha("data/MANIFEST.json"),
                           "commit": "c", "analysis_code": analysis_code(root)}, sort_keys=True).encode() + b"\n"
    (root / "prereg" / "manifest.json").write_bytes(manifest)
    prereg_hash = hashlib.sha256(manifest).hexdigest()
    (root / "prereg" / "HASH").write_text(prereg_hash + "\n")
    git("add", "prereg")
    git("commit", "-q", "-m", "pre-registration")
    origin = root.parent / f"{root.name}-origin.git"
    git("init", "-q", "--bare", str(origin), cwd=root.parent)
    git("remote", "add", "origin", str(origin))
    git("push", "-q", "origin", "HEAD:main")
    return prereg_hash


def b1k_choice(k, pilot=PILOT):
    """A stored choice of B1's few-shot k on the pilot, as `bench judge b1k` writes it."""
    from bench.judge.base import canonical
    raw = canonical({"judgment": "b1k", "reads": {}, "result": {"k": k, "pilot_ids": sorted(pilot, key=int)}})
    path = paths.ROOT / "judgments" / "b1k" / hashlib.sha256(raw).hexdigest() / "choice.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return relative(path)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("datasketch")
    from bench.curate import run_curate, write_datasets
    from bench.embed import run_embed
    from bench.judge import j2, j3, j5, j6, j7, j8
    from fixtures.fake import call, fake_embed, fake_tokens, write_run
    from fixtures.world import gold_correct, per_call_eval, replay, teacher, trained_on
    from synthetic import make_repo
    from test_curate import teacher_config

    _, config_path, config = make_repo(tmp_path, monkeypatch)
    config = teacher_config(config)
    config["roles"]["production_llm"]["model"] = "teacher-model"
    config["prices"] = {"as_of": "2026-09-30", "table": {
        "teacher-model": {"input_per_mtok": 3.0, "cached_input_per_mtok": 0.3, "output_per_mtok": 12.0, "batch_discount": 0.5},
        "engine-model": {"input_per_mtok": 0.2, "cached_input_per_mtok": 0.02, "output_per_mtok": 0.6, "batch_discount": 0.5}}}
    config["cost"].update(p95_slo_cap_ms=1000, slo_from="agent-B0-slo")
    config["modal"] = {"gpu_prices": {"as_of": "2026-09-30", "usd_per_s": {"L4": 0.8 / 3600},  # F3's key
                                      "cpu_usd_per_core_s": 0.0, "memory_usd_per_gib_s": 0.0}}
    config["stats"] = {"n_boot": 100}  # F2's key
    config["allocation"]["min_calls"] = 5
    config["clustering"].update(k_min=2, k_max=6, n_init=3)
    config["selection"]["triage"] = [
        {"candidate": "qwen3-8b", "capabilities": "pass", "benchmarks": "pass", "license": "pass", "footprint": "pass", "result": "zero-shot"},
        {"candidate": "granite-4.2-8b", "capabilities": "pass", "benchmarks": "pass", "license": "pass", "footprint": "pass", "result": "zero-shot"}]
    config_path.write_text(yaml.safe_dump(config))
    # registered (and published) once the configuration is final, before anything touches the test
    for args in (("init", "-q"), ("config", "user.email", "t@example.org"), ("config", "user.name", "t")):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    prereg_hash = register_analysis_code(tmp_path, config)
    monkeypatch.setattr(clusters, "embed", fake_embed)

    def run(run_id, manifest, calls):
        write_run(run_id, {"config_sha256": "", **manifest}, calls)
        return run_id

    # S1 and S2: the teacher on train, curated (its SQL runs on the synthetic database)
    train = teacher("agent-B0-train", "train", ["1", "2", "3"])
    from bench.contracts.config import config_sha256
    write_run("agent-B0-train", {"type": "agent", "arm": "B0", "split": "train", "question_ids": ["1", "2", "3"],
                                 "config_sha256": config_sha256(config)}, train, config)
    curated = run_curate(["agent-B0-train"], str(config_path)).name
    # S3: embed and cluster, with the teacher on calib for the assignment rate
    calib = teacher("agent-B0-calib", "calib", CALIB_IDS)
    run("agent-B0-calib", {"type": "agent", "arm": "B0", "split": "calib", "question_ids": CALIB_IDS}, calib)
    teacher_ok = gold_correct("teacher", 0.9)
    per_call_eval("eval-B0-calib-per-call", "agent-B0-calib", calib, teacher_ok)
    train_embed = run_embed(curated, str(config_path), embed_fn=fake_embed, count_tokens=fake_tokens).name
    calib_embed = run_embed("agent-B0-calib", str(config_path), embed_fn=fake_embed, count_tokens=fake_tokens).name
    j5_path, centroids = j5.run(curated, train_embed, calib_embed, config)
    datasets = write_datasets(curated, str(j5_path), str(config_path))
    # S4: zero-shot of both candidates on calib
    zeroshots = {}
    for name, quality in (("qwen3-8b", 0.9), ("granite-4.2-8b", 0.6)):
        calls = replay(calib, f"zeroshot-{name}", f"slm:{name}", quality)
        run(f"zeroshot-{name}", {"type": "replay", "source_run_id": "agent-B0-calib", "engine": f"slm:{name}", "split": "calib"}, calls)
        zeroshots[f"zeroshot-{name}"] = per_call_eval(f"eval-zeroshot-{name}", f"zeroshot-{name}", calls,
                                                      lambda c, q=quality: teacher_ok(c) and gold_correct(name, q)(c))
    j6_path, choice = j6.run(zeroshots, "eval-B0-calib-per-call", config)
    # S5 (F3's): one adapter per cluster
    payload, centroids_sha = facts.read_fact(str(centroids), "centroids")
    adapters = facts.write_fact("S5", "adapters", {"slm": "qwen3-8b", "choice": choice.parent.name, "centroids": centroids_sha,
                                                   **trained_on(config),
                                                   "adapters": {c: {"served_name": f"qwen3-8b-{c}", "sha256": "e" * 64}
                                                                for c in payload["clusters"]}})
    assigned = lambda c: clusters.assign(c["prompt_messages"], payload)  # noqa: E731
    # load test and S6 on calib
    loadtests = []
    for engine, slug in (("slm:qwen3-8b", "base"), ("slm:qwen3-8b+lora:c", "lora-c")):  # B3's base, B4/B5's adapter
        for concurrency, p95, rps in ((1, 300, 3.0), (8, 800, 12.0), (32, 2500, 20.0)):
            run_id = f"loadtest-{slug}-{concurrency}"
            write_run(run_id, {"type": "loadtest", "engine": engine, "gpu": "L4", "concurrency": concurrency,
                               "prefix_cache": True, "sweep_id": f"sweep-{slug}"},
                      files={"profile_export_aiperf.json": {"request_latency": {"unit": "ms", "p95": p95},
                                                             "request_throughput": {"unit": "requests/sec", "avg": rps}}})
            loadtests.append(run_id)
    # the SLO rule's pilot: a production LLM slower than the cap, so the cap is the SLO
    run("agent-B0-slo", {"type": "agent", "arm": "B0", "split": "calib"},
        [call("agent-B0-slo", "1", "select_tables", latency_ms=5000)])
    j8_path = j8.run(loadtests, config, slo_from="agent-B0-slo")
    b4_calib = replay(calib, "replay-B4-calib", "slm:qwen3-8b+lora:c", 0.99, cluster_of=assigned)
    run("replay-B4-calib", {"type": "replay", "source_run_id": "agent-B0-calib", "arm": "B4", "split": "calib",
                            "facts": {"choice": choice.parent.name, "centroids": centroids_sha, "adapters": adapters.parent.name}}, b4_calib)
    per_call_eval("eval-B4-calib", "replay-B4-calib", b4_calib, teacher_ok)
    cheap_calib = replay(calib, "replay-cheap-calib", "cheap_alt", 0.8)
    run("replay-cheap-calib", {"type": "replay", "source_run_id": "agent-B0-calib", "engine": "cheap_alt", "split": "calib"}, cheap_calib)
    per_call_eval("eval-cheap-calib", "replay-cheap-calib", cheap_calib, lambda c: teacher_ok(c) and unit("cheap", c["question_id"]) < 0.8)
    # the teacher against itself on the pilot questions (T4): the agreement bar's cap
    routine_sites = ["agent_ir", "extract_keywords", "filter_column", "select_tables"]
    self_calls = replay([c for c in calib if c["question_id"] in PILOT and c["call_site"] in routine_sites],
                        "replay-self-calib", "production_llm", 0.98, model="teacher-model")
    run("replay-self-calib", {"type": "replay", "source_run_id": "agent-B0-calib", "engine": "production_llm",
                              "split": "calib", "call_sites": routine_sites}, self_calls)
    j7_path, allocation = j7.run(str(centroids), str(adapters), {"cheap_alt": ("replay-cheap-calib", "eval-cheap-calib"),
                                                                "slm": ("replay-B4-calib", "eval-B4-calib")},
                                 "eval-B0-calib-per-call", str(j8_path), str(j6_path), config, noninferiority,
                                 teacher_self_replay="replay-self-calib", pilot_ids=PILOT,
                                 difficulty={q: ("simple", "moderate", "challenging")[int(q) % 3] for q in CALIB_IDS})
    allocated = facts.read_fact(str(allocation), "allocation")[0]["allocation"]

    # the arms on test
    b0 = teacher("agent-B0-test", "test", TEST_IDS)
    arms_calls = {"B0": b0, "B1": replay(b0, "agent-B1-test", "cheap_alt", 0.8),
                  "B2-production": [c for c in teacher("agent-B2-production-test", "test", TEST_IDS) if c["call_site"] == "generate_candidate"],
                  "B2-cheap": [c for c in replay(b0, "agent-B2-cheap-test", "cheap_alt", 0.7) if c["call_site"] == "generate_candidate"],
                  "B3": replay(b0, "agent-B3-test", "slm:qwen3-8b", 0.6),
                  "B4": replay(b0, "agent-B4-test", "slm:qwen3-8b+lora:c", 0.99, cluster_of=assigned)}
    b5 = []
    for c in b0:
        engine = allocated.get(assigned(c), "production_llm")
        if engine == "production_llm":
            b5.append({**c, "run_id": "agent-B5-test", "cluster": assigned(c)})
        else:
            b5 += replay([c], "agent-B5-test", "slm:qwen3-8b+lora:c" if engine == "slm" else "cheap_alt", 1.0,
                         cluster_of=assigned)
    arms_calls["B5"] = b5
    plan = {"split": "test", "arms": {}, "format": {}, "pilot": {}, "b1k": b1k_choice(config["arms"]["B1"]["few_shot"]["k"])}
    b0_ok = {q: unit("B0", q) < 0.8 for q in TEST_IDS + CALIB_IDS}

    def evaluation(eval_run_id, source_run_id, arm, split, ids):
        """An end-to-end eval execution, as F2 writes it: no status, the arm and engine it scored."""
        engine = {"B2-production": "production_llm", "B2-cheap": "cheap_alt"}.get(arm)
        rows = [{"question_id": q, "difficulty": ("simple", "moderate", "challenging")[int(q) % 3],
                 "correct": b0_ok[q] if unit(arm, "keep", q) < QUALITY[arm] else not b0_ok[q],
                 "gold_date_substituted": False, "gold_has_limit": False, "gold_error": None} for q in ids]
        write_run(eval_run_id, {"type": "eval", "source_run_id": source_run_id, "arm": arm.split("-")[0], "engine": engine,
                                "split": split, "n": len(rows), "status": None, "fixed_date": "2026-09-30",
                                "timeout_s": 60, "sqlite_version": "3.45.0",
                                "prereg_hash": prereg_hash if split == "test" else None,
                                "finished_at": "2026-09-30T12:00:00+00:00" if split == "calib" else "2026-10-02T12:00:00+00:00"},
                  files={"results.jsonl": rows})
    for arm, calls in arms_calls.items():
        run_id = f"agent-{arm}-test"
        calls = [{**c, "run_id": run_id} for c in calls]
        run(run_id, {"type": "agent", "arm": arm.split("-")[0], "split": "test", "question_ids": TEST_IDS}, calls)
        evaluation(f"eval-{arm}", run_id, arm, "test", TEST_IDS)
        if arm in ("B0", "B3"):  # the pilot: the zero-shot SLM against the production LLM, on calib
            evaluation(f"eval-{arm}-pilot", f"agent-{arm}-pilot", arm, "calib", CALIB_IDS)
            plan["pilot"][arm] = f"eval-{arm}-pilot"
        has_slm = any(c["model_role"] == "slm" for c in calls)
        cost = j3.run(run_id, f"eval-{arm}", str(j8_path) if has_slm else None, config)
        plan["arms"][arm] = {"eval": f"eval-{arm}", "cost": relative(cost)}
        if arm in ("B0", "B4"):
            reads, result = j2.judge_run(run_id)
            plan["format"][arm] = relative(write_result("J2", reads, result))
    # the per-call-site evaluation on the test inputs: B0's invocations replayed as B4
    b4_replay = replay(b0, "replay-B4-test", "slm:qwen3-8b+lora:c", 0.97, cluster_of=assigned)
    run("replay-B4-test", {"type": "replay", "source_run_id": "agent-B0-test", "arm": "B4", "split": "test"}, b4_replay)
    per_call_eval("eval-B0-test-per-call", "agent-B0-test", b0, gold_correct("teacher-test", 0.9), prereg_hash=prereg_hash)
    per_call_eval("eval-B4-test-per-call", "replay-B4-test", b4_replay, gold_correct("teacher-test", 0.9),
                  prereg_hash=prereg_hash)
    reads, result = j2.judge_replay("replay-B4-test", "eval-B4-test-per-call", "eval-B0-test-per-call")
    plan.update(per_call=relative(write_result("J2", reads, result)), j5=relative(j5_path), j6=relative(j6_path),
                j7=relative(j7_path), j8=relative(j8_path),
                teacher_train_cost=relative(j3.run("agent-B0-train", None, None, config)))
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan))
    # the test registry, as F1 commits it, under a pre-registration of the analysis code this repository holds
    (tmp_path / "registry" / "test").mkdir(parents=True)
    recorded = {"B3": {"choice": choice.parent.name},
                "B4": {"choice": choice.parent.name, "centroids": centroids_sha, "adapters": adapters.parent.name},
                "B5": {"choice": choice.parent.name, "centroids": centroids_sha, "adapters": adapters.parent.name,
                       "allocation": allocation.parent.name}}
    for arm in arms_calls:
        engine = {"B2-production": "production_llm", "B2-cheap": "cheap_alt"}.get(arm)
        identity = {"type": "agent", "arm": arm.split("-")[0], "engine": engine, "commit": "c" * 40, "prereg_hash": prereg_hash}
        (tmp_path / "registry" / "test" / f"agent-{arm}-test.intent.json").write_text(json.dumps(
            {**identity, "run_id": f"agent-{arm}-test", "split": "test", "started_at": "2026-10-01T09:00:00+00:00"}))
        (tmp_path / "registry" / "test" / f"agent-{arm}-test.manifest.json").write_text(json.dumps(
            {**identity, "status": "done", "facts": recorded.get(arm, {})}))
    replay_identity = {"type": "replay", "arm": "B4", "engine": None, "commit": "c" * 40, "prereg_hash": prereg_hash}
    (tmp_path / "registry" / "test" / "replay-B4-test.intent.json").write_text(json.dumps(
        {**replay_identity, "run_id": "replay-B4-test", "split": "test", "started_at": "2026-10-01T09:30:00+00:00"}))
    (tmp_path / "registry" / "test" / "replay-B4-test.manifest.json").write_text(json.dumps(
        {**replay_identity, "status": "done", "facts": recorded["B4"]}))
    subprocess.run(["git", "-C", str(tmp_path), "add", "registry"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "registry"], check=True, capture_output=True)
    publish(tmp_path)
    return {"config": config, "plan": plan_path, "allocation": allocated, "datasets": datasets, "j5": j5_path,
            "j6": j6_path, "root": tmp_path, "prereg_hash": prereg_hash}


def test_a_test_report_is_read_only_under_the_registration_in_force(pipeline):
    """The configuration every verdict reads (thresholds, seeds, the pilot) is the registered one, and the
    registration itself is intact: a manifest edited to fit new code, keeping the old HASH, is refused."""
    import copy
    run = lambda config: report.run(str(pipeline["plan"]), config, ex_table, ex_summary,  # noqa: E731
                                    noninferiority, pilot_ids=PILOT)
    looser = copy.deepcopy(pipeline["config"])
    looser["thresholds"]["delta_pp"] = 10  # the margin of every verdict, widened after the test
    with pytest.raises(JudgmentError, match="configuration differs from the pre-registered one"):
        run(looser)
    root = pipeline["root"]
    manifest = json.loads((root / "prereg" / "manifest.json").read_text())
    manifest["analysis_code"]["bench/judge/j4.py"] = "0" * 64
    (root / "prereg" / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    for args in (("commit", "-q", "-am", "a manifest fitted to new code"), ("push", "-q", "origin", "HEAD:main")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    with pytest.raises(JudgmentError, match="HASH is not the sha256 of prereg/manifest.json"):
        run(pipeline["config"])


def test_a_test_report_reads_the_registry_as_published(pipeline):
    """A registry record that only this clone holds is no record: it counts once pushed."""
    root = pipeline["root"]
    registered = json.loads((root / "registry" / "test" / "agent-B4-test.manifest.json").read_text())
    identity = {k: registered[k] for k in ("type", "arm", "engine", "commit", "prereg_hash")}
    (root / "registry" / "test" / "agent-B4-again.intent.json").write_text(json.dumps(
        {**identity, "run_id": "agent-B4-again", "split": "test", "started_at": "2026-10-01T11:00:00+00:00"}))
    (root / "registry" / "test" / "agent-B4-again.manifest.json").write_text(json.dumps({**identity, "status": "done"}))
    for args in (("add", "registry"), ("commit", "-q", "-m", "a record not pushed yet")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    run = lambda: json.loads((report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary,  # noqa: E731
                                         noninferiority, pilot_ids=PILOT) / "report.json").read_text())
    assert run()["tests"]["B0|B4"]["several_runs"] is None
    publish(root)
    assert run()["tests"]["B0|B4"]["several_runs"] == ["agent-B4-again", "agent-B4-test"]


def test_a_test_report_reads_no_evaluation_scored_under_another_registration(pipeline):
    for eval_dir in [d for d in paths.RUNS.glob("eval-*") if (d / "manifest.json").exists()]:
        found = json.loads((eval_dir / "manifest.json").read_text())
        if found.get("prereg_hash"):  # every test evaluation, all under one other registration (J1 sees no mix)
            (eval_dir / "manifest.json").write_text(json.dumps({**found, "prereg_hash": "0" * 64}))
    with pytest.raises(JudgmentError, match="was scored under pre-registration 0000"):
        report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)


def test_a_test_report_reads_costs_priced_as_the_configuration_says(pipeline):
    from bench.judge import j3, j8
    plan = yaml.safe_load(pipeline["plan"].read_text())
    other = json.loads(json.dumps(pipeline["config"]))
    other["prices"]["as_of"] = "2026-10-15"  # every arm repriced alike: consistent among themselves, not registered
    changed = json.loads(json.dumps(plan))
    for arm, spec in changed["arms"].items():
        slm = arm in ("B3", "B4", "B5")
        spec["cost"] = relative(j3.run(f"agent-{arm}-test", spec["eval"], str(paths.ROOT / plan["j8"]) if slm else None, other))
    pipeline["plan"].write_text(yaml.safe_dump(changed))
    with pytest.raises(JudgmentError, match="another price table than the configuration's"):
        report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)
    gpu = json.loads(json.dumps(pipeline["config"]))
    gpu["modal"]["gpu_prices"]["usd_per_s"]["L4"] = 1.6 / 3600
    loadtests = sorted(read_result(plan["j8"], "J8")["reads"]["loadtests"])
    pipeline["plan"].write_text(yaml.safe_dump({**plan, "j8": relative(j8.run(loadtests, gpu, slo_from="agent-B0-slo"))}))
    with pytest.raises(JudgmentError, match="priced the GPUs with another table"):
        report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)


def test_a_test_report_is_read_only_with_the_registered_analysis_code(pipeline):
    """A verdict rule changed after the registration (here J4's) is refused before anything is read."""
    j4_copy = pipeline["root"] / "bench" / "judge" / "j4.py"
    j4_copy.write_text(j4_copy.read_text() + "\n# the margin read differently after the test was seen\n")
    with pytest.raises(JudgmentError, match="analysis code differs from the pre-registered one: bench/judge/j4.py"):
        report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)


def test_report_end_to_end_on_a_fake_execution(pipeline):
    out = report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)
    data = json.loads((out / "report.json").read_text())
    markdown = (out / "report.md").read_text()
    for heading in ("## EX and cost per correct query", "## The SPEC §5 map", "## S1–S6", "## S4 desk triage",
                    "## Replaceable fraction (B5)", "## Per-call-site evaluation", "## Test registry"):
        assert heading in markdown
    assert set(data["arms"]) == set(QUALITY)
    # numbers come from the judgments, unchanged
    b0_eval = read_jsonl(paths.RUNS / "eval-B0" / "results.jsonl")
    assert data["arms"]["B0"]["ex"] == sum(r["correct"] for r in b0_eval) / len(b0_eval)
    plan = yaml.safe_load(pipeline["plan"].read_text())
    b4_cost = read_result(plan["arms"]["B4"]["cost"], "J3")["result"]["per_correct"]
    assert data["arms"]["B4"]["cost_per_correct"] == {u: b4_cost[f"standard@{u}"] for u in ("20%", "50%", "100%")}
    assert data["judgments"]["j7"]["allocation"] == pipeline["allocation"]
    assert len(data["map"]) == 12 and all(any(v in r["verdict"] for v in VOCABULARY) for r in data["map"])
    # the scope: one line above the map, and "(this workload)" on every claim (T15)
    assert data["scope"] == report.SCOPE and report.SCOPE in markdown
    assert all("(this workload): " in r["claim"] for r in data["map"])
    # V3's ratio at every configured utilization, with its interval around the point (T9)
    assert data["utilizations"] == ["20%", "50%", "100%"] and set(data["cost_ratios"]) == set(data["utilizations"])
    for u, found in data["cost_ratios"].items():
        base, slm = data["arms"][found["base"]]["cost_per_correct"][""], data["arms"][found["slm"]]["cost_per_correct"][u]
        assert found["ratio"] == pytest.approx(base / slm) and found["ci_low"] <= found["ratio"] <= found["ci_high"]
    assert "(95% CI [" in next(r for r in data["map"] if r["claim"].startswith("V3"))["result"]
    assert data["deviations"] is None and "## Deviations" not in markdown  # no re-registration: nothing to print
    # every margin is the fixed thresholds.delta_pp, whatever the pair's discordance; the pilot gives the planned power
    assert {t["delta"] for t in data["tests"].values()} == {0.05} == {t["delta"] for t in data["gold_tests"].values()}
    assert len({t["d"] for t in data["tests"].values()}) > 1 and data["repair_test"]["d_pilot"] is not None
    assert "no verdict" not in markdown and "not testable (" not in markdown
    assert data["d_pilot"] == data["tests"]["B0|B4"]["d_pilot"] == data["tests"]["B0|B1"]["d_pilot"]  # one pilot
    from bench.judge.j4 import power
    assert data["tests"]["B0|B4"]["planned_power"] == power(data["d_pilot"], len(TEST_IDS), 0.05)
    pilot_rows = {arm: {r["question_id"]: r["correct"] for r in read_jsonl(paths.RUNS / f"eval-{arm}-pilot" / "results.jsonl")}
                  for arm in ("B0", "B3")}
    assert data["d_pilot"] == sum(pilot_rows["B0"][q] != pilot_rows["B3"][q] for q in PILOT) / len(PILOT)  # the pilot ids only
    assert [r["run_id"] for r in data["registry"]["runs"]] == sorted(f"agent-{arm}-test" for arm in QUALITY) + ["replay-B4-test"]
    assert data["arms"]["B5"]["replaceable_fraction"] is not None
    assert [data["arms"][a]["slm_cost_basis"] for a in ("B3", "B4", "B5")] == [
        "measured", "extrapolated from per-adapter load tests", "extrapolated from per-adapter load tests"]
    b4_row = next(line for line in markdown.splitlines() if line.startswith("| B4 |"))
    assert b4_row.endswith("| measured; extrapolated from per-adapter load tests (B4) |")  # the arm table's labels
    j5 = read_result(pipeline["j5"], "J5")["result"]
    assert f"ARI {j5['ari_call_sites']:.3f}" in markdown
    svg = ET.fromstring((out / "ex_cost.svg").read_text())
    labels = {t.text for t in svg.iter("{http://www.w3.org/2000/svg}text")}
    assert set(QUALITY) <= labels
    assert out.name == __import__("hashlib").sha256((out / "report.json").read_bytes()).hexdigest()
    # the training files follow the clusters the router will use
    manifest = json.loads((pipeline["datasets"] / "manifest.json").read_text())
    assert set(manifest["clusters"]) == set(j5["clusters"])
    # re-running gives the same report
    assert report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT) == out


def test_a_test_report_needs_the_registry(pipeline):
    import shutil
    shutil.rmtree(paths.ROOT / ".git")
    with pytest.raises(JudgmentError, match="registry"):
        report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)


def test_without_a_pilot_the_verdicts_stand_and_only_the_planned_power_is_missing(pipeline):
    plan = yaml.safe_load(pipeline["plan"].read_text())
    del plan["pilot"]
    pipeline["plan"].write_text(yaml.safe_dump(plan))
    data = json.loads((report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary,
                                  noninferiority, pilot_ids=PILOT) / "report.json").read_text())
    assert all(t["noninferior"] is not None and t["planned_power"] is None for t in data["tests"].values())
    assert data["d_pilot"] is None and next(r for r in data["map"] if r["claim"].startswith("V1"))["power"] == "—"


def test_report_refuses_an_eval_of_another_arm(pipeline):
    plan = yaml.safe_load(pipeline["plan"].read_text())
    plan["arms"]["B1"]["eval"], plan["arms"]["B3"]["eval"] = plan["arms"]["B3"]["eval"], plan["arms"]["B1"]["eval"]
    pipeline["plan"].write_text(yaml.safe_dump(plan))
    with pytest.raises(JudgmentError, match="is not B1 on test"):
        report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)


def test_generation_and_repair_get_their_own_tests_with_the_pilot_restricted(pipeline):
    from bench.judge.j4 import discordance
    out = report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority,
                     pilot_ids=PILOT)
    data = json.loads((out / "report.json").read_text())
    assert set(data["gold_tests"]) == {"generate_candidate", "revise"} and data["repair_test"] == data["gold_tests"]["revise"]
    assert "## Clusters with gold" in (out / "report.md").read_text()
    j6 = read_result(pipeline["j6"], "J6")["result"]
    gold = j6["per_call_site"][j6["choice"]]["revise"]["gold"]["by_question"]
    d_on = lambda ids: discordance({q: gold["replay"][q] for q in ids}, {q: gold["teacher"][q] for q in ids})  # noqa: E731
    pilot = sorted(set(PILOT) & set(gold["teacher"]))
    assert data["repair_test"]["d_pilot"] == d_on(pilot) != d_on(sorted(gold["teacher"]))  # the pilot questions only
    a4 = next(r for r in data["map"] if r["claim"].startswith("A4"))
    assert a4["result"].endswith(f"B5 against B0: {report.outcome(data['tests']['B0|B5'])}")
    a6 = next(r for r in data["map"] if r["claim"].startswith("A6"))
    assert "[costs: extrapolated from per-adapter load tests (B5)]" in a6["verdict"]  # B5's SLM cost


def test_the_report_refuses_prices_of_different_dates_and_mismatched_results(pipeline):
    from bench.judge import j2, j3
    plan = yaml.safe_load(pipeline["plan"].read_text())
    dated = json.loads(json.dumps(pipeline["config"]))
    dated["prices"]["as_of"] = "2026-10-15"
    for key, value, message in (
            (("arms", "B1", "cost"), relative(j3.run("agent-B1-test", "eval-B1", None, dated)), "different price tables"),
            (("format", "B4"), plan["format"]["B0"], "not of B4's execution"),
            (("per_call",), relative(write_result("J2", *j2.judge_run("agent-B4-test"))), "replay routed as B4")):
        changed = json.loads(json.dumps(plan))
        node = changed
        for part in key[:-1]:
            node = node[part]
        node[key[-1]] = value
        pipeline["plan"].write_text(yaml.safe_dump(changed))
        with pytest.raises(JudgmentError, match=message):
            report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority,
                       pilot_ids=PILOT)


def test_a_price_table_of_the_same_date_but_other_prices_is_refused(pipeline):
    from bench.judge import j3
    plan = yaml.safe_load(pipeline["plan"].read_text())
    repriced = json.loads(json.dumps(pipeline["config"]))
    repriced["prices"]["table"]["engine-model"]["output_per_mtok"] = 9.9  # same as_of, another table
    plan["arms"]["B1"]["cost"] = relative(j3.run("agent-B1-test", "eval-B1", None, repriced))
    pipeline["plan"].write_text(yaml.safe_dump(plan))
    with pytest.raises(JudgmentError, match="different price tables"):
        report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)


def test_a_replay_of_the_gold_call_sites_decides_appendix_b_and_k4(pipeline):
    """The per-call replay declares the call sites it replayed. Appendix B and K4 rest on the calls with
    gold alone (T4): a replay of only those decides both, with no routine agreement beside them."""
    from bench.judge import j2
    from fixtures.fake import write_run
    root = pipeline["root"]
    b0 = j2.calls_of("agent-B0-test")
    gold_only = [c for c in b0 if c["call_site"] in ("generate_candidate", "revise")]
    calls = [{**c, "run_id": "replay-gold"} for c in json.loads(json.dumps(read_jsonl(paths.RUNS / "replay-B4-test" / "calls.jsonl")))
             if c["call_site"] in ("generate_candidate", "revise")]
    manifest = json.loads((paths.RUNS / "replay-B4-test" / "manifest.json").read_text())
    write_run("replay-gold", {**manifest, "run_id": "replay-gold", "call_sites": ["generate_candidate", "revise"]}, calls)
    evaluated = [r for r in read_jsonl(paths.RUNS / "eval-B4-test-per-call" / "results.jsonl")]
    write_run("eval-gold", {"type": "eval", "source_run_id": "replay-gold", "per_call": True, "status": None,
                            "prereg_hash": pipeline["prereg_hash"],
                            "finished_at": "2026-10-02T12:00:00+00:00"}, files={"results.jsonl": evaluated})
    registered = json.loads((root / "registry" / "test" / "replay-B4-test.manifest.json").read_text())
    identity = {k: registered[k] for k in ("type", "arm", "engine", "commit", "prereg_hash")}
    for name in ("replay-B4-test.intent.json", "replay-B4-test.manifest.json"):  # the gold-only replay is the only one
        subprocess.run(["git", "-C", str(root), "rm", "-q", f"registry/test/{name}"], check=True, capture_output=True)
    (root / "registry" / "test" / "replay-gold.intent.json").write_text(json.dumps(
        {**identity, "run_id": "replay-gold", "split": "test", "started_at": "2026-10-01T09:40:00+00:00"}))
    (root / "registry" / "test" / "replay-gold.manifest.json").write_text(json.dumps(
        {**identity, "status": "done", "facts": registered["facts"]}))
    subprocess.run(["git", "-C", str(root), "add", "registry"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "a gold-only replay"], check=True, capture_output=True)
    publish(root)
    plan = yaml.safe_load(pipeline["plan"].read_text())
    plan["per_call"] = relative(write_result("J2", *j2.judge_replay("replay-gold", "eval-gold", "eval-B0-test-per-call")))
    pipeline["plan"].write_text(yaml.safe_dump(plan))
    out = report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)
    data = json.loads((out / "report.json").read_text())
    appendix_b = next(r for r in data["map"] if r["claim"].startswith("Appendix B"))
    assert data["per_call_uncovered"] == [] and set(data["gold_tests"]) == {"generate_candidate", "revise"}
    assert appendix_b["verdict"].startswith(("confirms (the SLM is worse on repair)", "refutes (the SLM is non-inferior on repair)",
                                             "inconclusive (planned power "))
    assert appendix_b["result"].endswith("a proxy with no effect on the verdict (D15): none measured")
    assert "No data: the per-call replay covers only" not in (out / "report.md").read_text()
    assert gold_only  # the teacher had gold calls to replay


def test_several_registered_replays_leave_k4_and_appendix_b_without_a_verdict(pipeline):
    root = pipeline["root"]
    registered = json.loads((root / "registry" / "test" / "replay-B4-test.manifest.json").read_text())
    identity = {k: registered[k] for k in ("type", "arm", "engine", "commit", "prereg_hash")}
    (root / "registry" / "test" / "replay-B4-again.intent.json").write_text(json.dumps(
        {**identity, "run_id": "replay-B4-again", "split": "test", "started_at": "2026-10-01T10:00:00+00:00"}))
    (root / "registry" / "test" / "replay-B4-again.manifest.json").write_text(json.dumps({**identity, "status": "done"}))
    for args in (("add", "registry"), ("commit", "-q", "-m", "the replay again")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    publish(root)
    data = json.loads((report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority,
                                  pilot_ids=PILOT) / "report.json").read_text())
    several = "no verdict (several test runs of one configuration: replay-B4-again, replay-B4-test)"
    assert data["gold_tests"] and all(t["several_runs"] == ["replay-B4-again", "replay-B4-test"]
                                      for t in data["gold_tests"].values())
    assert next(r for r in data["map"] if r["claim"].startswith("Appendix B"))["verdict"] == several


def test_several_registered_runs_of_an_arm_leave_its_comparisons_without_a_verdict(pipeline):
    root = pipeline["root"]
    identity = json.loads((root / "registry" / "test" / "agent-B4-test.manifest.json").read_text())
    (root / "registry" / "test" / "agent-B4-again.intent.json").write_text(json.dumps(
        {"type": "agent", "arm": "B4", "engine": None, "prereg_hash": pipeline["prereg_hash"], "run_id": "agent-B4-again",
         "split": "test", "started_at": "2026-10-01T11:00:00+00:00"}))
    (root / "registry" / "test" / "agent-B4-again.manifest.json").write_text(json.dumps({**identity, "status": "done"}))
    subprocess.run(["git", "-C", str(root), "add", "registry"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "B4 again"], check=True, capture_output=True)
    publish(root)
    data = json.loads((report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority,
                                  pilot_ids=PILOT) / "report.json").read_text())
    several = "agent-B4-again, agent-B4-test"
    assert data["tests"]["B0|B4"]["several_runs"] == ["agent-B4-again", "agent-B4-test"]
    v1 = next(r for r in data["map"] if r["claim"].startswith("V1"))
    assert v1["verdict"] == f"no verdict (several test runs of one configuration: {several})"
    for claim in ("V3", "AV2"):
        verdict = next(r for r in data["map"] if r["claim"].startswith(claim))["verdict"]
        assert verdict == f"no verdict (several test runs of one configuration of B4: {several})"


def test_unfinished_and_superseded_runs_are_listed_and_never_block_a_verdict(pipeline):
    """T13: 'several runs' counts completed runs under the registration in force. A failed or interrupted
    re-run, and a run under an earlier registration, are listed as such; DEVIATIONS.md is printed."""
    root = pipeline["root"]
    before = json.loads((report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority,
                                    pilot_ids=PILOT) / "report.json").read_text())
    b4 = json.loads((root / "registry" / "test" / "agent-B4-test.manifest.json").read_text())
    identity = {k: b4[k] for k in ("type", "arm", "engine", "commit", "prereg_hash")}
    register(root, "agent-B4-failed", identity, status="failed")
    (root / "registry" / "test" / "agent-B4-interrupted.intent.json").write_text(json.dumps(
        {**identity, "run_id": "agent-B4-interrupted", "split": "test", "started_at": "2026-10-01T11:30:00+00:00"}))
    register(root, "agent-B4-before", {**identity, "prereg_hash": "0" * 64}, started="2026-10-01T08:00:00+00:00")
    (root / "prereg" / "DEVIATIONS.md").write_text("- 2026-10-01 · re-registered: the first registration left the pilot out\n")
    for args in (("add", "prereg"), ("commit", "-q", "-m", "the deviation")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    publish(root)
    out = report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)
    data, markdown = json.loads((out / "report.json").read_text()), (out / "report.md").read_text()
    assert all(t["several_runs"] is None for t in data["tests"].values()) and "no verdict" not in markdown
    assert [r["verdict"] for r in data["map"]] == [r["verdict"] for r in before["map"]]  # the verdicts did not move
    for shown in ("| agent-B4-failed | agent | B4 | not completed (failed) |",
                  "| agent-B4-interrupted | agent | B4 | not completed (interrupted (intent, no manifest)) |",
                  "| agent-B4-before | agent | B4 | superseded by re-registration |", "| agent-B4-test | agent | B4 | done |",
                  "## Deviations from the pre-registration (`prereg/DEVIATIONS.md`)\n\n"
                  "- 2026-10-01 · re-registered: the first registration left the pilot out"):
        assert shown in markdown
    assert data["deviations"] == "- 2026-10-01 · re-registered: the first registration left the pilot out\n"


def test_the_report_refuses_a_cost_that_cannot_say_what_it_is_or_used_another_load_test(pipeline):
    from bench.judge import j8
    plan = yaml.safe_load(pipeline["plan"].read_text())
    b1 = read_result(plan["arms"]["B1"]["cost"], "J3")
    silent = {k: v for k, v in b1["result"].items() if k not in ("upper_bound", "cache_not_reported")}
    changed = json.loads(json.dumps(plan))
    changed["arms"]["B1"]["cost"] = relative(write_result("J3", b1["reads"], silent))
    pipeline["plan"].write_text(yaml.safe_dump(changed))
    with pytest.raises(JudgmentError, match="records no upper_bound, cache_not_reported"):
        report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)
    only_adapter = [f"loadtest-lora-c-{c}" for c in (1, 8, 32)]
    other = relative(j8.run(only_adapter, pipeline["config"], slo_from="agent-B0-slo"))
    b4 = read_result(plan["arms"]["B4"]["cost"], "J3")
    from bench.judge import j3
    changed = json.loads(json.dumps(plan))
    changed["arms"]["B4"]["cost"] = relative(j3.run(b4["reads"]["run"]["run_id"], "eval-B4", other, pipeline["config"]))
    pipeline["plan"].write_text(yaml.safe_dump(changed))
    with pytest.raises(JudgmentError, match="another load test than the plan's j8"):
        report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)


def test_the_report_refuses_an_slo_or_a_few_shot_k_the_configuration_does_not_name(pipeline):
    """What the pilot fixed is read from the registered configuration: the execution the SLO is
    measured on (T8), and B1's k, which is the one the rule chose on the pilot questions (T18)."""
    plan = yaml.safe_load(pipeline["plan"].read_text())
    k = pipeline["config"]["arms"]["B1"]["few_shot"]["k"]

    def refused(changed, match):
        pipeline["plan"].write_text(yaml.safe_dump({**plan, **changed}))
        with pytest.raises(JudgmentError, match=match):
            report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)
    j8 = read_result(plan["j8"], "J8")
    elsewhere = write_result("J8", {**j8["reads"], "slo_from": {**j8["reads"]["slo_from"], "run_id": "agent-B0-other"}},
                             j8["result"])
    refused({"j8": relative(elsewhere)}, r"another execution than the configuration's cost.slo_from \(agent-B0-slo\)")
    refused({"b1k": b1k_choice(3 - k)}, f"arms.B1.few_shot.k is {k} and the plan's b1k chose {3 - k}")
    refused({"b1k": b1k_choice(k, PILOT[1:])}, "on other questions than the pilot's")
    refused({"b1k": None}, "a test report with B1 needs the plan's b1k")
    assert plan["b1k"].endswith("choice.json")  # and the plan as written is the one every other test reports on


def test_the_s3_row_reports_the_prompts_cut(pipeline):
    data = json.loads((report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority,
                                  pilot_ids=PILOT) / "report.json").read_text())
    j5 = read_result(pipeline["j5"], "J5")["result"]
    s3 = next(r for r in data["map"] if r["claim"].startswith("S3"))
    cut = j5["truncation"]["prompt"]["truncated_fraction"]
    assert s3["result"].endswith(f"{100 * cut:.1f}% of prompts cut (what the router embeds)")


def register(root, run_id, identity, status="done", facts=None, started="2026-10-01T11:00:00+00:00"):
    (root / "registry" / "test" / f"{run_id}.intent.json").write_text(json.dumps(
        {**identity, "run_id": run_id, "split": "test", "started_at": started}))
    (root / "registry" / "test" / f"{run_id}.manifest.json").write_text(json.dumps(
        {**identity, "status": status, "facts": facts or {}}))
    subprocess.run(["git", "-C", str(root), "add", "registry"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", run_id], check=True, capture_output=True)
    publish(root)


def test_several_runs_of_the_replay_or_of_b0_reach_k4_and_appendix_b(pipeline):
    root = pipeline["root"]
    replay = json.loads((root / "registry" / "test" / "replay-B4-test.manifest.json").read_text())
    identity = {k: replay[k] for k in ("type", "arm", "engine", "commit", "prereg_hash")}
    register(root, "replay-B4-again", identity)
    data = json.loads((report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority,
                                  pilot_ids=PILOT) / "report.json").read_text())
    assert {t["several_runs"] and tuple(t["several_runs"]) for t in data["gold_tests"].values()} == \
        {("replay-B4-again", "replay-B4-test")}
    appendix_b = next(r for r in data["map"] if r["claim"].startswith("Appendix B"))
    assert appendix_b["verdict"] == "no verdict (several test runs of one configuration: replay-B4-again, replay-B4-test)"


def test_several_runs_of_b0_reach_every_row_on_its_evidence(pipeline):
    root = pipeline["root"]
    b0 = json.loads((root / "registry" / "test" / "agent-B0-test.manifest.json").read_text())
    register(root, "agent-B0-again", {k: b0[k] for k in ("type", "arm", "engine", "commit", "prereg_hash")})
    data = json.loads((report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority,
                                  pilot_ids=PILOT) / "report.json").read_text())
    runs = "agent-B0-again, agent-B0-test"
    verdicts = {r["claim"].split(" (this workload)")[0]: r["verdict"] for r in data["map"]}
    assert verdicts["Appendix B"] == f"no verdict (several test runs of one configuration: {runs})"  # B0 is K4's teacher
    assert verdicts["A5"] == f"no verdict (several test runs of one configuration of B0: {runs})"
    assert verdicts["A4 / A11"] == f"no verdict (several test runs of one configuration of B0: {runs})"
    assert all(t["several_runs"] == ["agent-B0-again", "agent-B0-test"] for t in data["gold_tests"].values())


def test_a_replay_of_the_routine_only_decides_neither_appendix_b_nor_k4(pipeline):
    from bench.judge import j2
    from fixtures.fake import write_run
    root = pipeline["root"]
    routine = list(report.ROUTINE) + ["agent_ir"]
    calls = [{**c, "run_id": "replay-routine"} for c in read_jsonl(paths.RUNS / "replay-B4-test" / "calls.jsonl")
             if c["call_site"] in routine]
    manifest = json.loads((paths.RUNS / "replay-B4-test" / "manifest.json").read_text())
    write_run("replay-routine", {**manifest, "run_id": "replay-routine", "call_sites": routine}, calls)
    write_run("eval-routine", {"type": "eval", "source_run_id": "replay-routine", "per_call": True, "status": None,
                               "prereg_hash": pipeline["prereg_hash"],
                               "finished_at": "2026-10-02T12:00:00+00:00"}, files={"results.jsonl": []})
    registered = json.loads((root / "registry" / "test" / "replay-B4-test.manifest.json").read_text())
    for name in ("replay-B4-test.intent.json", "replay-B4-test.manifest.json"):  # the routine-only replay is the only one
        subprocess.run(["git", "-C", str(root), "rm", "-q", f"registry/test/{name}"], check=True, capture_output=True)
    register(root, "replay-routine", {k: registered[k] for k in ("type", "arm", "engine", "commit", "prereg_hash")},
             facts=registered["facts"])
    plan = yaml.safe_load(pipeline["plan"].read_text())
    plan["per_call"] = relative(write_result("J2", *j2.judge_replay("replay-routine", "eval-routine", "eval-B0-test-per-call")))
    pipeline["plan"].write_text(yaml.safe_dump(plan))
    data = json.loads((report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority,
                                  pilot_ids=PILOT) / "report.json").read_text())
    appendix_b = next(r for r in data["map"] if r["claim"].startswith("Appendix B"))
    assert data["per_call_uncovered"] == ["generate_candidate", "revise"]
    assert appendix_b["verdict"] == f"no data (the replay covers only {', '.join(routine)})"


def test_the_report_refuses_a_j2_without_its_call_sites_and_a_j7_on_other_prices(pipeline):
    from bench.judge import j7
    plan = yaml.safe_load(pipeline["plan"].read_text())
    per_call = read_result(plan["per_call"], "J2")
    silent = {k: v for k, v in per_call["result"].items() if k != "call_sites"}
    changed = json.loads(json.dumps(plan))
    changed["per_call"] = relative(write_result("J2", per_call["reads"], silent))
    pipeline["plan"].write_text(yaml.safe_dump(changed))
    with pytest.raises(JudgmentError, match="does not record which call sites"):
        report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)
    j7_result = read_result(plan["j7"], "J7")
    for key, value, message in ((("prices",), {"as_of": "2026-09-30", "sha256": "0" * 64}, "another price table"),
                                (None, None, "another load test")):
        result, reads = json.loads(json.dumps(j7_result["result"])), json.loads(json.dumps(j7_result["reads"]))
        if key:
            result["prices"] = value
        else:
            reads["j8"] = {"judgment": "J8", "sha256": "1" * 64}
        changed = json.loads(json.dumps(plan))
        changed["j7"] = relative(write_result("J7", reads, result))
        pipeline["plan"].write_text(yaml.safe_dump(changed))
        with pytest.raises(JudgmentError, match=message):
            report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)
