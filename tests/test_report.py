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
              "no verdict")


def j4(ok=True, diff=-0.01, pilot=0.1, ci_low=None, ci_high=None):
    """A J4 result as the report holds it: J4's test at the fixed margin, with the pilot's d and the
    planned power beside it (None without a pilot)."""
    return {"n": 498, "d": 0.1, "delta": 0.05, "diff": diff, "ci_low": ci_low if ci_low is not None else (0.0 if ok else -0.09),
            "ci_high": ci_high if ci_high is not None else diff + 0.05, "noninferior": ok, "power": 0.97,
            "d_pilot": pilot, "planned_power": 0.8 if pilot is not None else None}


def j4_results(ok4=True, ok5=True, diff=-0.01, pilot=0.1):
    return {"B0|B4": j4(ok4, diff, pilot), "B0|B5": j4(ok5, diff, pilot)}


def data_with(tests, costs, formats=None, repair=None, per_call=None):
    arms = {arm: {"cost_per_correct": c, "replaceable_fraction": None} for arm, c in costs.items()}
    return {"arms": arms, "tests": tests, "formats": formats or {}, "repair_test": repair,
            "judgments": {"j5": None, "per_call": per_call}, "concordance_min": 0.95, "v3_min_ratio": 3}


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
    assert report.outcome(j4(False, ci_low=-0.0501, ci_high=0.01)) == "inconclusive"


def test_planned_power_is_j4s_at_the_pilots_d_and_the_tests_n():
    """The pilot's d gives the planned power and nothing else: by hand, at d = 0.20, n = 498, Δ = 0.05,
    Φ(0.05 / sqrt(0.20 / 498) − 1.6448536) = Φ(2.494995 − 1.6448536) = Φ(0.850141) = 0.8024."""
    test = {"n": 498, "delta": 0.05, "noninferior": False, "ci_high": 0.0}
    with_pilot = report.planned(test, 0.20)
    assert with_pilot["d_pilot"] == 0.20 and with_pilot["planned_power"] == pytest.approx(0.8024, abs=1e-4)
    assert with_pilot["delta"] == 0.05 and report.outcome(with_pilot) == "inconclusive"  # the margin did not move
    assert report.planned(test, None)["planned_power"] is None


def test_v1_confirms_refutes_and_says_why_not():
    costs = {"B0": {"": 1.0}}
    assert row(report.claims_map(data_with(j4_results(), costs)), "V1")["verdict"] == "confirms"
    assert row(report.claims_map(data_with(j4_results(False, False, diff=-0.2), costs)), "V1")["verdict"] == "refutes"
    undecided = row(report.claims_map(data_with(j4_results(False, False), costs)), "V1")
    assert undecided["verdict"] == "inconclusive (planned power 0.80, 0.80)" and undecided["power"] == "0.80, 0.80"
    assert row(report.claims_map(data_with({}, costs)), "V1")["verdict"] == "no data"
    # without a pilot the margin is still the fixed one: a verdict, with no planned power
    no_pilot = row(report.claims_map(data_with(j4_results(pilot=None), costs)), "V1")
    assert no_pilot["verdict"] == "confirms" and no_pilot["power"] == "—" and "Δ 5.0 pp" in no_pilot["result"]
    unpowered = row(report.claims_map(data_with(j4_results(False, False, pilot=None), costs)), "V1")
    assert unpowered["verdict"] == "inconclusive (no pilot: planned power unknown)"


def test_cost_claims_per_utilization():
    tests = {**j4_results(), "B0|B1": j4(False)}
    costs = {"B0": {"": 1.0}, "B1": {"": 0.2}, "B4": {"20%": 0.5, "100%": 0.1}, "B5": {"20%": 0.4, "100%": 0.05}}
    rows = report.claims_map(data_with(tests, costs))
    # B1 is not non-inferior to B0, so the best arm without training is B0 at $1.00
    assert row(rows, "V3")["verdict"] == "20%: refutes (2.5× vs B0) · 100%: confirms (20.0× vs B0)"
    assert row(rows, "A6")["verdict"] == "20%: confirms · 100%: confirms"
    assert row(rows, "AV2")["verdict"] == "20%: refutes the paper (AV2 wins) · 100%: does not refute"
    worse = {**costs, "B5": {"20%": 2.0, "100%": 1.5}}
    assert row(report.claims_map(data_with(tests, worse)), "A6")["verdict"] == "20%: refutes · 100%: refutes"
    failing = row(report.claims_map(data_with(j4_results(False, False), costs)), "V3")["verdict"]
    assert failing.startswith("20%: inconclusive (planned power 0.80, 0.80) (no trained arm passes V1)")


def test_format_claim_compares_each_call_site():
    formats = {"B0": {"a": {"rate": 0.9}, "b": {"rate": 1.0}}, "B4": {"a": {"rate": 1.0}, "b": {"rate": 1.0}}}
    assert row(report.claims_map(data_with({}, {}, formats)), "A5")["verdict"] == "confirms"
    formats["B4"]["b"]["rate"] = 0.99  # better overall, worse on one call site
    a5 = row(report.claims_map(data_with({}, {}, formats)), "A5")
    assert a5["verdict"] == "refutes" and a5["result"].startswith("B4 below B0 on b")


def test_a4_states_b5s_own_outcome_and_a5_honours_several_runs():
    fraction = {"calls": 0.5, "tokens": 0.4, "cost_at_production_price": 0.3}
    for ok5, diff, said in ((True, -0.01, "non-inferior"), (False, -0.2, "worse"), (False, -0.02, "inconclusive")):
        data = data_with(j4_results(ok5=ok5, diff=diff), {"B0": {"": 1.0}})
        data["arms"]["B5"] = {"replaceable_fraction": fraction}
        assert row(report.claims_map(data), "A4")["result"].endswith(f"B5 against B0: {said}")
    formats = {"B0": {"a": {"rate": 0.9}}, "B4": {"a": {"rate": 1.0}}}
    data = data_with({}, {}, formats)
    data["arms"]["B4"] = {"several_runs": ["b4-a", "b4-b"]}
    assert row(report.claims_map(data), "A5")["verdict"] == \
        "no verdict (several test runs of one configuration of B4: b4-a, b4-b)"


def per_call(rates):
    return {"per_call_site": {site: {"agreement": {"rate": rate}} for site, rate in rates.items()}}


def test_appendix_b_confirms_only_when_repair_is_worse_and_the_routine_passes():
    routine = per_call({"filter_column": 0.97, "select_tables": 0.99})
    verdict = lambda repair, calls=routine: row(report.claims_map(data_with({}, {}, repair=repair, per_call=calls)),  # noqa: E731
                                               "Appendix B")["verdict"]
    assert verdict(j4(False, diff=-0.2)) == "confirms (the routine by the agreement proxy, which supports no per-cluster claim (D15))"
    assert verdict(j4(True)) == "refutes (the SLM ties on repair)"
    assert verdict(j4(False, diff=-0.02)) == "inconclusive (planned power 0.80)"
    assert verdict(j4(False, diff=-0.2), per_call({"filter_column": 0.90})).startswith("refutes (the SLM loses on the routine;")


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
    assert rows["S5"]["changed"] == "B4 − B3: +10.0 pp (CI low +0.0 pp), non-inferior"
    data["judgments"]["j7"] = {"adapters": "a" * 64, "allocation": {"c0": "slm", "c1": "production_llm"},
                               "clusters": {"c0": {"cost_dependent": True}, "c1": {"cost_dependent": False}}}
    s6 = {r["step"].split(" ")[0]: r for r in report.steps(data)}["S6"]["did"]
    assert s6 == ("allocation: c0 → slm, c1 → production_llm; 1 chosen on the SLM cost extrapolated from "
                  "per-adapter load tests alone (c0)")


def test_appendix_b_with_no_routine_measured_is_no_data():
    row_ = row(report.claims_map(data_with({}, {}, repair=j4(False, diff=-0.2), per_call=per_call({}))), "Appendix B")
    assert row_["verdict"] == "no data (no routine call site measured)"


def test_cost_verdicts_carry_their_labels_and_an_upper_bound_is_inconclusive():
    tests = {**j4_results()}
    costs = {"B0": {"": 1.0}, "B4": {"20%": 0.1}, "B5": {"20%": 0.05}}
    data = data_with(tests, costs)
    data["arms"]["B5"]["slm_cost_basis"] = "extrapolated from per-adapter load tests"
    assert row(report.claims_map(data), "V3")["verdict"] == \
        "20%: confirms (20.0× vs B0) [costs: extrapolated from per-adapter load tests (B5)]"
    data["arms"]["B0"].update(upper_bound=True, cache_not_reported=7)
    v3 = row(report.claims_map(data), "V3")["verdict"]
    assert v3.startswith("20%: inconclusive (rests on an upper-bound cost) [costs: ") and \
        "upper bound (cache not reported for 7 calls of B0)" in v3
    assert row(report.claims_map(data), "A6")["verdict"].startswith("20%: inconclusive (rests on an upper-bound cost)")
    data["arms"]["B5"].update(lower_bound=True, failed_unbilled=3)
    assert "lower bound (3 failed calls of B5 unpriced)" in row(report.claims_map(data), "A6")["verdict"]


def test_several_test_runs_of_one_configuration_get_no_verdict():
    tests = {"B0|B4": {**j4(True), "several_runs": ["agent-B4-a", "agent-B4-b"]}, "B0|B5": j4(True)}
    data = data_with(tests, {"B0": {"": 1.0}, "B4": {"20%": 0.1}})
    data["arms"]["B4"]["several_runs"] = ["agent-B4-a", "agent-B4-b"]
    rows = report.claims_map(data)
    assert row(rows, "V1")["verdict"] == "no verdict (several test runs of one configuration: agent-B4-a, agent-B4-b)"
    assert row(rows, "V3")["verdict"] == \
        "20%: no verdict (several test runs of one configuration of B4: agent-B4-a, agent-B4-b)"


def test_every_arm_of_a_set_counts_not_only_its_cheapest():
    """V3 takes the cheapest untrained arm: B0 here. B1 is dearer (it loses) but its cost is an upper
    bound, and B2-cheap has several registered runs; neither may be left out of the verdict."""
    tests = {**j4_results(), "B0|B1": j4(True), "B0|B2-cheap": j4(True)}
    costs = {"B0": {"": 1.0}, "B1": {"": 3.0}, "B4": {"20%": 0.1}, "B5": {"20%": 0.05}}
    data = data_with(tests, costs)
    data["arms"]["B1"].update(upper_bound=True, cache_not_reported=4)
    v3 = row(report.claims_map(data), "V3")["verdict"]
    assert v3.startswith("20%: inconclusive (rests on an upper-bound cost)") and "calls of B1" in v3
    data["arms"]["B2-cheap"] = {"cost_per_correct": {"": 5.0}, "several_runs": ["b2-a", "b2-b"]}
    for claim in ("V3", "A6"):
        assert row(report.claims_map(data), claim)["verdict"] == \
            "20%: no verdict (several test runs of one configuration of B2-cheap: b2-a, b2-b)"


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
    costs = {"B0": {"": 1.0}, "B1": {"": 0.5}, "B4": {"20%": 0.1}, "B5": {"20%": 0.3}}
    data = data_with(j4_results(), costs)
    assert row(report.claims_map(data), "AV2")["verdict"] == "20%: does not refute"
    data["arms"]["B5"].update(upper_bound=True, cache_not_reported=2)  # B4 is the cheaper: B5 loses
    assert row(report.claims_map(data), "AV2")["verdict"].startswith("20%: inconclusive (rests on an upper-bound cost)")


def test_rows_on_an_arm_with_several_runs_give_no_verdict():
    formats = {"B0": {"a": {"rate": 0.9}}, "B4": {"a": {"rate": 1.0}}}
    data = data_with({"B0|B4": j4(True)}, {"B0": {"": 1.0}, "B4": {"20%": 0.1}}, formats)
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


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("datasketch")
    from bench.curate import run_curate, write_datasets
    from bench.embed import run_embed
    from bench.judge import j2, j3, j5, j6, j7, j8
    from fixtures.fake import fake_embed, fake_tokens, write_run
    from fixtures.world import gold_correct, per_call_eval, replay, teacher, trained_on
    from synthetic import make_repo
    from test_curate import teacher_config

    _, config_path, config = make_repo(tmp_path, monkeypatch)
    config = teacher_config(config)
    config["roles"]["production_llm"]["model"] = "teacher-model"
    config["prices"] = {"as_of": "2026-09-30", "table": {
        "teacher-model": {"input_per_mtok": 3.0, "cached_input_per_mtok": 0.3, "output_per_mtok": 12.0, "batch_discount": 0.5},
        "engine-model": {"input_per_mtok": 0.2, "cached_input_per_mtok": 0.02, "output_per_mtok": 0.6, "batch_discount": 0.5}}}
    config["cost"]["p95_slo_ms"] = 1000
    config["modal"] = {"gpu_prices": {"as_of": "2026-09-30", "usd_per_s": {"L4": 0.8 / 3600}}}  # F3's key
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
    j8_path = j8.run(loadtests, config)
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
    plan = {"split": "test", "arms": {}, "format": {}, "pilot": {}}
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
    looser["thresholds"]["concordance_min"] = 0.5  # Appendix B's routine bar, lowered after the test
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
    (root / "registry" / "test" / "agent-B4-again.manifest.json").write_text(json.dumps({**identity, "status": "failed"}))
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
    pipeline["plan"].write_text(yaml.safe_dump({**plan, "j8": relative(j8.run(loadtests, gpu))}))
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


def test_a_replay_of_some_call_sites_decides_neither_appendix_b_nor_k4(pipeline):
    """The per-call replay declares the call sites it replayed: covering only the gold ones, it
    cannot say whether the SLM passes the routine, so Appendix B and K4 are no data."""
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
    assert appendix_b["verdict"] == "no data (the replay covers only generate_candidate, revise)"
    assert data["per_call_uncovered"] == sorted(report.ROUTINE)
    assert "No data: the per-call replay covers only generate_candidate, revise." in (out / "report.md").read_text()
    assert gold_only  # the teacher had gold calls to replay


def test_several_registered_replays_leave_k4_and_appendix_b_without_a_verdict(pipeline):
    root = pipeline["root"]
    registered = json.loads((root / "registry" / "test" / "replay-B4-test.manifest.json").read_text())
    identity = {k: registered[k] for k in ("type", "arm", "engine", "commit", "prereg_hash")}
    (root / "registry" / "test" / "replay-B4-again.intent.json").write_text(json.dumps(
        {**identity, "run_id": "replay-B4-again", "split": "test", "started_at": "2026-10-01T10:00:00+00:00"}))
    (root / "registry" / "test" / "replay-B4-again.manifest.json").write_text(json.dumps({**identity, "status": "failed"}))
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
    (root / "registry" / "test" / "agent-B4-again.manifest.json").write_text(json.dumps({**identity, "status": "failed"}))
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
        assert all(part.endswith(f"no verdict (several test runs of one configuration of B4: {several})")
                   for part in verdict.split(" · "))


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
    other = relative(j8.run(only_adapter, pipeline["config"]))
    b4 = read_result(plan["arms"]["B4"]["cost"], "J3")
    from bench.judge import j3
    changed = json.loads(json.dumps(plan))
    changed["arms"]["B4"]["cost"] = relative(j3.run(b4["reads"]["run"]["run_id"], "eval-B4", other, pipeline["config"]))
    pipeline["plan"].write_text(yaml.safe_dump(changed))
    with pytest.raises(JudgmentError, match="another load test than the plan's j8"):
        report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)


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
    register(root, "replay-B4-again", identity, status="failed")
    data = json.loads((report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority,
                                  pilot_ids=PILOT) / "report.json").read_text())
    assert {t["several_runs"] and tuple(t["several_runs"]) for t in data["gold_tests"].values()} == \
        {("replay-B4-again", "replay-B4-test")}
    appendix_b = next(r for r in data["map"] if r["claim"].startswith("Appendix B"))
    assert appendix_b["verdict"] == "no verdict (several test runs of one configuration: replay-B4-again, replay-B4-test)"


def test_several_runs_of_b0_reach_every_row_on_its_evidence(pipeline):
    root = pipeline["root"]
    b0 = json.loads((root / "registry" / "test" / "agent-B0-test.manifest.json").read_text())
    register(root, "agent-B0-again", {k: b0[k] for k in ("type", "arm", "engine", "commit", "prereg_hash")}, status="failed")
    data = json.loads((report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority,
                                  pilot_ids=PILOT) / "report.json").read_text())
    runs = "agent-B0-again, agent-B0-test"
    verdicts = {r["claim"].split(":")[0]: r["verdict"] for r in data["map"]}
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
