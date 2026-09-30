"""R · the report: the SPEC §5 map rules, the test registry, and the whole pipeline end to end on a
fake execution (S2 → S3 → S4 → S6, the costs, the per-call evaluation, the report)."""
import json
import subprocess
import xml.etree.ElementTree as ET

import pytest
import yaml

from bench import paths, report
from bench.contracts import clusters, facts
from bench.judge.base import JudgmentError, read_jsonl, read_result, relative, write_result
from bench.judge.j1 import ex_summary, ex_table
from bench.judge.j4 import margin, noninferiority
from fixtures.world import unit

VOCABULARY = ("confirms", "refutes", "inconclusive", "not testable", "descriptive", "no data", "does not refute",
              "no verdict")


def j4(ok=True, testable=True, diff=-0.01, pilot=0.1, ci_low=None, ci_high=None, margin_from=None):
    """A J4 result as F2's J4 returns it (with the upper bound F2 is adding)."""
    margin_from = margin_from or ("pilot" if pilot is not None else "pairs")
    return {"d": 0.1, "d_pilot": pilot, "margin_from": margin_from, "delta": 0.04,
            "diff": diff, "ci_low": ci_low if ci_low is not None else (0.0 if ok else -0.09),
            "ci_high": ci_high if ci_high is not None else diff + 0.05,
            "noninferior": (ok if testable else None) if margin_from != "pairs" else None, "power": 0.8, "testable": testable}


def j4_results(ok4=True, ok5=True, testable=True, diff=-0.01, pilot=0.1):
    return {"B0|B4": j4(ok4, testable, diff, pilot), "B0|B5": j4(ok5, testable, diff, pilot)}


def data_with(tests, costs, formats=None, repair=None, per_call=None):
    arms = {arm: {"cost_per_correct": c, "replaceable_fraction": None} for arm, c in costs.items()}
    return {"arms": arms, "tests": tests, "formats": formats or {}, "repair_test": repair,
            "judgments": {"j5": None, "per_call": per_call}, "concordance_min": 0.95, "v3_min_ratio": 3}


def row(rows, prefix):
    return next(r for r in rows if r["claim"].startswith(prefix))


def test_the_outcome_of_a_comparison():
    assert report.outcome(None) == "no data"
    assert report.outcome(j4(pilot=None)) == "no pilot"
    assert report.outcome(j4(testable=False)) == "not testable"
    assert report.outcome(j4(True)) == "non-inferior"
    assert report.outcome(j4(False, diff=-0.2)) == "worse"
    assert report.outcome(j4(False, diff=-0.02)) == "inconclusive"


def test_v1_confirms_refutes_and_says_why_not():
    costs = {"B0": {"": 1.0}}
    assert row(report.claims_map(data_with(j4_results(), costs)), "V1")["verdict"] == "confirms"
    assert row(report.claims_map(data_with(j4_results(False, False, diff=-0.2), costs)), "V1")["verdict"] == "refutes"
    assert row(report.claims_map(data_with(j4_results(False, False), costs)), "V1")["verdict"] == "inconclusive"
    assert row(report.claims_map(data_with(j4_results(testable=False), costs)), "V1")["verdict"] == "not testable"
    assert row(report.claims_map(data_with({}, costs)), "V1")["verdict"] == "no data"
    # without the pilot's d, J4 gives no verdict, and the report never reads it as a pass
    no_pilot = row(report.claims_map(data_with(j4_results(pilot=None), costs)), "V1")
    assert no_pilot["verdict"] == "no verdict (no pilot d)" and "from the pairs" in no_pilot["result"]


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
    unpiloted = row(report.claims_map(data_with(j4_results(pilot=None), costs)), "V3")["verdict"]
    assert unpiloted.startswith("20%: no verdict (no pilot d) (no trained arm passes V1)")


def test_format_claim_compares_each_call_site():
    formats = {"B0": {"a": {"rate": 0.9}, "b": {"rate": 1.0}}, "B4": {"a": {"rate": 1.0}, "b": {"rate": 1.0}}}
    assert row(report.claims_map(data_with({}, {}, formats)), "A5")["verdict"] == "confirms"
    formats["B4"]["b"]["rate"] = 0.99  # better overall, worse on one call site
    a5 = row(report.claims_map(data_with({}, {}, formats)), "A5")
    assert a5["verdict"] == "refutes" and a5["result"].startswith("B4 below B0 on b")


def per_call(rates):
    return {"per_call_site": {site: {"agreement": {"rate": rate}} for site, rate in rates.items()}}


def test_appendix_b_confirms_only_when_repair_is_worse_and_the_routine_passes():
    routine = per_call({"filter_column": 0.97, "select_tables": 0.99})
    verdict = lambda repair, calls=routine: row(report.claims_map(data_with({}, {}, repair=repair, per_call=calls)),  # noqa: E731
                                               "Appendix B")["verdict"]
    assert verdict(j4(False, diff=-0.2)) == "confirms (the routine by the agreement proxy, which supports no per-cluster claim (D15))"
    assert verdict(j4(True)) == "refutes (the SLM ties on repair)"
    assert verdict(j4(False, diff=-0.02)) == "inconclusive"
    assert verdict(j4(pilot=None)) == "no verdict (no pilot d)"
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


def test_no_verdict_is_read_from_j4_itself_and_refuting_takes_the_upper_bound():
    assert report.outcome(j4(True, pilot=None, margin_from="given")) == "non-inferior"  # J7-style margin, no d_pilot
    assert report.outcome(j4(True, pilot=0.1, margin_from="pairs")) == "no pilot"
    assert report.outcome({**j4(True), "noninferior": None}) == "no pilot"
    assert report.outcome(j4(False, diff=-0.2, ci_high=-0.02)) == "inconclusive"  # the point below −Δ, not the bound
    assert report.outcome({k: v for k, v in j4(False, diff=-0.2).items() if k != "ci_high"}) == "inconclusive"
    assert report.outcome(j4(False, diff=-0.2, ci_high=-0.05)) == "worse"


def test_appendix_b_with_no_routine_measured_is_no_data():
    row_ = row(report.claims_map(data_with({}, {}, repair=j4(False, diff=-0.2), per_call=per_call({}))), "Appendix B")
    assert row_["verdict"] == "no data (no routine call site measured)"


def test_cost_verdicts_carry_their_labels_and_an_upper_bound_is_inconclusive():
    tests = {**j4_results()}
    costs = {"B0": {"": 1.0}, "B4": {"20%": 0.1}, "B5": {"20%": 0.05}}
    data = data_with(tests, costs)
    data["arms"]["B5"]["slm_cost_basis"] = "extrapolated from per-adapter load tests"
    assert row(report.claims_map(data), "V3")["verdict"] == \
        "20%: confirms (20.0× vs B0) [costs: extrapolated from per-adapter load tests]"
    data["arms"]["B0"].update(upper_bound=True, cache_not_reported=7)
    v3 = row(report.claims_map(data), "V3")["verdict"]
    assert v3.startswith("20%: inconclusive (rests on an upper-bound cost) [costs: ") and \
        "upper bound (cache not reported for 7 calls of B0)" in v3
    assert row(report.claims_map(data), "A6")["verdict"].startswith("20%: inconclusive (rests on an upper-bound cost)")


def test_several_test_runs_of_one_configuration_get_no_verdict():
    tests = {"B0|B4": {**j4(True), "several_runs": ["agent-B4-a", "agent-B4-b"]}, "B0|B5": j4(True)}
    rows = report.claims_map(data_with(tests, {"B0": {"": 1.0}, "B4": {"20%": 0.1}}))
    assert row(rows, "V1")["verdict"] == "no verdict (several test runs of one configuration: agent-B4-a, agent-B4-b)"
    assert row(rows, "V3")["verdict"] == "20%: no verdict (several test runs of one configuration: agent-B4-a, agent-B4-b)"


def registry_of(*runs):
    return {"available": True, "runs": [{"run_id": r, "type": "agent", "arm": a, "engine": None, "status": st,
                                          "commit": "c", "prereg_hash": "h", "started_at": "2026-10-01T09:00:00+00:00",
                                          "facts": f} for r, a, st, f in runs]}


def test_a_test_report_is_bound_to_the_registry(tmp_path, monkeypatch):
    from fixtures.fake import repo, write_run
    repo(tmp_path, monkeypatch)
    arms = {"B0": {"run_id": "b0"}, "B4": {"run_id": "b4"}}
    judged = {"j6": {"choice_fact": {"sha256": "C"}}, "j5": {"centroids": {"sha256": "K"}},
              "j7": {"adapters": "A", "allocation_fact": {"sha256": "L"}}}
    good = registry_of(("b0", "B0", "done", {}), ("b4", "B4", "done", {"choice": "C", "centroids": "K", "adapters": "A"}))
    bind = lambda registry, reads={}, pilot={}, split="test": report._registry_bindings(  # noqa: E731
        split, {a: dict(v) for a, v in arms.items()}, registry, judged, reads, pilot)
    bind(good)
    with pytest.raises(JudgmentError, match="not a done entry"):
        bind(registry_of(("b0", "B0", "failed", {}), good["runs"][1].values().__iter__() and ("b4", "B4", "done", {})))
    with pytest.raises(JudgmentError, match="not registered"):
        bind(registry_of(("b4", "B4", "done", {"choice": "C", "centroids": "K", "adapters": "A"})))
    with pytest.raises(JudgmentError, match="recorded the centroids fact"):
        bind(registry_of(("b0", "B0", "done", {}), ("b4", "B4", "done", {"choice": "C", "centroids": "X", "adapters": "A"})))
    with pytest.raises(JudgmentError, match="another teacher run"):
        bind(good, reads={"per_call": {"teacher": {"run_id": "other-b0"}}})
    with pytest.raises(JudgmentError, match="test registry"):
        bind({"available": False, "reason": "not a git checkout", "runs": []})
    bind({"available": False, "reason": "x", "runs": []}, split="calib")  # only a test report is bound
    marked = {a: dict(v) for a, v in arms.items()}
    twice = registry_of(*[(r["run_id"], r["arm"], r["status"], r["facts"]) for r in good["runs"]], ("b4-again", "B4", "done", {}))
    report._registry_bindings("test", marked, twice, judged, {}, {})
    assert marked["B4"]["several_runs"] == ["b4", "b4-again"] and marked["B0"]["several_runs"] is None
    write_run("eval-pilot-late", {"type": "eval", "status": None, "finished_at": "2026-10-01T10:00:00+00:00"})
    write_run("eval-pilot-early", {"type": "eval", "status": None, "finished_at": "2026-09-30T10:00:00+00:00"})
    bind(good, pilot={"B0": "eval-pilot-early"})
    with pytest.raises(JudgmentError, match="did not finish before the first test"):
        bind(good, pilot={"B0": "eval-pilot-early", "B3": "eval-pilot-late"})


def test_the_chart_legend_reads_the_configured_utilizations():
    data = {"arms": {"B4": {"ex": 0.8, "cost_per_correct": {"30%": 0.2, "90%": 0.1}}}, "utilizations": ["30%", "90%"]}
    assert "one point per utilization, 30% (right) to 90% (left)" in report.chart_svg(data)


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


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("datasketch")
    from bench.curate import run_curate, write_datasets
    from bench.embed import run_embed
    from bench.judge import j2, j3, j5, j6, j7, j8
    from fixtures.fake import fake_embed, fake_tokens, write_run
    from fixtures.world import gold_correct, per_call_eval, replay, teacher
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
                                                   "adapters": {c: {"served_name": f"qwen3-8b-{c}", "sha256": "e" * 64}
                                                                for c in payload["clusters"]}})
    assigned = lambda c: clusters.assign(c["prompt_messages"], payload)  # noqa: E731
    # load test and S6 on calib
    for concurrency, p95, rps in ((1, 300, 3.0), (8, 800, 12.0), (32, 2500, 20.0)):
        write_run(f"loadtest-{concurrency}", {"type": "loadtest", "engine": "slm:qwen3-8b", "gpu": "L4", "concurrency": concurrency,
                                              "prefix_cache": True, "sweep_id": "sweep-qwen3-8b"},
                  files={"profile_export_aiperf.json": {"request_latency": {"unit": "ms", "p95": p95},
                                                         "request_throughput": {"unit": "requests/sec", "avg": rps}}})
    j8_path = j8.run(["loadtest-1", "loadtest-8", "loadtest-32"], config)
    b4_calib = replay(calib, "replay-B4-calib", "slm:qwen3-8b+lora:c", 0.99, cluster_of=assigned)
    run("replay-B4-calib", {"type": "replay", "source_run_id": "agent-B0-calib", "arm": "B4", "split": "calib",
                            "facts": {"choice": choice.parent.name, "centroids": centroids_sha, "adapters": adapters.parent.name}}, b4_calib)
    per_call_eval("eval-B4-calib", "replay-B4-calib", b4_calib, teacher_ok)
    cheap_calib = replay(calib, "replay-cheap-calib", "cheap_alt", 0.8)
    run("replay-cheap-calib", {"type": "replay", "source_run_id": "agent-B0-calib", "engine": "cheap_alt", "split": "calib"}, cheap_calib)
    per_call_eval("eval-cheap-calib", "replay-cheap-calib", cheap_calib, lambda c: teacher_ok(c) and unit("cheap", c["question_id"]) < 0.8)
    j7_path, allocation = j7.run(str(centroids), str(adapters), {"cheap_alt": ("replay-cheap-calib", "eval-cheap-calib"),
                                                                "slm": ("replay-B4-calib", "eval-B4-calib")},
                                 "eval-B0-calib-per-call", str(j8_path), str(j6_path), config, noninferiority,
                                 margin, len(TEST_IDS), PILOT)
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
                                "timeout_s": 60, "sqlite_version": "3.45.0", "prereg_hash": "d" * 64,
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
    per_call_eval("eval-B0-test-per-call", "agent-B0-test", b0, gold_correct("teacher-test", 0.9))
    per_call_eval("eval-B4-test-per-call", "replay-B4-test", b4_replay, gold_correct("teacher-test", 0.9))
    reads, result = j2.judge_replay("replay-B4-test", "eval-B4-test-per-call", "eval-B0-test-per-call")
    plan.update(per_call=relative(write_result("J2", reads, result)), j5=relative(j5_path), j6=relative(j6_path),
                j7=relative(j7_path), j8=relative(j8_path),
                teacher_train_cost=relative(j3.run("agent-B0-train", None, None, config)))
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan))
    # the test registry, as F1 commits it
    for args in (("init", "-q"), ("config", "user.email", "t@example.org"), ("config", "user.name", "t")):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    (tmp_path / "registry" / "test").mkdir(parents=True)
    recorded = {"B3": {"choice": choice.parent.name},
                "B4": {"choice": choice.parent.name, "centroids": centroids_sha, "adapters": adapters.parent.name},
                "B5": {"choice": choice.parent.name, "centroids": centroids_sha, "adapters": adapters.parent.name,
                       "allocation": allocation.parent.name}}
    for arm in arms_calls:
        engine = {"B2-production": "production_llm", "B2-cheap": "cheap_alt"}.get(arm)
        identity = {"type": "agent", "arm": arm.split("-")[0], "engine": engine, "commit": "c" * 40, "prereg_hash": "d" * 64}
        (tmp_path / "registry" / "test" / f"agent-{arm}-test.intent.json").write_text(json.dumps(
            {**identity, "run_id": f"agent-{arm}-test", "split": "test", "started_at": "2026-10-01T09:00:00+00:00"}))
        (tmp_path / "registry" / "test" / f"agent-{arm}-test.manifest.json").write_text(json.dumps(
            {**identity, "status": "done", "facts": recorded.get(arm, {})}))
    subprocess.run(["git", "-C", str(tmp_path), "add", "registry"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "registry"], check=True, capture_output=True)
    return {"config": config, "plan": plan_path, "allocation": allocated, "datasets": datasets, "j5": j5_path,
            "j6": j6_path, "root": tmp_path}


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
    # every margin came from the pilot: no comparison is left without a verdict
    assert all(t["margin_from"] == "pilot" for t in data["tests"].values()) and data["repair_test"]["d_pilot"] is not None
    assert "no verdict" not in markdown
    assert data["d_pilot"] == data["tests"]["B0|B4"]["d_pilot"] == data["tests"]["B0|B1"]["d_pilot"]  # one pilot
    pilot_rows = {arm: {r["question_id"]: r["correct"] for r in read_jsonl(paths.RUNS / f"eval-{arm}-pilot" / "results.jsonl")}
                  for arm in ("B0", "B3")}
    assert data["d_pilot"] == sum(pilot_rows["B0"][q] != pilot_rows["B3"][q] for q in PILOT) / len(PILOT)  # the pilot ids only
    assert [r["run_id"] for r in data["registry"]["runs"]] == sorted(f"agent-{arm}-test" for arm in QUALITY)
    assert data["arms"]["B5"]["replaceable_fraction"] is not None
    assert [data["arms"][a]["slm_cost_basis"] for a in ("B3", "B4", "B5")] == [
        "measured", "extrapolated from per-adapter load tests", "extrapolated from per-adapter load tests"]
    assert "SLM cost extrapolated from per-adapter load tests" in markdown
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


def test_without_a_pilot_no_comparison_gets_a_verdict(pipeline):
    plan = yaml.safe_load(pipeline["plan"].read_text())
    del plan["pilot"]
    pipeline["plan"].write_text(yaml.safe_dump(plan))
    data = json.loads((report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary,
                                  noninferiority, pilot_ids=PILOT) / "report.json").read_text())
    assert all(t["noninferior"] is None for t in data["tests"].values())
    assert report.v1_verdict(data["tests"]["B0|B4"], data["tests"]["B0|B5"]) == "no verdict (no pilot d)"


def test_report_refuses_an_eval_of_another_arm(pipeline):
    plan = yaml.safe_load(pipeline["plan"].read_text())
    plan["arms"]["B1"]["eval"], plan["arms"]["B3"]["eval"] = plan["arms"]["B3"]["eval"], plan["arms"]["B1"]["eval"]
    pipeline["plan"].write_text(yaml.safe_dump(plan))
    with pytest.raises(JudgmentError, match="is not B1 on test"):
        report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority, pilot_ids=PILOT)


def test_generation_and_repair_get_their_own_tests_with_the_pilot_restricted(pipeline):
    from bench.judge.j4 import noninferiority as j4_
    out = report.run(str(pipeline["plan"]), pipeline["config"], ex_table, ex_summary, noninferiority,
                     pilot_ids=PILOT)
    data = json.loads((out / "report.json").read_text())
    assert set(data["gold_tests"]) == {"generate_candidate", "revise"} and data["repair_test"] == data["gold_tests"]["revise"]
    assert "## Clusters with gold" in (out / "report.md").read_text()
    j6 = read_result(pipeline["j6"], "J6")["result"]
    gold = j6["per_call_site"][j6["choice"]]["revise"]["gold"]["by_question"]
    d_on = lambda ids: j4_({q: gold["replay"][q] for q in ids}, {q: gold["teacher"][q] for q in ids}, 5, 1, 10)["d"]  # noqa: E731
    pilot = sorted(set(PILOT) & set(gold["teacher"]))
    assert data["repair_test"]["d_pilot"] == d_on(pilot) != d_on(sorted(gold["teacher"]))  # the pilot questions only
    a4 = next(r for r in data["map"] if r["claim"].startswith("A4"))
    assert "B5 against B0: " in a4["result"]
    a6 = next(r for r in data["map"] if r["claim"].startswith("A6"))
    assert "[costs: extrapolated from per-adapter load tests]" in a6["verdict"]  # B5's SLM cost


def test_the_report_refuses_prices_of_different_dates_and_mismatched_results(pipeline):
    from bench.judge import j2, j3
    plan = yaml.safe_load(pipeline["plan"].read_text())
    dated = json.loads(json.dumps(pipeline["config"]))
    dated["prices"]["as_of"] = "2026-10-15"
    for key, value, message in (
            (("arms", "B1", "cost"), relative(j3.run("agent-B1-test", "eval-B1", None, dated)), "different dates"),
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
