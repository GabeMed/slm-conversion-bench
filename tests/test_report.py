"""R · the report: the SPEC §5 map rules, the test registry, and the whole pipeline end to end on a
fake execution (S2 → S3 → S4 → S6, the costs, the per-call evaluation, the report)."""
import json
import subprocess
import xml.etree.ElementTree as ET

import pytest
import yaml

from bench import paths, report
from bench.contracts import clusters, facts
from bench.judge.base import read_jsonl, read_result, relative, write_result
from fixtures.world import fake_noninferiority, unit

VOCABULARY = ("confirms", "refutes", "inconclusive", "not testable", "descriptive", "no data", "does not refute")


def j4_results(ok4=True, ok5=True, testable=True, diff=-0.01):
    t = lambda ok: {"d": 0.1, "delta": 0.04, "diff": diff, "ci_low": 0.0 if ok else -0.09,  # noqa: E731
                    "noninferior": ok, "power": 0.8, "testable": testable}
    return {"B0|B4": t(ok4), "B0|B5": t(ok5)}


def data_with(tests, costs, formats=None):
    arms = {arm: {"cost_per_correct": c, "replaceable_fraction": None} for arm, c in costs.items()}
    return {"arms": arms, "tests": tests, "formats": formats or {}, "repair_test": None,
            "judgments": {"j5": None, "per_call": None}, "concordance_min": 0.95}


def row(rows, prefix):
    return next(r for r in rows if r["claim"].startswith(prefix))


def test_v1_confirms_refutes_and_not_testable():
    costs = {"B0": {"": 1.0}}
    assert row(report.claims_map(data_with(j4_results(), costs)), "V1")["verdict"] == "confirms"
    assert row(report.claims_map(data_with(j4_results(False, False, diff=-0.2), costs)), "V1")["verdict"] == "refutes"
    assert row(report.claims_map(data_with(j4_results(False, False), costs)), "V1")["verdict"] == "inconclusive"
    assert row(report.claims_map(data_with(j4_results(testable=False), costs)), "V1")["verdict"] == "not testable"
    assert row(report.claims_map(data_with({}, costs)), "V1")["verdict"] == "no data"


def test_cost_claims_per_utilization():
    tests = {**j4_results(), "B0|B1": {"testable": True, "noninferior": False}}
    costs = {"B0": {"": 1.0}, "B1": {"": 0.2}, "B4": {"20%": 0.5, "100%": 0.1}, "B5": {"20%": 0.4, "100%": 0.05}}
    rows = report.claims_map(data_with(tests, costs))
    # B1 is not non-inferior to B0, so the best arm without training is B0 at $1.00
    assert row(rows, "V3")["verdict"] == "20%: refutes (2.5× vs B0) · 100%: confirms (20.0× vs B0)"
    assert row(rows, "A6")["verdict"] == "20%: confirms · 100%: confirms"
    assert row(rows, "AV2")["verdict"] == "20%: refutes the paper (AV2 wins) · 100%: does not refute"
    worse = {**costs, "B5": {"20%": 2.0, "100%": 1.5}}
    assert row(report.claims_map(data_with(tests, worse)), "A6")["verdict"] == "20%: refutes · 100%: refutes"


def test_format_claim_compares_all_invocations():
    formats = {"B0": {"a": {"n": 10, "valid": 9}}, "B4": {"a": {"n": 10, "valid": 10}}}
    assert row(report.claims_map(data_with({}, {}, formats)), "A5")["verdict"] == "confirms"
    formats["B4"]["a"]["valid"] = 8
    assert row(report.claims_map(data_with({}, {}, formats)), "A5")["verdict"] == "refutes"


def test_registry_reads_committed_test_manifests_and_intents(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init", "-q")
    for run_id, files in {"agent-B0-test-1": {"manifest.json": {"split": "test", "type": "agent", "arm": "B0",
                                                                  "status": "done", "commit": "a" * 40}},
                          "agent-B4-test-2": {"intent.json": {"arm": "B4"}},
                          "agent-B0-train-3": {"manifest.json": {"split": "train"}}}.items():
        (tmp_path / "runs" / run_id).mkdir(parents=True)
        for name, content in files.items():
            (tmp_path / "runs" / run_id / name).write_text(json.dumps(content))
    git("add", "-f", "runs")
    found = report.test_registry(tmp_path)
    assert found["available"] and [r["run_id"] for r in found["runs"]] == ["agent-B0-test-1", "agent-B4-test-2"]
    assert found["runs"][1]["status"].startswith("interrupted")


# ---------------------------------------------------------------- end to end

TEST_IDS = [str(q) for q in range(5000, 5400)]
CALIB_IDS = [str(q) for q in range(2000, 2060)]
QUALITY = {"B0": 1.0, "B1": 0.7, "B2-production": 0.6, "B2-cheap": 0.5, "B3": 0.5, "B4": 0.99, "B5": 0.995}


def fake_ex_table(eval_dirs):
    """A stand-in for F2's J1: the per-question table of eval executions."""
    rows = []
    for directory in eval_dirs:
        evaluated = json.loads((directory / "manifest.json").read_text())
        source = json.loads((paths.RUNS / evaluated["source_run_id"] / "manifest.json").read_text())
        rows += [{"arm": source["arm"], "split": source["split"], "question_id": r["question_id"],
                  "difficulty": r["difficulty"], "correct": r["correct"]}
                 for r in read_jsonl(directory / "results.jsonl")]
    return rows


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
    config["cost"].update(gpu_price_per_hour={"L4": 0.8}, p95_slo_ms=1000)
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
        write_run(f"loadtest-{concurrency}", {"type": "loadtest", "engine": "slm:qwen3-8b", "gpu": "L4", "concurrency": concurrency},
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
                                 "eval-B0-calib-per-call", str(j8_path), config, fake_noninferiority)
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
    plan = {"split": "test", "arms": {}, "format": {}}
    b0_ok = {q: unit("B0", q) < 0.8 for q in TEST_IDS}
    for arm, calls in arms_calls.items():
        run_id = f"agent-{arm}-test"
        calls = [{**c, "run_id": run_id} for c in calls]
        run(run_id, {"type": "agent", "arm": arm.split("-")[0], "split": "test", "question_ids": TEST_IDS}, calls)
        rows = [{"question_id": q, "difficulty": ("simple", "moderate", "challenging")[int(q) % 3],
                 "correct": b0_ok[q] if unit(arm, "keep", q) < QUALITY[arm] else not b0_ok[q]} for q in TEST_IDS]
        write_run(f"eval-{arm}", {"type": "eval", "source_run_id": run_id}, files={"results.jsonl": rows})
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
    return {"config": config, "plan": plan_path, "allocation": allocated, "datasets": datasets, "j5": j5_path}


def test_report_end_to_end_on_a_fake_execution(pipeline):
    out = report.run(str(pipeline["plan"]), pipeline["config"], fake_ex_table, fake_noninferiority)
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
    assert data["arms"]["B5"]["replaceable_fraction"] is not None
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
    assert report.run(str(pipeline["plan"]), pipeline["config"], fake_ex_table, fake_noninferiority) == out
