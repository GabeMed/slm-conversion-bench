"""J7 · S6: the cheapest engine that passes per cluster, the production LLM otherwise, and an
allocation fact the router accepts."""
import pytest

from bench import paths
from bench.contracts import facts, router
from bench.judge import j7
from bench.judge.base import JudgmentError, read_result, reference, relative, write_result
from bench.judge.j4 import noninferiority
from fixtures.fake import repo, write_run
from fixtures.world import GOLD_SITES, gold_correct, per_call_eval, replay, teacher, trained_on

SETTINGS = {"selection_delta_pp": 2.5, "seed": 1, "n_boot": 2000, "min_calls": 5, "concordance_min": 0.95}


def agreement(rate, n=100):
    return {"n": n, "rate": rate, "gold": None, "agreement": {"n": n, "agree": round(rate * n), "rate": rate}}


def evidence(entry):
    return {"n": entry["n"], "gold": None, "agreement": entry["agreement"]}


def gold(teacher_correct, engine_correct):
    n = len(teacher_correct)
    return {"n": n, "agreement": None, "gold": {
        "n": n, "ex_replay": sum(engine_correct.values()) / n, "ex_teacher": sum(teacher_correct.values()) / n,
        "by_question": {"teacher": teacher_correct, "replay": engine_correct}}}


def test_cheapest_passing_engine_else_the_next_else_production():
    ev = {"slm": {"c0": evidence(agreement(0.99)), "c1": evidence(agreement(0.90)), "c2": evidence(agreement(0.50))},
          "cheap_alt": {"c0": evidence(agreement(0.99)), "c1": evidence(agreement(0.97)), "c2": evidence(agreement(0.60))}}
    costs = {c: {"slm": 1.0, "cheap_alt": 2.0} for c in ("c0", "c1", "c2")}
    out = j7.allocate(["c0", "c1", "c2", "c3"], ev, costs, noninferiority, SETTINGS)
    assert {c: out[c]["engine"] for c in out} == {"c0": "slm", "c1": "cheap_alt", "c2": "production_llm",
                                                   "c3": "production_llm"}
    assert out["c3"]["evidence"]["slm"]["why"] == "no calib call in this cluster"
    costs["c0"] = {"slm": 3.0, "cheap_alt": 2.0}  # when the SLM costs more, the passing cheap_alt wins
    assert j7.allocate(["c0"], ev, costs, noninferiority, SETTINGS)["c0"]["engine"] == "cheap_alt"


def test_gold_clusters_need_non_inferiority_at_the_selection_margin():
    teacher_ok = {str(q): q % 10 != 0 for q in range(400)}
    same = dict(teacher_ok)
    worse = {q: ok and int(q) % 7 != 0 for q, ok in teacher_ok.items()}
    ev = {"slm": {"c0": gold(teacher_ok, worse)}, "cheap_alt": {"c0": gold(teacher_ok, same)}}
    out = j7.allocate(["c0"], ev, {"c0": {"slm": 1.0, "cheap_alt": 2.0}}, noninferiority, SETTINGS)
    assert out["c0"]["engine"] == "cheap_alt"
    assert out["c0"]["evidence"]["slm"]["gold"]["j4"]["noninferior"] is False
    few = {"slm": {"c0": gold({"1": True}, {"1": True})}, "cheap_alt": {}}
    assert j7.allocate(["c0"], few, {"c0": {"slm": 1.0}}, noninferiority, SETTINGS)["c0"]["engine"] == "production_llm"


def test_the_selection_margin_is_fixed_and_stricter_than_the_tests():
    """Discordant both ways, no difference: within 5 p.p. (the test's Δ), not within the 2.5 p.p. of the
    selection. The margin is thresholds.selection_delta_pp whatever the pairs' discordance (T3)."""
    teacher_ok = {str(q): q % 10 != 0 for q in range(400)}
    engine_ok = dict(teacher_ok)
    for q in range(0, 300, 10):   # 30 questions only the engine gets right
        engine_ok[str(q)] = True
    for q in range(1, 300, 10):   # 30 only the teacher does
        engine_ok[str(q)] = False
    verdict = j7.passes(gold(teacher_ok, engine_ok), noninferiority, SETTINGS)
    test = verdict["gold"]["j4"]
    # d = 60 / 400 = 0.15: the bound is near -1.6448536 * sqrt(0.15 / 400) = -0.0319
    assert test["delta"] == 0.025 and test["d"] == 0.15 and -0.05 < test["ci_low"] < -0.025
    assert not verdict["passes"] and verdict["why"] == "gold: not non-inferior at thresholds.selection_delta_pp"
    assert "delta_cluster" not in verdict["gold"]  # no margin derived from a discordance
    quiet = {**teacher_ok, "1": False, "11": False}  # d = 0.005: the same margin, and it passes
    passed = j7.passes(gold(teacher_ok, quiet), noninferiority, SETTINGS)
    assert passed["passes"] and passed["gold"]["j4"]["delta"] == 0.025
    wider = j7.passes(gold(teacher_ok, engine_ok), noninferiority, {**SETTINGS, "selection_delta_pp": 5})
    assert wider["passes"] and wider["gold"]["j4"]["delta"] == 0.05  # the margin comes from the settings alone


def test_a_choice_that_rests_on_the_cost_order_alone_is_flagged():
    ev = {"slm": {"c0": evidence(agreement(0.99))}, "cheap_alt": {"c0": evidence(agreement(0.98))}}
    out = j7.allocate(["c0"], ev, {"c0": {"slm": 1.0, "cheap_alt": 1.5}}, noninferiority, SETTINGS)
    assert out["c0"]["engine"] == "slm" and out["c0"]["cost_dependent"] and out["c0"]["cost_ratio"] == 1.5


def test_agreement_needs_enough_calls():
    verdict = j7.passes(evidence(agreement(1.0, n=4)), noninferiority, SETTINGS)
    assert not verdict["passes"] and "min_calls" in verdict["why"]
    assert j7.passes(evidence(agreement(1.0, n=5)), noninferiority, SETTINGS)["passes"]


def test_a_mixed_cluster_must_pass_both_tests():
    teacher_ok = {str(q): True for q in range(400)}
    mixed = gold(teacher_ok, dict(teacher_ok))
    mixed["agreement"] = agreement(0.80)["agreement"]
    verdict = j7.passes(mixed, noninferiority, SETTINGS)
    assert not verdict["passes"] and verdict["gold"]["passes"] and "concordance_min" in verdict["why"]


# ---------------------------------------------------------------- the execution

QUESTIONS = [str(q) for q in range(1000, 1200)]


def cluster_of(c):
    return "c0" if c["call_site"] in GOLD_SITES else "c1"


def allocation_world(tmp_path, monkeypatch):
    config_path, config = repo(tmp_path, monkeypatch, {
        "prices": {"as_of": "2026-09-30", "table": {"engine-model": {"input_per_mtok": 0.5, "cached_input_per_mtok": 0.05,
                                                                     "output_per_mtok": 1.5, "batch_discount": 0.5}}},
        "allocation": {"min_calls": 5}, "stats": {"n_boot": 100}})
    embedding = {"model": "fake-embedder", "revision": "0" * 40, "max_seq_length": 64, "truncation": "tail", "text": "prompt"}
    centroids = facts.write_fact("J5", "centroids", {"embedding": embedding, "clusters": {"c0": [1.0, 0.0], "c1": [0.0, 1.0]}})
    choice = facts.write_fact("J6", "choice", {"slm": "qwen3-8b"})
    adapters = facts.write_fact("S5", "adapters", {
        "slm": "qwen3-8b", "choice": choice.parent.name, "centroids": centroids.parent.name, **trained_on(config),
        "adapters": {c: {"served_name": f"qwen3-8b-{c}", "sha256": c[1] * 64} for c in ("c0", "c1")}})
    t = teacher("agent-B0-calib", "calib", QUESTIONS)
    write_run("agent-B0-calib", {"type": "agent", "arm": "B0", "split": "calib"}, t)
    teacher_ok = gold_correct("t", 0.9)
    per_call_eval("eval-t", "agent-B0-calib", t, teacher_ok)
    b4 = replay(t, "replay-b4", "slm:qwen3-8b+lora:x", 1.0, cluster_of=cluster_of)
    write_run("replay-b4", {"type": "replay", "source_run_id": "agent-B0-calib", "arm": "B4", "split": "calib",
                            "facts": {"choice": choice.parent.name, "centroids": centroids.parent.name,
                                      "adapters": adapters.parent.name}}, b4)
    per_call_eval("eval-b4", "replay-b4", b4, lambda c: teacher_ok(c) and gold_correct("slm", 0.5)(c))
    cheap = replay(t, "replay-cheap", "cheap_alt", 0.9)
    write_run("replay-cheap", {"type": "replay", "source_run_id": "agent-B0-calib", "engine": "cheap_alt", "split": "calib"}, cheap)
    per_call_eval("eval-cheap", "replay-cheap", cheap, teacher_ok)  # the teacher's EX exactly
    j8 = write_result("J8", {}, {"engine": "slm:qwen3-8b", "cost_per_request": {"20%": 0.0001, "50%": 0.00004, "100%": 0.00002}})
    # the pilot: S4's zero-shot replay of the chosen candidate, before any adapter
    zeroshot = replay(t, "zeroshot-qwen", "slm:qwen3-8b", 0.9)
    write_run("zeroshot-qwen", {"type": "replay", "source_run_id": "agent-B0-calib", "engine": "slm:qwen3-8b", "split": "calib"}, zeroshot)
    per_call_eval("eval-zeroshot", "zeroshot-qwen", zeroshot, lambda c: teacher_ok(c) and gold_correct("zs", 0.95)(c))
    j6 = write_result("J6", {"qwen3-8b": {"teacher": reference("agent-B0-calib"), "replay": reference("zeroshot-qwen"),
                                          "replay_eval": reference("eval-zeroshot"), "teacher_eval": reference("eval-t")}},
                      {"choice": "qwen3-8b", "choice_fact": {"sha256": choice.parent.name}})
    replays = {"cheap_alt": ("replay-cheap", "eval-cheap"), "slm": ("replay-b4", "eval-b4")}
    return config, centroids, choice, adapters, replays, j8, j6


def test_run_writes_an_allocation_the_router_accepts(tmp_path, monkeypatch):
    config, centroids, choice, adapters, replays, j8, j6 = allocation_world(tmp_path, monkeypatch)
    path, fact = j7.run(str(centroids), str(adapters), replays, "eval-t", str(j8), str(j6), config, noninferiority)
    result = read_result(path, "J7")["result"]
    assert result["settings"]["selection_delta_pp"] == config["thresholds"]["selection_delta_pp"] == 2.5
    assert result["clusters"]["c0"]["evidence"]["cheap_alt"]["gold"]["j4"]["delta"] == 0.025
    # gold cluster: the SLM loses EX, cheap_alt matches the teacher; the rest: the SLM agrees always
    assert result["allocation"] == {"c0": "cheap_alt", "c1": "slm"}
    assert result["clusters"]["c1"]["order"] == ["slm", "cheap_alt"]
    assert result["clusters"]["c1"]["costs"]["slm"] == pytest.approx(0.0001)
    payload, _ = facts.read_fact(str(fact), "allocation")
    assert payload == {"centroids": centroids.parent.name, "adapters": adapters.parent.name,
                       "allocation": {"c0": "cheap_alt", "c1": "slm"}}
    config["arms"]["B5"] = {"choice": relative(choice), "centroids": relative(centroids),
                            "adapters": relative(adapters), "allocation": relative(fact)}
    assert set(router.arm_facts("B5", config)) == {"choice", "centroids", "adapters", "allocation"}
    assert router.possible_engines("B5", config) == ["cheap_alt", "production_llm", "slm:qwen3-8b+lora:qwen3-8b-c1"]


def test_run_refuses_mismatched_facts_and_missing_settings(tmp_path, monkeypatch):
    config, centroids, choice, adapters, replays, j8, j6 = allocation_world(tmp_path, monkeypatch)
    other = facts.write_fact("J5", "centroids", {"embedding": facts.read_fact(str(centroids), "centroids")[0]["embedding"],
                                                  "clusters": {"c0": [1.0, 0.0]}})
    with pytest.raises(JudgmentError, match="not trained on these centroids"):
        j7.run(str(other), str(adapters), replays, "eval-t", str(j8), str(j6), config, noninferiority)
    manifest_path = paths.RUNS / "replay-b4" / "manifest.json"
    routed = __import__("json").loads(manifest_path.read_text())
    manifest_path.write_text(__import__("json").dumps({**routed, "facts": {}}))
    with pytest.raises(JudgmentError, match="does not record"):
        j7.run(str(centroids), str(adapters), replays, "eval-t", str(j8), str(j6), config, noninferiority)
    manifest_path.write_text(__import__("json").dumps(routed))
    other_j8 = write_result("J8", {}, {"engine": "slm:granite-4.2-8b", "cost_per_request": {"20%": 0.1}})
    with pytest.raises(JudgmentError, match="not the base of these adapters"):
        j7.run(str(centroids), str(adapters), replays, "eval-t", str(other_j8), str(j6), config, noninferiority)
    config["allocation"]["min_calls"] = None
    with pytest.raises(JudgmentError, match="min_calls"):
        j7.run(str(centroids), str(adapters), replays, "eval-t", str(j8), str(j6), config, noninferiority)
    assert not (paths.ROOT / "judgments" / "J7").exists()
