"""J7 · S6: the cheapest engine that passes per cluster, the production LLM otherwise, an agreement bar
tied to the teacher's agreement with itself (T4), and an allocation fact the router accepts."""
from fractions import Fraction

import pytest

from bench import paths
from bench.contracts import facts, router
from bench.judge import j7
from bench.judge.base import JudgmentError, read_result, reference, relative, write_result
from bench.judge.j4 import noninferiority
from fixtures.fake import repo, write_run
from fixtures.world import GOLD_SITES, gold_correct, per_call_eval, replay, teacher, trained_on, unit

SETTINGS = {"selection_delta_pp": 2.5, "seed": 1, "n_boot": 2000, "min_calls": 5, "concordance_min": 0.95,
            "concordance_slack_pp": 2}
BARS = {engine: {cluster: 0.95 for cluster in ("c0", "c1", "c2", "c3")} for engine in j7.ENGINES}  # an exact teacher


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
    out = j7.allocate(["c0", "c1", "c2", "c3"], ev, costs, noninferiority, SETTINGS, BARS)
    assert {c: out[c]["engine"] for c in out} == {"c0": "slm", "c1": "cheap_alt", "c2": "production_llm",
                                                   "c3": "production_llm"}
    assert out["c3"]["evidence"]["slm"]["why"] == "no calib call in this cluster"
    costs["c0"] = {"slm": 3.0, "cheap_alt": 2.0}  # when the SLM costs more, the passing cheap_alt wins
    assert j7.allocate(["c0"], ev, costs, noninferiority, SETTINGS, BARS)["c0"]["engine"] == "cheap_alt"


def test_gold_clusters_need_non_inferiority_at_the_selection_margin():
    teacher_ok = {str(q): q % 10 != 0 for q in range(400)}
    same = dict(teacher_ok)
    worse = {q: ok and int(q) % 7 != 0 for q, ok in teacher_ok.items()}
    ev = {"slm": {"c0": gold(teacher_ok, worse)}, "cheap_alt": {"c0": gold(teacher_ok, same)}}
    out = j7.allocate(["c0"], ev, {"c0": {"slm": 1.0, "cheap_alt": 2.0}}, noninferiority, SETTINGS, BARS)
    assert out["c0"]["engine"] == "cheap_alt"
    assert out["c0"]["evidence"]["slm"]["gold"]["j4"]["noninferior"] is False
    few = {"slm": {"c0": gold({"1": True}, {"1": True})}, "cheap_alt": {}}
    assert j7.allocate(["c0"], few, {"c0": {"slm": 1.0}}, noninferiority, SETTINGS, BARS)["c0"]["engine"] == "production_llm"


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
    out = j7.allocate(["c0"], ev, {"c0": {"slm": 1.0, "cheap_alt": 1.5}}, noninferiority, SETTINGS, BARS)
    assert out["c0"]["engine"] == "slm" and out["c0"]["cost_dependent"] and out["c0"]["cost_ratio"] == 1.5


def test_agreement_needs_enough_calls():
    verdict = j7.passes(evidence(agreement(1.0, n=4)), noninferiority, SETTINGS, 0.95)
    assert not verdict["passes"] and "min_calls" in verdict["why"]
    assert j7.passes(evidence(agreement(1.0, n=5)), noninferiority, SETTINGS, 0.95)["passes"]


def test_each_call_sites_bar_is_capped_by_the_teachers_agreement_with_itself():
    """T4: bar = min(thresholds.concordance_min, A_tt − thresholds.concordance_slack_pp / 100)."""
    measured = {site: agreement(rate)["agreement"] for site, rate in (("exact", 1.0), ("close", 0.97), ("noisy", 0.93))}
    measured["unmeasured"] = {"n": 0, "agree": 0, "rate": None}
    bars = j7.site_bars(measured, SETTINGS)
    assert bars == {"exact": Fraction(95, 100), "close": Fraction(95, 100), "noisy": Fraction(91, 100)}  # 0.95 caps; 0.93 − 0.02
    assert j7.site_bars({"noisy": measured["noisy"]}, {**SETTINGS, "concordance_slack_pp": 5}) == {"noisy": Fraction(88, 100)}
    # a cluster's bar: its call sites' bars weighted by its calls of each: (300 × 0.95 + 100 × 0.91) / 400 = 0.94
    assert j7.cluster_bar({"exact": 300, "noisy": 100}, bars) == Fraction(94, 100)
    assert j7.cluster_bar({"noisy": 7}, bars) == Fraction(91, 100) and j7.cluster_bar({}, bars) is None
    with pytest.raises(JudgmentError, match="does not measure its agreement on unmeasured"):
        j7.cluster_bar({"exact": 10, "unmeasured": 5}, bars)


def test_an_engine_passes_at_the_teachers_bar_not_at_a_bar_the_teacher_misses():
    """A teacher that agrees with itself 93% of the time: an engine at 92% passes (bar 0.91), where the
    untied 0.95 would keep the calls on the LLM whatever the engine did."""
    verdict = j7.passes(evidence(agreement(0.92)), noninferiority, SETTINGS, 0.91)
    assert verdict["passes"] and verdict["agreement"] == {"n": 100, "rate": 0.92, "bar": 0.91, "passes": True, "proxy": True}
    untied = j7.passes(evidence(agreement(0.92)), noninferiority, SETTINGS, 0.95)
    assert not untied["passes"] and "A_tt" in untied["why"]
    assert not j7.passes(evidence(agreement(0.90)), noninferiority, SETTINGS, 0.91)["passes"]
    assert not j7.passes(evidence(agreement(1.0)), noninferiority, SETTINGS)["passes"]  # no bar, no pass


def test_an_engine_exactly_on_its_bar_passes():
    """The bar is reached at equality, and the comparison is exact: a cluster of two call sites, 3 and 7
    calls, both at 0.95, has a bar of 0.95 (in floats, 3 × 0.95 + 7 × 0.95 over 10 is not 0.95)."""
    exact = {site: agreement(1.0)["agreement"] for site in ("a", "b")}
    bar = j7.cluster_bar({"a": 3, "b": 7}, j7.site_bars(exact, SETTINGS))
    assert bar == Fraction(95, 100)
    on_it = j7.passes(evidence(agreement(0.95)), noninferiority, SETTINGS, bar)
    assert on_it["passes"] and on_it["agreement"]["bar"] == 0.95
    assert not j7.passes(evidence(agreement(0.949, n=1000)), noninferiority, SETTINGS, bar)["passes"]
    # a teacher at 96% with itself: the bar is 0.96 − 0.02, and an engine at 94% is on it
    tied = j7.cluster_bar({"a": 3, "b": 7}, j7.site_bars({site: agreement(0.96)["agreement"] for site in ("a", "b")}, SETTINGS))
    assert tied == Fraction(94, 100) and j7.passes(evidence(agreement(0.94)), noninferiority, SETTINGS, tied)["passes"]
    assert not j7.passes(evidence(agreement(0.939, n=1000)), noninferiority, SETTINGS, tied)["passes"]


def test_a_mixed_cluster_must_pass_both_tests():
    teacher_ok = {str(q): True for q in range(400)}
    mixed = gold(teacher_ok, dict(teacher_ok))
    mixed["agreement"] = agreement(0.80)["agreement"]
    verdict = j7.passes(mixed, noninferiority, SETTINGS, 0.95)
    assert not verdict["passes"] and verdict["gold"]["passes"] and "concordance_min" in verdict["why"]


# ---------------------------------------------------------------- the execution

QUESTIONS = [str(q) for q in range(1000, 1200)]
PILOT = QUESTIONS[:50]
DIFFICULTY = {q: ("simple", "moderate", "challenging")[int(q) % 3] for q in QUESTIONS}
ROUTINE_SITES = ["agent_ir", "extract_keywords", "filter_column", "select_tables"]  # the world's call sites without gold


def self_replay(quality=1.0, questions=PILOT, run_id="replay-self", engine="production_llm", sites=ROUTINE_SITES):
    """The teacher's calib run replayed on the production LLM itself, on the pilot questions (T4)."""
    source = [c for c in teacher("agent-B0-calib", "calib", QUESTIONS) if c["question_id"] in questions and c["call_site"] in sites]
    write_run(run_id, {"type": "replay", "source_run_id": "agent-B0-calib", "engine": engine, "split": "calib",
                       "call_sites": sites}, replay(source, run_id, engine, quality, model="teacher-model"))
    return run_id


def run_j7(world, **kwargs):
    config, centroids, choice, adapters, replays, j8, j6 = world
    options = {"teacher_self_replay": "replay-self", "pilot_ids": PILOT, "difficulty": DIFFICULTY, **kwargs}
    return j7.run(str(centroids), str(adapters), replays, "eval-t", str(j8), str(j6), config, noninferiority, **options)


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
    self_replay()
    replays = {"cheap_alt": ("replay-cheap", "eval-cheap"), "slm": ("replay-b4", "eval-b4")}
    return config, centroids, choice, adapters, replays, j8, j6


def test_run_writes_an_allocation_the_router_accepts(tmp_path, monkeypatch):
    world = config, centroids, choice, adapters, replays, j8, j6 = allocation_world(tmp_path, monkeypatch)
    path, fact = run_j7(world)
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
    world = config, centroids, choice, adapters, replays, j8, j6 = allocation_world(tmp_path, monkeypatch)
    other = facts.write_fact("J5", "centroids", {"embedding": facts.read_fact(str(centroids), "centroids")[0]["embedding"],
                                                  "clusters": {"c0": [1.0, 0.0]}})
    with pytest.raises(JudgmentError, match="not trained on these centroids"):
        run_j7((config, other, choice, adapters, replays, j8, j6))
    manifest_path = paths.RUNS / "replay-b4" / "manifest.json"
    routed = __import__("json").loads(manifest_path.read_text())
    manifest_path.write_text(__import__("json").dumps({**routed, "facts": {}}))
    with pytest.raises(JudgmentError, match="does not record"):
        run_j7(world)
    manifest_path.write_text(__import__("json").dumps(routed))
    other_j8 = write_result("J8", {}, {"engine": "slm:granite-4.2-8b", "cost_per_request": {"20%": 0.1}})
    with pytest.raises(JudgmentError, match="not the base of these adapters"):
        run_j7((config, centroids, choice, adapters, replays, other_j8, j6))
    config["allocation"]["min_calls"] = None
    with pytest.raises(JudgmentError, match="min_calls"):
        run_j7(world)
    assert not (paths.ROOT / "judgments" / "J7").exists()


def test_j7_refuses_without_the_teachers_self_replay(tmp_path, monkeypatch):
    """T4: with no A_tt the agreement bar would be untied from what the teacher itself reaches."""
    world = allocation_world(tmp_path, monkeypatch)
    with pytest.raises(JudgmentError, match="--teacher-self-replay"):
        run_j7(world, teacher_self_replay=None)
    with pytest.raises(JudgmentError, match="not a calib replay on production_llm"):
        run_j7(world, teacher_self_replay=self_replay(run_id="replay-self-cheap", engine="cheap_alt"))
    with pytest.raises(JudgmentError, match="not on the pilot questions"):
        run_j7(world, teacher_self_replay=self_replay(run_id="replay-self-other", questions=QUESTIONS[10:60]))
    write_run("agent-B0-calib-again", {"type": "agent", "arm": "B0", "split": "calib"}, teacher("agent-B0-calib-again", "calib", PILOT))
    again = replay([c for c in teacher("agent-B0-calib-again", "calib", PILOT) if c["call_site"] in ROUTINE_SITES],
                   "replay-self-again", "production_llm", 1.0)
    write_run("replay-self-again", {"type": "replay", "source_run_id": "agent-B0-calib-again", "engine": "production_llm",
                                    "split": "calib", "call_sites": ROUTINE_SITES}, again)
    with pytest.raises(JudgmentError, match="replays another teacher run"):
        run_j7(world, teacher_self_replay="replay-self-again")
    partial = self_replay(run_id="replay-self-partial", sites=[s for s in ROUTINE_SITES if s != "agent_ir"])
    with pytest.raises(JudgmentError, match="does not measure its agreement on agent_ir"):
        run_j7(world, teacher_self_replay=partial)  # a call site the clusters hold and the self-replay left out
    assert not (paths.ROOT / "judgments" / "J7").exists()


def test_the_bar_follows_the_teachers_self_agreement_by_call_site_and_difficulty(tmp_path, monkeypatch):
    """A teacher that repeats itself 90% of the time: A_tt per call site is its agreement with its own
    replay on the pilot questions, by difficulty too; each site's bar is min(0.95, A_tt − 0.02); the
    routine cluster's bar is its sites' bars weighted by its calls."""
    world = config, *_ = allocation_world(tmp_path, monkeypatch)
    noisy = self_replay(quality=0.9, run_id="replay-self-noisy")
    result = read_result(run_j7(world, teacher_self_replay=noisy)[0], "J7")
    measured, calls = result["result"]["teacher_self_agreement"], teacher("agent-B0-calib", "calib", QUESTIONS)
    assert sorted(measured) == sorted(ROUTINE_SITES)
    for site in ROUTINE_SITES:  # the fixture's rule, recomputed here: it repeats the teacher when unit(...) < quality
        mine = [c for c in calls if c["call_site"] == site and c["question_id"] in PILOT]
        agree = [unit("production_llm", c["question_id"], site, c["invocation_key"]) < 0.9 for c in mine]
        assert (measured[site]["n"], measured[site]["agree"]) == (len(mine), sum(agree))
        assert measured[site]["bar"] == pytest.approx(min(0.95, sum(agree) / len(mine) - 0.02))
        levels = measured[site]["by_difficulty"]
        assert sorted(levels) == ["challenging", "moderate", "simple"]
        for level, found in levels.items():
            of_level = [ok for c, ok in zip(mine, agree) if DIFFICULTY[c["question_id"]] == level]
            assert (found["n"], found["agree"], found["rate"]) == (len(of_level), sum(of_level), sum(of_level) / len(of_level))
    assert any(found["bar"] < 0.95 for found in measured.values())  # the cap moved with the teacher
    per_site = {site: sum(c["call_site"] == site for c in calls) for site in ROUTINE_SITES}  # c1 holds all of them
    expected = sum(n * measured[site]["bar"] for site, n in per_site.items()) / sum(per_site.values())
    for engine in j7.ENGINES:
        found = result["result"]["clusters"]["c1"]["evidence"][engine]["agreement"]
        assert found["bar"] == pytest.approx(expected) and found["passes"] == (found["rate"] >= found["bar"])
    assert result["result"]["settings"]["concordance_slack_pp"] == 2 and result["result"]["pilot_ids"] == PILOT
    assert result["reads"]["teacher_self_replay"] == reference(noisy)
    assert result["reads"]["config"] == j7.CONFIG_KEYS
    assert {"thresholds.concordance_slack_pp", "thresholds.selection_delta_pp", "allocation.min_calls",
            "seeds.calib_split", "stats.pilot_mix", "stats.pilot_size"} <= set(j7.CONFIG_KEYS)  # the pilot's keys too (bench.data.pilot_ids)


def test_the_defaults_are_the_pilot_accessor_and_the_calib_difficulties(tmp_path, monkeypatch):
    """Without pilot_ids and difficulty, J7 reads the pilot through bench.data.pilot_ids and the
    difficulty of the calib questions through bench.data.questions_for."""
    from bench import data
    world = config, *_ = allocation_world(tmp_path, monkeypatch)
    config["stats"]["pilot_size"] = 50
    monkeypatch.setattr(data, "load_splits", lambda: {"train": [], "calib": QUESTIONS, "test": [], "excluded": []})
    monkeypatch.setattr(data, "questions_for", lambda config, split: {q: {"difficulty": DIFFICULTY[q]} for q in QUESTIONS}
                        if split == "calib" else {})
    pilot = data.pilot_ids(config)
    assert pilot != PILOT and len(pilot) == 50  # the registered draw, not the first ids
    registered = self_replay(run_id="replay-self-registered", questions=pilot)
    result = read_result(run_j7(world, teacher_self_replay=registered, pilot_ids=None, difficulty=None)[0], "J7")["result"]
    assert result["pilot_ids"] == pilot
    assert set(result["teacher_self_agreement"]["filter_column"]["by_difficulty"]) == {"simple", "moderate", "challenging"}
    with pytest.raises(JudgmentError, match="not on the pilot questions"):  # the first 50 ids are not the pilot
        run_j7(world, pilot_ids=None, difficulty=None)
