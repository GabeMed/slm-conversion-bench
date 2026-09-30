"""J6 · S4: the per-call-site score on calib, the pre-registered tie-break, and the choice fact."""
import pytest

from bench import paths
from bench.contracts import facts
from bench.judge import j6
from bench.judge.base import JudgmentError, read_result
from fixtures.fake import repo, write_run
from fixtures.world import gold_correct, per_call_eval, replay, teacher

QUESTIONS = [str(q) for q in range(100, 130)]


def site(gold=None, agreement=None):
    return {"gold": {"ex_replay": gold} if gold is not None else None,
            "agreement": {"rate": agreement} if agreement is not None else None}


def test_mean_over_call_sites_decides():
    a = {"generate_candidate": site(gold=0.9), "filter_column": site(agreement=0.5)}   # 0.70
    b = {"generate_candidate": site(gold=0.6), "filter_column": site(agreement=0.9)}   # 0.75
    choice, table = j6.choose({"a": a, "b": b}, {"a": 8.0, "b": 16.0}, 0.0)
    assert choice == "b" and table["decided_by"] == "score" and table["score"]["a"] == pytest.approx(0.7)


def test_a_tie_goes_to_the_smaller_footprint_and_never_silently():
    same = {"filter_column": site(agreement=0.8)}
    close = {"filter_column": site(agreement=0.79)}
    assert j6.choose({"big": same, "small": close}, {"big": 16.0, "small": 8.0}, 0.02)[0] == "small"
    assert j6.choose({"big": same, "small": close}, {"big": 16.0, "small": 8.0}, 0.0)[0] == "big"
    with pytest.raises(JudgmentError, match="footprint_gb is not recorded"):
        j6.choose({"big": same, "small": same}, {"big": 16.0, "small": None}, 0.0)
    with pytest.raises(JudgmentError, match="does not decide"):
        j6.choose({"big": same, "small": same}, {"big": 8.0, "small": 8.0}, 0.0)


def test_call_sites_without_a_score_are_left_out_for_every_candidate():
    a = {"filter_column": site(agreement=0.9), "revise": site()}
    b = {"filter_column": site(agreement=0.8), "revise": site(gold=1.0)}
    choice, table = j6.choose({"a": a, "b": b}, {}, 0.0)
    assert table["call_sites"] == ["filter_column"] and choice == "a"


def zeroshot_world(tmp_path, monkeypatch, overrides=None):
    config_path, config = repo(tmp_path, monkeypatch, overrides)
    t = teacher("agent-B0-calib", "calib", QUESTIONS)
    write_run("agent-B0-calib", {"type": "agent", "arm": "B0", "split": "calib"}, t)
    per_call_eval("eval-t", "agent-B0-calib", t, gold_correct("t", 0.9))
    runs = {}
    for name, quality in (("qwen3-8b", 0.9), ("granite-4.2-8b", 0.5)):
        run_id = f"replay-{name}"
        r = replay(t, run_id, f"slm:{name}", quality)
        write_run(run_id, {"type": "replay", "source_run_id": "agent-B0-calib", "engine": f"slm:{name}", "split": "calib"}, r)
        runs[run_id] = per_call_eval(f"eval-{name}", run_id, r, gold_correct(name, quality))
    return config, runs


def test_run_writes_the_choice_fact(tmp_path, monkeypatch):
    config, runs = zeroshot_world(tmp_path, monkeypatch)
    path, fact = j6.run(runs, "eval-t", config)
    result = read_result(path, "J6")["result"]
    assert result["choice"] == "qwen3-8b" and result["score"]["qwen3-8b"] > result["score"]["granite-4.2-8b"]
    assert facts.read_fact(str(fact), "choice")[0] == {"slm": "qwen3-8b"}
    assert result["choice_fact"]["sha256"] == fact.parent.name


def test_run_refuses_what_is_not_a_zero_shot_on_calib(tmp_path, monkeypatch):
    config, runs = zeroshot_world(tmp_path, monkeypatch)
    write_run("replay-lora", {"type": "replay", "source_run_id": "agent-B0-calib", "engine": "slm:qwen3-8b+lora:c0",
                              "split": "calib"}, [])
    with pytest.raises(JudgmentError, match="not the base"):
        j6.run({**runs, "replay-lora": "eval-t"}, "eval-t", config)
    unknown = {k: v for k, v in runs.items()}
    config["roles"]["slm_candidates"] = config["roles"]["slm_candidates"][:1]
    with pytest.raises(JudgmentError, match="not the base"):
        j6.run(unknown, "eval-t", config)


def test_s4_compares_exactly_the_configured_candidates_on_complete_replays(tmp_path, monkeypatch):
    config, runs = zeroshot_world(tmp_path, monkeypatch)
    one = dict(list(runs.items())[:1])
    with pytest.raises(JudgmentError, match="exactly the configured candidates"):
        j6.run(one, "eval-t", config)
    manifest = paths.RUNS / "replay-granite-4.2-8b" / "manifest.json"
    manifest.write_text(__import__("json").dumps({**__import__("json").loads(manifest.read_text()),
                                                  "call_sites": sorted(set(GOLD_OR_ALL))}))
    with pytest.raises(JudgmentError, match="complete zero-shot replay"):
        j6.run(runs, "eval-t", config)


GOLD_OR_ALL = ["agent_ir", "extract_keywords", "filter_column", "select_tables", "generate_candidate", "revise"]
