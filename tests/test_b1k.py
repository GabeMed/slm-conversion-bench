"""b1k · B1's few-shot k, chosen on the pilot: 3 only if it beats 0 on the gold call sites by the
paired bootstrap's lower bound, 0 otherwise; read from two pilot replays on cheap_alt."""
import hashlib
import json

import pytest

from bench import cli, data
from bench.judge import b1k
from bench.judge.base import JudgmentError
from bench.judge.j4 import noninferiority
from fixtures.fake import repo, write_run
from fixtures.world import gold_correct, per_call_eval, replay, teacher

CALIB = [str(q) for q in range(1000, 1080)]
PILOT = CALIB[:50]


def invocation(q):
    return (str(q), "generate_candidate", "generate_candidate_one:0")


def correct(right):
    """Per-call correctness of invocations 100..199: right for the question numbers in `right`."""
    return {invocation(q): q in right for q in range(100, 200)}


def test_k3_is_chosen_only_when_its_lower_bound_is_above_zero():
    half, most = set(range(100, 150)), set(range(100, 190))
    better = b1k.choose(correct(half), correct(most), seed=1, resamples=2000)
    assert (better["k"], better["n"], better["ex_k0"], better["ex_k3"]) == (3, 100, 0.5, 0.9)
    assert better["diff"] == pytest.approx(0.4) and 0 < better["ci_low"] < 0.4
    # a tie: the bound is exactly 0, which is not above 0
    tie = b1k.choose(correct(half), correct(half), seed=1, resamples=2000)
    assert (tie["k"], tie["diff"], tie["ci_low"]) == (0, 0.0, 0.0)
    # ahead, but within the noise: 3 invocations gained, 2 lost
    noise = b1k.choose(correct(half), correct((half - {100, 101}) | {150, 151, 152}), seed=1, resamples=2000)
    assert noise["k"] == 0 and noise["diff"] == pytest.approx(0.01) and noise["ci_low"] < 0
    worse = b1k.choose(correct(most), correct(half), seed=1, resamples=2000)
    assert worse["k"] == 0 and worse["ci_low"] < 0
    # every invocation gained: the bound is the difference itself
    assert b1k.choose(correct(set()), correct(set(range(100, 200))), seed=1, resamples=200)["ci_low"] == 1.0


def test_the_bound_is_j4s_paired_bootstrap():
    k0, k3 = correct(set(range(100, 150))), correct(set(range(120, 185)))
    mine = b1k.choose(k0, k3, seed=7, resamples=1000)
    j4 = noninferiority({i[0]: ok for i, ok in k3.items()}, {i[0]: ok for i, ok in k0.items()}, 5, 7, 1000)
    assert (mine["diff"], mine["ci_low"]) == (j4["diff"], j4["ci_low"])  # one gold call per question: the same bootstrap


def test_the_question_is_the_bootstrap_unit():
    """Ten questions with ten gold calls each: k = 3 gains every call of six questions and loses every
    call of four. Resampled call by call, 100 independent pairs would put the bound above 0; a
    question's calls move together, so there are ten units, and the bound is below 0."""
    k0 = {(str(q), "revise", f"revise_{r}:0"): q >= 6 for q in range(10) for r in range(10)}
    k3 = {identity: not ok for identity, ok in k0.items()}
    result = b1k.choose(k0, k3, seed=3, resamples=2000)
    assert (result["n"], result["n_questions"], result["ex_k0"], result["ex_k3"]) == (100, 10, 0.4, 0.6)
    assert result["diff"] == pytest.approx(0.2) and result["ci_low"] < 0 and result["k"] == 0
    # questions with more calls weigh more: EX is per call, not per question
    uneven = {("1", "revise", f"revise_{r}:0"): False for r in range(3)} | {("2", "generate_candidate", "g:0"): True}
    gained = b1k.choose(uneven, {identity: True for identity in uneven}, seed=3, resamples=200)
    assert (gained["ex_k0"], gained["diff"]) == (0.25, 0.75)


def test_the_two_replays_must_answer_the_same_invocations():
    k0 = correct(set())
    with pytest.raises(JudgmentError, match="1 are in one only"):
        b1k.choose(k0, {i: ok for i, ok in k0.items() if i != invocation(100)}, seed=1, resamples=10)
    with pytest.raises(JudgmentError, match="at least one"):
        b1k.choose({}, {}, seed=1, resamples=10)


def world(tmp_path, monkeypatch, quality_k0, quality_k3):
    """The teacher on calib, and its pilot questions replayed on cheap_alt with k = 0 and k = 3, each
    with its per-call evaluation: a gold call is right with the given probability."""
    config_path, config = repo(tmp_path, monkeypatch, {"stats": {"n_boot": 500}})
    monkeypatch.setattr(data, "pilot_ids", lambda config: list(PILOT), raising=False)  # D's accessor (design §6.2)
    t = teacher("agent-B0-calib", "calib", CALIB)
    write_run("agent-B0-calib", {"type": "agent", "arm": "B0", "split": "calib", "question_ids": CALIB}, t)
    per_call_eval("eval-t", "agent-B0-calib", t, gold_correct("t", 0.9))
    on_pilot = [c for c in t if c["question_id"] in PILOT]
    for name, k, quality in (("k0", 0, quality_k0), ("k3", 3, quality_k3)):
        calls = replay(on_pilot, f"replay-{name}", "cheap_alt", 0.9)
        write_run(f"replay-{name}", {"type": "replay", "source_run_id": "agent-B0-calib", "engine": "cheap_alt",
                                     "split": "calib", "question_ids": PILOT, "few_shot_k": k}, calls)
        per_call_eval(f"eval-{name}", f"replay-{name}", calls, gold_correct("b1", quality))
    return config_path, config


def test_run_chooses_k3_when_the_examples_earn_their_tokens(tmp_path, monkeypatch):
    _, config = world(tmp_path, monkeypatch, 0.4, 0.95)
    path = b1k.run(("replay-k0", "eval-k0"), ("replay-k3", "eval-k3"), "eval-t", config)
    assert path.name == "choice.json" and path.parent.parent.name == "b1k"
    assert path.parent.name == hashlib.sha256(path.read_bytes()).hexdigest()  # stored by content
    payload = json.loads(path.read_bytes())
    result = payload["result"]
    n = sum(1 for q in PILOT for _ in (["generate"] + (["revise"] if int(q) % 3 == 0 else [])))  # the gold calls of the pilot
    assert (result["k"], result["n"], result["n_questions"]) == (3, n, len(PILOT))
    assert result["ex_k3"] > result["ex_k0"] and result["ci_low"] > 0
    assert result["pilot_ids"] == PILOT and (result["seed"], result["n_boot"]) == (config["seeds"]["bootstrap"], 500)
    assert 0 < result["ex_teacher"] <= 1
    assert payload["judgment"] == "b1k" and payload["reads"]["config"] == ["seeds.bootstrap", "stats.n_boot"]
    assert {name: payload["reads"][name]["replay"]["run_id"] for name in ("k0", "k3")} == {"k0": "replay-k0", "k3": "replay-k3"}
    assert payload["reads"]["k0"]["teacher"]["run_id"] == "agent-B0-calib"


def test_run_chooses_k0_when_k3_does_no_better(tmp_path, monkeypatch):
    _, config = world(tmp_path, monkeypatch, 0.7, 0.7)  # the same gold calls right in both
    result = json.loads(b1k.run(("replay-k0", "eval-k0"), ("replay-k3", "eval-k3"), "eval-t", config).read_bytes())["result"]
    assert (result["k"], result["diff"], result["ci_low"]) == (0, 0.0, 0.0) and result["ex_k0"] == result["ex_k3"]


def test_run_refuses_replays_that_are_not_the_pilot_on_cheap_alt_with_that_k(tmp_path, monkeypatch):
    _, config = world(tmp_path, monkeypatch, 0.4, 0.95)
    with pytest.raises(JudgmentError, match="replay-k3 ran with few_shot_k 3, not 0"):  # the two swapped
        b1k.run(("replay-k3", "eval-k3"), ("replay-k0", "eval-k0"), "eval-t", config)
    monkeypatch.setattr(data, "pilot_ids", lambda config: PILOT[:40])
    with pytest.raises(JudgmentError, match="exactly the pilot questions"):
        b1k.run(("replay-k0", "eval-k0"), ("replay-k3", "eval-k3"), "eval-t", config)
    monkeypatch.setattr(data, "pilot_ids", lambda config: list(PILOT))
    slm = replay([c for c in teacher("agent-B0-calib", "calib", PILOT)], "replay-slm", "slm:qwen3-8b", 0.9)
    write_run("replay-slm", {"type": "replay", "source_run_id": "agent-B0-calib", "engine": "slm:qwen3-8b", "split": "calib",
                             "question_ids": PILOT, "few_shot_k": None}, slm)
    per_call_eval("eval-slm", "replay-slm", slm, gold_correct("b1", 0.5))
    with pytest.raises(JudgmentError, match="not a replay on cheap_alt"):
        b1k.run(("replay-slm", "eval-slm"), ("replay-k3", "eval-k3"), "eval-t", config)
    with pytest.raises(JudgmentError, match="evaluated replay-k3, not replay-k0"):
        b1k.run(("replay-k0", "eval-k3"), ("replay-k3", "eval-k3"), "eval-t", config)


def test_the_command_prints_the_choice_and_reports_a_refusal(tmp_path, monkeypatch, capsys):
    config_path, _ = world(tmp_path, monkeypatch, 0.4, 0.95)
    judge = ["judge", "b1k", "--teacher-eval", "eval-t", "--config", str(config_path)]
    assert cli.main(judge + ["--k0", "replay-k0=eval-k0", "--k3", "replay-k3=eval-k3"]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed.endswith("choice.json") and "/judgments/b1k/" in printed
    assert json.loads(open(printed).read())["result"]["k"] == 3
    assert cli.main(judge + ["--k0", "replay-k3=eval-k3", "--k3", "replay-k0=eval-k0"]) == 2
    assert "few_shot_k" in capsys.readouterr().err
    assert cli.main(judge + ["--k0", "replay-k0", "--k3", "replay-k3=eval-k3"]) == 2
    assert "expected RUN=EVAL" in capsys.readouterr().err
