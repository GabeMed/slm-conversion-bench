"""b1k · B1's few-shot k, chosen on the pilot: 3 only if it beats 0 end to end by the paired
bootstrap's lower bound, 0 otherwise; read from B1's two agent executions on the pilot and their
end-to-end evaluations."""
import hashlib
import json

import pytest

from bench import cli, data, verify
from bench.judge import b1k
from bench.judge.base import JudgmentError
from bench.judge.j4 import noninferiority
from fixtures.fake import call, few_shot, repo, write_run

CALIB = [str(q) for q in range(1000, 1080)]
PILOT = CALIB[:50]


def correct(right, questions=range(100, 200)):
    """End-to-end correctness of questions 100..199: right for the question numbers in `right`."""
    return {str(q): q in right for q in questions}


def test_k3_is_chosen_only_when_its_lower_bound_is_above_zero():
    half, most = set(range(100, 150)), set(range(100, 190))
    better = b1k.choose(correct(half), correct(most), seed=1, resamples=2000)
    assert (better["k"], better["n"], better["ex_k0"], better["ex_k3"]) == (3, 100, 0.5, 0.9)
    assert better["diff"] == pytest.approx(0.4) and 0 < better["ci_low"] < 0.4
    assert (better["k3_only"], better["k0_only"]) == (40, 0) and better["ids"] == [str(q) for q in range(100, 200)]
    # a tie: the bound is exactly 0, which is not above 0
    tie = b1k.choose(correct(half), correct(half), seed=1, resamples=2000)
    assert (tie["k"], tie["diff"], tie["ci_low"]) == (0, 0.0, 0.0)
    # ahead, but within the noise: 3 questions gained, 2 lost
    noise = b1k.choose(correct(half), correct((half - {100, 101}) | {150, 151, 152}), seed=1, resamples=2000)
    assert noise["k"] == 0 and noise["diff"] == pytest.approx(0.01) and noise["ci_low"] < 0
    worse = b1k.choose(correct(most), correct(half), seed=1, resamples=2000)
    assert worse["k"] == 0 and worse["ci_low"] < 0
    # every question gained: the bound is the difference itself
    assert b1k.choose(correct(set()), correct(set(range(100, 200))), seed=1, resamples=200)["ci_low"] == 1.0


def test_the_bound_is_j4s_paired_bootstrap():
    k0, k3 = correct(set(range(100, 150))), correct(set(range(120, 185)))
    mine = b1k.choose(k0, k3, seed=7, resamples=1000)
    j4 = noninferiority(k3, k0, 7, 1000, margin=0.05)
    assert (mine["diff"], mine["ci_low"], mine["n"]) == (j4["diff"], j4["ci_low"], j4["n"])


def test_the_two_executions_must_be_evaluated_on_the_same_questions():
    k0 = correct(set())
    with pytest.raises(JudgmentError, match="1 are in one only"):
        b1k.choose(k0, {q: ok for q, ok in k0.items() if q != "100"}, seed=1, resamples=10)
    with pytest.raises(JudgmentError, match="at least one"):
        b1k.choose({}, {}, seed=1, resamples=10)


def agent_run(run_id, k, right, *, arm="B1", split="calib", questions=PILOT, engine="cheap_alt", status="done",
              no_sql=(), sampling="today"):
    """B1's agent execution on `questions` with k examples, and its end-to-end eval: right on `right`,
    no SQL on `no_sql`."""
    calls = [call(run_id, q, "generate_candidate", "generate_candidate_one:0", parsed={"SQL": "SELECT 1"},
                  role="cheap_alt" if engine == "cheap_alt" else "slm", engine=engine, model="engine-model")
             for q in questions]
    write_run(run_id, {"type": "agent", "arm": arm, "split": split, "question_ids": list(questions),
                       **few_shot(k, sampling), "status": status}, calls)
    rows = [{"question_id": q, "predicted_sql": None if q in no_sql else "SELECT 1", "correct": q in right}
            for q in questions]
    write_run(f"eval-{run_id}", {"type": "eval", "source_run_id": run_id, "status": None}, files={"results.jsonl": rows})
    return run_id, f"eval-{run_id}"


def world(tmp_path, monkeypatch, right_k0, right_k3):
    """B1 on the pilot with k = 0 and k = 3, right end to end on the given questions. The arms' facts
    are those before RUNBOOK steps 4 and 8 (test_verify.NO_CENTROIDS): this world holds no J6 and no J5."""
    config_path, config = repo(tmp_path, monkeypatch, {"stats": {"n_boot": 500},
                                                       "arms": {"B3": {"choice": None},
                                                                "B4": {"choice": None, "centroids": None},
                                                                "B5": {"choice": None, "centroids": None}}})
    monkeypatch.setattr(data, "pilot_ids", lambda config: list(PILOT))  # D's accessor (design §6.2)
    k0 = agent_run("agent-k0", 0, right_k0, no_sql=PILOT[40:])
    k3 = agent_run("agent-k3", 3, right_k3)
    return config_path, config, k0, k3


def test_run_chooses_k3_when_the_examples_earn_their_tokens(tmp_path, monkeypatch):
    _, config, k0, k3 = world(tmp_path, monkeypatch, set(PILOT[:12]), set(PILOT[:30]))
    path = b1k.run(k0, k3, config)
    assert path.name == "choice.json" and path.parent.parent.name == "b1k"
    assert path.parent.name == hashlib.sha256(path.read_bytes()).hexdigest()  # stored by content
    payload = json.loads(path.read_bytes())
    result = payload["result"]
    assert (result["k"], result["n"], result["ex_k0"], result["ex_k3"]) == (3, 50, 0.24, 0.6)
    assert result["diff"] == pytest.approx(0.36) and 0 < result["ci_low"] < 0.36
    assert result["without_sql"] == {"k0": 10, "k3": 0}
    assert result["pilot_ids"] == PILOT == result["ids"] and (result["seed"], result["n_boot"]) == (config["seeds"]["bootstrap"], 500)
    assert payload["judgment"] == "b1k" and payload["reads"]["config"] == list(b1k.CONFIG_KEYS)
    assert {name: (payload["reads"][name]["run"]["run_id"], payload["reads"][name]["eval"]["run_id"]) for name in ("k0", "k3")} \
        == {"k0": k0, "k3": k3}
    assert b1k.run(k0, k3, config) == path  # the same inputs, the same bytes


def test_run_chooses_k0_when_k3_does_no_better(tmp_path, monkeypatch):
    _, config, k0, k3 = world(tmp_path, monkeypatch, set(PILOT[:25]), set(PILOT[:25]))
    result = json.loads(b1k.run(k0, k3, config).read_bytes())["result"]
    assert (result["k"], result["diff"], result["ci_low"]) == (0, 0.0, 0.0) and result["ex_k0"] == result["ex_k3"]
    _, config, k0, k3 = world(tmp_path / "noise", monkeypatch, set(PILOT[:25]), set(PILOT[:27]))
    result = json.loads(b1k.run(k0, k3, config).read_bytes())["result"]
    assert result["k"] == 0 and result["diff"] == pytest.approx(0.04) and result["ci_low"] <= 0


def test_run_refuses_executions_that_are_not_b1_on_the_pilot_on_cheap_alt_with_that_k(tmp_path, monkeypatch):
    _, config, k0, k3 = world(tmp_path, monkeypatch, set(PILOT[:12]), set(PILOT[:30]))
    with pytest.raises(JudgmentError, match="agent-k3 ran with few_shot_k 3, not 0"):  # the two swapped
        b1k.run(k3, k0, config)
    refusals = {
        "arm is 'B0'": agent_run("agent-b0", 0, set(), arm="B0"),
        "split is 'train'": agent_run("agent-train", 0, set(), split="train"),
        "not 'done'": agent_run("agent-failed", 0, set(), status="failed"),
        "exactly the pilot questions": agent_run("agent-short", 0, set(), questions=PILOT[:40]),
        "not an execution on cheap_alt": agent_run("agent-slm", 0, set(), engine="slm:qwen3-8b"),
    }
    for reason, inputs in refusals.items():
        with pytest.raises(JudgmentError, match=reason):
            b1k.run(inputs, k3, config)
    with pytest.raises(JudgmentError, match="evaluated agent-k3, not agent-k0"):
        b1k.run(("agent-k0", "eval-agent-k3"), k3, config)
    with pytest.raises(JudgmentError, match="type is 'agent'"):  # an execution in the eval's place
        b1k.run(("agent-k0", "agent-k3"), k3, config)
    write_run("eval-per-call", {"type": "eval", "source_run_id": "agent-k0", "per_call": True, "status": None},
              files={"results.jsonl": []})
    with pytest.raises(JudgmentError, match="per-call evaluation"):
        b1k.run(("agent-k0", "eval-per-call"), k3, config)
    write_run("eval-partial", {"type": "eval", "source_run_id": "agent-k0", "status": None},
              files={"results.jsonl": [{"question_id": q, "correct": False} for q in PILOT[:49]]})
    with pytest.raises(JudgmentError, match="each pilot question exactly once"):
        b1k.run(("agent-k0", "eval-partial"), k3, config)
    monkeypatch.setattr(data, "pilot_ids", lambda config: PILOT[:40])
    with pytest.raises(JudgmentError, match="exactly the pilot questions"):
        b1k.run(k0, k3, config)


def test_run_refuses_a_k3_execution_drawn_by_another_sampling_rule(tmp_path, monkeypatch):
    """A k = 3 execution made before the per-step sampling rule (its manifest records no rule, or another)
    is another B1: choosing on it would write a second, wrong choice. A k = 0 execution has no examples
    to draw, so no rule; it must have the empty prefix."""
    _, config, k0, k3 = world(tmp_path, monkeypatch, set(PILOT[:12]), set(PILOT[:30]))
    for run_id, sampling in (("agent-k3-old", None), ("agent-k3-other", "one prefix per call site")):
        old = agent_run(run_id, 3, set(PILOT[:30]), sampling=sampling)
        with pytest.raises(JudgmentError, match=f"{run_id} drew its 3 few-shot examples by another sampling rule"):
            b1k.run(k0, old, config)
    path = tmp_path / "runs" / "agent-k0" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["few_shot_sha256"] = dict(manifest["few_shot_sha256"], agent_ir="0" * 64)
    path.write_text(json.dumps(manifest))
    with pytest.raises(JudgmentError, match="agent-k0 ran with few_shot_k 0 but a non-empty prefix"):
        b1k.run(k0, k3, config)
    assert not (tmp_path / "judgments" / "b1k").exists()


def test_bench_verify_recomputes_the_choice(tmp_path, monkeypatch):
    _, config, k0, k3 = world(tmp_path, monkeypatch, set(PILOT[:12]), set(PILOT[:30]))
    path = b1k.run(k0, k3, config)
    plan = {"b1k": str(path.relative_to(tmp_path))}
    assert verify.verify(config, plan) == ([plan["b1k"]], [])
    other = dict(config, seeds=dict(config["seeds"], bootstrap=config["seeds"]["bootstrap"] + 1))
    _, divergences = verify.verify(other, plan)
    assert len(divergences) == 1 and "seeds.bootstrap" in divergences[0]
    assert sorted(p.parent.name for p in (tmp_path / "judgments" / "b1k").glob("*/choice.json")) == [path.parent.name]


def test_the_command_prints_the_choice_and_reports_a_refusal(tmp_path, monkeypatch, capsys):
    config_path, _, _, _ = world(tmp_path, monkeypatch, set(PILOT[:12]), set(PILOT[:30]))
    judge = ["judge", "b1k", "--config", str(config_path)]
    assert cli.main(judge + ["--k0", "agent-k0=eval-agent-k0", "--k3", "agent-k3=eval-agent-k3"]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed.endswith("choice.json") and "/judgments/b1k/" in printed
    assert json.loads(open(printed).read())["result"]["k"] == 3
    assert cli.main(judge + ["--k0", "agent-k3=eval-agent-k3", "--k3", "agent-k0=eval-agent-k0"]) == 2
    assert "few_shot_k" in capsys.readouterr().err
    assert cli.main(judge + ["--k0", "agent-k0", "--k3", "agent-k3=eval-agent-k3"]) == 2
    assert "expected RUN=EVAL" in capsys.readouterr().err
