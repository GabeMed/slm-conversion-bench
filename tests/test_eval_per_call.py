"""`bench eval <run_id> --per-call` (design §5.1): the SQL of every generate_candidate and revise
invocation in a run's calls.jsonl, from the last attempt that parsed, scored against the gold of its
question, under the same checks as the end-to-end eval."""
import json
import uuid

import pytest
import yaml

from bench import paths
from bench.contracts.calls import validate_calls
from bench.contracts.config import config_sha256, load_config
from bench.data import DataError
from bench.evaluate import evaluate_per_call
from synthetic import GOLD, make_repo, sha256


GOLDS = {1: GOLD,                                                    # 1 row: 1
         2: "SELECT count(*) FROM gas_t WHERE country = 'CZE'",   # 2
         3: "SELECT count(*) FROM gas_t"}                          # 3


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """The synthetic repository, with a different gold per train question, so pairing shows."""
    root, config_path, _ = make_repo(tmp_path, monkeypatch)
    dev_path = paths.RAW / "bird_dev_questions.json"
    dev_path.write_text(json.dumps([{**q, "SQL": GOLDS[q["question_id"]]} for q in json.loads(dev_path.read_text())]))
    raw = yaml.safe_load(config_path.read_text())
    raw["data"]["bird_dev_questions"]["sha256"] = sha256(dev_path)
    config_path.write_text(yaml.safe_dump(raw))
    return root, config_path, load_config(config_path)


def call(question_id, call_site, invocation_key, parsed_output, attempt=1, retry_of=None):
    ok = parsed_output is not None
    return {"run_id": "replay-x", "call_id": str(uuid.uuid4()), "retry_of": retry_of, "attempt": attempt,
            "question_id": question_id, "call_site": call_site, "invocation_key": invocation_key,
            "cluster": None, "engine": "slm:qwen3-8b", "model_role": "slm", "model": "qwen3-8b", "endpoint": "fake",
            "prompt_messages": [{"role": "user", "content": "..."}], "response_text": "...",
            "parsed_output": parsed_output, "parsed_ok": ok, "latency_ms": 1, "temperature": 0.0,
            "usage": {"input": 1, "cached_input": 0, "output": 1, "source": "api"},
            "started_at": "2026-09-30T12:00:00+00:00", "error": None if ok else "unparsable"}


def calls_of_a_replay():
    failed = call("1", "generate_candidate", "generate_candidate_one:0", None)
    return [
        failed,
        call("1", "generate_candidate", "generate_candidate_one:0", {"SQL": GOLD, "plan": ""}, 2, failed["call_id"]),
        call("1", "revise", "revise_1:0", {"refined_sql_query": "SELECT 0"}),
        call("2", "agent_cg", "cg:0", {"tool": "generate_candidate"}),
        call("2", "generate_candidate", "generate_candidate_one:0", None),
        call("3", "select_tables", "single", {"table_names": ["gas_t"]}),
        call("3", "generate_candidate", "generate_candidate_one:0", {"SQL": "SELECT count(id) FROM gas_t"}),
    ]


def replay_run(repo, calls, status="done", split="train", question_ids=("1", "2", "3")):
    _, _, config = repo
    assert validate_calls(calls) == [] or status != "done"
    run_dir = paths.RUNS / "replay-x"
    run_dir.mkdir(parents=True)
    (run_dir / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "replay-x", "type": "replay", "engine": "slm:qwen3-8b", "split": split, "status": status,
        "commit": "c", "config_sha256": config_sha256(config), "question_ids": list(question_ids)}))
    return "replay-x"


def results_of(out):
    return [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]


def test_every_generation_and_repair_invocation_is_scored_against_its_gold(repo):
    out = evaluate_per_call(replay_run(repo, calls_of_a_replay()))
    rows = results_of(out)
    assert [(r["question_id"], r["call_site"], r["invocation_key"], r["correct"]) for r in rows] == [
        ("1", "generate_candidate", "generate_candidate_one:0", True),   # the retry that parsed
        ("1", "revise", "revise_1:0", False),
        ("2", "generate_candidate", "generate_candidate_one:0", False),  # no attempt parsed
        ("3", "generate_candidate", "generate_candidate_one:0", True)]
    assert rows[0]["attempt"] == 2 and rows[2]["pred_error"] == "no attempt parsed"
    assert all(r["gold_sql"] == GOLDS[int(r["question_id"])] and r["difficulty"] == "simple" for r in rows)
    manifest = json.loads((out / "manifest.json").read_text())
    assert (manifest["type"], manifest["per_call"], manifest["source_run_id"], manifest["n"], manifest["correct"]) == \
        ("eval", True, "replay-x", 4, 2)


def test_per_call_refuses_what_the_end_to_end_eval_refuses(repo):
    run_id = replay_run(repo, calls_of_a_replay(), status="interrupted")
    with pytest.raises(DataError, match="not 'done'"):
        evaluate_per_call(run_id)


def test_per_call_refuses_calls_outside_the_split_and_invalid_calls(repo):
    with pytest.raises(DataError, match="without a gold"):
        evaluate_per_call(replay_run(repo, calls_of_a_replay() + [call("9", "revise", "revise_1:0", {"refined_sql_query": GOLD})],
                                     question_ids=("1", "2", "3", "9")))
    manifest_path = paths.RUNS / "replay-x" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "question_ids": ["1", "2"]}))
    with pytest.raises(DataError, match="outside its question ids"):
        evaluate_per_call("replay-x")
    manifest_path.write_text(json.dumps({k: v for k, v in manifest.items() if k != "question_ids"}))
    with pytest.raises(DataError, match="question ids"):
        evaluate_per_call("replay-x")
    manifest_path.write_text(json.dumps({**manifest, "question_ids": ["1", "2", "3", "9"]}))
    broken = calls_of_a_replay()
    broken[1]["retry_of"] = None  # a second attempt that points nowhere
    (paths.RUNS / "replay-x" / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in broken))
    with pytest.raises(DataError, match="C1"):
        evaluate_per_call("replay-x")


def test_the_command_takes_per_call(repo, capsys):
    from bench.cli import main
    run_id = replay_run(repo, calls_of_a_replay())
    assert main(["eval", run_id, "--per-call"]) == 0
    out = capsys.readouterr().out.strip()
    assert "eval-per-call-replay-x-" in out and json.loads((paths.RUNS / out.rsplit("/", 1)[1] / "manifest.json").read_text())["per_call"]
