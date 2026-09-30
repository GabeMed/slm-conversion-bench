"""The `embed` execution: prompt and prompt+action vectors per invocation, token counts and
truncation, never the call site."""
import json

import pytest

np = pytest.importorskip("numpy")

from bench import paths  # noqa: E402
from bench.contracts import clusters  # noqa: E402
from bench.embed import run_embed, texts_of  # noqa: E402
from bench.judge.base import JudgmentError, read_jsonl  # noqa: E402
from fixtures.fake import call, curate_run, example, fake_embed, fake_tokens, repo, write_run  # noqa: E402

SMALL = {"clustering": {"embedding": {"max_seq_length": 60}}}


def test_embeds_curated_examples_prompt_and_prompt_with_action(tmp_path, monkeypatch):
    config_path, config = repo(tmp_path, monkeypatch, SMALL)
    examples = [example("1", "select_tables"), example("2", "filter_column", "t.c")]
    curate_run("curate-x", examples)
    out = run_embed("curate-x", str(config_path), embed_fn=fake_embed, count_tokens=fake_tokens)
    manifest = json.loads((out / "manifest.json").read_text())
    index = read_jsonl(out / "index.jsonl")
    prompts, actions = np.load(out / "prompt.npy"), np.load(out / "prompt_action.npy")
    assert [r["call_id"] for r in index] == [e["call_id"] for e in examples]
    assert all("call_site" not in r for r in index)  # S3 never sees the call site
    first = clusters.prompt_text(examples[0]["prompt"])
    assert np.allclose(prompts[0], fake_embed([first], None)[0])
    with_action = clusters.prompt_text(examples[0]["prompt"] + examples[0]["completion"])
    assert np.allclose(actions[index[0]["action_row"]], fake_embed([with_action], None)[0])
    assert manifest["embedding"] == config["clustering"]["embedding"] and manifest["source"]["run_id"] == "curate-x"
    assert index[0]["prompt_tokens"] == len(first.split())
    # every template is longer than 60 words here: all cut, and reported so
    assert manifest["tokens"]["prompt"]["truncated_fraction"] == 1.0 and manifest["tokens"]["prompt"]["n"] == 2


def test_counts_only_the_texts_longer_than_the_limit(tmp_path, monkeypatch):
    config_path, _ = repo(tmp_path, monkeypatch, {"clustering": {"embedding": {"max_seq_length": 100}}})
    short = {**example("1", "select_tables"), "prompt": [{"role": "user", "content": "a short prompt"}]}
    curate_run("curate-y", [short, example("2", "filter_column"), example("3", "revise")])
    out = run_embed("curate-y", str(config_path), embed_fn=fake_embed, count_tokens=fake_tokens)
    tokens = json.loads((out / "manifest.json").read_text())["tokens"]["prompt"]
    assert (tokens["n"], tokens["truncated"]) == (3, 2) and tokens["truncated_fraction"] == pytest.approx(2 / 3)


def test_embeds_an_agent_run_by_invocation_first_prompt_and_last_parsed_action(tmp_path, monkeypatch):
    config_path, _ = repo(tmp_path, monkeypatch)
    failed = call("agent-c", "5", "select_tables", response="x", parsed_ok=False)
    retry = call("agent-c", "5", "select_tables", response="{}", parsed={}, attempt=2, retry_of=failed["call_id"])
    never = call("agent-c", "6", "select_tables", response="?", parsed_ok=False)
    write_run("agent-c", {"type": "agent", "arm": "B0", "split": "calib"}, [failed, retry, never])
    rows = texts_of("agent-c")
    assert [r["call_id"] for r in rows] == [failed["call_id"], never["call_id"]]
    assert rows[0]["action"].endswith("assistant: {}") and rows[1]["action"] is None
    out = run_embed("agent-c", str(config_path), embed_fn=fake_embed, count_tokens=fake_tokens)
    index = read_jsonl(out / "index.jsonl")
    assert [r["action_row"] for r in index] == [0, None]
    assert np.load(out / "prompt_action.npy").shape == (1, 64)


def test_refuses_an_unfinished_source(tmp_path, monkeypatch):
    config_path, _ = repo(tmp_path, monkeypatch)
    write_run("agent-f", {"type": "agent", "arm": "B0", "split": "calib", "status": "failed"}, [])
    with pytest.raises(JudgmentError, match="not 'done'"):
        run_embed("agent-f", str(config_path), embed_fn=fake_embed, count_tokens=fake_tokens)
    assert not any(p.name.startswith("embed-") for p in paths.RUNS.iterdir())
