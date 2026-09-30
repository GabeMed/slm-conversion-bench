"""`bench replay`: the first attempt of every invocation of a B0 run, resent to another engine with
the same identity, parser and retry policy; C1 and a manifest only. Agent environment only."""
import json

import pytest
import yaml

pytest.importorskip("langchain_core")
import httpx  # noqa: E402
import openai  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from bench import paths  # noqa: E402
from bench.agent import hooks, runner  # noqa: E402
from bench.agent.replay import replay  # noqa: E402
from bench.barrier import TestSplitLocked  # noqa: E402
from bench.contracts import clusters  # noqa: E402
from bench.contracts.calls import CALL_SITES, read_calls, validate_calls  # noqa: E402
from bench.contracts.facts import write_fact  # noqa: E402
from test_agent_arms import Revising, adapters_for, by_column_filter, facts, rel, repo, set_config  # noqa: E402,F401
from test_agent_wiring import ScriptedChess  # noqa: E402


@pytest.fixture
def source(monkeypatch, repo):
    """A done B0 run on train where every call site is called, some twice (a parse retry)."""
    class RetryOnce(Revising):
        seen = set()

        def answer(self, text):
            if '"table_names"' in text and text not in self.seen:
                self.seen.add(text)
                return "the tables are gas_t"  # the first select_tables answer never parses
            return super().answer(text)
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: RetryOnce())
    run_dir = runner.run_agent(str(repo), "B0", "train", ids=["1", "2"])
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "done"
    return run_dir.name


def replayed(monkeypatch, repo, source, model=None, **kwargs):
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: model or Revising())
    run_dir = replay(str(repo), source, **kwargs)
    return run_dir, json.loads((run_dir / "manifest.json").read_text()), read_calls(run_dir / "calls.jsonl")


def identity(call):
    return call["question_id"], call["call_site"], call["invocation_key"]


def test_every_first_attempt_is_resent_with_its_identity(monkeypatch, repo, source):
    run_dir, manifest, calls = replayed(monkeypatch, repo, source, engine="slm:qwen3-8b")  # the zero-shot of S4
    teacher = read_calls(paths.RUNS / source / "calls.jsonl")
    firsts = {identity(c): c for c in teacher if c["attempt"] == 1}
    assert len(firsts) < len(teacher)  # the source has a retry, which is not resent
    assert {identity(c) for c in calls} == set(firsts) and all(c["attempt"] == 1 for c in calls)
    for call in calls:
        assert call["prompt_messages"] == firsts[identity(call)]["prompt_messages"]
        assert (call["engine"], call["model_role"], call["run_id"]) == ("slm:qwen3-8b", "slm", run_dir.name)
        assert call["parsed_ok"] and call["temperature"] == firsts[identity(call)]["temperature"]
    agent_calls = [c for c in calls if c["call_site"].startswith("agent_")]
    assert agent_calls and all(c["parsed_output"] == firsts[identity(c)]["parsed_output"] for c in agent_calls)
    assert validate_calls(calls) == [] and manifest["c1_errors"] == 0
    assert (manifest["type"], manifest["status"], manifest["source_run_id"]) == ("replay", "done", source)
    assert manifest["n_invocations"] == len(firsts) and manifest["call_sites_seen"] == sorted(CALL_SITES)
    assert sorted(p.name for p in run_dir.iterdir()) == ["calls.jsonl", "config.json", "manifest.json"]


def test_call_sites_filter_what_is_resent(monkeypatch, repo, source):
    _, manifest, calls = replayed(monkeypatch, repo, source, engine="production_llm",
                                  call_sites=["generate_candidate", "revise"])
    assert {c["call_site"] for c in calls} == {"generate_candidate", "revise"}
    assert manifest["call_sites"] == ["generate_candidate", "revise"] and manifest["status"] == "done"


def test_the_same_retry_policy_applies(monkeypatch, repo, source):
    class Flaky(Revising):
        def __init__(self):
            super().__init__()
            self.failed = set()

        def invoke(self, messages):
            text = messages[-1].content
            if "is_column_information_relevant" in text and text not in self.failed:
                self.failed.add(text)
                raise openai.RateLimitError("429", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None)
            if '"table_names"' in text:
                return AIMessage(content="not json", response_metadata={})
            return super().invoke(messages)
    monkeypatch.setattr(hooks, "_sleep", lambda s: None)
    _, manifest, calls = replayed(monkeypatch, repo, source, model=Flaky(), engine="production_llm")
    assert manifest["status"] == "done" and manifest["model_failures"] == {"select_tables": 2}  # parse retries exhausted
    filtered = [c for c in calls if c["call_site"] == "filter_column"]
    assert {c["attempt"] for c in filtered} == {1, 2} and all(c["parsed_ok"] for c in filtered if c["attempt"] == 2)
    tables = [c for c in calls if c["call_site"] == "select_tables"]
    assert sorted(c["attempt"] for c in tables) == [1, 1, 2, 2] and validate_calls(calls) == []


def test_a_harness_failure_stops_the_replay(monkeypatch, repo, source):
    class Down(Revising):
        def invoke(self, messages):
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
    monkeypatch.setattr(hooks, "_sleep", lambda s: None)
    _, manifest, calls = replayed(monkeypatch, repo, source, model=Down(), engine="production_llm")
    assert manifest["status"] == "failed" and list(manifest["harness_errors"]) == ["1"]
    assert manifest["model_failures"] == {}  # the engine being down is not the model's failure
    assert {c["question_id"] for c in calls} == {"1"} and manifest["n_questions_replayed"] == 0


def test_routed_by_an_arm_with_the_few_shot_on_cheap_alt(monkeypatch, repo, source):
    choice, centroids = facts(paths.ROOT)
    adapters = adapters_for(choice, centroids, ("c0", "c1"))
    allocation = write_fact("J7", "allocation", {"centroids": centroids.parent.name, "adapters": adapters.parent.name,
                                                 "allocation": {"c0": "slm", "c1": "cheap_alt"}})
    set_config(repo, B1={"few_shot": {"k": 1, "source_run": source}},
               B5={"choice": rel(choice), "centroids": rel(centroids), "adapters": rel(adapters), "allocation": rel(allocation)})
    monkeypatch.setattr(clusters, "embed", by_column_filter)
    _, manifest, calls = replayed(monkeypatch, repo, source, arm="B5")
    assert manifest["status"] == "done" and set(manifest["facts"]) == {"choice", "centroids", "adapters", "allocation"}
    assert manifest["few_shot_k"] == 1 and set(manifest["few_shot_sha256"]) == set(CALL_SITES)
    teacher = {identity(c): c for c in read_calls(paths.RUNS / source / "calls.jsonl") if c["attempt"] == 1}
    for call in calls:
        if call["engine"] == "cheap_alt":
            assert call["cluster"] == "c1" and call["prompt_messages"][2:] == teacher[identity(call)]["prompt_messages"]
        else:
            assert call["engine"] == "slm:qwen3-8b+lora:qwen3-8b-c0" and call["call_site"] == "filter_column"


def test_only_a_done_b0_run_is_replayed_and_the_barrier_holds(monkeypatch, repo, source):
    with pytest.raises(runner.data.DataError, match="one of the two"):
        replay(str(repo), source)
    other, _, _ = replayed(monkeypatch, repo, source, engine="production_llm")
    with pytest.raises(runner.data.DataError, match="done B0 agent run"):
        replay(str(repo), other.name, engine="production_llm")  # a replay of a replay
    with pytest.raises(runner.data.DataError, match="unknown call sites"):
        replay(str(repo), source, engine="production_llm", call_sites=["select_everything"])
    manifest_path = paths.RUNS / source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "split": "test"}))
    runs = set(paths.RUNS.iterdir())
    with pytest.raises(TestSplitLocked, match="origin/main|prereg"):  # the barrier itself, not the registry
        replay(str(repo), source, engine="production_llm")
    assert set(paths.RUNS.iterdir()) == runs


def test_a_replay_needs_no_retrieval_embeddings(monkeypatch, repo, source):
    config = yaml.safe_load(repo.read_text())
    config["embeddings"]["provider"] = "openai"  # the default; nothing is retrieved, so no key is needed
    repo.write_text(yaml.safe_dump(config))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _, manifest, calls = replayed(monkeypatch, repo, source, engine="slm:qwen3-8b")
    assert manifest["status"] == "done" and calls


@pytest.mark.parametrize("block,path,value", [
    ("call_sites", ("select_tables", "temperature"), 0.7),
    ("retries", ("parse_max_attempts",), 5),
    ("agent", ("team_agents", "schema_selector", "tools", "select_tables", "parser_name"), "select_columns"),
])
def test_a_replay_runs_under_the_sources_policy(monkeypatch, repo, source, block, path, value):
    config = yaml.safe_load(repo.read_text())
    node = config[block]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    repo.write_text(yaml.safe_dump(config))
    runs = set(paths.RUNS.iterdir())
    with pytest.raises(runner.data.DataError, match=".".join((block, *path))):  # the difference, named
        replay(str(repo), source, engine="production_llm")
    assert set(paths.RUNS.iterdir()) == runs


def test_an_empty_replay_fails_with_its_reason(monkeypatch, repo):
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    plain = runner.run_agent(str(repo), "B0", "train", ids=["1"])  # no SQL to repair: no revise call
    _, manifest, calls = replayed(monkeypatch, repo, plain.name, engine="production_llm", call_sites=["revise"])
    assert (manifest["status"], manifest["n_invocations"], calls) == ("failed", 0, [])
    assert "no invocation to replay" in manifest["problems"][0] and "revise" in manifest["problems"][0]


def test_an_api_error_in_a_replay_fails_it(monkeypatch, repo, source):
    class Unpaid(Revising):
        def invoke(self, messages):
            raise openai.APIStatusError("402", response=httpx.Response(402, request=httpx.Request("POST", "http://x")), body=None)
    _, manifest, _ = replayed(monkeypatch, repo, source, model=Unpaid(), engine="production_llm")
    assert manifest["status"] == "failed" and manifest["model_failures"] == {} and manifest["harness_errors"]


def test_a_source_whose_snapshot_changed_is_refused(monkeypatch, repo, source):
    snapshot = paths.RUNS / source / "config.json"
    config = json.loads(snapshot.read_text())
    config["eval"]["timeout_s"] += 1  # outside the policy blocks: only the hash against the manifest sees it
    snapshot.write_text(json.dumps(config))
    with pytest.raises(runner.data.DataError, match="does not match its manifest"):
        replay(str(repo), source, engine="production_llm")
