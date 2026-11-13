"""EXPLORATORY local patch (scratch worktree, never merged): the constant column filter.

`BENCH_CONST_FILTER=No|Yes bench run --arm B5` on a split that is not `test` answers every `filter_column`
invocation with the constant, calling no model for it; every other call site is routed as B5 routes it.
The real patched CHESS on the synthetic repository, with the scripted model of test_agent_wiring.py and
the facts of test_agent_arms.py: no network. Agent environment only."""
import json
import pickle

import pytest

pytest.importorskip("langchain_core")

from bench import barrier, paths  # noqa: E402
from bench.agent import hooks, runner  # noqa: E402
from bench.contracts import clusters  # noqa: E402
from bench.contracts.calls import read_calls, validate_calls  # noqa: E402
from bench.contracts.facts import write_fact  # noqa: E402
from bench.data import DataError  # noqa: E402
from bench.evaluate import evaluate  # noqa: E402
from test_agent_arms import adapters_for, rel, repo, set_config  # noqa: E402,F401  (repo: a fixture)
from test_agent_wiring import ScriptedChess  # noqa: E402

FILTER_ADAPTER = "slm:qwen3-8b+lora:qwen3-8b-c0"
AGENT_ADAPTER = "slm:qwen3-8b+lora:qwen3-8b-c3"
MARKER = "is_column_information_relevant"  # only the column filter's prompt carries it


def three_ways(texts, embedding):
    """A stand-in for the embedding model, shaped like the real allocation: the column filter's prompts
    to c0, the agents' own prompts to c3, every other prompt to c1."""
    return [[1.0, 0.0, 0.0] if MARKER in t else [0.0, 0.0, 1.0] if "agent, in a team" in t else [0.0, 1.0, 0.0]
            for t in texts]


class Recording(ScriptedChess):
    """The scripted model of one engine: it records every prompt that reaches it."""

    def __init__(self, engine, seen):
        super().__init__()
        self.engine, self.seen = engine, seen

    def invoke(self, messages):
        self.seen.append((self.engine, messages[-1].content))
        return super().invoke(messages)


@pytest.fixture
def b5(monkeypatch, repo):  # noqa: F811
    """B5 as registered, in small: c0 (the column filter) and c3 (the agents) on their adapters, c1 left with
    the production LLM. Returns a function that runs it and what it saw."""
    for name in (hooks.CONST_FILTER_ENV, hooks.CONST_FILTER_CLUSTER_ENV):
        monkeypatch.delenv(name, raising=False)
    choice = write_fact("J6", "choice", {"slm": "qwen3-8b"})
    embedding = {"model": "unit-test", "revision": "0" * 40, "max_seq_length": 128, "truncation": "head", "text": "prompt"}
    centroids = write_fact("J5", "centroids", {"embedding": embedding, "clusters": {
        "c0": [1.0, 0.0, 0.0], "c1": [0.0, 1.0, 0.0], "c3": [0.0, 0.0, 1.0]}})
    adapters = adapters_for(choice, centroids, ("c0", "c1", "c3"))
    allocation = write_fact("J7", "allocation", {"centroids": centroids.parent.name, "adapters": adapters.parent.name,
                                                 "allocation": {"c0": "slm", "c3": "slm"}})
    set_config(repo, B5={"choice": rel(choice), "centroids": rel(centroids), "adapters": rel(adapters),
                         "allocation": rel(allocation)})
    monkeypatch.setattr(clusters, "embed", three_ways)

    def run(answer=None, cluster="c0", split="train", ids=("1", "2"), arm="B5", **kwargs):
        if answer is not None:
            monkeypatch.setenv(hooks.CONST_FILTER_ENV, answer)
            if cluster:
                monkeypatch.setenv(hooks.CONST_FILTER_CLUSTER_ENV, cluster)
        built, seen = [], []
        monkeypatch.setattr(hooks, "chat_model",
                            lambda engine, temperature: built.append(engine) or Recording(engine, seen))
        run_dir = runner.run_agent(str(repo), arm, split, ids=list(ids), **kwargs)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        return run_dir, manifest, read_calls(run_dir / "calls.jsonl"), built, seen
    return run


def routed(calls):
    """What the router decided for every call but the column filter's, in the run's order."""
    return [(c["question_id"], c["call_site"], c["invocation_key"], c["cluster"], c["engine"])
            for c in calls if c["call_site"] != "filter_column"]


def schema_sent_to(calls, site="select_tables", question="1"):
    return next(c["prompt_messages"][-1]["content"] for c in calls
                if c["call_site"] == site and c["question_id"] == question)




@pytest.mark.parametrize("answer", ["No", "Yes"])
def test_core(b5, answer):
    run_dir, manifest, calls, built, seen = b5(answer)
    assert manifest["status"] == "done" and manifest["c1_errors"] == 0
    filtered = [c for c in calls if c["call_site"] == "filter_column"]
    assert len(filtered) >= 4
    for call in filtered:
        assert call["parsed_ok"] and call["parsed_output"]["is_column_information_relevant"] == answer
        assert (call["engine"], call["endpoint"]) == (f"const:{answer}", "fake") and call["latency_ms"] == 0
    assert not [engine for engine, text in seen if MARKER in text]
    others = {c["engine"] for c in calls if c["call_site"] != "filter_column"}
    assert "production_llm" in others and AGENT_ADAPTER in others


def test_plain_b5_still_calls_the_filter_adapter(b5):
    run_dir, manifest, calls, built, seen = b5(None)
    assert {c["engine"] for c in calls if c["call_site"] == "filter_column"} == {FILTER_ADAPTER}
