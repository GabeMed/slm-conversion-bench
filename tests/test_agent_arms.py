"""Every arm end to end on a synthetic repository, driving the real patched CHESS with a scripted
model (the pattern of test_agent_wiring.py): B1 with its few-shot prefix, B2 in one call, B3 on the
base SLM, B4 on the adapter of each call's cluster (`assign` with a real, tiny sentence-transformers
model), B5 by its allocation. Facts are written with `write_fact`. Agent environment only."""
import json

import pytest
import yaml

pytest.importorskip("langchain_core")

from bench import paths  # noqa: E402
from bench.agent import hooks, runner  # noqa: E402
from bench.contracts import clusters  # noqa: E402
from bench.contracts.calls import CALL_SITES, read_calls, validate_calls  # noqa: E402
from bench.contracts.config import load_config  # noqa: E402
from bench.contracts.facts import write_fact  # noqa: E402
from bench.evaluate import evaluate  # noqa: E402
from synthetic import GOLD  # noqa: E402
from test_agent_patches import tiny_sentence_transformer  # noqa: E402,F401  (a fixture)
from test_agent_wiring import ScriptedChess, make_concurrent_repo  # noqa: E402

BROKEN_SQL = "SELECT no_such_column FROM gas_t"


class Revising(ScriptedChess):
    """Generates a SQL that fails, so the agent repairs it: every call site of the agent is called."""

    def answer(self, text):
        if text.startswith("**Task Description:**\nYou are an SQL database expert tasked with correcting"):
            return f"fixed\n<FINAL_ANSWER>\n{GOLD}\n</FINAL_ANSWER>"
        if text.startswith("You are an experienced database expert"):
            return f"plan\n<FINAL_ANSWER>\n{BROKEN_SQL}\n</FINAL_ANSWER>"
        return super().answer(text)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    config_path, _ = make_concurrent_repo(tmp_path, monkeypatch)
    config = yaml.safe_load(config_path.read_text())
    for candidate in config["roles"]["slm_candidates"]:
        candidate["endpoint"]["base_url"] = "http://slm.example/v1"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def set_config(config_path, **arms):
    config = yaml.safe_load(config_path.read_text())
    for arm, settings in arms.items():
        config["arms"][arm] = {**(config["arms"].get(arm) or {}), **settings}
    config_path.write_text(yaml.safe_dump(config))


def run(monkeypatch, config_path, arm, ids, model=None, **kwargs):
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: model or ScriptedChess())
    run_dir = runner.run_agent(str(config_path), arm, "train", ids=list(ids), **kwargs)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    calls = read_calls(run_dir / "calls.jsonl")
    assert validate_calls(calls) == [] and manifest["c1_errors"] == 0
    return run_dir, manifest, calls


def rel(path):
    return str(path.relative_to(paths.ROOT))


def teacher_run(monkeypatch, config_path):
    """A done B0 run on train where every call site succeeds at least twice: the few-shot source."""
    run_dir, manifest, calls = run(monkeypatch, config_path, "B0", ("1", "2"), model=Revising())
    assert manifest["status"] == "done" and {c["call_site"] for c in calls} == set(CALL_SITES)
    return run_dir.name


# ---------------------------------------------------------------- B1

def test_b1_prefixes_each_call_site_with_the_teachers_examples(monkeypatch, repo):
    source = teacher_run(monkeypatch, repo)
    set_config(repo, B1={"few_shot": {"k": 2, "source_run": source}})
    routed = []
    real_route = hooks.route
    monkeypatch.setattr(hooks, "route", lambda arm, site, messages, config: routed.append(messages) or real_route(arm, site, messages, config))
    _, manifest, calls = run(monkeypatch, repo, "B1", ("3",))
    assert manifest["status"] == "done" and manifest["few_shot"]["source_run"] == source
    assert all(len(messages) == 1 for messages in routed)  # the router sees CHESS's prompt only
    teacher = {(c["call_site"], c["question_id"], c["invocation_key"]): c
               for c in read_calls(paths.RUNS / source / "calls.jsonl") if c["parsed_ok"]}
    for call in calls:
        assert (call["engine"], call["model_role"]) == ("cheap_alt", "cheap_alt")
        prefix, prompt = call["prompt_messages"][:4], call["prompt_messages"][4:]
        assert [m["role"] for m in prefix] == ["user", "assistant", "user", "assistant"] and len(prompt) == 1
        examples = manifest["few_shot"]["examples"][call["call_site"]]
        for (question_id, key), (user, assistant) in zip(examples, zip(prefix[::2], prefix[1::2])):
            example = teacher[(call["call_site"], question_id, key)]
            assert [user] == example["prompt_messages"] and assistant["content"] == example["response_text"]
    per_site = {}
    for call in calls:  # one fixed prefix per call site, so a provider can cache it
        per_site.setdefault(call["call_site"], set()).add(json.dumps(call["prompt_messages"][:4]))
    assert all(len(prefixes) == 1 for prefixes in per_site.values())


def test_the_examples_are_seeded(monkeypatch, repo):
    source = teacher_run(monkeypatch, repo)
    from bench.agent import few_shot
    config = yaml.safe_load(repo.read_text())
    config["arms"]["B1"]["few_shot"] = {"k": 1, "source_run": source}
    first = few_shot.build(config)
    assert few_shot.build(config) == first
    seeds = {json.dumps(few_shot.build({**config, "seeds": {**config["seeds"], "few_shot": s}})[1]["examples"])
             for s in range(8)}
    assert len(seeds) > 1


def test_cheap_alt_never_runs_without_its_few_shot(monkeypatch, repo):
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    with pytest.raises(hooks.HarnessError, match="source_run is not set"):
        runner.run_agent(str(repo), "B1", "train", ids=["1"])  # k: 3, no source: refused before anything runs
    set_config(repo, B1={"few_shot": {"k": 3, "source_run": "agent-B0-train-missing"}})
    with pytest.raises(hooks.HarnessError, match="does not exist"):
        runner.run_agent(str(repo), "B1", "train", ids=["1"])
    source = run(monkeypatch, repo, "B0", ("1",))[0].name  # no revise call: too few examples
    set_config(repo, B1={"few_shot": {"k": 1, "source_run": source}})
    with pytest.raises(hooks.HarnessError, match="0 successful revise"):
        runner.run_agent(str(repo), "B1", "train", ids=["1"])
    assert sorted(p.name for p in paths.RUNS.iterdir()) == [source]
    hooks.configure(yaml.safe_load(repo.read_text()))
    hooks.start_run("r", "B1", paths.RUNS / "r" / "calls.jsonl")  # a runner that forgot to load it
    hooks.set_question("1")
    try:
        with pytest.raises(hooks.HarnessError, match="without its few-shot"):
            hooks.invoke_agent_call("agent_ir", "ir:0", "state", lambda r: {"done": True})
    finally:
        hooks.end_run()


def test_k_zero_is_an_explicit_empty_prefix(monkeypatch, repo):
    set_config(repo, B1={"few_shot": {"k": 0, "source_run": None}})
    _, manifest, calls = run(monkeypatch, repo, "B1", ("1",))
    assert manifest["status"] == "done" and manifest["few_shot"] == {"k": 0, "source_run": None}
    assert manifest["few_shot_k"] == 0 and set(manifest["few_shot_sha256"]) == set(CALL_SITES)
    assert all(len(c["prompt_messages"]) == 1 for c in calls)


# ---------------------------------------------------------------- B2

@pytest.mark.parametrize("status", [400, 401])  # the model's failure (failures) and the harness's (stopped_by)
def test_a_providers_error_echoing_a_key_never_reaches_the_manifest(monkeypatch, repo, status):
    import contextlib

    import httpx
    import openai
    monkeypatch.setenv("OPENAI_API_KEY", "embeddings-key-0001")  # a credential of every configuration (C2)
    error = {400: openai.BadRequestError, 401: openai.AuthenticationError}[status]

    class Refusing(ScriptedChess):
        def invoke(self, messages):
            raise error(f"Error code: {status} - the key embeddings-key-0001 was refused", body=None,
                        response=httpx.Response(status, request=httpx.Request("POST", "http://x")))
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: Refusing())
    with contextlib.suppress(Exception):
        runner.run_agent(str(repo), "B2", "train", ids=["1"], engine="production_llm")
    (run_dir,) = paths.RUNS.glob("agent-B2-production_llm-train-*")
    manifest = (run_dir / "manifest.json").read_text()
    assert "embeddings-key-0001" not in manifest and "<redacted>" in manifest


@pytest.mark.parametrize("engine", ["production_llm", "cheap_alt"])
def test_b2_is_one_call_per_question_on_the_complete_schema(monkeypatch, repo, engine):
    run_dir, manifest, calls = run(monkeypatch, repo, "B2", ("1", "2"), engine=engine)
    assert (manifest["status"], manifest["mode"], manifest["engine"]) == ("done", "single_call", engine)
    assert [(c["question_id"], c["call_site"], c["invocation_key"], c["engine"]) for c in calls] == [
        ("1", "generate_candidate", "b2:0", engine), ("2", "generate_candidate", "b2:0", engine)]
    (prompt,) = calls[0]["prompt_messages"]  # no few-shot prefix, cheap_alt included
    assert (manifest["few_shot"], manifest["few_shot_k"], manifest["few_shot_sha256"]) == (None, None, None)
    assert prompt["content"].startswith("You are an experienced database expert")
    assert "CREATE TABLE gas_t" in prompt["content"] and "CREATE TABLE other_u" in prompt["content"]
    assert "price segment" not in prompt["content"]  # no retrieval: no column description from the vector DB
    assert "`CZE`" in prompt["content"]  # the example values CHESS's schema string reads from the database itself
    assert json.loads((run_dir / "predictions.json").read_text()) == {"1": f" {GOLD} ", "2": f" {GOLD} "}
    eval_manifest = json.loads((evaluate(run_dir.name) / "manifest.json").read_text())
    assert (eval_manifest["n"], eval_manifest["correct"]) == (2, 2)


def test_b2_needs_its_engine_and_other_arms_refuse_one(monkeypatch, repo):
    with pytest.raises(runner.data.DataError, match="B2 needs --engine"):
        runner.run_agent(str(repo), "B2", "train", ids=["1"])
    with pytest.raises(runner.data.DataError, match="--engine is for B2"):
        runner.run_agent(str(repo), "B0", "train", ids=["1"], engine="cheap_alt")


def test_b2_keeps_going_when_the_model_fails_a_question(monkeypatch, repo):
    class Unparseable(ScriptedChess):
        def answer(self, text):
            return ""  # empty output, every attempt
    run_dir, manifest, calls = run(monkeypatch, repo, "B2", ("1", "2"), model=Unparseable(), engine="production_llm")
    assert manifest["status"] == "done" and set(manifest["tool_errors"]) == {"1", "2"}
    assert manifest["errors_by_call_site"] == {"generate_candidate": {"empty output": 4}}  # visible without C1
    assert json.loads((run_dir / "predictions.json").read_text()) == {"1": None, "2": None}
    assert [c["attempt"] for c in calls] == [1, 2, 1, 2]


# ---------------------------------------------------------------- B3, B4, B5

def facts(repo_root, served=("c0", "c1"), embedding=None):
    choice = write_fact("J6", "choice", {"slm": "qwen3-8b"})
    embedding = embedding or {"model": "unit-test", "revision": "0" * 40, "max_seq_length": 128,
                              "truncation": "head", "text": "prompt"}
    centroids = write_fact("J5", "centroids", {"embedding": embedding, "clusters": {"c0": [1.0, 0.0], "c1": [0.0, 1.0]}})
    return choice, centroids


QWEN = next(c for c in load_config(paths.ROOT / "config.yaml")["roles"]["slm_candidates"] if c["name"] == "qwen3-8b")


def adapters_for(choice, centroids, clusters_):
    return write_fact("S5", "adapters", {
        "slm": "qwen3-8b", "choice": choice.parent.name, "centroids": centroids.parent.name,
        "base_revision": QWEN["hf"]["revision"], "chat_template_kwargs": QWEN["chat_template_kwargs"],
        "adapters": {c: {"served_name": f"qwen3-8b-{c}", "sha256": f"{i}" * 64} for i, c in enumerate(clusters_)}})


def by_column_filter(texts, embedding):
    """A stand-in for the embedding model: column-filter prompts to c0, every other prompt to c1."""
    return [[1.0, 0.0] if "is_column_information_relevant" in t else [0.0, 1.0] for t in texts]


def test_b3_runs_every_call_on_the_chosen_base(monkeypatch, repo):
    choice, _ = facts(paths.ROOT)
    set_config(repo, B3={"choice": rel(choice)})
    _, manifest, calls = run(monkeypatch, repo, "B3", ("1",))
    assert manifest["status"] == "done" and manifest["facts"] == {"choice": choice.parent.name}
    assert {(c["engine"], c["model"], c["model_role"], c["endpoint"], c["cluster"]) for c in calls} == {
        ("slm:qwen3-8b", "qwen3-8b", "slm", "vllm", None)}


def test_b4_serves_each_call_with_the_adapter_of_its_cluster(monkeypatch, repo):
    choice, centroids = facts(paths.ROOT)
    adapters = adapters_for(choice, centroids, ("c0", "c1"))
    set_config(repo, B4={"choice": rel(choice), "centroids": rel(centroids), "adapters": rel(adapters)})
    monkeypatch.setattr(clusters, "embed", by_column_filter)
    _, manifest, calls = run(monkeypatch, repo, "B4", ("1",))
    assert manifest["status"] == "done" and set(manifest["facts"]) == {"choice", "centroids", "adapters"}
    for call in calls:
        cluster = "c0" if call["call_site"] == "filter_column" else "c1"
        assert (call["cluster"], call["engine"], call["model"]) == (
            cluster, f"slm:qwen3-8b+lora:qwen3-8b-{cluster}", f"qwen3-8b-{cluster}")


def test_b4_assigns_with_the_pinned_sentence_transformer(monkeypatch, repo, tiny_sentence_transformer):
    embedding = {"model": tiny_sentence_transformer, "revision": "0" * 40, "max_seq_length": 128,
                 "truncation": "head", "text": "prompt"}
    anchors = clusters.embed(["is_column_information_relevant", "You are an experienced database expert"], embedding)
    choice = write_fact("J6", "choice", {"slm": "qwen3-8b"})
    centroids = write_fact("J5", "centroids", {"embedding": embedding, "clusters": {"c0": anchors[0], "c1": anchors[1]}})
    adapters = adapters_for(choice, centroids, ("c0", "c1"))
    set_config(repo, B4={"choice": rel(choice), "centroids": rel(centroids), "adapters": rel(adapters)})
    _, manifest, calls = run(monkeypatch, repo, "B4", ("1",))
    assert manifest["status"] == "done"
    fact = json.loads(centroids.read_text())
    for call in calls:
        assert call["cluster"] == clusters.assign(call["prompt_messages"], fact)
        assert call["engine"] == f"slm:qwen3-8b+lora:qwen3-8b-{call['cluster']}"


def test_b5_follows_the_allocation_and_prefixes_only_cheap_alt(monkeypatch, repo):
    source = teacher_run(monkeypatch, repo)
    choice, centroids = facts(paths.ROOT)
    adapters = adapters_for(choice, centroids, ("c0", "c1"))
    allocation = write_fact("J7", "allocation", {"centroids": centroids.parent.name, "adapters": adapters.parent.name,
                                                 "allocation": {"c0": "slm", "c1": "cheap_alt"}})
    set_config(repo, B1={"few_shot": {"k": 1, "source_run": source}},
               B5={"choice": rel(choice), "centroids": rel(centroids), "adapters": rel(adapters), "allocation": rel(allocation)})
    embedded = []
    monkeypatch.setattr(clusters, "embed", lambda texts, embedding: embedded.extend(texts) or by_column_filter(texts, embedding))
    _, manifest, calls = run(monkeypatch, repo, "B5", ("3",))
    assert manifest["status"] == "done" and manifest["few_shot"]["source_run"] == source
    assert all(text.count("user: ") == 1 for text in embedded)  # assign sees CHESS's prompt, never the prefix
    for call in calls:
        if call["call_site"] == "filter_column":
            assert (call["cluster"], call["engine"], len(call["prompt_messages"])) == ("c0", "slm:qwen3-8b+lora:qwen3-8b-c0", 1)
        else:
            assert (call["cluster"], call["engine"], len(call["prompt_messages"])) == ("c1", "cheap_alt", 3)


def test_b5_leaves_an_unallocated_cluster_with_the_production_llm(monkeypatch, repo):
    choice, centroids = facts(paths.ROOT)
    adapters = adapters_for(choice, centroids, ("c0", "c1"))
    allocation = write_fact("J7", "allocation", {"centroids": centroids.parent.name, "adapters": adapters.parent.name,
                                                 "allocation": {"c0": "slm"}})
    set_config(repo, B5={"choice": rel(choice), "centroids": rel(centroids), "adapters": rel(adapters), "allocation": rel(allocation)})
    monkeypatch.setattr(clusters, "embed", by_column_filter)
    _, manifest, calls = run(monkeypatch, repo, "B5", ("1",))
    assert manifest["status"] == "done" and manifest["few_shot"] is None  # cheap_alt unreachable: no prefix needed
    assert {c["engine"] for c in calls if c["call_site"] != "filter_column"} == {"production_llm"}


def test_examples_are_only_the_teachers_successful_answers(monkeypatch, repo):
    class FirstTablesAnswerUnparseable(Revising):
        seen = set()

        def answer(self, text):
            if '"table_names"' in text and text not in self.seen:
                self.seen.add(text)
                return "the tables are gas_t"
            return super().answer(text)
    source = run(monkeypatch, repo, "B0", ("1", "2"), model=FirstTablesAnswerUnparseable())[0].name
    from bench.agent import few_shot
    config = yaml.safe_load(repo.read_text())
    for seed in range(8):
        config["seeds"]["few_shot"] = seed
        config["arms"]["B1"]["few_shot"] = {"k": 2, "source_run": source}
        prefix, _ = few_shot.build(config)
        # 2 of the 4 select_tables answers did not parse: over 8 seeds, a draw from all 4 would pick one
        assert all(m["content"] != "the tables are gas_t" for m in prefix["select_tables"])


@pytest.mark.parametrize("field,value", [("arm", "B1"), ("split", "calib"), ("status", "failed"), ("type", "replay")])
def test_the_examples_come_only_from_a_done_b0_run_on_train(monkeypatch, repo, field, value):
    source = teacher_run(monkeypatch, repo)
    manifest_path = paths.RUNS / source / "manifest.json"
    manifest_path.write_text(json.dumps({**json.loads(manifest_path.read_text()), field: value}))
    from bench.agent import few_shot
    config = yaml.safe_load(repo.read_text())
    config["arms"]["B1"]["few_shot"] = {"k": 1, "source_run": source}
    with pytest.raises(hooks.HarnessError, match="not a done B0 agent run on train"):
        few_shot.build(config)  # no calib or test answer can reach a prefix


def test_b2_checks_its_template_before_a_run_exists(monkeypatch, repo):
    config = yaml.safe_load(repo.read_text())
    generator = config["agent"]["team_agents"]["candidate_generator"]["tools"]["generate_candidate"]["generator_configs"][0]
    generator["template_name"] = "generate_candidate_two"
    repo.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    with pytest.raises(runner.data.DataError, match="generate_candidate_one"):
        runner.run_agent(str(repo), "B2", "train", ids=["1"], engine="production_llm")
    assert not paths.RUNS.exists() or not any(paths.RUNS.iterdir())


def test_the_manifest_records_k_and_the_sha256_of_each_prefix_as_sent(monkeypatch, repo):
    import hashlib
    from bench.contracts.facts import canonical
    source = teacher_run(monkeypatch, repo)
    set_config(repo, B1={"few_shot": {"k": 2, "source_run": source}})
    _, manifest, calls = run(monkeypatch, repo, "B1", ("3",))
    assert manifest["few_shot_k"] == 2 and set(manifest["few_shot_sha256"]) == set(CALL_SITES)
    for call in calls:
        sent = hashlib.sha256(canonical(call["prompt_messages"][:4])).hexdigest()
        assert manifest["few_shot_sha256"][call["call_site"]] == sent


def test_the_examples_are_questions_of_todays_train_split(monkeypatch, repo):
    from bench.agent import few_shot
    source = teacher_run(monkeypatch, repo)
    config = yaml.safe_load(repo.read_text())
    config["arms"]["B1"]["few_shot"] = {"k": 2, "source_run": source}
    manifest_path = paths.RUNS / source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "splits_sha256": "0" * 64}))
    with pytest.raises(hooks.HarnessError, match="other splits"):
        few_shot.build(config)
    manifest_path.write_text(json.dumps(manifest))
    splits = json.loads(paths.SPLITS.read_text())  # question 2 leaves train (the source's splits sha follows)
    paths.SPLITS.write_text(json.dumps({**splits, "train": ["1", "3"], "calib": ["2"]}))
    manifest_path.write_text(json.dumps({**manifest, "splits_sha256": runner.data.sha256_file(paths.SPLITS)}))
    with pytest.raises(hooks.HarnessError, match=r"not in the current train split: \['2'\]"):
        few_shot.build(config)


def test_an_api_error_in_b2_fails_the_run(monkeypatch, repo):
    import httpx
    import openai

    class Gone(ScriptedChess):
        def invoke(self, messages):
            raise openai.NotFoundError("no such model", response=httpx.Response(404, request=httpx.Request("POST", "http://x")), body=None)
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: Gone())
    run_dir = runner.run_agent(str(repo), "B2", "train", ids=["1", "2"], engine="production_llm")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed" and list(manifest["harness_errors"]) == ["1"]  # stopped at the first
    assert manifest["tool_errors"] == {} and "HarnessError" in manifest["failures"]["1"]  # never the model's
