"""The test registry (REQ-013) and the call-site assertion (REQ-001; SPEC 7.1), on a temporary git
repository with a published pre-registration: an execution on `test` commits its intent before
anything runs and its manifest when it ends, never anything else; it runs only when the call sites
of train and calib are registered and committed, and a call site outside them aborts it."""
import hashlib
import json
import os
import subprocess

import pytest

pytest.importorskip("langchain_core")
import httpx  # noqa: E402
import openai  # noqa: E402

from bench import barrier, paths  # noqa: E402
from bench.data import DataError  # noqa: E402
from bench.agent import hooks, registry, runner  # noqa: E402
from bench.contracts.calls import CALL_SITES, read_calls  # noqa: E402
from bench.contracts.config import config_sha256  # noqa: E402
from test_agent_wiring import ScriptedChess, make_concurrent_repo  # noqa: E402


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A clone of a bare origin, with the pre-registration of this configuration published."""
    origin, root = tmp_path / "origin.git", tmp_path / "repo"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    root.mkdir()
    config_path, config = make_concurrent_repo(root, monkeypatch)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    git(root, "remote", "add", "origin", str(origin))
    (root / ".gitignore").write_text("runs/\ndata/raw/\ndata/bird_dev/\n")
    (root / "SPEC.md").write_text("protocol\n")
    (root / "bench" / "agent").mkdir(parents=True)
    (root / "bench" / "agent" / "code.py").write_text("x = 1\n")  # execution code the registration hashes
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    publish_prereg(root, config)
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    return root, config_path


def publish_prereg(root, config, **override):
    """Register HEAD as `bench prereg` does (the commit it records, then the commit of prereg/), and push."""
    (root / "prereg").mkdir(exist_ok=True)
    prereg = {"spec_sha256": sha(root / "SPEC.md"), "config_sha256": config_sha256(config),
              "splits_sha256": sha(paths.SPLITS), "data_manifest_sha256": sha(paths.DATA_MANIFEST),
              "commit": git(root, "rev-parse", "HEAD").strip(), "execution_code": barrier.execution_code(root),
              **override}
    (root / "prereg" / "manifest.json").write_text(json.dumps(prereg, sort_keys=True) + "\n")
    (root / "prereg" / "HASH").write_text(sha(root / "prereg" / "manifest.json") + "\n")
    git(root, "add", "prereg")
    git(root, "commit", "-q", "-m", "register")
    git(root, "push", "-q", "origin", "main")
    return prereg


def deviation(root, run_id, defect="a defect"):
    """The line a redo on test needs first (SPEC D22): prereg/DEVIATIONS.md, committed and pushed."""
    path = root / "prereg" / "DEVIATIONS.md"
    with open(path, "a") as fh:
        fh.write(f"- 2026-11-10 · {run_id} run again from its first question: {defect}. Direction: none\n")
    git(root, "add", "prereg/DEVIATIONS.md")
    git(root, "commit", "-q", "-m", "a deviation")
    git(root, "push", "-q", "origin", "main")


def register(root, config_path, call_sites=None, commit=True):
    train = runner.run_agent(str(config_path), "B0", "train", ids=["1"])
    path = registry.update_call_sites([train.name])
    if call_sites is not None:
        registered = json.loads(path.read_text())
        path.write_text(json.dumps({**registered, "call_sites": call_sites}))
    if commit:
        git(root, "add", registry.CALL_SITES_FILE)
        git(root, "commit", "-q", "-m", "register the call sites")
        git(root, "push", "-q", "origin", "main")
    return train


def test_the_intent_is_committed_before_anything_runs_and_the_manifest_after(repo, monkeypatch):
    root, config_path = repo
    register(root, config_path)
    before = git(root, "rev-parse", "HEAD").strip()
    seen_while_running = []

    class Watching(ScriptedChess):
        def invoke(self, messages):
            if not seen_while_running:
                seen_while_running.append(git(root, "ls-files", "registry/test").split())
            if "relevant keywords" in messages[-1].content:  # an error whose text carries a local path
                raise OSError(f"cannot read {root}/secret/prompt.txt")  # unrecognised: fails the run
            return super().invoke(messages)
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: Watching())
    run_dir = runner.run_agent(str(config_path), "B0", "test")
    intent, manifest_rel = f"registry/test/{run_dir.name}.intent.json", f"registry/test/{run_dir.name}.manifest.json"
    assert seen_while_running == [[intent, manifest_rel]]  # a run that dies from here on leaves its manifest `running`
    started = json.loads(git(root, "show", f"HEAD~1:{manifest_rel}"))
    assert (started["status"], started["run_id"], started["finished_at"]) == ("running", run_dir.name, None)
    assert sorted(git(root, "ls-files", "registry/test").split()) == [intent, manifest_rel]
    assert git(root, "diff", "--name-only", before, "HEAD").split() == [intent, manifest_rel]  # nothing else, ever
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed" and json.loads(git(root, "show", f"HEAD:{manifest_rel}")) == manifest
    assert manifest["prereg_hash"] == (root / "prereg" / "HASH").read_text().strip()
    assert manifest["call_sites_registry_sha256"] == sha(root / registry.CALL_SITES_FILE)
    assert manifest["unregistered_call_sites"] == [] and manifest["commit"] == before
    recorded = json.loads(git(root, "show", f"HEAD~1:{intent}"))
    assert (recorded["run_id"], recorded["arm"], recorded["prereg_hash"]) == (run_dir.name, "B0", manifest["prereg_hash"])
    committed = git(root, "show", f"HEAD:{manifest_rel}")
    assert "<repo>/secret/prompt.txt" in committed  # the error reached the committed record, scrubbed
    assert str(root) not in committed + git(root, "show", f"HEAD~1:{intent}")


def test_the_intent_is_on_origin_before_the_first_question(repo, monkeypatch):
    """Published, not only committed: while the run answers its first call, origin's main already
    holds the intent (and not the manifest, which does not exist yet)."""
    root, config_path = repo
    register(root, config_path)
    origin, on_origin = root.parent / "origin.git", []

    class Watching(ScriptedChess):
        def invoke(self, messages):
            if not on_origin:
                on_origin.append(git(origin, "ls-tree", "-r", "--name-only", "main", "registry/test").split())
            return super().invoke(messages)
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: Watching())
    run_dir = runner.run_agent(str(config_path), "B0", "test")
    assert on_origin == [[f"registry/test/{run_dir.name}.intent.json", f"registry/test/{run_dir.name}.manifest.json"]]
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "done"


def refused_before_spending(root, config_path, monkeypatch, match):
    """Run on test expecting the intent's publication to refuse it; returns the intent left committed."""
    calls = []

    class Counting(ScriptedChess):
        def invoke(self, messages):
            calls.append(1)
            return super().invoke(messages)
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: Counting())
    with pytest.raises(barrier.TestSplitLocked, match=match):
        runner.run_agent(str(config_path), "B0", "test")
    assert calls == []  # not one model call
    assert not any(p.name.startswith("agent-B0-test") for p in paths.RUNS.iterdir())  # and no run
    intent, started = git(root, "ls-files", "registry/test").split()
    assert intent.endswith(".intent.json") and started == intent.replace(".intent.json", ".manifest.json")
    return intent


def test_a_failed_push_of_the_intent_refuses_the_run_before_any_spend(repo, monkeypatch):
    root, config_path = repo
    register(root, config_path)
    origin = root.parent / "origin.git"
    hook = origin / "hooks" / "pre-receive"  # origin refuses every push from here on (fetches still work)
    hook.write_text("#!/bin/sh\necho 'pushes are closed' >&2\nexit 1\n")
    hook.chmod(0o755)
    intent = refused_before_spending(root, config_path, monkeypatch, "not published \\(the push to origin/main failed")
    assert git(origin, "ls-tree", "-r", "--name-only", "main", "registry/").split() == [registry.CALL_SITES_FILE]
    # the intent stays committed here, the record of the attempt; until it is pushed nothing else runs on test
    with pytest.raises(barrier.TestSplitLocked, match="push the registry first"):
        runner.run_agent(str(config_path), "B0", "test")
    hook.unlink()
    git(root, "push", "-q", "origin", "main")
    # published now, the refused attempt is an earlier attempt: running again from scratch states why
    with pytest.raises(barrier.TestSplitLocked, match="already attempted it"):
        runner.run_agent(str(config_path), "B0", "test")
    refused_id = intent.rsplit("/", 1)[1].rsplit(".", 2)[0]
    # it never ran, so it cannot be resumed (no run directory): a redo, once DEVIATIONS.md names it
    with pytest.raises(barrier.TestSplitLocked, match=f"DEVIATIONS.md, as published, has no line naming {refused_id}"):
        runner.run_agent(str(config_path), "B0", "test", redo_reason="the intent could not be pushed")
    deviation(root, refused_id, "its intent could not be pushed")
    run_dir = runner.run_agent(str(config_path), "B0", "test", redo_reason="the intent could not be pushed")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "done" and manifest["redo_reason"] == "the intent could not be pushed"
    assert manifest["redo"]["of"] == refused_id and manifest["redo"]["affects_completed"] is False
    assert "has no run directory here" in manifest["redo"]["not_resumable_because"]
    assert [a["run_id"] for a in manifest["earlier_attempts"]] == [intent.rsplit("/", 1)[1].rsplit(".", 2)[0]]
    assert manifest["earlier_attempts"][0]["status"] == "running"  # it never ran: its manifest is as it started
    assert intent in git(origin, "ls-tree", "-r", "--name-only", "main", "registry/test").split()


def test_a_push_that_succeeds_is_still_confirmed_by_what_a_fetch_finds(repo, monkeypatch):
    """The push's own answer is not the evidence: a push that lands somewhere else than the origin a
    fetch reads (here, a separate push URL) leaves origin's main without the intent, and the run is refused."""
    root, config_path = repo
    register(root, config_path)
    elsewhere = root.parent / "elsewhere.git"
    git(root.parent, "clone", "-q", "--bare", str(root.parent / "origin.git"), str(elsewhere))
    git(root, "config", "remote.origin.pushurl", str(elsewhere))
    refused_before_spending(root, config_path, monkeypatch, "origin/main does not contain it after the push")
    assert len(git(elsewhere, "ls-tree", "-r", "--name-only", "main", "registry/test").split()) == 2  # it was pushed


def test_a_single_call_execution_on_test_is_registered_too(repo):
    root, config_path = repo
    register(root, config_path)
    run_dir = runner.run_agent(str(config_path), "B2", "test", engine="production_llm")
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "done"
    assert sorted(git(root, "ls-files", "registry/test").split()) == [
        f"registry/test/{run_dir.name}.intent.json", f"registry/test/{run_dir.name}.manifest.json"]


@pytest.mark.parametrize("selection", [{"ids": ["9"]}, {"limit": 1}])
def test_part_of_the_test_split_is_refused_before_anything_is_recorded(repo, selection):
    """A configuration has one run on test: a `done` run of part of the split would be it."""
    root, config_path = repo
    register(root, config_path)
    with pytest.raises(DataError, match="--ids and --limit are refused on the test split"):
        runner.run_agent(str(config_path), "B0", "test", **selection)
    assert not (root / "registry" / "test").exists()
    assert all(not p.name.startswith("agent-B0-test") for p in paths.RUNS.iterdir())


def test_the_send_modes_are_never_read_from_the_shell(repo, monkeypatch):
    """BENCH_ANTHROPIC_BATCH / BENCH_VIA_CLAUDE_CODE left exported are cleared: only --batch and
    --via-claude-code (K1 only) choose how the production LLM's calls are sent."""
    root, config_path = repo
    register(root, config_path)
    monkeypatch.setenv("BENCH_ANTHROPIC_BATCH", "1")
    monkeypatch.setenv("BENCH_VIA_CLAUDE_CODE", "1")
    seen = []

    class Watching(ScriptedChess):
        def invoke(self, messages):
            seen.append((os.environ.get("BENCH_ANTHROPIC_BATCH"), os.environ.get("BENCH_VIA_CLAUDE_CODE")))
            return super().invoke(messages)
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: Watching())
    run_dir = runner.run_agent(str(config_path), "B0", "test")
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "done"
    assert seen and set(seen) == {(None, None)}


def test_a_call_site_outside_the_registered_set_aborts_the_run(repo):
    root, config_path = repo
    register(root, config_path, call_sites=[s for s in CALL_SITES if s != "select_columns"])
    run_dir = runner.run_agent(str(config_path), "B0", "test")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["unregistered_call_sites"] == ["select_columns"]
    assert "REQ-001" in json.dumps(manifest["harness_errors"])
    seen = {c["call_site"] for c in read_calls(run_dir / "calls.jsonl")}
    assert "select_columns" not in seen  # stopped before the call
    assert not seen & {"agent_cg", "generate_candidate"}  # and nothing after it in that question either
    assert json.loads(git(root, "show", f"HEAD:registry/test/{run_dir.name}.manifest.json"))["status"] == "failed"


def test_no_registered_call_sites_no_test_run(repo):
    root, config_path = repo
    with pytest.raises(barrier.TestSplitLocked, match="does not exist"):
        runner.run_agent(str(config_path), "B0", "test")
    register(root, config_path, commit=False)
    with pytest.raises(barrier.TestSplitLocked, match="not committed"):
        runner.run_agent(str(config_path), "B0", "test")
    git(root, "add", registry.CALL_SITES_FILE)
    git(root, "commit", "-q", "-m", "register")
    (root / registry.CALL_SITES_FILE).write_text('{"call_sites": ["agent_ir"]}')
    with pytest.raises(barrier.TestSplitLocked, match="uncommitted"):
        runner.run_agent(str(config_path), "B0", "test")
    assert not (root / "registry" / "test").exists()
    assert all(not p.name.startswith("agent-B0-test") for p in paths.RUNS.iterdir())


def test_only_done_train_and_calib_runs_register_call_sites(repo):
    root, config_path = repo
    train = register(root, config_path)
    registered = json.loads((root / registry.CALL_SITES_FILE).read_text())
    assert registered["call_sites"] == sorted(set(CALL_SITES) - {"revise"})  # what the run called, nothing more
    assert registered["sources"][train.name]["split"] == "train"
    test_run = runner.run_agent(str(config_path), "B0", "test")
    with pytest.raises(registry.RegistryError, match="only train and calib"):
        registry.update_call_sites([test_run.name])
    manifest = json.loads((train / "manifest.json").read_text())
    (train / "manifest.json").write_text(json.dumps({**manifest, "status": "failed"}))
    with pytest.raises(registry.RegistryError, match="not 'done'"):
        registry.update_call_sites([train.name])


def test_the_registry_commits_its_file_and_nothing_else(repo):
    """And a test execution starts only on a clean tree, so there is nothing else to leave alone but what
    .gitignore ignores."""
    root, config_path = repo
    register(root, config_path)
    (root / "staged.txt").write_text("staged, not committed\n")
    git(root, "add", "staged.txt")
    with pytest.raises(barrier.TestSplitLocked, match="not clean \\(A staged.txt\\)"):
        runner.run_agent(str(config_path), "B0", "test")
    git(root, "rm", "-q", "--cached", "staged.txt")
    (root / "staged.txt").unlink()
    (root / "runs" / "scratch.txt").write_text("ignored output\n")
    before = git(root, "rev-parse", "HEAD").strip()
    run_dir = runner.run_agent(str(config_path), "B0", "test")
    assert git(root, "diff", "--name-only", before, "HEAD").split() == [
        f"registry/test/{run_dir.name}.intent.json", f"registry/test/{run_dir.name}.manifest.json"]
    assert (root / "runs" / "scratch.txt").read_text() == "ignored output\n"


def test_a_manifest_the_registry_cannot_commit_fails_the_run(repo, monkeypatch):
    root, config_path = repo
    register(root, config_path)

    def refused(run_dir, config):
        raise registry.RegistryError("git commit failed: signing required")
    monkeypatch.setattr(registry, "commit_manifest", refused)
    with pytest.raises(hooks.HarnessError, match="signing required"):  # a harness error: the CLI reports it
        runner.run_agent(str(config_path), "B0", "test")
    (run_dir,) = [p for p in paths.RUNS.iterdir() if p.name.startswith("agent-B0-test")]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "registry" in manifest["problems"][-1]
    from bench.data import DataError
    from bench.evaluate import evaluate
    with pytest.raises(DataError, match="not 'done'"):
        evaluate(run_dir.name)  # never scored while the registry shows only its intent


def test_the_registry_is_pushed_by_what_a_fetch_finds_not_by_a_local_ref(repo):
    """origin/main moved locally (or left ahead of a rewound remote) must not make a record look pushed. In a
    clone whose refspec does not map main no fetch corrects that ref, so only the fetched commit tells."""
    root, config_path = repo
    register(root, config_path)
    registry.registry_pushed()
    git(root, "config", "--unset-all", "remote.origin.fetch")
    (root / "registry" / "test").mkdir(parents=True, exist_ok=True)
    (root / "registry" / "test" / "local-only.intent.json").write_text("{}\n")
    git(root, "add", "registry")
    git(root, "commit", "-q", "-m", "a record never pushed")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")  # a local ref that says it is
    with pytest.raises(barrier.TestSplitLocked, match="1 commit\\(s\\) of registry/ are not on origin/main"):
        registry.registry_pushed()


def test_registry_failures_reach_the_cli_as_errors(repo, capsys):
    from bench.cli import main
    assert main(["call-sites", "agent-B0-train-no-such-run"]) == 2
    assert "no run agent-B0-train-no-such-run" in capsys.readouterr().err


def test_what_is_committed_never_carries_a_key(monkeypatch):
    config = json.loads(json.dumps(runner.load_config(paths.ROOT / "config.yaml")))
    config["roles"]["cheap_alt"]["endpoint"]["api_key_env"] = "BENCH_TEST_KEY"
    monkeypatch.setenv("BENCH_TEST_KEY", "plain-secret-value-123")
    monkeypatch.setenv("OPENAI_API_KEY", "embeddings-secret-456")  # the retrieval embeddings' key, not in any role
    text = ("refused: plain-secret-value-123; embeddings: embeddings-secret-456; "
            "Incorrect API key provided: sk-proj-abc1***wxyz; hf_AbCdEf123456")
    assert registry.redact(text, config) == ("refused: <redacted>; embeddings: <redacted>; "
                                             "Incorrect API key provided: <redacted>; <redacted>")
    monkeypatch.setenv("OPENAI_API_KEY", "1")  # a local server's placeholder, not a secret
    assert registry.redact('{"question_ids": ["1", "12"]}', config) == '{"question_ids": ["1", "12"]}'  # intact


def test_what_is_committed_never_carries_a_header_credential(monkeypatch):
    """`headers_env` values (the Modal proxy token of the candidates on Modal) are credentials like any key:
    redacted from the public record, and checked for length before a test execution. The shipped
    configuration (RunPod) sends none: vLLM's key is the only one its candidates use."""
    config = json.loads(json.dumps(runner.load_config(paths.ROOT / "config.yaml")))
    assert config["roles"]["slm_candidates"][0]["endpoint"]["headers_env"] is None
    assert set(runner.used_key_envs(config, ["slm:qwen3-8b+lora:c0-aa"], retrieval=False)) == {"SLM_VLLM_API_KEY"}
    for entry in config["roles"]["slm_candidates"]:  # on Modal, behind its proxy auth
        entry["endpoint"]["headers_env"] = {"Modal-Key": "SLM_MODAL_KEY", "Modal-Secret": "SLM_MODAL_SECRET"}
    monkeypatch.setenv("SLM_MODAL_KEY", "wk-modal-key-0001")
    monkeypatch.setenv("SLM_MODAL_SECRET", "ws-modal-secret-0002")
    assert registry.redact("proxy refused wk-modal-key-0001 / ws-modal-secret-0002", config) == \
        "proxy refused <redacted> / <redacted>"
    used = runner.used_key_envs(config, ["slm:qwen3-8b+lora:c0-aa"], retrieval=False)
    assert set(used) == {"SLM_VLLM_API_KEY", "SLM_MODAL_KEY", "SLM_MODAL_SECRET"}
    monkeypatch.setenv("SLM_MODAL_KEY", "wk-modal-key-0001")
    monkeypatch.setenv("SLM_MODAL_SECRET", "wk-modal-key-0001-and-more")  # one value containing another
    assert registry.redact("secret wk-modal-key-0001-and-more", config) == "secret <redacted>"
    monkeypatch.setenv("SLM_MODAL_SECRET", "short")
    with pytest.raises(barrier.TestSplitLocked, match="SLM_MODAL_SECRET"):
        registry.check_keys(used)


def test_a_committed_manifest_never_echoes_a_providers_key(repo, monkeypatch):
    import httpx
    import openai
    root, config_path = repo
    register(root, config_path)
    body = "Incorrect API key provided: sk-proj-abc1***wxyz"

    class Refusing(ScriptedChess):
        def invoke(self, messages):
            raise openai.AuthenticationError(body, response=httpx.Response(401, request=httpx.Request("POST", "http://x")), body=None)
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: Refusing())
    run_dir = runner.run_agent(str(config_path), "B0", "test")
    local = (run_dir / "manifest.json").read_text()  # no manifest keeps a credential, the local one included
    assert "sk-proj" not in local and "<redacted>" in local and "Incorrect API key provided" in local
    committed = git(root, "show", f"HEAD:registry/test/{run_dir.name}.manifest.json")
    assert "sk-proj" not in committed and "<redacted>" in committed


def test_a_test_run_waits_until_the_registry_is_pushed(repo):
    root, config_path = repo
    register(root, config_path)
    (root / "registry" / "note.txt").write_text("a registry commit only this clone has\n")
    git(root, "add", "registry/note.txt")
    git(root, "commit", "-q", "-m", "unpushed registry commit")
    with pytest.raises(barrier.TestSplitLocked, match="push the registry first"):
        runner.run_agent(str(config_path), "B0", "test")
    assert not any(p.name.startswith("agent-B0-test") for p in paths.RUNS.iterdir())
    git(root, "push", "-q", "origin", "main")
    run_dir = runner.run_agent(str(config_path), "B0", "test")
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "done"


def test_a_ctrl_c_commits_the_manifest_as_interrupted(repo, monkeypatch):
    root, config_path = repo
    register(root, config_path)

    class Interrupting(ScriptedChess):
        def invoke(self, messages):
            if messages[-1].content.startswith("<system>"):
                raise KeyboardInterrupt  # on the agent's own call, in the main thread, as a Ctrl-C would
            return super().invoke(messages)
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: Interrupting())
    with pytest.raises(KeyboardInterrupt):
        runner.run_agent(str(config_path), "B0", "test")
    (run_dir,) = [p for p in paths.RUNS.iterdir() if p.name.startswith("agent-B0-test")]
    committed = json.loads(git(root, "show", f"HEAD:registry/test/{run_dir.name}.manifest.json"))
    assert (committed["status"], committed["stopped_by"]) == ("interrupted", "KeyboardInterrupt: ")
    assert committed["end_reason"] == "KeyboardInterrupt: "  # why, and what it had answered so far
    assert committed["predictions_sha256"] == sha(run_dir / "predictions.json")


def test_a_key_too_short_to_redact_keeps_a_test_run_from_starting(repo, monkeypatch):
    import yaml
    root, config_path = repo
    register(root, config_path)
    config = yaml.safe_load(config_path.read_text())
    config["roles"]["production_llm"]["endpoint"]["api_key_env"] = "BENCH_TEST_PROVIDER_KEY"
    config_path.write_text(yaml.safe_dump(config))
    git(root, "commit", "-q", "-am", "the production LLM needs a key")  # (the barrier then refuses: re-register)
    publish_prereg(root, runner.load_config(config_path))
    monkeypatch.setenv("BENCH_TEST_PROVIDER_KEY", "abc12")  # a key, and too short to redact without damage
    with pytest.raises(barrier.TestSplitLocked, match="BENCH_TEST_PROVIDER_KEY is shorter than 8"):
        runner.run_agent(str(config_path), "B0", "test")
    assert not (root / "registry" / "test").exists()
    monkeypatch.setenv("BENCH_TEST_PROVIDER_KEY", "a-long-enough-provider-key")
    monkeypatch.setenv("OPENAI_API_KEY", "1")  # set, but this run never uses it (fake embeddings): no refusal
    run_dir = runner.run_agent(str(config_path), "B0", "test")
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "done"


def test_a_second_completed_run_of_one_configuration_is_refused_before_its_intent(repo, monkeypatch):
    """SPEC 6.1: one completed run per configuration (type, arm, engine, pre-registration), as the report
    keys it. A run that did not complete may be run again; once one is done, another is refused before
    its intent is committed and before any model call. Another configuration still runs."""
    root, config_path = repo
    register(root, config_path)

    class Failing(ScriptedChess):
        def invoke(self, messages):
            raise OSError("the network went away")
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: Failing())
    failed = runner.run_agent(str(config_path), "B0", "test")
    assert json.loads((failed / "manifest.json").read_text())["status"] == "failed"
    git(root, "push", "-q", "origin", "main")
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    done = runner.run_agent(str(config_path), "B0", "test", resume_from=failed.name)  # after a failure: resumed
    assert json.loads((done / "manifest.json").read_text())["status"] == "done"
    git(root, "push", "-q", "origin", "main")
    before = git(root, "rev-parse", "HEAD").strip()
    calls = []

    class Counting(ScriptedChess):
        def invoke(self, messages):
            calls.append(1)
            return super().invoke(messages)
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: Counting())
    runs_before = set(paths.RUNS.iterdir())
    with pytest.raises(barrier.TestSplitLocked, match=f"{done.name} already completed it"):
        runner.run_agent(str(config_path), "B0", "test")
    assert calls == [] and set(paths.RUNS.iterdir()) == runs_before
    assert git(root, "rev-parse", "HEAD").strip() == before  # no intent committed
    other = runner.run_agent(str(config_path), "B2", "test", engine="production_llm")
    assert json.loads((other / "manifest.json").read_text())["status"] == "done"


def test_done_runs_are_matched_by_type_arm_engine_and_registration(repo):
    root, _ = repo
    (root / registry.TEST_DIR).mkdir(parents=True)
    entries = {
        "a": {"type": "agent", "arm": "B2", "engine": "cheap_alt", "prereg_hash": "h1", "status": "done"},
        "b": {"type": "agent", "arm": "B2", "engine": "production_llm", "prereg_hash": "h1", "status": "done"},
        "c": {"type": "agent", "arm": "B2", "engine": "cheap_alt", "prereg_hash": "h0", "status": "done"},
        "d": {"type": "agent", "arm": "B2", "engine": "cheap_alt", "prereg_hash": "h1", "status": "failed"},
        "e": {"type": "replay", "arm": "B2", "engine": "cheap_alt", "prereg_hash": "h1", "status": "done"},
    }
    for run_id, entry in entries.items():
        (root / registry.TEST_DIR / f"{run_id}.manifest.json").write_text(json.dumps({"run_id": run_id, **entry}))
    (root / registry.TEST_DIR / "f.intent.json").write_text(json.dumps({"run_id": "f", **entries["a"]}))
    git(root, "add", "registry")
    git(root, "commit", "-q", "-m", "entries")
    assert registry.done_runs_of("agent", "B2", "cheap_alt", "h1", "HEAD") == ["a"]
    assert registry.done_runs_of("agent", "B0", None, "h1", "HEAD") == []
    assert registry.done_runs_of("replay", "B2", "cheap_alt", "h1", "HEAD") == ["e"]


def test_a_registry_commit_waits_for_another_gits_lock(repo, monkeypatch):
    """A git process that holds the index for a moment (an editor, a `git status` elsewhere) must not
    fail a run that finished: the commit waits for the lock, a bounded number of times."""
    import threading
    root, _ = repo
    lock = root / ".git" / "index.lock"
    lock.write_text("")
    threading.Timer(0.3, lock.unlink).start()
    registry._write("registry/note.json", {"a": 1})
    registry._commit("registry/note.json", "a note")
    assert git(root, "log", "-1", "--format=%s").strip() == "a note"
    assert git(root, "show", "--name-only", "--format=", "HEAD").split() == ["registry/note.json"]


def test_a_lock_that_outlasts_the_waits_is_still_an_error(repo, monkeypatch):
    root, _ = repo
    waits = []
    monkeypatch.setattr(registry.time, "sleep", waits.append)
    (root / ".git" / "index.lock").write_text("")
    registry._write("registry/note.json", {"a": 1})
    with pytest.raises(registry.RegistryError, match="index.lock"):
        registry._commit("registry/note.json", "a note")
    assert waits == list(registry.LOCK_WAITS_S)  # bounded: every wait once, then the error
    (root / ".git" / "index.lock").unlink()


def test_a_lock_at_the_end_of_a_test_run_leaves_it_done(repo, monkeypatch):
    root, config_path = repo
    register(root, config_path)
    real, locked = registry._git, []

    def busy_once(*args):
        if args[:1] == ("commit",) and "manifest of test execution" in " ".join(args) and not locked:
            locked.append(args)
            return subprocess.CompletedProcess(args, 128, "", "fatal: Unable to create '/x/.git/index.lock': File exists.")
        return real(*args)
    monkeypatch.setattr(registry, "_git", busy_once)
    monkeypatch.setattr(registry.time, "sleep", lambda s: None)
    run_dir = runner.run_agent(str(config_path), "B0", "test")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert locked and manifest["status"] == "done" and manifest["problems"] == []
    assert json.loads(git(root, "show", f"HEAD:registry/test/{run_dir.name}.manifest.json"))["status"] == "done"


# ---------------------------------------------------------------- failures on test: resume, redo, the record

class NetworkDown(ScriptedChess):
    def invoke(self, messages):
        raise OSError("the network went away")


def failed_test_run(root, config_path, monkeypatch, **kwargs):
    """A test run that fails (its harness: the network), committed and pushed as the RUNBOOK's loop does."""
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: NetworkDown())
    run_dir = runner.run_agent(str(config_path), "B0", "test", **kwargs)
    git(root, "push", "-q", "origin", "main")
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    return run_dir


def test_an_execution_that_did_not_end_records_why_and_what_it_had_produced(repo, monkeypatch):
    root, config_path = repo
    register(root, config_path)
    run_dir = failed_test_run(root, config_path, monkeypatch)
    committed = json.loads(git(root, "show", f"HEAD:registry/test/{run_dir.name}.manifest.json"))
    assert committed["status"] == "failed" and "the network went away" in committed["end_reason"]
    assert committed["predictions_sha256"] == sha(run_dir / "predictions.json")
    assert committed["calls_sha256"] == sha(run_dir / "calls.jsonl")
    assert committed["earlier_attempts"] == [] and committed["redo_reason"] is None
    done = runner.run_agent(str(config_path), "B0", "train", ids=["1"])
    assert json.loads((done / "manifest.json").read_text())["end_reason"] is None


def test_a_failed_test_run_is_resumed_under_the_fixed_rule(repo, monkeypatch):
    root, config_path = repo
    register(root, config_path)
    source = failed_test_run(root, config_path, monkeypatch)
    run_dir = runner.run_agent(str(config_path), "B0", "test", resume_from=source.name)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    source_manifest = json.loads((source / "manifest.json").read_text())
    assert manifest["status"] == "done"
    resumed = manifest["resumed_from"]
    assert (resumed["run_id"], resumed["prereg_hash"], resumed["dirty"]) == (source.name, manifest["prereg_hash"], False)
    assert resumed["predictions_sha256"] == source_manifest["predictions_sha256"]
    assert resumed["end_reason"] == source_manifest["end_reason"]
    assert [(a["run_id"], a["status"]) for a in manifest["earlier_attempts"]] == [(source.name, "failed")]
    intent = json.loads(git(root, "show", f"HEAD:registry/test/{run_dir.name}.intent.json"))
    assert (intent["resumed_from"], intent["earlier_attempts"], intent["redo_reason"]) == (source.name, [source.name], None)
    assert len(manifest["sessions"]) == 2 and not manifest["dirty"]


def refused_resume(root, config_path, source, match, **kwargs):
    before = git(root, "rev-parse", "HEAD").strip()
    runs_before = set(paths.RUNS.iterdir())
    with pytest.raises(barrier.TestSplitLocked, match=match):
        runner.run_agent(str(config_path), "B0", "test", resume_from=source, **kwargs)
    assert git(root, "rev-parse", "HEAD").strip() == before and set(paths.RUNS.iterdir()) == runs_before  # no intent, no run


def test_a_test_resume_is_refused_when_the_source_outputs_changed_after_it_ended(repo, monkeypatch):
    root, config_path = repo
    register(root, config_path)
    source = failed_test_run(root, config_path, monkeypatch)
    (source / "predictions.json").write_text("{}")
    refused_resume(root, config_path, source.name, "predictions.json does not hash")


def test_a_test_resume_is_refused_when_the_local_manifest_is_not_the_published_one(repo, monkeypatch):
    root, config_path = repo
    register(root, config_path)
    source = failed_test_run(root, config_path, monkeypatch)
    path = source / "manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "dirty": False, "workers": 3}))
    refused_resume(root, config_path, source.name, "differs from the one the registry published \\(workers\\)")


def test_a_resumable_attempt_is_resumed_and_redone_only_for_a_defect_in_what_it_completed(repo, monkeypatch):
    """A redo is no alternative to a resume: an attempt that can be resumed is redone only when the defect
    affects questions it completed, and only once DEVIATIONS.md names it. Only the latest is resumed."""
    root, config_path = repo
    register(root, config_path)
    first = failed_test_run(root, config_path, monkeypatch)
    with pytest.raises(barrier.TestSplitLocked, match=f"{first.name} already attempted it"):
        runner.run_agent(str(config_path), "B0", "test")
    with pytest.raises(barrier.TestSplitLocked, match=f"can be resumed. Resume it with --resume-from {first.name}"):
        runner.run_agent(str(config_path), "B0", "test", redo_reason="the endpoint was misconfigured")
    with pytest.raises(barrier.TestSplitLocked, match="has no line naming"):
        runner.run_agent(str(config_path), "B0", "test", redo_reason="the endpoint was misconfigured",
                         redo_affects_completed=True)
    deviation(root, first.name, "the endpoint was misconfigured, which every completed question went through")
    second = failed_test_run(root, config_path, monkeypatch, redo_reason="  the endpoint\nwas misconfigured ",
                             redo_affects_completed=True)
    manifest = json.loads((second / "manifest.json").read_text())
    assert manifest["redo_reason"] == "the endpoint was misconfigured"  # one line
    assert manifest["redo"] == {"of": first.name, "affects_completed": True, "not_resumable_because": None}
    assert [a["run_id"] for a in manifest["earlier_attempts"]] == [first.name]
    intent = json.loads(git(root, "show", f"origin/main:registry/test/{second.name}.intent.json"))
    assert intent["redo_reason"] == "the endpoint was misconfigured" and intent["redo"]["affects_completed"] is True
    refused_resume(root, config_path, first.name, f"not the latest attempt of its configuration \\({second.name} is\\)")
    run_dir = runner.run_agent(str(config_path), "B0", "test", resume_from=second.name)
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "done"
    # both earlier attempts stay in the registry
    listed = git(root, "ls-files", "registry/test").split()
    assert {f"registry/test/{r.name}.manifest.json" for r in (first, second, run_dir)} <= set(listed)


def test_a_redo_reason_needs_an_earlier_attempt_on_test_and_excludes_a_resume(repo, monkeypatch):
    from bench.data import DataError
    root, config_path = repo
    register(root, config_path)
    with pytest.raises(barrier.TestSplitLocked, match="nothing to run again"):
        runner.run_agent(str(config_path), "B0", "test", redo_reason="a defect")
    with pytest.raises(DataError, match="for the test split"):
        runner.run_agent(str(config_path), "B0", "train", ids=["1"], redo_reason="a defect")
    source = failed_test_run(root, config_path, monkeypatch)
    with pytest.raises(DataError, match="choose one"):
        runner.run_agent(str(config_path), "B0", "test", resume_from=source.name, redo_reason="a defect")
    with pytest.raises(DataError, match="qualifies --redo-reason"):
        runner.run_agent(str(config_path), "B0", "test", redo_affects_completed=True)
    assert not (root / "registry" / "test" / "x").exists()


def test_a_test_resume_is_refused_unless_the_source_ran_clean_at_the_registered_code(repo, monkeypatch):
    """The source's own record decides: under another registration, on a dirty tree, or at other code."""
    from bench.agent import resume
    root, config_path = repo
    register(root, config_path)
    source_dir = failed_test_run(root, config_path, monkeypatch)
    source = json.loads((source_dir / "manifest.json").read_text())
    attempts = [{"run_id": source["run_id"], "started_at": source["started_at"], "status": "failed"}]
    in_force = source["prereg_hash"]
    resume.check_test_source(source, in_force, attempts, source)  # as it is: allowed
    git(root, "checkout", "-q", "-b", "elsewhere")
    (root / "bench" / "agent" / "code.py").write_text("x = 2\n")
    git(root, "commit", "-q", "-am", "other code")
    other_code = git(root, "rev-parse", "HEAD").strip()
    git(root, "checkout", "-q", "main")
    for changed, match in (({"prereg_hash": "h0"}, "under pre-registration h0"),
                           ({"dirty": True}, "not clean"),
                           ({"commit": other_code}, "not at the registered code")):
        edited = {**source, **changed}
        with pytest.raises(barrier.TestSplitLocked, match=match):
            resume.check_test_source(edited, in_force, attempts, edited)
    with pytest.raises(barrier.TestSplitLocked, match="holds no manifest"):
        resume.check_test_source(source, in_force, attempts, None)


def test_a_test_execution_is_refused_on_code_changed_after_the_registration(repo, monkeypatch):
    root, config_path = repo
    register(root, config_path)
    (root / "bench" / "agent" / "code.py").write_text("x = 2\n")
    with pytest.raises(barrier.TestSplitLocked, match="not clean \\(M bench/agent/code.py\\)"):
        runner.run_agent(str(config_path), "B0", "test")
    git(root, "commit", "-q", "-am", "a fix after the registration")
    git(root, "push", "-q", "origin", "main")
    with pytest.raises(barrier.TestSplitLocked, match="differs from the registered commit .* bench/agent/code.py"):
        runner.run_agent(str(config_path), "B0", "test")
    assert not (root / "registry" / "test").exists()


class Killed(BaseException):
    """Stands for SIGKILL: nothing after it runs in the process."""


def test_a_test_session_killed_outright_is_resumed_with_its_outputs_hashed_first(repo, monkeypatch):
    import os
    root, config_path = repo
    register(root, config_path)
    real_write_json = runner._write_json

    def write_json(path, value):
        if path.name == "predictions.json":
            os.close(runner._SESSION_LOCKS.pop(path.parent))  # the dying process's lock goes with it
            raise Killed()
        real_write_json(path, value)
    monkeypatch.setattr(runner, "_write_json", write_json)
    with pytest.raises(Killed):
        runner.run_agent(str(config_path), "B0", "test")
    monkeypatch.setattr(runner, "_write_json", real_write_json)
    (source,) = [p for p in paths.RUNS.iterdir() if p.name.startswith("agent-B0-test")]
    published = json.loads(git(root, "show", f"origin/main:registry/test/{source.name}.manifest.json"))
    assert published["status"] == "running"  # all the registry knows of a killed session
    hashes = {p.name: sha(p) for p in sorted(source.glob("*.jsonl"))}
    assert set(hashes) == {"answers.jsonl", "calls.jsonl"}

    with pytest.raises(barrier.TestSplitLocked, match="already attempted it"):
        runner.run_agent(str(config_path), "B0", "test")  # a redo still needs its reason
    run_dir = runner.run_agent(str(config_path), "B0", "test", resume_from=source.name)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "done" and manifest["resumed_question_ids"] == ["9"]
    assert manifest["unregistered_source_hashes"] == hashes
    intent = json.loads(git(root, "show", f"HEAD:registry/test/{run_dir.name}.intent.json"))
    assert intent["unregistered_source_hashes"] == hashes and intent["resumed_from"] == source.name
    assert [(a["run_id"], a["status"]) for a in manifest["earlier_attempts"]] == [(source.name, "running")]


# ---------------------------------------------------------------- a replay on test, end to end

@pytest.fixture
def failed_replay(repo, monkeypatch):
    """A done B0 run on test, and a replay of it on test that failed on the harness (the network),
    both published as the RUNBOOK's loop publishes them."""
    from bench.agent import replay
    root, config_path = repo
    register(root, config_path)
    source = runner.run_agent(str(config_path), "B0", "test")
    git(root, "push", "-q", "origin", "main")
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: NetworkDown())
    failed = replay.replay(str(config_path), source.name, engine="production_llm")
    git(root, "push", "-q", "origin", "main")
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    manifest = json.loads((failed / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["harness_errors"] and manifest["calls_sha256"]
    return root, config_path, source.name, failed


def refused_replay(root, config_path, source, failed, match, **kwargs):
    from bench.agent import replay
    before, runs_before = git(root, "rev-parse", "HEAD").strip(), set(paths.RUNS.iterdir())
    with pytest.raises(barrier.TestSplitLocked, match=match):
        replay.replay(str(config_path), source, engine="production_llm", resume_from=failed, **kwargs)
    assert git(root, "rev-parse", "HEAD").strip() == before and set(paths.RUNS.iterdir()) == runs_before


def test_a_failed_replay_on_test_is_resumed_under_the_fixed_rule(failed_replay):
    from bench.agent import replay
    root, config_path, source, failed = failed_replay
    run_dir = replay.replay(str(config_path), source, engine="production_llm", resume_from=failed.name)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "done" and manifest["resumed_from"]["run_id"] == failed.name
    assert manifest["resumed_from"]["calls_sha256"] == json.loads((failed / "manifest.json").read_text())["calls_sha256"]
    assert [(a["run_id"], a["status"]) for a in manifest["earlier_attempts"]] == [(failed.name, "failed")]
    intent = json.loads(git(root, "show", f"HEAD:registry/test/{run_dir.name}.intent.json"))
    assert intent["resumed_from"] == failed.name and intent["type"] == "replay"


def test_a_replay_on_test_is_not_resumed_when_its_calls_changed(failed_replay):
    root, config_path, source, failed = failed_replay
    with open(failed / "calls.jsonl", "a") as fh:
        fh.write("\n")
    refused_replay(root, config_path, source, failed.name, "calls.jsonl does not hash")


def test_a_replay_on_test_is_not_resumed_when_its_local_manifest_is_not_the_published_one(failed_replay):
    root, config_path, source, failed = failed_replay
    path = failed / "manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "workers": 2}))
    refused_replay(root, config_path, source, failed.name, "differs from the one the registry published \\(workers\\)")


@pytest.mark.parametrize("field, value, match", [
    ("prereg_hash", "0" * 64, "under pre-registration 0000"),
    ("dirty", True, "not clean"),
    ("commit", "0" * 40, "not at the registered code"),
])
def test_a_replay_on_test_is_not_resumed_from_a_source_outside_the_rule(failed_replay, field, value, match):
    """Its record, local and published alike, says it ran under another registration, on a dirty tree, or at
    other code."""
    root, config_path, source, failed = failed_replay
    rel = f"registry/test/{failed.name}.manifest.json"
    edited = {**json.loads((failed / "manifest.json").read_text()), field: value}
    for path in (failed / "manifest.json", root / rel):
        path.write_text(json.dumps(edited, indent=2) + "\n")
    git(root, "commit", "-q", "-am", "a record edited for the test")
    git(root, "push", "-q", "origin", "main")
    refused_replay(root, config_path, source, failed.name, match)


def test_a_replay_on_test_is_redone_only_with_a_reason_and_only_the_latest_is_resumed(failed_replay, monkeypatch):
    from bench.agent import replay
    root, config_path, source, failed = failed_replay
    with pytest.raises(barrier.TestSplitLocked, match=f"{failed.name} already attempted it"):
        replay.replay(str(config_path), source, engine="production_llm")
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: NetworkDown())
    with pytest.raises(barrier.TestSplitLocked, match=f"Resume it with --resume-from {failed.name}"):
        replay.replay(str(config_path), source, engine="production_llm", redo_reason="the endpoint was wrong")
    deviation(root, failed.name, "the endpoint was wrong for every invocation it completed")
    second = replay.replay(str(config_path), source, engine="production_llm", redo_reason="the endpoint was wrong",
                           redo_affects_completed=True)
    git(root, "push", "-q", "origin", "main")
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    assert json.loads((second / "manifest.json").read_text())["redo_reason"] == "the endpoint was wrong"
    refused_replay(root, config_path, source, failed.name, f"not the latest attempt of its configuration \\({second.name} is\\)")
    done = replay.replay(str(config_path), source, engine="production_llm", resume_from=second.name)
    assert json.loads((done / "manifest.json").read_text())["status"] == "done"


def test_a_replay_on_test_is_not_resumed_on_a_dirty_tree(failed_replay):
    root, config_path, source, failed = failed_replay
    (root / "notes.txt").write_text("not committed\n")
    refused_replay(root, config_path, source, failed.name, "not clean \\(\\?\\? notes.txt\\)")


def test_an_attempt_that_cannot_be_resumed_is_redone_with_its_deviation_and_why(repo, monkeypatch):
    """Outputs that no longer hash to their record make resuming impossible: a redo, with no claim about
    completed questions, once DEVIATIONS.md names the attempt; the manifest says why it was not resumed."""
    root, config_path = repo
    register(root, config_path)
    first = failed_test_run(root, config_path, monkeypatch)
    (first / "calls.jsonl").write_text("")
    refused_resume(root, config_path, first.name, "calls.jsonl does not hash")
    deviation(root, first.name, "its calls.jsonl was lost")
    run_dir = runner.run_agent(str(config_path), "B0", "test", redo_reason="its calls.jsonl was lost")
    redo = json.loads((run_dir / "manifest.json").read_text())["redo"]
    assert (redo["of"], redo["affects_completed"]) == (first.name, False)
    assert "calls.jsonl does not hash" in redo["not_resumable_because"]
