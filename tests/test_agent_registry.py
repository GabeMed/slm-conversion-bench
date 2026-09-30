"""The test registry (REQ-013) and the call-site assertion (REQ-001; SPEC 7.1), on a temporary git
repository with a published pre-registration: an execution on `test` commits its intent before
anything runs and its manifest when it ends, never anything else; it runs only when the call sites
of train and calib are registered and committed, and a call site outside them aborts it."""
import hashlib
import json
import subprocess

import pytest

pytest.importorskip("langchain_core")
import httpx  # noqa: E402
import openai  # noqa: E402

from bench import barrier, paths  # noqa: E402
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
    (root / "prereg").mkdir()
    prereg = {"spec_sha256": sha(root / "SPEC.md"), "config_sha256": config_sha256(config),
              "splits_sha256": sha(paths.SPLITS), "data_manifest_sha256": sha(paths.DATA_MANIFEST), "commit": "c"}
    (root / "prereg" / "manifest.json").write_text(json.dumps(prereg, sort_keys=True) + "\n")
    (root / "prereg" / "HASH").write_text(sha(root / "prereg" / "manifest.json") + "\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    git(root, "push", "-q", "origin", "main")
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    return root, config_path


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
    run_dir = runner.run_agent(str(config_path), "B0", "test", ids=["9"])
    intent, manifest_rel = f"registry/test/{run_dir.name}.intent.json", f"registry/test/{run_dir.name}.manifest.json"
    assert seen_while_running == [[intent]]  # a run that dies from here on leaves an intent with no manifest
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


def test_a_single_call_execution_on_test_is_registered_too(repo):
    root, config_path = repo
    register(root, config_path)
    run_dir = runner.run_agent(str(config_path), "B2", "test", ids=["9"], engine="production_llm")
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "done"
    assert sorted(git(root, "ls-files", "registry/test").split()) == [
        f"registry/test/{run_dir.name}.intent.json", f"registry/test/{run_dir.name}.manifest.json"]


def test_a_call_site_outside_the_registered_set_aborts_the_run(repo):
    root, config_path = repo
    register(root, config_path, call_sites=[s for s in CALL_SITES if s != "select_columns"])
    run_dir = runner.run_agent(str(config_path), "B0", "test", ids=["9"])
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
        runner.run_agent(str(config_path), "B0", "test", ids=["9"])
    register(root, config_path, commit=False)
    with pytest.raises(barrier.TestSplitLocked, match="not committed"):
        runner.run_agent(str(config_path), "B0", "test", ids=["9"])
    git(root, "add", registry.CALL_SITES_FILE)
    git(root, "commit", "-q", "-m", "register")
    (root / registry.CALL_SITES_FILE).write_text('{"call_sites": ["agent_ir"]}')
    with pytest.raises(barrier.TestSplitLocked, match="uncommitted"):
        runner.run_agent(str(config_path), "B0", "test", ids=["9"])
    assert not (root / "registry" / "test").exists()
    assert all(not p.name.startswith("agent-B0-test") for p in paths.RUNS.iterdir())


def test_only_done_train_and_calib_runs_register_call_sites(repo):
    root, config_path = repo
    train = register(root, config_path)
    registered = json.loads((root / registry.CALL_SITES_FILE).read_text())
    assert registered["call_sites"] == sorted(set(CALL_SITES) - {"revise"})  # what the run called, nothing more
    assert registered["sources"][train.name]["split"] == "train"
    test_run = runner.run_agent(str(config_path), "B0", "test", ids=["9"])
    with pytest.raises(registry.RegistryError, match="only train and calib"):
        registry.update_call_sites([test_run.name])
    manifest = json.loads((train / "manifest.json").read_text())
    (train / "manifest.json").write_text(json.dumps({**manifest, "status": "failed"}))
    with pytest.raises(registry.RegistryError, match="not 'done'"):
        registry.update_call_sites([train.name])


def test_the_registry_commits_its_file_and_nothing_else(repo):
    root, config_path = repo
    register(root, config_path)
    (root / "notes.txt").write_text("tracked\n")
    git(root, "add", "notes.txt")
    git(root, "commit", "-q", "-m", "notes")
    (root / "notes.txt").write_text("changed, not committed\n")
    (root / "staged.txt").write_text("staged, not committed\n")
    git(root, "add", "staged.txt")
    before = git(root, "rev-parse", "HEAD").strip()
    run_dir = runner.run_agent(str(config_path), "B0", "test", ids=["9"])
    assert git(root, "diff", "--name-only", before, "HEAD").split() == [
        f"registry/test/{run_dir.name}.intent.json", f"registry/test/{run_dir.name}.manifest.json"]
    status = git(root, "status", "--porcelain", "--", "notes.txt", "staged.txt").splitlines()
    assert sorted(status) == [" M notes.txt", "A  staged.txt"]  # left exactly as they were


def test_a_manifest_the_registry_cannot_commit_fails_the_run(repo, monkeypatch):
    root, config_path = repo
    register(root, config_path)

    def refused(run_dir, config):
        raise registry.RegistryError("git commit failed: signing required")
    monkeypatch.setattr(registry, "commit_manifest", refused)
    with pytest.raises(hooks.HarnessError, match="signing required"):  # a harness error: the CLI reports it
        runner.run_agent(str(config_path), "B0", "test", ids=["9"])
    (run_dir,) = [p for p in paths.RUNS.iterdir() if p.name.startswith("agent-B0-test")]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "registry" in manifest["problems"][-1]
    from bench.data import DataError
    from bench.evaluate import evaluate
    with pytest.raises(DataError, match="not 'done'"):
        evaluate(run_dir.name)  # never scored while the registry shows only its intent


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
    """`headers_env` values (the Modal proxy token of config.yaml's candidates) are credentials like any key:
    redacted from the public record, and checked for length before a test execution."""
    config = json.loads(json.dumps(runner.load_config(paths.ROOT / "config.yaml")))
    headers = config["roles"]["slm_candidates"][0]["endpoint"]["headers_env"]
    assert set(headers.values()) == {"SLM_MODAL_KEY", "SLM_MODAL_SECRET"}
    monkeypatch.setenv("SLM_MODAL_KEY", "wk-modal-key-0001")
    monkeypatch.setenv("SLM_MODAL_SECRET", "ws-modal-secret-0002")
    assert registry.redact("proxy refused wk-modal-key-0001 / ws-modal-secret-0002", config) == \
        "proxy refused <redacted> / <redacted>"
    used = runner.used_key_envs(config, ["slm:qwen3-8b+lora:c0-aa"], retrieval=False)
    assert set(used) == {"SLM_VLLM_API_KEY", "SLM_MODAL_KEY", "SLM_MODAL_SECRET"}
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
    run_dir = runner.run_agent(str(config_path), "B0", "test", ids=["9"])
    assert "sk-proj-abc1" in (run_dir / "manifest.json").read_text()  # the local record keeps the provider's words
    committed = git(root, "show", f"HEAD:registry/test/{run_dir.name}.manifest.json")
    assert "sk-proj" not in committed and "<redacted>" in committed


def test_a_test_run_waits_until_the_registry_is_pushed(repo):
    root, config_path = repo
    register(root, config_path)
    (root / "registry" / "note.txt").write_text("a registry commit only this clone has\n")
    git(root, "add", "registry/note.txt")
    git(root, "commit", "-q", "-m", "unpushed registry commit")
    with pytest.raises(barrier.TestSplitLocked, match="push the registry first"):
        runner.run_agent(str(config_path), "B0", "test", ids=["9"])
    assert not any(p.name.startswith("agent-B0-test") for p in paths.RUNS.iterdir())
    git(root, "push", "-q", "origin", "main")
    run_dir = runner.run_agent(str(config_path), "B0", "test", ids=["9"])
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
        runner.run_agent(str(config_path), "B0", "test", ids=["9"])
    (run_dir,) = [p for p in paths.RUNS.iterdir() if p.name.startswith("agent-B0-test")]
    committed = json.loads(git(root, "show", f"HEAD:registry/test/{run_dir.name}.manifest.json"))
    assert (committed["status"], committed["stopped_by"]) == ("interrupted", "KeyboardInterrupt: ")


def test_a_key_too_short_to_redact_keeps_a_test_run_from_starting(repo, monkeypatch):
    import yaml
    root, config_path = repo
    register(root, config_path)
    config = yaml.safe_load(config_path.read_text())
    config["roles"]["production_llm"]["endpoint"]["api_key_env"] = "BENCH_TEST_PROVIDER_KEY"
    config_path.write_text(yaml.safe_dump(config))
    git(root, "commit", "-q", "-am", "the production LLM needs a key")  # (the barrier then refuses: re-register)
    prereg = json.loads((root / "prereg" / "manifest.json").read_text())
    prereg["config_sha256"] = runner.config_sha256(runner.load_config(config_path))
    (root / "prereg" / "manifest.json").write_text(json.dumps(prereg, sort_keys=True) + "\n")
    (root / "prereg" / "HASH").write_text(sha(root / "prereg" / "manifest.json") + "\n")
    git(root, "commit", "-q", "-am", "re-register")
    git(root, "push", "-q", "origin", "main")
    monkeypatch.setenv("BENCH_TEST_PROVIDER_KEY", "abc12")  # a key, and too short to redact without damage
    with pytest.raises(barrier.TestSplitLocked, match="BENCH_TEST_PROVIDER_KEY is shorter than 8"):
        runner.run_agent(str(config_path), "B0", "test", ids=["9"])
    assert not (root / "registry" / "test").exists()
    monkeypatch.setenv("BENCH_TEST_PROVIDER_KEY", "a-long-enough-provider-key")
    monkeypatch.setenv("OPENAI_API_KEY", "1")  # set, but this run never uses it (fake embeddings): no refusal
    run_dir = runner.run_agent(str(config_path), "B0", "test", ids=["9"])
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "done"
