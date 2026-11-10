"""The test barrier: the test split opens only when the pre-registration is published on origin/main
and the execution uses exactly what was registered (configuration, splits, SPEC, data manifest)."""
import hashlib
import json
import re
import subprocess

import pytest

from bench import paths
from bench.barrier import TestSplitLocked, ensure_split_allowed, prereg_published
from bench.contracts.config import config_sha256, load_config

CONFIG = load_config(paths.ROOT / "config.yaml")


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clone(tmp_path, origin, name):
    path = tmp_path / name
    git(tmp_path, "clone", "-q", str(origin), str(path))
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "t")
    return path


@pytest.fixture
def repo(tmp_path):
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    repo = clone(tmp_path, origin, "clone")
    (repo / "data").mkdir()
    (repo / "SPEC.md").write_text("protocol\n")
    (repo / "data" / "splits.json").write_text('{"test": ["9"]}\n')
    (repo / "data" / "MANIFEST.json").write_text('{"databases": {}}\n')
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    git(repo, "push", "-q", "origin", "main")
    return repo


def register(repo, config=CONFIG, publish=True, **override):
    (repo / "prereg").mkdir(exist_ok=True)
    manifest = {"spec_sha256": sha(repo / "SPEC.md"), "config_sha256": config_sha256(config),
                "splits_sha256": sha(repo / "data" / "splits.json"),
                "data_manifest_sha256": sha(repo / "data" / "MANIFEST.json"), "commit": "c", **override}
    (repo / "prereg" / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    (repo / "prereg" / "HASH").write_text(sha(repo / "prereg" / "manifest.json") + "\n")
    git(repo, "add", "prereg")
    git(repo, "commit", "-q", "-m", "prereg")
    if publish:
        git(repo, "push", "-q", "origin", "main")


def test_opens_only_when_published_and_matching(repo):
    assert "does not exist" in prereg_published(CONFIG, repo)
    register(repo, publish=False)
    assert "not on origin/main" in prereg_published(CONFIG, repo)
    git(repo, "push", "-q", "origin", "main")
    assert prereg_published(CONFIG, repo) == ""
    ensure_split_allowed("test", CONFIG, repo)


def test_an_execution_with_another_configuration_is_refused(repo):
    register(repo)
    other = {**CONFIG, "seeds": {**CONFIG["seeds"], "bootstrap": 1}}
    assert "configuration differs" in prereg_published(other, repo)
    assert "must present its configuration" in prereg_published(None, repo)


@pytest.mark.parametrize("rel", ["SPEC.md", "data/splits.json", "data/MANIFEST.json"])
def test_changed_registered_inputs_are_refused(repo, rel):
    register(repo)
    (repo / rel).write_text("changed after the pre-registration\n")
    git(repo, "commit", "-q", "-am", "change")
    git(repo, "push", "-q", "origin", "main")
    assert f"{rel} differs from the pre-registered one" in prereg_published(CONFIG, repo)


def test_hash_must_be_the_manifests(repo):
    register(repo)
    (repo / "prereg" / "HASH").write_text("0" * 64 + "\n")
    git(repo, "commit", "-q", "-am", "tamper")
    git(repo, "push", "-q", "origin", "main")
    assert "is not the sha256" in prereg_published(CONFIG, repo)


def test_a_hash_that_is_not_text_is_a_refusal_not_a_crash(repo):
    register(repo)
    (repo / "prereg" / "HASH").write_bytes(b"\xff\xfe\n")
    git(repo, "commit", "-q", "-am", "a HASH that is not UTF-8")
    git(repo, "push", "-q", "origin", "main")
    assert "is not the sha256" in prereg_published(CONFIG, repo)


def test_local_changes_and_a_newer_remote_are_refused(repo, tmp_path):
    register(repo)
    (repo / "prereg" / "HASH").write_text("edited\n")
    assert "uncommitted" in prereg_published(CONFIG, repo)
    git(repo, "checkout", "--", "prereg/HASH")
    other = clone(tmp_path, tmp_path / "origin.git", "other")  # someone else re-registers on the remote
    register(other, splits_sha256="0" * 64)
    assert "differs from origin/main" in prereg_published(CONFIG, repo)  # seen only because the barrier fetches


def test_the_remote_is_what_the_fetch_brought_not_a_stale_ref(repo, tmp_path):
    """A clone whose refspec does not map main keeps a stale origin/main after `git fetch origin main`:
    the barrier compares with the fetched commit, so a re-registration on the remote is still seen."""
    register(repo)
    git(repo, "config", "--unset-all", "remote.origin.fetch")
    other = clone(tmp_path, tmp_path / "origin.git", "other")
    register(other, splits_sha256="0" * 64)
    assert "differs from origin/main" in prereg_published(CONFIG, repo)


def test_the_fetch_never_prompts_and_times_out(repo, monkeypatch):
    import bench.barrier as barrier
    register(repo)
    real, seen = subprocess.run, []

    def run(cmd, **kwargs):
        if "fetch" in cmd:
            seen.append((cmd, kwargs))
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        return real(cmd, **kwargs)
    monkeypatch.setattr(barrier.subprocess, "run", run)
    for name in ("GIT_TERMINAL_PROMPT", "GIT_SSH_COMMAND"):  # whatever the shell running the tests exports
        monkeypatch.delenv(name, raising=False)
    assert "timed out" in prereg_published(CONFIG, repo)
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i my-key")  # a user's own ssh command is kept
    assert "timed out" in prereg_published(CONFIG, repo)
    (cmd, first), (_, second) = seen
    assert first["env"]["GIT_TERMINAL_PROMPT"] == "0" and first["timeout"] == barrier.FETCH_TIMEOUT_S
    assert first["env"]["GIT_SSH_COMMAND"] == "ssh -o BatchMode=yes" and second["env"]["GIT_SSH_COMMAND"] == "ssh -i my-key"
    assert cmd[-1] == "refs/heads/main"  # the branch in full: a tag called main would win over a short name


def test_a_failed_fetch_refuses_even_after_one_that_succeeded(repo):
    """The earlier fetch left a FETCH_HEAD: a failed fetch must not compare with it."""
    register(repo)
    assert prereg_published(CONFIG, repo) == ""
    git(repo, "remote", "set-url", "origin", str(repo.parent / "gone.git"))
    assert "could not fetch origin/main" in prereg_published(CONFIG, repo)


def test_a_fetch_that_leaves_no_commit_refuses(repo, monkeypatch):
    """An empty FETCH_HEAD (a concurrent fetch had just emptied it) would read as the index, which holds
    the committed files: the barrier would open. It refuses."""
    import bench.barrier as barrier
    register(repo)
    real = barrier._git
    monkeypatch.setattr(barrier, "_git", lambda root, *args: subprocess.CompletedProcess(args, 0, "", "")
                        if "FETCH_HEAD^{commit}" in args else real(root, *args))
    assert "left no commit to compare with" in prereg_published(CONFIG, repo)


@pytest.mark.parametrize("content", ["[not JSON]\n", "[]\n", "1\n"])
def test_a_malformed_registration_is_a_refusal_not_a_crash(repo, content):
    (repo / "prereg").mkdir()
    (repo / "prereg" / "manifest.json").write_text(content)
    (repo / "prereg" / "HASH").write_text(sha(repo / "prereg" / "manifest.json") + "\n")
    git(repo, "add", "prereg")
    git(repo, "commit", "-q", "-m", "a malformed registration")
    git(repo, "push", "-q", "origin", "main")
    assert "is not a JSON object" in prereg_published(CONFIG, repo)


def test_no_remote_no_test(tmp_path):
    assert "could not fetch" in prereg_published(CONFIG, tmp_path)
    with pytest.raises(TestSplitLocked):
        ensure_split_allowed("test", CONFIG, tmp_path)


def test_other_splits_are_never_locked(tmp_path):
    ensure_split_allowed("train", None, tmp_path)
    ensure_split_allowed("calib", None, tmp_path)


# ---------------------------------------------------------------- an execution on test runs the registered code

def git_out(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def coded(repo):
    """The repository with execution code and a .gitignore, registered at HEAD as `bench prereg` does."""
    from bench.barrier import execution_code
    (repo / ".gitignore").write_text("runs/\n")
    (repo / "bench" / "agent").mkdir(parents=True)
    (repo / "bench" / "agent" / "runner.py").write_text("x = 1\n")
    (repo / "vendor" / "chess").mkdir(parents=True)
    (repo / "vendor" / "chess" / "agent.py").write_text("y = 1\n")
    (repo / "bench" / "judge").mkdir()
    (repo / "bench" / "judge" / "j4.py").write_text("z = 1\n")  # analysis code: not the execution code's
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "code")
    registered = {"commit": git_out(repo, "rev-parse", "HEAD").strip(), "execution_code": execution_code(repo)}
    register(repo, **registered)
    return repo, registered


def test_the_execution_code_is_the_agent_chess_the_serving_plan_and_the_curation(coded):
    from bench.barrier import EXECUTION_CODE
    repo, registered = coded
    assert sorted(registered["execution_code"]) == ["bench/agent/runner.py", "vendor/chess/agent.py"]
    assert registered["execution_code"]["vendor/chess/agent.py"] == sha(repo / "vendor" / "chess" / "agent.py")
    assert {"bench/agent", "vendor/chess", "bench/curate.py", "modal_apps/common.py", "runpod_apps"} <= set(EXECUTION_CODE)


def test_a_clean_checkout_of_the_registered_code_runs_records_after_it_included(coded):
    from bench.barrier import code_in_force, ensure_execution_allowed
    repo, registered = coded
    assert code_in_force(registered, repo) == ""
    (repo / "registry" / "test").mkdir(parents=True)  # what test executions commit after the registration
    (repo / "registry" / "test" / "a.intent.json").write_text("{}\n")
    (repo / "prereg" / "DEVIATIONS.md").write_text("- a line\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "records")
    (repo / "runs").mkdir()
    (repo / "runs" / "out.json").write_text("{}\n")  # ignored: what executions write
    assert code_in_force(registered, repo) == ""
    git(repo, "push", "-q", "origin", "main")
    assert ensure_execution_allowed(CONFIG, repo) == (repo / "prereg" / "HASH").read_text().strip()


@pytest.mark.parametrize("change, expected", [
    (lambda repo: (repo / "bench" / "agent" / "runner.py").write_text("x = 2\n"), "not clean \\(M bench/agent/runner.py\\)"),
    (lambda repo: (repo / "notes.txt").write_text("n\n"), "not clean \\(\\?\\? notes.txt\\)"),
    (lambda repo: ((repo / "judgments").mkdir(), (repo / "judgments" / "j.json").write_text("{}")),
     "not clean \\(\\?\\? judgments/"),  # a judgment the registered commit does not hold
])
def test_a_dirty_tree_is_refused(coded, change, expected):
    from bench.barrier import code_in_force
    repo, registered = coded
    change(repo)
    assert re.search(expected, code_in_force(registered, repo))


def test_code_committed_after_the_registration_is_refused(coded):
    from bench.barrier import code_in_force
    repo, registered = coded
    (repo / "bench" / "judge" / "j4.py").write_text("z = 2\n")
    git(repo, "commit", "-q", "-am", "a later fix")
    assert "differs from the registered commit" in code_in_force(registered, repo)
    assert "bench/judge/j4.py" in code_in_force(registered, repo)


def test_a_commit_that_does_not_descend_from_the_registered_one_is_refused(coded):
    from bench.barrier import code_in_force
    repo, registered = coded
    assert "is not in this repository" in code_in_force({**registered, "commit": "0" * 40}, repo)
    assert "is not in this repository" in code_in_force({**registered, "commit": None}, repo)
    git(repo, "checkout", "-q", "--orphan", "other")
    git(repo, "commit", "-q", "-m", "same files, other history")
    assert "does not descend" in code_in_force(registered, repo)


def test_the_execution_code_must_be_registered_and_match(coded):
    from bench.barrier import code_in_force
    repo, registered = coded
    assert "records no execution code" in code_in_force({"commit": registered["commit"]}, repo)
    tampered = {**registered["execution_code"], "vendor/chess/agent.py": "0" * 64}
    assert "execution code differs" in code_in_force({**registered, "execution_code": tampered}, repo)
    assert "vendor/chess/agent.py" in code_in_force({**registered, "execution_code": tampered}, repo)


def test_python_that_gitignore_hides_under_the_execution_code_is_refused(coded):
    from bench.barrier import code_in_force
    repo, registered = coded
    (repo / ".git" / "info" / "exclude").write_text("bench/agent/hidden.py\n")
    (repo / "bench" / "agent" / "hidden.py").write_text("import os\n")
    assert "ignored by git would run: bench/agent/hidden.py" in code_in_force(registered, repo)
