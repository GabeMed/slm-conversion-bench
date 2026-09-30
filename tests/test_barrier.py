"""The test barrier: the test split opens only when the pre-registration is published on origin/main
and the execution uses exactly what was registered (configuration, splits, SPEC, data manifest)."""
import hashlib
import json
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


def test_local_changes_and_a_newer_remote_are_refused(repo, tmp_path):
    register(repo)
    (repo / "prereg" / "HASH").write_text("edited\n")
    assert "uncommitted" in prereg_published(CONFIG, repo)
    git(repo, "checkout", "--", "prereg/HASH")
    other = clone(tmp_path, tmp_path / "origin.git", "other")  # someone else re-registers on the remote
    register(other, splits_sha256="0" * 64)
    assert "differs from origin/main" in prereg_published(CONFIG, repo)  # seen only because the barrier fetches


def test_no_remote_no_test(tmp_path):
    assert "could not fetch" in prereg_published(CONFIG, tmp_path)
    with pytest.raises(TestSplitLocked):
        ensure_split_allowed("test", CONFIG, tmp_path)


def test_other_splits_are_never_locked(tmp_path):
    ensure_split_allowed("train", None, tmp_path)
    ensure_split_allowed("calib", None, tmp_path)
