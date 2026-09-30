"""The test barrier: the test split is refused until prereg/HASH is committed and on origin/main."""
import subprocess

import pytest

from bench.barrier import TestSplitLocked, ensure_split_allowed, prereg_published


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    origin, clone = tmp_path / "origin.git", tmp_path / "clone"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    git(tmp_path, "clone", "-q", str(origin), str(clone))
    git(clone, "config", "user.email", "t@example.com")
    git(clone, "config", "user.name", "t")
    (clone / "README").write_text("x\n")
    git(clone, "add", "README")
    git(clone, "commit", "-q", "-m", "init")
    git(clone, "push", "-q", "origin", "main")
    return clone


def write_hash(repo, content="abc\n"):
    (repo / "prereg").mkdir(exist_ok=True)
    (repo / "prereg" / "HASH").write_text(content)


def test_missing(repo):
    assert "does not exist" in prereg_published(repo)
    with pytest.raises(TestSplitLocked):
        ensure_split_allowed("test", repo)


def test_uncommitted(repo):
    write_hash(repo)
    assert "not committed" in prereg_published(repo)


def test_committed_but_not_pushed(repo):
    write_hash(repo)
    git(repo, "add", "prereg/HASH")
    git(repo, "commit", "-q", "-m", "prereg")
    assert "not on origin/main" in prereg_published(repo)


def test_published_then_modified(repo):
    write_hash(repo)
    git(repo, "add", "prereg/HASH")
    git(repo, "commit", "-q", "-m", "prereg")
    git(repo, "push", "-q", "origin", "main")
    assert prereg_published(repo) == ""
    ensure_split_allowed("test", repo)
    write_hash(repo, "changed\n")
    assert "uncommitted" in prereg_published(repo)
    git(repo, "commit", "-q", "-am", "change")
    assert "differs from origin/main" in prereg_published(repo)


def test_other_splits_are_never_locked(repo):
    ensure_split_allowed("train", repo)
    ensure_split_allowed("calib", repo)
