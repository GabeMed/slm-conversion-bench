"""What an execution records about the code: the commit, `dirty` (a bool, as every reader expects it), and
which paths made the tree dirty."""
import subprocess

from bench import provenance


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    for name in ("a.py", "b.py", "old.py"):
        (tmp_path / name).write_text(name)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_a_clean_tree_records_no_dirty_path(tmp_path):
    state = provenance.git_state(repo(tmp_path))
    assert len(state["commit"]) == 40 and state["dirty"] is False
    assert (state["dirty_paths"], state["dirty_paths_total"]) == ([], 0)


def test_a_dirty_tree_records_its_paths_and_status_codes_sorted(tmp_path):
    root = repo(tmp_path)
    (root / "b.py").write_text("changed")
    (root / "new dir").mkdir()
    (root / "new dir" / "z.txt").write_text("untracked")
    git(root, "mv", "old.py", "renamed.py")
    git(root, "rm", "-q", "a.py")
    state = provenance.git_state(root)
    assert state["dirty"] is True and state["dirty_paths_total"] == 4
    assert state["dirty_paths"] == [{"status": "D ", "path": "a.py"}, {"status": " M", "path": "b.py"},
                                    {"status": "??", "path": "new dir/"},
                                    {"status": "R ", "path": "renamed.py", "from": "old.py"}]


def test_the_dirty_paths_are_capped_and_counted(tmp_path, monkeypatch):
    root = repo(tmp_path)
    monkeypatch.setattr(provenance, "DIRTY_PATHS_MAX", 2)
    for i in range(5):
        (root / f"u{i}.txt").write_text("x")
    state = provenance.git_state(root)
    assert state["dirty"] is True and state["dirty_paths_total"] == 5
    assert [e["path"] for e in state["dirty_paths"]] == ["u0.txt", "u1.txt"]
