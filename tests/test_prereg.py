"""`bench prereg` (SPEC 6.1): what it writes and commits opens the test barrier for the registered
configuration and no other, once the author pushes it; and `bench eval` scores a test run only if
the run was made under the registration in force."""
import hashlib
import json
import subprocess

import pytest
import yaml

from bench import paths
from bench.barrier import prereg_published
from bench.contracts.config import config_sha256, load_config
from bench.data import DataError
from bench.evaluate import evaluate, evaluate_per_call
from bench.judge import j4
from bench.prereg import PreregError, register
from synthetic import make_repo, sha256

CONFIG = load_config(paths.ROOT / "config.yaml")


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def init_published(root, origin):
    """`root` becomes a clone of a bare `origin`, with everything in it committed and pushed."""
    git(origin.parent, "init", "-q", "--bare", "-b", "main", str(origin))
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    git(root, "remote", "add", "origin", str(origin))
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    git(root, "push", "-q", "-u", "origin", "main")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "clone"
    (root / "data").mkdir(parents=True)
    (root / "SPEC.md").write_text("protocol\n")
    (root / "data" / "splits.json").write_text('{"test": ["9"]}\n')
    (root / "data" / "MANIFEST.json").write_text('{"databases": {}}\n')
    (root / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    init_published(root, tmp_path / "origin.git")
    return root


def test_register_writes_the_contract_the_barrier_reads_and_commits_without_pushing(repo):
    before = git(repo, "rev-parse", "HEAD")
    registered = register("config.yaml", root=repo)
    manifest_bytes = (repo / "prereg" / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert registered["hash"] == hashlib.sha256(manifest_bytes).hexdigest() == (repo / "prereg" / "HASH").read_text().strip()
    assert (repo / "prereg" / "HASH").read_text() == registered["hash"] + "\n"
    assert manifest["config_sha256"] == config_sha256(CONFIG) and manifest["commit"] == before
    assert manifest["spec_sha256"] == sha256(repo / "SPEC.md")
    assert manifest["splits_sha256"] == sha256(repo / "data" / "splits.json")
    assert manifest["data_manifest_sha256"] == sha256(repo / "data" / "MANIFEST.json")
    rule = manifest["delta_rule"]
    assert (rule["cap_pp"], rule["bootstrap"]["n_boot"], rule["bootstrap"]["seed"]) == \
        (CONFIG["thresholds"]["delta_cap_pp"], CONFIG["stats"]["n_boot"], CONFIG["seeds"]["bootstrap"])
    assert rule["implementation"] == {"path": "bench/judge/j4.py", "sha256": sha256(j4.__file__)}
    assert git(repo, "rev-parse", "HEAD^") == before and git(repo, "status", "--porcelain") == ""
    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == ["prereg/HASH", "prereg/manifest.json"]
    assert "not on origin/main" in prereg_published(CONFIG, repo)  # publishing is the author's act
    git(repo, "push", "-q", "origin", "main")
    assert prereg_published(CONFIG, repo) == ""
    other = {**CONFIG, "eval": {**CONFIG["eval"], "fixed_date": "2026-10-01"}}
    assert "configuration differs" in prereg_published(other, repo)


@pytest.mark.parametrize("change", ["untracked", "modified"])
def test_register_refuses_a_dirty_tree(repo, change):
    if change == "untracked":
        (repo / "notes.txt").write_text("x\n")
    else:
        (repo / "SPEC.md").write_text("protocol, edited\n")
    with pytest.raises(PreregError, match="uncommitted"):
        register("config.yaml", root=repo)
    assert not (repo / "prereg").exists()


def test_registering_again_changes_nothing_and_replacing_needs_asking(repo):
    first = register("config.yaml", root=repo)
    head = git(repo, "rev-parse", "HEAD")
    assert register("config.yaml", root=repo) == {**first, "commit": head, "new": False}
    assert git(repo, "rev-parse", "HEAD") == head
    (repo / "SPEC.md").write_text("protocol, v2\n")
    git(repo, "commit", "-q", "-am", "spec v2")
    with pytest.raises(PreregError, match="--replace"):
        register("config.yaml", root=repo)
    second = register("config.yaml", root=repo, replace=True)
    assert second["hash"] != first["hash"] and (repo / "prereg" / "HASH").read_text().strip() == second["hash"]


def test_register_refuses_a_configuration_outside_the_repository_or_without_stats(repo, tmp_path):
    outside = tmp_path / "elsewhere.yaml"
    outside.write_text(yaml.safe_dump(CONFIG))
    with pytest.raises(PreregError, match="inside the repository"):
        register(str(outside), root=repo)
    (repo / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in CONFIG.items() if k != "stats"}))
    git(repo, "commit", "-q", "-am", "no stats")
    with pytest.raises(PreregError, match="stats.n_boot"):
        register("config.yaml", root=repo)


# ---------------------------------------------------------------- the test split, through a published registration

@pytest.fixture
def published(tmp_path, monkeypatch, tmp_path_factory):
    """The synthetic repository (test id 9) as a git clone with its pre-registration pushed."""
    root, config_path, _ = make_repo(tmp_path, monkeypatch)
    (paths.RAW / "mini_dev.json").write_text(json.dumps([{"question_id": 9, "db_id": "tiny", "difficulty": "challenging"}]))
    raw = yaml.safe_load(config_path.read_text())
    raw["data"]["mini_dev"]["sha256"] = sha256(paths.RAW / "mini_dev.json")
    config_path.write_text(yaml.safe_dump(raw))
    (root / "SPEC.md").write_text("protocol\n")
    (root / ".gitignore").write_text("runs/\ndata/raw/\ndata/bird_dev/\n")
    init_published(root, tmp_path_factory.mktemp("remote") / "origin.git")
    registered = register(str(config_path.relative_to(root)), root=root)
    git(root, "push", "-q", "origin", "main")
    return load_config(config_path), registered["hash"]


def test_run(config, prereg_hash, predictions=None):
    run_dir = paths.RUNS / "agent-B0-test-x"
    run_dir.mkdir(parents=True)
    predictions = predictions or {"9": "SELECT 9"}
    (run_dir / "predictions.json").write_text(json.dumps(predictions))
    (run_dir / "calls.jsonl").write_text("")
    (run_dir / "config.json").write_text(json.dumps(config))
    manifest = {"run_id": run_dir.name, "type": "agent", "arm": "B0", "split": "test", "status": "done", "commit": "c",
                "question_ids": sorted(predictions), "config_sha256": config_sha256(config)}
    if prereg_hash is not None:
        manifest["prereg_hash"] = prereg_hash
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir.name


test_run.__test__ = False  # a helper, not a test


def test_a_test_run_under_the_registration_in_force_is_scored(published):
    config, prereg_hash = published
    out = evaluate(test_run(config, prereg_hash))
    result = json.loads((out / "results.jsonl").read_text())
    assert (result["question_id"], result["correct"], result["difficulty"]) == ("9", True, "challenging")
    assert json.loads((out / "manifest.json").read_text())["prereg_hash"] == prereg_hash


@pytest.mark.parametrize("prereg_hash", ["0" * 64, None])
def test_a_test_run_under_another_registration_is_refused(published, prereg_hash):
    config, _ = published
    run_id = test_run(config, prereg_hash)
    with pytest.raises(DataError, match="pre-registration"):
        evaluate(run_id)
    with pytest.raises(DataError, match="pre-registration"):
        evaluate_per_call(run_id)
    assert not any(paths.RUNS.glob("eval-*"))


def test_the_command_registers_and_refuses_through_the_cli(repo, monkeypatch, capsys):
    from bench.cli import main
    monkeypatch.setattr(paths, "ROOT", repo)
    assert main(["prereg", "--config", str(repo / "config.yaml")]) == 0
    out = capsys.readouterr()
    assert (repo / "prereg" / "HASH").read_text().strip() in out.out and "arms.B3.choice" in out.err
    (repo / "notes.txt").write_text("x\n")
    assert main(["prereg", "--config", str(repo / "config.yaml")]) == 2
    assert "uncommitted" in capsys.readouterr().err
