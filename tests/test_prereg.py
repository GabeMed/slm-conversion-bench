"""`bench prereg` (SPEC 6.1): what it writes and commits opens the test barrier for the registered
configuration and no other, once the author pushes it; and `bench eval` scores a test run only if
the run was made under the registration in force."""
import hashlib
import json
import shutil
import subprocess

import pytest
import yaml

from bench import paths
from bench.barrier import prereg_published
from bench.contracts.config import config_sha256, load_config
from bench.data import DataError
from bench.evaluate import evaluate, evaluate_per_call
from bench.prereg import PreregError, check_registered_analysis_code, register
from synthetic import make_repo, sha256

# distinct seeds, so reading one seed key for another shows
CONFIG = {**load_config(paths.ROOT / "config.yaml"),
          "seeds": {"calib_split": 101, "schema_shuffle": 202, "bootstrap": 303, "few_shot": 404}}


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


CODE = paths.ROOT  # this repository, taken before any test points bench.paths elsewhere
ANALYSIS = ["bench/data.py", "bench/evaluate.py", "bench/judge/__init__.py", "bench/judge/j1.py", "bench/judge/j4.py",
            "bench/paths.py", *git(CODE, "ls-files", "bench/contracts").splitlines()]


def copy_analysis_code(root):
    """The analysis code the registration hashes, as the repository holds it."""
    for rel in ANALYSIS:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(CODE / rel, root / rel)


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
    (root / "data" / "splits.json").write_text(json.dumps({"calib": [str(i) for i in range(100, 160)], "test": ["9"]}) + "\n")
    (root / "data" / "MANIFEST.json").write_text('{"databases": {}}\n')
    (root / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    copy_analysis_code(root)
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
    rule = manifest["delta_rule"]  # the fixed margin (T3); the pilot only gives the planned power
    assert (rule["margin_pp"], rule["selection_margin_pp"], rule["bootstrap"]["n_boot"], rule["bootstrap"]["seed"]) == \
        (CONFIG["thresholds"]["delta_pp"], CONFIG["thresholds"]["selection_delta_pp"], CONFIG["stats"]["n_boot"],
         CONFIG["seeds"]["bootstrap"]) == (5, 2.5, 10000, 303)
    assert "cap_pp" not in rule and "sqrt(d / n)" in rule["planned_power"]["formula"]
    assert rule["pilot"] == {"accessor": "bench.data.pilot_ids(config)", "size": CONFIG["stats"]["pilot_size"]}
    assert manifest["analysis_code"] == {rel: sha256(repo / rel) for rel in ANALYSIS}
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
    with pytest.raises(PreregError, match="outside the repository"):
        register(str(outside), root=repo)
    (repo / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in CONFIG.items() if k != "stats"}))
    git(repo, "commit", "-q", "-am", "no stats")
    with pytest.raises(PreregError, match="stats.n_boot"):
        register("config.yaml", root=repo)


@pytest.mark.parametrize("rel", ANALYSIS + ["bench/report.py", "bench/judge/j9.py"])
def test_editing_any_analysis_file_changes_the_registration(repo, rel):
    first = register("config.yaml", root=repo)
    path = repo / rel
    path.write_text((path.read_text() if path.exists() else '"""the report"""\n') + "# a reading changed\n")
    git(repo, "add", rel)
    git(repo, "commit", "-q", "-m", f"edit {rel}")
    with pytest.raises(PreregError, match="--replace"):
        register("config.yaml", root=repo)
    second = register("config.yaml", root=repo, replace=True)
    manifest = json.loads((repo / "prereg" / "manifest.json").read_text())
    assert second["hash"] != first["hash"] and manifest["analysis_code"][rel] == sha256(path)


@pytest.mark.parametrize("ignored", [False, True])
def test_analysis_code_the_commit_would_not_hold_is_refused(repo, ignored):
    if ignored:
        (repo / ".gitignore").write_text("bench/judge/local_*.py\n")
        git(repo, "add", ".gitignore")
        git(repo, "commit", "-q", "-m", "ignore local judges")
    (repo / "bench" / "judge" / "local_j9.py").write_text("READING = 'changed'\n")
    with pytest.raises(PreregError, match="ignored by git: bench/judge/local_j9.py" if ignored else "uncommitted"):
        register("config.yaml", root=repo)


def test_ignored_bytecode_beside_the_analysis_code_is_not_a_refusal(repo):
    """Every checkout that imported bench.judge has an ignored __pycache__, and a Mac writes .DS_Store
    where Finder looked: neither is unregistered code."""
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n.DS_Store\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore bytecode")
    cache = repo / "bench" / "judge" / "__pycache__"
    cache.mkdir()
    (cache / "j4.cpython-311.pyc").write_bytes(b"\0")
    (repo / "bench" / "contracts" / ".DS_Store").write_bytes(b"\0")
    assert register("config.yaml", root=repo)["hash"]
    check_registered_analysis_code(repo)


def test_a_malformed_registration_is_replaced_only_on_request(repo):
    (repo / "prereg").mkdir()
    (repo / "prereg" / "manifest.json").write_text("[]\n")
    (repo / "prereg" / "HASH").write_text("x\n")
    git(repo, "add", "prereg")
    git(repo, "commit", "-q", "-m", "a malformed registration")
    with pytest.raises(PreregError, match="the registration in prereg/ is broken"):
        register("config.yaml", root=repo)
    with pytest.raises(PreregError, match="not a JSON object: no analysis code is registered"):
        check_registered_analysis_code(repo)
    assert register("config.yaml", root=repo, replace=True)["new"]


@pytest.mark.parametrize("damage", [b"0" * 64 + b"\n", b"\xff\xfe\n", None])  # wrong, not UTF-8, missing
def test_a_registration_whose_hash_is_not_its_manifests_is_repaired_only_on_request(repo, damage):
    """`bench prereg` checks both halves, as the barrier does: never "already registered" with a broken HASH."""
    registered = register("config.yaml", root=repo)
    hash_path = repo / "prereg" / "HASH"
    if damage is None:
        git(repo, "rm", "-q", "prereg/HASH")
    else:
        hash_path.write_bytes(damage)
        git(repo, "add", "prereg/HASH")
    git(repo, "commit", "-q", "-m", "a damaged HASH")
    with pytest.raises(PreregError, match="the registration in prereg/ is broken"):
        register("config.yaml", root=repo)
    repaired = register("config.yaml", root=repo, replace=True)  # a new registration (it records its own commit), intact
    manifest_sha = hashlib.sha256((repo / "prereg" / "manifest.json").read_bytes()).hexdigest()
    assert repaired["new"] and repaired["hash"] == hash_path.read_text().strip() == manifest_sha != registered["hash"]


def test_register_refuses_without_the_analysis_code(repo):
    git(repo, "rm", "-q", "bench/evaluate.py")
    git(repo, "commit", "-q", "-m", "drop the evaluator")
    with pytest.raises(PreregError, match="bench/evaluate.py"):
        register("config.yaml", root=repo)


def commit_config(repo, name, text):
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_text(text)
    git(repo, "add", "-f", name)
    git(repo, "commit", "-q", "-m", f"add {name}")


def test_register_refuses_an_extends_chain_that_the_commit_does_not_hold(repo, tmp_path):
    # untracked because ignored: the tree looks clean, but the commit would not hold the parent
    (repo / ".gitignore").write_text("local.yaml\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore local.yaml")
    (repo / "local.yaml").write_text(yaml.safe_dump(CONFIG))
    commit_config(repo, "via-ignored.yaml", "extends: local.yaml\n")
    assert git(repo, "status", "--porcelain") == ""
    with pytest.raises(PreregError, match="local.yaml is not tracked"):
        register("via-ignored.yaml", root=repo)
    # tracked, but matched by .gitignore (force-added): refused as well
    git(repo, "add", "-f", "local.yaml")
    git(repo, "commit", "-q", "-m", "force-add local.yaml")
    with pytest.raises(PreregError, match="local.yaml is ignored"):
        register("via-ignored.yaml", root=repo)
    # a parent outside the repository
    (tmp_path / "outside.yaml").write_text(yaml.safe_dump(CONFIG))
    commit_config(repo, "via-outside.yaml", "extends: ../outside.yaml\n")
    with pytest.raises(PreregError, match="outside the repository"):
        register("via-outside.yaml", root=repo)
    assert not (repo / "prereg").exists()


def test_the_extends_chain_resolves_each_parent_from_its_own_file(repo):
    # sub/child.yaml -> sub/deeper/mid.yaml -> config.yaml: resolved from the child's folder or the root,
    # the second link would point outside the repository
    commit_config(repo, "sub/deeper/mid.yaml", "extends: ../../config.yaml\n")
    commit_config(repo, "sub/child.yaml", "extends: deeper/mid.yaml\n")
    assert register("sub/child.yaml", root=repo)["new"] is True
    (repo / "local.yaml").write_text(yaml.safe_dump(CONFIG))
    commit_config(repo, ".gitignore", "local.yaml\n")
    commit_config(repo, "sub/deeper/mid2.yaml", "extends: ../../local.yaml\n")
    commit_config(repo, "sub/child2.yaml", "extends: deeper/mid2.yaml\n")
    with pytest.raises(PreregError, match="local.yaml is not tracked"):  # checked at every depth
        register("sub/child2.yaml", root=repo, replace=True)


def test_register_accepts_a_tracked_extends_chain(repo):
    commit_config(repo, "child.yaml", "extends: config.yaml\n")
    assert register("child.yaml", root=repo)["new"] is True


# ---------------------------------------------------------------- the test split, through a published registration

@pytest.fixture
def published(tmp_path, monkeypatch, tmp_path_factory):
    """The synthetic repository (test id 9) as a git clone with its pre-registration pushed."""
    root, config_path, _ = make_repo(tmp_path, monkeypatch)
    (paths.RAW / "mini_dev.json").write_text(json.dumps([{"question_id": 9, "db_id": "tiny", "difficulty": "challenging"}]))
    raw = yaml.safe_load(config_path.read_text())
    raw["data"]["mini_dev"]["sha256"] = sha256(paths.RAW / "mini_dev.json")
    raw["stats"]["pilot_size"] = 1  # the synthetic calibration split has a single id
    config_path.write_text(yaml.safe_dump(raw))
    paths.SPLITS.write_text(json.dumps({"train": ["1", "2"], "calib": ["3"], "test": ["9"], "excluded": []}))
    (root / "SPEC.md").write_text("protocol\n")
    (root / ".gitignore").write_text("runs/\ndata/raw/\ndata/bird_dev/\n")
    copy_analysis_code(root)
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


@pytest.mark.parametrize("rel", ["bench/judge/j4.py", "bench/contracts/concordance.py", "bench/paths.py"])
def test_a_test_run_is_not_scored_with_analysis_code_other_than_the_registered(published, rel):
    config, prereg_hash = published
    changed = paths.ROOT / rel  # the synthetic repository's copy
    changed.write_text(changed.read_text() + "\n# a reading changed after the pre-registration\n")
    with pytest.raises(DataError, match=f"analysis code differs from the pre-registered one: {rel}"):
        evaluate(test_run(config, prereg_hash))


def test_a_test_run_is_not_scored_with_analysis_code_no_commit_holds(published):
    config, prereg_hash = published
    (paths.ROOT / "bench" / "judge" / "j9.py").write_text("READING = 'new'\n")
    with pytest.raises(DataError, match="analysis code not committed: bench/judge/j9.py"):
        evaluate(test_run(config, prereg_hash))


def test_a_registration_that_recorded_no_analysis_code_matches_none(repo):
    register("config.yaml", root=repo)
    check_registered_analysis_code(repo)
    manifest = json.loads((repo / "prereg" / "manifest.json").read_text())
    del manifest["analysis_code"]
    (repo / "prereg" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(PreregError, match="differs from the pre-registered one: bench/contracts/"):
        check_registered_analysis_code(repo)
