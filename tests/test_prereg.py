"""`bench prereg` (SPEC 6.1): what it writes and commits opens the test barrier for the registered
configuration and no other, once someone pushes it; and `bench eval` scores a test run only if
the run was made under the registration in force."""
import hashlib
import json
import re
import shutil
import subprocess

import pytest
import yaml

from bench import paths
from bench.barrier import prereg_published
from bench.contracts.config import config_sha256, load_config
from bench.contracts.facts import canonical
from bench.data import DataError
from bench.evaluate import evaluate, evaluate_per_call
from bench.prereg import PreregError, check_registered_analysis_code, register
from synthetic import make_repo, sha256

# distinct seeds, so reading one seed key for another shows
SHIPPED = load_config(paths.ROOT / "config.yaml")
CONFIG = {**SHIPPED, "seeds": {"calib_split": 101, "schema_shuffle": 202, "bootstrap": 303, "few_shot": 404},
          "cost": {**SHIPPED["cost"], "slo_from": "agent-B0-calib"},  # what the pilot fixes before the registration
          # every fact the arms follow, decided (the repo fixture writes and commits each)
          "arms": {**SHIPPED["arms"], "B3": {"choice": "judgments/J6/c/choice.json"},
                   "B4": {"choice": "judgments/J6/c/choice.json", "centroids": "judgments/J5/c/centroids.json",
                          "adapters": "judgments/S5/a/adapters.json"},
                   "B5": {"choice": "judgments/J6/c/choice.json", "centroids": "judgments/J5/c/centroids.json",
                          "adapters": "judgments/S5/a/adapters.json", "allocation": "judgments/J7/a/allocation.json"}}}
ARM_FACTS = sorted({rel for arm in CONFIG["arms"].values() if isinstance(arm, dict)
                    for name, rel in arm.items() if name != "few_shot" and isinstance(rel, str)})
PILOT = [str(i) for i in range(100, 150)]  # what bench.data.pilot_ids gives in these tests (no BIRD here)


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


CODE = paths.ROOT  # this repository, taken before any test points bench.paths elsewhere
REASON = "the first registration hashed the wrong splits"  # what --replace must say (T13)
ANALYSIS = ["bench/data.py", "bench/evaluate.py", "bench/judge/__init__.py", "bench/judge/j1.py", "bench/judge/j4.py",
            "bench/paths.py", *git(CODE, "ls-files", "bench/contracts").splitlines()]


EXECUTION = ["bench/agent/resume.py", "bench/curate.py", "vendor/chess/src/runner/run_manager.py"]


def copy_analysis_code(root):
    """The analysis code the registration hashes, as the repository holds it, and some execution code."""
    for rel in ANALYSIS + EXECUTION:
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


def write_choice(root, k, pilot=PILOT, rule=None):
    """B1's k as bench judge b1k stores it: by the sha256 of its bytes."""
    from bench.judge import b1k
    raw = canonical({"judgment": "b1k", "reads": {}, "result": {"k": k, "rule": rule or b1k.RULE, "pilot_ids": pilot}})
    path = root / "judgments" / "b1k" / hashlib.sha256(raw).hexdigest() / "choice.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


@pytest.fixture(autouse=True)
def pilot(monkeypatch):
    from bench import data
    monkeypatch.setattr(data, "pilot_ids", lambda config: list(PILOT))


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "clone"
    (root / "data").mkdir(parents=True)
    (root / "SPEC.md").write_text("protocol\n")
    (root / "data" / "splits.json").write_text(json.dumps({"calib": [str(i) for i in range(100, 160)], "test": ["9"]}) + "\n")
    (root / "data" / "MANIFEST.json").write_text('{"databases": {}}\n')
    (root / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    write_choice(root, CONFIG["arms"]["B1"]["few_shot"]["k"])  # the pilot chose the configured k
    for rel in ARM_FACTS:  # prereg reads only that git holds them
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}\n")
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
    assert rule["pilot"] == {"accessor": "bench.data.pilot_ids(config)", "size": 50,
                             "mix": {"simple": 15, "moderate": 25, "challenging": 10}}
    assert manifest["analysis_code"] == {rel: sha256(repo / rel) for rel in ANALYSIS}
    assert manifest["execution_code"] == {rel: sha256(repo / rel) for rel in EXECUTION}  # all of bench/ but analysis
    assert git(repo, "rev-parse", "HEAD^") == before and git(repo, "status", "--porcelain") == ""
    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == ["prereg/HASH", "prereg/manifest.json"]
    assert "not on origin/main" in prereg_published(CONFIG, repo)  # publishing is left to a person
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
    second = register("config.yaml", root=repo, replace=True, reason=REASON)
    assert second["hash"] != first["hash"] and (repo / "prereg" / "HASH").read_text().strip() == second["hash"]


def test_replacing_a_registration_needs_its_reason_which_becomes_a_line_of_deviations(repo):
    """T13: `--replace` needs `--reason`; the defect is one dated line of prereg/DEVIATIONS.md, committed
    with the new manifest and HASH."""
    deviations = repo / "prereg" / "DEVIATIONS.md"
    first = register("config.yaml", root=repo)
    assert not deviations.exists()  # a first registration deviates from nothing
    (repo / "SPEC.md").write_text("protocol, v2\n")
    git(repo, "commit", "-q", "-am", "spec v2")
    head = git(repo, "rev-parse", "HEAD")
    for no_reason in (None, "", "  \n "):
        with pytest.raises(PreregError, match="--replace needs --reason"):
            register("config.yaml", root=repo, replace=True, reason=no_reason)
    with pytest.raises(PreregError, match="a different pre-registration is in force"):
        register("config.yaml", root=repo, reason=REASON)  # a reason alone replaces nothing
    assert git(repo, "rev-parse", "HEAD") == head and not deviations.exists() and git(repo, "status", "--porcelain") == ""
    second = register("config.yaml", root=repo, replace=True, reason="the SPEC named\n the wrong pilot ")
    (line,) = deviations.read_text().splitlines()
    assert re.fullmatch(rf"- \d{{4}}-\d{{2}}-\d{{2}} · {second['hash']} replaces {first['hash']}: the SPEC named the wrong pilot", line)
    assert git(repo, "status", "--porcelain") == ""  # committed with the registration, in one commit
    assert sorted(git(repo, "show", "--name-only", "--format=", "HEAD").split()) == [
        "prereg/DEVIATIONS.md", "prereg/HASH", "prereg/manifest.json"]
    # registering the same inputs again replaces nothing: no new line, with or without --replace
    assert register("config.yaml", root=repo, replace=True, reason=REASON)["new"] is False
    assert deviations.read_text().splitlines() == [line]
    (repo / "SPEC.md").write_text("protocol, v3\n")
    git(repo, "commit", "-q", "-am", "spec v3")
    third = register("config.yaml", root=repo, replace=True, reason=REASON)
    assert deviations.read_text().splitlines()[0] == line and len(deviations.read_text().splitlines()) == 2
    assert deviations.read_text().splitlines()[1].endswith(f"· {third['hash']} replaces {second['hash']}: {REASON}")
    check_registered_analysis_code(repo)  # the file changes nothing the barrier or the registered code reads


def test_register_refuses_a_configuration_outside_the_repository_or_without_stats(repo, tmp_path):
    outside = tmp_path / "elsewhere.yaml"
    outside.write_text(yaml.safe_dump(CONFIG))
    with pytest.raises(PreregError, match="outside the repository"):
        register(str(outside), root=repo)
    (repo / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in CONFIG.items() if k != "stats"}))
    git(repo, "commit", "-q", "-am", "no stats")
    with pytest.raises(PreregError, match="stats.n_boot"):
        register("config.yaml", root=repo)
    short = {**CONFIG, "stats": {**CONFIG["stats"], "pilot_mix": {"simple": 15, "moderate": 25, "challenging": 9}}}
    (repo / "config.yaml").write_text(yaml.safe_dump(short))
    git(repo, "commit", "-q", "-am", "a mix of 49")
    with pytest.raises(PreregError, match="stats.pilot_mix does not give stats.pilot_size questions"):
        register("config.yaml", root=repo)
    (repo / "config.yaml").write_text(yaml.safe_dump({**CONFIG, "stats": {k: v for k, v in CONFIG["stats"].items()
                                                                          if k != "pilot_mix"}}))
    git(repo, "commit", "-q", "-am", "no mix")
    with pytest.raises(PreregError, match="stats.pilot_mix does not give"):
        register("config.yaml", root=repo)


def test_register_refuses_what_the_report_would_refuse_after_the_test(repo):
    """The execution the SLO is measured on, and B1's k as the pilot chose it: found wrong by the report,
    after the test runs, either would need a new registration and every test run would be superseded."""
    def commit(config, message):
        (repo / "config.yaml").write_text(yaml.safe_dump(config))
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", message)
    commit({**CONFIG, "cost": {**CONFIG["cost"], "slo_from": None}}, "no SLO execution")
    with pytest.raises(PreregError, match="no cost.slo_from"):
        register("config.yaml", root=repo)
    k = CONFIG["arms"]["B1"]["few_shot"]["k"]
    shutil.rmtree(repo / "judgments" / "b1k")
    other = write_choice(repo, 3 - k)
    commit(CONFIG, "the pilot chose the other k")
    with pytest.raises(PreregError, match=rf"arms.B1.few_shot.k is {k}, and the stored choice .* is {3 - k}"):
        register("config.yaml", root=repo)
    write_choice(repo, k)
    commit(CONFIG, "two choices")
    with pytest.raises(PreregError, match="holds 2 choices of B1's k, not one"):
        register("config.yaml", root=repo)
    other.unlink()
    commit(CONFIG, "the pilot chose this k")
    assert register("config.yaml", root=repo)["new"] is True


@pytest.mark.parametrize("damage", ["none", "edited", "rule", "pilot"])
def test_register_refuses_a_b1k_choice_that_is_missing_or_not_the_pilots(repo, damage):
    """B1's k must have been chosen by bench judge b1k, by today's rule, on today's pilot: a clone with no
    choice, or with one edited after it was written, or chosen by another rule or on other questions,
    does not let any k through."""
    k = CONFIG["arms"]["B1"]["few_shot"]["k"]
    shutil.rmtree(repo / "judgments" / "b1k")
    if damage == "edited":
        path = write_choice(repo, k)
        path.write_bytes(path.read_bytes().replace(b'"k":', b'"k": '))
    elif damage == "rule":
        write_choice(repo, k, rule="k = 3 by a replay of the teacher's calls")
    elif damage == "pilot":
        write_choice(repo, k, pilot=[str(i) for i in range(110, 160)])
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", damage)
    reason = {"none": "holds 0 choices", "edited": "not stored by its sha256", "rule": "not chosen by today's rule",
              "pilot": "other pilot questions"}[damage]
    with pytest.raises(PreregError, match=reason):
        register("config.yaml", root=repo)
    assert not (repo / "prereg" / "manifest.json").exists()


def test_register_refuses_a_fact_the_configuration_names_that_git_does_not_track(repo):
    """The b1k choice and every arms.* fact (J6's choice, J5's centroids, the adapters, the allocation) must
    be in the registration commit: a clone without them could not tell which decision was registered."""
    rel = CONFIG["arms"]["B5"]["allocation"]
    git(repo, "rm", "-q", "--cached", "--", rel)
    (repo / ".gitignore").write_text("judgments/J7/\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the allocation, ignored")
    with pytest.raises(PreregError, match=rf"{rel}, a fact the configuration names, is not tracked"):
        register("config.yaml", root=repo)
    git(repo, "add", "-f", "--", rel)
    git(repo, "commit", "-q", "-m", "the allocation, forced in")
    with pytest.raises(PreregError, match="is ignored by git"):
        register("config.yaml", root=repo)
    (repo / ".gitignore").unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the allocation, tracked")
    assert register("config.yaml", root=repo)["new"] is True
    git(repo, "rm", "-q", "--", CONFIG["arms"]["B4"]["adapters"])
    git(repo, "commit", "-q", "-m", "the adapters fact, gone")
    with pytest.raises(PreregError, match="judgments/S5/a/adapters.json, a fact the configuration names, does not exist"):
        register("config.yaml", root=repo, replace=True, reason=REASON)
    git(repo, "checkout", "-q", "HEAD~1", "--", CONFIG["arms"]["B4"]["adapters"])
    (repo / "judgments" / "b1k").rename(repo / "b1k-elsewhere")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "no choice in the clone")
    with pytest.raises(PreregError, match="holds 0 choices"):
        register("config.yaml", root=repo, replace=True, reason=REASON)


@pytest.mark.parametrize("unset_fact", [("B4", "adapters"), ("B5", "adapters"), ("B5", "allocation"), ("B3", "choice")])
def test_register_refuses_an_arm_whose_fact_is_still_null(repo, unset_fact):
    """B4's and B5's adapters and B5's allocation (and every other fact the router reads) are decided before
    the registration: registered null, the arm could not run on test, and its decision would come after."""
    arm, name = unset_fact
    config = {**CONFIG, "arms": {**CONFIG["arms"], arm: {**CONFIG["arms"][arm], name: None}}}
    (repo / "config.yaml").write_text(yaml.safe_dump(config))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", f"arms.{arm}.{name} null")
    with pytest.raises(PreregError, match=rf"arms.{arm}.{name} not set"):
        register("config.yaml", root=repo)
    assert not (repo / "prereg" / "manifest.json").exists()


@pytest.mark.parametrize("rel", ANALYSIS + ["bench/report.py", "bench/judge/j9.py"])
def test_editing_any_analysis_file_changes_the_registration(repo, rel):
    first = register("config.yaml", root=repo)
    path = repo / rel
    path.write_text((path.read_text() if path.exists() else '"""the report"""\n') + "# a reading changed\n")
    git(repo, "add", rel)
    git(repo, "commit", "-q", "-m", f"edit {rel}")
    with pytest.raises(PreregError, match="--replace"):
        register("config.yaml", root=repo)
    second = register("config.yaml", root=repo, replace=True, reason=REASON)
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
    assert register("config.yaml", root=repo, replace=True, reason=REASON)["new"]
    assert (repo / "prereg" / "DEVIATIONS.md").read_text().endswith(f" replaces a broken registration: {REASON}\n")


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
    repaired = register("config.yaml", root=repo, replace=True, reason=REASON)  # a new registration (it records its own commit), intact
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
        register("sub/child2.yaml", root=repo, replace=True, reason=REASON)


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
    raw["stats"]["pilot_mix"] = {"simple": 1, "moderate": 0, "challenging": 0}
    raw.setdefault("cost", {})["slo_from"] = "agent-B0-calib"
    raw["arms"] = {**raw["arms"], **{arm: CONFIG["arms"][arm] for arm in ("B3", "B4", "B5")}}  # every fact decided
    for rel in ARM_FACTS:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}\n")
    write_choice(root, raw["arms"]["B1"]["few_shot"]["k"])
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
    assert (repo / "prereg" / "HASH").read_text().strip() in out.out and "train.sft.max_steps" in out.err  # a value still null (not a fact an arm follows)
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


def test_the_execution_code_is_registered_and_what_the_commit_would_not_hold_is_refused(repo):
    """The code a test execution runs, beyond the analysis code: the barrier compares it file by file."""
    from bench.barrier import code_in_force
    registered = register("config.yaml", root=repo)
    manifest = json.loads((repo / "prereg" / "manifest.json").read_text())
    assert code_in_force(manifest, repo) == ""
    (repo / "vendor" / "chess" / "src" / "runner" / "run_manager.py").write_text("changed = True\n")
    git(repo, "commit", "-q", "-am", "a change to CHESS after the registration")
    assert "vendor/chess/src/runner/run_manager.py" in code_in_force(manifest, repo)
    (repo / ".gitignore").write_text("bench/agent/local_*.py\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore local hooks")
    (repo / "bench" / "agent" / "local_hook.py").write_text("x = 1\n")  # would run, and no commit holds it
    with pytest.raises(PreregError, match="execution code ignored by git: bench/agent/local_hook.py"):
        register("config.yaml", root=repo, replace=True, reason=REASON)
    assert registered["new"]


def test_a_reading_of_the_test_needs_the_registered_code_but_no_clean_tree(repo):
    """bench eval and the report write judgments/, so they need no clean tree; they need the registered
    code: HEAD holds the registered tree outside prereg/, registry/ and judgments/."""
    register("config.yaml", root=repo)
    (repo / "judgments" / "j3").mkdir(parents=True)
    (repo / "judgments" / "j3" / "result.json").write_text("{}\n")
    (repo / "notes.txt").write_text("not committed\n")  # a dirty tree: no matter for a reading
    check_registered_analysis_code(repo)
    git(repo, "add", "judgments")
    git(repo, "commit", "-q", "-m", "judgments after the test")
    check_registered_analysis_code(repo)
    (repo / "bench" / "agent" / "resume.py").write_text("changed = True\n")  # execution code, not analysis code
    git(repo, "commit", "-q", "-am", "a change after the registration")
    with pytest.raises(PreregError, match="not the registered one: HEAD differs .* bench/agent/resume.py"):
        check_registered_analysis_code(repo)


def test_an_abandoned_test_attempt_is_scored_partially_on_the_predictions_it_recorded(published):
    """`bench eval --partial`: for the report's list of attempts, on the questions the attempt completed,
    and only on the predictions its manifest recorded when it ended; never for a done run, never off test."""
    from bench.data import sha256_file
    config, prereg_hash = published
    run_id = test_run(config, prereg_hash, predictions={"9": "SELECT 9"})
    run_dir = paths.RUNS / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    with pytest.raises(DataError, match="--partial scores an abandoned test attempt"):
        evaluate(run_id, partial=True)  # a done run is scored whole
    failed = {**manifest, "status": "failed", "predictions_sha256": sha256_file(run_dir / "predictions.json")}
    (run_dir / "manifest.json").write_text(json.dumps(failed))
    with pytest.raises(DataError, match="not 'done'"):
        evaluate(run_id)
    out = evaluate(run_id, partial=True)
    scored = json.loads((out / "manifest.json").read_text())
    assert (scored["partial"], scored["n"], scored["correct"], scored["of_questions"]) == (True, 1, 1, 1)
    assert scored["predictions_sha256"] == failed["predictions_sha256"] and out.name.startswith("eval-partial-")
    (run_dir / "manifest.json").write_text(json.dumps({**failed, "harness_errors": {"9": ["a dead pod"]}}))
    with pytest.raises(DataError, match="completed no question"):
        evaluate(run_id, partial=True)
    (run_dir / "predictions.json").write_text(json.dumps({"9": "SELECT 8"}))
    (run_dir / "manifest.json").write_text(json.dumps(failed))
    with pytest.raises(DataError, match="do not hash to what its manifest recorded"):
        evaluate(run_id, partial=True)


def test_a_test_attempt_killed_outright_is_scored_partially_on_its_journal(published):
    """D3: an attempt killed outright never closed its manifest (`running`, no predictions_sha256). Once its
    session's lock is free (the rule a resume reads), `--partial` scores the questions its session journaled
    (answers*.jsonl; a last line cut off by the kill is dropped), the model's failures as wrong, and checks
    the hashes of what it left against the ones a resume of it recorded, or records that none existed."""
    import fcntl
    import os
    from bench.agent.resume import unregistered_source_hashes
    from bench.agent.runner import SESSION_LOCK
    config, prereg_hash = published
    run_id = test_run(config, prereg_hash, predictions={"9": "SELECT 9"})
    run_dir = paths.RUNS / run_id
    (run_dir / "predictions.json").unlink()  # killed before the end: no predictions.json
    manifest = {**json.loads((run_dir / "manifest.json").read_text()), "status": "running"}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "answers.jsonl").write_text(json.dumps({"question_id": "9", "prediction": "SELECT 9", "tool_errors": {}})
                                           + '\n{"question_id": "1')  # the kill cut the last line
    with pytest.raises(DataError, match="left no session lock"):
        evaluate(run_id, partial=True)
    fd = os.open(run_dir / SESSION_LOCK, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(DataError, match="still running"):
            evaluate(run_id, partial=True)
    finally:
        os.close(fd)
    with pytest.raises(DataError, match="not 'done'"):
        evaluate(run_id)  # never scored whole
    out = evaluate(run_id, partial=True)
    scored = json.loads((out / "manifest.json").read_text())
    assert (scored["partial"], scored["n"], scored["correct"], scored["of_questions"]) == (True, 1, 1, 1)
    assert (scored["source_status"], scored["predictions_sha256"], scored["source_hashes_recorded_by"]) == ("running", None, None)
    assert scored["source_hashes"] == unregistered_source_hashes(manifest) and "answers.jsonl" in scored["source_hashes"]

    registry = paths.ROOT / "registry" / "test"  # a resume of it published the hashes in its intent
    registry.mkdir(parents=True, exist_ok=True)
    intent = {"run_id": "agent-B0-test-y", "started_at": "2026-10-02T09:00:00+00:00", "resumed_from": run_id,
              "unregistered_source_hashes": unregistered_source_hashes(manifest)}
    (registry / "agent-B0-test-y.intent.json").write_text(json.dumps(intent))
    assert json.loads((evaluate(run_id, partial=True) / "manifest.json").read_text())["source_hashes_recorded_by"] == \
        "agent-B0-test-y"
    (run_dir / "answers.jsonl").write_text(json.dumps({"question_id": "9", "prediction": None,
                                                       "tool_errors": {"generate_candidate": "APIStatusError"}}) + "\n")
    with pytest.raises(DataError, match="do not hash to what agent-B0-test-y recorded when it resumed it"):
        evaluate(run_id, partial=True)
    (registry / "agent-B0-test-y.intent.json").unlink()
    out = evaluate(run_id, partial=True)  # the model's failure, journaled: scored, as wrong
    assert [json.loads(line)["pred_error"] for line in (out / "results.jsonl").read_text().splitlines()] == ["no prediction"]
    assert json.loads((out / "manifest.json").read_text())["correct"] == 0
    (run_dir / "answers.jsonl").write_text("")  # it journaled nothing: scored on no question, and still listed
    assert json.loads((evaluate(run_id, partial=True) / "manifest.json").read_text())["n"] == 0
    (run_dir / "manifest.json").write_text(json.dumps({**manifest, "split": "calib"}))
    with pytest.raises(DataError, match="--partial scores an abandoned test attempt"):
        evaluate(run_id, partial=True)


def test_a_partial_evaluation_scores_the_models_failures_as_wrong_and_leaves_out_what_did_not_finish(published):
    """A question the model failed (a null prediction, its failed call a tool error) ended: it is scored,
    as wrong. One that raised or met a harness error did not finish: it is left out, as a resume answers
    it again."""
    from bench.data import sha256_file
    config, prereg_hash = published
    run_id = test_run(config, prereg_hash, predictions={"9": None})
    run_dir = paths.RUNS / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    failed = {**manifest, "status": "failed", "predictions_sha256": sha256_file(run_dir / "predictions.json"),
              "tool_errors": {"9": {"generate_candidate": "APIStatusError: Error code: 400"}}}
    (run_dir / "manifest.json").write_text(json.dumps(failed))
    out = evaluate(run_id, partial=True)
    scored = json.loads((out / "manifest.json").read_text())
    assert (scored["n"], scored["correct"]) == (1, 0)
    result = json.loads((out / "results.jsonl").read_text())
    assert (result["question_id"], result["correct"], result["pred_error"]) == ("9", False, "no prediction")
    for unfinished in ({"failures": {"9": "RuntimeError: boom"}}, {"harness_errors": {"9": ["a dead pod"]}}):
        (run_dir / "manifest.json").write_text(json.dumps({**failed, **unfinished}))
        with pytest.raises(DataError, match="completed no question"):
            evaluate(run_id, partial=True)
