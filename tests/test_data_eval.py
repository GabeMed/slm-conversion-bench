"""Splits and pinned inputs (bench data), the evaluator (bench eval), the barrier at the commands,
and CHESS's final-SQL rule."""
import copy
import hashlib
import json
import random
import sqlite3
from types import SimpleNamespace

import pytest
import yaml

from bench import data, paths
from bench.agent.runner import final_sql, run_agent
from bench.barrier import TestSplitLocked
from bench.contracts.config import config_sha256, load_config
from bench.data import DataError, build_splits, calib_sample
from bench.evaluate import evaluate, execute, score

CONFIG = load_config(paths.ROOT / "config.yaml")


# ---------------------------------------------------------------- splits

def synthetic(test_size=498):
    dev = [{"question_id": str(i), "db_id": f"db{i % 11}"} for i in range(1534)]
    mini = random.Random(0).sample(range(1534), 500)
    excluded = [str(i) for i in mini[:2]]
    test = [{"question_id": str(i), "db_id": f"db{i % 11}"} for i in mini[2:2 + test_size]]
    return dev, test, excluded


def config_with(excluded):
    return {**CONFIG, "splits": {**CONFIG["splits"], "excluded": excluded}}


def test_splits_partition_bird_dev():
    dev, test, excluded = synthetic()
    splits = build_splits(config_with(excluded), dev, test)
    assert (len(splits["train"]), len(splits["calib"]), len(splits["test"])) == (834, 200, 498)
    groups = [set(splits[k]) for k in ("train", "calib", "test", "excluded")]
    assert sum(map(len, groups)) == 1534 and set().union(*groups) == {q["question_id"] for q in dev}
    assert build_splits(config_with(excluded), dev, test) == splits  # seeded


def test_calib_sample_does_not_depend_on_python_random():
    pool = [str(i) for i in range(50)]
    expected = sorted(sorted(pool, key=lambda q: hashlib.sha256(f"7:{q}".encode()).hexdigest())[:5], key=int)
    assert calib_sample(pool, 5, 7) == expected
    assert calib_sample(list(reversed(pool)), 5, 7) == expected


def test_the_committed_splits():
    splits = json.loads(paths.SPLITS.read_text())
    assert {k: len(splits[k]) for k in ("train", "calib", "test", "excluded")} == \
        {"train": 834, "calib": 200, "test": 498, "excluded": 2}
    groups = [set(splits[k]) for k in ("train", "calib", "test", "excluded")]
    assert sum(map(len, groups)) == len(set().union(*groups)) == 1534
    assert splits["excluded"] == ["119", "120"] == CONFIG["splits"]["excluded"]
    pool = splits["train"] + splits["calib"]
    assert splits["calib"] == calib_sample(pool, CONFIG["splits"]["calib_size"], CONFIG["seeds"]["calib_split"])


@pytest.mark.parametrize("breakage", ["missing_dev", "excluded_in_test", "db_mismatch", "short_test"])
def test_splits_refuse_inconsistent_inputs(breakage):
    dev, test, excluded = synthetic()
    if breakage == "missing_dev":
        dev = dev[:-1]
    elif breakage == "excluded_in_test":
        test[0] = {"question_id": excluded[0], "db_id": dev[int(excluded[0])]["db_id"]}
    elif breakage == "db_mismatch":
        test[0] = {**test[0], "db_id": "other"}
    else:
        test = test[:-1]
    with pytest.raises(DataError):
        build_splits(config_with(excluded), dev, test)


# ---------------------------------------------------------------- execution

@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE t (id INTEGER, v TEXT)")
    connection.executemany("INSERT INTO t VALUES (?, ?)", [(i, f"v{i}") for i in range(12)])
    connection.commit()
    connection.close()
    return path


def test_pairing_is_by_id_not_by_position(db):
    # the gold file order differs from the numeric order the predictions are scored in
    gold = {q: {"question_id": q, "db_id": "t", "SQL": f"SELECT v FROM t WHERE id = {q}"} for q in ("9", "1", "10")}
    predictions = {"10": "SELECT v FROM t WHERE id = 10", "1": "SELECT v FROM t WHERE id = 1",
                   "9": "SELECT v FROM t WHERE id = 8"}
    results = {r["question_id"]: r for r in score(predictions, gold, lambda _: db, 5)}
    assert {q: r["correct"] for q, r in results.items()} == {"1": True, "9": False, "10": True}
    assert all(r["gold_sql"] == gold[q]["SQL"] for q, r in results.items())


def test_rows_compare_as_sets_and_missing_prediction_is_wrong(db):
    gold = {"1": {"db_id": "t", "SQL": "SELECT v FROM t WHERE id < 3 ORDER BY id"},
            "2": {"db_id": "t", "SQL": "SELECT 1"}}
    results = score({"1": "SELECT v FROM t WHERE id < 3 ORDER BY id DESC", "2": None}, gold, lambda _: db, 5)
    assert [r["correct"] for r in results] == [True, False]
    assert results[1]["pred_error"] == "no prediction"


def test_timeout_and_read_only(db):
    endless = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r"
    assert execute(db, endless, 0.2) == (None, "timeout")
    rows, error = execute(db, "DELETE FROM t", 5)
    assert rows is None and "readonly" in error


# ---------------------------------------------------------------- the commands, on a synthetic repository

def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def repo(tmp_path, monkeypatch, db):
    """A repository with one tiny database, pinned inputs, splits, and one finished run."""
    for name, value in {"ROOT": tmp_path, "DATA": tmp_path / "data", "RAW": tmp_path / "data" / "raw",
                        "SPLITS": tmp_path / "data" / "splits.json", "RUNS": tmp_path / "runs",
                        "DATA_MANIFEST": tmp_path / "data" / "MANIFEST.json"}.items():
        monkeypatch.setattr(paths, name, value)
    config = copy.deepcopy(CONFIG)
    root = paths.bird_root(config) / "dev_databases" / "tiny"
    root.mkdir(parents=True)
    (root / "tiny.sqlite").write_bytes(db.read_bytes())
    paths.RAW.mkdir(parents=True)
    dev = [{"question_id": q, "db_id": "tiny", "question": "?", "evidence": None, "difficulty": "simple",
            "SQL": f"SELECT v FROM t WHERE id = {q}"} for q in (1, 2, 3)]
    test = [{"question_id": "9", "db_id": "tiny", "question": "?", "evidence": "", "SQL": "SELECT 9"}]
    for name, items in (("bird_dev_questions", dev), ("plat_sql_test", test)):
        path = paths.RAW / f"{name}.json"
        path.write_text(json.dumps(items))
        config["data"][name]["sha256"] = _sha(path)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    paths.SPLITS.write_text(json.dumps({"train": ["1", "2", "3"], "calib": [], "test": ["9"], "excluded": []}))
    paths.DATA_MANIFEST.write_text(json.dumps({"inputs": {}, "databases": {
        "tiny": {"sqlite": _sha(root / "tiny.sqlite"), "descriptions": {}}}}))
    return tmp_path, config


def _finished_run(repo, split="train", status="done", predictions=None, question_ids=None):
    tmp_path, config = repo
    predictions = predictions if predictions is not None else {"1": "SELECT v FROM t WHERE id = 1",
                                                               "2": "SELECT v FROM t WHERE id = 5"}
    run_dir = paths.RUNS / f"agent-B0-{split}-x"
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.json").write_text(json.dumps(predictions))
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_dir.name, "split": split, "status": status, "commit": "c",
        "question_ids": question_ids or sorted(predictions), "config_path": "config.yaml",
        "config_sha256": config_sha256(load_config(tmp_path / "config.yaml"))}))
    return run_dir.name


def test_evaluate_scores_a_finished_run(repo):
    out = evaluate(_finished_run(repo))
    results = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [(r["question_id"], r["correct"]) for r in results] == [("1", True), ("2", False)]
    recorded = json.loads(paths.DATA_MANIFEST.read_text())["databases"]["tiny"]["sqlite"]
    assert json.loads((out / "manifest.json").read_text())["databases"] == {"tiny": recorded}


def test_evaluate_refuses_a_changed_database(repo):
    run_id = _finished_run(repo)
    sqlite = paths.bird_root(repo[1]) / "dev_databases" / "tiny" / "tiny.sqlite"
    connection = sqlite3.connect(sqlite)
    connection.execute("INSERT INTO t VALUES (99, 'x')")
    connection.commit()
    connection.close()
    with pytest.raises(DataError, match="tiny.sqlite"):
        evaluate(run_id)


def test_evaluate_refuses_a_changed_gold_file(repo):
    run_id = _finished_run(repo)
    path = paths.RAW / "bird_dev_questions.json"
    path.write_text(path.read_text().replace("id = 2", "id = 5"))
    with pytest.raises(DataError, match="differs from the pin"):
        evaluate(run_id)


@pytest.mark.parametrize("status,ids", [("interrupted", None), ("failed", None), ("done", ["1", "2", "3"])])
def test_evaluate_refuses_incomplete_runs(repo, status, ids):
    with pytest.raises(DataError):
        evaluate(_finished_run(repo, status=status, question_ids=ids))


def test_the_test_split_is_refused_by_every_command(repo):
    with pytest.raises(TestSplitLocked):
        evaluate(_finished_run(repo, split="test", predictions={"9": "SELECT 9"}))
    with pytest.raises(TestSplitLocked):
        run_agent(str(repo[0] / "config.yaml"), "B0", "test", limit=1)
    with pytest.raises(TestSplitLocked):
        data.questions_for(repo[1], "test")
    assert not any(paths.RUNS.glob("agent-B0-test-2*"))  # refused before a run directory exists


def test_run_refuses_ids_outside_the_split_and_duplicates(repo):
    with pytest.raises(DataError, match="not in the train split"):
        run_agent(str(repo[0] / "config.yaml"), "B0", "train", ids=["9"])
    with pytest.raises(DataError, match="more than once"):
        run_agent(str(repo[0] / "config.yaml"), "B0", "train", ids=["1", "1"])


# ---------------------------------------------------------------- CHESS's final SQL

def test_final_sql_is_the_first_sql_of_the_last_key():
    def info(sql):
        return SimpleNamespace(SQL=sql)
    state = SimpleNamespace(SQL_meta_infos={"generate_candidate": [info("a")], "revise_1": [info("b"), info("c")]})
    assert final_sql(state) == "b"
    assert final_sql(SimpleNamespace(SQL_meta_infos={})) is None
    assert final_sql(SimpleNamespace(SQL_meta_infos={"generate_candidate": [info("a")], "revise_1": []})) is None


def test_a_run_is_done_only_when_nothing_went_wrong():
    from bench.agent.runner import run_status
    assert run_status({"1": "x", "2": None}, ["1", "2"], {}, []) == "done"
    assert run_status({"1": "x"}, ["1", "2"], {}, []) == "interrupted"
    assert run_status({"1": "x"}, ["1"], {"1": ["route: FactError"]}, []) == "failed"
    assert run_status({"1": "x"}, ["1"], {}, ["tiny"]) == "failed"
