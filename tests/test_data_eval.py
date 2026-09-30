"""Splits (bench data), pairing by id and timeout (bench eval), CHESS's final-SQL rule."""
import random
import sqlite3
from types import SimpleNamespace

import pytest

from bench import paths
from bench.agent.runner import final_sql
from bench.contracts.config import load_config
from bench.data import DataError, build_splits
from bench.evaluate import execute, score

CONFIG = load_config(paths.ROOT / "config.yaml")


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


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE t (id INTEGER, v TEXT)")
    connection.executemany("INSERT INTO t VALUES (?, ?)", [(i, f"v{i}") for i in range(10)])
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


def test_final_sql_is_the_first_sql_of_the_last_key():
    def info(sql):
        return SimpleNamespace(SQL=sql)
    state = SimpleNamespace(SQL_meta_infos={"generate_candidate": [info("a")], "revise_1": [info("b"), info("c")]})
    assert final_sql(state) == "b"
    assert final_sql(SimpleNamespace(SQL_meta_infos={})) is None
    assert final_sql(SimpleNamespace(SQL_meta_infos={"generate_candidate": [info("a")], "revise_1": []})) is None
