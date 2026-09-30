"""`bench data`, complete: the test questions carry the Mini-Dev difficulty, test ∪ excluded is checked
to be exactly the 500 Mini-Dev ids, and every gold of train, calib and test is executed on the pinned
databases, the ones that fail written into data/MANIFEST.json. The last tests read the real pinned
inputs and are skipped where `bench data` has not downloaded them (CI)."""
import collections
import json
import re
import sqlite3

import pytest
import yaml

from bench import data, paths
from bench.contracts.config import load_config
from bench.data import DataError, attach_difficulty, check_mini_dev
from bench.evaluate import check_golds, fix_date, gold_has_limit
from synthetic import GOLD, make_repo, sha256

CONFIG = load_config(paths.ROOT / "config.yaml")
DAY = CONFIG["eval"]["fixed_date"]


def mini_dev_of(test, excluded, difficulty="simple"):
    return [{"question_id": q["question_id"], "db_id": q["db_id"], "difficulty": difficulty} for q in test] + \
        [{"question_id": q, "db_id": "financial", "difficulty": "moderate"} for q in excluded]


def small():
    test = [{"question_id": str(i), "db_id": f"db{i % 3}"} for i in (5, 7, 9)]
    return test, ["119", "120"], mini_dev_of(test, ["119", "120"])


def test_the_mini_dev_source_is_pinned_at_a_revision():
    pin = CONFIG["data"]["mini_dev"]
    assert re.search(r"/resolve/[0-9a-f]{40}/", pin["url"]) and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])


def test_mini_dev_must_be_exactly_test_plus_excluded():
    test, excluded, mini_dev = small()
    check_mini_dev(test, excluded, mini_dev)
    for broken in (mini_dev[:-1],                                                   # an exclusion not in Mini-Dev
                   mini_dev + [{"question_id": "11", "db_id": "db2", "difficulty": "simple"}],  # an id in neither
                   mini_dev + mini_dev[:1],                                         # a duplicate
                   [{**mini_dev[0], "db_id": "other"}] + mini_dev[1:],              # a database mismatch
                   [{**mini_dev[0], "difficulty": "hard"}] + mini_dev[1:]):         # an unknown difficulty
        with pytest.raises(DataError):
            check_mini_dev(test, excluded, broken)


def test_test_questions_take_the_mini_dev_difficulty_by_id():
    test, _, mini_dev = small()
    mini_dev[1]["difficulty"] = "challenging"
    joined = attach_difficulty(test, list(reversed(mini_dev)))
    assert [(q["question_id"], q["difficulty"]) for q in joined] == [("5", "simple"), ("7", "challenging"), ("9", "simple")]
    with pytest.raises(DataError, match="no Mini-Dev difficulty"):
        attach_difficulty(test + [{"question_id": "13", "db_id": "db1"}], mini_dev)


def test_the_gold_check_registers_errors_and_only_observes_the_rest(tmp_path):
    db = tmp_path / "t.sqlite"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE t (id INTEGER)")
    connection.commit()
    connection.close()
    endless = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r"
    questions = {"train": [{"question_id": "2", "db_id": "t", "SQL": "SELECT nope FROM t"},
                           {"question_id": "1", "db_id": "t", "SQL": "SELECT id FROM t WHERE date('now') > '2000'"},
                           {"question_id": "4", "db_id": "t", "SQL": endless}],
                 "calib": [{"question_id": "5", "db_id": "t", "SQL": "-- nothing"}],
                 "test": [{"question_id": "9", "db_id": "t", "SQL": "SELECT * FROM missing"}]}
    check = check_golds(questions, lambda _: db, 0.3, DAY)
    registered, observed = check["registered"], check["observed"]
    # what reproduces anywhere: the golds that fail with an execution error
    assert registered["checked"] == {"train": 3, "calib": 1, "test": 1} and registered["fixed_date"] == DAY
    assert [(e["split"], e["question_id"]) for e in registered["errors"]] == [("train", "2"), ("calib", "5"), ("test", "9")]
    assert "no such column" in registered["errors"][0]["error"] and registered["errors"][1]["error"] == "no statement"
    # what this machine saw: the timeout, the golds with zero rows, the SQLite
    assert [(t["split"], t["question_id"]) for t in observed["timeouts"]] == [("train", "4")]
    assert [(e["split"], e["question_id"]) for e in observed["empty"]] == [("train", "1")]
    assert (observed["sqlite_version"], observed["timeout_s"]) == (sqlite3.sqlite_version, 0.3)


# ---------------------------------------------------------------- `bench data` end to end, offline

@pytest.fixture
def full_repo(tmp_path, monkeypatch):
    """A synthetic repository with inputs of the real sizes (1534 dev, 498 test, 500 Mini-Dev) on the
    tiny database, every file already in data/raw with its pin, so `bench data` needs no network."""
    _, config_path, _ = make_repo(tmp_path, monkeypatch)
    raw = yaml.safe_load(config_path.read_text())
    dev = [{"question_id": i, "db_id": "tiny", "question": "?", "evidence": None, "difficulty": "simple",
            "SQL": "SELECT nope FROM gas_t" if i == 7 else GOLD} for i in range(1534)]
    mini = list(range(1000, 1500))
    test = [{"question_id": str(i), "db_id": "tiny", "question": "?", "evidence": "", "SQL": GOLD} for i in mini[:498]]
    mini_dev = [{"question_id": i, "db_id": "tiny", "difficulty": "moderate"} for i in mini]
    (paths.RAW / "bird_dev_databases.zip").write_bytes(b"stands for dev.zip; already extracted")
    for name, items in (("bird_dev_questions", dev), ("plat_sql_test", test), ("mini_dev", mini_dev)):
        (paths.RAW / f"{name}.json").write_text(json.dumps(items))
    for name in ("bird_dev_questions", "plat_sql_test", "mini_dev", "bird_dev_databases"):
        suffix = ".zip" if name == "bird_dev_databases" else ".json"
        raw["data"][name]["sha256"] = sha256(paths.RAW / f"{name}{suffix}")
    raw["splits"]["excluded"] = [str(i) for i in mini[498:]]
    config_path.write_text(yaml.safe_dump(raw))
    paths.SPLITS.unlink()
    paths.DATA_MANIFEST.unlink()
    return load_config(config_path)


def test_bench_data_records_the_gold_check_and_joins_difficulty(full_repo):
    summary = data.run(full_repo)
    assert summary == {"databases": 1, "train": 834, "calib": 200, "test": 498,
                       "gold_errors": 1, "gold_timeouts": 0, "gold_empty": 0}
    manifest = json.loads(paths.DATA_MANIFEST.read_text())
    assert set(manifest["inputs"]) == {"bird_dev_databases", "bird_dev_questions", "plat_sql_test", "mini_dev"}
    check = manifest["gold_check"]
    assert check["checked"] == {"train": 834, "calib": 200, "test": 498}
    assert [e["question_id"] for e in check["errors"]] == ["7"]
    assert manifest["gold_observed"]["sqlite_version"] == sqlite3.sqlite_version
    assert {q["difficulty"] for q in data._test_questions(full_repo)} == {"moderate"}
    assert data.run(full_repo) == summary  # rebuilding gives the same facts


def test_bench_data_refuses_a_mini_dev_that_is_not_test_plus_exclusions(full_repo):
    path = paths.RAW / "mini_dev.json"
    items = json.loads(path.read_text())
    items[-1]["question_id"] = 1600  # a Mini-Dev whose second exclusion is not the configured one
    path.write_text(json.dumps(items))
    config = {**full_repo, "data": {**full_repo["data"], "mini_dev": {**full_repo["data"]["mini_dev"], "sha256": sha256(path)}}}
    with pytest.raises(DataError, match="Mini-Dev ids"):
        data.run(config)


def test_what_another_machine_observes_neither_refuses_nor_rewrites(full_repo):
    data.run(full_repo)
    manifest = json.loads(paths.DATA_MANIFEST.read_text())
    manifest["gold_observed"]["sqlite_version"] = "3.0.0"  # recorded with another SQLite, on a slower machine
    manifest["gold_observed"]["timeouts"] = [{"split": "test", "question_id": "1001", "db_id": "tiny"}]
    manifest["gold_check"]["errors"][0]["error"] = "OperationalError: worded by another SQLite"
    paths.DATA_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    before = paths.DATA_MANIFEST.read_bytes()
    data.run(full_repo)
    assert paths.DATA_MANIFEST.read_bytes() == before  # the registered file keeps its hash


@pytest.mark.parametrize("recorded", [
    {"errors": []},                                            # recorded when gold 7 still executed
    {"fixed_date": "2025-01-01"},                              # recorded under another fixed date
    {"checked": {"train": 834, "calib": 200, "test": 497}},   # recorded over other questions
])
def test_a_changed_gold_check_is_refused(full_repo, recorded):
    data.run(full_repo)
    manifest = json.loads(paths.DATA_MANIFEST.read_text())
    manifest["gold_check"].update(recorded)
    manifest["gold_observed"]["sqlite_version"] = "3.31.0"
    paths.DATA_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    before = paths.DATA_MANIFEST.read_bytes()
    with pytest.raises(DataError, match="gold check differs") as refused:
        data.run(full_repo)
    assert f"SQLite 3.31.0 then, {sqlite3.sqlite_version} now" in str(refused.value)  # a suspect worth naming
    assert paths.DATA_MANIFEST.read_bytes() == before


# ---------------------------------------------------------------- the real pinned inputs

REAL = all((paths.RAW / f"{name}.json").exists() for name in ("bird_dev_questions", "plat_sql_test", "mini_dev"))
real = pytest.mark.skipif(not REAL, reason="the pinned inputs are not downloaded (run `bench data`)")
DATE_DEPENDENT_TEST_GOLDS = {"1156", "1227", "1229", "1232", "1235", "1239", "1242", "1243", "1257", "1031", "898", "194"}


@real
def test_real_test_difficulty_and_mini_dev_ids():
    test = data._test_questions(CONFIG)
    assert collections.Counter(q["difficulty"] for q in test) == {"moderate": 248, "simple": 148, "challenging": 102}
    check_mini_dev(test, CONFIG["splits"]["excluded"], data._read_pinned(CONFIG, "mini_dev"))


@real
def test_real_sensitivity_sets_are_the_twelve_date_golds_and_the_limit_golds():
    test = data._test_questions(CONFIG)
    assert {q["question_id"] for q in test if fix_date(q["SQL"], DAY)[1]} == DATE_DEPENDENT_TEST_GOLDS
    assert sum(gold_has_limit(q["SQL"]) for q in test) == 86


def test_the_committed_manifest_records_the_gold_check():
    manifest = json.loads(paths.DATA_MANIFEST.read_text())
    check = manifest["gold_check"]
    assert set(check) == {"fixed_date", "checked", "errors"} and "gold_observed" in manifest
    assert check["checked"] == {"train": 834, "calib": 200, "test": 498} and check["fixed_date"] == DAY
    assert all(e["error"] != "timeout" for e in check["errors"])
    assert set(manifest["inputs"]) == set(CONFIG["data"])
    assert all(manifest["inputs"][name]["sha256"] == CONFIG["data"][name]["sha256"] for name in CONFIG["data"])
