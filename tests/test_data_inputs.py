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


def test_golds_that_do_not_execute_are_listed(tmp_path):
    db = tmp_path / "t.sqlite"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE t (id INTEGER)")
    connection.commit()
    connection.close()
    questions = {"train": [{"question_id": "2", "db_id": "t", "SQL": "SELECT nope FROM t"},
                           {"question_id": "1", "db_id": "t", "SQL": "SELECT id FROM t WHERE date('now') > '2000'"}],
                 "calib": [],
                 "test": [{"question_id": "9", "db_id": "t", "SQL": "SELECT * FROM missing"}]}
    check = check_golds(questions, lambda _: db, 5, DAY)
    assert check["checked"] == {"train": 2, "calib": 0, "test": 1}
    assert [(f["split"], f["question_id"]) for f in check["failures"]] == [("train", "2"), ("test", "9")]
    assert "no such column" in check["failures"][0]["error"] and check["fixed_date"] == DAY


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
    assert summary == {"databases": 1, "train": 834, "calib": 200, "test": 498, "gold_failures": 1}
    manifest = json.loads(paths.DATA_MANIFEST.read_text())
    assert set(manifest["inputs"]) == {"bird_dev_databases", "bird_dev_questions", "plat_sql_test", "mini_dev"}
    check = manifest["gold_check"]
    assert check["checked"] == {"train": 834, "calib": 200, "test": 498}
    assert [f["question_id"] for f in check["failures"]] == ["7"]
    assert {q["difficulty"] for q in data._test_questions(full_repo)} == {"moderate"}
    assert data.run(full_repo) == summary  # rebuilding gives the same facts


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
    assert check["checked"] == {"train": 834, "calib": 200, "test": 498} and check["fixed_date"] == DAY
    assert set(manifest["inputs"]) == set(CONFIG["data"])
    assert all(manifest["inputs"][name]["sha256"] == CONFIG["data"][name]["sha256"] for name in CONFIG["data"])
