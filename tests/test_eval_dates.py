"""The evaluator's fixed date: `'now'`, CURRENT_TIMESTAMP, CURRENT_DATE (and the forms SQLite reads as
'now') become the pre-registered `eval.fixed_date` in prediction and gold alike, whatever the day
and the machine's time zone, and every result says whether a substitution happened. Golds with
LIMIT are flagged too, for the sensitivity analysis."""
import datetime
import sqlite3

import pytest

from bench import paths
from bench.contracts.config import load_config
from bench.evaluate import execute, fix_date, gold_has_limit, score

CONFIG = load_config(paths.ROOT / "config.yaml")
DAY = "2026-09-30"
STAMP = f"'{DAY} 00:00:00'"


def test_the_fixed_date_is_preregistered_as_an_iso_date_string():
    fixed = CONFIG["eval"]["fixed_date"]
    assert isinstance(fixed, str)  # an unquoted YAML date would load as datetime.date and break the config hash
    assert datetime.date.fromisoformat(fixed).isoformat() == fixed


@pytest.mark.parametrize("sql,expected", [
    ("SELECT STRFTIME('%Y', CURRENT_TIMESTAMP)", f"SELECT STRFTIME('%Y', {STAMP})"),
    ("SELECT current_date, CURRENT_TIME", f"SELECT '{DAY}', '00:00:00'"),
    ("SELECT STRFTIME('%Y', date('NOW'))", f"SELECT STRFTIME('%Y', date({STAMP}))"),
    ("SELECT julianday('now') - julianday(\"now\")", f"SELECT julianday({STAMP}) - julianday({STAMP})"),
    ("SELECT date(), datetime( ), julianday(), unixepoch(), time()",
     f"SELECT date({STAMP}), datetime({STAMP}), julianday({STAMP}), unixepoch({STAMP}), time({STAMP})"),
    ("SELECT strftime('%Y') - strftime('%Y', dob)", f"SELECT strftime('%Y', {STAMP}) - strftime('%Y', dob)"),
])
def test_now_and_its_synonyms_become_the_fixed_date(sql, expected):
    assert fix_date(sql, DAY) == (expected, True)


@pytest.mark.parametrize("sql", [
    "SELECT 'CURRENT_DATE', 'now()', 'right now', \"CURRENT_TIMESTAMP\" FROM t",  # literals and identifiers
    "SELECT current_dates, my_date() FROM [current_date] WHERE x LIKE '%now%'",
    "SELECT date(dob), strftime('%Y', dob) FROM t",
])
def test_only_the_current_moment_is_substituted(sql):
    assert fix_date(sql, DAY) == (sql, False)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE p (id INTEGER, birthday TEXT)")
    connection.executemany("INSERT INTO p VALUES (?, ?)", [(1, "1950-10-01"), (2, "1980-01-15"), (3, "2000-09-29")])
    connection.commit()
    connection.close()
    return path


def test_prediction_and_gold_both_see_the_fixed_date(db):
    # gold with CURRENT_TIMESTAMP, prediction with date('now'): equal only if both got the same date
    gold = {"1": {"db_id": "t", "SQL": "SELECT id, STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', birthday) FROM p"},
            "2": {"db_id": "t", "SQL": "SELECT count(*) FROM p"},
            "3": {"db_id": "t", "SQL": "SELECT CURRENT_DATE"}}
    predictions = {"1": "SELECT id, CAST(STRFTIME('%Y', date('now')) AS INT) - STRFTIME('%Y', birthday) FROM p",
                   "2": "SELECT count(id) FROM p",
                   "3": "SELECT '1999-12-31'"}
    results = {r["question_id"]: r for r in score(predictions, gold, lambda _: db, 5, "1999-12-31")}
    assert {q: r["correct"] for q, r in results.items()} == {"1": True, "2": True, "3": True}
    assert {q: (r["pred_date_substituted"], r["gold_date_substituted"]) for q, r in results.items()} == \
        {"1": (True, True), "2": (False, False), "3": (False, True)}
    assert results["3"]["gold_sql"] == "SELECT CURRENT_DATE"  # the record keeps the SQL as written


def test_the_machine_time_zone_does_not_move_the_date(db, monkeypatch):
    # at midnight UTC, 'localtime' west of Greenwich is the day before: the evaluator pins UTC
    monkeypatch.setenv("TZ", "America/Sao_Paulo")
    gold = {"1": {"db_id": "t", "SQL": "SELECT date('now')"}}
    results = score({"1": "SELECT date('now', 'localtime')"}, gold, lambda _: db, 5, "2026-01-01")
    assert results[0]["correct"] is True
    rows, _ = execute(db, "SELECT date('2026-01-01 00:00:00', 'localtime')", 5)
    assert rows == [("2026-01-01",)]


@pytest.mark.parametrize("sql,expected", [
    ("SELECT name FROM t ORDER BY dob DESC LIMIT 1", True),
    ("select name from t order by dob limit 1 offset 2", True),
    ("SELECT 'no LIMIT here' FROM t", False),
    ("SELECT name FROM t", False),
])
def test_golds_with_limit_are_flagged(sql, expected):
    assert gold_has_limit(sql) is expected


def test_every_result_carries_the_sensitivity_flags(db):
    gold = {"1": {"db_id": "t", "SQL": "SELECT id FROM p ORDER BY birthday LIMIT 1"}}
    result = score({"1": "SELECT min(id) FROM p"}, gold, lambda _: db, 5, DAY)[0]
    assert (result["gold_has_limit"], result["gold_date_substituted"], result["pred_date_substituted"]) == \
        (True, False, False)
