"""The evaluator's fixed date: `'now'`, CURRENT_TIMESTAMP, CURRENT_DATE (and the forms SQLite reads as
'now') become the pre-registered `eval.fixed_date` in prediction and gold alike, whatever the day
and the machine's time zone, and every result says whether a substitution happened. Golds with
LIMIT are flagged too, for the sensitivity analysis."""
import datetime
import os
import sqlite3
import time

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


@pytest.mark.parametrize("value", [None, "", "2026-9-30", "20260930", "2026-W40-3", "2026-09-30T00:00"])
def test_the_fixed_date_must_be_a_plain_iso_date(value):
    from bench.data import DataError
    from bench.evaluate import fixed_date
    with pytest.raises(DataError, match="fixed_date"):
        fixed_date({"eval": {"fixed_date": value}})
    assert fixed_date({"eval": {"fixed_date": "2026-09-30"}}) == "2026-09-30"


@pytest.mark.parametrize("sql,expected", [
    ("SELECT STRFTIME('%Y', CURRENT_TIMESTAMP)", f"SELECT STRFTIME('%Y', {STAMP})"),
    ("SELECT current_date, CURRENT_TIME", f"SELECT '{DAY}', '00:00:00'"),
    ("SELECT STRFTIME('%Y', date('NOW'))", f"SELECT STRFTIME('%Y', date({STAMP}))"),
    ("SELECT julianday('now') - julianday(\"now\")", f"SELECT julianday({STAMP}) - julianday({STAMP})"),
    ("SELECT date(), datetime( ), julianday(), unixepoch(), time()",
     f"SELECT date({STAMP}), datetime({STAMP}), julianday({STAMP}), unixepoch({STAMP}), time({STAMP})"),
    ("SELECT strftime('%Y') - strftime('%Y', dob)", f"SELECT strftime('%Y', {STAMP}) - strftime('%Y', dob)"),
    ('SELECT strftime("%Y") - 1', f'SELECT strftime("%Y", {STAMP}) - 1'),
    ("SELECT datetime('now', 'localtime'), time( 'now' ), unixepoch('NOW'), strftime(\"%Y\", 'now')",
     f"SELECT datetime({STAMP}, 'localtime'), time( {STAMP} ), unixepoch({STAMP}), strftime(\"%Y\", {STAMP})"),
    # an apostrophe inside a comment opens no string literal
    ("SELECT -- the member's age\n STRFTIME('%Y', 'now') /* it's */ - CURRENT_DATE",
     f"SELECT -- the member's age\n STRFTIME('%Y', {STAMP}) /* it's */ - '{DAY}'"),
])
def test_now_and_its_synonyms_become_the_fixed_date(sql, expected):
    assert fix_date(sql, DAY) == (expected, True)


@pytest.mark.parametrize("sql", [
    "SELECT 'CURRENT_DATE', 'now()', 'right now', \"CURRENT_TIMESTAMP\" FROM t",  # literals and identifiers
    "SELECT current_dates, my_date() FROM [current_date] WHERE x LIKE '%now%'",
    "SELECT date(dob), strftime('%Y', dob) FROM t",
    "SELECT dob FROM t -- as of CURRENT_DATE, i.e. date('now')\n",
    "SELECT * FROM t WHERE end_date < 'now'",           # not a time value: a TEXT comparison in SQLite
    "SELECT coalesce(end_date, 'now'), 'now' FROM t",   # an argument, but not of a date function
    "SELECT strftime('now', dob) FROM t",               # the format, not the time value
])
def test_only_the_current_moment_is_substituted(sql):
    assert fix_date(sql, DAY) == (sql, False)


def test_a_bare_now_is_text_and_stays_text(db):
    # every ISO date sorts before the text 'now'; read as a date at 1990-01-01 it would count 2, not 3
    gold = {"1": {"db_id": "t", "SQL": "SELECT count(*) FROM p WHERE birthday < 'now'"}}
    result = score({"1": "SELECT 3"}, gold, lambda _: db, 5, "1990-01-01")[0]
    assert (result["correct"], result["gold_date_substituted"]) == (True, False)
    dated = score({"1": "SELECT 2"}, {"1": {"db_id": "t", "SQL": "SELECT count(*) FROM p WHERE birthday < date('now')"}},
                  lambda _: db, 5, "1990-01-01")[0]
    assert (dated["correct"], dated["gold_date_substituted"]) == (True, True)


@pytest.mark.parametrize("sql", ["-- x", "   ", "/* nothing */", "\n-- only a comment\n"])
def test_no_statement_is_an_execution_error(db, sql):
    assert execute(db, sql, 5) == (None, "no statement")


def test_no_statement_is_not_an_empty_answer(db):
    # [] from a comment is not the zero rows of a query: neither side may score it as an answer
    empty = "SELECT id FROM p WHERE id > 99"
    gold = {"1": {"db_id": "t", "SQL": empty}, "2": {"db_id": "t", "SQL": "-- x"}}
    results = score({"1": "-- the model answered with a comment", "2": empty}, gold, lambda _: db, 5, DAY)
    assert [(r["correct"], r["pred_error"], r["gold_error"]) for r in results] == \
        [(False, "no statement", None), (False, None, "no statement")]


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


@pytest.fixture
def west_of_greenwich():
    """The process in a zone behind UTC, applied with tzset (glibc's localtime_r reads it only then),
    and restored the same way so no later test inherits it."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "America/Sao_Paulo"
    time.tzset()
    yield
    if previous is None:
        del os.environ["TZ"]
    else:
        os.environ["TZ"] = previous
    time.tzset()


def test_the_machine_time_zone_does_not_move_the_date(db, west_of_greenwich):
    # at midnight UTC, 'localtime' west of Greenwich is the day before: the evaluator pins UTC
    assert sqlite3.connect(":memory:").execute("SELECT date('2026-01-01 00:00:00', 'localtime')").fetchone() == \
        ("2025-12-31",)  # the zone is in force outside the evaluator
    gold = {"1": {"db_id": "t", "SQL": "SELECT date('now')"}}
    results = score({"1": "SELECT date('now', 'localtime')"}, gold, lambda _: db, 5, "2026-01-01")
    assert results[0]["correct"] is True
    rows, _ = execute(db, "SELECT date('2026-01-01 00:00:00', 'localtime')", 5)
    assert rows == [("2026-01-01",)]


@pytest.mark.parametrize("sql,expected", [
    ("SELECT name FROM t ORDER BY dob DESC LIMIT 1", True),
    ("select name from t order by dob limit 1 offset 2", True),
    ("SELECT 'no LIMIT here' FROM t", False),
    ("SELECT name FROM t -- LIMIT 1 would be wrong", False),
    ("SELECT name FROM t", False),
])
def test_golds_with_limit_are_flagged(sql, expected):
    assert gold_has_limit(sql) is expected


def test_every_result_carries_the_sensitivity_flags(db):
    gold = {"1": {"db_id": "t", "SQL": "SELECT id FROM p ORDER BY birthday LIMIT 1"}}
    result = score({"1": "SELECT min(id) FROM p"}, gold, lambda _: db, 5, DAY)[0]
    assert (result["gold_has_limit"], result["gold_date_substituted"], result["pred_date_substituted"]) == \
        (True, False, False)
