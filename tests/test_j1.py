"""J1 (SPEC 6.3, D9): the per-question table read from `eval` executions, and EX per arm, per split
and per difficulty, with the sensitivity without the date-dependent golds and without the LIMIT golds."""
import json

import pytest

from bench.judge import JudgeError
from bench.judge.j1 import correct, ex_summary, ex_table

QUESTIONS = [  # question_id, difficulty, date-dependent gold, gold with LIMIT
    ("1", "simple", False, False), ("2", "simple", False, False),
    ("3", "moderate", True, False), ("4", "challenging", False, True)]


INSTRUMENT = {"fixed_date": "2026-09-30", "timeout_s": 60, "sqlite_version": "3.53.1", "prereg_hash": None}


def eval_dir(tmp_path, name, arm, split, outcomes, **manifest):
    run_dir = tmp_path / name
    run_dir.mkdir()
    rows = [{"question_id": q, "difficulty": d, "correct": ok, "gold_date_substituted": dd, "gold_has_limit": lim,
             "gold_error": None, "pred_error": None} for (q, d, dd, lim), ok in zip(QUESTIONS, outcomes)]
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": name, "type": "eval", "arm": arm, "engine": None, "split": split, "source_run_id": f"agent-{arm}",
        "n": len(rows), **INSTRUMENT, **manifest}))
    return run_dir


def test_the_table_has_one_row_per_arm_and_question(tmp_path):
    rows = ex_table([eval_dir(tmp_path, "e0", "B0", "calib", [True, False, True, True])])
    assert rows[2] == {"arm": "B0", "engine": None, "split": "calib", "question_id": "3", "difficulty": "moderate",
                       "correct": True, "date_dependent": True, "limit": False, "gold_error": False,
                       "source_run_id": "agent-B0"}


def test_ex_per_arm_per_difficulty_and_the_sensitivity(tmp_path):
    rows = ex_table([eval_dir(tmp_path, "e0", "B0", "calib", [True, False, True, False]),
                     eval_dir(tmp_path, "e4", "B4", "calib", [False, True, False, False])])
    b0, b4 = ex_summary(rows)
    assert (b0["arm"], b0["engine"], b0["split"], b0["n"], b0["correct"], b0["ex"]) == ("B0", None, "calib", 4, 2, 0.5)
    assert b0["by_difficulty"] == {"simple": {"n": 2, "correct": 1, "ex": 0.5},
                                   "moderate": {"n": 1, "correct": 1, "ex": 1.0},
                                   "challenging": {"n": 1, "correct": 0, "ex": 0.0}}
    assert b0["without_date_dependent"] == {"n": 3, "correct": 1, "ex": pytest.approx(1 / 3)}  # 1, 2, 4
    assert b0["without_limit"] == {"n": 3, "correct": 2, "ex": pytest.approx(2 / 3)}           # 1, 2, 3
    assert (b4["arm"], b4["correct"], b4["ex"], b4["gold_errors"]) == ("B4", 1, 0.25, 0)
    assert correct(rows, "B4", "calib") == {"1": False, "2": True, "3": False, "4": False}


def test_the_two_single_call_engines_of_b2_are_two_configurations(tmp_path):
    rows = ex_table([eval_dir(tmp_path, "p", "B2", "calib", [True] * 4, engine="production_llm"),
                     eval_dir(tmp_path, "c", "B2", "calib", [False] * 4, engine="cheap_alt")])
    assert [(s["arm"], s["engine"], s["ex"]) for s in ex_summary(rows)] == \
        [("B2", "cheap_alt", 0.0), ("B2", "production_llm", 1.0)]
    assert correct(rows, "B2", "calib", engine="production_llm") == {q: True for q in ("1", "2", "3", "4")}


@pytest.mark.parametrize("breakage", ["duplicate", "per_call", "count", "no_arm", "instrument", "prereg", "pre_f2"])
def test_the_table_refuses_ambiguous_or_broken_evaluations(tmp_path, breakage):
    dirs = [eval_dir(tmp_path, "e0", "B0", "calib", [True] * 4)]
    if breakage == "duplicate":  # the same arm and split scored twice: which one counts?
        dirs.append(eval_dir(tmp_path, "e1", "B0", "calib", [False] * 4))
    elif breakage == "per_call":
        dirs = [eval_dir(tmp_path, "p", "B0", "calib", [True] * 4, per_call=True)]
    elif breakage == "count":
        dirs = [eval_dir(tmp_path, "c", "B0", "calib", [True] * 4, n=5)]
    elif breakage == "no_arm":
        dirs = [eval_dir(tmp_path, "a", None, "calib", [True] * 4)]
    elif breakage == "instrument":  # two arms of one split measured with different SQLite: not comparable
        dirs.append(eval_dir(tmp_path, "e4", "B4", "calib", [True] * 4, sqlite_version="3.45.0"))
    elif breakage == "prereg":  # two arms of one split scored under two pre-registrations
        dirs = [eval_dir(tmp_path, "t0", "B0", "test", [True] * 4, prereg_hash="a" * 64),
                eval_dir(tmp_path, "t4", "B4", "test", [True] * 4, prereg_hash="b" * 64)]
    else:  # an eval written before the fixed date existed
        dirs = [eval_dir(tmp_path, "old", "B0", "calib", [True] * 4, fixed_date=None)]
    with pytest.raises(JudgeError):
        ex_table(dirs)
