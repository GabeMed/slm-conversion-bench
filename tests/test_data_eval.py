"""Splits and pinned inputs (bench data), the pilot stratified by difficulty, the evaluator (bench eval),
the barrier at the commands, and CHESS's final-SQL rule."""
import collections
import copy
import hashlib
import json
import random
import sqlite3
from types import SimpleNamespace

import pytest

from bench import data, paths
from bench.agent.runner import final_sql, run_agent
from bench.barrier import TestSplitLocked
from bench.contracts.config import config_sha256, load_config
from bench.data import DataError, build_splits, calib_sample, pilot_ids, pilot_sample
from bench.evaluate import evaluate, execute, score
from synthetic import GOLD, make_repo, sha256

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


def test_the_pilots_draw_is_the_ids_with_the_smallest_seeded_hash():
    calib = [str(i) for i in range(60)]
    expected = sorted(sorted(calib, key=lambda q: hashlib.sha256(f"7:pilot:{q}".encode()).hexdigest())[:5], key=int)
    assert pilot_sample(calib, 5, 7) == pilot_sample(list(reversed(calib)), 5, 7) == expected
    assert pilot_sample(calib, 5, 7) != calib_sample(calib, 5, 7)  # its own draw, not the calibration's
    with pytest.raises(DataError, match="pilot"):
        pilot_sample(calib[:3], 5, 7)


CALIB = {"simple": range(100, 112), "moderate": range(200, 206), "challenging": range(300, 304)}


def pilot_repo(tmp_path, monkeypatch, mix, size=None):
    """A synthetic repository whose calibration split has 12 simple, 6 moderate and 4 challenging
    questions, beside train questions of every difficulty; the configuration with `mix`."""
    _, _, config = make_repo(tmp_path, monkeypatch)
    train = {"simple": range(1, 9), "moderate": range(9, 17), "challenging": range(17, 25)}
    dev = [{"question_id": i, "db_id": "tiny", "question": "?", "evidence": None, "difficulty": difficulty, "SQL": GOLD}
           for ids in (train, CALIB) for difficulty, block in ids.items() for i in block]
    path = paths.RAW / "bird_dev_questions.json"
    path.write_text(json.dumps(dev))
    paths.SPLITS.write_text(json.dumps({"train": [str(i) for i in range(1, 25)], "test": ["9"], "excluded": [],
                                        "calib": [str(i) for block in CALIB.values() for i in block]}))
    config = copy.deepcopy(config)
    config["data"]["bird_dev_questions"]["sha256"] = sha256(path)
    config["stats"].update(pilot_mix=mix, pilot_size=sum(mix.values()) if size is None else size)
    return config


def test_the_pilot_has_the_configured_mix_drawn_within_each_difficulty(tmp_path, monkeypatch):
    mix = {"simple": 3, "moderate": 4, "challenging": 2}
    config = pilot_repo(tmp_path, monkeypatch, mix)
    seed = config["seeds"]["calib_split"]
    pilot = pilot_ids(config)
    assert {d: sum(int(q) in CALIB[d] for q in pilot) for d in mix} == mix and len(pilot) == len(set(pilot)) == 9
    expected = [q for d in mix for q in sorted(map(str, CALIB[d]),
                                               key=lambda q: hashlib.sha256(f"{seed}:pilot:{q}".encode()).hexdigest())[:mix[d]]]
    assert pilot == sorted(expected, key=int)  # the seeded rank, within the difficulty
    assert pilot == pilot_ids(config)
    other = {**config, "seeds": {**config["seeds"], "calib_split": seed + 1}}
    assert pilot_ids(other) != pilot and len(pilot_ids(other)) == 9
    # not the draw over the whole calibration split, which takes the difficulties as they come
    assert pilot != pilot_sample(json.loads(paths.SPLITS.read_text())["calib"], 9, seed)
    none_of_one = pilot_repo(tmp_path / "zero", monkeypatch, {"simple": 2, "moderate": 0, "challenging": 0})
    assert len(pilot_ids(none_of_one)) == 2 and all(int(q) in CALIB["simple"] for q in pilot_ids(none_of_one))


def test_a_difficulty_short_of_its_count_is_an_error_never_filled_from_another(tmp_path, monkeypatch):
    config = pilot_repo(tmp_path, monkeypatch, {"simple": 3, "moderate": 4, "challenging": 5})
    with pytest.raises(DataError, match="needs 5 challenging calibration questions, there are 4"):
        pilot_ids(config)
    config["stats"].update(pilot_mix={"simple": 3, "moderate": 4, "challenging": 4}, pilot_size=11)
    assert len(pilot_ids(config)) == 11  # every challenging question there is: enough


def test_a_mix_that_does_not_sum_to_the_pilot_size_or_misnames_a_difficulty_is_refused(tmp_path, monkeypatch):
    config = pilot_repo(tmp_path, monkeypatch, {"simple": 3, "moderate": 4, "challenging": 2}, size=10)
    with pytest.raises(DataError, match="sums to 9, not stats.pilot_size = 10"):
        pilot_ids(config)
    for mix in ({"simple": 6, "moderate": 4}, {"simple": 3, "moderate": 4, "challenging": 2, "hard": 1}):
        config["stats"].update(pilot_mix=mix, pilot_size=10)
        with pytest.raises(DataError, match="a count for each of"):
            pilot_ids(config)


@pytest.mark.skipif(not (paths.RAW / "bird_dev_questions.json").exists(),
                    reason="the pinned inputs are not downloaded (run `bench data`)")
def test_the_registered_pilot_has_the_tests_mix_on_the_committed_splits():
    """15/25/10 on the difficulty of the file the splits come from, where the calibration split has
    134 simple, 34 moderate and 32 challenging questions."""
    questions = data.questions_for(CONFIG, "calib")
    difficulty = lambda ids: dict(collections.Counter(questions[q]["difficulty"] for q in ids))  # noqa: E731
    assert difficulty(questions) == {"simple": 134, "moderate": 34, "challenging": 32}
    pilot = pilot_ids(CONFIG)
    assert difficulty(pilot) == CONFIG["stats"]["pilot_mix"] == {"simple": 15, "moderate": 25, "challenging": 10}
    assert CONFIG["stats"]["pilot_size"] == len(pilot) == len(set(pilot)) == 50


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
    results = {r["question_id"]: r for r in score(predictions, gold, lambda _: db, 5, "2026-09-30")}
    assert {q: r["correct"] for q, r in results.items()} == {"1": True, "9": False, "10": True}
    assert all(r["gold_sql"] == gold[q]["SQL"] for q, r in results.items())


def test_rows_compare_as_sets_and_missing_prediction_is_wrong(db):
    gold = {"1": {"db_id": "t", "SQL": "SELECT v FROM t WHERE id < 3 ORDER BY id"},
            "2": {"db_id": "t", "SQL": "SELECT 1"}}
    results = score({"1": "SELECT v FROM t WHERE id < 3 ORDER BY id DESC", "2": None}, gold, lambda _: db, 5, "2026-09-30")
    assert [r["correct"] for r in results] == [True, False]
    assert results[1]["pred_error"] == "no prediction"


def test_timeout_and_read_only(db):
    endless = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r"
    assert execute(db, endless, 0.2, "2026-09-30") == (None, "timeout", False)
    rows, error, _ = execute(db, "DELETE FROM t", 5, "2026-09-30")
    assert rows is None and "readonly" in error


# ---------------------------------------------------------------- the commands, on a synthetic repository

@pytest.fixture
def repo(tmp_path, monkeypatch):
    return make_repo(tmp_path, monkeypatch)


def _finished_run(repo, split="train", status="done", predictions=None, question_ids=None, config=None):
    config = config or repo[2]
    predictions = predictions if predictions is not None else {"1": GOLD, "2": "SELECT 0"}
    run_dir = paths.RUNS / f"agent-B0-{split}-x"
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.json").write_text(json.dumps(predictions))
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_dir.name, "arm": "B0", "split": split, "status": status, "commit": "c",
        "question_ids": question_ids or sorted(predictions), "config_path": "config.yaml",
        "config_sha256": config_sha256(config)}))
    return run_dir.name


def test_evaluate_scores_a_finished_run(repo):
    out = evaluate(_finished_run(repo))
    results = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [(r["question_id"], r["correct"]) for r in results] == [("1", True), ("2", False)]
    recorded = json.loads(paths.DATA_MANIFEST.read_text())["databases"]["tiny"]["sqlite"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["databases"] == {"tiny": recorded}
    assert (manifest["arm"], manifest["split"], manifest["per_call"] if "per_call" in manifest else False) == ("B0", "train", False)
    assert manifest["fixed_date"] == repo[2]["eval"]["fixed_date"] and manifest["sqlite_version"] == sqlite3.sqlite_version


def test_evaluate_records_the_engine_of_a_single_call_run(repo):
    run_id = _finished_run(repo)
    manifest_path = paths.RUNS / run_id / "manifest.json"
    manifest_path.write_text(json.dumps({**json.loads(manifest_path.read_text()), "arm": "B2", "mode": "single_call",
                                         "engine": "cheap_alt"}))
    manifest = json.loads((evaluate(run_id) / "manifest.json").read_text())
    assert (manifest["arm"], manifest["engine"], manifest["mode"]) == ("B2", "cheap_alt", "single_call")


def test_evaluate_uses_the_runs_configuration_not_todays(repo):
    run_id = _finished_run(repo)
    (repo[0] / "config.yaml").write_text("changed: after the run\n")  # e.g. a fact pointed at in `arms`
    assert evaluate(run_id).exists()
    snapshot = paths.RUNS / run_id / "config.json"
    tampered = json.loads(snapshot.read_text())
    tampered["eval"]["timeout_s"] += 1
    snapshot.write_text(json.dumps(tampered))
    with pytest.raises(DataError, match="snapshot"):
        evaluate(run_id)


def test_evaluate_refuses_a_snapshot_without_a_fixed_date(repo):
    undated = {**repo[2], "eval": {**repo[2]["eval"], "fixed_date": None}}
    with pytest.raises(DataError, match="fixed_date"):
        evaluate(_finished_run(repo, config=undated))
    assert not any(paths.RUNS.glob("eval-*"))  # no fallback to today, and nothing written


def test_evaluate_refuses_a_changed_database(repo):
    run_id = _finished_run(repo)
    connection = sqlite3.connect(paths.sqlite_path(repo[2], "tiny"))
    connection.execute("INSERT INTO gas_t VALUES (99, 'CZE', 'Premium')")
    connection.commit()
    connection.close()
    with pytest.raises(DataError, match="tiny.sqlite"):
        evaluate(run_id)


def test_evaluate_refuses_a_changed_gold_file(repo):
    run_id = _finished_run(repo)
    path = paths.RAW / "bird_dev_questions.json"
    path.write_text(path.read_text().replace("Premium", "Value"))
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
        run_agent(str(repo[1]), "B0", "test", limit=1)
    with pytest.raises(TestSplitLocked):
        data.questions_for(repo[2], "test")
    assert not any(paths.RUNS.glob("agent-B0-test-2*"))  # refused before a run directory exists


def test_run_refuses_ids_outside_the_split_and_duplicates(repo):
    with pytest.raises(DataError, match="not in the train split"):
        run_agent(str(repo[1]), "B0", "train", ids=["9"])
    with pytest.raises(DataError, match="more than once"):
        run_agent(str(repo[1]), "B0", "train", ids=["1", "1"])


def test_rebuilt_splits_must_equal_the_committed_file(repo):
    committed = json.loads(paths.SPLITS.read_text())
    data.check_splits_unchanged(committed)
    with pytest.raises(DataError, match="differ"):
        data.check_splits_unchanged({**committed, "calib": ["1"]})


def test_a_run_is_done_only_when_nothing_went_wrong():
    from bench.agent.runner import new_outcome, run_status

    def outcome(**parts):
        return {**new_outcome(), "predictions": {"1": "x", "2": None}, **parts}
    assert run_status(["1", "2"], outcome(), 0, []) == "done"
    assert run_status(["1", "2"], outcome(tool_errors={"1": {"select_tables": "x"}}), 0, []) == "done"
    assert run_status(["1", "2", "3"], outcome(), 0, []) == "interrupted"
    assert run_status(["1", "2"], outcome(harness_errors={"1": ["route"]}), 0, []) == "failed"
    assert run_status(["1", "2"], outcome(failures={"1": "KeyError"}), 0, []) == "failed"
    assert run_status(["1", "2"], outcome(), 1, []) == "failed"
    assert run_status(["1", "2"], outcome(), 0, ["tiny"]) == "failed"


# ---------------------------------------------------------------- CHESS's final SQL

def test_final_sql_is_the_first_sql_of_the_last_key():
    def info(sql):
        return SimpleNamespace(SQL=sql)
    state = SimpleNamespace(SQL_meta_infos={"generate_candidate": [info("a")], "revise_1": [info("b"), info("c")]})
    assert final_sql(state) == "b"
    assert final_sql(SimpleNamespace(SQL_meta_infos={})) is None
    assert final_sql(SimpleNamespace(SQL_meta_infos={"generate_candidate": [info("a")], "revise_1": []})) is None
