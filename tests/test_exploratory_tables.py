"""The exploratory tables of SPEC 6.1 (scripts/exploratory_tables.py): outside the registered code,
checked here on synthetic executions only."""
import importlib.util
import json
import re
import sqlite3
from pathlib import Path

import pytest

from bench.contracts import concordance

np = pytest.importorskip("numpy")  # the intervals: in the agent and the analysis environments, not the core one

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("exploratory_tables", ROOT / "scripts" / "exploratory_tables.py")
tables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tables)

BOOT = tables.Bootstrap(200, 7)
SCHEMA = ("Database Schema Overview:\nCREATE TABLE gas_t\n(\n\tid INTEGER primary key,\n\t`price segment` TEXT null, -- x\n"
          "\tcountry TEXT null,\n\tforeign key (id) references other_u (id),\n);\n\nCREATE TABLE other_u\n(\n\tid INTEGER,\n);\n\nQuestion: ?")


# ------------------------------------------------------------------------------------------ a synthetic repository

def call(question, site, key, output, parsed=True, attempt=1, cluster=None, prompt="p"):
    return {"question_id": question, "call_site": site, "invocation_key": key, "attempt": attempt, "parsed_ok": parsed,
            "parsed_output": output if parsed else None, "cluster": cluster,
            "prompt_messages": [{"role": "user", "content": prompt}]}


def keep(answer):
    return {"is_column_information_relevant": answer}


def teacher_calls(questions=("1", "2", "3", "4")):
    """Per question: 3 column-filter calls (Yes, No, No), one of each selection, keywords, one generation and
    two agent steps on two fixed prompts; question 1 also repairs."""
    calls = []
    for q in questions:
        calls += [call(q, "agent_ir", "ir:0", {"tool": "extract_keywords"}, prompt="IR0"),
                  call(q, "agent_ir", "ir:1", "DONE", prompt="IR1"),
                  call(q, "extract_keywords", "single", ["a", "b"]),
                  call(q, "filter_column", "gas_t.id", keep("Yes")),
                  call(q, "filter_column", "gas_t.country", keep("No")),
                  call(q, "filter_column", "other_u.id", keep("No")),
                  call(q, "select_tables", "single", {"table_names": ["gas_t"]}, prompt=SCHEMA),
                  call(q, "select_columns", "single", {"gas_t": ["id", "`price segment`"]}, prompt=SCHEMA),
                  call(q, "generate_candidate", "g:0", {"SQL": "SELECT 1"})]
    calls.append(call("1", "revise", "r:0", {"refined_sql_query": "SELECT 1"}))
    return calls


CLUSTER = {"filter_column": "c0", "select_tables": "c1", "select_columns": "c1", "generate_candidate": "c1",
           "revise": "c1", "extract_keywords": "c2", "agent_ir": "c3"}


def engine_calls(source, routed=False):
    """An engine that differs from the teacher on question 1: it keeps `country`, drops `id`, selects another
    table, half the keywords, and answers the second agent prompt differently; question 2's keywords do not parse."""
    calls = []
    for c in source:
        q, site, key = c["question_id"], c["call_site"], c["invocation_key"]
        output, parsed = c["parsed_output"], True
        if q == "1":
            output = {"filter_column": keep("No" if key == "gas_t.id" else "Yes" if key == "gas_t.country" else "No"),
                      "select_tables": {"table_names": ["other_u"]}, "extract_keywords": ["a"],
                      "agent_ir": "DONE" if key == "ir:0" else c["parsed_output"]}.get(site, output)
        if q == "2" and site == "extract_keywords":
            parsed = False
        calls.append(call(q, site, key, output, parsed=parsed, cluster=CLUSTER[site] if routed else None))
    return calls


def write_run(root, run_id, manifest, calls=None, results=None, config=None):
    path = root / "runs" / run_id
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"run_id": run_id, **manifest}))
    if calls is not None:
        (path / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    if results is not None:
        (path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    if config is not None:
        (path / "config.json").write_text(json.dumps(config))


CONFIG = {"stats": {"n_boot": 200}, "seeds": {"bootstrap": 7}, "data": {"bird_dev_databases": {"root": "dev"}}}
QUESTIONS = ["1", "2", "3", "4"]
DIFFICULTY = {"1": "simple", "2": "simple", "3": "moderate", "4": "challenging"}


def per_call(calls, right):
    return [{"question_id": c["question_id"], "call_site": c["call_site"], "invocation_key": c["invocation_key"],
             "correct": c["question_id"] in right, "difficulty": DIFFICULTY[c["question_id"]], "db_id": "tiny"}
            for c in calls if c["call_site"] in ("generate_candidate", "revise")]


def end_to_end(predictions, right):
    return [{"question_id": q, "db_id": "tiny", "difficulty": DIFFICULTY[q], "correct": q in right, "predicted_sql": sql,
             "gold_sql": "SELECT id FROM t ORDER BY id", "pred_error": None, "gold_error": None}
            for q, sql in predictions.items()]


@pytest.fixture
def repo(tmp_path):
    """A teacher run with its per-call evaluation, base and adapter replays (no cheap alternative, no
    teacher replay), and the arms B0 and B4 with their evaluations, over a tiny database."""
    db = tmp_path / "data" / "bird_dev" / "dev" / "dev_databases" / "tiny"
    db.mkdir(parents=True)
    connection = sqlite3.connect(db / "tiny.sqlite")
    connection.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    connection.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    connection.commit()
    connection.close()
    teacher = teacher_calls()
    agent = {"type": "agent", "split": "calib", "status": "done", "question_ids": QUESTIONS}
    write_run(tmp_path, "agent-B0-calib-1", {**agent, "arm": "B0"}, teacher, config=CONFIG)
    write_run(tmp_path, "eval-per-call-B0", {"type": "eval", "per_call": True, "source_run_id": "agent-B0-calib-1", "split": "calib"},
              results=per_call(teacher, {"1", "2", "3"}))
    for name, routed, right in (("base", False, {"1"}), ("adapter", True, {"1", "2"})):
        replay = engine_calls(teacher, routed)
        write_run(tmp_path, f"replay-{name}", {"type": "replay", "split": "calib", "status": "done", "call_sites": None,
                                               "source_run_id": "agent-B0-calib-1", "question_ids": QUESTIONS}, replay)
        write_run(tmp_path, f"eval-per-call-{name}", {"type": "eval", "per_call": True, "source_run_id": f"replay-{name}",
                                                      "split": "calib"}, results=per_call(replay, right))
    evals = {"fixed_date": "2026-09-30", "timeout_s": 5, "split": "calib", "type": "eval"}
    write_run(tmp_path, "eval-B0", {**evals, "source_run_id": "agent-B0-calib-1"},
              results=end_to_end({q: "SELECT id FROM t ORDER BY id" for q in QUESTIONS}, {"1", "2", "3"}))
    b4 = engine_calls(teacher, routed=True)
    write_run(tmp_path, "agent-B4-calib-1", {**agent, "arm": "B4"}, b4, config=CONFIG)
    write_run(tmp_path, "eval-B4", {**evals, "source_run_id": "agent-B4-calib-1"}, results=end_to_end(
        {"1": "SELECT id FROM t", "2": "SELECT id, name FROM t", "3": "SELECT name, id FROM t", "4": "SELECT name FROM t"}, {"1"}))
    write_run(tmp_path, "eval-per-call-B4", {"type": "eval", "per_call": True, "source_run_id": "agent-B4-calib-1", "split": "calib"},
              results=per_call(b4, {"1", "3"}))
    (tmp_path / "train" / "datasets").mkdir(parents=True)
    (tmp_path / "train" / "datasets" / "manifest.json").write_text(json.dumps(
        {"clusters": {"c0": {"n": 11982}, "c1": {"n": 2517}, "c2": {"n": 833}, "c3": {"n": 9164}}}))
    return tmp_path


ARGS = ["--teacher", "agent-B0-calib-1", "--teacher-per-call-eval", "eval-per-call-B0",
        "--replay", "base=replay-base,eval-per-call-base", "--replay", "adapter=replay-adapter,eval-per-call-adapter",
        "--live", "B0=agent-B0-calib-1,eval-B0,eval-per-call-B0", "--live", "B4=agent-B4-calib-1,eval-B4,eval-per-call-B4"]


def build(repo, *extra):
    return tables.build(tables.parse_args(["--root", str(repo), *ARGS, *extra]))


# ------------------------------------------------------------------------------------------ reading an output

@pytest.mark.parametrize("site,a,b", [
    ("extract_keywords", ["a", "b"], ["b", "a", "a"]), ("extract_keywords", ["a"], ["a", "b"]), ("extract_keywords", [], []),
    ("extract_keywords", [["x"]], ["a"]), ("extract_keywords", None, ["a"]), ("extract_keywords", 3, 3),
    ("filter_column", keep("yes"), keep("YES")), ("filter_column", keep("No"), {}), ("filter_column", keep("No"), "text"),
    ("filter_column", keep("Yes"), keep(None)),
    ("select_tables", {"table_names": ["a", "b"]}, {"table_names": ["b", "a"]}), ("select_tables", {}, {"table_names": []}),
    ("select_tables", {"table_names": None}, {"table_names": None}), ("select_tables", ["a"], ["a"]),
    ("select_columns", {"t": ["`a b`"], "chain_of_thought_reasoning": "x"}, {"t": ["a b"]}), ("select_columns", {"t": ["a"]}, {"t": ["b"]}),
    ("select_columns", {"t": None}, {"t": None}), ("agent_ir", {"tool": "x"}, {"tool": "x"}), ("agent_ir", "DONE", {"tool": "x"}),
])
def test_two_decisions_are_equal_exactly_when_the_contract_says_the_outputs_agree(site, a, b):
    mine, theirs = (tables.decision(site, o) if o is not None else None for o in (a, b))
    assert (mine is not None and theirs is not None and mine == theirs) == concordance.agree(site, a, b)


def test_the_candidates_of_a_prompt_are_the_tables_and_columns_of_its_schema():
    assert tables.schema_candidates(SCHEMA) == {"gas_t": ["id", "price segment", "country"], "other_u": ["id"]}
    assert tables.all_candidates("select_tables", SCHEMA) == frozenset({"gas_t", "other_u"})
    assert ("gas_t", "price segment") in tables.all_candidates("select_columns", SCHEMA)


def test_the_last_attempt_is_the_invocations_output_and_no_prompt_is_kept(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in [
        call("1", "filter_column", "k", keep("Yes")), call("1", "filter_column", "k", None, parsed=False, attempt=2),
        call("1", "agent_ir", "ir:0", "DONE", prompt="P")]))
    found = tables.load_invocations(path, teacher=True)
    assert found[("1", "filter_column", "k")]["parsed"] is False and found[("1", "filter_column", "k")]["attempts"] == 2
    assert "'P'" not in str(found) and "prompt_messages" not in str(found) and len(found[("1", "agent_ir", "ir:0")]["prompt"]) == 64


# ------------------------------------------------------------------------------------------ the cells

def cells(repo, column="base"):
    return {site: row[column] for site, row in build(repo)["tables"]["per_call_site"]["rows"].items()}


def test_column_filter_recall_agreement_and_kappa(repo):
    cell = cells(repo)["filter_column"]
    # the teacher keeps 4 and drops 8; on question 1 the engine drops the kept one and keeps a dropped one
    assert cell["counts"] == {"both_keep": 3, "teacher_keeps_only": 1, "engine_keeps_only": 1, "both_drop": 7}
    metrics = cell["metrics"]
    assert (metrics["recall_kept"]["numerator"], metrics["recall_kept"]["denominator"]) == (3, 4)
    assert (metrics["agreement_dropped"]["numerator"], metrics["agreement_dropped"]["denominator"]) == (7, 8)
    assert metrics["raw_agreement"]["value"] == pytest.approx(10 / 12)
    chance = (4 * 4 + 8 * 8) / 144
    assert metrics["kappa"]["value"] == pytest.approx((10 / 12 - chance) / (1 - chance))
    assert (cell["calls"], cell["questions"]) == (12, 4)
    assert metrics["recall_kept"]["low"] <= metrics["recall_kept"]["value"] <= metrics["recall_kept"]["high"]


def test_the_trivial_rule_of_the_column_filter_is_always_no(repo):
    metrics = cells(repo, "trivial")["filter_column"]["metrics"]
    assert metrics["recall_kept"]["value"] == 0 and metrics["agreement_dropped"]["value"] == 1
    assert metrics["kappa"]["value"] == 0 and metrics["raw_agreement"]["value"] == pytest.approx(8 / 12)


def test_an_engine_output_that_did_not_parse_disagrees():
    teacher = {("1", "filter_column", k): {"parsed": True, "decision": d} for k, d in (("a", True), ("b", False))}
    engine = {i: {"parsed": False, "decision": None} for i in teacher}
    cell = tables.site_cell("filter_column", teacher, engine, list(teacher), BOOT)
    assert cell["counts"] == {"both_keep": 0, "teacher_keeps_only": 1, "engine_keeps_only": 1, "both_drop": 0}
    assert cell["metrics"]["raw_agreement"]["value"] == 0 and cell["unparsed"] == 2


def test_a_teacher_output_that_did_not_parse_is_left_out_and_counted():
    teacher = {("1", "filter_column", "a"): {"parsed": True, "decision": True},
               ("1", "filter_column", "b"): {"parsed": False, "decision": None}}
    engine = {i: {"parsed": True, "decision": True} for i in teacher}
    cell = tables.site_cell("filter_column", teacher, engine, list(teacher), BOOT)
    assert cell["teacher_unparsed"] == 1 and cell["metrics"]["raw_agreement"]["denominator"] == 1


def test_selection_gives_exact_set_agreement_and_leaves_the_recall_to_its_open_question(repo):
    base, trivial = cells(repo), cells(repo, "trivial")
    assert base["select_tables"]["metrics"]["exact_set_agreement"]["numerator"] == 3
    assert base["select_tables"]["metrics"]["recall"] == {"status": "not computed", "open_question": "Q1"}
    # every candidate of the prompt: the teacher never selects both tables, nor every column
    assert trivial["select_tables"]["metrics"]["exact_set_agreement"]["value"] == 0
    assert trivial["select_tables"]["metrics"]["recall"] == {"status": "1 by construction"}
    assert trivial["select_columns"]["teacher_set_outside_prompt"] == 0


def test_keywords_mean_jaccard_and_exact_set_agreement(repo):
    cell = cells(repo)["extract_keywords"]
    # question 1: {a} against {a, b} = 0.5; question 2 did not parse = 0; questions 3 and 4 = 1
    assert cell["metrics"]["mean_jaccard"]["value"] == pytest.approx(2.5 / 4)
    assert cell["metrics"]["exact_set_agreement"]["numerator"] == 2 and cell["unparsed"] == 1


def test_two_empty_keyword_sets_have_a_jaccard_index_of_one_and_are_counted():
    teacher = {("1", "extract_keywords", "single"): {"parsed": True, "decision": frozenset()}}
    cell = tables.site_cell("extract_keywords", teacher, teacher, list(teacher), BOOT)
    assert cell["metrics"]["mean_jaccard"]["value"] == 1 and cell["both_sets_empty"] == 1


def test_sql_generation_is_ex_per_call_and_an_unparsed_answer_is_wrong(repo):
    assert cells(repo)["generate_candidate"]["metrics"]["ex"]["numerator"] == 1
    assert cells(repo, "adapter")["generate_candidate"]["metrics"]["ex"]["numerator"] == 2
    identity = ("1", "generate_candidate", "g:0")
    assert not tables.call_correct(identity, {identity: {"parsed": False}}, {identity: {"correct": True}})
    with pytest.raises(tables.TablesError):
        tables.call_correct(identity, {identity: {"parsed": True}}, {})


def test_the_agents_steps_are_counted_per_distinct_prompt_with_no_rate_and_no_interval(repo):
    cell = cells(repo)["agent_ir"]
    assert cell["metrics"] == {} and cell["open_question"] == "Q5"
    prompts = cell["prompts"]  # IR0: question 1 answers otherwise (3 of 4 agree); IR1: every call agrees
    assert (prompts["distinct"], prompts["every_call_agrees"], prompts["some_calls_agree"], prompts["no_call_agrees"]) == (2, 1, 1, 0)
    assert sorted((p["calls"], p["agree"]) for p in prompts["by_prompt"]) == [(4, 3), (4, 4)]


def test_what_the_spec_does_not_fix_is_not_computed_and_names_its_question(repo):
    trivial = cells(repo, "trivial")
    found = {site: trivial[site].get("open_question") for site in ("extract_keywords", "generate_candidate", "revise", "agent_ir")}
    assert found == {"extract_keywords": "Q2", "generate_candidate": "Q3", "revise": "Q3", "agent_ir": "Q4"}
    assert all(trivial[site]["status"] == "not computed" for site in found)


# ------------------------------------------------------------------------------------------ not run, never dropped

def test_a_column_whose_replay_was_not_run_prints_not_run_at_every_call_site(repo):
    result = build(repo)
    rows = result["tables"]["per_call_site"]["rows"]
    assert set(rows) == set(tables.SITE_ORDER) and all(set(row) == set(tables.COLUMNS) for row in rows.values())
    assert all(rows[site]["cheap"]["status"] == "not run" for site in tables.SITE_ORDER)
    assert rows["filter_column"]["teacher"]["status"] == "not run"
    # the call sites with gold still show the teacher's EX on its own run
    assert rows["generate_candidate"]["teacher"]["from_own_run"]["metrics"]["ex"]["numerator"] == 3
    assert rows["generate_candidate"]["teacher"]["from_replay"]["status"] == "not run"
    text = tables.render(result)
    header = next(line for line in text.splitlines() if line.startswith("| call site |"))
    assert all(tables.COLUMN_TITLES[c] in header for c in tables.COLUMNS)
    assert result["tables"]["per_call_site"]["inputs"]["replays"]["cheap"] == "not run"


def test_without_a_teacher_replay_the_pilots_self_agreement_is_shown_and_labelled(repo):
    pilot_source = teacher_calls(("7", "8"))
    write_run(repo, "agent-B0-calib-pilot", {"type": "agent", "arm": "B0", "split": "calib", "status": "done",
                                             "question_ids": ["7", "8"]}, pilot_source)
    sites = ["agent_ir", "extract_keywords", "filter_column", "select_columns", "select_tables"]
    write_run(repo, "replay-pilot", {"type": "replay", "split": "calib", "status": "done", "call_sites": sites,
                                     "source_run_id": "agent-B0-calib-pilot", "question_ids": ["7", "8"]},
              [c for c in pilot_source if c["call_site"] in sites])
    result = build(repo, "--teacher-pilot-replay", "replay-pilot")
    cell = result["tables"]["per_call_site"]["rows"]["filter_column"]["teacher"]
    assert cell["pilot"] and list(cell["metrics"]) == ["raw_agreement"] and cell["metrics"]["raw_agreement"]["value"] == 1
    assert "PILOT, not these calls" in tables.render(result)
    # in the roll-up the pilot's calls have no cluster of this replay: not run there
    assert result["tables"]["per_cluster"]["clusters"]["c0"]["rows"]["filter_column"]["teacher"]["status"] == "not run"


def test_a_replay_is_read_within_its_questions_and_refused_when_it_lacks_an_invocation():
    teacher = {(q, "filter_column", "k"): {} for q in ("1", "2")}
    replay = {("1", "filter_column", "k"): {}}
    assert tables.replay_scope(teacher, replay, {"question_ids": ["1"], "call_sites": None}, "r") == [("1", "filter_column", "k")]
    with pytest.raises(tables.TablesError):
        tables.replay_scope(teacher, replay, {"question_ids": ["1", "2"], "call_sites": None}, "r")
    with pytest.raises(tables.TablesError):
        tables.replay_scope(teacher, {**replay, ("1", "revise", "x"): {}}, {"question_ids": ["1"], "call_sites": None}, "r")


# ------------------------------------------------------------------------------------------ clusters

def test_the_roll_up_groups_by_the_adapter_replays_cluster_and_pools_nothing(repo):
    cluster = build(repo)["tables"]["per_cluster"]
    assert {name: list(found["rows"]) for name, found in cluster["clusters"].items()} == {
        "c0": ["filter_column"], "c1": ["select_tables", "select_columns", "generate_candidate", "revise"],
        "c2": ["extract_keywords"], "c3": ["agent_ir"]}
    assert cluster["clusters"]["c0"]["training_rows"] == 11982 and cluster["clusters"]["c1"]["calls"] == 13
    # the base replay carries no cluster of its own: it is grouped by the adapter replay's
    assert cluster["clusters"]["c0"]["rows"]["filter_column"]["base"]["counts"]["both_drop"] == 7
    assert cluster["clusters"]["c1"]["rows"]["generate_candidate"]["trivial"]["open_question"] == "Q3"


def test_the_roll_up_is_not_run_without_the_adapter_replay(repo):
    args = [a for a in ARGS if not a.startswith("adapter=")]
    args = [a for i, a in enumerate(args) if not (a == "--replay" and (i + 1 == len(args) or args[i + 1].startswith("--")))]
    result = tables.build(tables.parse_args(["--root", str(repo), *args]))
    assert result["tables"]["per_cluster"]["status"] == "not run"
    assert "not run: the cluster of a call" in tables.render(result)


def test_training_rows_against_the_rule_of_thumb():
    assert tables.rule_reading(11982) == "within 10k-100k" and tables.rule_reading(833) == "below 10k (12.0x below)"
    assert tables.rule_reading(None) == "training rows not found" and tables.rule_reading(200_000) == "above 100k"


# ------------------------------------------------------------------------------------------ the cuts

def test_the_arms_own_execution_counts_path_and_ex(repo):
    rows = build(repo)["tables"]["own_execution"]["arms"]["B4"]["rows"]
    assert (rows["filter_column"]["invocations"], rows["filter_column"]["questions_reaching"]) == (12, 4)
    assert rows["filter_column"]["invocations_per_question_in_run"] == 3
    assert (rows["revise"]["questions_reaching"], rows["revise"]["invocations_per_reaching_question"]) == (1, 1)
    assert rows["generate_candidate"]["ex"]["numerator"] == 2 and "ex" not in rows["filter_column"]
    assert rows["agent_cg"]["invocations"] == 0 and rows["agent_cg"]["invocations_per_reaching_question"] is None


def test_the_difficulty_cut_of_the_arms_the_comparisons_and_sql_generation(repo):
    difficulty = build(repo)["tables"]["difficulty"]
    assert {d: (c["correct"], c["n"]) for d, c in difficulty["arms"]["B0"].items()} == {
        "simple": (2, 2), "moderate": (1, 1), "challenging": (0, 1)}
    simple = difficulty["comparisons"]["B4 - B0"]["by_difficulty"]["simple"]
    assert (simple["n"], simple["a_only"], simple["b_only"], simple["value"]) == (2, 0, 1, -0.5)
    generation = difficulty["sql_generation"]
    assert generation["on_the_teachers_inputs"]["adapter"]["by"]["simple"]["correct"] == 2
    assert generation["on_the_teachers_inputs"]["cheap"]["status"] == "not run"
    assert generation["on_the_teachers_inputs"]["trivial"]["open_question"] == "Q3"
    assert generation["own_execution"]["B4"]["by"]["moderate"]["correct"] == 1


def test_the_database_cut_is_counts_only(repo):
    database = build(repo)["tables"]["database"]
    assert database["databases"] == ["tiny"] and database["arms"]["B4"]["tiny"] == {"n": 4, "correct": 1}
    assert set(database["sql_generation"]["on_the_teachers_inputs"]["base"]["by"]["tiny"]) == {"calls", "questions", "correct"}


def test_replay_against_live_counts_questions_four_ways(repo):
    cross = build(repo)["tables"]["replay_against_live"]["arms"]
    # the adapter replay is right on questions 1 and 2, B4's own execution on question 1
    assert cross["B4"]["by_call_site"]["generate_candidate"] == {
        "questions": 4, "replay_only_questions": 0, "right_both": 1, "right_on_inputs_only": 1, "right_live_only": 0, "wrong_both": 2}
    assert cross["B3"]["status"] == "not run" and cross["B1"]["status"] == "not run"


# ------------------------------------------------------------------------------------------ projection-tolerant EX

def test_an_ordered_projection_keeps_the_predictions_column_order():
    gold = [(1, "a"), (2, "b")]
    assert tables.ordered_projection([(1, "x", "a"), (2, "y", "b")], gold) is True
    assert tables.ordered_projection([("a", 1), ("b", 2)], gold) is False  # the columns are there, in another order
    assert tables.ordered_projection([(1,), (2,)], gold) is False  # fewer columns than the gold
    assert tables.ordered_projection([(1, "a", 9), (2, "b", 9), (3, "c", 9)], gold) is False  # another set of rows
    assert tables.ordered_projection([(1, 1, "a"), (2, 2, "b"), (2, 2, "b")], gold) is True  # sets of rows, as BIRD's EX
    assert tables.ordered_projection([(1, 2), (2, 1)], [(1,), (2,)]) is True
    assert tables.ordered_projection([(0,) * 6], [(0,) * 3], max_checks=0) is None  # the search was cut off


def test_projection_tolerant_ex_per_arm(repo):
    arms = build(repo)["tables"]["projection_tolerant_ex"]["arms"]
    # B4: question 1 right; 2 and 3 return the gold's column and one more, after it and before it; 4 lacks it
    assert arms["B4"]["right_only_once_extra_columns_are_dropped"] == ["2", "3"]
    assert (arms["B4"]["ex"]["numerator"], arms["B4"]["projection_tolerant_ex"]["numerator"], arms["B4"]["n"]) == (1, 3, 4)
    # B0's question 4 was scored wrong and returns the gold's rows now: left as scored, and listed
    assert arms["B0"]["right_only_once_extra_columns_are_dropped"] == [] and arms["B0"]["not_reexecuted"] == []
    assert arms["B0"]["same_rows_when_executed_again"] == ["4"] and arms["B0"]["projection_tolerant_ex"]["numerator"] == 3


def test_ex_without_a_question(repo):
    without = build(repo, "--without", "1")["tables"]["without_questions"]["arms"]
    assert without["B0"] == {"status": "computed", "n": 3, "correct": 2, "ex": pytest.approx(2 / 3), "removed": 1}


# ------------------------------------------------------------------------------------------ intervals

def test_an_interval_resamples_questions_and_keeps_a_questions_calls_together():
    # one question holds every miss: resampling questions moves the rate in steps of whole questions
    hits = {"1": [0, 10], "2": [10, 10], "3": [10, 10]}
    found = tables.rate_metric(tables.Bootstrap(2000, 1), hits)
    assert found["value"] == pytest.approx(2 / 3) and found["low"] == 0 and found["high"] == 1
    assert tables.rate_metric(tables.Bootstrap(2000, 1), hits) == found  # the seed fixes the resamples
    assert tables.rate_metric(tables.Bootstrap(50, 1), {"1": [3, 4]})["low"] == 0.75  # one question: no spread


def test_the_interval_is_two_sided_at_95_percent():
    boot = tables.Bootstrap(4000, 3)
    hits = {str(q): [q % 2, 1] for q in range(100)}
    found = tables.rate_metric(boot, hits)
    rng = np.random.Generator(np.random.PCG64(3))
    counts = rng.multinomial(100, np.full(100, 0.01), size=4000)
    means = np.sort(counts @ np.array([q % 2 for q in sorted(range(100), key=lambda q: (q, ""))], dtype=float) / 100)
    assert (found["low"], found["high"]) == (means[99], means[3900])  # the 100th smallest and the 100th largest


def test_a_resample_with_no_denominator_is_left_out_and_counted():
    found = tables.rate_metric(tables.Bootstrap(500, 1), {"1": [1, 1], "2": [0, 0]})
    assert found["value"] == 1 and found["undefined_resamples"] > 0 and found["low"] == 1


# ------------------------------------------------------------------------------------------ the page and the guard

VERDICT_WORDS = ("confirm", "refute", "inconclusive", "non-inferior", "noninferior", "significan", "passes", "fails",
                 "better", "worse", "superior", "wins", "beats")


def test_every_table_is_labelled_exploratory_names_its_inputs_and_uses_no_verdict_word(repo):
    text = tables.render(build(repo, "--without", "1"))
    headings = [line for line in text.splitlines() if re.match(r"#{2,3} ", line)]
    labelled = [h for h in headings if not h.startswith(("## Open questions", "## Conventions"))]
    assert len(labelled) >= 12 and all("(exploratory)" in h for h in labelled)
    assert "on the teacher's inputs" in text
    assert text.count("Inputs: `") == 8  # one per table of the page
    assert "agent-B0-calib-1" in text and "eval-per-call-adapter" in text
    assert not [w for w in VERDICT_WORDS if w in text.lower()]
    for key in tables.OPEN:
        assert f"**{key}**" in text


def test_a_test_execution_is_refused_without_the_flag(repo, capsys):
    write_run(repo, "agent-B0-test-1", {"type": "agent", "arm": "B0", "split": "test", "status": "done", "question_ids": []}, [])
    assert tables.main(["--root", str(repo), "--teacher", "agent-B0-test-1"]) == 2
    assert "--allow-test" in capsys.readouterr().err
    write_run(repo, "agent-B0-other", {"type": "agent", "arm": "B0", "split": "test", "status": "done", "question_ids": []}, [])
    with pytest.raises(tables.TablesError, match="test split"):  # by its manifest, whatever its id
        tables.read_manifest(repo / "runs", "agent-B0-other", False)
    with pytest.raises(tables.TablesError, match="test execution"):  # by its id, before anything of it is opened
        tables.run_path(repo / "runs", "eval-agent-B0-test-missing", False)


def test_inputs_that_do_not_belong_together_are_refused(repo):
    with pytest.raises(tables.TablesError, match="source_run_id"):  # an evaluation of another run
        tables.build(tables.parse_args(["--root", str(repo), "--teacher", "agent-B0-calib-1", "--teacher-per-call-eval", "eval-per-call-base"]))
    with pytest.raises(tables.TablesError, match="end-to-end"):
        tables.build(tables.parse_args(["--root", str(repo), "--teacher", "agent-B0-calib-1", "--teacher-per-call-eval", "eval-B0"]))
    with pytest.raises(tables.TablesError, match="not one of"):
        tables.parse_args(["--teacher", "x", "--replay", "gpt=replay-base"])


def test_the_command_writes_the_page_and_every_cell(repo, tmp_path):
    out, cells_path = tmp_path / "tables.md", tmp_path / "tables.json"
    assert tables.main(["--root", str(repo), *ARGS, "--out", str(out), "--json", str(cells_path)]) == 0
    written = json.loads(cells_path.read_text())
    assert written["label"] == "exploratory" and written["interval"] == {"level": 0.95, "sides": 2, "unit": "question", "n_boot": 200, "seed": 7,
                                                                          "numpy": np.__version__}
    assert set(written["tables"]) == {"per_call_site", "per_cluster", "own_execution", "difficulty", "database",
                                      "replay_against_live", "projection_tolerant_ex"}
    assert out.read_text().startswith("# Exploratory tables (SPEC 6.1), split `calib`")
