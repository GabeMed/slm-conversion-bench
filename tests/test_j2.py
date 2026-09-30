"""J2 · per call site: EX with the teacher's context where there is gold, agreement (C3) elsewhere,
format validity everywhere."""
import pytest

from bench.judge import j2
from bench.judge.base import JudgmentError, read_result, write_result
from fixtures.fake import call, repo, write_run
from fixtures.world import gold_correct, per_call_eval, replay, teacher

QUESTIONS = [str(q) for q in range(100, 112)]


def test_agreement_follows_the_agents_decision_not_the_text():
    t = [call("t", "1", "filter_column", "t.a", parsed={"is_column_information_relevant": "Yes", "chain_of_thought_reasoning": "a"}),
         call("t", "1", "select_tables", parsed={"table_names": ["x", "y"], "chain_of_thought_reasoning": "a"})]
    r = [call("r", "1", "filter_column", "t.a", parsed={"is_column_information_relevant": "yes", "chain_of_thought_reasoning": "b"}),
         call("r", "1", "select_tables", parsed={"table_names": ["y", "x"], "chain_of_thought_reasoning": "b"})]
    table = j2.compare(t, r)
    assert table["filter_column"]["agreement"] == {"n": 1, "agree": 1, "teacher_unparsed": 0, "rate": 1.0}
    assert table["select_tables"]["agreement"]["rate"] == 1.0 and table["select_tables"]["gold"] is None


def test_unparsed_outputs_disagree_and_unparsed_teacher_outputs_are_left_out():
    t = [call("t", "1", "select_tables", parsed={"table_names": ["x"]}),
         call("t", "2", "select_tables", response="?", parsed_ok=False),
         call("t", "3", "select_tables", parsed={"table_names": ["x"]})]
    r = [call("r", "1", "select_tables", response="?", parsed_ok=False),
         call("r", "2", "select_tables", parsed={"table_names": ["x"]}),
         call("r", "3", "select_tables", parsed={"table_names": ["x"]})]
    entry = j2.compare(t, r)["select_tables"]
    assert entry["agreement"] == {"n": 2, "agree": 1, "teacher_unparsed": 1, "rate": 0.5}
    assert entry["format_valid_rate"] == pytest.approx(2 / 3)


def test_gold_call_sites_are_judged_by_the_per_call_evaluation():
    t = teacher("t", "calib", QUESTIONS)
    r = replay(t, "r", "cheap_alt", 0.5, fail_rate=0.2)
    replay_ok, teacher_ok = gold_correct("r", 0.6), gold_correct("t", 0.9)
    replay_eval = {(c["question_id"], c["call_site"], c["invocation_key"]): replay_ok(c) for c in r if c["parsed_ok"]}
    teacher_eval = {(c["question_id"], c["call_site"], c["invocation_key"]): teacher_ok(c) for c in t}
    table = j2.compare(t, r, replay_eval, teacher_eval)
    gold = table["generate_candidate"]["gold"]
    generated = [c for c in r if c["call_site"] == "generate_candidate"]
    expected = sum(c["parsed_ok"] and replay_ok(c) for c in generated)
    assert gold["n"] == len(QUESTIONS) and gold["correct_replay"] == expected
    assert gold["ex_replay"] == expected / len(QUESTIONS)
    assert gold["correct_teacher"] == sum(teacher_ok(c) for c in t if c["call_site"] == "generate_candidate")
    # the question is the unit: correct when every one of its gold invocations in the group is
    revise_questions = {c["question_id"] for c in t if c["call_site"] == "revise"}
    by_site = j2.compare(t, r, replay_eval, teacher_eval, group=lambda identity, attempts: "all" if identity[1] in ("generate_candidate", "revise") else None)
    for q in revise_questions:
        both = [c for c in r if c["question_id"] == q and c["call_site"] in ("generate_candidate", "revise")]
        assert by_site["all"]["gold"]["by_question"]["replay"][q] == all(c["parsed_ok"] and replay_ok(c) for c in both)


def test_refuses_incomplete_evidence():
    t = [call("t", "1", "generate_candidate", "g:0", parsed={"SQL": "SELECT 1"})]
    r = [call("r", "1", "generate_candidate", "g:0", parsed={"SQL": "SELECT 1"})]
    with pytest.raises(JudgmentError, match="per-call evaluations"):
        j2.compare(t, r)
    with pytest.raises(JudgmentError, match="no result"):
        j2.compare(t, r, {}, {("1", "generate_candidate", "g:0"): True})
    with pytest.raises(JudgmentError, match="does not"):
        j2.compare([], r)


def test_format_validity_of_one_run():
    first = call("a", "1", "select_tables", response="?", parsed_ok=False)
    calls = [first, call("a", "1", "select_tables", parsed={}, attempt=2, retry_of=first["call_id"]),
             call("a", "2", "select_tables", response="?", parsed_ok=False)]
    assert j2.format_validity(calls)["select_tables"] == {"n": 2, "valid": 1, "attempts": 3, "rate": 0.5}


def world(tmp_path, monkeypatch):
    repo(tmp_path, monkeypatch)
    t = teacher("agent-B0-calib", "calib", QUESTIONS)
    write_run("agent-B0-calib", {"type": "agent", "arm": "B0", "split": "calib", "question_ids": QUESTIONS}, t)
    per_call_eval("eval-t", "agent-B0-calib", t, gold_correct("t", 0.9))
    r = replay(t, "replay-cheap", "cheap_alt", 0.7)
    write_run("replay-cheap", {"type": "replay", "source_run_id": "agent-B0-calib", "engine": "cheap_alt", "split": "calib"}, r)
    per_call_eval("eval-r", "replay-cheap", r, gold_correct("r", 0.6))
    return t, r


def test_judge_replay_reads_the_executions_and_writes_a_result(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch)
    reads, result = j2.judge_replay("replay-cheap", "eval-r", "eval-t")
    assert set(reads) == {"teacher", "replay", "replay_eval", "teacher_eval"} and result["engine"] == "cheap_alt"
    path = write_result("J2", reads, result)
    assert read_result(path, "J2")["result"]["per_call_site"]["filter_column"]["agreement"]["n"] == 2 * len(QUESTIONS)
    with pytest.raises(JudgmentError, match="evaluated"):
        j2.judge_replay("replay-cheap", "eval-t", "eval-t")
