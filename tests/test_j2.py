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


def test_truncation_counts_cut_off_answers_and_leaves_lines_without_the_field_unknown():
    def answered(question, finish_reason="absent", response="ok", site="select_tables", **kwargs):
        line = call("a", question, site, response=response, **kwargs)
        return line if finish_reason == "absent" else {**line, "finish_reason": finish_reason}
    first = answered("3", "length", parsed_ok=False)
    calls = [answered("1", "stop"), answered("2", "stop"),
             first, answered("3", "stop", attempt=2, retry_of=first["call_id"]),  # every attempt is a call
             answered("4", "stop", response=""), answered("5", "stop", response=" \n"),  # empty content: cut, whatever the reason
             answered("6", None),                      # the provider reported no reason: unknown
             answered("7", None, response=""),         # ... but an empty answer is cut
             answered("8"), answered("9", response=""),  # no field (an older log): unknown, even when empty
             answered("10", None, response=None, parsed_ok=False),  # failed before any answer: not counted
             answered("1", "length", site="revise", key="revise_1:0"), answered("2", "tool_calls", site="revise", key="revise_1:0")]
    table = j2.truncation(calls)
    assert table["select_tables"] == {"answers": 10, "truncated": 4, "unknown": 3, "rate": 4 / 7}
    assert table["revise"] == {"answers": 2, "truncated": 1, "unknown": 0, "rate": 0.5}
    old_log = j2.truncation([answered("1"), answered("2")])["select_tables"]
    assert old_log == {"answers": 2, "truncated": 0, "unknown": 2, "rate": None}  # nothing known: no rate, never 0%


def test_judge_run_reports_truncation_by_call_site_and_records_what_it_read(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch)
    payload = read_result(j2.run("replay-cheap"), "J2")
    assert payload["reads"] == {"run": payload["reads"]["run"], "config": []} and payload["reads"]["run"]["run_id"] == "replay-cheap"
    result = payload["result"]
    assert (result["mode"], result["engine"]) == ("run", "cheap_alt")
    assert set(result["truncation"]) == set(result["format_validity"])
    # these C1 lines predate `finish_reason`: every answer is unknown
    assert result["truncation"]["filter_column"] == {"answers": 2 * len(QUESTIONS), "truncated": 0,
                                                    "unknown": 2 * len(QUESTIONS), "rate": None}
    replayed = read_result(j2.run(replay_run_id="replay-cheap", replay_eval_run_id="eval-r", teacher_eval_run_id="eval-t"), "J2")
    assert set(replayed["reads"]) == {"teacher", "replay", "replay_eval", "teacher_eval", "config"}
    assert replayed["result"]["mode"] == "replay"


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
    write_run("eval-e2e", {"type": "eval", "source_run_id": "replay-cheap", "status": None}, files={"results.jsonl": []})
    with pytest.raises(JudgmentError, match="per-call"):
        j2.judge_replay("replay-cheap", "eval-e2e", "eval-t")


def test_a_replay_is_judged_on_its_declared_call_sites_and_must_cover_them():
    t = [call("t", "1", "select_tables", parsed={"table_names": ["x"]}),
         call("t", "1", "filter_column", "t.a", parsed={"is_column_information_relevant": "Yes"}),
         call("t", "2", "select_tables", parsed={"table_names": ["x"]})]
    only_tables = [call("r", "1", "select_tables", parsed={"table_names": ["x"]}),
                   call("r", "2", "select_tables", parsed={"table_names": ["y"]})]
    table = j2.compare(t, only_tables, call_sites=["select_tables"])
    assert set(table) == {"select_tables"} and table["select_tables"]["agreement"]["rate"] == 0.5
    with pytest.raises(JudgmentError, match="lacks 1"):  # the column filter is in scope when nothing is declared
        j2.compare(t, only_tables)
    with pytest.raises(JudgmentError, match="lacks 1"):
        j2.compare(t, only_tables[:1], call_sites=["select_tables"])
