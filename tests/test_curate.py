"""S2 · `bench curate` and `bench datasets`: the production-signal filter, masking, exact and
near-duplicate removal, the cap per (question, call site) with its coverage report, the do-not-train
rule, and the per-cluster training files."""
import hashlib
import json

import pytest

pytest.importorskip("datasketch")

from bench import paths  # noqa: E402
from bench.curate import (CurationError, Masker, cap, coverage, curate, near_duplicates, run_curate,  # noqa: E402
                          write_datasets)
from bench.contracts.config import config_sha256, load_config  # noqa: E402
from bench.judge.base import JudgmentError, read_jsonl, reference, write_result  # noqa: E402
from fixtures.fake import call, prompt, write_run  # noqa: E402
from synthetic import GOLD, make_repo  # noqa: E402

SETTINGS = load_config(paths.ROOT / "config.yaml")["curation"]
SEED = 7  # the cap's; the examples of these tests are under its ceiling unless they say otherwise
DB_OF = {str(q): "db" for q in range(1, 10)}  # question id -> database
KEEP = '{"chain_of_thought_reasoning": "not needed", "is_column_information_relevant": "No"}'


def fake_sql(results):
    """run_sql(question_id, sql) answering from a table {sql: (rows, error)}."""
    return lambda question_id, sql: results[sql]


# ---------------------------------------------------------------- the filter

def test_success_filter_uses_the_production_signal_never_the_gold():
    run = "r"
    calls = [
        call(run, "1", "generate_candidate", "t:0", response="ok", parsed={"SQL": "SELECT good"}),
        call(run, "2", "generate_candidate", "t:0", response="err", parsed={"SQL": "SELECT broken"}),
        call(run, "3", "generate_candidate", "t:0", response="none", parsed={"SQL": "SELECT nothing"}),
        call(run, "4", "revise", "revise_1:0", response="fixed", parsed={"refined_sql_query": "SELECT good"}),
        call(run, "5", "filter_column", "t.c", response="garbled", parsed_ok=False),
    ]
    first = call(run, "6", "select_tables", response="bad", parsed_ok=False)
    retry = call(run, "6", "select_tables", response='{"table_names": ["t"]}', parsed={"table_names": ["t"]},
                 attempt=2, retry_of=first["call_id"])
    examples, counts = curate(calls + [first, retry], SETTINGS, fake_sql({
        "SELECT good": ([(1,)], None), "SELECT broken": (None, "OperationalError: no such column"),
        "SELECT nothing": ([], None)}), SEED, DB_OF)
    assert {(e["question_id"], e["call_site"]) for e in examples} == {("1", "generate_candidate"), ("4", "revise"),
                                                                      ("6", "select_tables")}
    assert counts["per_call_site"]["generate_candidate"] | {} == {
        "invocations": 3, "unparsed": 0, "sql_error": 1, "sql_empty": 1, "passed_filter": 1, "masked_sql": 0,
        "exact_duplicates": 0, "near_duplicates": 0, "over_cap": 0, "kept": 1}
    assert counts["per_call_site"]["filter_column"]["unparsed"] == 1
    six = next(e for e in examples if e["question_id"] == "6")
    assert six["completion"] == [{"role": "assistant", "content": '{"table_names": ["t"]}'}]  # the attempt that parsed
    assert six["call_id"] == retry["call_id"]
    assert counts["total"]["kept"] == 3 and "not applied" in counts["paraphrase"]


def test_masking_that_would_change_a_sql_completion_drops_the_example():
    sql = "SELECT * FROM users WHERE email = 'bob@corp.com'"
    c = call("r", "1", "generate_candidate", "t:0", response=sql, parsed={"SQL": sql})
    examples, counts = curate([c], SETTINGS, fake_sql({sql: ([(1,)], None)}), SEED, DB_OF)
    assert examples == [] and counts["per_call_site"]["generate_candidate"]["masked_sql"] == 1
    ids = "SELECT * FROM schools WHERE CDSCode = '01100170109835'"  # a 14-digit id, not a card
    c = call("r", "2", "generate_candidate", "t:0", response=ids, parsed={"SQL": ids})
    examples, _ = curate([c], SETTINGS, fake_sql({ids: ([(1,)], None)}), SEED, DB_OF)
    assert examples[0]["completion"][0]["content"] == ids


def test_an_unparsed_last_attempt_is_dropped_even_if_an_earlier_one_parsed():
    first = call("r", "1", "select_tables", response="{}", parsed={"table_names": []})
    second = call("r", "1", "select_tables", response="x", parsed_ok=False, attempt=2, retry_of=first["call_id"])
    examples, counts = curate([first, second], SETTINGS, fake_sql({}), SEED, DB_OF)
    assert examples == [] and counts["total"]["unparsed"] == 1


# ---------------------------------------------------------------- masking

def test_masking_replaces_in_prompt_and_completion_and_counts():
    mask = Masker(SETTINGS["mask"])
    text = ("mail ana@example.org, card 4111 1111 1111 1111, not a card 4111 1111 1111 1112, "
            "date 2012-01-01, id 1234567, ip 192.168.0.1, phone +1 415 555 0100")
    masked = mask(text)
    assert "[EMAIL]" in masked and "[IP]" in masked and "[PHONE]" in masked
    assert masked.count("[CARD]") == 1 and "4111 1111 1111 1112" in masked  # the Luhn check keeps ids
    assert "2012-01-01" in masked and "1234567" in masked
    assert dict(mask.detections) == {"card": 1, "email": 1, "ipv4": 1, "phone": 1}

    c = call("r", "1", "select_tables", messages=prompt("select_tables", "who is bob@corp.com?"),
             response='{"table_names": ["users"]} bob@corp.com', parsed={"table_names": ["users"]})
    examples, counts = curate([c], SETTINGS, fake_sql({}), SEED, DB_OF)
    assert "bob@corp.com" not in json.dumps(examples) and counts["mask_detections"] == {"email": 2}


# ---------------------------------------------------------------- duplicates

def test_exact_duplicates_are_removed_after_masking():
    a = call("r", "1", "select_tables", messages=prompt("select_tables", "a@x.org"), response="{}", parsed={})
    b = call("r", "2", "select_tables", messages=prompt("select_tables", "b@y.org"), response="{}", parsed={})
    examples, counts = curate([a, b], SETTINGS, fake_sql({}), SEED, DB_OF)
    assert [e["question_id"] for e in examples] == ["1"]  # the same once the emails are masked
    assert counts["per_call_site"]["select_tables"]["exact_duplicates"] == 1


def filter_example(question: str, column: str, completion: str = KEEP) -> dict:
    return {"prompt": prompt("filter_column", question, column), "completion": [{"role": "assistant", "content": completion}]}


def test_near_duplicates_are_judged_without_the_shared_template():
    distinct = [filter_example("How many schools are in Alameda county?", "schools.County"),
                filter_example("What is the average SAT score of charter schools?", "satscores.AvgScrRead")]
    assert near_duplicates(distinct, SETTINGS["near_duplicate"]) == []
    # compared whole, two prompts of one call site look alike (the template is most of the text)
    whole = {**SETTINGS["near_duplicate"], "template_share": 1.0}
    assert near_duplicates(distinct, whole) == [1]


def test_near_duplicates_need_both_prompt_and_completion_alike():
    long = " ".join(f"word{i}" for i in range(120))
    pair = [filter_example(long + " end.", "t.c"), filter_example(long + " end!", "t.c")]
    other = [filter_example("unrelated question " * 3, "t.d")]
    assert near_duplicates(pair + other, SETTINGS["near_duplicate"]) == [1]
    answers_differ = [pair[0], filter_example(long + " end!", "t.c", '{"is_column_information_relevant": "Yes", '
                                                                        '"chain_of_thought_reasoning": "needed for the count"}')]
    assert near_duplicates(answers_differ + other, SETTINGS["near_duplicate"]) == []


# ---------------------------------------------------------------- the cap

def example(question_id: str, site: str, key: str = "single") -> dict:
    return {"call_id": f"{question_id}|{site}|{key}", "question_id": question_id, "call_site": site,
            "invocation_key": key}


def test_the_cap_keeps_a_seeded_sample_of_each_question_and_call_site():
    many = [example("1", "filter_column", f"t.c{i}") for i in range(10)]
    few = [example("2", "filter_column", f"t.c{i}") for i in range(3)]
    other = [example("1", "select_tables")]  # the same question, another call site: its own ceiling
    examples = many + few + other
    kept = cap(examples, 4, 7)
    ranked = sorted(many, key=lambda e: hashlib.sha256(f"7:1:filter_column:{e['invocation_key']}".encode()).hexdigest())
    assert [e for e in kept if e in many] == [e for e in many if e in ranked[:4]]  # the rule, in the order given
    assert [e for e in kept if e not in many] == few + other  # under the ceiling: all kept
    assert cap(examples, 4, 7) == kept and cap(examples[::-1], 4, 7) == kept[::-1]  # seeded, whatever the order
    assert len({tuple(e["call_id"] for e in cap(many, 4, seed)) for seed in range(5)}) > 1  # the seed draws
    assert cap(examples, 1, 7) == [e for e in examples if e in (ranked[0], *cap(few, 1, 7), *other)]


def test_the_cap_is_uniform_so_it_keeps_the_teachers_decisions_as_they_are():
    """Two columns in ten are kept by the teacher. Over many seeds every invocation is drawn about as
    often as any other, so the kept share of "yes" stays two in ten (a draw balanced by decision would
    give five in ten)."""
    examples = [example("1", "filter_column", f"t.c{i}") for i in range(10)]
    yes = {examples[0]["call_id"], examples[1]["call_id"]}
    seeds = range(400)
    drawn = [e["call_id"] for seed in seeds for e in cap(examples, 4, seed)]
    assert len(drawn) == 4 * len(seeds)
    assert all(120 <= drawn.count(e["call_id"]) <= 200 for e in examples)  # 160 expected of each
    assert 0.15 <= sum(c in yes for c in drawn) / len(drawn) <= 0.25


def test_coverage_counts_rows_and_the_filters_tables_columns_and_decisions():
    examples = [example("1", "filter_column", key) for key in ("a.x", "a.y", "b.z", "a.x@2")] + \
        [example("2", "filter_column", "a.x"), example("3", "filter_column", "u.v"), example("3", "select_tables")]
    db_of = {"1": "db1", "2": "db1", "3": "db2"}
    keeps = {e["call_id"]: e["invocation_key"].startswith("a.x") for e in examples if e["call_site"] == "filter_column"}
    assert coverage(examples, db_of, keeps) == {
        "filter_column": {"rows": 6, "yes": 3, "no": 3, "yes_share": 0.5,
                          # a repeated invocation (`@2`) and another question's are the same column
                          "databases": {"db1": {"tables": 2, "columns": 3}, "db2": {"tables": 1, "columns": 1}}},
        "select_tables": {"rows": 1}}
    assert coverage([examples[-1]], db_of, keeps) == {"select_tables": {"rows": 1}}


def column_call(question_id: str, key: str, parsed, response: str = "answer") -> dict:
    return call("r", question_id, "filter_column", key, messages=prompt("filter_column", f"question {question_id}", key),
                response=response, parsed=parsed)


def test_curation_counts_what_the_cap_drops_and_keeps_what_is_left():
    """`kept` is what the examples hold, after the cap; what the cap dropped is its own step."""
    calls = [column_call("1", f"t.c{i}", {"is_column_information_relevant": "No"}) for i in range(6)] + \
        [column_call("2", "t.c0", {"is_column_information_relevant": "Yes"}),
         call("r", "1", "select_tables", response='{"table_names": ["t"]}', parsed={"table_names": ["t"]})]
    examples, counts = curate(calls, {**SETTINGS, "max_per_question_call_site": 2}, fake_sql({}), SEED, DB_OF)
    assert examples == cap(curate(calls, {**SETTINGS, "max_per_question_call_site": 9}, fake_sql({}), SEED, DB_OF)[0], 2, SEED)
    assert counts["per_call_site"]["filter_column"] | {} == {
        "invocations": 7, "unparsed": 0, "sql_error": 0, "sql_empty": 0, "passed_filter": 7, "masked_sql": 0,
        "exact_duplicates": 0, "near_duplicates": 0, "over_cap": 4, "kept": 3}
    assert (counts["total"]["over_cap"], counts["total"]["kept"]) == (4, 4) and len(examples) == 4
    assert counts["cap"] == {
        "max_per_question_call_site": 2, "seed": SEED,
        "before": {"filter_column": {"rows": 7, "yes": 1, "no": 6, "yes_share": 0.1429,
                                     "databases": {"db": {"tables": 1, "columns": 6}}},
                   "select_tables": {"rows": 1}},
        "after": {"filter_column": {"rows": 3, "yes": 1, "no": 2, "yes_share": 0.3333,
                                    "databases": {"db": {"tables": 1, "columns": 2 + (
                                        "t.c0" not in [e["invocation_key"] for e in examples if e["question_id"] == "1"])}}},
                  "select_tables": {"rows": 1}}}
    other = curate(calls, {**SETTINGS, "max_per_question_call_site": 2}, fake_sql({}), SEED + 1, DB_OF)
    assert other[1]["cap"]["seed"] == SEED + 1 and other[1]["total"] == counts["total"]


@pytest.mark.parametrize("unreadable", [{"chain_of_thought_reasoning": "the answer was cut before the decis"},
                                        {"is_column_information_relevant": True}, ["Yes"]])
def test_a_filter_output_the_agent_cannot_read_counts_as_a_dropped_column(unreadable):
    """A cut-off answer still parses (the parser takes partial JSON); the agent then drops the column."""
    calls = [column_call("1", "t.a", unreadable), column_call("1", "t.b", {"is_column_information_relevant": "YES"})]
    examples, counts = curate(calls, SETTINGS, fake_sql({}), SEED, DB_OF)
    assert len(examples) == 2 and counts["cap"]["after"]["filter_column"] | {} == {
        "rows": 2, "yes": 1, "no": 1, "yes_share": 0.5, "databases": {"db": {"tables": 1, "columns": 2}}}


# ---------------------------------------------------------------- the execution

def teacher_config(config):
    config = json.loads(json.dumps(config))
    config["roles"]["production_llm"]["terms"] = {"weights_license": "https://example.org/license",
                                                   "provider_terms": "https://example.org/terms",
                                                   "checked_on": "2026-09-30"}
    return config


def b0_train_run(config, run_id="agent-B0-train-1", calls=None, **manifest):
    calls = calls if calls is not None else [
        call(run_id, "1", "generate_candidate", "t:0", response=GOLD, parsed={"SQL": GOLD}),
        call(run_id, "2", "generate_candidate", "t:0", response="x", parsed={"SQL": "SELECT nope FROM gas_t"}),
        call(run_id, "3", "generate_candidate", "t:0", response="y",
             parsed={"SQL": "SELECT id FROM gas_t WHERE country = 'XXX'"}),
        call(run_id, "3", "select_tables", response='{"table_names": ["gas_t"]}', parsed={"table_names": ["gas_t"]}),
    ]
    write_run(run_id, {"type": "agent", "arm": "B0", "split": "train", "question_ids": ["1", "2", "3"],
                       "config_sha256": config_sha256(config), **manifest}, calls, config)
    return run_id


def test_run_curate_executes_the_sql_on_the_pinned_database(tmp_path, monkeypatch):
    _, config_path, config = make_repo(tmp_path, monkeypatch)
    other = "SELECT id FROM gas_t WHERE segment = 'Value'"  # not the gold, returns a row: kept
    calls = [call("agent-B0-train-1", "1", "generate_candidate", "t:0", response=GOLD, parsed={"SQL": GOLD}),
             call("agent-B0-train-1", "2", "generate_candidate", "t:0", response="x", parsed={"SQL": "SELECT nope FROM gas_t"}),
             call("agent-B0-train-1", "3", "generate_candidate", "t:0", response=other, parsed={"SQL": other}),
             call("agent-B0-train-1", "3", "revise", "revise_1:0", response="y",
                  parsed={"refined_sql_query": "SELECT id FROM gas_t WHERE country = 'XXX'"}),
             call("agent-B0-train-1", "3", "select_tables", response='{"table_names": ["gas_t"]}', parsed={"table_names": ["gas_t"]})]
    source = b0_train_run(teacher_config(config), calls=calls)
    out = run_curate([source], str(config_path))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["type"] == "curate" and manifest["sources"] == [reference(source)]
    assert manifest["counts"]["per_call_site"]["generate_candidate"]["sql_error"] == 1
    assert manifest["counts"]["per_call_site"]["revise"]["sql_empty"] == 1
    examples = read_jsonl(out / "examples.jsonl")
    assert sorted((e["question_id"], e["call_site"]) for e in examples) == [
        ("1", "generate_candidate"), ("3", "generate_candidate"), ("3", "select_tables")]


def test_run_curate_caps_each_question_and_call_site_and_reports_the_coverage(tmp_path, monkeypatch):
    _, config_path, config = make_repo(tmp_path, monkeypatch)
    limit = config["curation"]["max_per_question_call_site"]
    assert limit == 4 and config["seeds"]["curation_sample"] == 20260934
    columns = ["gas_t.id", "gas_t.country", "gas_t.segment", "other_u.id", "other_u.note"]
    run = "agent-B0-train-1"
    calls = [call(run, q, "filter_column", key, messages=prompt("filter_column", f"question {q}", key),
                  response=answer, parsed={"is_column_information_relevant": answer})
             for q, keys in (("1", columns), ("2", columns[:2])) for key, answer in zip(keys, ("Yes", "No", "no", "YES", "No"))]
    calls.append(call(run, "1", "select_tables", response='{"table_names": ["gas_t"]}', parsed={"table_names": ["gas_t"]}))
    source = b0_train_run(teacher_config(config), calls=calls)
    out = run_curate([source], str(config_path))
    manifest, examples = json.loads((out / "manifest.json").read_text()), read_jsonl(out / "examples.jsonl")
    total = manifest["counts"]["total"]  # one over the ceiling; `kept` is what the examples hold
    assert (total["over_cap"], total["kept"]) == (1, 7) and manifest["n"] == len(examples) == 7
    by_group = {}
    for e in examples:
        by_group.setdefault((e["question_id"], e["call_site"]), []).append(e["invocation_key"])
    assert {group: len(keys) for group, keys in by_group.items()} == {
        ("1", "filter_column"): limit, ("2", "filter_column"): 2, ("1", "select_tables"): 1}
    dropped = (set(columns) - set(by_group[("1", "filter_column")])).pop()
    yes = 2 + 1 - (dropped in ("gas_t.id", "other_u.id"))  # the teacher's "Yes"/"YES", read as the agent reads them
    assert manifest["counts"]["cap"] == {
        "max_per_question_call_site": limit, "seed": 20260934,
        "before": {"filter_column": {"rows": 7, "yes": 3, "no": 4, "yes_share": 0.4286,
                                     "databases": {"tiny": {"tables": 2, "columns": 5}}},
                   "select_tables": {"rows": 1}},
        "after": {"filter_column": {"rows": 6, "yes": yes, "no": 6 - yes, "yes_share": round(yes / 6, 4),
                                    "databases": {"tiny": {"tables": 2, "columns": 5 - (dropped not in columns[:2])}}},
                  "select_tables": {"rows": 1}}}
    again = run_curate([source], str(config_path))  # the same seed, the same sample
    assert read_jsonl(again / "examples.jsonl") == examples
    raw = json.loads(json.dumps(config))
    raw["seeds"]["curation_sample"] = next(  # another seed draws another sample
        seed for seed in range(100) if cap(calls_as_examples(calls), limit, seed) != cap(calls_as_examples(calls), limit, 20260934))
    config_path.write_text(json.dumps(raw))
    other = read_jsonl(run_curate([source], str(config_path)) / "examples.jsonl")
    assert len(other) == 7 and other != examples


def calls_as_examples(calls):
    return [{k: c[k] for k in ("call_id", "question_id", "call_site", "invocation_key")} for c in calls]


def test_run_curate_refuses_outputs_it_may_not_train_on(tmp_path, monkeypatch):
    _, config_path, config = make_repo(tmp_path, monkeypatch)
    with pytest.raises(CurationError, match="terms"):
        run_curate([b0_train_run(config, "unchecked")], str(config_path))
    student = [call("mixed", "1", "select_tables", response="{}", parsed={}, role="cheap_alt", engine="cheap_alt")]
    with pytest.raises(CurationError, match="only the teacher"):
        run_curate([b0_train_run(teacher_config(config), "mixed", student)], str(config_path))
    with pytest.raises(JudgmentError, match="arm"):
        run_curate([b0_train_run(teacher_config(config), "b1", arm="B1")], str(config_path))
    with pytest.raises(JudgmentError, match="not 'done'"):
        run_curate([b0_train_run(teacher_config(config), "failed", status="failed")], str(config_path))


def test_datasets_follow_the_s3_clusters_in_trl_shape(tmp_path, monkeypatch):
    _, config_path, config = make_repo(tmp_path, monkeypatch)
    curated = run_curate([b0_train_run(teacher_config(config))], str(config_path)).name
    examples = read_jsonl(paths.RUNS / curated / "examples.jsonl")
    members = {examples[0]["call_id"]: "c0", examples[1]["call_id"]: "c1"}
    result = write_result("J5", {"curate": reference(curated)},
                          {"members": members, "clusters": ["c0", "c1", "c2"], "centroids": {"sha256": "ab" * 32}})
    (paths.ROOT / "train" / "datasets").mkdir(parents=True)
    (paths.ROOT / "train" / "datasets" / "c7.jsonl").write_text("{}\n")  # an earlier J5's cluster
    out = write_datasets(curated, str(result), str(config_path))
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["c0.jsonl", "c1.jsonl", "c2.jsonl"]
    lines = read_jsonl(out / "c0.jsonl")
    assert lines == [{"prompt": examples[0]["prompt"], "completion": examples[0]["completion"]}]
    assert lines[0]["completion"][0]["role"] == "assistant"
    assert read_jsonl(out / "c2.jsonl") == []
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["clusters"]["c0"] == {"n": 1, "rule_of_thumb": "below"} and manifest["rule_of_thumb"] == [10000, 100000]
    wrong = write_result("J5", {"curate": reference(curated)}, {"members": {examples[0]["call_id"]: "c0"},
                                                               "clusters": ["c0"], "centroids": {"sha256": "cd" * 32}})
    with pytest.raises(CurationError, match="exactly the curated"):
        write_datasets(curated, str(wrong), str(config_path))


def test_run_curate_refuses_overlapping_sources_and_a_changed_snapshot(tmp_path, monkeypatch):
    _, config_path, config = make_repo(tmp_path, monkeypatch)
    first = b0_train_run(teacher_config(config), "agent-B0-train-a")
    second = b0_train_run(teacher_config(config), "agent-B0-train-b")  # the same questions 1-3
    with pytest.raises(CurationError, match="repeats questions"):
        run_curate([first, second], str(config_path))
    snapshot = paths.RUNS / first / "config.json"
    changed = json.loads(snapshot.read_text())
    changed["splits"]["calib_size"] = 7
    snapshot.write_text(json.dumps(changed))
    with pytest.raises(CurationError, match="does not match its manifest"):
        run_curate([first], str(config_path))


def test_curation_runs_the_sql_at_the_pre_registered_date(tmp_path, monkeypatch):
    """The day is the source run's snapshot's `eval.fixed_date`, here one that is neither today's config.yaml
    nor the day the tests run: a query about "today" keeps its row only on that day."""
    _, config_path, config = make_repo(tmp_path, monkeypatch)
    fixed = teacher_config(config)
    fixed["eval"]["fixed_date"] = "2031-01-15"
    assert config["eval"]["fixed_date"] != "2031-01-15"
    that_day = "SELECT id FROM gas_t WHERE date('now') = '2031-01-15'"  # rows only on the snapshot's day
    nothing = "-- a comment, no statement"
    calls = [call("agent-d", "1", "generate_candidate", "t:0", response=that_day, parsed={"SQL": that_day}),
             call("agent-d", "2", "generate_candidate", "t:0", response=nothing, parsed={"SQL": nothing})]
    out = run_curate([b0_train_run(fixed, "agent-d", calls)], str(config_path))
    assert [e["question_id"] for e in read_jsonl(out / "examples.jsonl")] == ["1"]
    counts = json.loads((out / "manifest.json").read_text())["counts"]["per_call_site"]["generate_candidate"]
    assert counts["sql_error"] == 1  # no statement is an error, as eval reads it
    undated = json.loads(json.dumps(fixed))
    undated["eval"]["fixed_date"] = None
    with pytest.raises(CurationError, match="eval.fixed_date must be a pre-registered YYYY-MM-DD"):
        run_curate([b0_train_run(undated, "agent-u", calls)], str(config_path))
    other_day = json.loads(json.dumps(fixed))
    other_day["eval"]["fixed_date"] = "2031-01-16"
    first = b0_train_run(fixed, "agent-a", calls[:1], question_ids=["1"])
    second = b0_train_run(other_day, "agent-b", [call("agent-b", "2", "generate_candidate", "t:0", response=GOLD,
                                                      parsed={"SQL": GOLD})], question_ids=["2"])
    with pytest.raises(CurationError, match="different fixed dates"):
        run_curate([first, second], str(config_path))


def test_sources_run_under_different_fixed_dates_are_refused(tmp_path, monkeypatch):
    _, config_path, config = make_repo(tmp_path, monkeypatch)
    first = teacher_config(config)
    other = json.loads(json.dumps(first))
    other["eval"]["fixed_date"] = "2001-02-03"
    one = [call("agent-1", "1", "select_tables", parsed={"table_names": ["gas_t"]})]
    two = [call("agent-2", "2", "select_tables", parsed={"table_names": ["gas_t"]})]
    a = b0_train_run(first, "agent-1", one, question_ids=["1"])
    b = b0_train_run(other, "agent-2", two, question_ids=["2"])
    with pytest.raises(CurationError, match="different fixed dates"):
        run_curate([a, b], str(config_path))
