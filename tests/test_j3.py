"""J3 · cost: usage × dated prices, variants from the same usage, missing refused, estimated
labelled, SLM calls at the load test's cost, per question and per correct query, each question's
own cost (`by_question`), and a price that is one provider's."""
import pytest

from bench.judge import j3
from bench.judge.base import JudgmentError, canonical, read_result, write_result
from fixtures.fake import call, repo, usage, write_run

PRICES = {"as_of": "2026-09-30", "table": {
    "teacher-model": {"provider": "provider-x", "input_per_mtok": 2.0, "cached_input_per_mtok": 0.5, "output_per_mtok": 8.0, "batch_discount": 0.5},
    "cheap-model": {"provider": "provider-x", "input_per_mtok": 0.2, "cached_input_per_mtok": 0.2, "output_per_mtok": 0.6}}}
SLM = {"20%": 0.010, "50%": 0.004, "100%": 0.002}


def teacher_call(q, i=0, use=None, **kw):
    return call("r", q, "select_tables", f"k{i}", parsed={}, use=use or usage(1_000_000, 400_000, 100_000), **kw)


def test_prices_one_call_by_hand():
    cost = j3.api_cost(teacher_call("1"), PRICES["table"]["teacher-model"])
    assert cost["standard"] == pytest.approx(0.6 * 2.0 + 0.4 * 0.5 + 0.1 * 8.0)      # 2.2
    assert cost["no_cache"] == pytest.approx(1.0 * 2.0 + 0.1 * 8.0)                  # 2.8
    assert cost["batch"] == pytest.approx(2.2 * 0.5)


def test_totals_per_question_and_per_correct():
    calls = [teacher_call("1"), teacher_call("1", 1), teacher_call("2")]
    result = j3.judge(calls, ["1", "2"], {"1": True, "2": False}, PRICES, "teacher-model")
    assert result["total"]["standard"] == pytest.approx(6.6)
    assert result["per_question"]["standard"] == pytest.approx(3.3)
    assert result["per_correct"]["standard"] == pytest.approx(6.6)
    assert result["label"] == "measured" and result["replaceable_fraction"] is None
    assert result["by_call_site"]["select_tables"]["calls"] == 3
    none_right = j3.judge(calls, ["1", "2"], {"1": False, "2": False}, PRICES, "teacher-model")
    assert none_right["per_correct"]["standard"] is None


def test_batch_variant_needs_a_discount_in_the_table():
    cheap = teacher_call("1", model="cheap-model")
    assert j3.judge([cheap], ["1"], None, PRICES, "teacher-model")["total"]["batch"] is None


def test_usage_missing_is_refused_estimated_is_labelled_failures_are_unbilled():
    with pytest.raises(JudgmentError, match="no usage"):
        j3.judge([teacher_call("1", use=usage(source="missing"))], ["1"], None, PRICES, "teacher-model")
    failed = call("r", "1", "select_tables", "k9", response=None, parsed_ok=False, use=usage(source="missing"))
    estimated = teacher_call("1", use=usage(1_000_000, 0, 0, source="estimated"))
    result = j3.judge([failed, estimated], ["1"], None, PRICES, "teacher-model")
    assert result["failed_unbilled"] == 1 and result["estimated"] == 1 and result["label"] == "estimated"
    assert result["total"]["standard"] == pytest.approx(2.0) and result["lower_bound"]
    assert result["per_call"]["standard"] == pytest.approx(1.0)  # two calls, one unpriced


def test_refuses_an_undated_table_and_an_unknown_model():
    with pytest.raises(JudgmentError, match="as_of"):
        j3.judge([teacher_call("1")], ["1"], None, {**PRICES, "as_of": None}, "teacher-model")
    with pytest.raises(JudgmentError, match="no entry"):
        j3.judge([teacher_call("1", model="nobody")], ["1"], None, PRICES, "teacher-model")


def test_slm_calls_cost_the_load_test_per_request_at_each_utilization():
    slm = call("r", "1", "filter_column", "t.a", parsed={}, role="slm", engine="slm:qwen3-8b+lora:c0",
               model="qwen3-8b-c0", use=usage(1_000_000, 0, 100_000))
    api = teacher_call("1")
    with pytest.raises(JudgmentError, match="J8"):
        j3.judge([slm, api], ["1"], None, PRICES, "teacher-model")
    result = j3.judge([slm, api], ["1"], {"1": True}, PRICES, "teacher-model", SLM)
    assert result["total"]["standard@20%"] == pytest.approx(2.2 + 0.010)
    assert result["total"]["no_cache@100%"] == pytest.approx(2.8 + 0.002)
    assert "standard" not in result["total"] and result["slm_calls"] == 1
    fraction = result["replaceable_fraction"]
    assert fraction["calls"] == 0.5 and fraction["tokens"] == pytest.approx(1.1 / 2.2)
    assert fraction["cost_at_production_price"] == pytest.approx((1.0 * 2.0 + 0.1 * 8.0) / (2.8 + 2.2))


def test_run_reads_the_execution_its_eval_and_j8(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, {"prices": PRICES, "roles": {"production_llm": {"model": "teacher-model"}}})
    calls = [call("agent-B4", "1", "select_tables", parsed={}, role="slm", engine="slm:qwen3-8b+lora:c0",
                  model="qwen3-8b-c0", use=usage(1000, 0, 10))]
    write_run("agent-B4", {"type": "agent", "arm": "B4", "split": "test", "question_ids": ["1", "2"]}, calls)
    write_run("eval-B4", {"type": "eval", "source_run_id": "agent-B4", "status": None},
              files={"results.jsonl": [{"question_id": "1", "correct": True}, {"question_id": "2", "correct": False}]})
    j8 = write_result("J8", {}, {"engine": "slm:qwen3-8b", "cost_per_request": SLM,
                                 "combined": {"rule": "r", "engines": ["slm:qwen3-8b+lora:c0"], "loadtests": ["lt"]}})
    result = read_result(j3.run("agent-B4", "eval-B4", str(j8), config), "J3")
    assert result["result"]["per_correct"]["standard@50%"] == pytest.approx(0.004)
    assert result["result"]["per_question"]["standard@50%"] == pytest.approx(0.002)
    assert result["reads"]["j8"]["sha256"] == j8.parent.name and result["result"]["arm"] == "B4"
    assert not result["result"]["lower_bound"]
    write_run("eval-B4-per-call", {"type": "eval", "source_run_id": "agent-B4", "per_call": True, "status": None},
              files={"results.jsonl": [{"question_id": "1", "call_site": "select_tables", "invocation_key": "single", "correct": True}]})
    with pytest.raises(JudgmentError, match="per-call"):
        j3.run("agent-B4", "eval-B4-per-call", str(j8), config)
    other = write_result("J8", {}, {"engine": "slm:granite-4.2-8b", "cost_per_request": SLM})
    with pytest.raises(JudgmentError, match="measured"):
        j3.run("agent-B4", "eval-B4", str(other), config)


def test_slm_cost_through_adapters_is_labelled_extrapolated(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, {"prices": PRICES, "roles": {"production_llm": {"model": "teacher-model"}}})
    for run_id, engine in (("agent-B3", "slm:qwen3-8b"), ("agent-B4x", "slm:qwen3-8b+lora:c0")):
        write_run(run_id, {"type": "agent", "arm": run_id[6:8], "split": "test", "question_ids": ["1"]},
                  [call(run_id, "1", "select_tables", parsed={}, role="slm", engine=engine, model="m", use=usage(10, 0, 1))])
    base = write_result("J8", {}, {"engine": "slm:qwen3-8b", "cost_per_request": SLM,
                                   "combined": {"rule": "r", "engines": ["slm:qwen3-8b"], "loadtests": ["lt"]}})
    b3 = read_result(j3.run("agent-B3", None, str(base), config), "J3")["result"]
    assert b3["slm_cost_basis"]["basis"] == "measured"
    with pytest.raises(JudgmentError, match="did not measure every SLM engine"):  # the adapter was never load-tested
        j3.run("agent-B4x", None, str(base), config)
    # even when the load test measured that very adapter: a B4/B5 server serves many adapters at once
    adapter = write_result("J8", {}, {"engine": "slm:qwen3-8b", "cost_per_request": SLM,
                                      "combined": {"rule": "r", "engines": ["slm:qwen3-8b+lora:c0"], "loadtests": ["lt"]}})
    alone = read_result(j3.run("agent-B4x", None, str(adapter), config), "J3")["result"]
    assert alone["slm_cost_basis"]["basis"] == "extrapolated from per-adapter load tests"
    assert alone["slm_cost_basis"]["call_engines"] == ["slm:qwen3-8b+lora:c0"]


def test_a_cache_not_reported_is_priced_without_discount_as_an_upper_bound():
    unreported = teacher_call("1", use={"input": 1_000_000, "cached_input": None, "output": 0, "source": "api"})
    result = j3.judge([unreported, teacher_call("2")], ["1", "2"], None, PRICES, "teacher-model")
    assert result["cache_not_reported"] == 1 and result["upper_bound"]
    assert result["total"]["standard"] == pytest.approx(1.0 * 2.0 + 2.2)  # all its input at the input price
    assert not j3.judge([teacher_call("2")], ["2"], None, PRICES, "teacher-model")["upper_bound"]


def test_the_price_table_used_is_recorded():
    result = j3.judge([teacher_call("1")], ["1"], None, PRICES, "teacher-model")
    expected = __import__("hashlib").sha256(canonical(PRICES["table"])).hexdigest()
    assert result["prices"] == {"as_of": "2026-09-30", "sha256": expected}


def test_failed_slm_calls_count_on_both_sides_of_the_fraction():
    slm_ok = call("r", "1", "filter_column", "t.a", parsed={}, role="slm", engine="slm:q+lora:c0", model="m",
                  use=usage(1000, 0, 0))
    slm_failed = call("r", "1", "filter_column", "t.b", response=None, parsed_ok=False, role="slm", engine="slm:q+lora:c0",
                      model="m", use=usage(source="missing"))
    llm = teacher_call("1", use=usage(1000, 0, 0))
    fraction = j3.judge([slm_ok, slm_failed, llm], ["1"], None, PRICES, "teacher-model", SLM)["replaceable_fraction"]
    assert fraction["calls"] == pytest.approx(2 / 3)  # both SLM calls, of all three
    assert fraction["tokens"] == pytest.approx(0.5)   # billed calls on both sides


def test_slm_calls_are_priced_at_the_engines_they_used(tmp_path, monkeypatch):
    """One load test measures the base and an adapter: the base's calls cost the base's price, not
    the dearer adapter's."""
    _, config = repo(tmp_path, monkeypatch, {"prices": PRICES, "roles": {"production_llm": {"model": "teacher-model"}}})
    write_run("agent-B3y", {"type": "agent", "arm": "B3", "split": "test", "question_ids": ["1"]},
              [call("agent-B3y", "1", "select_tables", parsed={}, role="slm", engine="slm:qwen3-8b", model="m",
                    use=usage(10, 0, 1))])
    both = write_result("J8", {}, {"engine": "slm:qwen3-8b", "cost_per_request": {"100%": 0.5},
                                   "engines": {"slm:qwen3-8b": {"cost_per_request": {"100%": 0.1}},
                                               "slm:qwen3-8b+lora:c0": {"cost_per_request": {"100%": 0.5}}},
                                   "combined": {"rule": "r", "engines": ["slm:qwen3-8b", "slm:qwen3-8b+lora:c0"]}})
    result = read_result(j3.run("agent-B3y", None, str(both), config), "J3")["result"]
    assert result["total"]["standard@100%"] == pytest.approx(0.1) and result["slm_cost_basis"]["basis"] == "measured"


def test_by_question_is_each_questions_cost_in_every_scenario():
    """T9: {question: {scenario: cost}}, what the report resamples for the cost ratio's interval."""
    calls = [teacher_call("1"), teacher_call("1", 1), teacher_call("2")]
    result = j3.judge(calls, ["1", "2", "3"], {"1": True, "2": False, "3": False}, PRICES, "teacher-model")
    assert result["by_question"]["1"] == {"standard": pytest.approx(4.4), "no_cache": pytest.approx(5.6), "batch": pytest.approx(2.2)}
    assert result["by_question"]["2"]["standard"] == pytest.approx(2.2)
    assert result["by_question"]["3"] == {"standard": 0.0, "no_cache": 0.0, "batch": 0.0}  # no call: it costs nothing
    assert sum(q["standard"] for q in result["by_question"].values()) == pytest.approx(result["total"]["standard"])
    with pytest.raises(JudgmentError, match="question 2, which the execution does not list"):
        j3.judge(calls, ["1"], None, PRICES, "teacher-model")


def test_by_question_of_an_slm_arm_has_the_utilizations_on_every_question():
    slm = call("r", "1", "filter_column", "t.a", parsed={}, role="slm", engine="slm:qwen3-8b+lora:c0",
               model="qwen3-8b-c0", use=usage(1_000_000, 0, 100_000))
    result = j3.judge([slm, teacher_call("1"), teacher_call("2")], ["1", "2", "3"], None, PRICES, "teacher-model", SLM)
    assert set(result["by_question"]["1"]) == set(result["by_question"]["2"]) == set(result["by_question"]["3"]) \
        == set(result["total"])
    assert result["by_question"]["1"]["standard@20%"] == pytest.approx(2.2 + 0.010)
    assert result["by_question"]["2"]["standard@20%"] == pytest.approx(2.2)   # an API-only question of an SLM arm
    assert result["by_question"]["3"]["standard@100%"] == 0.0
    for scenario in ("standard@20%", "standard@50%", "standard@100%"):
        assert sum(q[scenario] for q in result["by_question"].values()) == pytest.approx(result["total"][scenario])


def test_a_call_is_priced_only_at_its_own_providers_price():
    """T6: the price entry of an API model carries `provider`, and a call another provider served is
    refused, never priced at the wrong provider's price."""
    prices = {"as_of": "2026-09-30", "table": {"teacher-model": {**PRICES["table"]["teacher-model"], "provider": "provider-a"}}}
    served_by = lambda provider: teacher_call("1", provider=provider)  # noqa: E731
    assert j3.judge([served_by("provider-a")], ["1"], None, prices, "teacher-model")["total"]["standard"] == pytest.approx(2.2)
    with pytest.raises(JudgmentError, match="served by provider 'provider-b'.*is of provider 'provider-a'"):
        j3.judge([served_by("provider-b")], ["1"], None, prices, "teacher-model")
    with pytest.raises(JudgmentError, match="served by provider None"):  # a C1 line that records no provider
        j3.judge([served_by(None)], ["1"], None, prices, "teacher-model")
    unnamed = {"as_of": "2026-09-30", "table": {"teacher-model": {k: v for k, v in PRICES["table"]["teacher-model"].items()
                                                                  if k != "provider"}}}
    for provider in ("provider-a", None):  # an entry that names none prices nothing, a call that names none included
        with pytest.raises(JudgmentError, match="is of provider None"):
            j3.judge([served_by(provider)], ["1"], None, unnamed, "teacher-model")
    slm = {**call("r", "1", "filter_column", "t.a", parsed={}, role="slm", engine="slm:q+lora:c0", model="m",
                  use=usage(1000, 0, 0)), "provider": None}
    assert j3.judge([slm], ["1"], None, prices, "teacher-model", SLM)["slm_calls"] == 1  # the SLM has no price entry


def test_run_records_what_it_read_of_the_configuration(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, {"prices": PRICES, "roles": {"production_llm": {"model": "teacher-model"}}})
    write_run("agent-B0", {"type": "agent", "arm": "B0", "split": "calib", "question_ids": ["1"]}, [{**teacher_call("1"), "run_id": "agent-B0"}])
    reads = read_result(j3.run("agent-B0", None, None, config), "J3")["reads"]
    assert reads["config"] == ["prices", "roles.production_llm.model"] and reads["run"]["run_id"] == "agent-B0"
