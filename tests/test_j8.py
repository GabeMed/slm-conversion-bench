"""J8 · load: sustained throughput within the p95 bound, and the cost per request at each
utilization, from AIPerf exports."""
import pytest

from bench.judge import j8
from bench.judge.base import JudgmentError, read_result
from fixtures.fake import repo, write_run


def export(p95_ms, rps):
    """The two blocks J8 reads, as AIPerf's JSON export writes them."""
    return {"schema_version": "1.5", "request_latency": {"unit": "ms", "avg": p95_ms / 2, "p50": p95_ms / 2, "p95": p95_ms},
            "request_throughput": {"unit": "requests/sec", "avg": rps}}


def loadtest(run_id, concurrency, p95_ms, rps, engine="slm:qwen3-8b", gpu="L4", cache=True):
    write_run(run_id, {"type": "loadtest", "engine": engine, "gpu": gpu, "concurrency": concurrency, "prefix_cache": cache},
              files={j8.EXPORT: export(p95_ms, rps)})
    return run_id


def test_sustained_throughput_is_the_best_within_the_bound():
    levels = [{"concurrency": 1, "p95_ms": 400, "throughput_rps": 2.0},
              {"concurrency": 8, "p95_ms": 900, "throughput_rps": 10.0},
              {"concurrency": 32, "p95_ms": 3000, "throughput_rps": 14.0}]
    result = j8.judge(levels, 0.80, 1000, [0.2, 0.5, 1.0])
    assert result["sustained"]["concurrency"] == 8 and result["requests_per_hour"] == 36000
    assert result["cost_per_request"] == pytest.approx({"20%": 0.80 / 7200, "50%": 0.80 / 18000, "100%": 0.80 / 36000})
    with pytest.raises(JudgmentError, match="no load level"):
        j8.judge(levels, 0.80, 100, [0.5])


def test_units_are_checked():
    assert j8.level(export(500, 3.0)) == {"p95_ms": 500.0, "throughput_rps": 3.0}
    wrong = export(500, 3.0)
    wrong["request_latency"]["unit"] = "s"
    with pytest.raises(JudgmentError, match="units"):
        j8.level(wrong)


def test_run_reads_loadtests_and_pre_registered_settings(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, {"cost": {"p95_slo_ms": 1000},
                                             "modal": {"gpu_prices": {"as_of": "2026-09-30", "usd_per_s": {"L4": 0.8 / 3600}}}})
    runs = [loadtest("loadtest-1", 1, 400, 2.0), loadtest("loadtest-8", 8, 900, 10.0)]
    result = read_result(j8.run(runs, config), "J8")
    assert result["result"]["engine"] == "slm:qwen3-8b" and result["result"]["gpu"] == "L4"
    assert result["result"]["cost_per_request"]["50%"] == pytest.approx(0.8 / 18000)
    assert result["result"]["gpu_prices_as_of"] == "2026-09-30"
    assert set(result["reads"]["loadtests"]) == set(runs)
    loadtest("loadtest-other", 4, 500, 5.0, engine="slm:granite-4.2-8b")
    with pytest.raises(JudgmentError, match="one base"):
        j8.run(runs + ["loadtest-other"], config)
    loadtest("loadtest-nocache", 2, 500, 5.0, cache=False)
    with pytest.raises(JudgmentError, match="prefix cache"):
        j8.run(runs + ["loadtest-nocache"], config)
    config["modal"]["gpu_prices"]["usd_per_s"] = {}
    with pytest.raises(JudgmentError, match="no dated price"):
        j8.run(runs, config)
    config["cost"]["p95_slo_ms"] = None
    with pytest.raises(JudgmentError, match="p95_slo_ms"):
        j8.run(runs, config)


def test_adapters_are_measured_apart_and_combined_by_the_stated_rule(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, {"cost": {"p95_slo_ms": 1000},
                                             "modal": {"gpu_prices": {"as_of": "2026-09-30", "usd_per_s": {"L4": 0.8 / 3600}}}})
    runs = [loadtest("lt-c0-1", 1, 400, 4.0, engine="slm:qwen3-8b+lora:c0"),
            loadtest("lt-c0-8", 8, 900, 10.0, engine="slm:qwen3-8b+lora:c0"),
            loadtest("lt-c1-8", 8, 800, 5.0, engine="slm:qwen3-8b+lora:c1")]
    result = read_result(j8.run(runs, config), "J8")["result"]
    assert result["engine"] == "slm:qwen3-8b"
    assert result["engines"]["slm:qwen3-8b+lora:c0"]["sustained"]["throughput_rps"] == 10.0
    # combined: the dearer engine's cost per request (c1 sustains 5 rps)
    assert result["cost_per_request"]["100%"] == pytest.approx(0.8 / (5.0 * 3600))
    assert result["combined"] == {"rule": "the highest cost per request among the engines measured",
                                  "engines": ["slm:qwen3-8b+lora:c0", "slm:qwen3-8b+lora:c1"], "loadtests": sorted(runs)}
