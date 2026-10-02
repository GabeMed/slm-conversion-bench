"""J8 · load: the SLO rule, the container's all-in price, sustained throughput within the SLO, and
the cost per request at each utilization, from AIPerf exports."""
import hashlib
import json

import pytest

from bench.judge import j8
from bench.judge.base import JudgmentError, canonical, read_result
from fixtures.fake import call, repo, write_run

PILOT = "agent-B0-calib"


def settings(cap_ms=1000, gpu_per_hour=0.8, cpu=0.0, memory=0.0):
    """Overrides of the shipped configuration: the SLO cap and the prices of an L4 container (CPU
    and memory free unless a test prices them, so the GPU's price is the container's)."""
    return {"cost": {"p95_slo_cap_ms": cap_ms},
            "modal": {"gpu_prices": {"as_of": "2026-09-30", "usd_per_s": {"L4": gpu_per_hour / 3600},
                                     "cpu_usd_per_core_s": cpu, "memory_usd_per_gib_s": memory}}}


def pilot(latencies_ms=(5000,), run_id=PILOT, **manifest):
    """The pilot's B0 execution on calib, one first attempt per latency."""
    write_run(run_id, {"type": "agent", "arm": "B0", "split": "calib", **manifest},
              [call(run_id, str(q), "select_tables", latency_ms=ms) for q, ms in enumerate(latencies_ms, start=1)])
    return run_id


def export(p95_ms, rps):
    """The two blocks J8 reads, as AIPerf's JSON export writes them."""
    return {"schema_version": "1.5", "request_latency": {"unit": "ms", "avg": p95_ms / 2, "p50": p95_ms / 2, "p95": p95_ms},
            "request_throughput": {"unit": "requests/sec", "avg": rps}}


def loadtest(run_id, concurrency, p95_ms, rps, engine="slm:qwen3-8b", gpu="L4", cache=True, sweep=None):
    write_run(run_id, {"type": "loadtest", "engine": engine, "gpu": gpu, "concurrency": concurrency, "prefix_cache": cache,
                       "sweep_id": sweep or f"sweep-{engine}"}, files={j8.EXPORT: export(p95_ms, rps)})
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
    _, config = repo(tmp_path, monkeypatch, settings())
    runs = [loadtest("loadtest-1", 1, 400, 2.0), loadtest("loadtest-8", 8, 900, 10.0)]
    result = read_result(j8.run(runs, config, slo_from=pilot()), "J8")
    assert result["result"]["engine"] == "slm:qwen3-8b" and result["result"]["gpu"] == "L4"
    assert result["result"]["cost_per_request"]["50%"] == pytest.approx(0.8 / 18000)
    assert result["result"]["gpu_prices_as_of"] == "2026-09-30"
    assert set(result["reads"]["loadtests"]) == set(runs)
    assert result["reads"]["slo_from"]["run_id"] == PILOT
    assert result["reads"]["config"] == ["cost.p95_slo_cap_ms", "cost.slo_from", "cost.utilizations", "modal.gpu_prices",
                                         "serving.cpu", "serving.memory_gib"]  # the configuration keys J8 read
    loadtest("loadtest-other", 4, 500, 5.0, engine="slm:granite-4.2-8b")
    with pytest.raises(JudgmentError, match="one base"):
        j8.run(runs + ["loadtest-other"], config, slo_from=PILOT)
    loadtest("loadtest-nocache", 2, 500, 5.0, cache=False)
    with pytest.raises(JudgmentError, match="prefix cache"):
        j8.run(runs + ["loadtest-nocache"], config, slo_from=PILOT)
    config["modal"]["gpu_prices"]["usd_per_s"] = {}
    with pytest.raises(JudgmentError, match="no dated price"):
        j8.run(runs, config, slo_from=PILOT)
    config["cost"]["p95_slo_cap_ms"] = None
    with pytest.raises(JudgmentError, match="p95_slo_cap_ms"):
        j8.run(runs, config, slo_from=PILOT)


def test_the_slos_execution_is_the_one_the_configuration_names(tmp_path, monkeypatch):
    """Once `cost.slo_from` is set, the SLO cannot be measured on another execution chosen after the
    load test; unset (before the pilot ran), whoever runs J8 names it."""
    _, config = repo(tmp_path, monkeypatch, settings())
    runs = [loadtest(f"loadtest-{c}", c, p95, rps) for c, p95, rps in ((1, 300, 2.0), (4, 700, 6.0))]
    config["cost"]["slo_from"] = pilot()
    named = read_result(j8.run(runs, config, slo_from=PILOT), "J8")
    assert named["reads"]["slo_from"]["run_id"] == PILOT and "cost.slo_from" in named["reads"]["config"]
    with pytest.raises(JudgmentError, match=f"cost.slo_from names {PILOT}, not pilot-other"):
        j8.run(runs, config, slo_from=pilot(run_id="pilot-other"))


def test_p95_interpolates_between_order_statistics():
    assert j8.p95([700]) == 700.0
    assert j8.p95(list(range(1, 101))) == pytest.approx(95.05)  # rank 0.95 × 99 = 94.05: between the 95th and 96th
    assert j8.p95([100, 200, 300, 400, 500, 600, 700, 800, 900, 5000]) == pytest.approx(3155.0)  # 900 + 0.55 × 4100
    assert j8.p95([30, 10, 20]) == pytest.approx(29.0)  # order does not matter


def test_the_slo_is_the_stricter_of_the_cap_and_the_llms_own_p95(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(cap_ms=1000))
    runs = [loadtest("loadtest-1", 1, 400, 2.0), loadtest("loadtest-8", 8, 900, 10.0)]
    # the LLM is slower than the cap: the cap binds, and the level at 900 ms is sustained
    slow = read_result(j8.run(runs, config, slo_from=pilot([2000] * 20, "pilot-slow")), "J8")["result"]
    assert slow["slo"] == {"cap_ms": 1000, "llm_p95_ms": 2000.0, "slo_ms": 1000, "first_attempts": 20}
    assert slow["engines"]["slm:qwen3-8b"]["sustained"]["concurrency"] == 8
    # the LLM is faster than the cap: its p95 binds, and the SLM must keep up with it
    fast = read_result(j8.run(runs, config, slo_from=pilot([500] * 20, "pilot-fast")), "J8")["result"]
    assert (fast["slo"]["llm_p95_ms"], fast["slo"]["slo_ms"]) == (500.0, 500.0)
    assert fast["engines"]["slm:qwen3-8b"]["sustained"]["concurrency"] == 1
    assert fast["cost_per_request"]["100%"] == pytest.approx(0.8 / (2.0 * 3600))
    # no level answers as fast as the LLM: no sustained throughput, never a silent fall back to the cap
    with pytest.raises(JudgmentError, match="no load level keeps p95 within 300.0 ms"):
        j8.run(runs, config, slo_from=pilot([300] * 20, "pilot-faster"))


def test_the_llms_p95_is_over_the_first_attempts_of_a_done_b0_run_on_calib(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(cap_ms=10000))
    runs = [loadtest("loadtest-4", 4, 3000, 6.0), loadtest("loadtest-8", 8, 4000, 10.0)]
    # the SLO is the p95 of the latencies, not their largest, smallest or first: 900 + 0.55 × (5000 − 900)
    spread = read_result(j8.run(runs, config, slo_from=pilot([700, 100, 200, 300, 400, 500, 600, 800, 900, 5000], "pilot-spread")),
                         "J8")["result"]
    assert spread["slo"] == {"cap_ms": 10000, "llm_p95_ms": 3155.0, "slo_ms": 3155.0, "first_attempts": 10}
    assert spread["engines"]["slm:qwen3-8b"]["sustained"]["concurrency"] == 4  # 3000 ms is within it, 4000 ms is not
    first = call("pilot-retries", "1", "select_tables", latency_ms=1000, parsed_ok=False)
    retry = call("pilot-retries", "1", "select_tables", latency_ms=9000, attempt=2, retry_of=first["call_id"])
    failed = call("pilot-retries", "2", "select_tables", latency_ms=3500, response=None, parsed_ok=False)
    write_run("pilot-retries", {"type": "agent", "arm": "B0", "split": "calib"}, [first, retry, failed])
    measured = read_result(j8.run(runs, config, slo_from="pilot-retries"), "J8")["result"]["slo"]
    # the retry's 9000 ms is not a first attempt; the first attempt that failed is one: 1000 + 0.95 × 2500
    assert (measured["llm_p95_ms"], measured["first_attempts"]) == (3375.0, 2)
    for run_id, manifest, reason in (("pilot-b1", {"arm": "B1"}, "arm is 'B1'"), ("pilot-train", {"split": "train"}, "split is 'train'"),
                                     ("pilot-failed", {"status": "failed"}, "not 'done'"),
                                     ("pilot-replay", {"type": "replay"}, "type is 'replay'")):
        with pytest.raises(JudgmentError, match=reason):
            j8.run(runs, config, slo_from=pilot(run_id=run_id, **manifest))
    write_run("pilot-empty", {"type": "agent", "arm": "B0", "split": "calib"}, [])
    with pytest.raises(JudgmentError, match="has no call"):
        j8.run(runs, config, slo_from="pilot-empty")


def test_the_price_is_the_containers_gpu_cpu_and_memory(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(gpu_per_hour=1.8, cpu=0.00001, memory=0.000002))
    config["serving"].update(cpu=8, memory_gib=32)
    runs = [loadtest("loadtest-8", 8, 900, 10.0)]
    result = read_result(j8.run(runs, config, slo_from=pilot()), "J8")["result"]
    per_second = {"gpu": 1.8 / 3600, "cpu": 8 * 0.00001, "memory": 32 * 0.000002}
    assert result["price_per_second"] == pytest.approx(per_second)
    assert result["price_per_hour"] == pytest.approx(1.8 + 8 * 0.036 + 32 * 0.0072)  # 2.3184, not the GPU's 1.8
    assert result["cost_per_request"]["100%"] == pytest.approx(2.3184 / 36000)
    # the hash is of the whole price block, the one the report compares with the configuration's
    assert result["gpu_prices_sha256"] == hashlib.sha256(canonical(config["modal"]["gpu_prices"])).hexdigest()
    dearer = {**config, "modal": {"gpu_prices": {**config["modal"]["gpu_prices"], "cpu_usd_per_core_s": 0.00002}}}
    other = read_result(j8.run(runs, dearer, slo_from=PILOT), "J8")["result"]
    assert other["gpu_prices_sha256"] != result["gpu_prices_sha256"] and other["price_per_hour"] > result["price_per_hour"]
    for where, key, name in (("modal", "cpu_usd_per_core_s", "modal.gpu_prices.cpu_usd_per_core_s"),
                             ("modal", "memory_usd_per_gib_s", "modal.gpu_prices.memory_usd_per_gib_s"),
                             ("serving", "cpu", "serving.cpu"), ("serving", "memory_gib", "serving.memory_gib")):
        broken = {**config, "modal": {"gpu_prices": dict(config["modal"]["gpu_prices"])}, "serving": dict(config["serving"])}
        (broken["modal"]["gpu_prices"] if where == "modal" else broken["serving"])[key] = None
        with pytest.raises(JudgmentError, match=f"{name} not set"):  # never the GPU alone
            j8.run(runs, broken, slo_from=PILOT)


def test_the_shipped_configuration_prices_the_container_all_in(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch)
    assert config["cost"]["p95_slo_cap_ms"] == 17000 and config["loadtest"]["concurrency"] == [4, 8, 16, 32, 64, 128]
    per_second = j8.container_price(config, "L40S")
    assert per_second == pytest.approx({"gpu": 0.000542, "cpu": 8 * 0.0000131, "memory": 32 * 0.00000222})
    assert sum(per_second.values()) * 3600 == pytest.approx(2.584224)  # US$2.58/h, against the GPU's 1.95


def test_adapters_are_measured_apart_and_combined_by_the_stated_rule(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings())
    pilot()
    runs = [loadtest("lt-c0-1", 1, 400, 4.0, engine="slm:qwen3-8b+lora:c0"),
            loadtest("lt-c0-8", 8, 900, 10.0, engine="slm:qwen3-8b+lora:c0"),
            loadtest("lt-c1-8", 8, 800, 5.0, engine="slm:qwen3-8b+lora:c1")]
    result = read_result(j8.run(runs, config, slo_from=PILOT), "J8")["result"]
    assert result["engine"] == "slm:qwen3-8b"
    assert result["engines"]["slm:qwen3-8b+lora:c0"]["sustained"]["throughput_rps"] == 10.0
    # combined: the dearer engine's cost per request (c1 sustains 5 rps)
    assert result["cost_per_request"]["100%"] == pytest.approx(0.8 / (5.0 * 3600))
    assert result["combined"] == {"rule": "the highest cost per request among the engines measured",
                                  "engines": ["slm:qwen3-8b+lora:c0", "slm:qwen3-8b+lora:c1"], "loadtests": sorted(runs),
                                  "sweeps": ["sweep-slm:qwen3-8b+lora:c0", "sweep-slm:qwen3-8b+lora:c1"]}


def test_one_sweep_per_engine_unless_one_is_named(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings())
    pilot()
    runs = [loadtest("lt-a1", 1, 400, 2.0, sweep="sweep-a"), loadtest("lt-a8", 8, 900, 10.0, sweep="sweep-a"),
            loadtest("lt-b8", 8, 900, 4.0, sweep="sweep-b")]
    with pytest.raises(JudgmentError, match="several sweeps"):
        j8.run(runs, config, slo_from=PILOT)
    chosen = read_result(j8.run(runs, config, sweeps=["sweep-b"], slo_from=PILOT), "J8")["result"]
    assert chosen["combined"]["sweeps"] == ["sweep-b"] and chosen["combined"]["loadtests"] == ["lt-b8"]
    assert chosen["cost_per_request"]["100%"] == pytest.approx(0.8 / (4.0 * 3600))
    write_run("lt-none", {"type": "loadtest", "engine": "slm:qwen3-8b", "gpu": "L4", "concurrency": 1, "prefix_cache": True},
              files={j8.EXPORT: export(400, 2.0)})
    with pytest.raises(JudgmentError, match="no sweep_id"):
        j8.run(["lt-none"], config, slo_from=PILOT)


def test_naming_a_sweep_never_drops_an_engine_with_one(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings())
    pilot()
    runs = [loadtest("c0-a", 8, 900, 10.0, engine="slm:qwen3-8b+lora:c0", sweep="sweep-c0-a"),
            loadtest("c0-b", 8, 900, 6.0, engine="slm:qwen3-8b+lora:c0", sweep="sweep-c0-b"),
            loadtest("c1", 8, 900, 5.0, engine="slm:qwen3-8b+lora:c1", sweep="sweep-c1")]
    result = read_result(j8.run(runs, config, sweeps=["sweep-c0-b"], slo_from=PILOT), "J8")["result"]
    assert result["combined"]["engines"] == ["slm:qwen3-8b+lora:c0", "slm:qwen3-8b+lora:c1"]  # c1 kept, not named
    assert result["combined"]["sweeps"] == ["sweep-c0-b", "sweep-c1"]
    with pytest.raises(JudgmentError, match="name exactly one"):
        j8.run(runs, config, sweeps=["sweep-c0-a", "sweep-c0-b"], slo_from=PILOT)
    with pytest.raises(JudgmentError, match=r"no load test has the named sweep\(s\) \['sweep-c0-typo'\]"):
        j8.run(runs, config, sweeps=["sweep-c0-b", "sweep-c0-typo"], slo_from=PILOT)  # a typo is refused, never ignored
    assert len(result["gpu_prices_sha256"]) == 64


def test_the_configuration_refuses_an_unusable_slo_cap_price_or_memory(tmp_path, monkeypatch):
    from bench.contracts.config import validate_config
    _, config = repo(tmp_path, monkeypatch)
    assert validate_config(config) == []
    for where, key, wrong, reason in ((("cost",), "p95_slo_cap_ms", "17s", "cost.p95_slo_cap_ms must be a number > 0"),
                                      (("cost",), "p95_slo_cap_ms", 0, "cost.p95_slo_cap_ms must be a number > 0"),
                                      (("cost",), "p95_slo_cap_ms", None, "cost.p95_slo_cap_ms must be a number > 0"),
                                      (("cost",), "slo_from", 7, "cost.slo_from must be a run id or null"),
                                      (("serving",), "memory_gib", -32, "serving.memory_gib must be a number > 0"),
                                      (("serving",), "memory_gib", True, "serving.memory_gib must be a number > 0"),
                                      (("modal", "gpu_prices"), "cpu_usd_per_core_s", -0.1,
                                       "modal.gpu_prices.cpu_usd_per_core_s must be a number >= 0"),
                                      (("modal", "gpu_prices"), "memory_usd_per_gib_s", "free",
                                       "modal.gpu_prices.memory_usd_per_gib_s must be a number >= 0")):
        broken = json.loads(json.dumps(config))
        node = broken
        for part in where:
            node = node[part]
        node[key] = wrong
        assert validate_config(broken) == [reason]
    free = json.loads(json.dumps(config))
    free["modal"]["gpu_prices"].update(cpu_usd_per_core_s=0, memory_usd_per_gib_s=0.0)  # a price may be zero
    del free["cost"]["p95_slo_cap_ms"]  # a key that is absent is J8's to refuse, when it runs
    assert validate_config(free) == []
