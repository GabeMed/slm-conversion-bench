"""J8 · load: the SLO rule, the container's all-in price, sustained throughput within the SLO, and
the cost per request at each utilization, from AIPerf exports."""
import hashlib
import json

import pytest

from bench.judge import j8
from bench.judge.base import JudgmentError, canonical, read_result
from fixtures.fake import aiperf_records as records, call, repo, write_run

PILOT = "agent-B0-calib"


def settings(cap_ms=1000, gpu_per_hour=0.8, cpu=0.0, memory=0.0):
    """Overrides of the shipped configuration: the SLO cap, no SLO execution fixed yet (the shipped one
    names the real pilot's; each test names its own), and the prices of an L4 container (CPU and memory
    free unless a test prices them, so the GPU's price is the container's)."""
    return {"cost": {"p95_slo_cap_ms": cap_ms, "slo_from": None, "limit_min_calls": 1, "limit_pool": {},
                     "steady_inflight_share": 0.9, "steady_min_requests": 100, "steady_min_share": 0.5},
            "modal": {"gpu_prices": {"as_of": "2026-09-30", "usd_per_s": {"L4": gpu_per_hour / 3600},
                                     "cpu_usd_per_core_s": cpu, "memory_usd_per_gib_s": memory},
                      "gpu_prices_sensitivity": None}}  # none, unless a test prices one


def pilot(latencies_ms=(5000,), run_id=PILOT, **manifest):
    """The pilot's B0 execution on calib, one first attempt per latency (no cache break: no warm-up gate)."""
    write_run(run_id, {"type": "agent", "arm": "B0", "split": "calib", **manifest},
              [call(run_id, str(q), "select_tables", latency_ms=ms) for q, ms in enumerate(latencies_ms, start=1)],
              config={})
    return run_id


def export(p95_ms, rps, **status):
    """The blocks J8 reads, as AIPerf's JSON export writes them."""
    return {"schema_version": "1.5", "request_latency": {"unit": "ms", "avg": p95_ms / 2, "p50": p95_ms / 2, "p95": p95_ms},
            "request_throughput": {"unit": "requests/sec", "avg": rps},
            **{"error_summary": [], "was_cancelled": False, "is_complete": True, **status}}


def loadtest(run_id, concurrency, p95_ms, rps, engine="slm:qwen3-8b", gpu="L4", cache=True, sweep=None,
             sites=None, kind="base", source=None, rows=None, **status):
    """One level as F3 writes it: its sweep's pool by call site (`sites`), its traffic kind, its source run, and
    AIPerf's export and per-request records (`rows`, by default a steady level at this p95 and throughput)."""
    write_run(run_id, {"type": "loadtest", "engine": engine, "gpu": gpu, "concurrency": concurrency, "prefix_cache": cache,
                       "sweep_id": sweep or f"sweep-{engine}", "traffic_kind": kind, "source_run_id": source,
                       "composition": {"source": {"call_sites": sites or {"select_tables": 100}}}},
              files={j8.EXPORT: export(p95_ms, rps, **status),
                     j8.RECORDS: rows if rows is not None else records(concurrency, p95_ms, rps)})
    return run_id


def test_sustained_throughput_is_the_best_within_the_bound():
    levels = [{"concurrency": 1, "p95_ms": 400, "throughput_rps": 2.0},
              {"concurrency": 8, "p95_ms": 900, "throughput_rps": 10.0},
              {"concurrency": 32, "p95_ms": 3000, "throughput_rps": 14.0}]
    result = j8.judge(levels, 0.80, 1000, [0.2, 0.5, 1.0])
    assert result["sustained"]["concurrency"] == 8 and result["requests_per_hour"] == 36000
    assert result["cost_per_request"] == pytest.approx({"20%": 0.80 / 7200, "50%": 0.80 / 18000, "100%": 0.80 / 36000})
    assert result["outside_slo"] is False
    # the router's time is added to each level's p95: 900 + 150 is over 1000, so level 1 is the one sustained
    assert j8.judge(levels, 0.80, 1000, [1.0], added_ms=150)["sustained"]["concurrency"] == 1
    # no level within the limit is a result: the engine sustains none, and has no cost
    outside = j8.judge(levels, 0.80, 100, [0.5])
    assert outside["outside_slo"] is True and outside["levels"] == levels and outside["p95_slo_ms"] == 100
    assert (outside["sustained"], outside["requests_per_hour"], outside["cost_per_request"]) == (None, None, None)


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
    assert result["reads"]["config"] == ["cost.latency_groups", "cost.limit_min_calls", "cost.limit_pool",
                                         "cost.p95_slo_cap_ms", "cost.slo_from", "cost.steady_inflight_share",
                                         "cost.steady_min_requests", "cost.steady_min_share", "cost.utilizations",
                                         "modal.gpu_prices", "modal.gpu_prices_sensitivity",
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
    assert slow["slo"] == {"cap_ms": 1000, "llm_p95_ms": 2000.0, "slo_ms": 1000, "first_attempts": 20,
                           "warmup_excluded": 0, "llm_p95_ms_with_warmup": 2000.0}
    assert slow["engines"]["slm:qwen3-8b"]["sustained"]["concurrency"] == 8
    # the LLM is faster than the cap: its p95 binds, and the SLM must keep up with it
    fast = read_result(j8.run(runs, config, slo_from=pilot([500] * 20, "pilot-fast")), "J8")["result"]
    assert (fast["slo"]["llm_p95_ms"], fast["slo"]["slo_ms"]) == (500.0, 500.0)
    assert fast["engines"]["slm:qwen3-8b"]["sustained"]["concurrency"] == 1
    assert fast["cost_per_request"]["100%"] == pytest.approx(0.8 / (2.0 * 3600))
    # no level answers as fast as the LLM: no sustained throughput and no cost, never a silent fall back to the cap
    faster = read_result(j8.run(runs, config, slo_from=pilot([300] * 20, "pilot-faster")), "J8")["result"]
    assert faster["engines"]["slm:qwen3-8b"]["outside_slo"] is True and faster["cost_per_request"] is None
    assert faster["outside_slo"] == ["slm:qwen3-8b"] and faster["combined"]["outside_slo"] == ["slm:qwen3-8b"]


def test_the_llms_p95_is_over_the_first_attempts_of_a_done_b0_run_on_calib(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(cap_ms=10000))
    runs = [loadtest("loadtest-4", 4, 3000, 6.0), loadtest("loadtest-8", 8, 4000, 10.0)]
    # the SLO is the p95 of the latencies, not their largest, smallest or first: 900 + 0.55 × (5000 − 900)
    spread = read_result(j8.run(runs, config, slo_from=pilot([700, 100, 200, 300, 400, 500, 600, 800, 900, 5000], "pilot-spread")),
                         "J8")["result"]
    assert spread["slo"] == {"cap_ms": 10000, "llm_p95_ms": 3155.0, "slo_ms": 3155.0, "first_attempts": 10,
                             "warmup_excluded": 0, "llm_p95_ms_with_warmup": 3155.0}
    # the sweep's limit is the weighted quantile (inverse distribution): with 10 calls, the 10th, 5000 ms
    assert spread["engines"]["slm:qwen3-8b"]["limit"]["slo_ms"] == 5000.0
    assert spread["engines"]["slm:qwen3-8b"]["sustained"]["concurrency"] == 8  # 4000 ms is within it
    first = call("pilot-retries", "1", "select_tables", latency_ms=1000, parsed_ok=False)
    retry = call("pilot-retries", "1", "select_tables", latency_ms=9000, attempt=2, retry_of=first["call_id"])
    failed = call("pilot-retries", "2", "select_tables", latency_ms=3500, response=None, parsed_ok=False)
    write_run("pilot-retries", {"type": "agent", "arm": "B0", "split": "calib"}, [first, retry, failed], config={})
    measured = read_result(j8.run(runs, config, slo_from="pilot-retries"), "J8")["result"]["slo"]
    # the retry's 9000 ms is not a first attempt; the first attempt that failed is one: 1000 + 0.95 × 2500
    assert (measured["llm_p95_ms"], measured["first_attempts"]) == (3375.0, 2)
    for run_id, manifest, reason in (("pilot-b1", {"arm": "B1"}, "arm is 'B1'"), ("pilot-train", {"split": "train"}, "split is 'train'"),
                                     ("pilot-failed", {"status": "failed"}, "not 'done'"),
                                     ("pilot-replay", {"type": "replay"}, "type is 'replay'")):
        with pytest.raises(JudgmentError, match=reason):
            j8.run(runs, config, slo_from=pilot(run_id=run_id, **manifest))
    write_run("pilot-empty", {"type": "agent", "arm": "B0", "split": "calib"}, [], config={})
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
    assert config["cost"]["p95_slo_cap_ms"] == 17000 and config["loadtest"]["concurrency"] == [1, 2, 4, 8, 16, 32, 64, 128]
    assert (config["cost"]["limit_min_calls"], config["cost"]["limit_pool"]) == (30, {"revise": "generate_candidate"})
    assert config["loadtest"]["cold_prefix_cache"] == "always"
    per_second = j8.container_price(config, config["serving"]["gpu"])  # RunPod's A100 80GB PCIe pod
    assert per_second == pytest.approx({"gpu": 0.000497, "cpu": 0, "memory": 0})  # CPU and RAM bundled in the pod
    assert sum(per_second.values()) * 3600 == pytest.approx(1.7892)  # US$1.79/h, the pod's price
    # the sensitivity: Modal's price for the same GPU, its CPU and memory billed apart
    modal = j8.container_price(config, config["serving"]["gpu"], "gpu_prices_sensitivity")
    assert sum(modal.values()) * 3600 == pytest.approx((0.000694 + 8 * 0.0000131 + 32 * 0.00000222) * 3600)


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
    assert result["combined"] == {"rule": "the highest cost per request among the engines measured, none if one is "
                                          "outside its limit; B4 as deployed apart",
                                  "engines": ["slm:qwen3-8b+lora:c0", "slm:qwen3-8b+lora:c1"], "outside_slo": [],
                                  "loadtests": sorted(runs),
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


# ---------------------------------------------------------------- one limit rule: the teacher reweighted to the sweep

def mixed_pilot(run_id="pilot-mixed"):
    """The teacher's first attempts: 40 select_tables at 1000 ms, 40 generate_candidate at 9000 ms, 2 revise
    at 20000 ms (too few on their own: pooled with generate_candidate)."""
    calls = ([call(run_id, f"s{q}", "select_tables", latency_ms=1000) for q in range(40)]
             + [call(run_id, f"g{q}", "generate_candidate", latency_ms=9000) for q in range(40)]
             + [call(run_id, f"r{q}", "revise", latency_ms=20000) for q in range(2)])
    write_run(run_id, {"type": "agent", "arm": "B0", "split": "calib"}, calls, config={})
    return run_id


def test_the_weighted_quantile_is_the_inverse_of_the_weighted_distribution():
    from fractions import Fraction

    assert j8.weighted_quantile([1, 2, 3], [Fraction(1)] * 3) == 3.0  # 95% of 3 is reached at the third
    assert j8.weighted_quantile([3, 1, 2], [Fraction(1), Fraction(97), Fraction(2)]) == 1.0  # 97% at the first
    assert j8.weighted_quantile([5, 1], [Fraction(1, 20), Fraction(19, 20)]) == 1.0  # exactly 95%: reached


def test_each_sweeps_limit_is_the_teachers_p95_reweighted_to_its_own_call_site_mix(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, {**settings(cap_ms=17000)})
    config["cost"].update(limit_min_calls=30, limit_pool={"revise": "generate_candidate"})
    runs = [loadtest("a-8", 8, 900, 10.0, engine="slm:qwen3-8b+lora:c1", sites={"select_tables": 100}),
            loadtest("b-8", 8, 8000, 5.0, engine="slm:qwen3-8b+lora:c2",
                     sites={"select_tables": 10, "generate_candidate": 80, "revise": 10})]
    result = read_result(j8.run(runs, config, slo_from=mixed_pilot()), "J8")["result"]
    a, b = (result["engines"][e]["limit"] for e in ("slm:qwen3-8b+lora:c1", "slm:qwen3-8b+lora:c2"))
    # all select_tables: the teacher's select_tables alone, 1000 ms
    assert (a["weighted_p95_ms"], a["slo_ms"], a["shares"]) == (1000.0, 1000.0, {"select_tables": 1.0})
    # 10% select_tables, 90% the SQL generation (revise pooled with generate_candidate: 42 calls weigh 0.9)
    assert b["shares"] == {"generate_candidate": 0.9, "select_tables": 0.1} and b["pooled"] == {"revise": "generate_candidate"}
    assert b["n"] == {"generate_candidate": 40, "revise": 2, "select_tables": 40}
    # cumulative weight: 0.1 through the select_tables calls, 0.957 through the generate_candidate ones: 9000 ms
    assert b["weighted_p95_ms"] == 9000.0 and b["slo_ms"] == 9000.0
    # the teacher's own mix (equal weights) is the sensitivity: 82 calls, the 78th, 9000 ms for both sweeps
    assert b["unweighted_p95_ms"] == 9000.0 and a["unweighted_slo_ms"] == 9000.0
    assert result["engines"]["slm:qwen3-8b+lora:c1"]["outside_slo"] is False
    assert result["engines"]["slm:qwen3-8b+lora:c1"]["unweighted_limit"]["outside_slo"] is False
    # a rare call site with no pool is refused, never weighed on 2 calls
    config["cost"]["limit_pool"] = {}
    with pytest.raises(JudgmentError, match="call site revise has 2 teacher calls"):
        j8.run(runs, config, slo_from="pilot-mixed")


def routed_source(run_id="agent-B4-calib", embeds=(100.0,) * 19 + (300.0,), waits=(5000.0,) * 20):
    write_run(run_id, {"type": "agent", "arm": "B4", "split": "calib"}, [],
              files={"routes.jsonl": [{"question_id": str(i), "embed_ms": e, "embed_wait_ms": w}
                                      for i, (e, w) in enumerate(zip(embeds, waits))]})
    return run_id


def test_the_routers_embedding_time_is_added_to_a_routed_sweep_and_never_its_queue(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(cap_ms=5000))
    pilot([1000] * 20)
    source = routed_source()
    runs = [loadtest("c0-1", 1, 850, 2.0, engine="slm:qwen3-8b+lora:c0", kind="adapter", source=source),
            loadtest("c0-8", 8, 950, 10.0, engine="slm:qwen3-8b+lora:c0", kind="adapter", source=source),
            loadtest("base-8", 8, 950, 10.0, engine="slm:qwen3-8b", kind="base")]
    result = read_result(j8.run(runs, config, slo_from=PILOT), "J8")
    adapter, base = result["result"]["engines"]["slm:qwen3-8b+lora:c0"], result["result"]["engines"]["slm:qwen3-8b"]
    # p95 of the embedding: 100 + 0.05 x 200 = 110 ms; the 5000 ms waits for the lock are reported, not added
    routes_sha256 = hashlib.sha256((tmp_path / "runs" / source / "routes.jsonl").read_bytes()).hexdigest()
    assert adapter["router"] == {"run_id": source, "routes_sha256": routes_sha256, "n": 20,
                                 "embed_p95_ms": pytest.approx(110.0), "embed_wait_p95_ms": 5000.0}
    assert adapter["added_ms"] == pytest.approx(110.0) and adapter["sustained"]["concurrency"] == 1  # 950 + 110 > 1000
    assert adapter["without_router"]["sustained"]["concurrency"] == 8  # the sensitivity: without the addition
    assert (base["router"], base["added_ms"], base["sustained"]["concurrency"]) == (None, 0.0, 8)  # B3 is not routed
    # what J8 read of the source: its manifest and the routes.jsonl itself, each hashed
    assert result["reads"]["routes"][source]["run_id"] == source
    assert result["reads"]["routes"][source]["routes.jsonl"] == routes_sha256
    write_run("agent-B4-old", {"type": "agent", "arm": "B4", "split": "calib"}, [])
    loadtest("old-8", 8, 900, 10.0, engine="slm:qwen3-8b+lora:c1", kind="adapter", source="agent-B4-old")
    with pytest.raises(JudgmentError, match="has no routes.jsonl"):
        j8.run(runs + ["old-8"], config, slo_from=PILOT)


def test_b4_as_deployed_is_judged_apart_from_the_combination(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(cap_ms=5000))
    pilot([1000] * 20)
    source = routed_source(embeds=(10.0,) * 20)
    runs = [loadtest("c0-8", 8, 900, 10.0, engine="slm:qwen3-8b+lora:c0", kind="adapter", source=source),
            loadtest("dep-8", 8, 900, 4.0, engine="slm:qwen3-8b+as-deployed", kind="as_deployed", source=source)]
    result = read_result(j8.run(runs, config, slo_from=PILOT), "J8")["result"]
    assert result["engine"] == "slm:qwen3-8b" and result["combined"]["engines"] == ["slm:qwen3-8b+lora:c0"]
    assert result["cost_per_request"]["100%"] == pytest.approx(0.8 / (10.0 * 3600))  # not the as-deployed sweep's
    deployed = result["as_deployed"]["slm:qwen3-8b+as-deployed"]
    assert deployed["added_ms"] == 10.0 and deployed["cost_per_request"]["100%"] == pytest.approx(0.8 / (4.0 * 3600))


def test_one_engine_outside_its_limit_leaves_the_others_judged(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(cap_ms=1000))
    pilot([1000] * 20)
    runs = [loadtest("c0-8", 8, 900, 10.0, engine="slm:qwen3-8b+lora:c0"),
            loadtest("c1-8", 8, 3000, 5.0, engine="slm:qwen3-8b+lora:c1")]
    result = read_result(j8.run(runs, config, slo_from=PILOT), "J8")["result"]
    assert result["engines"]["slm:qwen3-8b+lora:c0"]["cost_per_request"]["100%"] == pytest.approx(0.8 / 36000)
    assert result["outside_slo"] == ["slm:qwen3-8b+lora:c1"] and result["cost_per_request"] is None


def test_every_cost_is_also_given_at_the_sensitivity_price(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(gpu_per_hour=1.79))
    config["serving"].update(cpu=8, memory_gib=32)
    config["modal"]["gpu_prices_sensitivity"] = {"as_of": "2026-11-09", "provider": "modal",
                                                 "usd_per_s": {"L4": 0.000694}, "cpu_usd_per_core_s": 0.0000131,
                                                 "memory_usd_per_gib_s": 0.00000222}
    result = read_result(j8.run([loadtest("lt-8", 8, 900, 10.0)], config, slo_from=pilot()), "J8")["result"]
    modal_hour = (0.000694 + 8 * 0.0000131 + 32 * 0.00000222) * 3600
    assert result["sensitivity"]["price_per_hour"] == pytest.approx(modal_hour) and result["sensitivity"]["provider"] == "modal"
    assert result["cost_per_request_sensitivity"]["20%"] == pytest.approx(modal_hour / (10.0 * 3600 * 0.2))
    assert result["cost_per_request"]["20%"] == pytest.approx(1.79 / (10.0 * 3600 * 0.2))
    del config["modal"]["gpu_prices_sensitivity"]
    plain = read_result(j8.run(["lt-8"], config, slo_from=PILOT), "J8")["result"]
    assert "sensitivity" not in plain and "cost_per_request_sensitivity" not in plain["engines"]["slm:qwen3-8b"]


# ---------------------------------------------------------------- the steady state of a level

def span(start_ms, end_ms, n=0, **extra):
    return {"metadata": {"session_num": n, "request_start_ns": int(start_ms * 1e6), "request_end_ns": int(end_ms * 1e6),
                         "benchmark_phase": "profiling", "was_cancelled": False}, **extra}


def ramped(concurrency=4, per_slot=60, latency=100, stagger=1000):
    """`concurrency` slots that start `stagger` ms apart (the ramp-up), each `per_slot` requests back to back, so
    they also end `stagger` ms apart (the drain)."""
    rows = []
    for slot in range(concurrency):
        t = slot * stagger
        for _ in range(per_slot):
            rows.append(span(t, t + latency, len(rows)))
            t += latency
    return rows


def test_the_steady_window_leaves_out_the_ramp_up_and_the_drain():
    # 4 slots at 100 ms, starting 0, 1, 2, 3 s apart, 60 requests each (6 s): 4 in flight from 3 s to 6 s
    rows = ramped()
    window = j8.steady(rows, 4, 0.9, 10, 0.3)
    assert (window["window_start_ns"], window["window_end_ns"]) == (3_000_000_000, 6_000_000_000)
    assert window["window_s"] == 3.0 and window["window_share"] == pytest.approx(3 / 9)
    # 4 slots × 30 requests end in (3 s, 6 s]; the whole ones are the same 120, at 100 ms
    assert window["completed_in_window"] == 120 and window["throughput_rps"] == pytest.approx(40.0)
    assert window["whole_requests"] == 120 and window["p95_ms"] == pytest.approx(100.0)
    assert window["measured"] is True and window["reasons"] == []
    # the rule's two floors: whole requests and the share of the level
    assert j8.steady(rows, 4, 0.9, 121, 0.3)["reasons"] == [
        "120 whole requests in the steady window (< cost.steady_min_requests 121)"]
    assert j8.steady(rows, 4, 0.9, 10, 0.5)["reasons"] == [
        "the steady window lasts 0.333 of the level (< cost.steady_min_share 0.5)"]
    # a request that straddles an edge is a completion inside but not a whole request
    straddle = [span(0, 10_000, 0)] + [span(1000 + 100 * i, 1100 + 100 * i, i + 1) for i in range(50)]
    one = j8.steady(straddle, 2, 0.9, 1, 0.0)
    assert (one["window_start_ns"], one["window_end_ns"]) == (1_000_000_000, 6_000_000_000)
    assert one["whole_requests"] == 50 and one["completed_in_window"] == 50
    # in flight never reaches 90% of the level: not measured
    assert j8.steady(rows, 8, 0.9, 1, 0.0)["reasons"] == ["the requests in flight never reached 0.9 x 8"]
    assert j8.steady([], 4, 0.9, 1, 0.0)["reasons"] == ["no request record"]


def test_back_to_back_requests_never_dip_out_of_the_window():
    rows = [span(100 * i, 100 * (i + 1), i) for i in range(200)]  # concurrency 1, each starts as the last ends
    window = j8.steady(rows, 1, 0.9, 100, 0.5)
    assert (window["window_s"], window["whole_requests"], window["measured"]) == (20.0, 200, True)


def test_an_unmeasured_level_counts_neither_for_nor_against(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(cap_ms=1000))
    pilot([1000] * 20)
    short = records(12, 500, 30.0, per_slot=8)  # 12 in flight throughout, but only 96 requests in all
    runs = [loadtest("lt-1", 1, 400, 2.0), loadtest("lt-8", 8, 900, 4.0),
            loadtest("lt-12", 12, 500, 30.0, rows=short)]  # the best on AIPerf's aggregates, but not measured
    result = read_result(j8.run(runs, config, slo_from=PILOT), "J8")["result"]
    engine = result["engines"]["slm:qwen3-8b"]
    assert engine["sustained"]["concurrency"] == 8 and engine["sustained"]["throughput_rps"] == pytest.approx(4.0)
    assert engine["unmeasured"] == [{"run_id": "lt-12", "concurrency": 12, "reasons": [
        "96 whole requests in the steady window (< cost.steady_min_requests 100)"]}]
    assert result["unmeasured"] == {"slm:qwen3-8b": engine["unmeasured"]}
    # the sensitivity on AIPerf's whole-level aggregates takes the level the steady state leaves out
    assert engine["aiperf_aggregates"]["sustained"]["concurrency"] == 12
    level8 = next(lv for lv in engine["levels"] if lv["concurrency"] == 8)
    assert level8["aiperf"] == {"p95_ms": 900.0, "throughput_rps": 4.0} and level8["steady"]["whole_requests"] == 800
    assert result["steady_rule"] == {"steady_inflight_share": 0.9, "steady_min_requests": 100, "steady_min_share": 0.5}
    # an engine with no measured level is refused, never judged on nothing
    with pytest.raises(JudgmentError, match="no measured level"):
        j8.run(["lt-12"], config, slo_from=PILOT)


def test_a_level_with_any_failure_is_not_measured(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(cap_ms=1000))
    pilot([1000] * 20)
    failed_row = records(8, 900, 4.0)
    failed_row[5]["error"] = {"code": 500, "type": "server", "message": "boom"}
    runs = [loadtest("ok-1", 1, 400, 2.0),
            loadtest("err-8", 8, 900, 4.0, rows=failed_row, error_summary=[{"error_details": {"code": 500}, "count": 1}]),
            loadtest("cancel-4", 4, 900, 3.0, was_cancelled=True),
            loadtest("partial-2", 2, 900, 2.1, is_complete=False)]
    engine = read_result(j8.run(runs, config, slo_from=PILOT), "J8")["result"]["engines"]["slm:qwen3-8b"]
    assert engine["sustained"]["run_id"] == "ok-1"
    reasons = {lv["run_id"]: lv["reasons"] for lv in engine["unmeasured"]}
    assert reasons == {"partial-2": ["AIPerf's run is not complete"], "cancel-4": ["AIPerf was cancelled"],
                       "err-8": ["AIPerf's error_summary lists 1 error type(s)", "1 request(s) failed"]}
    err = next(lv for lv in engine["levels"] if lv["run_id"] == "err-8")
    assert err["failures"]["records_with_error"] == 1 and err["failures"]["none"] is False
    assert engine["aiperf_aggregates"]["sustained"]["run_id"] == "ok-1"  # failures are out of the sensitivity too


def test_a_level_needs_its_per_request_records(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings())
    pilot()
    write_run("lt-old", {"type": "loadtest", "engine": "slm:qwen3-8b", "gpu": "L4", "concurrency": 1, "prefix_cache": True,
                         "sweep_id": "s", "composition": {"source": {"call_sites": {"select_tables": 1}}}},
              files={j8.EXPORT: export(400, 2.0)})
    with pytest.raises(JudgmentError, match="has no profile_export.jsonl"):
        j8.run(["lt-old"], config, slo_from=PILOT)
    del config["cost"]["steady_min_requests"]
    with pytest.raises(JudgmentError, match="cost.steady_min_requests not set"):
        j8.run([loadtest("lt-1", 1, 400, 2.0)], config, slo_from=PILOT)


def test_the_shipped_steady_rule_and_groups():
    from bench.contracts.calls import CALL_SITES
    from bench.contracts.config import load_config
    from bench import paths
    cost = load_config(paths.ROOT / "config.yaml")["cost"]
    assert (cost["steady_inflight_share"], cost["steady_min_requests"], cost["steady_min_share"]) == (0.9, 100, 0.5)
    grouped = [site for members in cost["latency_groups"].values() for site in members]
    assert sorted(grouped) == sorted(set(grouped)) and set(grouped) <= set(CALL_SITES)


# ---------------------------------------------------------------- the teacher's warm-up gate

BREAK = "Column information:\n"
GATED = {"roles": {"production_llm": {"params": {"cache_breaks": [BREAK]}}}}


def gated_call(run_id, question, at_ms, latency_ms, site="filter_column", prefix="schema A", session=None, attempt=1):
    found = call(session or run_id, question, site, f"k{at_ms}", latency_ms=latency_ms, attempt=attempt,
                 messages=[{"role": "user", "content": f"{prefix}\n{BREAK}column {at_ms}"}])
    seconds = 12 * 3600 + at_ms / 1000
    found["started_at"] = f"2026-09-30T{int(seconds // 3600):02d}:{int(seconds % 3600 // 60):02d}:{seconds % 60:09.6f}+00:00"
    return found


def test_the_cache_prefix_is_the_engines_own_key():
    messages = [{"role": "user", "content": f"intro\n{BREAK}rest"}]
    assert j8.cache_prefix(messages, [BREAK]) == f"None|[]intro\n{BREAK}"
    assert j8.cache_prefix([{"role": "user", "content": f"intro\n{BREAK}"}], [BREAK]) is None  # nothing after it
    assert j8.cache_prefix([{"role": "user", "content": "no marker"}], [BREAK]) is None
    pytest.importorskip("anthropic")
    pytest.importorskip("langchain_core")
    import hashlib as h
    from langchain_core.messages import HumanMessage, SystemMessage
    from bench.agent.anthropic_engine import AnthropicChat
    client = AnthropicChat.__new__(AnthropicChat)
    client.model, client.max_tokens, client.model_kwargs, client.extra_body = "m", 10, {}, None
    client.cache_breaks, client.cache_ttl = [BREAK, "Question: "], "1h"
    for lc, recorded in (([HumanMessage(content=f"a\n{BREAK}b Question: c")], [{"role": "user", "content": f"a\n{BREAK}b Question: c"}]),
                         ([SystemMessage(content="sys"), HumanMessage(content="x Question: y")],
                          [{"role": "system", "content": "sys"}, {"role": "user", "content": "x Question: y"}])):
        _, engine_key = client.request(lc)
        key = j8.cache_prefix(recorded, client.cache_breaks)
        system, prefix = key.split("|", 1)
        assert engine_key == h.sha256(f"{system}|{h.sha256(prefix.encode()).hexdigest()}".encode()).hexdigest()


def test_the_calls_the_warm_up_gate_held_are_left_out_of_the_teachers_latency(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(cap_ms=17000))
    run_id = "pilot-gated"
    # worker 0 takes questions 1 and 3, worker 1 question 2 (worker_parts over [1, 2, 3], 2 workers)
    calls = [gated_call(run_id, "1", 0, 1000),            # worker 0's leader of prefix A: ends at 1000
             gated_call(run_id, "1", 2, 1900),            # started while it ran: held (ends at 1902)
             gated_call(run_id, "1", 5, 300, site="select_tables", prefix="other"),  # another prefix: its own leader
             gated_call(run_id, "1", 1500, 400),          # after the leader ended: not held
             gated_call(run_id, "3", 600, 2000),          # still worker 0, prefix A, during the leader: held
             gated_call(run_id, "2", 10, 800),            # worker 1: its own leader of prefix A
             gated_call(run_id, "2", 20, 1600)]           # held by worker 1's leader
    write_run(run_id, {"type": "agent", "arm": "B0", "split": "calib", "workers": 2, "question_ids": ["1", "2", "3"]},
              calls, config=GATED)
    teacher = j8.teacher_calls(run_id)
    assert teacher["warmup"]["excluded"] == {"filter_column": 3} and teacher["warmup"]["groups"] == 3
    assert sorted(c["latency_ms"] for c in teacher["kept"]) == [300, 400, 800, 1000]
    runs = [loadtest("lt-1", 1, 900, 1.0, sites={"filter_column": 100})]
    result = read_result(j8.run(runs, config, slo_from=run_id), "J8")["result"]
    limit = result["engines"]["slm:qwen3-8b"]["limit"]
    assert limit["slo_ms"] == 1000.0 and limit["warmup"]["slo_ms_with_warmup"] == 2000.0
    assert result["engines"]["slm:qwen3-8b"]["with_warmup_limit"]["slo_ms"] == 2000.0
    assert result["slo"]["warmup_excluded"] == 3
    # the leader is the call that ends first, not the first stamped: a call stamped a hair earlier waited too
    early = [gated_call("p2", "1", 0, 2600), gated_call("p2", "1", 1, 1000), gated_call("p2", "1", 2, 2500)]
    write_run("p2", {"type": "agent", "arm": "B0", "split": "calib", "workers": 1, "question_ids": ["1"]}, early, config=GATED)
    assert sorted(c["latency_ms"] for c in j8.teacher_calls("p2")["kept"]) == [1000]


def test_a_resumed_runs_processes_are_its_sessions_workers(tmp_path, monkeypatch):
    repo(tmp_path, monkeypatch, settings())
    old, new = "agent-old", "pilot-resumed"
    # the first session asked 1, 2, 3, 4 on 2 workers (0: 1, 3; 1: 2, 4) and kept 1 and 2; the second asked 3 and 4
    calls = [gated_call(new, "1", 0, 1000, session=old), gated_call(new, "2", 0, 1000, session=old),
             gated_call(new, "3", 5000, 1000), gated_call(new, "4", 5100, 1000)]
    found = {"run_id": new, "workers": 2, "question_ids": ["1", "2", "3", "4"], "resumed_question_ids": ["1", "2"],
             "resumed_from": {"run_id": old, "workers": 2}, "sessions": [{}, {}]}
    assert j8.processes(calls, found) == [(old, 0), (old, 1), (new, 0), (new, 1)]
    # one process in two blocks: the reconstruction does not match the runner's merge, refused
    with pytest.raises(JudgmentError, match="not one block"):
        j8.processes([calls[0], calls[1], gated_call(new, "1", 9, 9, session=old)], found)
    with pytest.raises(JudgmentError, match="resumed more than once"):
        j8.processes(calls, {**found, "sessions": [{}, {}, {}]})
    # the gate holds each process apart: worker 1's call at 5100 was not held by worker 0's leader
    assert j8.warmup_gated(calls, found, GATED)["call_ids"] == set()
    assert j8.warmup_gated(calls, {**found, "via_claude_code": True}, GATED)["call_ids"] == set()


def test_the_pilot_needs_its_config_for_the_gate(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings())
    write_run("pilot-noconf", {"type": "agent", "arm": "B0", "split": "calib"}, [call("pilot-noconf", "1", "select_tables")])
    with pytest.raises(JudgmentError, match="has no config.json"):
        j8.run([loadtest("lt-1", 1, 400, 2.0)], config, slo_from="pilot-noconf")


# ---------------------------------------------------------------- p95 per group of call sites, and Modal's CPU

def test_each_group_of_call_sites_is_set_beside_the_teachers(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(cap_ms=17000))
    config["cost"]["latency_groups"] = {"sql": ["generate_candidate"], "selection": ["select_tables"]}
    write_run("pilot-groups", {"type": "agent", "arm": "B0", "split": "calib"},
              [call("pilot-groups", f"s{q}", "select_tables", latency_ms=1000) for q in range(10)]
              + [call("pilot-groups", f"g{q}", "generate_candidate", latency_ms=8000) for q in range(10)], config={})
    source = [call("agent-B3-calib", str(q), "select_tables" if q % 2 else "generate_candidate", engine="slm:qwen3-8b",
                   role="slm") for q in range(200)]
    write_run("agent-B3-calib", {"type": "agent", "arm": "B3", "split": "calib"}, source)
    rows = records(2, 900, 2.5)  # 200 requests; even session_num: generate_candidate
    payloads = [{"model": "m", "messages": c["prompt_messages"]} for c in source]
    loadtest("lt-2", 2, 900, 2.5, source="agent-B3-calib", rows=rows,
             sites={"select_tables": 100, "generate_candidate": 100})
    (tmp_path / "runs" / "lt-2" / j8.PAYLOADS).write_text("".join(json.dumps(p) + "\n" for p in payloads))
    result = read_result(j8.run(["lt-2"], config, slo_from="pilot-groups"), "J8")
    groups = result["result"]["engines"]["slm:qwen3-8b"]["by_group"]
    assert groups["teacher"] == {"selection": {"p95_ms": 1000.0, "n": 10}, "sql": {"p95_ms": 8000.0, "n": 10}}
    level = groups["levels"]["2"]
    assert set(level) == {"selection", "sql"} and level["sql"]["n"] + level["selection"]["n"] == 200
    assert result["reads"]["sources"]["agent-B3-calib"]["run_id"] == "agent-B3-calib"
    # without payloads the group reading is absent, never a guess
    loadtest("lt-4", 4, 900, 4.0, source="agent-B3-calib", sites={"select_tables": 100, "generate_candidate": 100},
             sweep="other")
    other = read_result(j8.run(["lt-4"], config, slo_from="pilot-groups"), "J8")["result"]
    assert other["engines"]["slm:qwen3-8b"]["by_group"]["levels"] == {"4": None}


def test_modals_cpu_is_the_higher_of_the_request_and_the_measured_use(tmp_path, monkeypatch):
    _, config = repo(tmp_path, monkeypatch, settings(gpu_per_hour=1.79))
    config["serving"].update(cpu=8, memory_gib=32)
    config["modal"]["gpu_prices_sensitivity"] = {"as_of": "2026-11-09", "provider": "modal",
                                                 "usd_per_s": {"L4": 0.000694}, "cpu_usd_per_core_s": 0.0000131,
                                                 "memory_usd_per_gib_s": 0.00000222}
    pilot()
    for name, cores in (("lt-light", 0.5), ("lt-heavy", 12.0)):
        loadtest(name, 8, 900, 4.0, sweep=name)
        found = json.loads((tmp_path / "runs" / name / "manifest.json").read_text())
        found["cpu_sample"] = {"server": {"cores": cores}}
        (tmp_path / "runs" / name / "manifest.json").write_text(json.dumps(found))
    light = read_result(j8.run(["lt-light"], config, slo_from=PILOT), "J8")["result"]["sensitivity"]
    assert light["cpu_cores"]["priced"] == 8 and light["cpu_cores"]["measured_max"] == 0.5
    assert light["price_per_hour"] == pytest.approx((0.000694 + 8 * 0.0000131 + 32 * 0.00000222) * 3600)
    heavy = read_result(j8.run(["lt-heavy"], config, slo_from=PILOT), "J8")["result"]["sensitivity"]
    assert heavy["cpu_cores"]["priced"] == 12.0
    assert heavy["price_per_hour"] == pytest.approx((0.000694 + 12 * 0.0000131 + 32 * 0.00000222) * 3600)
