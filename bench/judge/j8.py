"""J8 · load (SPEC §6.6): the SLM's sustained throughput, and its cost per request and per
question at 20, 50 and 100% utilization.

Reads `loadtest` executions (F3): per load level, AIPerf v0.13.0's `profile_export_aiperf.json` as
it comes out, and a manifest naming the engine, the GPU and the concurrency. From the export it
reads two blocks (AIPerf's JSON export schema): `request_latency` (unit `ms`, with `p95`) and
`request_throughput` (unit `requests/sec`, with `avg`); any other unit is refused.

**Sustained throughput within the p95** is read as: the highest request throughput among the load
levels whose p95 request latency is within the SLO. No level within it is an error, not a number.
Every level must have run with the prefix cache on (SPEC §6.6).

**The SLO is a rule, not a number chosen after a load test**: the stricter of
`cost.p95_slo_cap_ms`, an external reference, and the production LLM's own p95 latency per call,
over the first attempts of the pilot's B0 execution on calib (`slo_from`). That execution is fixed
in the configuration before the load test (`cost.slo_from`, registered with it): once set, J8
refuses any other. The SLM must answer at least as fast as the API it replaces, and never slower than the
reference. Every first attempt counts, a failed one too: it is the latency the agent met. That p95
is the 95th percentile by linear interpolation between order statistics, the definition AIPerf's
`request_latency.p95` follows on the SLM's side. The result records the cap, the LLM's p95 and
the SLO used (`slo`).

**The price is the serving container's, all-in**: Modal bills the GPU, the CPU and the memory, so
the price per second is the GPU's (F3's dated `modal.gpu_prices.usd_per_s`, keyed by the
manifest's `gpu`) plus `serving.cpu` cores at `cpu_usd_per_core_s` plus `serving.memory_gib` GiB at
`memory_usd_per_gib_s`. Then:

    cost per request at utilization u = price per hour / (sustained requests per hour × u)

J8 gives the cost **per request**. The load test replays the agent's real calls, so a request is
one SLM call: J3 prices each SLM call of an execution at this cost, which makes the cost per question
the execution's SLM calls per question times it. That holds as far as the load test's mix of calls
is the execution's (F3 replays a source execution's calls; the manifest records which).

**One sweep per engine**: F3 runs an engine's load levels as one sweep (`sweep_id`), in order on one
warm server. J8 takes exactly one sweep of each engine: an engine with one sweep is always kept;
several sweeps of one engine are refused unless the invocation names exactly one of them
(`sweeps`); naming sweeps only chooses among an engine's own, it never drops an engine.

**One engine per load test** (F3): the base (`slm:<candidate>`, B3's traffic) or one adapter
(`slm:<candidate>+lora:<name>`). J8 takes the load tests of one or more engines of **one base on one
GPU**, judges each engine apart, and combines them by one stated rule: the highest cost per request
among the engines measured (the conservative one). B4 and B5 spread their calls over several adapters
on one server, which no single-engine load test measures, so their SLM cost from J8 is an
**extrapolation**; the result records the engines and load tests it combined, and J3 labels it.
"""
import hashlib
import json
from fractions import Fraction
from typing import Any, Dict, List, Optional

from bench.judge.base import JudgmentError, calls_of, canonical, reference, require_done, run_dir, write_result

JUDGMENT = "J8"
EXPORT = "profile_export_aiperf.json"
# the configuration keys J8 reads, recorded in the result's `reads` (design §6.2)
CONFIG_KEYS = ("cost.p95_slo_cap_ms", "cost.slo_from", "cost.utilizations", "modal.gpu_prices", "serving.cpu",
               "serving.memory_gib")


def level(export: Dict[str, Any]) -> Dict[str, float]:
    """(p95 latency in ms, throughput in requests/s) of one AIPerf export."""
    latency, throughput = export.get("request_latency") or {}, export.get("request_throughput") or {}
    if latency.get("unit") != "ms" or throughput.get("unit") != "requests/sec":
        raise JudgmentError(f"unexpected AIPerf units: request_latency in {latency.get('unit')!r}, "
                            f"request_throughput in {throughput.get('unit')!r}")
    if latency.get("p95") is None or throughput.get("avg") is None:
        raise JudgmentError("the AIPerf export has no request_latency.p95 or request_throughput.avg")
    return {"p95_ms": float(latency["p95"]), "throughput_rps": float(throughput["avg"])}


def judge(levels: List[Dict[str, Any]], price_per_hour: float, slo_ms: float, utilizations: List[float]) -> Dict[str, Any]:
    """`levels`: [{"concurrency", "p95_ms", "throughput_rps", "run_id"}], one engine on one GPU."""
    within = [lv for lv in levels if lv["p95_ms"] <= slo_ms]
    if not within:
        raise JudgmentError(f"no load level keeps p95 within {slo_ms} ms: the SLM has no sustained throughput")
    best = max(within, key=lambda lv: (lv["throughput_rps"], -lv["concurrency"]))
    per_hour = best["throughput_rps"] * 3600
    return {"levels": sorted(levels, key=lambda lv: lv["concurrency"]), "p95_slo_ms": slo_ms,
            "sustained": best, "requests_per_hour": per_hour, "price_per_hour": price_per_hour,
            "cost_per_request": {f"{round(u * 100)}%": price_per_hour / (per_hour * u) for u in utilizations}}


def p95(values: List[float]) -> float:
    """The 95th percentile by linear interpolation between order statistics (numpy's default), the
    rank computed in integers."""
    ordered = sorted(values)
    rank = Fraction(95 * (len(ordered) - 1), 100)
    low = rank.numerator // rank.denominator
    if low == len(ordered) - 1:
        return float(ordered[low])
    return float(ordered[low] + (rank - low) * (ordered[low + 1] - ordered[low]))


def slo(slo_from: str, cap_ms: float) -> Dict[str, Any]:
    """The SLO rule: the stricter of the cap and the production LLM's p95 latency per call, over the
    first attempts of the pilot's B0 execution on calib."""
    require_done(slo_from, type="agent", arm="B0", split="calib")
    first = [call["latency_ms"] for call in calls_of(slo_from) if call["attempt"] == 1]
    if not first:
        raise JudgmentError(f"{slo_from} has no call: the production LLM's p95 latency cannot be measured")
    measured = p95(first)
    return {"cap_ms": cap_ms, "llm_p95_ms": measured, "slo_ms": min(cap_ms, measured), "first_attempts": len(first)}


def container_price(config: Dict[str, Any], gpu: str) -> Dict[str, float]:
    """USD per second of one serving container, by what Modal bills: the GPU, the CPU cores and the memory."""
    prices, serving = (config.get("modal") or {}).get("gpu_prices") or {}, config.get("serving") or {}
    per_gpu = (prices.get("usd_per_s") or {}).get(gpu)
    if per_gpu is None or not prices.get("as_of"):
        raise JudgmentError(f"modal.gpu_prices (F3) has no dated price for {gpu!r}")
    missing = [key for key, value in (("modal.gpu_prices.cpu_usd_per_core_s", prices.get("cpu_usd_per_core_s")),
                                      ("modal.gpu_prices.memory_usd_per_gib_s", prices.get("memory_usd_per_gib_s")),
                                      ("serving.cpu", serving.get("cpu")), ("serving.memory_gib", serving.get("memory_gib")))
               if value is None]
    if missing:
        raise JudgmentError(f"{', '.join(missing)} not set: the container's price counts its CPU and memory, not the GPU alone")
    return {"gpu": per_gpu, "cpu": serving["cpu"] * prices["cpu_usd_per_core_s"],
            "memory": serving["memory_gib"] * prices["memory_usd_per_gib_s"]}


def run(loadtest_run_ids: List[str], config: Dict[str, Any], sweeps: Optional[List[str]] = None, *, slo_from: str):
    cost = config["cost"]
    if cost.get("p95_slo_cap_ms") is None:
        raise JudgmentError("cost.p95_slo_cap_ms is not set: the SLO needs its pre-registered cap")
    if cost.get("slo_from") not in (None, slo_from):
        raise JudgmentError(f"cost.slo_from names {cost['slo_from']}, not {slo_from}: the SLO's execution is fixed in "
                            f"the configuration, before the load test")
    bound = slo(slo_from, cost["p95_slo_cap_ms"])
    levels = []
    for run_id in sorted(loadtest_run_ids):
        found = require_done(run_id, type="loadtest")
        if found.get("prefix_cache") is not True:
            raise JudgmentError(f"{run_id} did not run with the prefix cache on (SPEC §6.6)")
        export = json.loads((run_dir(run_id) / EXPORT).read_text())
        if not found.get("sweep_id"):
            raise JudgmentError(f"{run_id} records no sweep_id: J8 takes one sweep per engine")
        levels.append({"run_id": run_id, "engine": found["engine"], "concurrency": found["concurrency"],
                       "sweep_id": found["sweep_id"], "source_run_id": found.get("source_run_id"), **level(export)})
    unknown = sorted(set(sweeps or []) - {lv["sweep_id"] for lv in levels})
    if unknown:
        raise JudgmentError(f"no load test has the named sweep(s) {unknown}")
    kept = []
    for engine in sorted({lv["engine"] for lv in levels}):
        found_sweeps = sorted({lv["sweep_id"] for lv in levels if lv["engine"] == engine})
        if len(found_sweeps) > 1:
            named = [sweep for sweep in found_sweeps if sweep in (sweeps or [])]
            if len(named) != 1:
                raise JudgmentError(f"{engine} has several sweeps {found_sweeps}: name exactly one of them")
            found_sweeps = named
        kept += [lv for lv in levels if lv["engine"] == engine and lv["sweep_id"] == found_sweeps[0]]
    levels = kept
    reads = {lv["run_id"]: reference(lv["run_id"]) for lv in levels}
    gpus = {require_done(lv["run_id"])["gpu"] for lv in levels}
    engines = sorted({lv["engine"] for lv in levels})
    bases = {e.split("+lora:")[0] for e in engines}
    if len(bases) != 1 or len(gpus) != 1:
        raise JudgmentError(f"the load tests must be engines of one base on one GPU: {engines} on {sorted(gpus)}")
    gpu = gpus.pop()
    per_second = container_price(config, gpu)
    per_hour = sum(per_second.values()) * 3600
    prices = config["modal"]["gpu_prices"]
    per_engine = {engine: judge([lv for lv in levels if lv["engine"] == engine], per_hour,
                                bound["slo_ms"], cost["utilizations"]) for engine in engines}
    combined = {u: max(result["cost_per_request"][u] for result in per_engine.values())
                for u in next(iter(per_engine.values()))["cost_per_request"]}
    result = {"engine": bases.pop(), "gpu": gpu, "gpu_prices_as_of": prices["as_of"], "price_per_hour": per_hour,
              "price_per_second": per_second, "gpu_prices_sha256": hashlib.sha256(canonical(prices)).hexdigest(),
              "slo": bound,
              "source_run_ids": sorted({lv.get("source_run_id") for lv in levels if lv.get("source_run_id")}),
              "engines": per_engine, "cost_per_request": combined,
              "combined": {"rule": "the highest cost per request among the engines measured", "engines": engines,
                           "loadtests": sorted(reads), "sweeps": sorted({lv["sweep_id"] for lv in levels})}}
    return write_result(JUDGMENT, {"loadtests": reads, "slo_from": reference(slo_from), "config": list(CONFIG_KEYS)}, result)
