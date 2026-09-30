"""J8 · load (SPEC §6.6): the SLM's sustained throughput, and its cost per request and per
question at 20, 50 and 100% utilization.

Reads `loadtest` executions (F3): per load level, AIPerf v0.13.0's `profile_export_aiperf.json` as
it comes out, and a manifest naming the engine, the GPU and the concurrency. From the export it
reads two blocks (AIPerf's JSON export schema): `request_latency` (unit `ms`, with `p95`) and
`request_throughput` (unit `requests/sec`, with `avg`); any other unit is refused.

**Sustained throughput within the p95** is read as: the highest request throughput among the load
levels whose p95 request latency is within `cost.p95_slo_ms`, a pre-registered bound. No level
within it is an error, not a number. Every level must have run with the prefix cache on (SPEC §6.6).
Then, with the GPU's price (F3's dated `modal.gpu_prices.usd_per_s`, keyed by the manifest's `gpu`):

    cost per request at utilization u = price per hour / (sustained requests per hour × u)

J8 gives the cost **per request**. The load test replays the agent's real calls, so a request is
one SLM call: J3 prices each SLM call of an execution at this cost, which makes the cost per question
the execution's SLM calls per question times it. That holds as far as the load test's mix of calls
is the execution's (F3 replays a source execution's calls; the manifest records which).
"""
import json
from typing import Any, Dict, List

from bench.judge.base import JudgmentError, reference, require_done, run_dir, write_result

JUDGMENT = "J8"
EXPORT = "profile_export_aiperf.json"


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


def run(loadtest_run_ids: List[str], config: Dict[str, Any]):
    cost = config["cost"]
    if cost.get("p95_slo_ms") is None:
        raise JudgmentError("cost.p95_slo_ms is not set: sustained throughput needs its pre-registered bound")
    levels, engines, gpus, reads = [], set(), set(), {}
    for run_id in sorted(loadtest_run_ids):
        found = require_done(run_id, type="loadtest")
        if found.get("prefix_cache") is not True:
            raise JudgmentError(f"{run_id} did not run with the prefix cache on (SPEC §6.6)")
        engines.add(found["engine"])
        gpus.add(found["gpu"])
        export = json.loads((run_dir(run_id) / EXPORT).read_text())
        levels.append({"run_id": run_id, "concurrency": found["concurrency"], "source_run_id": found.get("source_run_id"),
                       **level(export)})
        reads[run_id] = reference(run_id)
    if len(engines) != 1 or len(gpus) != 1:
        raise JudgmentError(f"the load levels must be one engine on one GPU: {sorted(engines)} on {sorted(gpus)}")
    gpu = gpus.pop()
    prices = (config.get("modal") or {}).get("gpu_prices") or {}
    per_second = (prices.get("usd_per_s") or {}).get(gpu)
    if per_second is None or not prices.get("as_of"):
        raise JudgmentError(f"modal.gpu_prices (F3) has no dated price for {gpu!r}")
    result = {"engine": engines.pop(), "gpu": gpu, "gpu_prices_as_of": prices["as_of"],
              "source_run_ids": sorted({lv.get("source_run_id") for lv in levels if lv.get("source_run_id")}),
              **judge(levels, per_second * 3600, cost["p95_slo_ms"], cost["utilizations"])}
    return write_result(JUDGMENT, {"loadtests": reads}, result)
