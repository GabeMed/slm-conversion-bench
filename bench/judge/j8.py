"""J8 · load (SPEC §6.6): the SLM's sustained throughput, and its cost per request and per
question at 20, 50 and 100% utilization.

Reads `loadtest` executions (F3): per load level, AIPerf v0.13.0's `profile_export_aiperf.json` as
it comes out, and a manifest naming the engine, the GPU and the concurrency. From the export it
reads two blocks (AIPerf's JSON export schema): `request_latency` (unit `ms`, with `p95`) and
`request_throughput` (unit `requests/sec`, with `avg`); any other unit is refused.

**Sustained throughput within the limit** is read as: the highest request throughput among the load
levels whose p95 request latency, plus the router's time where the router feeds the sweep (below), is within
the sweep's limit. **No level within it is a result, not an error**: the engine is `outside_slo`, with no
sustained throughput and no cost (the SLM does not keep up with the API it replaces on that traffic); the
other engines are judged as usual. Every level must have run with the prefix cache on (SPEC §6.6).

**The limit is one rule for every sweep, fixed before any load test**: the stricter of
`cost.p95_slo_cap_ms`, an external reference, and the production LLM's own p95 latency per call **reweighted
to the sweep's call-site mix**, over the first attempts of the pilot's B0 execution on calib (`slo_from`). The
mix is the sweep's pool as F3 recorded it (`composition.source.call_sites`: the base's is B3's, an adapter's its
cluster's calls of B4, as deployed all of B4's). Each teacher call of call site s weighs share(s) / n(s), where
share(s) is s's share of the sweep's calls and n(s) the teacher's first attempts at s, so each call site counts
as much as it does in the sweep. **A call site with fewer than `cost.limit_min_calls` teacher calls is pooled**
with the call site `cost.limit_pool` names for it (revise with generate_candidate: the same SQL generation, 5
calls on the pilot), their shares and calls added; one without such a pool is refused. The weighted p95 is the
inverse of the weighted distribution: the smallest latency at which the cumulative weight reaches 95%, computed
in exact fractions, so it is deterministic. The result records, per engine, the shares, the n per call site, the
pools, the weighted p95 and the limit, and the teacher's own-mix p95 by the same quantile with equal weights
(`limit.unweighted_*`, a sensitivity). That execution is fixed in the configuration before the load test
(`cost.slo_from`, registered with it): once set, J8 refuses any other. Every first attempt counts, a failed one
too: it is the latency the agent met. `slo` keeps the mix's p95 by linear interpolation (the definition AIPerf's
`request_latency.p95` follows), as before, for reference.

**The router's time is added** to each level's p95 for a sweep the router feeds (an adapter, B4 as deployed;
B5's engines are B4's adapters): the p95 of the embedding time (`embed_ms`) over B4's end-to-end calibration
run (the sweep's source, its `routes.jsonl`), not the wait for the embedding model's lock, an artifact of one
model per worker process. B5 routes every call with the same router whatever the allocation, so the time does
not depend on J7. The base (B3) is not routed and adds nothing. The cost without the addition is a sensitivity
(`without_router`).

**The price is the serving container's, all-in**: Modal bills the GPU, the CPU and the memory, so
the price per second is the GPU's (F3's dated `modal.gpu_prices.usd_per_s`, keyed by the
manifest's `gpu`) plus `serving.cpu` cores at `cpu_usd_per_core_s` plus `serving.memory_gib` GiB at
`memory_usd_per_gib_s`. Then:

    cost per request at utilization u = price per hour / (sustained requests per hour × u)

**The provider's price, and Modal's as sensitivity**: where `modal.gpu_prices_sensitivity` is set (Modal's
price for the same GPU, CPU and memory, dated), every cost is also given at that price
(`cost_per_request_sensitivity`); a V3 or A6 confirmation must hold at both (the report reads both).

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
**extrapolation**; the result records the engines and load tests it combined, and J3 labels it. The
combination leaves out engines outside the limit (`combined.outside_slo` names them; then there is no
combined cost) and B4 as deployed (`slm:<candidate>+as-deployed`): that sweep is B4's own traffic on one
server with every adapter, so its cost is B4's measured one (`as_deployed`), which J3 prices B4 with.
"""
import hashlib
import json
from fractions import Fraction
from typing import Any, Dict, List, Optional

from bench.judge.base import JudgmentError, calls_of, canonical, reference, require_done, run_dir, write_result

JUDGMENT = "J8"
EXPORT = "profile_export_aiperf.json"
# the configuration keys J8 reads, recorded in the result's `reads` (design §6.2)
CONFIG_KEYS = ("cost.limit_min_calls", "cost.limit_pool", "cost.p95_slo_cap_ms", "cost.slo_from", "cost.utilizations",
               "modal.gpu_prices", "modal.gpu_prices_sensitivity", "serving.cpu", "serving.memory_gib")
AS_DEPLOYED = "+as-deployed"  # bench/loadtest.py's label of B4 served as deployed
ROUTED = ("adapter", "as_deployed")  # the sweeps the router feeds (F3's traffic_kind)
ROUTES = "routes.jsonl"


def level(export: Dict[str, Any]) -> Dict[str, float]:
    """(p95 latency in ms, throughput in requests/s) of one AIPerf export."""
    latency, throughput = export.get("request_latency") or {}, export.get("request_throughput") or {}
    if latency.get("unit") != "ms" or throughput.get("unit") != "requests/sec":
        raise JudgmentError(f"unexpected AIPerf units: request_latency in {latency.get('unit')!r}, "
                            f"request_throughput in {throughput.get('unit')!r}")
    if latency.get("p95") is None or throughput.get("avg") is None:
        raise JudgmentError("the AIPerf export has no request_latency.p95 or request_throughput.avg")
    return {"p95_ms": float(latency["p95"]), "throughput_rps": float(throughput["avg"])}


def judge(levels: List[Dict[str, Any]], price_per_hour: float, slo_ms: float, utilizations: List[float],
          added_ms: float = 0.0, sensitivity_per_hour: Optional[float] = None) -> Dict[str, Any]:
    """`levels`: [{"concurrency", "p95_ms", "throughput_rps", "run_id"}], one engine on one GPU; `added_ms` is
    added to each level's p95 before the comparison (the router's time). No level within `slo_ms` is a result:
    `outside_slo`, no sustained throughput, no cost."""
    within = [lv for lv in levels if lv["p95_ms"] + added_ms <= slo_ms]
    out = {"levels": sorted(levels, key=lambda lv: lv["concurrency"]), "p95_slo_ms": slo_ms, "added_ms": added_ms,
           "outside_slo": not within, "price_per_hour": price_per_hour}
    if not within:
        return {**out, "sustained": None, "requests_per_hour": None, "cost_per_request": None,
                **({"cost_per_request_sensitivity": None} if sensitivity_per_hour is not None else {})}
    best = max(within, key=lambda lv: (lv["throughput_rps"], -lv["concurrency"]))
    per_hour = best["throughput_rps"] * 3600

    def cost(price: float) -> Dict[str, float]:
        return {f"{round(u * 100)}%": price / (per_hour * u) for u in utilizations}
    out.update(sustained=best, requests_per_hour=per_hour, cost_per_request=cost(price_per_hour))
    if sensitivity_per_hour is not None:
        out["cost_per_request_sensitivity"] = cost(sensitivity_per_hour)
    return out


def p95(values: List[float]) -> float:
    """The 95th percentile by linear interpolation between order statistics (numpy's default), the
    rank computed in integers."""
    ordered = sorted(values)
    rank = Fraction(95 * (len(ordered) - 1), 100)
    low = rank.numerator // rank.denominator
    if low == len(ordered) - 1:
        return float(ordered[low])
    return float(ordered[low] + (rank - low) * (ordered[low + 1] - ordered[low]))


def weighted_quantile(values: List[float], weights: List[Fraction], q: Fraction = Fraction(95, 100)) -> float:
    """The smallest value at which the cumulative weight reaches the fraction `q` of the total (the inverse of
    the weighted distribution), in exact fractions: deterministic, whatever the order of `values`."""
    pairs = sorted(zip(values, weights), key=lambda pair: pair[0])
    total, running = sum(weights), Fraction(0)
    for value, weight in pairs:
        running += weight
        if running >= q * total:
            return float(value)
    return float(pairs[-1][0])


def limit(first: List[dict], sweep_sites: Dict[str, int], cap_ms: float, min_calls: int,
          pool: Dict[str, str]) -> Dict[str, Any]:
    """The limit of one sweep: the teacher's first attempts (`first`) reweighted to the sweep's call-site counts
    (`sweep_sites`); a call site with fewer than `min_calls` teacher calls counts with the one `pool` names for
    it. Records the shares, the n, the pools, the weighted p95, the limit, and the unweighted ones."""
    n = {}
    for call in first:
        n[call["call_site"]] = n.get(call["call_site"], 0) + 1
    group = {}
    for site in set(n) | set(sweep_sites):
        if n.get(site, 0) >= min_calls:
            group[site] = site
        elif pool.get(site) and n.get(pool[site], 0) >= min_calls:
            group[site] = pool[site]
        elif sweep_sites.get(site):
            raise JudgmentError(f"call site {site} has {n.get(site, 0)} teacher calls (< cost.limit_min_calls "
                                f"{min_calls}) and cost.limit_pool pools it with none that has enough")
        else:
            group[site] = site  # in the teacher's calls only: weight 0 in this sweep
    total = sum(sweep_sites.values())
    if not total:
        raise JudgmentError("the sweep records no call-site mix (composition.source.call_sites)")
    share = {g: Fraction(0) for g in set(group.values())}
    for site, count in sweep_sites.items():
        share[group[site]] += Fraction(count, total)
    n_group = {g: sum(k for site, k in n.items() if group[site] == g) for g in share}
    latencies = [call["latency_ms"] for call in first]
    weights = [share[group[call["call_site"]]] / n_group[group[call["call_site"]]] for call in first]
    weighted = weighted_quantile(latencies, weights)
    unweighted = weighted_quantile(latencies, [Fraction(1)] * len(latencies))
    return {"cap_ms": cap_ms, "shares": {g: float(v) for g, v in sorted(share.items()) if v},
            "n": dict(sorted(n.items())), "pooled": {site: g for site, g in sorted(group.items()) if site != g},
            "weighted_p95_ms": weighted, "slo_ms": min(cap_ms, weighted),
            "unweighted_p95_ms": unweighted, "unweighted_slo_ms": min(cap_ms, unweighted)}


def router_ms(source_run_id: str) -> Dict[str, Any]:
    """The p95 of the embedding time over a routed run (B4's on calib: the sweep's source), from its
    routes.jsonl, with the sha256 of the file read (the run's manifest does not cover it); the wait for the
    embedding model's lock is left out and reported."""
    from bench.judge.base import read_jsonl

    path = run_dir(source_run_id) / ROUTES
    if not path.is_file():
        raise JudgmentError(f"{source_run_id} has no {ROUTES}: the router's time cannot be added to its sweep")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = read_jsonl(path)
    embeds = [r["embed_ms"] for r in rows if r.get("embed_ms") is not None]
    if not embeds:
        raise JudgmentError(f"{source_run_id}/{ROUTES} records no embedding time")
    waits = [r["embed_wait_ms"] for r in rows if r.get("embed_wait_ms") is not None]
    return {"run_id": source_run_id, "routes_sha256": sha256, "n": len(embeds), "embed_p95_ms": p95(embeds),
            "embed_wait_p95_ms": p95(waits) if waits else None}


def slo(slo_from: str, cap_ms: float) -> Dict[str, Any]:
    """The SLO rule: the stricter of the cap and the production LLM's p95 latency per call, over the
    first attempts of the pilot's B0 execution on calib."""
    require_done(slo_from, type="agent", arm="B0", split="calib")
    first = [call["latency_ms"] for call in calls_of(slo_from) if call["attempt"] == 1]
    if not first:
        raise JudgmentError(f"{slo_from} has no call: the production LLM's p95 latency cannot be measured")
    measured = p95(first)
    return {"cap_ms": cap_ms, "llm_p95_ms": measured, "slo_ms": min(cap_ms, measured), "first_attempts": len(first)}


def container_price(config: Dict[str, Any], gpu: str, key: str = "gpu_prices") -> Dict[str, float]:
    """USD per second of one serving container, by what the provider bills: the GPU, the CPU cores and the
    memory, at `modal.<key>` (the provider's price, or Modal's as sensitivity)."""
    prices, serving = (config.get("modal") or {}).get(key) or {}, config.get("serving") or {}
    per_gpu = (prices.get("usd_per_s") or {}).get(gpu)
    if per_gpu is None or not prices.get("as_of"):
        raise JudgmentError(f"modal.{key} (F3) has no dated price for {gpu!r}")
    missing = [name for name, value in ((f"modal.{key}.cpu_usd_per_core_s", prices.get("cpu_usd_per_core_s")),
                                        (f"modal.{key}.memory_usd_per_gib_s", prices.get("memory_usd_per_gib_s")),
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
                       "sweep_id": found["sweep_id"], "source_run_id": found.get("source_run_id"),
                       "traffic_kind": found.get("traffic_kind") or "base",
                       "call_sites": ((found.get("composition") or {}).get("source") or {}).get("call_sites"),
                       **level(export)})
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
    bases = {e.split("+", 1)[0] for e in engines}
    if len(bases) != 1 or len(gpus) != 1:
        raise JudgmentError(f"the load tests must be engines of one base on one GPU: {engines} on {sorted(gpus)}")
    gpu = gpus.pop()
    per_second = container_price(config, gpu)
    per_hour = sum(per_second.values()) * 3600
    prices = config["modal"]["gpu_prices"]
    sensitivity = (config.get("modal") or {}).get("gpu_prices_sensitivity")
    sensitivity_per_hour = sum(container_price(config, gpu, "gpu_prices_sensitivity").values()) * 3600 if sensitivity else None
    if cost.get("limit_min_calls") is None:
        raise JudgmentError("cost.limit_min_calls is not set: the limit's pooling of rare call sites must be pre-registered")
    first = [call for call in calls_of(slo_from) if call["attempt"] == 1]
    per_engine, routed_by = {}, {}
    for engine in engines:
        mine = [lv for lv in levels if lv["engine"] == engine]
        mixes = {json.dumps(lv["call_sites"], sort_keys=True) for lv in mine}
        kinds = {lv["traffic_kind"] for lv in mine}
        if len(mixes) != 1 or len(kinds) != 1 or mine[0]["call_sites"] is None:
            raise JudgmentError(f"the sweep of {engine} records no single call-site mix and traffic kind")
        bound_engine = limit(first, mine[0]["call_sites"], cost["p95_slo_cap_ms"], cost["limit_min_calls"],
                             cost.get("limit_pool") or {})
        kind = kinds.pop()
        router = None
        if kind in ROUTED:
            sources = {lv["source_run_id"] for lv in mine}
            if len(sources) != 1:
                raise JudgmentError(f"the sweep of {engine} replays several sources {sorted(sources)}")
            router = routed_by.setdefault(next(iter(sources)), router_ms(next(iter(sources))))
        added = router["embed_p95_ms"] if router else 0.0
        judged = judge(mine, per_hour, bound_engine["slo_ms"], cost["utilizations"], added, sensitivity_per_hour)
        plain = judge(mine, per_hour, bound_engine["slo_ms"], cost["utilizations"], 0.0, sensitivity_per_hour)
        unweighted = judge(mine, per_hour, bound_engine["unweighted_slo_ms"], cost["utilizations"], added,
                           sensitivity_per_hour)
        per_engine[engine] = {"traffic_kind": kind, "limit": bound_engine, "router": router, **judged,
                              "without_router": {k: plain[k] for k in ("outside_slo", "sustained", "cost_per_request")},
                              "unweighted_limit": {k: unweighted[k] for k in ("outside_slo", "sustained", "cost_per_request")}}
    single = [e for e in engines if not e.endswith(AS_DEPLOYED)]
    outside = [e for e in single if per_engine[e]["outside_slo"]]

    def highest(key: str) -> Optional[Dict[str, float]]:
        if not single or outside:
            return None
        return {u: max(per_engine[e][key][u] for e in single) for u in per_engine[single[0]][key]}
    result = {"engine": bases.pop(), "gpu": gpu, "gpu_prices_as_of": prices["as_of"], "price_per_hour": per_hour,
              "price_per_second": per_second, "gpu_prices_sha256": hashlib.sha256(canonical(prices)).hexdigest(),
              "slo": bound,
              "source_run_ids": sorted({lv.get("source_run_id") for lv in levels if lv.get("source_run_id")}),
              "utilizations": [f"{round(u * 100)}%" for u in cost["utilizations"]],
              "engines": per_engine, "cost_per_request": highest("cost_per_request"),
              "outside_slo": [e for e in engines if per_engine[e]["outside_slo"]],
              "as_deployed": {e: per_engine[e] for e in engines if e.endswith(AS_DEPLOYED)} or None,
              "combined": {"rule": "the highest cost per request among the engines measured, none if one is outside "
                                   "its limit; B4 as deployed apart", "engines": single, "outside_slo": outside,
                           "loadtests": sorted(reads), "sweeps": sorted({lv["sweep_id"] for lv in levels})}}
    if sensitivity:
        result.update(sensitivity={"as_of": sensitivity.get("as_of"), "provider": sensitivity.get("provider"),
                                   "price_per_hour": sensitivity_per_hour,
                                   "sha256": hashlib.sha256(canonical(sensitivity)).hexdigest()},
                      cost_per_request_sensitivity=highest("cost_per_request_sensitivity"))
    reads_out = {"loadtests": reads, "slo_from": reference(slo_from), "config": list(CONFIG_KEYS)}
    if routed_by:
        reads_out["routes"] = {run_id: {**reference(run_id), ROUTES: routed_by[run_id]["routes_sha256"]}
                               for run_id in sorted(routed_by)}
    return write_result(JUDGMENT, reads_out, result)
