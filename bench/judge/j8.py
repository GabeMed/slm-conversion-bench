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

**The teacher's warm-up gate is left out of its latency**: the harness sends the first call of each cache prefix
alone in each process and holds the calls of the same prefix that arrive meanwhile until it ends
(`bench.agent.anthropic_engine._Warmup`, timed inside the call by `hooks._invoke`), so their latency counts a
wait the API never imposed. The rule follows the gate's own key: the calls of `slo_from` are grouped by
process (the session, the record's `run_id`, and the worker, `runner.worker_parts` over that session's
questions, each process checked to be one contiguous block of calls.jsonl) and by cache prefix (rebuilt from
`prompt_messages` and the run's `config.json` cache_breaks for the call's role, exactly as `_cache_split` cuts
it); in each group the gate's leader is the call that ends first (every call that waited ends after it), and
every other call of the group that started before that end is left out. The limit with them is a sensitivity
(`with_warmup_limit`); `limit.warmup` records how many each call site lost.

**Each level is read in its steady state** (`profile_export.jsonl`, AIPerf's per-request records): the window
runs from the first moment the requests in flight reach `cost.steady_inflight_share` of the level's concurrency
to the last moment they are still there; the throughput is the completions inside it over its length, and the
p95 is over the requests that both start and end inside it (the ramp-up and the drain, where the server serves
fewer than the level, are out). A level is **measured** only if at least `cost.steady_min_requests` whole
requests fall in the window, the window lasts at least `cost.steady_min_share` of the level, and no request
failed or was cancelled (AIPerf's `error_summary`, `was_cancelled`, `is_complete`, and each record's error);
J8 chooses among the measured levels only: an unmeasured one counts neither for nor against, and the result
lists it with its reasons (`unmeasured`). An engine with no measured level is refused. AIPerf's aggregates over
the whole level (`request_latency.p95`, `request_throughput.avg`) are kept per level and judged as a
sensitivity (`aiperf_aggregates`). Per level, the p95 of each group of call sites (`cost.latency_groups`) is
set beside the teacher's p95 in the same group, a declared sensitivity with no verdict (`by_group`): each
request's call site is its payload's in the sweep's source execution.

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
(`cost_per_request_sensitivity`); a V3 or A6 confirmation must hold at both (the report reads both). Modal bills
a container's CPU at the higher of its request and its use: the sensitivity prices `serving.cpu` cores, or the
serving process's measured cores (the load tests' `cpu_sample.server.cores`, the highest level) where that is
more (`sensitivity.cpu_cores`); the memory is `serving.memory_gib`, declared, as no load test measures it.

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
from typing import Any, Dict, List, Optional, Tuple

from bench.judge.base import JudgmentError, calls_of, canonical, reference, require_done, run_dir, write_result

JUDGMENT = "J8"
EXPORT = "profile_export_aiperf.json"
# the configuration keys J8 reads, recorded in the result's `reads` (design §6.2)
CONFIG_KEYS = ("cost.latency_groups", "cost.limit_min_calls", "cost.limit_pool", "cost.p95_slo_cap_ms", "cost.slo_from",
               "cost.steady_inflight_share", "cost.steady_min_requests", "cost.steady_min_share", "cost.utilizations",
               "modal.gpu_prices", "modal.gpu_prices_sensitivity", "serving.cpu", "serving.memory_gib")
RECORDS = "profile_export.jsonl"  # AIPerf's per-request records
PAYLOADS = "payloads.jsonl"       # the lines a level replayed, in order (F3); AIPerf's session_num is the line's index
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
    added to each level's p95 before the comparison (the router's time). A level with `measured` false is left
    out of the choice and listed under `unmeasured`. No measured level within `slo_ms` is a result:
    `outside_slo`, no sustained throughput, no cost."""
    measured = [lv for lv in levels if lv.get("measured", True)]  # an unmeasured level counts neither way
    kept = {id(lv) for lv in measured}
    within = [lv for lv in measured if lv["p95_ms"] + added_ms <= slo_ms]
    out = {"levels": sorted(levels, key=lambda lv: lv["concurrency"]), "p95_slo_ms": slo_ms, "added_ms": added_ms,
           "outside_slo": not within, "price_per_hour": price_per_hour,
           "unmeasured": [{"run_id": lv.get("run_id"), "concurrency": lv["concurrency"], "reasons": lv.get("reasons", [])}
                          for lv in sorted(levels, key=lambda lv: lv["concurrency"]) if id(lv) not in kept]}
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


# ---------------------------------------------------------------- the steady state of a level

def steady(records: List[dict], concurrency: int, inflight_share: float, min_requests: int,
           min_share: float) -> Dict[str, Any]:
    """A level's steady state from AIPerf's per-request records (`metadata.request_start_ns`/`request_end_ns`):
    the window from the first moment the requests in flight reach `inflight_share` × `concurrency` to the moment
    they last fall below it (in flight is a step function: every start and end at one instant applies before it
    is read, so back-to-back requests never dip); the throughput is the completions in the window over its
    length; the p95 (by `p95`, AIPerf's own definition) is over the requests that both start and end in it.

    Measured only if at least `min_requests` whole requests fall in the window, and the window lasts at least
    `min_share` of the level (its first start to its last end). Why 100 (`cost.steady_min_requests`): the
    p95 by interpolation sits at rank 0.95 (n − 1); with n = 100 it falls between the 95th and the 96th order
    statistics, four requests above it, so no single slow request moves it; with n = 20 it is between the 19th
    and the 20th, almost the maximum. Equivalently, the requests above the true p95 are Binomial(n, 0.05): 5 on
    average at n = 100, and the chance that none is (the sample p95 then estimating nothing above the body) is
    0.95^100 < 0.6%, against 36% at n = 20. Why half the level (`cost.steady_min_share`): a window shorter than
    the ramp-up and the drain together is a transient, not the regime the level was meant to hold."""
    spans = [(int(r["metadata"]["request_start_ns"]), int(r["metadata"]["request_end_ns"])) for r in records]
    out: Dict[str, Any] = {"records": len(spans), "inflight_threshold": inflight_share * concurrency,
                           "min_requests": min_requests, "min_share": min_share}
    if not spans:
        return {**out, "measured": False, "reasons": ["no request record"]}
    first, last = min(s for s, _ in spans), max(e for _, e in spans)
    deltas: Dict[int, int] = {}
    for start, end in spans:
        deltas[start] = deltas.get(start, 0) + 1
        deltas[end] = deltas.get(end, 0) - 1
    threshold = Fraction(inflight_share).limit_denominator(10 ** 6) * concurrency
    inflight, t0, t1 = 0, None, None
    for t in sorted(deltas):
        before, inflight = inflight, inflight + deltas[t]
        if inflight >= threshold and t0 is None:
            t0 = t
        if before >= threshold > inflight:
            t1 = t
    level_ns = last - first
    out["level_s"] = level_ns / 1e9
    if t0 is None or t1 is None or t1 <= t0:
        return {**out, "measured": False, "window_s": 0.0, "window_share": 0.0, "whole_requests": 0,
                "reasons": [f"the requests in flight never reached {inflight_share:g} x {concurrency}"]}
    whole = [end - start for start, end in spans if start >= t0 and end <= t1]
    completed = sum(1 for _, end in spans if t0 < end <= t1)
    window_ns = t1 - t0
    share = window_ns / level_ns if level_ns else 0.0
    reasons = []
    if len(whole) < min_requests:
        reasons.append(f"{len(whole)} whole requests in the steady window (< cost.steady_min_requests {min_requests})")
    if share < min_share:
        reasons.append(f"the steady window lasts {share:.3f} of the level (< cost.steady_min_share {min_share:g})")
    return {**out, "measured": not reasons, "reasons": reasons, "window_start_ns": t0, "window_end_ns": t1,
            "window_s": window_ns / 1e9, "window_share": share, "whole_requests": len(whole),
            "completed_in_window": completed, "throughput_rps": completed / (window_ns / 1e9),
            "p95_ms": p95([ns / 1e6 for ns in whole]) if whole else None}


def failures(export: Dict[str, Any], records: List[dict]) -> Dict[str, Any]:
    """What failed in a level, from AIPerf's export and its records: none of it may be non-zero for the level
    to be measured (a failed request is one the server did not answer, not a fast one)."""
    summary = export.get("error_summary")
    missing = [key for key in ("error_summary", "was_cancelled", "is_complete") if key not in export]
    out = {"error_summary": len(summary or []), "was_cancelled": export.get("was_cancelled"),
           "is_complete": export.get("is_complete"),
           "records_with_error": sum(1 for r in records if r.get("error")),
           "records_cancelled": sum(1 for r in records if (r.get("metadata") or {}).get("was_cancelled"))}
    reasons = [f"the AIPerf export has no {', '.join(missing)}"] if missing else []
    if out["error_summary"]:
        reasons.append(f"AIPerf's error_summary lists {out['error_summary']} error type(s)")
    if out["was_cancelled"]:
        reasons.append("AIPerf was cancelled")
    if "is_complete" in export and not out["is_complete"]:
        reasons.append("AIPerf's run is not complete")
    if out["records_with_error"]:
        reasons.append(f"{out['records_with_error']} request(s) failed")
    if out["records_cancelled"]:
        reasons.append(f"{out['records_cancelled']} request(s) cancelled")
    return {**out, "reasons": reasons}


def read_level(run_id: str, concurrency: int, cost: Dict[str, Any]) -> Dict[str, Any]:
    """One level: AIPerf's aggregates (a sensitivity), its steady state, its failures, and whether it is measured;
    `p95_ms` and `throughput_rps` are the steady state's (None when it has none)."""
    folder = run_dir(run_id)
    export = json.loads((folder / EXPORT).read_text())
    path = folder / RECORDS
    if not path.is_file():
        raise JudgmentError(f"{run_id} has no {RECORDS}: its steady state cannot be read")
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    profiling = [r for r in records if (r.get("metadata") or {}).get("benchmark_phase", "profiling") == "profiling"]
    window = steady([r for r in profiling if not r.get("error")], concurrency, cost["steady_inflight_share"],
                    cost["steady_min_requests"], cost["steady_min_share"])
    failed = failures(export, profiling)
    reasons = failed["reasons"] + window["reasons"]
    return {"p95_ms": window.get("p95_ms"), "throughput_rps": window.get("throughput_rps"), "measured": not reasons,
            "reasons": reasons, "aiperf": level(export), "steady": {k: v for k, v in window.items() if k != "reasons"},
            "failures": {**{k: v for k, v in failed.items() if k != "reasons"}, "none": not failed["reasons"]},
            "records_sha256": hashlib.sha256(raw).hexdigest(), "_records": profiling}


# ---------------------------------------------------------------- the teacher's warm-up gate

def cache_prefix(messages: List[dict], markers: List[str]) -> Optional[str]:
    """The key the harness's warm-up gate holds a call on (`anthropic_engine.AnthropicChat.request`): the system
    text and the last user message up to the first of `markers` it contains (its last occurrence), with the
    turns before it; None where nothing splits (no gate). Kept unhashed: the same key, the same groups."""
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system") or None
    turns = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") != "system"]
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["role"] == "user" and isinstance(turns[i]["content"], str):
            break
    else:
        return None
    text = turns[i]["content"]
    for marker in markers:
        at = text.rfind(marker)
        if at >= 0:
            cut = at + len(marker)
            return None if cut >= len(text) else f"{system}|{repr(turns[:i]) + text[:cut]}"
    return None


def processes(calls: List[dict], found: Dict[str, Any]) -> List[Tuple[str, int]]:
    """Each call's process: its session (the record's run_id) and its worker, `runner.worker_parts` over that
    session's questions (worker k takes every `workers`-th from the k-th). The run's own session answered its
    questions less the resumed ones; the session it resumed (one, at most) answered them all. Each process must
    be one contiguous block of calls.jsonl, as the runner merges them, or the reconstruction is refused."""
    resumed = found.get("resumed_from") or {}
    if len(found.get("sessions") or [None]) > 2:
        raise JudgmentError(f"{found['run_id']} was resumed more than once: its calls' processes are not reconstructed")
    kept = set(found.get("resumed_question_ids") or [])
    questions = [str(q) for q in found.get("question_ids") or []]
    sessions = {found["run_id"]: ([q for q in questions if q not in kept], found.get("workers") or 1)}
    if resumed.get("run_id"):
        sessions[resumed["run_id"]] = (questions, resumed.get("workers") or 1)
    worker: Dict[str, Dict[str, int]] = {}
    for session, (asked, workers) in sessions.items():
        worker[session] = {q: i % workers for i, q in enumerate(asked)} if workers > 1 else {}
    out = []
    for call in calls:
        session = call["run_id"]
        if session not in sessions:
            raise JudgmentError(f"a call of {found['run_id']} names the session {session}, which the manifest does not")
        if sessions[session][1] > 1 and call["question_id"] not in worker[session]:
            raise JudgmentError(f"{found['run_id']}: question {call['question_id']} is in no worker of {session}")
        out.append((session, worker[session].get(call["question_id"], 0)))
    seen, previous = set(), None
    for process in out:
        if process != previous:
            if process in seen:
                raise JudgmentError(f"{found['run_id']}: the calls of process {process} are not one block of calls.jsonl")
            seen.add(process)
            previous = process
    return out


def warmup_gated(calls: List[dict], found: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """The calls the warm-up gate held: per process and cache prefix, the gate's leader is the call that ends
    first (each held call ends after it: it waits for its end), and every other call of the group that started
    before that end was held. {"call_ids": set, "by_call_site": {site: n first attempts held}, "rule"}."""
    from datetime import datetime, timedelta

    if found.get("via_claude_code"):  # the Claude Code path returns before the gate
        return {"call_ids": set(), "by_call_site": {}, "groups": 0}
    roles = config.get("roles") or {}
    groups: Dict[Tuple, List[Tuple[datetime, datetime, dict]]] = {}
    for call, process in zip(calls, processes(calls, found)):
        markers = ((roles.get(call.get("model_role")) or {}).get("params") or {}).get("cache_breaks") or []
        key = cache_prefix(call["prompt_messages"], markers) if markers else None
        if key is None:
            continue
        start = datetime.fromisoformat(call["started_at"])
        groups.setdefault((process, key), []).append((start, start + timedelta(milliseconds=call["latency_ms"]), call))
    held = set()
    for members in groups.values():
        leader = min(members, key=lambda m: (m[1], m[0]))
        held |= {call["call_id"] for start, _, call in members if call is not leader[2] and start < leader[1]}
    by_site: Dict[str, int] = {}
    for call in calls:
        if call["call_id"] in held and call["attempt"] == 1:
            by_site[call["call_site"]] = by_site.get(call["call_site"], 0) + 1
    return {"call_ids": held, "by_call_site": dict(sorted(by_site.items())), "groups": len(groups)}


# ---------------------------------------------------------------- p95 per group of call sites (no verdict)

def request_sites(run_id: str, source_calls: List[dict]) -> Optional[Dict[int, str]]:
    """Each request's call site in a level, by its payload (AIPerf's session_num is its line in payloads.jsonl)
    matched on its messages with the source execution's calls; None where a payload matches no single site."""
    path = run_dir(run_id) / PAYLOADS
    if not path.is_file():
        return None
    by_messages: Dict[str, set] = {}
    for call in source_calls:
        by_messages.setdefault(json.dumps(call["prompt_messages"], sort_keys=True), set()).add(call["call_site"])
    sites = {}
    for i, line in enumerate(l for l in path.read_text().splitlines() if l.strip()):
        found = by_messages.get(json.dumps(json.loads(line).get("messages"), sort_keys=True)) or set()
        if len(found) != 1:
            return None
        sites[i] = next(iter(found))
    return sites


def by_group(levels: List[Dict[str, Any]], teacher: List[dict], groups: Dict[str, List[str]],
             sites_of: Dict[str, Optional[Dict[int, str]]]) -> Dict[str, Any]:
    """Per level, the p95 of each group's whole requests in the steady window, beside the teacher's p95 in the
    same group (its first attempts, warm-up gate out): a declared sensitivity, no verdict."""
    group_of = {site: name for name, members in groups.items() for site in members}
    mine: Dict[str, List[float]] = {}
    for call in teacher:
        mine.setdefault(group_of.get(call["call_site"], "other"), []).append(call["latency_ms"])
    out = {"teacher": {g: {"p95_ms": p95(v), "n": len(v)} for g, v in sorted(mine.items())}, "levels": {}}
    for lv in sorted(levels, key=lambda lv: lv["concurrency"]):
        sites, window = sites_of.get(lv["run_id"]), lv["steady"]
        if sites is None or window.get("window_start_ns") is None:
            out["levels"][str(lv["concurrency"])] = None
            continue
        latencies: Dict[str, List[float]] = {}
        for r in lv["_records"]:
            meta = r["metadata"]
            if r.get("error") or not (window["window_start_ns"] <= meta["request_start_ns"]
                                      and meta["request_end_ns"] <= window["window_end_ns"]):
                continue
            group = group_of.get(sites.get(meta.get("session_num")), "other")
            latencies.setdefault(group, []).append((meta["request_end_ns"] - meta["request_start_ns"]) / 1e6)
        out["levels"][str(lv["concurrency"])] = {g: {"p95_ms": p95(v), "n": len(v)} for g, v in sorted(latencies.items())}
    return out


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


def teacher_calls(slo_from: str) -> Dict[str, Any]:
    """The first attempts of the pilot's B0 execution on calib, all of them and without the calls the warm-up
    gate held, and what the gate rule read (the run's config.json, hashed)."""
    found = require_done(slo_from, type="agent", arm="B0", split="calib")
    calls = calls_of(slo_from)
    path = run_dir(slo_from) / "config.json"
    if not path.is_file():
        raise JudgmentError(f"{slo_from} has no config.json: the warm-up gate's cache prefixes cannot be rebuilt")
    gated = warmup_gated(calls, found, json.loads(path.read_text()))
    first = [call for call in calls if call["attempt"] == 1]
    return {"first": first, "kept": [c for c in first if c["call_id"] not in gated["call_ids"]],
            "warmup": {"excluded": gated["by_call_site"], "groups": gated["groups"],
                       "rule": "per process and cache prefix, the calls that started before the first of the group "
                               "ended, that one (the gate's leader) excepted"},
            "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def slo(slo_from: str, cap_ms: float, teacher: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The SLO rule: the stricter of the cap and the production LLM's p95 latency per call, over the
    first attempts of the pilot's B0 execution on calib, the warm-up gate's held calls out (with them, a
    sensitivity)."""
    teacher = teacher or teacher_calls(slo_from)
    if not teacher["first"]:
        raise JudgmentError(f"{slo_from} has no call: the production LLM's p95 latency cannot be measured")
    measured = p95([call["latency_ms"] for call in teacher["kept"]])
    return {"cap_ms": cap_ms, "llm_p95_ms": measured, "slo_ms": min(cap_ms, measured),
            "first_attempts": len(teacher["kept"]), "warmup_excluded": len(teacher["first"]) - len(teacher["kept"]),
            "llm_p95_ms_with_warmup": p95([call["latency_ms"] for call in teacher["first"]])}


def container_price(config: Dict[str, Any], gpu: str, key: str = "gpu_prices",
                    cpu_cores: Optional[float] = None) -> Dict[str, float]:
    """USD per second of one serving container, by what the provider bills: the GPU, the CPU cores and the
    memory, at `modal.<key>` (the provider's price, or Modal's as sensitivity). `cpu_cores` replaces
    `serving.cpu` where given (Modal's: the higher of the request and the measured use)."""
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
    cores = serving["cpu"] if cpu_cores is None else cpu_cores
    return {"gpu": per_gpu, "cpu": cores * prices["cpu_usd_per_core_s"],
            "memory": serving["memory_gib"] * prices["memory_usd_per_gib_s"]}


STEADY_KEYS = ("steady_inflight_share", "steady_min_requests", "steady_min_share")


def run(loadtest_run_ids: List[str], config: Dict[str, Any], sweeps: Optional[List[str]] = None, *, slo_from: str):
    cost = config["cost"]
    if cost.get("p95_slo_cap_ms") is None:
        raise JudgmentError("cost.p95_slo_cap_ms is not set: the SLO needs its pre-registered cap")
    if cost.get("slo_from") not in (None, slo_from):
        raise JudgmentError(f"cost.slo_from names {cost['slo_from']}, not {slo_from}: the SLO's execution is fixed in "
                            f"the configuration, before the load test")
    unset = [f"cost.{key}" for key in STEADY_KEYS if cost.get(key) is None]
    if unset:
        raise JudgmentError(f"{', '.join(unset)} not set: a level's steady state is read by a pre-registered rule")
    teacher = teacher_calls(slo_from)
    bound = slo(slo_from, cost["p95_slo_cap_ms"], teacher)
    levels, records = [], {}
    for run_id in sorted(loadtest_run_ids):
        found = require_done(run_id, type="loadtest")
        if found.get("prefix_cache") is not True:
            raise JudgmentError(f"{run_id} did not run with the prefix cache on (SPEC §6.6)")
        if not found.get("sweep_id"):
            raise JudgmentError(f"{run_id} records no sweep_id: J8 takes one sweep per engine")
        read = read_level(run_id, found["concurrency"], cost)
        records[run_id] = read.pop("_records")
        levels.append({"run_id": run_id, "engine": found["engine"], "concurrency": found["concurrency"],
                       "sweep_id": found["sweep_id"], "source_run_id": found.get("source_run_id"),
                       "traffic_kind": found.get("traffic_kind") or "base",
                       "call_sites": ((found.get("composition") or {}).get("source") or {}).get("call_sites"),
                       "server_cores": ((found.get("cpu_sample") or {}).get("server") or {}).get("cores"), **read})
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
    reads = {lv["run_id"]: {**reference(lv["run_id"]), RECORDS: lv["records_sha256"]} for lv in levels}
    gpus = {require_done(lv["run_id"])["gpu"] for lv in levels}
    engines = sorted({lv["engine"] for lv in levels})
    bases = {e.split("+", 1)[0] for e in engines}
    if len(bases) != 1 or len(gpus) != 1:
        raise JudgmentError(f"the load tests must be engines of one base on one GPU: {engines} on {sorted(gpus)}")
    for engine in engines:
        if not any(lv["measured"] for lv in levels if lv["engine"] == engine):
            reasons = {lv["concurrency"]: lv["reasons"] for lv in levels if lv["engine"] == engine}
            raise JudgmentError(f"{engine} has no measured level (steady state or failures): {reasons}")
    gpu = gpus.pop()
    per_second = container_price(config, gpu)
    per_hour = sum(per_second.values()) * 3600
    prices = config["modal"]["gpu_prices"]
    sensitivity = (config.get("modal") or {}).get("gpu_prices_sensitivity")
    observed = [lv["server_cores"] for lv in levels if lv.get("server_cores") is not None]
    cores = max([config["serving"]["cpu"]] + observed) if sensitivity and (config.get("serving") or {}).get("cpu") is not None else None
    sensitivity_per_hour = (sum(container_price(config, gpu, "gpu_prices_sensitivity", cores).values()) * 3600
                            if sensitivity else None)
    if cost.get("limit_min_calls") is None:
        raise JudgmentError("cost.limit_min_calls is not set: the limit's pooling of rare call sites must be pre-registered")
    groups = cost.get("latency_groups") or {}
    per_engine, routed_by, sources = {}, {}, {}
    for engine in engines:
        mine = [lv for lv in levels if lv["engine"] == engine]
        mixes = {json.dumps(lv["call_sites"], sort_keys=True) for lv in mine}
        kinds = {lv["traffic_kind"] for lv in mine}
        if len(mixes) != 1 or len(kinds) != 1 or mine[0]["call_sites"] is None:
            raise JudgmentError(f"the sweep of {engine} records no single call-site mix and traffic kind")
        bound_engine = limit(teacher["kept"], mine[0]["call_sites"], cost["p95_slo_cap_ms"], cost["limit_min_calls"],
                             cost.get("limit_pool") or {})
        with_warmup = limit(teacher["first"], mine[0]["call_sites"], cost["p95_slo_cap_ms"], cost["limit_min_calls"],
                            cost.get("limit_pool") or {})
        bound_engine["warmup"] = {**teacher["warmup"], "weighted_p95_ms_with_warmup": with_warmup["weighted_p95_ms"],
                                  "slo_ms_with_warmup": with_warmup["slo_ms"]}
        kind = kinds.pop()
        router = None
        if kind in ROUTED:
            sources_of = {lv["source_run_id"] for lv in mine}
            if len(sources_of) != 1:
                raise JudgmentError(f"the sweep of {engine} replays several sources {sorted(sources_of)}")
            router = routed_by.setdefault(next(iter(sources_of)), router_ms(next(iter(sources_of))))
        added = router["embed_p95_ms"] if router else 0.0
        args = (per_hour, bound_engine["slo_ms"], cost["utilizations"])
        judged = judge(mine, *args, added, sensitivity_per_hour)
        plain = judge(mine, *args, 0.0, sensitivity_per_hour)
        unweighted = judge(mine, per_hour, bound_engine["unweighted_slo_ms"], cost["utilizations"], added,
                           sensitivity_per_hour)
        warm = judge(mine, per_hour, with_warmup["slo_ms"], cost["utilizations"], added, sensitivity_per_hour)
        # AIPerf's whole-level aggregates, over every level with no failure: the reading before the steady state
        aggregates = judge([{**lv, **lv["aiperf"], "measured": lv["failures"]["none"]} for lv in mine], *args, added,
                           sensitivity_per_hour)
        sites_of = {}
        for lv in mine:
            source = lv.get("source_run_id")
            if source and source not in sources:
                try:
                    sources[source] = calls_of(source)
                except JudgmentError:
                    sources[source] = None
            sites_of[lv["run_id"]] = request_sites(lv["run_id"], sources[source]) if source and sources[source] else None
        keys = ("outside_slo", "sustained", "cost_per_request")
        per_engine[engine] = {"traffic_kind": kind, "limit": bound_engine, "router": router, **judged,
                              "without_router": {k: plain[k] for k in keys},
                              "unweighted_limit": {k: unweighted[k] for k in keys},
                              "with_warmup_limit": {"slo_ms": with_warmup["slo_ms"], **{k: warm[k] for k in keys}},
                              "aiperf_aggregates": {k: aggregates[k] for k in keys + ("unmeasured",)},
                              "by_group": {"rule": "p95 per group of call sites in each level's steady window beside the "
                                                   "teacher's in the same group: a sensitivity, no verdict",
                                           **by_group([{**lv, "_records": records[lv["run_id"]]} for lv in mine],
                                                      teacher["kept"], groups, sites_of)}}
    single = [e for e in engines if not e.endswith(AS_DEPLOYED)]
    outside = [e for e in single if per_engine[e]["outside_slo"]]

    def highest(key: str) -> Optional[Dict[str, float]]:
        if not single or outside:
            return None
        return {u: max(per_engine[e][key][u] for e in single) for u in per_engine[single[0]][key]}
    result = {"engine": bases.pop(), "gpu": gpu, "gpu_prices_as_of": prices["as_of"], "price_per_hour": per_hour,
              "price_per_second": per_second, "gpu_prices_sha256": hashlib.sha256(canonical(prices)).hexdigest(),
              "slo": bound, "steady_rule": {key: cost[key] for key in STEADY_KEYS},
              "source_run_ids": sorted({lv.get("source_run_id") for lv in levels if lv.get("source_run_id")}),
              "utilizations": [f"{round(u * 100)}%" for u in cost["utilizations"]],
              "engines": per_engine, "cost_per_request": highest("cost_per_request"),
              "outside_slo": [e for e in engines if per_engine[e]["outside_slo"]],
              "unmeasured": {e: per_engine[e]["unmeasured"] for e in engines if per_engine[e]["unmeasured"]},
              "as_deployed": {e: per_engine[e] for e in engines if e.endswith(AS_DEPLOYED)} or None,
              "combined": {"rule": "the highest cost per request among the engines measured, none if one is outside "
                                   "its limit; B4 as deployed apart", "engines": single, "outside_slo": outside,
                           "loadtests": sorted(reads), "sweeps": sorted({lv["sweep_id"] for lv in levels})}}
    if sensitivity:
        result.update(sensitivity={"as_of": sensitivity.get("as_of"), "provider": sensitivity.get("provider"),
                                   "price_per_hour": sensitivity_per_hour,
                                   "cpu_cores": {"priced": cores, "requested": config["serving"]["cpu"],
                                                 "measured_max": max(observed) if observed else None,
                                                 "rule": "the higher of serving.cpu and the serving process's measured "
                                                         "cores (Modal bills the higher of request and use)"},
                                   "sha256": hashlib.sha256(canonical(sensitivity)).hexdigest()},
                      cost_per_request_sensitivity=highest("cost_per_request_sensitivity"))
    reads_out = {"loadtests": reads, "slo_from": {**reference(slo_from), "config.json": teacher["config_sha256"]},
                 "config": list(CONFIG_KEYS)}
    read_sources = sorted(source for source, calls in sources.items() if calls is not None)
    if read_sources:  # read only to give each request its call site (by_group)
        reads_out["sources"] = {source: reference(source) for source in read_sources}
    if routed_by:
        reads_out["routes"] = {run_id: {**reference(run_id), ROUTES: routed_by[run_id]["routes_sha256"]}
                               for run_id in sorted(routed_by)}
    return write_result(JUDGMENT, reads_out, result)
