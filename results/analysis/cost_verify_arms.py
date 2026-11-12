"""Independent re-derivation of the per-arm cost / latency / capacity numbers.
Streams calls.jsonl one file at a time, keeps only numeric / id fields, and prices them with the
registered code (bench.judge.j3.judge) on the stripped calls. Read-only on the repository."""
import json, sys, glob, bisect
from datetime import datetime
from collections import defaultdict
import yaml
from bench import paths
from bench.judge import j3

ROOT = "<repo>"
assert str(paths.ROOT) == ROOT, paths.ROOT
OUT = "<scratch>"
J8P = f"{ROOT}/judgments/J8/de3754a76358b70edb33ff0b29f88616d84755923e8becf79cad333750ab5e66/result.json"
RUNS = {
    "B0": "agent-B0-test-20261111T060814.371294Z",
    "B2-prod": "agent-B2-production_llm-test-20261111T144852.703285Z",
    "B2-cheap": "agent-B2-cheap_alt-test-20261111T081959.861177Z",
    "B3": "agent-B3-test-20261111T012223.392984Z",
    "B4": "agent-B4-test-20261111T032707.693347Z",
    "B5": "agent-B5-test-20261111T200809.693765Z",
}
SITE2CL = {"filter_column": "c0", "select_tables": "c1", "select_columns": "c1", "generate_candidate": "c1",
           "revise": "c1", "extract_keywords": "c2", "agent_ir": "c3", "agent_ss": "c3", "agent_cg": "c3"}
config = yaml.safe_load(open(f"{ROOT}/config.yaml"))
PRICES = config["prices"]
PROD = config["roles"]["production_llm"]["model"]
GPU_H = 1.79


def ts(s):
    return datetime.fromisoformat(s).timestamp()


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def pp(xs):
    return {"n": len(xs), "p50": pct(xs, .5), "p95": pct(xs, .95), "p99": pct(xs, .99), "max": max(xs) if xs else None}


def load(rid):
    calls = []
    with open(f"{ROOT}/runs/{rid}/calls.jsonl") as f:
        for line in f:
            c = json.loads(line)
            calls.append({
                "call_id": c["call_id"], "question_id": c["question_id"], "call_site": c["call_site"],
                "cluster": c.get("cluster"), "engine": c["engine"], "model_role": c["model_role"], "model": c["model"],
                "provider": c.get("provider"), "usage": c["usage"], "latency_ms": c.get("latency_ms"),
                "t0": ts(c["started_at"]), "run_id": c["run_id"], "error": c.get("error") is not None,
                "attempt": c.get("attempt") or 1, "response_text": None if c["response_text"] is None else "",
                "invocation_key": c.get("invocation_key"),
            })
            del c, line
    return calls


def window_max(times, weights, w):
    """Largest sum of weights over events whose time falls in a sliding window of w seconds."""
    order = sorted(range(len(times)), key=lambda i: times[i])
    t = [times[i] for i in order]
    cum = [0]
    for i in order:
        cum.append(cum[-1] + weights[i])
    best = 0
    for i in range(len(t)):
        j = bisect.bisect_right(t, t[i] + w)
        best = max(best, cum[j] - cum[i])
    return best


out = {}
only = sys.argv[1:] or list(RUNS)
for arm in only:
    rid = RUNS[arm]
    man = json.load(open(f"{ROOT}/runs/{rid}/manifest.json"))
    calls = load(rid)
    ev = sorted(glob.glob(f"{ROOT}/runs/eval-{rid}-*"))
    evals = []
    for d in ev:
        corr = {}
        with open(d + "/results.jsonl") as f:
            for line in f:
                r = json.loads(line)
                corr[r["question_id"]] = bool(r["correct"])
        evals.append((d.split("/")[-1], sum(corr.values()), len(corr)))
    correct = corr
    slm, read = j3.slm_cost_per_request(J8P, calls, man.get("arm"))
    res = j3.judge(calls, man["question_ids"], correct, PRICES, PROD, slm)
    by_site_reg = res.pop("by_call_site")
    by_q = res.pop("by_question")
    o = {"run_id": rid, "evals": evals, "j3": res, "basis": read.get("slm_cost_basis")}
    # per call site
    sites = defaultdict(lambda: dict(calls=0, slm=0, errors=0, lat_all=[], lat_slm=[], lat_slm_ok=[], lat_api=[]))
    for c in calls:
        s = sites[c["call_site"]]
        s["calls"] += 1
        s["errors"] += c["error"]
        lat = c["latency_ms"] or 0
        s["lat_all"].append(lat)
        if c["model_role"] == "slm":
            s["slm"] += 1
            s["lat_slm"].append(lat)
            if not c["error"]:
                s["lat_slm_ok"].append(lat)
        else:
            s["lat_api"].append(lat)
    o["sites"] = {}
    for site, s in sites.items():
        p = by_site_reg[site]
        o["sites"][site] = {"cluster": SITE2CL[site], "calls": s["calls"], "slm_role_calls": s["slm"], "errors": s["errors"],
                            "priced": {k: p[k] for k in ("calls", "slm_calls", "failed_unbilled", "tokens", "api", "slm")},
                            "lat_all": pp(s["lat_all"]), "lat_slm": pp(s["lat_slm"]), "lat_slm_ok": pp(s["lat_slm_ok"]),
                            "lat_api": pp(s["lat_api"])}
    # recorded cluster and engine split
    eng = defaultdict(lambda: [0, 0])
    for c in calls:
        e = eng[(c["cluster"], c["engine"], c["model_role"])]
        e[0] += 1
        e[1] += c["usage"].get("source") == "missing"
    o["engines"] = [[k[0], k[1], k[2], v[0], v[1]] for k, v in sorted(eng.items(), key=lambda x: str(x[0]))]
    # per-call latency
    def lats(f):
        return pp([c["latency_ms"] or 0 for c in calls if f(c)])
    o["lat"] = {"all": lats(lambda c: True), "all_ok": lats(lambda c: not c["error"]),
                "slm": lats(lambda c: c["model_role"] == "slm"), "slm_ok": lats(lambda c: c["model_role"] == "slm" and not c["error"]),
                "api": lats(lambda c: c["model_role"] != "slm"), "api_ok": lats(lambda c: c["model_role"] != "slm" and not c["error"]),
                "null_latency": sum(c["latency_ms"] is None for c in calls)}
    o["errors"] = sum(c["error"] for c in calls)
    o["retries"] = sum(c["attempt"] > 1 for c in calls)
    # sessions
    sessions = man.get("sessions") or [{"started_at": man["started_at"], "finished_at": man["finished_at"], "workers": man.get("workers")}]
    sess = [(ts(s["started_at"]), ts(s["finished_at"]), s.get("workers"), bool(s.get("killed"))) for s in sessions]
    sess_s = sum(b - a for a, b, _, _ in sess)
    o["sessions"] = {"n": len(sess), "wall_s": sess_s, "workers": [w for _, _, w, _ in sess], "killed": [k for *_, k in sess],
                     "each_s": [b - a for a, b, _, _ in sess], "final_s": sess[-1][1] - sess[-1][0]}
    # question wall
    q_or = defaultdict(set)
    span = defaultdict(lambda: [1e18, 0.0, 0.0, 0])
    for c in calls:
        q_or[c["question_id"]].add(c["run_id"])
        sp = span[c["question_id"]]
        lat = (c["latency_ms"] or 0) / 1000
        sp[0] = min(sp[0], c["t0"]); sp[1] = max(sp[1], c["t0"] + lat); sp[2] += lat; sp[3] += 1
    o["q_multi_origin"] = sum(len(v) > 1 for v in q_or.values())
    o["q_with_calls"] = len(span)
    wall = {q: sp[1] - sp[0] for q, sp in span.items()}
    o["q_wall"] = pp(list(wall.values()))
    o["q_wall_mean"] = sum(wall.values()) / len(wall)
    o["q_sumlat"] = pp([sp[2] for sp in span.values()])
    o["q_calls"] = pp([sp[3] for sp in span.values()])
    # by worker count of the session the question was answered in
    def workers_of(t):
        for a, b, w, _ in sess:
            if a - 1 <= t <= b + 1:
                return w
        return None
    byw = defaultdict(list)
    for q, sp in span.items():
        byw[workers_of(sp[0])].append(wall[q])
    o["q_wall_by_workers"] = {str(k): pp(v) for k, v in byw.items()}
    # calls outside any session window
    o["calls_outside_sessions"] = sum(workers_of(c["t0"]) is None for c in calls)
    # per question cost p50/p95 (standard, 100%)
    key = "standard@100%" if slm else "standard"
    qc = [v[key] for v in by_q.values() if v.get(key) is not None]
    o["q_cost"] = pp(qc)
    o["q_cost_key"] = key
    # SLM capacity as run
    S = [c for c in calls if c["model_role"] == "slm"]
    if S:
        evs = []
        for c in S:
            lat = (c["latency_ms"] or 0) / 1000
            evs.append((c["t0"], 1)); evs.append((c["t0"] + lat, -1))
        evs.sort()
        cur = peak = 0; busy = 0.0; area = 0.0; prev = evs[0][0]
        for t, d in evs:
            if cur > 0:
                busy += t - prev
            area += cur * (t - prev)
            prev = t
            cur += d; peak = max(peak, cur)
        ok = [c for c in S if c["usage"].get("source") != "missing"]
        tin = sum(c["usage"]["input"] for c in ok); tout = sum(c["usage"]["output"] for c in ok)
        tc = sum(c["usage"].get("cached_input") or 0 for c in ok)
        ends = [c["t0"] + (c["latency_ms"] or 0) / 1000 for c in ok]
        cap = {"slm_role_calls": len(S), "slm_billed": len(ok), "in": tin, "out": tout, "cached": tc,
               "session_wall_s": sess_s, "busy_s": busy, "sum_latency_s": area, "peak_inflight": peak,
               "mean_inflight_wall": area / sess_s, "mean_inflight_busy": area / busy,
               "calls_per_s_wall": len(ok) / sess_s, "calls_per_s_busy": len(ok) / busy,
               "in_tok_per_s_wall": tin / sess_s, "out_tok_per_s_wall": tout / sess_s,
               "in_tok_per_s_busy": tin / busy, "out_tok_per_s_busy": tout / busy,
               "gpu_time_usd": sess_s / 3600 * GPU_H, "gpu_busy_usd": busy / 3600 * GPU_H}
        for w in (60, 300):
            cap[f"max_calls_per_s_{w}s"] = window_max(ends, [1] * len(ends), w) / w
            cap[f"max_in_tok_per_s_{w}s"] = window_max(ends, [c["usage"]["input"] for c in ok], w) / w
            cap[f"max_out_tok_per_s_{w}s"] = window_max(ends, [c["usage"]["output"] for c in ok], w) / w
        # final session only
        a, b = sess[-1][0], sess[-1][1]
        fin = [c for c in ok if a <= c["t0"] <= b]
        cap["final_session"] = {"s": b - a, "calls": len(fin), "calls_per_s": len(fin) / (b - a),
                                "in_tok_per_s": sum(c["usage"]["input"] for c in fin) / (b - a),
                                "out_tok_per_s": sum(c["usage"]["output"] for c in fin) / (b - a),
                                "mean_inflight": sum((c["latency_ms"] or 0) / 1000 for c in fin) / (b - a)}
        o["capacity"] = cap
    out[arm] = o
    json.dump(o, open(f"{OUT}/arm_{arm}.json", "w"), indent=1, default=str)
    print(arm, "done", res["calls"], res["total"], flush=True)
    del calls, by_q
