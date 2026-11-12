"""Ledger of every logged model call outside the *-B1-* directories (never opened), de-duplicated by call_id
in directory-name order, at list price. Adapted from the reviewer's ledger.py: list price only, unpriced models
named, partial last lines tolerated, pods still running listed apart. Streams line by line."""
import json, os, glob, re, hashlib
from datetime import datetime
from collections import defaultdict

ROOT = "<repo>"
OUT = "<scratch>"
PRICES = {"claude-sonnet-5-5": dict(i=2.0, c=0.20, w=4.0, o=10.0), "openai/gpt-oss-120b": dict(i=0.03, c=0.03, w=0.0, o=0.17)}


def pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


seen = set()
G = defaultdict(lambda: dict(dirs=0, lines=0, new=0, std=0.0, nocache=0.0, inp=0, cached=0, out=0, slm=0, unbilled=0,
                             unknown=defaultdict(lambda: [0, 0, 0]), batch_dirs=0, flag_dirs=0, bad_lines=0, first=None, last=None))
dirs = sorted(glob.glob(f"{ROOT}/runs/*/calls.jsonl") + glob.glob(f"{ROOT}/runs/_invalid/*/calls.jsonl"))
skipped = 0
for path in dirs:
    d = os.path.dirname(path); name = os.path.basename(d)
    if "-B1-" in name or name.endswith("-B1") or "B1" in name.split("-")[:3]:
        skipped += 1
        continue
    try:
        man = json.load(open(d + "/manifest.json"))
    except Exception:
        man = {}
    key = re.sub(r"-\d{8}T[\d.]+Z.*$", "", name)
    if "_invalid" in d:
        key = "_invalid/" + key
    g = G[key]; g["dirs"] += 1
    g["batch_dirs"] += bool(man.get("batch")); g["flag_dirs"] += bool(man.get("via_claude_code"))
    with open(path) as f:
        for line in f:
            try:
                c = json.loads(line)
            except Exception:
                g["bad_lines"] += 1
                continue
            cid = c["call_id"]; u = c["usage"]; role = c["model_role"]; model = c["model"]; st = c.get("started_at")
            del line, c
            g["lines"] += 1
            h = int.from_bytes(hashlib.blake2b(cid.encode(), digest_size=8).digest(), "big")
            if h in seen:
                continue
            seen.add(h); g["new"] += 1
            if st:
                g["first"] = st if g["first"] is None or st < g["first"] else g["first"]
                g["last"] = st if g["last"] is None or st > g["last"] else g["last"]
            if u.get("source") == "missing":
                g["unbilled"] += 1
                continue
            inp = u.get("input") or 0; ca = u.get("cached_input") or 0; w = u.get("cache_write") or 0; o = u.get("output") or 0
            g["inp"] += inp; g["out"] += o; g["cached"] += ca
            if role == "slm":
                g["slm"] += 1; continue
            p = PRICES.get(model)
            if p is None:
                x = g["unknown"][model]; x[0] += 1; x[1] += inp; x[2] += o
                continue
            g["std"] += ((inp - ca - w) * p["i"] + ca * p["c"] + w * p["w"] + o * p["o"]) / 1e6
            g["nocache"] += (inp * p["i"] + o * p["o"]) / 1e6
    print(key, name, g["lines"], g["new"], round(g["std"], 2), flush=True)

res = {"groups": {k: {**{kk: vv for kk, vv in v.items() if kk != "unknown"}, "unknown": {m: x for m, x in v["unknown"].items()}} for k, v in G.items()},
       "skipped_b1_dirs": skipped, "computed_at": datetime.now().astimezone().isoformat()}

st = json.load(open(f"{ROOT}/.runpod/state.json"))
pods = []
for p in st["terminated"] + list(st.get("pods", {}).values()):
    c = datetime.fromisoformat(p["created_at"]); t = p.get("terminated_at")
    h = (datetime.fromisoformat(t) - c).total_seconds() / 3600 if t else None
    pods.append({"id": p["id"], "role": p.get("role"), "candidate": p.get("candidate"), "created": p["created_at"], "terminated": t,
                 "hours": h, "usd_h": p.get("cost_per_hour"), "usd": h * p.get("cost_per_hour", 0) if h is not None else None,
                 "adapters": len(p.get("adapters") or {})})
res["pods"] = pods
res["volume"] = {k: v for k, v in (st.get("volume") or {}).items() if k in ("size_gb", "sizeGb", "size", "created_at", "datacenter", "dataCenterId")}

tr = {}
for c in "0123":
    m = json.load(open(f"{ROOT}/train/adapters/c{c}/manifest.json")); s = m["stats"]
    tr[f"c{c}"] = {"rows": m["dataset"]["rows"], "tokens": s["tokens"], "tokens_seen": s["tokens_seen"], "completion_tokens": s["completion_tokens"],
                   "steps": s["global_step"], "train_seconds": s["train_seconds"], "tok_s": s["tokens_per_second"],
                   "cost_usd": m.get("cost_usd"), "gpu": m.get("gpu"), "gpu_seconds": m.get("gpu_seconds"), "backend": s.get("backend"),
                   "started_at": m.get("started_at"), "finished_at": m.get("finished_at")}
res["training"] = tr

rt = {}
for arm, rid in (("B4", "agent-B4-test-20261111T032707.693347Z"), ("B5", "agent-B5-test-20261111T200809.693765Z")):
    r, e, w = [], [], []
    with open(f"{ROOT}/runs/{rid}/routes.jsonl") as f:
        for line in f:
            x = json.loads(line); r.append(x["route_ms"]); e.append(x["embed_ms"]); w.append(x["embed_wait_ms"])
    rt[arm] = {"n": len(r), "embed_p50": pct(e, .5), "embed_p95": pct(e, .95), "wait_p50": pct(w, .5), "wait_p95": pct(w, .95),
               "route_p50": pct(r, .5), "route_p95": pct(r, .95), "route_s_per_q": sum(r) / 1000 / 498, "wait_s_per_q": sum(w) / 1000 / 498,
               "embed_s_per_q": sum(e) / 1000 / 498}
res["routes"] = rt
json.dump(res, open(f"{OUT}/ledger.json", "w"), indent=1, default=str)
print("done")
