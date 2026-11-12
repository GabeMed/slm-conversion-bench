import json
from datetime import datetime
from collections import defaultdict
ROOT="<repo>"
J8=json.load(open(f"{ROOT}/judgments/J8/de3754a76358b70edb33ff0b29f88616d84755923e8becf79cad333750ab5e66/result.json"))["result"]
RUNS={"B3":"agent-B3-test-20261111T012223.392984Z","B4":"agent-B4-test-20261111T032707.693347Z","B5":"agent-B5-test-20261111T200809.693765Z"}
GROUP={"filter_column":"filter","select_tables":"selection","select_columns":"selection","generate_candidate":"sql","revise":"sql","extract_keywords":"keywords","agent_ir":"agent","agent_ss":"agent","agent_cg":"agent"}
TEACH={k:v["p95_ms"] for k,v in J8["engines"]["slm:qwen3-8b"]["by_group"]["teacher"].items()}
def ts(s): return datetime.fromisoformat(s).timestamp()
def pct(xs,p):
    if not xs: return None
    xs=sorted(xs);k=(len(xs)-1)*p;f=int(k);c=min(f+1,len(xs)-1);return xs[f]+(xs[c]-xs[f])*(k-f)
out={"teacher_group_p95":TEACH}
for arm,rid in RUNS.items():
    man=json.load(open(f"{ROOT}/runs/{rid}/manifest.json"))
    sess=[(ts(s["started_at"]),ts(s["finished_at"]),s.get("workers")) for s in (man.get("sessions") or [{"started_at":man["started_at"],"finished_at":man["finished_at"],"workers":man.get("workers")}])]
    def lim(engine):
        if arm=="B4": return J8["as_deployed"]["slm:qwen3-8b+as-deployed"]["p95_slo_ms"]
        return J8["engines"][engine]["p95_slo_ms"]
    eng=defaultdict(list); site=defaultdict(list); byw=defaultdict(list); evs=[]; over=defaultdict(lambda:[0,0]); grp=defaultdict(list)
    with open(f"{ROOT}/runs/{rid}/calls.jsonl") as f:
        for line in f:
            c=json.loads(line)
            if c["model_role"]!="slm": continue
            lat=c.get("latency_ms") or 0; t0=ts(c["started_at"]); e=c["engine"]; s=c["call_site"]
            del c,line
            eng[e].append(lat); site[(s,e)].append(lat); grp[GROUP[s]].append(lat)
            w=next((w for a,b,w in sess if a-1<=t0<=b+1),None); byw[w].append(lat)
            L=lim(e); o=over[e]; o[0]+=1; o[1]+=lat>L
            evs.append((t0,1)); evs.append((t0+lat/1000,-1))
    evs.sort(); cur=0; prev=evs[0][0]; hist=defaultdict(float)
    for t,d in evs:
        if cur>0: hist[cur]+=t-prev
        prev=t; cur+=d
    tot=sum(hist.values()); acc=0; q={}
    for k in sorted(hist):
        acc+=hist[k]
        for p in (.5,.95):
            if p not in q and acc>=p*tot: q[p]=k
    o={"engines":{e:{"n":len(v),"p50":pct(v,.5),"p95":pct(v,.95),"limit":lim(e),"share_over":over[e][1]/over[e][0]} for e,v in eng.items()},
       "site_engine":{f"{s}|{e}":{"n":len(v),"p50":pct(v,.5),"p95":pct(v,.95)} for (s,e),v in site.items()},
       "groups":{g:{"n":len(v),"p50":pct(v,.5),"p95":pct(v,.95),"teacher_p95":TEACH[g]} for g,v in grp.items()},
       "by_workers":{str(w):{"n":len(v),"p50":pct(v,.5),"p95":pct(v,.95)} for w,v in byw.items()},
       "inflight_busy_p50":q[.5],"inflight_busy_p95":q[.95],"all_over":sum(x[1] for x in over.values())/sum(x[0] for x in over.values())}
    out[arm]=o; print(arm,json.dumps(o,indent=0)[:6000],flush=True)
# K1 final run at list price
P={"claude-sonnet-5-5":dict(i=2.0,c=0.20,w=4.0,o=10.0)}
rid="agent-B0-train-20261109T214533.609401Z"
man=json.load(open(f"{ROOT}/runs/{rid}/manifest.json"))
n=0;cost=0;inp=0;outp=0;qs=set();other=0
with open(f"{ROOT}/runs/{rid}/calls.jsonl") as f:
    for line in f:
        c=json.loads(line);u=c["usage"];n+=1;qs.add(c["question_id"])
        if u.get("source")=="missing": continue
        p=P.get(c["model"])
        if not p: other+=1; continue
        i=u.get("input") or 0;ca=u.get("cached_input") or 0;w=u.get("cache_write") or 0;o=u.get("output") or 0
        inp+=i;outp+=o;cost+=((i-ca-w)*p["i"]+ca*p["c"]+w*p["w"]+o*p["o"])/1e6
        del c,line
ss=man.get("sessions") or []
out["K1"]={"run":rid,"calls":n,"questions":len(qs),"in":inp,"out":outp,"list":cost,"other":other,"sessions":len(ss),"session_h":sum(ts(s["finished_at"])-ts(s["started_at"]) for s in ss)/3600,"status":man.get("status"),"n_q":len(man.get("question_ids") or [])}
print(out["K1"])
json.dump(out,open("extra.json","w"),indent=1)
