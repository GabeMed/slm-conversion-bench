import json, sys
ROOT="<repo>"
J8=json.load(open(f"{ROOT}/judgments/J8/de3754a76358b70edb33ff0b29f88616d84755923e8becf79cad333750ab5e66/result.json"))["result"]
A={a:json.load(open(f"arm_{a}.json")) for a in ["B0","B2-prod","B2-cheap","B3","B4","B5"]}
X=json.load(open("extra.json")); L=json.load(open("ledger.json"))
NAME={"B0":"B0","B2-prod":"B2, production LLM","B2-cheap":"B2, cheaper LLM","B3":"B3","B4":"B4","B5":"B5"}
N=498; GPU=1.79
SITES=["filter_column","select_tables","select_columns","generate_candidate","revise","extract_keywords","agent_ir","agent_ss","agent_cg"]
CL={"filter_column":"c0","select_tables":"c1","select_columns":"c1","generate_candidate":"c1","revise":"c1","extract_keywords":"c2","agent_ir":"c3","agent_ss":"c3","agent_cg":"c3"}
T={}
def tab(h,rows,align=None):
    s="| "+" | ".join(h)+" |\n|"+"|".join("---" for _ in h)+"|\n"
    return s+"\n".join("| "+" | ".join(str(x) for x in r)+" |" for r in rows)
def tot(a,k):
    t=A[a]["j3"]["total"]; return t.get(k)
def api(a,v="standard"):
    j=A[a]["j3"]["total"]
    if v in j: return j[v]
    # arm with SLM calls: API part = total@100% - slm@100%
    return sum(s["priced"]["api"][v] or 0 for s in A[a]["sites"].values())
def slm(a,u): return sum(s["priced"]["slm"].get(u) or 0 for s in A[a]["sites"].values())
def has_slm(a): return A[a]["j3"]["slm_calls"]>0
# T1
rows=[]
for a in A:
    j=A[a]["j3"]; tk=j["tokens"]
    rows.append([NAME[a], f"{j['n_correct']} / {N} = {100*j['n_correct']/N:.1f}%", f"{j['calls']:,}", f"{j['calls']/N:.1f}", f"{j['slm_calls']:,}" if has_slm(a) else "0",
                 f"{tk['input']/1e6:.2f}", f"{100*tk['cached_input']/tk['input']:.0f}%", f"{tk['output']/1e6:.3f}", f"{tk['input']/N:,.0f}", f"{tk['output']/N:,.0f}", j["failed_unbilled"]])
T["ARMS_TOKENS"]=tab(["Arm","EX","Calls","Calls per question","Of which on the SLM (priced)","Input tokens, millions","Of which read from cache","Output tokens, millions","Input tokens per question","Output tokens per question","Failed calls, not priced"],rows)
rows=[]
for a in A:
    g=A[a].get("capacity")
    rows.append([NAME[a], f"{api(a):.2f}", f"{api(a,'no_cache'):.2f}",
                 " / ".join(f"{slm(a,u):.2f}" for u in ("20%","50%","100%")) if has_slm(a) else "–",
                 f"{g['gpu_time_usd']:.2f}" if g else "–", f"{g['gpu_busy_usd']:.2f}" if g else "–"])
T["ARMS_COST"]=tab(["Arm","API, list price, with caching","API, list price, no caching","SLM serving, load-test price at 20% / 50% / 100% utilisation","SLM serving, GPU time as run (session wall time)","SLM serving, GPU time with a call in flight"],rows)
rows=[]
for a in A:
    j=A[a]["j3"]; n=j["n_correct"]; g=A[a].get("capacity")
    if has_slm(a):
        c={u:api(a)+slm(a,u) for u in ("20%","50%","100%")}; gt=api(a)+g["gpu_time_usd"]
        rows.append([NAME[a], f"{100*n/N:.1f}%", " / ".join(f"{c[u]/N:.4f}" for u in c), f"{gt/N:.4f}", " / ".join(f"{c[u]/n:.4f}" for u in c), f"{gt/n:.4f}", f"{api(a,'no_cache')/N+slm(a,'100%')/N:.4f}"])
    else:
        rows.append([NAME[a], f"{100*n/N:.1f}%", f"{api(a)/N:.4f}", "–", f"{api(a)/n:.4f}", "–", f"{api(a,'no_cache')/N:.4f}"])
T["ARMS_PER"]=tab(["Arm","EX","Cost per question (SLM arms: at 20% / 50% / 100% utilisation)","Cost per question, GPU time as run","Cost per correct answer (SLM arms: at 20% / 50% / 100%)","Cost per correct answer, GPU time as run","Cost per question without caching (SLM at 100%)"],rows)
# clusters
def cluster_rows(a):
    S=A[a]["sites"]; j=A[a]["j3"]; total100=api(a)+slm(a,"100%"); total20=api(a)+slm(a,"20%"); tt=j["tokens"]["input"]+j["tokens"]["output"]
    agg={}
    for s,v in S.items():
        p=v["priced"]; g=agg.setdefault(CL[s],dict(calls=0,slm=0,tok=0,c100=0,c20=0,api=0))
        g["calls"]+=v["calls"]; g["slm"]+=p["slm_calls"]; g["tok"]+=p["tokens"]["input"]+p["tokens"]["output"]
        g["api"]+=p["api"]["standard"]; g["c100"]+=p["api"]["standard"]+(p["slm"].get("100%") or 0); g["c20"]+=p["api"]["standard"]+(p["slm"].get("20%") or 0)
    return agg,total100,total20,tt,j["calls"]
rows=[]
for a in ("B0","B4","B5"):
    agg,t100,t20,tt,nc=cluster_rows(a)
    for cl in sorted(agg):
        g=agg[cl]
        rows.append([a,cl,f"{g['calls']/N:.2f}",f"{100*g['calls']/nc:.1f}%",f"{100*g['slm']/g['calls']:.0f}%",f"{100*g['tok']/tt:.1f}%",
                     f"{g['c100']/N:.5f}",f"{100*g['c100']/t100:.1f}%", (f"{g['c20']/N:.5f}" if has_slm(a) else "–"), (f"{100*g['c20']/t20:.1f}%" if has_slm(a) else "–")])
T["CLUSTERS"]=tab(["Arm","Cluster (by call site)","Calls per question","Share of calls","Served by the SLM","Share of tokens","Cost per question (SLM at 100%)","Share of cost (SLM at 100%)","Cost per question (SLM at 20%)","Share of cost (SLM at 20%)"],rows)
rows=[]
for a in ("B0","B4","B5"):
    S=A[a]["sites"]; t100=api(a)+slm(a,"100%"); j=A[a]["j3"]; tt=j["tokens"]["input"]+j["tokens"]["output"]
    for s in SITES:
        v=S[s]; p=v["priced"]; c100=p["api"]["standard"]+(p["slm"].get("100%") or 0)
        rows.append([a,CL[s],f"`{s}`",f"{v['calls']/N:.2f}",f"{100*v['calls']/j['calls']:.1f}%",f"{100*p['slm_calls']/v['calls']:.0f}%",f"{p['tokens']['input']/N:,.0f}",
                     f"{100*p['tokens']['cached_input']/p['tokens']['input']:.0f}%",f"{p['tokens']['output']/N:,.0f}",f"{100*(p['tokens']['input']+p['tokens']['output'])/tt:.1f}%",f"{c100/N:.5f}",f"{100*c100/t100:.1f}%"])
T["SITES"]=tab(["Arm","Cluster","Call site","Calls per question","Share of calls","Served by the SLM","Input tokens per question","Read from cache","Output tokens per question","Share of tokens","Cost per question (SLM at 100%)","Share of cost"],rows)
# B5 vs B0 by cluster
a0,_,_,_,_=cluster_rows("B0"); a5,_,_,_,_=cluster_rows("B5")
rows=[]
for cl in sorted(a0):
    rows.append([cl,f"{a0[cl]['c100']:.2f}",f"{a5[cl]['api']:.2f}",f"{a5[cl]['c100']-a5[cl]['api']:.2f} / {a5[cl]['c20']-a5[cl]['api']:.2f}",f"{a0[cl]['c100']-a5[cl]['c100']:.2f} / {a0[cl]['c100']-a5[cl]['c20']:.2f}"])
b0=api("B0"); 
rows.append(["Total",f"{b0:.2f}",f"{api('B5'):.2f}",f"{slm('B5','100%'):.2f} / {slm('B5','20%'):.2f}",f"{b0-api('B5')-slm('B5','100%'):.2f} / {b0-api('B5')-slm('B5','20%'):.2f}"])
T["B5_REMOVES"]=tab(["Cluster (by call site)","B0, API (US$)","B5, API (US$)","B5, SLM serving at 100% / 20% (US$)","B0 minus B5 at 100% / 20% (US$)"],rows)
T["B5_REMOVES_TXT"]=json.dumps({"api_only":1-api('B5')/b0,"100":1-(api('B5')+slm('B5','100%'))/b0,"50":1-(api('B5')+slm('B5','50%'))/b0,"20":1-(api('B5')+slm('B5','20%'))/b0,
   "gpu":1-(api('B5')+A['B5']['capacity']['gpu_time_usd'])/b0,"sens20":1-A['B5']['j3']['total']['standard@20%@sensitivity']/b0,"repl":A['B5']['j3']['replaceable_fraction'],
   "nocache100":1-(A['B5']['j3']['total']['no_cache@100%'])/A['B0']['j3']['total']['no_cache'],"batch100":1-A['B5']['j3']['total']['batch@100%']/A['B0']['j3']['total']['batch'],"batch20":1-A['B5']['j3']['total']['batch@20%']/A['B0']['j3']['total']['batch']})
# latency
def s(x,d=1): return "–" if x is None else f"{x/1000:.{d}f}"
rows=[]
for a in A:
    o=A[a]; ss=o["sessions"]; w=[]
    for x in ss["workers"]:
        if x not in w: w.append(x)
    bw=o["q_wall_by_workers"]
    rows.append([NAME[a], ss["n"], " / ".join(str(x) for x in w), f"{ss['wall_s']/60:.0f}", f"{o['q_wall']['p50']:.1f} / {o['q_wall']['p95']:.1f}",
                 "; ".join(f"{k} workers: {v['p50']:.1f} / {v['p95']:.1f} (n = {v['n']})" for k,v in bw.items()) if len(bw)>1 else "–",
                 f"{o['q_sumlat']['p50']:.0f}"])
T["LAT_Q"]=tab(["Arm","Sessions","Workers (questions at once), by session","Session wall time, minutes","Seconds per question, p50 / p95","Same, by the worker count of the session that answered the question","Sum of the question's call latencies, p50, seconds"],rows)
rows=[]
for a in A:
    l=A[a]["lat"]
    f=lambda k: f"{s(l[k]['p50'],2)} / {s(l[k]['p95'],2)}" if l[k]["n"] else "–"
    rows.append([NAME[a], f"{l['all']['n']:,}", f("all"), f("slm"), f("api"), A[a]["errors"]])
T["LAT_CALL"]=tab(["Arm","Calls","All calls, p50 / p95 (s)","SLM calls, p50 / p95 (s)","API calls, p50 / p95 (s)","Calls that ended in an error (included)"],rows)
# SLM latency vs limit
rows=[]
def lim_of(arm,e):
    return J8["as_deployed"]["slm:qwen3-8b+as-deployed"]["p95_slo_ms"] if arm=="B4" else J8["engines"][e]["p95_slo_ms"]
for a in ("B3","B4","B5"):
    x=X[a]
    if a in("B3","B4"):
        l=A[a]["lat"]["slm"]; e=list(x["engines"])[0]; L_=lim_of(a,e)
        so=x["all_over"]
        rows.append([a,"all SLM calls ("+("base sweep" if a=="B3" else "as-deployed sweep")+")",f"{l['n']:,}",f"{l['p50']/1000:.2f}",f"{l['p95']/1000:.2f}",f"{L_/1000:.2f}",f"{l['p95']/L_:.1f}x",f"{100*so:.0f}%"])
    else:
        for e,v in sorted(x["engines"].items()):
            rows.append([a,f"adapter {e.split('lora:')[1].split('-')[0]} (its own sweep)",f"{v['n']:,}",f"{v['p50']/1000:.2f}",f"{v['p95']/1000:.2f}",f"{v['limit']/1000:.2f}",f"{v['p95']/v['limit']:.1f}x",f"{100*v['share_over']:.0f}%" if v['share_over']>=0.01 else f"{100*v['share_over']:.1f}%"])
T["SLM_LIMIT"]=tab(["Arm","Traffic priced by one sweep","SLM calls","As run, p50 (s)","As run, p95 (s)","p95 limit of the sweep (s)","As-run p95 over the limit","Calls slower than the limit"],rows)
rows=[]
G={"filter_column":"filter","select_tables":"selection","select_columns":"selection","generate_candidate":"sql","revise":"sql","extract_keywords":"keywords","agent_ir":"agent","agent_ss":"agent","agent_cg":"agent"}
for a in ("B3","B4","B5"):
    for st in SITES:
        v=A[a]["sites"][st]["lat_slm"]
        if not v["n"]: continue
        tp=X["teacher_group_p95"][G[st]]
        rows.append([a,f"`{st}`",f"{v['n']:,}",f"{v['p50']/1000:.2f}",f"{v['p95']/1000:.2f}",f"{tp/1000:.2f}","over" if v["p95"]>tp else "under"])
T["SLM_SITES"]=tab(["Arm","Call site","SLM calls","As run, p50 (s)","As run, p95 (s)","Production LLM's p95 on the same group of call sites, calibration (s)","As-run p95 against it"],rows)
T["B4_WORKERS"]=json.dumps({a:X[a]["by_workers"] for a in ("B3","B4","B5")})
# capacity
rows=[]
for a in ("B3","B4","B5"):
    c=A[a]["capacity"]; x=X[a]
    rows.append([a,f"{c['slm_role_calls']:,}",f"{c['session_wall_s']/3600:.2f}",f"{c['busy_s']/3600:.2f}",f"{c['calls_per_s_wall']:.2f}",f"{c['calls_per_s_busy']:.2f}",f"{c['max_calls_per_s_300s']:.2f}",f"{c['max_calls_per_s_60s']:.2f}",
                 f"{c['in_tok_per_s_busy']:,.0f}",f"{c['out_tok_per_s_busy']:,.0f}",f"{c['max_in_tok_per_s_300s']:,.0f}",f"{c['max_out_tok_per_s_300s']:,.0f}",f"{100*c['cached']/c['in']:.0f}%",
                 f"{c['mean_inflight_busy']:.1f}",f"{x['inflight_busy_p50']} / {x['inflight_busy_p95']}",c["peak_inflight"],f"{N/(c['session_wall_s']/3600):.0f}"])
T["CAPACITY"]=tab(["Arm","SLM calls","Session wall time (h)","Time with a call in flight (h)","Calls/s over session wall time","Calls/s while a call was in flight","Best 5 minutes, calls/s","Best minute, calls/s",
  "Input tokens/s while in flight","Output tokens/s while in flight","Best 5 minutes, input tokens/s","Best 5 minutes, output tokens/s","Input read from the prefix cache","Calls in flight, mean","Calls in flight, p50 / p95 over time","Calls in flight, peak","Questions per hour over session wall time"],rows)
rows=[]
for nme,e in list(J8["engines"].items()):
    su=e.get("sustained") or {}
    best=max(e["levels"],key=lambda l:l["throughput_rps"])
    label={"slm:qwen3-8b":"base model, B3's calls","slm:qwen3-8b+as-deployed":"four adapters on one server, B4's calls"}.get(nme, "adapter "+nme.split("lora:")[-1].split("-")[0]+" alone")
    cpr=e.get("cost_per_request")
    rows.append([label,f"{e['p95_slo_ms']/1000:.2f}",su.get("concurrency","none"),f"{su['throughput_rps']:.2f}" if su else "–",f"{e['requests_per_hour']:,.0f}" if e.get("requests_per_hour") else "–",
                 " / ".join(f"{cpr[u]*1000:.4f}" for u in ("20%","50%","100%")) if cpr else "no cost (outside the limit at every level)",
                 f"{best['throughput_rps']:.2f} at concurrency {best['concurrency']} (p95 {best['p95_ms']/1000:.1f} s)"])
T["J8"]=tab(["Sweep (calibration traffic)","p95 limit (s)","Highest concurrency within the limit","Requests/s there","Requests per hour there","US$ per 1,000 requests at 20% / 50% / 100% utilisation","Highest requests/s at any level"],rows)
# break-even
def cap(a):
    if a=="B4": return J8["as_deployed"]["slm:qwen3-8b+as-deployed"]["requests_per_hour"]/(A[a]["j3"]["calls"]/N)
    if a=="B3": return J8["engines"]["slm:qwen3-8b"]["requests_per_hour"]/(A[a]["j3"]["calls"]/N)
    c0=A[a]["sites"]["filter_column"]["priced"]["slm_calls"]/N
    return J8["engines"]["slm:qwen3-8b+lora:c0-842db723c47e"]["requests_per_hour"]/c0
rows=[]; BE={}
for slm_arm in ("B5","B4"):
    for comp in ("B0","B2-prod","B2-cheap"):
        before=api(comp)/N; after=api(slm_arm)/N; sv=before-after; c=cap(slm_arm)
        dEX=100*(A[slm_arm]["j3"]["n_correct"]-A[comp]["j3"]["n_correct"])/N
        if sv<=0: be="never (the SLM arm's API calls alone cost more)"; day="–"; ut="–"; sens="–"
        else:
            q=GPU/sv; be=f"{q:,.1f}" if q<100 else f"{q:,.0f}"; day=f"{q*24:,.0f}"; ut=f"{100*q/c:.1f}%" if q<=c else f"{100*q/c:,.0f}%: above the capacity of one GPU"; sens=f"{3.131424/sv:,.1f}" if 3.131424/sv<100 else f"{3.131424/sv:,.0f}"
            BE[f"{slm_arm}|{comp}"]=q
        rows.append([f"{NAME[slm_arm]} in place of {NAME[comp]}",f"{dEX:+.1f}".replace("-","−"),f"{before:.4f}",f"{after:.4f}",f"{sv:.4f}".replace("-","−"),be,day,f"{c:,.0f}",ut,sens])
T["BREAKEVEN"]=tab(["Comparison","EX difference, points","API cost per question before (US$)","API cost per question after (US$)","API saving per question (US$)","Break-even, questions per hour","Break-even, questions per day (24 h)","Questions per hour one GPU sustains within the latency limit","Break-even as a share of that capacity","Break-even at the sensitivity price (US$ 3.13/h), questions per hour"],rows)
# ledger
g=L["groups"]
def grp(keys): 
    ks=[k for k in g if any(k==x or k.startswith(x) for x in keys)]
    return ks
CATS=[("Production-LLM traces on the training questions (the final collection run, its interrupted attempts and the earlier probes)",["_invalid/agent-B0-train","agent-B0-train"]),
      ("Single-call probes on training questions",["agent-B2-cheap_alt-train","agent-B2-production_llm-train"]),
      ("Calibration: agent on the production LLM",["agent-B0-calib"]),
      ("Calibration: single call, cheaper LLM",["agent-B2-cheap_alt-calib"]),
      ("Calibration: replays on the cheaper LLM",["replay-cheap_alt-calib"]),
      ("Calibration: replay on the production LLM",["replay-production_llm-calib"]),
      ("Calibration: SLM agents and replays (B3, B4, two SLM candidates)",["agent-B3-calib","agent-B4-calib","replay-B4-calib","replay-slm_granite-4.2-8b-calib","replay-slm_qwen3-8b-calib"]),
      ("Test: B0, every attempt",["agent-B0-test"]),
      ("Test: B2, production LLM",["agent-B2-production_llm-test"]),
      ("Test: B2, cheaper LLM",["agent-B2-cheap_alt-test"]),
      ("Test: B3 and B4 (SLM only)",["agent-B3-test","agent-B4-test"]),
      ("Test: B5, every attempt",["agent-B5-test"]),
      ("Test: replay on the production LLM (100 questions)",["replay-production_llm-test"]),
      ("Test: replays on the SLM",["replay-B4-test","replay-slm_qwen3-8b-test"])]
rows=[]; used=set(); T_=dict(std=0,new=0,slm=0,dirs=0)
for name,keys in CATS:
    ks=[k for k in keys if k in g]; used|=set(ks)
    d=sum(g[k]["dirs"] for k in ks); new=sum(g[k]["new"] for k in ks); sl=sum(g[k]["slm"] for k in ks); i=sum(g[k]["inp"] for k in ks); o=sum(g[k]["out"] for k in ks); st=sum(g[k]["std"] for k in ks)
    T_["std"]+=st;T_["new"]+=new;T_["slm"]+=sl;T_["dirs"]+=d
    rows.append([name,d,f"{new:,}",f"{sl:,}",f"{i/1e6:,.1f}",f"{o/1e6:,.2f}",f"{st:,.2f}"])
assert used==set(g), set(g)-used
rows.append(["**Total, B1 excluded**",T_["dirs"],f"{T_['new']:,}",f"{T_['slm']:,}","","",f"**{T_['std']:,.2f}**"])
T["LEDGER_API"]=tab(["Category","Run directories","Distinct calls","Of which on the SLM","Input tokens, millions","Output tokens, millions","API cost at list price (US$)"],rows)
T["LEDGER_TXT"]=json.dumps({"unknown":g["agent-B0-train"]["unknown"],"skipped":L["skipped_b1_dirs"],"at":L["computed_at"],"batch_dirs":g["agent-B0-train"]["batch_dirs"],"flag":g["agent-B0-train"]["flag_dirs"],
   "test_api":sum(g[k]["std"] for k in g if k.endswith("-test")), "calib_api":sum(g[k]["std"] for k in g if k.endswith("-calib")), "train_api":sum(g[k]["std"] for k in g if k.endswith("-train"))})
rows=[]; ph=pc=0
import datetime as dt
for p in L["pods"]:
    if p["hours"] is None: continue
    ph+=p["hours"]; pc+=p["usd"]
    use={"ffy86d4gfktfpv":"download of the weights","6n8mywpcubaivr":"the second SLM candidate","05qx1swv1oq2sn":"the second SLM candidate","9mg4iongeb1lxi":"base-model replays and the B3 pilot on calibration",
         "kw329b06mou4yh":"a check with one adapter loaded","86ovr3x89eh3bx":"B3 and B4 on calibration, the load tests behind J8, B3 and B4 on the test","sg5e07uhpcomz6":"B5 on the test, session 1","ejuv1ctuz0z2lw":"B5 on the test, sessions 2 to 4",
         "92e3cqaubrafmj":"B5 on the test, sessions 5 to 11","5z556nr7hs2ql2":"replays on the test"}[p["id"]]
    rows.append([p["created"][:16].replace("T"," "),p["terminated"][:16].replace("T"," "),f"{p['hours']:.2f}",f"{p['usd']:.2f}",use])
rows.append(["**Total**","",f"**{ph:.2f}**",f"**{pc:.2f}**",""])
T["PODS"]=tab(["Created (UTC)","Terminated (UTC)","Hours","US$ at 1.79/h","What ran on it"],rows)
tr=L["training"]; rows=[]
for c,v in tr.items():
    rows.append([c,f"{v['rows']:,}",f"{v['tokens']/1e6:.2f}",f"{v['tokens_seen']/1e6:.2f}",v["steps"],f"{v['train_seconds']/60:.1f}","not recorded (null)"])
rows.append(["**Total**",f"{sum(v['rows'] for v in tr.values()):,}",f"{sum(v['tokens'] for v in tr.values())/1e6:.2f}",f"{sum(v['tokens_seen'] for v in tr.values())/1e6:.2f}",sum(v['steps'] for v in tr.values()),f"{sum(v['train_seconds'] for v in tr.values())/60:.1f}","not recorded"])
T["TRAIN"]=tab(["Adapter","Training rows","Dataset tokens, millions","Tokens seen in 2 epochs, millions","Steps","Training time, minutes","`cost_usd` in the manifest"],rows)
r=L["routes"]; rows=[]
for a,v in r.items():
    rows.append([a,f"{v['n']:,}",f"{v['embed_p50']:.0f} / {v['embed_p95']:.0f}",f"{v['wait_p50']:.0f} / {v['wait_p95']:,.0f}",f"{v['route_p50']:.0f} / {v['route_p95']:,.0f}",f"{v['route_s_per_q']:.0f}",f"{v['wait_s_per_q']:.0f}"])
T["ROUTER"]=tab(["Arm","Routed invocations","Embedding, p50 / p95 (ms)","Wait for the embedding model, p50 / p95 (ms)","Whole route, p50 / p95 (ms)","Route time summed per question (s)","Of which waiting (s)"],rows)
json.dump(T,open("tables.json","w"),indent=1)
if len(sys.argv)>1:
    doc=open(sys.argv[1]).read()
    import re
    miss=[m for m in re.findall(r"\{\{(\w+)\}\}",doc) if m not in T]
    assert not miss, miss
    for k,v in T.items(): doc=doc.replace("{{"+k+"}}",v)
    open(sys.argv[2],"w").write(doc)
else:
    for k,v in T.items(): print("##",k); print(v); print()
    print(BE)
