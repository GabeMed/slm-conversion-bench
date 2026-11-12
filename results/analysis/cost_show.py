import json
R=json.load(open("<repo>/.ref/reviews-20261112/reviewer3/summary.json"))
for arm in ["B0","B2-prod","B2-cheap","B3","B4","B5"]:
    o=json.load(open(f"arm_{arm}.json")); j=o["j3"]; r=R[arm]; t=r["tot"]
    print("=====",arm, o["evals"])
    print(" calls",j["calls"],t["calls"]," slm",j["slm_calls"],t["slm_calls"]," unbilled",j["failed_unbilled"],t["unbilled"]," ncorr",j["n_correct"],r["ncorr"], "label",j["label"],"lb",j["lower_bound"],"ub",j["upper_bound"],"est",j["estimated"])
    print(" tokens",j["tokens"]," rev in/cached/written/out",t["inp"],t["cached"],t["written"],t["out"])
    print(" total",{k:round(v,4) for k,v in j["total"].items() if v is not None and "sens" not in k})
    print(" sens",{k:round(v,4) for k,v in j["total"].items() if v is not None and "sens" in k and k.startswith("standard")})
    print(" rev api",round(t["api"],4),round(t["api_nocache"],4),{k:round(v,4) for k,v in r["slm_u"].items()}," gpu",round(r["gpu_cost"],4), "sess_s",round(r["sess_s"],1))
    print(" per_q",{k:round(v,5) for k,v in j["per_question"].items() if v is not None and "sens" not in k and k.startswith("standard")})
    print(" per_correct",{k:round(v,5) for k,v in j["per_correct"].items() if v is not None and "sens" not in k and k.startswith("standard")})
    print(" replaceable",j["replaceable_fraction"]); print(" basis",o["basis"] and {k:o["basis"][k] for k in("basis","rule","outside_slo")})
    print(" errors",o["errors"],t["errors"]," retries",o["retries"],t["retries"])
    print(" lat",{k:(v if isinstance(v,int) else {a:(round(b/1000,2) if b is not None and a!='n' else b) for a,b in v.items()}) for k,v in o["lat"].items()})
    print(" sessions",o["sessions"])
    print(" q_multi_origin",o["q_multi_origin"],"q_with_calls",o["q_with_calls"],"outside",o["calls_outside_sessions"])
    print(" q_wall",{a:round(b,1) for a,b in o["q_wall"].items()},"mean",round(o["q_wall_mean"],1)," sumlat",{a:round(b,1) for a,b in o["q_sumlat"].items()}," calls",o["q_calls"])
    print(" q_wall_by_workers",{k:{a:round(b,1) for a,b in v.items()} for k,v in o["q_wall_by_workers"].items()})
    print(" q_cost",o["q_cost_key"],{a:round(b,5) for a,b in o["q_cost"].items()})
    print(" engines",o["engines"])
    if "capacity" in o:
        print(" cap",{k:(round(v,3) if isinstance(v,float) else v) for k,v in o["capacity"].items()})
    for s,v in o["sites"].items():
        p=v["priced"]
        print(f"  {s:20s} {v['cluster']} calls {v['calls']:6d} slm {p['slm_calls']:6d} unb {p['failed_unbilled']:4d} err {v['errors']:4d} in {p['tokens']['input']:10d} cached {p['tokens']['cached_input']:10d} out {p['tokens']['output']:8d} api {p['api']['standard']:.3f} slm100 {(p['slm'].get('100%') or 0):.4f} slm20 {(p['slm'].get('20%') or 0):.4f} | all p50/p95 {v['lat_all']['p50']/1000:.2f}/{v['lat_all']['p95']/1000:.2f} slm {(v['lat_slm']['p50'] or 0)/1000:.2f}/{(v['lat_slm']['p95'] or 0)/1000:.2f} slm_ok {(v['lat_slm_ok']['p50'] or 0)/1000:.2f}/{(v['lat_slm_ok']['p95'] or 0)/1000:.2f} api {(v['lat_api']['p50'] or 0)/1000:.2f}/{(v['lat_api']['p95'] or 0)/1000:.2f}")
