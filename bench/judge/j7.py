"""J7 · S6, the allocation of B5 (SPEC §4 S6, §6.4; D15), written as the `allocation` fact.

Per cluster of the centroids, **the cheapest engine that passes on calib**, else the production LLM:
- the calls of a cluster that have gold (SQL generation and repair) pass by **non-inferiority with
  margin Δ/2** (J4, F2's: the engine is the candidate, the teacher the reference, the question the
  unit), when they number at least `allocation.min_calls`: J4, given the margin Δ/2, finds the engine
  non-inferior to the teacher on the calib pairs. **Δ is SPEC §6.4's**, J4's `margin(d, n, cap)` at
  n = the test's questions (498), and **d is the pilot's** (SPEC §6.4, §7.3): the discordance J4
  measures, on this cluster's calls of the pre-registered pilot questions (`bench.data.pilot_sample`),
  between the zero-shot replay of the candidate S4 chose (J6) and the teacher, both before any
  adapter exists. It is one margin per cluster, fixed before any engine is judged, never the
  discordance of the pairs being judged. A cluster with no pilot pairs, or whose Δ is above the cap
  (not testable), gets no verdict and stays with the production LLM;
- the calls without gold pass by **agreement with the teacher** of at least
  `thresholds.concordance_min`, over at least `allocation.min_calls` calls: a proxy, declared as
  such (D15), which supports no claim per cluster;
- a cluster whose calls mix both must pass both; a cluster with no calib call, or without enough
  of them, stays with `production_llm` (SPEC S6).

The engines are `cheap_alt` (a replay of the teacher's calib inputs on it) and `slm`, the cluster's
adapter (a replay routed as B4, so each call went to the adapter of the cluster the router assigned
it). **The cluster of a calib call is the router's own assignment**, as the B4 replay recorded it in
C1, never recomputed: it is the one B5 will make. The cheap_alt replay is grouped by the same
assignment, invocation by invocation.

**Cheapest**, per cluster: the mean cost per invocation on that cluster's calib calls, retries
included: J3's `standard` price for cheap_alt; J8's cost per request for the SLM, at the lowest
utilization of `cost.utilizations` (the most expensive SLM, so the SLM is preferred only when it is
cheaper even there).
"""
import hashlib
from typing import Any, Callable, Dict, List, Optional

from bench.contracts.facts import read_fact, write_fact
from bench.judge import j2, j3
from bench.judge.base import (Identity, JudgmentError, calls_of, canonical, invocations, n_boot, read_result, reference,
                              require_done, result_reference, write_result)

JUDGMENT = "J7"
ENGINES = ("cheap_alt", "slm")
NonInferiority = Callable[..., Dict[str, Any]]


def plain(value: Any) -> Any:
    """J4's output as plain JSON values (a numpy scalar has `.item()`)."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value.item() if hasattr(value, "item") else value


def passes(entry: Optional[Dict[str, Any]], noninferiority: NonInferiority, settings: Dict[str, Any],
           d_pilot: Optional[float] = None, margin: Optional[Callable[..., Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Whether one engine passes on one cluster, and why; `d_pilot` is the cluster's pilot discordance,
    `margin` J4's Δ for a discordance over n questions."""
    if entry is None:
        return {"passes": False, "why": "no calib call in this cluster"}
    verdict: Dict[str, Any] = {"passes": True, "why": []}
    if entry["gold"]:
        gold = entry["gold"]
        engine, teacher = gold["by_question"]["replay"], gold["by_question"]["teacher"]
        spec = plain(margin(d_pilot, settings["n_test"], settings["delta_cap_pp"])) if d_pilot is not None else None
        test = plain(noninferiority(engine, teacher, settings["delta_cap_pp"], settings["seed"], settings["n_boot"],
                                    margin=spec["delta"] / 2)) if spec and spec["testable"] else None
        ok = gold["n"] >= settings["min_calls"] and test is not None and bool(test["noninferior"])
        verdict["gold"] = {"n": gold["n"], "ex_engine": gold["ex_replay"], "ex_teacher": gold["ex_teacher"],
                           "j4": test, "delta_cluster": dict(spec or {}, d_pilot=d_pilot, n=settings["n_test"]),
                           "passes": bool(ok)}
        if not ok:
            verdict["why"].append("gold: fewer calls than allocation.min_calls" if gold["n"] < settings["min_calls"] else
                                  "gold: no pilot pairs in this cluster" if d_pilot is None else
                                  "gold: not testable (Δ above the cap)" if not spec["testable"] else
                                  "gold: not non-inferior at Δ/2")
    if entry["agreement"]:
        agreement = entry["agreement"]
        ok = agreement["n"] >= settings["min_calls"] and agreement["rate"] is not None \
            and agreement["rate"] >= settings["concordance_min"]
        verdict["agreement"] = {"n": agreement["n"], "rate": agreement["rate"], "passes": bool(ok), "proxy": True}
        if not ok:
            verdict["why"].append("agreement: fewer calls than allocation.min_calls" if agreement["n"] < settings["min_calls"]
                                  else "agreement: below thresholds.concordance_min")
    verdict["passes"] = not verdict["why"]
    verdict["why"] = "; ".join(verdict["why"]) or "passes"
    return verdict


def allocate(clusters: List[str], evidence: Dict[str, Dict[str, Dict[str, Any]]], costs: Dict[str, Dict[str, float]],
             noninferiority: NonInferiority, settings: Dict[str, Any], pilot: Dict[str, Optional[float]],
             margin: Callable[..., Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{cluster: {"engine", "order", "evidence", ...}}; `evidence` is {engine: J2 compare by cluster},
    `costs` is {cluster: {engine: mean cost per invocation}}, `pilot` is {cluster: d_pilot}. When more
    than one engine passes, the choice rests on the cost order alone (`cost_dependent`), and the SLM's
    cost is extrapolated from per-adapter load tests: the ratio between the two says how much."""
    out = {}
    for cluster in sorted(clusters):
        order = sorted(costs.get(cluster, {}), key=lambda e: (costs[cluster][e], e))
        verdicts = {engine: passes(evidence[engine].get(cluster), noninferiority, settings, pilot.get(cluster), margin)
                    for engine in ENGINES}
        passing = [engine for engine in order if verdicts[engine]["passes"]]
        chosen = passing[0] if passing else "production_llm"
        out[cluster] = {"engine": chosen, "order": order, "costs": costs.get(cluster, {}), "d_pilot": pilot.get(cluster),
                        "cost_dependent": len(passing) > 1,
                        "cost_ratio": costs[cluster][passing[1]] / costs[cluster][passing[0]]
                        if len(passing) > 1 and costs[cluster][passing[0]] else None,
                        "evidence": verdicts}
    return out


# ---------------------------------------------------------------- reading the executions

def router_clusters(b4_calls: List[dict], clusters: List[str]) -> Dict[Identity, str]:
    assigned = {}
    for identity, attempts in invocations(b4_calls).items():
        cluster = attempts[0]["cluster"]
        if cluster not in clusters:
            raise JudgmentError(f"the B4 replay assigned {identity} to {cluster!r}, not a cluster of the centroids")
        assigned[identity] = cluster
    return assigned


def mean_costs(calls: List[dict], cluster_of: Dict[Identity, str], prices: Dict[str, Any],
               slm_per_request: Optional[float]) -> Dict[str, float]:
    by_cluster: Dict[str, List[dict]] = {}
    for identity, attempts in invocations(calls).items():
        if identity in cluster_of:
            by_cluster.setdefault(cluster_of[identity], []).append(attempts)
    out = {}
    for cluster, groups in by_cluster.items():
        flat = [a for attempts in groups for a in attempts]
        if slm_per_request is not None:
            out[cluster] = slm_per_request * len(flat) / len(groups)
        else:
            out[cluster] = j3.price_calls(flat, prices)["api"]["standard"] / len(groups)
    return out


def pilot_discordance(zeroshot: Dict[str, Dict[str, Any]], noninferiority: NonInferiority, settings: Dict[str, Any],
                      pilot_ids: List[str]) -> Dict[str, Optional[float]]:
    """{cluster: J4's discordance between the zero-shot candidate and the teacher on its gold calls of
    the pilot questions}; None when the cluster has none."""
    out = {}
    for cluster, entry in zeroshot.items():
        gold = entry["gold"]
        ids = sorted(set(pilot_ids) & set(gold["by_question"]["teacher"])) if gold else []
        out[cluster] = plain(noninferiority({q: gold["by_question"]["replay"][q] for q in ids},
                                            {q: gold["by_question"]["teacher"][q] for q in ids},
                                            settings["delta_cap_pp"], settings["seed"], settings["n_boot"]))["d"] \
            if ids else None
    return out


def registered_pilot(config: Dict[str, Any]) -> List[str]:
    """The pre-registered pilot questions: F2's `bench.data.pilot_sample` over the calib split."""
    from bench.data import load_splits, pilot_sample  # F2's
    return pilot_sample(load_splits()["calib"], config["stats"]["pilot_size"], config["seeds"]["calib_split"])


def run(centroids_path: str, adapters_path: str, replays: Dict[str, tuple], teacher_eval_run_id: str,
        j8_path: str, j6_path: str, config: Dict[str, Any], noninferiority: Optional[NonInferiority] = None,
        margin: Optional[Callable[..., Dict[str, Any]]] = None, n_test: Optional[int] = None,
        pilot_ids: Optional[List[str]] = None):
    """`replays` is {"cheap_alt": (replay run, per-call eval run), "slm": (B4 replay run, per-call eval run)};
    `j6_path` the J6 result whose chosen candidate's zero-shot replay is the pilot. Returns (result path, fact path)."""
    if set(replays) != set(ENGINES):
        raise JudgmentError(f"J7 needs a calib replay of each of {ENGINES}")
    if config["allocation"].get("min_calls") is None:
        raise JudgmentError("allocation.min_calls is not set: 'enough evidence' must be pre-registered")
    if noninferiority is None:
        from bench.judge.j4 import noninferiority  # F2's J4
    if margin is None:
        from bench.judge.j4 import margin  # F2's J4
    if n_test is None:
        from bench.data import load_splits
        n_test = len(load_splits()["test"])
    if pilot_ids is None:
        pilot_ids = registered_pilot(config)
    centroids, centroids_sha = read_fact(centroids_path, "centroids")
    adapters, adapters_sha = read_fact(adapters_path, "adapters")
    if adapters["centroids"] != centroids_sha or set(adapters["adapters"]) != set(centroids["clusters"]):
        raise JudgmentError("the adapters were not trained on these centroids, one per cluster")
    clusters = sorted(centroids["clusters"])

    b4_run, _ = replays["slm"]
    b4 = require_done(b4_run, type="replay", arm="B4", split="calib")
    recorded = b4.get("facts") or {}
    if (recorded.get("centroids"), recorded.get("adapters")) != (centroids_sha, adapters_sha):
        raise JudgmentError(f"{b4_run} does not record that it was routed with these centroids and adapters")
    cluster_of = router_clusters(calls_of(b4_run), clusters)
    group = lambda identity, attempts: cluster_of.get(identity)  # noqa: E731

    j6 = read_result(j6_path, "J6")
    if j6["result"]["choice_fact"]["sha256"] != adapters["choice"]:
        raise JudgmentError("the adapters were not trained on the base this J6 result chose")
    pilot_reads = j6["reads"][j6["result"]["choice"]]
    j8 = read_result(j8_path, "J8")["result"]
    if j8["engine"].split("+lora:")[0] != f"slm:{adapters['slm']}":
        raise JudgmentError(f"the load test measured {j8['engine']!r}, not the base of these adapters, slm:{adapters['slm']}")
    lowest = j3.utilization_label(min(config["cost"]["utilizations"]))
    evidence, costs, reads = {}, {}, {"teacher_eval": reference(teacher_eval_run_id), "j8": result_reference(j8_path),
                                      "j6": result_reference(j6_path)}
    for engine in ENGINES:
        replay_run, eval_run = replays[engine]
        run_reads, teacher, replay, replay_eval, teacher_eval, found = j2.replay_inputs(replay_run, eval_run, teacher_eval_run_id)
        if found.get("split") != "calib":
            raise JudgmentError(f"{replay_run} is on {found.get('split')!r}: S6 allocates on calib only")
        if engine == "cheap_alt" and found.get("engine") != "cheap_alt":
            raise JudgmentError(f"{replay_run} did not run on cheap_alt")
        missing = set(invocations(teacher)) - set(cluster_of)
        if missing:
            raise JudgmentError(f"the B4 replay did not route {len(missing)} teacher invocations, e.g. {sorted(missing)[:2]}")
        evidence[engine] = j2.compare(teacher, replay, replay_eval, teacher_eval, group, call_sites=found.get("call_sites"))
        for cluster, cost in mean_costs(replay, cluster_of, config["prices"],
                                        j8["cost_per_request"][lowest] if engine == "slm" else None).items():
            costs.setdefault(cluster, {})[engine] = cost
        reads[engine] = run_reads
    settings = {"delta_cap_pp": config["thresholds"]["delta_cap_pp"], "seed": config["seeds"]["bootstrap"],
                "n_boot": n_boot(config), "min_calls": config["allocation"]["min_calls"],
                "concordance_min": config["thresholds"]["concordance_min"], "n_test": n_test}
    _, teacher, zeroshot, zeroshot_eval, teacher_eval, zs_manifest = j2.replay_inputs(
        pilot_reads["replay"]["run_id"], pilot_reads["replay_eval"]["run_id"], teacher_eval_run_id)
    if teacher[0]["run_id"] != reads["slm"]["teacher"]["run_id"]:
        raise JudgmentError("the pilot (J6's zero-shot) replays another teacher run than the calib replays")
    pilot = pilot_discordance(j2.compare(teacher, zeroshot, zeroshot_eval, teacher_eval, group,
                                         call_sites=zs_manifest.get("call_sites")), noninferiority,
                              settings, pilot_ids)
    decided = allocate(clusters, evidence, costs, noninferiority, settings, pilot, margin)
    fact = write_fact(JUDGMENT, "allocation", {"centroids": centroids_sha, "adapters": adapters_sha,
                                                "allocation": {c: d["engine"] for c, d in decided.items()}})
    result = {"allocation": {c: d["engine"] for c, d in decided.items()}, "clusters": decided,
              "centroids": centroids_sha, "adapters": adapters_sha, "allocation_fact": {"sha256": fact.parent.name},
              "pilot_ids": sorted(pilot_ids, key=int), "slm_cost_utilization": lowest,
              "prices": {"as_of": config["prices"].get("as_of"),
                         "sha256": hashlib.sha256(canonical(config["prices"].get("table") or {})).hexdigest()}, "slm_cost_basis": "extrapolated from per-adapter load tests",
              "slm_cost_combined": j8.get("combined"), "settings": settings}
    return write_result(JUDGMENT, reads, result), fact
