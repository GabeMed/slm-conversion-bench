"""J7 · S6, the allocation of B5 (SPEC §4 S6, §6.4; D15), written as the `allocation` fact.

Per cluster of the centroids, **the cheapest engine that passes on calib**, else the production LLM:
- the calls of a cluster that have gold (SQL generation and repair) pass by **non-inferiority at the
  fixed selection margin** `thresholds.selection_delta_pp` (J4: the engine is the candidate, the teacher
  the reference, the question the unit), when they number at least `allocation.min_calls`. The margin
  is fixed before any data and never derived from a discordance (design §6.3 T3);
- the calls without gold pass by **agreement with the teacher** at the cluster's bar, over at least
  `allocation.min_calls` calls: a proxy, declared as such (D15), which supports no claim per cluster.
  **The bar is tied to the teacher's own reproducibility** (design §6.3 T4): no engine can agree with
  one teacher sample more often than the teacher agrees with itself. Each call site's bar is
  min(`thresholds.concordance_min`, A_tt − `thresholds.concordance_slack_pp`/100), where A_tt is the
  agreement (C3) between the teacher's calib run and its own replay on the production LLM, on the pilot
  questions (`bench.data.pilot_ids`); a cluster's bar is the mean of its call sites' bars, weighted by
  the cluster's calls of each. Without that self-replay J7 refuses: the bar would be untied. A_tt is
  reported by call site and by difficulty. It measures reproducibility, not correctness;
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
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional

from bench.contracts.concordance import GOLD_CALL_SITES
from bench.contracts.facts import read_fact, write_fact
from bench.judge import j2, j3
from bench.judge.base import (Identity, JudgmentError, calls_of, canonical, invocations, n_boot, read_result, reference,
                              require_done, result_reference, write_result)

JUDGMENT = "J7"
# the keys of config.yaml this judgment reads (design §6.2: `reads` names them, for `bench verify`).
# seeds.calib_split and stats.pilot_size are what bench.data.pilot_ids reads today: they follow that accessor
CONFIG_KEYS = ["allocation.min_calls", "cost.utilizations", "data.bird_dev_questions", "prices", "seeds.bootstrap",
               "seeds.calib_split", "stats.n_boot", "stats.pilot_size", "thresholds.concordance_min",
               "thresholds.concordance_slack_pp", "thresholds.selection_delta_pp"]
ENGINES = ("cheap_alt", "slm")
NonInferiority = Callable[..., Dict[str, Any]]


def plain(value: Any) -> Any:
    """J4's output as plain JSON values (a numpy scalar has `.item()`)."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value.item() if hasattr(value, "item") else value


def site_bars(self_agreement: Dict[str, Dict[str, Any]], settings: Dict[str, Any]) -> Dict[str, Fraction]:
    """Each call site's agreement bar (T4): min(concordance_min, A_tt − slack). A site whose A_tt was not
    measured has none. Exact fractions, so an engine exactly on its bar passes whatever the rounding."""
    floor, slack = Fraction(str(settings["concordance_min"])), Fraction(str(settings["concordance_slack_pp"])) / 100
    return {site: min(floor, Fraction(found["agree"], found["n"]) - slack)
            for site, found in self_agreement.items() if found["n"]}


def cluster_bar(sites: Dict[str, int], bars: Dict[str, Fraction]) -> Optional[Fraction]:
    """A cluster's bar: the mean of its call sites' bars, weighted by the cluster's agreement calls of each
    (`sites`: {call site: calls}); None for a cluster with no such call."""
    missing = sorted(site for site in sites if site not in bars)
    if missing:
        raise JudgmentError(f"the teacher's self-replay does not measure its agreement on {', '.join(missing)}: "
                            f"their bar cannot be tied to the teacher")
    total = sum(sites.values())
    return sum(n * bars[site] for site, n in sites.items()) / total if total else None


def passes(entry: Optional[Dict[str, Any]], noninferiority: NonInferiority, settings: Dict[str, Any],
           bar: Optional[Fraction] = None) -> Dict[str, Any]:
    """Whether one engine passes on one cluster, and why; `bar` is the cluster's agreement bar, which the
    engine's agreement (agree / n, exactly) must reach."""
    if entry is None:
        return {"passes": False, "why": "no calib call in this cluster"}
    verdict: Dict[str, Any] = {"passes": True, "why": []}
    if entry["gold"]:
        gold = entry["gold"]
        test = plain(noninferiority(gold["by_question"]["replay"], gold["by_question"]["teacher"], settings["seed"],
                                    settings["n_boot"], margin=settings["selection_delta_pp"] / 100))
        ok = gold["n"] >= settings["min_calls"] and bool(test["noninferior"])
        verdict["gold"] = {"n": gold["n"], "ex_engine": gold["ex_replay"], "ex_teacher": gold["ex_teacher"],
                           "j4": test, "passes": bool(ok)}
        if not ok:
            verdict["why"].append("gold: fewer calls than allocation.min_calls" if gold["n"] < settings["min_calls"] else
                                  "gold: not non-inferior at thresholds.selection_delta_pp")
    if entry["agreement"]:
        agreement = entry["agreement"]
        ok = agreement["n"] >= settings["min_calls"] and agreement["rate"] is not None and bar is not None \
            and Fraction(agreement["agree"], agreement["n"]) >= bar
        verdict["agreement"] = {"n": agreement["n"], "rate": agreement["rate"], "bar": None if bar is None else float(bar),
                                "passes": bool(ok), "proxy": True}
        if not ok:
            verdict["why"].append("agreement: fewer calls than allocation.min_calls" if agreement["n"] < settings["min_calls"]
                                  else "agreement: below the bar, min(thresholds.concordance_min, A_tt − slack)")
    verdict["passes"] = not verdict["why"]
    verdict["why"] = "; ".join(verdict["why"]) or "passes"
    return verdict


def allocate(clusters: List[str], evidence: Dict[str, Dict[str, Dict[str, Any]]], costs: Dict[str, Dict[str, float]],
             noninferiority: NonInferiority, settings: Dict[str, Any],
             bars: Dict[str, Dict[str, Optional[Fraction]]]) -> Dict[str, Dict[str, Any]]:
    """{cluster: {"engine", "order", "evidence", ...}}; `evidence` is {engine: J2 compare by cluster},
    `costs` is {cluster: {engine: mean cost per invocation}}, `bars` is {engine: {cluster: agreement
    bar}} (an engine's replay may cover its own call sites). When more than one engine passes, the
    choice rests on the cost order alone (`cost_dependent`), and the SLM's cost is extrapolated from
    per-adapter load tests: the ratio between the two says how much."""
    out = {}
    for cluster in sorted(clusters):
        order = sorted(costs.get(cluster, {}), key=lambda e: (costs[cluster][e], e))
        verdicts = {engine: passes(evidence[engine].get(cluster), noninferiority, settings, bars[engine].get(cluster))
                    for engine in ENGINES}
        passing = [engine for engine in order if verdicts[engine]["passes"]]
        chosen = passing[0] if passing else "production_llm"
        out[cluster] = {"engine": chosen, "order": order, "costs": costs.get(cluster, {}),
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


def teacher_self_agreement(self_replay_run_id: str, teacher_run_id: str, pilot_ids: List[str],
                           difficulty: Dict[str, str]):
    """A_tt (T4): per call site without gold, the agreement (C3) between the teacher's calib run and its
    own replay on the production LLM, on the pilot questions, in total and by difficulty. Returns (what
    was read, {call site: {"n", "agree", "rate", "by_difficulty"}})."""
    reads, teacher, replay, _, _, found = j2.replay_inputs(self_replay_run_id, None, None)
    if (found.get("split"), found.get("engine")) != ("calib", "production_llm"):
        raise JudgmentError(f"{self_replay_run_id} is not a calib replay on production_llm: the bar is tied to the "
                            f"teacher's agreement with itself")
    if reads["teacher"]["run_id"] != teacher_run_id:
        raise JudgmentError(f"{self_replay_run_id} replays another teacher run than the calib replays ({teacher_run_id})")
    pilot = set(pilot_ids)
    if {c["question_id"] for c in replay} != pilot:
        raise JudgmentError(f"{self_replay_run_id} is not on the pilot questions (bench.data.pilot_ids)")
    scope = found.get("call_sites")
    scope = None if scope is None else [site for site in scope if site not in GOLD_CALL_SITES]

    def routine(calls: List[dict]) -> List[dict]:
        return [c for c in calls if c["call_site"] not in GOLD_CALL_SITES and c["question_id"] in pilot]
    by_site = j2.compare(routine(teacher), routine(replay), call_sites=scope)
    by_level = j2.compare(routine(teacher), routine(replay), call_sites=scope,
                          group=lambda identity, attempts: (identity[1], difficulty[identity[0]]))
    counts = lambda entry: {k: entry["agreement"][k] for k in ("n", "agree", "rate")}  # noqa: E731
    return reads, {site: {**counts(entry), "by_difficulty": {level: counts(e) for (s, level), e in by_level.items() if s == site}}
                   for site, entry in by_site.items()}


def run(centroids_path: str, adapters_path: str, replays: Dict[str, tuple], teacher_eval_run_id: str,
        j8_path: str, j6_path: str, config: Dict[str, Any], noninferiority: Optional[NonInferiority] = None, *,
        teacher_self_replay: Optional[str] = None, pilot_ids: Optional[List[str]] = None,
        difficulty: Optional[Dict[str, str]] = None):
    """`replays` is {"cheap_alt": (replay run, per-call eval run), "slm": (B4 replay run, per-call eval run)};
    `j6_path` the J6 result that chose the adapters' base; `teacher_self_replay` the replay of the teacher's
    calib run on the production LLM, on the pilot questions. Returns (result path, fact path)."""
    if set(replays) != set(ENGINES):
        raise JudgmentError(f"J7 needs a calib replay of each of {ENGINES}")
    if not teacher_self_replay:
        raise JudgmentError("J7 needs --teacher-self-replay, the teacher's calib run replayed on production_llm on the "
                            "pilot questions: without it the agreement bar is not tied to the teacher")
    if config["allocation"].get("min_calls") is None:
        raise JudgmentError("allocation.min_calls is not set: 'enough evidence' must be pre-registered")
    if noninferiority is None:
        from bench.judge.j4 import noninferiority
    if pilot_ids is None:
        from bench.data import pilot_ids as registered_pilot
        pilot_ids = registered_pilot(config)
    if difficulty is None:
        from bench.data import questions_for
        difficulty = {q: found["difficulty"] for q, found in questions_for(config, "calib").items()}
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
    j8 = read_result(j8_path, "J8")["result"]
    if j8["engine"].split("+lora:")[0] != f"slm:{adapters['slm']}":
        raise JudgmentError(f"the load test measured {j8['engine']!r}, not the base of these adapters, slm:{adapters['slm']}")
    lowest = j3.utilization_label(min(config["cost"]["utilizations"]))
    evidence, costs, sites = {}, {}, {}
    reads: Dict[str, Any] = {"teacher_eval": reference(teacher_eval_run_id), "j8": result_reference(j8_path),
                             "j6": result_reference(j6_path), "config": CONFIG_KEYS}
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
        by_site = j2.compare(teacher, replay, replay_eval, teacher_eval, call_sites=found.get("call_sites"),
                             group=lambda identity, attempts: (cluster_of[identity], identity[1]))
        sites[engine] = {}  # {cluster: {call site without gold: its agreement calls}}
        for (cluster, site), entry in by_site.items():
            if entry["agreement"] and entry["agreement"]["n"]:
                sites[engine].setdefault(cluster, {})[site] = entry["agreement"]["n"]
        for cluster, cost in mean_costs(replay, cluster_of, config["prices"],
                                        j8["cost_per_request"][lowest] if engine == "slm" else None).items():
            costs.setdefault(cluster, {})[engine] = cost
        reads[engine] = run_reads
    settings = {"selection_delta_pp": config["thresholds"]["selection_delta_pp"], "seed": config["seeds"]["bootstrap"],
                "n_boot": n_boot(config), "min_calls": config["allocation"]["min_calls"],
                "concordance_min": config["thresholds"]["concordance_min"],
                "concordance_slack_pp": config["thresholds"]["concordance_slack_pp"]}
    self_reads, a_tt = teacher_self_agreement(teacher_self_replay, reads["slm"]["teacher"]["run_id"], pilot_ids, difficulty)
    reads["teacher_self_replay"] = self_reads["replay"]
    site_bar = site_bars(a_tt, settings)
    bars = {engine: {cluster: cluster_bar(found, site_bar) for cluster, found in sites[engine].items()} for engine in ENGINES}
    decided = allocate(clusters, evidence, costs, noninferiority, settings, bars)
    fact = write_fact(JUDGMENT, "allocation", {"centroids": centroids_sha, "adapters": adapters_sha,
                                                "allocation": {c: d["engine"] for c, d in decided.items()}})
    result = {"allocation": {c: d["engine"] for c, d in decided.items()}, "clusters": decided,
              "centroids": centroids_sha, "adapters": adapters_sha, "allocation_fact": {"sha256": fact.parent.name},
              "pilot_ids": sorted(pilot_ids, key=int), "slm_cost_utilization": lowest,
              "teacher_self_agreement": {site: {**found, "bar": float(site_bar[site]) if site in site_bar else None}
                                         for site, found in a_tt.items()},
              "prices": {"as_of": config["prices"].get("as_of"),
                         "sha256": hashlib.sha256(canonical(config["prices"].get("table") or {})).hexdigest()}, "slm_cost_basis": "extrapolated from per-adapter load tests",
              "slm_cost_combined": j8.get("combined"), "settings": settings}
    return write_result(JUDGMENT, reads, result), fact
