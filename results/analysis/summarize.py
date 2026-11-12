"""EXPLORATORY, post-registration (per-cluster learning curves). Never read by a verdict.

Every metric of every point, per call site, of the four clusters' curves: each replay of B0 calib's own inputs
against the teacher on the same invocations, read in process (nothing is written to judgments/).

Points: 0% is the registered base replay (all call sites, restricted here to each cluster's); 25/50/100% are the
curve replays given with --point; "B4" is the registered B4 calib replay restricted to the cluster's call sites (a
cross-check of the 100% point: same adapters, other serving session); "teacher" is the teacher's self-replay, on
the pilot questions only and without the gold sites (the ceiling for agreement).

Per call site and point:
- n: invocations, questions, distinct teacher prompts; `routed`: how many of them the router gave this cluster in
  the B4 replay (bench replay filters by call site, not by cluster);
- J2's score as bench.judge.j2.compare computes it: EX against gold at generate_candidate / revise (the teacher's EX
  beside it), raw agreement with the teacher's decision elsewhere (bench.contracts.concordance.agree);
- the primary descriptive metric of SPEC 6.1: column filter: recall of the columns the teacher kept, agreement on
  the ones it dropped, Cohen's kappa; table / column selection: recall of the teacher's set; keywords: mean Jaccard
  with the teacher's set; SQL: EX against gold; the agent's steps: agreement averaged over distinct prompts;
- parsed-ok rate (the last attempt parsed), latency p50 and p95 (ms), input and output tokens per call (means);
- 95% intervals that resample questions (all of a question's calls together), seeded.

    PYTHONPATH=<worktree> .venv-agent/bin/python .ref/curves/summarize.py \
        --point c0:25=<replay>  --point c1:25=<replay>:<its per-call eval>  ...       # c1 needs the eval (gold sites)
    PYTHONPATH=<worktree> .venv-agent/bin/python .ref/curves/summarize.py              # the free points only (0%, B4, teacher)
Writes .ref/curves/results/curves.{json,md}.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict

SITES = {"c0": ["filter_column"], "c1": ["select_tables", "select_columns", "generate_candidate", "revise"],
         "c2": ["extract_keywords"], "c3": ["agent_ir", "agent_ss", "agent_cg"]}
TEACHER = "agent-B0-calib-20261109T151141.039667Z"
TEACHER_EVAL = "eval-per-call-agent-B0-calib-20261109T151141.039667Z-20261109T153122.727095Z"
BASE = ("replay-slm_qwen3-8b-calib-20261110T025007.659032Z",
        "eval-per-call-replay-slm_qwen3-8b-calib-20261110T025007.659032Z-20261110T031423.931711Z")
B4 = ("replay-B4-calib-20261110T174825.799047Z",
      "eval-per-call-replay-B4-calib-20261110T174825.799047Z-20261110T181832.677536Z")
SELF_REPLAY = "replay-production_llm-calib-20261110T021304.240338Z"
SEED, N_BOOT = 20261111, 2000
LABEL = ("EXPLORATORY, post-registration learning curves per cluster (calib replay of B0's own inputs, each adapter at "
         "its cluster's call sites). Outside the preregistered plan; not a test-split result; never read by a verdict. "
         "Fixed epochs: rows and optimiser steps vary together. One seed per point.")


def ratio(a, b):
    return a / b if b else None


def metrics(c: Counter, prompts) -> dict:
    """Every rate of one cell from its summed counts; `prompts` the distinct prompt hashes of the cell (agent sites)."""
    out = {"parsed_ok": ratio(c["ok"], c["n"]), "agreement": ratio(c["agree"], c["n_agree"]),
           "ex": ratio(c["ex_replay"], c["n_gold"]), "ex_teacher": ratio(c["ex_teacher"], c["n_gold"]),
           "recall_kept": ratio(c["tp"], c["tp"] + c["fn"]), "agreement_dropped": ratio(c["tn"], c["tn"] + c["fp"]),
           "set_recall": ratio(c["recall_sum"], c["recall_n"]), "jaccard": ratio(c["jaccard_sum"], c["jaccard_n"])}
    n = c["tp"] + c["fn"] + c["tn"] + c["fp"]
    if n:
        po = (c["tp"] + c["tn"]) / n
        pe = ((c["tp"] + c["fn"]) * (c["tp"] + c["fp"]) + (c["tn"] + c["fp"]) * (c["tn"] + c["fn"])) / (n * n)
        out["kappa"] = ratio(po - pe, 1 - pe)
    rates = [c[("pa", p)] / c[("pn", p)] for p in prompts if c[("pn", p)]]
    out["agreement_per_prompt"] = ratio(sum(rates), len(rates))
    return {k: v for k, v in out.items() if v is not None}


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]  # nearest rank


def cell(site, teacher, replay, replay_eval, teacher_eval, questions, routed):
    """One (point, call site): counts per question, then the rates and their question-resampled intervals."""
    from bench.contracts.calls import AGENT_CALL_SITES
    from bench.contracts.concordance import GOLD_CALL_SITES, _column_pairs, _keeps_column, _tables, agree
    from bench.judge.base import final
    from bench.judge.j2 import call_correct

    per_question = defaultdict(Counter)
    prompts, latency, tokens_in, tokens_out, n_routed = set(), [], [], [], 0
    for identity in sorted(teacher):
        question_id, call_site, _ = identity
        if call_site != site or (questions is not None and question_id not in questions):
            continue
        mine = replay.get(identity)
        if mine is None:
            sys.exit(f"the replay lacks {identity}: a missing answer is not evidence")
        c = per_question[question_id]
        t, m = final(teacher[identity]), final(mine)
        c["n"] += 1
        c["ok"] += bool(m["parsed_ok"])
        n_routed += routed.get(identity, False)
        prompt = hashlib.sha256(json.dumps(teacher[identity][0]["prompt_messages"], sort_keys=True).encode()).hexdigest()
        prompts.add(prompt)
        latency.append(m["latency_ms"])
        tokens_in.append((m.get("usage") or {}).get("input") or 0)
        tokens_out.append((m.get("usage") or {}).get("output") or 0)
        if site in GOLD_CALL_SITES:
            if replay_eval is None or teacher_eval is None:
                sys.exit(f"{site} is judged by execution: give the replay's per-call evaluation (--point c1:P=REPLAY:EVAL)")
            c["n_gold"] += 1
            c["ex_replay"] += call_correct(identity, mine, replay_eval, "replay")
            c["ex_teacher"] += call_correct(identity, teacher[identity], teacher_eval, "teacher")
            continue
        if not t["parsed_ok"]:
            continue  # no teacher decision to agree with (J2)
        a, b = t["parsed_output"], (m["parsed_output"] if m["parsed_ok"] else None)
        same = agree(site, a, b)
        c["n_agree"] += 1
        c["agree"] += same
        try:
            if site == "filter_column":
                kept, mine_keeps = _keeps_column(a), b is not None and _keeps_column(b)
                c[("tp" if mine_keeps else "fn") if kept else ("fp" if mine_keeps else "tn")] += 1
            elif site in ("select_tables", "select_columns"):
                read = _tables if site == "select_tables" else _column_pairs
                wanted, got = read(a), (read(b) if b is not None else frozenset())
                if wanted:
                    c["recall_sum"] += len(wanted & got) / len(wanted)
                    c["recall_n"] += 1
            elif site == "extract_keywords":
                wanted, got = set(map(str, a or [])), (set(map(str, b)) if isinstance(b, list) else set())
                c["jaccard_sum"] += len(wanted & got) / len(wanted | got) if wanted | got else 1.0
                c["jaccard_n"] += 1
            elif site in AGENT_CALL_SITES:
                c[("pn", prompt)] += 1
                c[("pa", prompt)] += same
        except (TypeError, AttributeError):  # an output of a shape nothing can be read from: counted as a miss
            if site in ("select_tables", "select_columns"):
                c["recall_n"] += 1
            elif site == "extract_keywords":
                c["jaccard_n"] += 1
    if not per_question:
        return None
    ids = sorted(per_question)
    total = sum((per_question[q] for q in ids), Counter())
    point = metrics(total, prompts)
    rng = random.Random(f"{SEED}:{site}")
    boots = defaultdict(list)
    for _ in range(N_BOOT):
        sample = Counter()
        for _ in ids:
            sample.update(per_question[ids[rng.randrange(len(ids))]])
        for name, value in metrics(sample, prompts).items():
            boots[name].append(value)
    ci = {name: [percentile(values, 0.025), percentile(values, 0.975)] for name, values in boots.items()}
    return {"n_calls": total["n"], "n_questions": len(ids), "distinct_prompts": len(prompts), "routed_to_cluster": n_routed,
            "teacher_unparsed": total["n"] - total["n_gold"] - total["n_agree"], **point, "ci95": ci,
            "latency_ms": {"p50": percentile(latency, 0.5), "p95": percentile(latency, 0.95)},
            "tokens_per_call": {"input": sum(tokens_in) / len(tokens_in), "output": sum(tokens_out) / len(tokens_out)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", action="append", default=[], metavar="CLUSTER:PCT=REPLAY[:EVAL]")
    ap.add_argument("--clusters", nargs="+", default=sorted(SITES), choices=sorted(SITES))
    args = ap.parse_args()

    from bench import paths
    from bench.judge import j2, j7
    from bench.judge.base import calls_of, invocations, manifest

    def slim(run_id):  # a replay's calls without the prompts and answers' text (272 MB a run): the teacher's keep theirs
        calls = calls_of(run_id)
        for call in calls:
            call.pop("prompt_messages", None)
            call.pop("response_text", None)
        return calls

    teacher = invocations(calls_of(TEACHER))
    teacher_eval = j2.per_call_eval(TEACHER_EVAL, TEACHER)
    b4_calls = slim(B4[0])
    cluster_of = j7.router_clusters(b4_calls, sorted(SITES))
    engines_b4 = defaultdict(Counter)
    for identity, attempts in invocations(b4_calls).items():
        engines_b4[cluster_of[identity]][attempts[0]["engine"]] += 1

    loaded = {}

    def load(replay_id, eval_id, sites):
        if replay_id not in loaded:
            loaded[replay_id] = _load(replay_id, eval_id)
        replay, replay_eval, questions, found = loaded[replay_id]
        scope = found.get("call_sites")
        if scope is not None and not set(sites) <= set(scope):
            sys.exit(f"{replay_id} replayed {scope}, which leaves out some of {sites}")
        return replay, replay_eval, questions, found

    def _load(replay_id, eval_id):
        found = manifest(replay_id)
        if found.get("status") != "done" or found.get("type") != "replay" or found.get("source_run_id") != TEACHER:
            sys.exit(f"{replay_id} is not a done replay of {TEACHER}")
        replay = invocations(b4_calls if replay_id == B4[0] else slim(replay_id))
        questions = set(found["question_ids"]) if found.get("question_ids") else {q for q, _, _ in replay}
        return replay, (j2.per_call_eval(eval_id, replay_id) if eval_id else None), questions, found

    given = defaultdict(dict)
    for spec in args.point:
        where, _, pair = spec.partition("=")
        cluster, _, pct = where.partition(":")
        replay_id, _, eval_id = pair.partition(":")
        if cluster not in SITES or pct not in ("25", "50", "100"):
            sys.exit(f"--point {spec}: CLUSTER:PCT is one of {sorted(SITES)} and 25, 50 or 100")
        given[cluster][f"{pct}%"] = (replay_id, eval_id or None)

    results = {"label": LABEL, "teacher": TEACHER, "seed": SEED, "n_boot": N_BOOT,
               "b4_engines_by_cluster": {c: dict(v) for c, v in sorted(engines_b4.items())}, "clusters": {}}
    lines = [f"> {LABEL}", ""]
    for cluster in args.clusters:
        sites = SITES[cluster]
        routed = {identity: cluster_of.get(identity) == cluster for identity in teacher}
        gold = cluster == "c1"
        points = {"0% (base)": BASE, **given.get(cluster, {}), "B4 (100%, cross-check)": B4,
                  "teacher (self-replay, pilot)": (SELF_REPLAY, None)}
        out = results["clusters"][cluster] = {}
        for name, (replay_id, eval_id) in points.items():
            replay, replay_eval, questions, found = load(replay_id, eval_id, [] if name.startswith("teacher") else sites)
            scope = found.get("call_sites")
            per_site = {}
            for site in sites:
                if scope is not None and site not in scope:
                    continue  # the teacher's self-replay leaves the gold sites out
                entry = cell(site, teacher, replay, replay_eval, teacher_eval if gold else None, questions, routed)
                if entry:
                    per_site[site] = entry
            out[name] = {"replay": replay_id, "eval": eval_id, "engine": found.get("engine") or found.get("arm"),
                         "per_call_site": per_site}
        lines += [f"## {cluster}", "",
                  "| call site | point | calls (questions, distinct prompts; routed to the cluster) | J2 score | primary (SPEC 6.1) | "
                  "parsed ok | latency p50 / p95 ms | tokens in / out per call |", "|---|---|---|---|---|---|---|---|"]
        for site in sites:
            for name, point in out.items():
                e = point["per_call_site"].get(site)
                if not e:
                    continue
                show = lambda key: (f"{e[key]:.3f} [{e['ci95'][key][0]:.3f}, {e['ci95'][key][1]:.3f}]"  # noqa: E731
                                    if key in e else "n/a")
                if "ex" in e:
                    score = primary = f"EX {show('ex')} (teacher {e['ex_teacher']:.3f})"
                else:
                    score = f"agreement {show('agreement')}"
                    primary = {"filter_column": lambda: f"recall kept {show('recall_kept')}; agreement dropped "
                                                        f"{show('agreement_dropped')}; kappa {show('kappa')}",
                               "select_tables": lambda: f"set recall {show('set_recall')}",
                               "select_columns": lambda: f"set recall {show('set_recall')}",
                               "extract_keywords": lambda: f"Jaccard {show('jaccard')}"}.get(
                        site, lambda: f"agreement per distinct prompt {show('agreement_per_prompt')}")()
                lines.append(f"| {site} | {name} | {e['n_calls']} ({e['n_questions']}, {e['distinct_prompts']}; {e['routed_to_cluster']}) | "
                             f"{score} | {primary} | {show('parsed_ok')} | {e['latency_ms']['p50']} / {e['latency_ms']['p95']} | "
                             f"{e['tokens_per_call']['input']:.0f} / {e['tokens_per_call']['output']:.0f} |")
        lines.append("")
    out_dir = paths.ROOT / ".ref" / "curves" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "curves.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    (out_dir / "curves.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
