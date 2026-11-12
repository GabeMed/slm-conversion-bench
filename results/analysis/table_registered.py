"""EXPLORATORY per-call-site table in the form SPEC 6.1 fixes ("Exploratory readings of the test, without a verdict").

Read-only over runs/ and train/datasets/. Nothing is executed against a database and no model is called: the EX
cells come from per-call evaluations that already exist. Call files are streamed line by line (prompts are hashed
and dropped), so the process stays small.

    cd <scratch checkout> && PYTHONPATH=<scratch checkout> .venv-agent/bin/python table_registered.py --split test
    ... --split calib        # the same table over the calibration replays

Adapted from .ref/curves/summarize.py (primary metrics, question bootstrap) and .ref/percallsite-test/percallsite_test.py
(training rows per call site). Writes table_<split>_registered.{md,json} beside this script.

Trivial rules (the column that uses no language model), as SPEC 6.1 names them:
- filter_column: always "No";
- select_tables / select_columns: every candidate in the prompt (every CREATE TABLE of the prompt's schema; every
  column line of those tables);
- agent steps: a lookup by prompt (prompt -> the action the c3 training rows give that exact prompt; the SPEC does
  not name the source of the lookup table, so the cell says which one was used);
- extract_keywords, generate_candidate, revise: NOT computed, see NOT_DEFINED below.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench.contracts.calls import AGENT_CALL_SITES
from bench.contracts.concordance import GOLD_CALL_SITES, _column_pairs, _keeps_column, _tables, agree

MAIN = Path("<repo>")
RUNS = MAIN / "runs"
OUT = Path(__file__).resolve().parent
SITES = {"c0": ["filter_column"], "c1": ["select_tables", "select_columns", "generate_candidate", "revise"],
         "c2": ["extract_keywords"], "c3": ["agent_ir", "agent_ss", "agent_cg"]}
CLUSTER_OF_SITE = {s: c for c, ss in SITES.items() for s in ss}
ORDER = [s for c in ("c0", "c1", "c2", "c3") for s in SITES[c]]
SEED, N_BOOT = 20261112, 2000

INPUTS = {
    "test": {
        "teacher": ("agent-B0-test-20261111T060814.371294Z",
                    "eval-per-call-agent-B0-test-20261111T060814.371294Z-20261112T043501.196637Z"),
        "base": ("replay-slm_qwen3-8b-test-20261111T225111.186203Z",
                 "eval-per-call-replay-slm_qwen3-8b-test-20261111T225111.186203Z-20261112T042104.945806Z"),
        "adapter": ("replay-B4-test-20261111T225020.625646Z",
                    "eval-per-call-replay-B4-test-20261111T225020.625646Z-20261112T041858.585025Z"),
        "cheap": None,
        "self": ("replay-production_llm-test-20261111T202527.888323Z",
                 "eval-per-call-replay-production_llm-test-20261111T202527.888323Z-20261112T043430.618816Z"),
    },
    "calib": {
        "teacher": ("agent-B0-calib-20261109T151141.039667Z",
                    "eval-per-call-agent-B0-calib-20261109T151141.039667Z-20261109T153122.727095Z"),
        "base": ("replay-slm_qwen3-8b-calib-20261110T025007.659032Z",
                 "eval-per-call-replay-slm_qwen3-8b-calib-20261110T025007.659032Z-20261110T031423.931711Z"),
        "adapter": ("replay-B4-calib-20261110T174825.799047Z",
                    "eval-per-call-replay-B4-calib-20261110T174825.799047Z-20261110T181832.677536Z"),
        "cheap": ("replay-cheap_alt-calib-20261110T160847.505204Z",
                  "eval-per-call-replay-cheap_alt-calib-20261110T160847.505204Z-20261110T162517.048684Z"),
        "self": ("replay-production_llm-calib-20261110T021304.240338Z", None),
    },
}
COLUMNS = [("trivial", "trivial rule (no language model)"), ("base", "base SLM (Qwen3-8B, no adapter)"),
           ("adapter", "SLM + adapters (B4's replay)"), ("cheap", "cheaper LLM (gpt-oss)"),
           ("self", "teacher's own repetition")]
NOT_RUN = "not run"
NOT_DEFINED = {
    "extract_keywords": "not defined in SPEC (the stop-word list and the tokenisation of \"the question's words\" are not fixed)",
    "generate_candidate": "not defined in SPEC (the distance behind \"nearest training question\" is not fixed; its EX "
                          "would also need a gold evaluation that does not exist)",
    "revise": "not defined in SPEC (the distance behind \"nearest training question\" is not fixed; its EX would also "
              "need a gold evaluation that does not exist)",
}
PRIMARY = {"filter_column": "recall of the columns the teacher kept; agreement on the columns it dropped; Cohen's kappa",
           "select_tables": "recall of the teacher's set; exact-set agreement beside it",
           "select_columns": "recall of the teacher's set; exact-set agreement beside it",
           "extract_keywords": "mean Jaccard against the teacher's set; exact-set agreement beside it",
           "generate_candidate": "EX against the gold", "revise": "EX against the gold",
           "agent_ir": "agreement per distinct prompt (no rate over calls)",
           "agent_ss": "agreement per distinct prompt (no rate over calls)",
           "agent_cg": "agreement per distinct prompt (no rate over calls)"}
KEYS = ["n", "ok", "n_agree", "agree", "tp", "fn", "tn", "fp", "recall_sum", "recall_n", "jac_sum", "jac_n",
        "n_gold", "ex", "ex_teacher"]
KI = {k: i for i, k in enumerate(KEYS)}


# ---------------------------------------------------------------- reading

def manifest(run_id):
    return json.loads((RUNS / run_id / "manifest.json").read_text())


def prompt_hash(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def slim_output(site, output):
    if isinstance(output, dict) and "chain_of_thought_reasoning" in output:
        output = {k: v for k, v in output.items() if k != "chain_of_thought_reasoning"}
    return output


_TABLE = re.compile(r"^CREATE TABLE (.+?)\s*$", re.M)
_SKIP = ("foreign key", "primary key", "constraint", "unique", "check")


def _name(text):
    text = text.strip()
    for quote in "`\"":
        if text.startswith(quote):
            return text[1:text.index(quote, 1)]
    if text.startswith("["):
        return text[1:text.index("]")]
    return text.split()[0].rstrip(",")


def schema_candidates(prompt_text):
    """{table: [columns]} of every CREATE TABLE block in the prompt, as written (quotes removed)."""
    out = {}
    for match in _TABLE.finditer(prompt_text):
        table = _name(match.group(1))
        columns = []
        for line in prompt_text[match.end():].split("\n")[1:]:
            stripped = line.strip()
            if stripped.startswith(")"):
                break
            if not stripped or stripped == "(" or stripped.lower().startswith(_SKIP):
                continue
            columns.append(_name(stripped))
        out[table] = columns
    return out


def agent_action(text):
    """CHESS's reading, simplified: DONE anywhere is done; else the name in the first <tool_call> block."""
    if "DONE" in text:
        return {"done": True}
    found = re.search(r"<tool_call>\s*([A-Za-z_]+)", text)
    return {"tool": found.group(1)} if found else None


def agent_lookup():
    """prompt hash -> (majority action of the c3 training rows with that exact prompt, its share, rows)."""
    by_prompt = defaultdict(Counter)
    with open(MAIN / "train/datasets/c3.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            action = agent_action(row["completion"][-1]["content"])
            by_prompt[prompt_hash(row["prompt"])][json.dumps(action, sort_keys=True)] += 1
    return {h: (json.loads(c.most_common(1)[0][0]), c.most_common(1)[0][1] / sum(c.values()), sum(c.values()))
            for h, c in by_prompt.items()}


def load_calls(run_id, teacher=False, lookup=None):
    """{identity: [slim attempts, in attempt order]}; for the teacher, also per identity the prompt hash, the
    prompt prefix and the trivial rule's output, taken from the first attempt's prompt."""
    grouped, extra, stats = {}, {}, Counter()
    with open(RUNS / run_id / "calls.jsonl") as fh:
        for line in fh:
            if not line.strip():
                continue
            call = json.loads(line)
            site = call["call_site"]
            identity = (str(call["question_id"]), site, call["invocation_key"])
            usage = call.get("usage") or {}
            grouped.setdefault(identity, []).append({
                "attempt": call["attempt"], "parsed_ok": call["parsed_ok"],
                "parsed_output": slim_output(site, call.get("parsed_output")), "latency_ms": call.get("latency_ms"),
                "input": usage.get("input") or 0, "output": usage.get("output") or 0,
                "cluster": call.get("cluster"), "engine": call.get("engine")})
            if teacher and (identity not in extra or call["attempt"] < extra[identity]["attempt"]):
                messages = call["prompt_messages"]
                digest = prompt_hash(messages)
                text = "\n".join(m["content"] for m in messages)
                trivial = None
                if site == "filter_column":
                    trivial = {"is_column_information_relevant": "No"}
                elif site == "select_tables":
                    trivial = {"table_names": list(schema_candidates(text))}
                elif site == "select_columns":
                    trivial = schema_candidates(text)
                elif site in AGENT_CALL_SITES:
                    hit = lookup.get(digest)
                    stats["lookup_hit" if hit else "lookup_miss"] += 1
                    if hit:
                        trivial = hit[0]
                        stats["lookup_not_unanimous"] += hit[1] < 1
                extra[identity] = {"attempt": call["attempt"], "prompt": digest,
                                   "prefix": messages[0]["content"][:70], "trivial": trivial}
            del call
    for attempts in grouped.values():
        attempts.sort(key=lambda a: a["attempt"])
    return (grouped, extra, stats) if teacher else grouped


def eval_rows(eval_id, source):
    found = manifest(eval_id)
    assert found.get("type") == "eval" and found.get("per_call") and found.get("source_run_id") == source, eval_id
    index, difficulty = {}, {}
    with open(RUNS / eval_id / "results.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            identity = (str(row["question_id"]), row["call_site"], row["invocation_key"])
            assert identity not in index, identity
            index[identity] = bool(row["correct"])
            difficulty[str(row["question_id"])] = row["difficulty"]
    return index, difficulty


def correct(identity, attempts, results):
    if attempts is None or not attempts[-1]["parsed_ok"]:
        return False  # no output, no SQL: wrong (bench.judge.j2.call_correct)
    return results[identity]  # KeyError: the evaluation lacks a parsed call, which must not pass silently


# ---------------------------------------------------------------- metrics

def rates(m):
    """m: array (..., len(KEYS)) of summed counts -> {name: array (...)}; nan where the denominator is 0."""
    def div(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)
    g = lambda k: m[..., KI[k]]  # noqa: E731
    tp, fn, tn, fp = g("tp"), g("fn"), g("tn"), g("fp")
    n4 = tp + fn + tn + fp
    po = div(tp + tn, n4)
    pe = div((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn), n4 * n4)
    return {"parsed_ok": div(g("ok"), g("n")), "raw_agreement": div(g("agree"), g("n_agree")),
            "recall_kept": div(tp, tp + fn), "agreement_dropped": div(tn, tn + fp),
            "kappa": np.where(pe < 1, (po - pe) / np.where(pe < 1, 1 - pe, 1), np.nan),
            "set_recall": div(g("recall_sum"), g("recall_n")), "jaccard": div(g("jac_sum"), g("jac_n")),
            "ex": div(g("ex"), g("n_gold")), "ex_teacher": div(g("ex_teacher"), g("n_gold"))}


def cell(site, teacher, extra, replay, replay_eval, teacher_eval, questions=None):
    """One (call site, column): per-question counts, the rates, and intervals that resample questions (all of a
    question's calls together). Agent steps: per distinct prompt, no interval."""
    per_question = defaultdict(lambda: np.zeros(len(KEYS)))
    per_prompt = defaultdict(lambda: [0, 0])
    missing = 0
    for identity in teacher:
        question_id, call_site, _ = identity
        if call_site != site or (questions is not None and question_id not in questions):
            continue
        mine = replay.get(identity)
        if mine is None:
            missing += 1
            continue
        c = per_question[question_id]
        t, m = teacher[identity][-1], mine[-1]
        c[KI["n"]] += 1
        c[KI["ok"]] += bool(m["parsed_ok"])
        if site in GOLD_CALL_SITES:
            c[KI["n_gold"]] += 1
            c[KI["ex"]] += correct(identity, mine, replay_eval)
            c[KI["ex_teacher"]] += correct(identity, teacher[identity], teacher_eval)
            continue
        if not t["parsed_ok"]:
            continue  # no teacher decision to agree with (J2)
        a, b = t["parsed_output"], (m["parsed_output"] if m["parsed_ok"] else None)
        same = agree(site, a, b)
        c[KI["n_agree"]] += 1
        c[KI["agree"]] += same
        try:
            if site == "filter_column":
                kept, mine_keeps = _keeps_column(a), b is not None and _keeps_column(b)
                c[KI[("tp" if mine_keeps else "fn") if kept else ("fp" if mine_keeps else "tn")]] += 1
            elif site in ("select_tables", "select_columns"):
                read = _tables if site == "select_tables" else _column_pairs
                wanted, got = read(a), (read(b) if b is not None else frozenset())
                if wanted:
                    c[KI["recall_sum"]] += len(wanted & got) / len(wanted)
                    c[KI["recall_n"]] += 1
            elif site == "extract_keywords":
                wanted, got = set(map(str, a or [])), (set(map(str, b)) if isinstance(b, list) else set())
                c[KI["jac_sum"]] += len(wanted & got) / len(wanted | got) if wanted | got else 1.0
                c[KI["jac_n"]] += 1
            elif site in AGENT_CALL_SITES:
                per_prompt[extra[identity]["prompt"]][0] += 1
                per_prompt[extra[identity]["prompt"]][1] += same
        except (TypeError, AttributeError):  # an output of a shape nothing can be read from: a miss
            if site in ("select_tables", "select_columns"):
                c[KI["recall_n"]] += 1
            elif site == "extract_keywords":
                c[KI["jac_n"]] += 1
    if not per_question:
        return {"n_calls": 0, "n_questions": 0, "missing_in_replay": missing}
    ids = sorted(per_question)
    matrix = np.stack([per_question[q] for q in ids])
    total = matrix.sum(axis=0)
    out = {"n_calls": int(total[KI["n"]]), "n_questions": len(ids), "missing_in_replay": missing,
           "counts": {k: (float(total[KI[k]]) if k.endswith("_sum") else int(total[KI[k]])) for k in KEYS},
           "teacher_unparsed": int(total[KI["n"]] - total[KI["n_gold"]] - total[KI["n_agree"]])}
    point = {k: (None if np.isnan(v) else float(v)) for k, v in rates(total).items()}
    out.update({k: v for k, v in point.items() if v is not None})
    if site in AGENT_CALL_SITES:
        shares = [agreed / n for n, agreed in per_prompt.values()]
        out["distinct_prompts"] = len(shares)
        out["agreement_per_prompt"] = sum(shares) / len(shares) if shares else None
        out["prompts_fully_agreeing"] = sum(s == 1 for s in shares)
        out["per_prompt"] = sorted([[n, agreed] for n, agreed in per_prompt.values()], reverse=True)
        return out  # no interval over calls: the effective sample is the distinct prompts
    rng = np.random.default_rng([SEED, ORDER.index(site)])
    draws = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    boots = rates(np.stack([matrix[d].sum(axis=0) for d in draws]))
    out["ci95"] = {}
    for name in point:
        if point[name] is None:
            continue
        values = boots[name][~np.isnan(boots[name])]
        out["ci95"][name] = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if len(values) else None
    return out


def percentile(values, q):
    values = sorted(v for v in values if v is not None)
    return values[max(0, int(np.ceil(q * len(values))) - 1)] if values else None  # nearest rank


def cost(site, calls, questions=None):
    """Latency and tokens of the final attempt of each invocation of a call site."""
    last = [a[-1] for (q, s, _), a in calls.items() if s == site and (questions is None or q in questions)]
    if not last:
        return None
    return {"n_calls": len(last), "latency_ms_p50": percentile([a["latency_ms"] for a in last], 0.5),
            "latency_ms_p95": percentile([a["latency_ms"] for a in last], 0.95),
            "tokens_in_mean": sum(a["input"] for a in last) / len(last),
            "tokens_out_mean": sum(a["output"] for a in last) / len(last),
            "attempts_per_invocation": sum(len(a) for (q, s, _), a in calls.items()
                                           if s == site and (questions is None or q in questions)) / len(last)}


# ---------------------------------------------------------------- printing

def show(e, key, digits=3):
    if key not in e:
        return "n/a"
    ci = (e.get("ci95") or {}).get(key)
    return f"{e[key]:.{digits}f}" + (f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else "")


def render(site, e, column=None):
    """A cell's text. Raw agreement is always printed beside the primary metric."""
    if isinstance(e, str):
        return e
    if not e or not e.get("n_calls"):
        return "no invocation of this call site in this column's questions"
    scope = f" (n = {e['n_questions']} questions, {e['n_calls']} calls)"
    if site in GOLD_CALL_SITES:
        c = e["counts"]
        text = f"EX {c['ex']}/{c['n_gold']} = {show(e, 'ex')}"
        if column == "self":
            text = (f"replay: EX {c['ex']}/{c['n_gold']} = {show(e, 'ex')}; the registered run on the same invocations: "
                    f"EX {c['ex_teacher']}/{c['n_gold']} = {show(e, 'ex_teacher')}")
        return text + "; raw agreement: not applicable (judged by execution)" + scope
    if site in AGENT_CALL_SITES:
        c = e["counts"]
        return (f"agreement per distinct prompt {e['agreement_per_prompt']:.3f} (k = {e['distinct_prompts']} prompts, "
                f"{e['prompts_fully_agreeing']} with every call agreeing); raw agreement, as counts only: "
                f"{c['agree']} of {c['n_agree']} calls" + scope)
    raw = f"raw agreement {show(e, 'raw_agreement')}"
    if site == "filter_column":
        return (f"recall kept {show(e, 'recall_kept')}; agreement dropped {show(e, 'agreement_dropped')}; "
                f"kappa {show(e, 'kappa')}; {raw}" + scope)
    if site in ("select_tables", "select_columns"):
        return f"set recall {show(e, 'set_recall')}; exact-set agreement (= {raw})" + scope
    return f"Jaccard {show(e, 'jaccard')}; exact-set agreement (= {raw})" + scope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(INPUTS), default="test")
    split = ap.parse_args().split
    inputs = INPUTS[split]
    teacher_id, teacher_eval_id = inputs["teacher"]
    assert "B1-test" not in json.dumps(inputs)

    lookup = agent_lookup()
    teacher, extra, stats = load_calls(teacher_id, teacher=True, lookup=lookup)
    teacher_eval, difficulty = eval_rows(teacher_eval_id, teacher_id)
    all_questions = {q for q, _, _ in teacher}
    trivial = {i: [{"attempt": 1, "parsed_ok": True, "parsed_output": x["trivial"]}]
               for i, x in extra.items() if x["trivial"] is not None}

    # training rows per call site (prompt prefix, as .ref/percallsite-test does) and per cluster
    prefix_site = {}
    for identity, x in extra.items():
        prefix_site.setdefault(x["prefix"], identity[1])
    train_rows, unknown = defaultdict(Counter), 0
    for cluster in SITES:
        with open(MAIN / f"train/datasets/{cluster}.jsonl") as fh:
            for line in fh:
                row = json.loads(line)
                site = prefix_site.get(row["prompt"][0]["content"][:70])
                unknown += site is None
                train_rows[cluster][site or "?"] += 1
                del row
    cluster_rows = {c: v["n"] for c, v in json.loads((MAIN / "train/datasets/manifest.json").read_text())["clusters"].items()}
    assert all(sum(train_rows[c].values()) == cluster_rows[c] for c in SITES)

    engines, scope, evals, found = {"teacher_run": teacher}, {}, {}, {}
    for name in ("base", "adapter", "cheap", "self"):
        if inputs[name] is None:
            continue
        run_id, eval_id = inputs[name]
        found[name] = m = manifest(run_id)
        assert m["status"] == "done" and m["type"] == "replay" and m["source_run_id"] == teacher_id, run_id
        engines[name] = load_calls(run_id)
        evals[name] = eval_rows(eval_id, run_id)[0] if eval_id else None
        replayed = {q for q, _, _ in engines[name]}
        scope[name] = (None if replayed == all_questions else replayed, m.get("call_sites"))
        assert not set(engines[name]) - set(teacher), f"{run_id} has invocations its source lacks"
    routed = Counter()
    for (q, site, _), attempts in engines["adapter"].items():
        routed[(site, attempts[0]["cluster"])] += 1

    def column_cell(name, site, questions=None):
        if name == "trivial":
            if site in NOT_DEFINED:
                return NOT_DEFINED[site]
            return cell(site, teacher, extra, trivial, None, None, questions)
        if inputs[name] is None:
            return NOT_RUN
        only, sites = scope[name]
        if sites is not None and site not in sites:
            return f"{NOT_RUN} at this call site (the replay covered {', '.join(sorted(sites))})"
        if site in GOLD_CALL_SITES and evals[name] is None:
            return "replayed, but no per-call evaluation exists"
        subset = only if questions is None else (questions if only is None else questions & only)
        return cell(site, teacher, extra, engines[name], evals[name], teacher_eval, subset)

    rows, costs = {}, {}
    for site in ORDER:
        mine = [i for i in teacher if i[1] == site]
        rows[site] = {
            "cluster": CLUSTER_OF_SITE[site],
            "routed_by_b4_replay": {c: n for (s, c), n in sorted(routed.items()) if s == site},
            "n_questions": len({q for q, _, _ in mine}), "n_calls": len(mine),
            "distinct_prompts": len({extra[i]["prompt"] for i in mine}),
            "teacher_unparsed": sum(not teacher[i][-1]["parsed_ok"] for i in mine),
            "train_rows_site": sum(train_rows[c].get(site, 0) for c in SITES),
            "train_rows_site_by_cluster": {c: train_rows[c][site] for c in SITES if train_rows[c].get(site)},
            "train_rows_cluster": cluster_rows[CLUSTER_OF_SITE[site]],
            "primary_metric": PRIMARY[site],
            "columns": {name: column_cell(name, site) for name, _ in COLUMNS}}
        if site in GOLD_CALL_SITES and mine:
            c = cell(site, teacher, extra, teacher, teacher_eval, teacher_eval)
            rows[site]["teacher_registered_run_ex"] = c
        costs[site] = {"teacher (registered B0 run)": cost(site, teacher)}
        for name, label in COLUMNS[1:]:
            costs[site][label] = NOT_RUN if inputs[name] is None else cost(site, engines[name])
        costs[site][COLUMNS[0][1]] = "no model call"

    levels = ["simple", "moderate", "challenging"]
    by_difficulty = {}
    for level in levels:
        questions = {q for q in all_questions if difficulty.get(q) == level}
        by_difficulty[level] = {}
        for site in GOLD_CALL_SITES:
            entry = {name: column_cell(name, site, questions) for name, _ in COLUMNS}
            entry["teacher_registered_run_ex"] = cell(site, teacher, extra, teacher, teacher_eval, teacher_eval, questions)
            by_difficulty[level][site] = entry

    label = (f"EXPLORATORY, no verdict. Per call site, on the teacher's inputs ({split} split): each engine answers the "
             f"calls of the teacher's registered run `{teacher_id}`. Descriptive only; unadjusted 95% intervals; "
             "these numbers allocate nothing and enter no decision rule.")
    result = {"label": label, "split": split, "inputs": inputs, "seed": SEED, "n_boot": N_BOOT,
              "interval": "percentile bootstrap that resamples questions, all of a question's calls together",
              "trivial_rule_stats": dict(stats), "unclassified_train_rows": unknown, "cluster_train_rows": cluster_rows,
              "self_replay_questions": len(scope["self"][0]) if scope.get("self") and scope["self"][0] else None,
              "rows": rows, "sql_by_difficulty": by_difficulty, "latency_and_tokens": costs}

    L = [f"# Per-call-site table, {split} split — exploratory, on the teacher's inputs", "", f"> {label}", ""]
    L += ["Inputs:", ""]
    for name in ("teacher", "base", "adapter", "cheap", "self"):
        L.append(f"- {name}: " + (NOT_RUN if inputs[name] is None else " ; ".join(f"`{x}`" for x in inputs[name] if x)))
    n_self = result["self_replay_questions"]
    L += ["", f"Intervals: 95%, unadjusted, percentile bootstrap over QUESTIONS ({N_BOOT} resamples, seed {SEED}); all of a "
          "question's calls stay together. The agent's steps get no interval: their effective sample is the number of "
          "distinct prompts.",
          f"The teacher's repetition covers {n_self if n_self else 'all'} questions"
          + (" (a seeded sample in the test's difficulty mix); its cells are computed on those questions only." if n_self else "."),
          "", "## Call site × the five columns", "",
          "| call site (cluster) | questions / calls (distinct prompts; teacher outputs that did not parse) | training rows: "
          "call site / its cluster | primary metric | " + " | ".join(lbl for _, lbl in COLUMNS) + " |",
          "|---|---|---|---|" + "---|" * len(COLUMNS)]
    for site in ORDER:
        r = rows[site]
        route = ", ".join(f"{n} routed to {c}" for c, n in r["routed_by_b4_replay"].items())
        site_rows = " + ".join(f"{n} in {c}" for c, n in r["train_rows_site_by_cluster"].items())
        L.append(f"| {site} ({r['cluster']}; B4's replay: {route}) | {r['n_questions']} / {r['n_calls']} "
                 f"({r['distinct_prompts']}; {r['teacher_unparsed']}) | {r['train_rows_site']} ({site_rows}) / "
                 f"{r['train_rows_cluster']} | {r['primary_metric']} | "
                 + " | ".join(render(site, r["columns"][name], name) for name, _ in COLUMNS) + " |")
    L += ["", "The teacher's EX in its registered run, on all invocations of the call sites with gold:", ""]
    for site in GOLD_CALL_SITES:
        e = rows[site].get("teacher_registered_run_ex")
        if e and e.get("n_calls"):
            L.append(f"- {site}: EX {e['counts']['ex']}/{e['counts']['n_gold']} = {show(e, 'ex')} "
                     f"(n = {e['n_questions']} questions, {e['n_calls']} calls)")
    L += ["", "Clusters (a cluster's row lists its call sites; metrics that differ are not pooled):", ""]
    for cluster, sites in SITES.items():
        L.append(f"- {cluster} ({cluster_rows[cluster]} training rows): " + ", ".join(sites))
    L += ["", "Trivial rules used: filter_column: always \"No\". select_tables / select_columns: every table (every column) "
          "of the CREATE TABLE blocks in the call's own prompt. Agent steps: a lookup by prompt, built from the c3 training "
          f"rows (prompt → the action those rows give it; the SPEC does not name the lookup's source). Lookup: "
          f"{stats['lookup_hit']} calls found their prompt, {stats['lookup_miss']} did not, "
          f"{stats['lookup_not_unanimous']} hit a prompt whose training rows give more than one action.",
          "", "Warnings published with the table (SPEC 6.1): exact-set agreement on keywords is rarely met even by the "
          "teacher, so a low value there says little about the decision. The agent's steps have about 11 distinct prompts "
          "(the prompt does not depend on the question), so their effective n is the number of steps, not of calls. "
          "Repair has a small n (the teacher hardly repairs) and 6 training rows, and is read as such. Raw agreement is "
          "not the primary metric where the teacher's answers are imbalanced or its self-agreement is below 95%.",
          "", "## SQL generation and repair by difficulty (exploratory)", "",
          "| difficulty | call site | teacher, registered run | " + " | ".join(lbl for _, lbl in COLUMNS) + " |",
          "|---|---|---|" + "---|" * len(COLUMNS)]
    for level in levels:
        for site in GOLD_CALL_SITES:
            entry = by_difficulty[level][site]
            t = entry["teacher_registered_run_ex"]
            t_text = (f"EX {t['counts']['ex']}/{t['counts']['n_gold']} = {show(t, 'ex')} (n = {t['n_questions']} questions, "
                      f"{t['n_calls']} calls)") if t.get("n_calls") else "no invocation at this difficulty"
            L.append(f"| {level} | {site} | {t_text} | " + " | ".join(
                (render(site, entry[name], name) if not (isinstance(entry[name], dict) and not entry[name].get("n_calls"))
                 else "no invocation at this difficulty in this column's questions") for name, _ in COLUMNS) + " |")
    L += ["", "## Latency and tokens per call, by engine (final attempt of each invocation)", "",
          "| call site | engine | calls | latency p50 / p95 (ms) | mean tokens in / out per call | attempts per invocation |",
          "|---|---|---|---|---|---|"]
    for site in ORDER:
        for engine, c in costs[site].items():
            if isinstance(c, str) or c is None:
                L.append(f"| {site} | {engine} | {c or 'no invocation'} | {c or '—'} | {c or '—'} | {c or '—'} |")
            else:
                L.append(f"| {site} | {engine} | {c['n_calls']} | {c['latency_ms_p50']} / {c['latency_ms_p95']} | "
                         f"{c['tokens_in_mean']:.0f} / {c['tokens_out_mean']:.0f} | {c['attempts_per_invocation']:.3f} |")
    L += ["", "The teacher's repetition rows cover its sampled questions only. Latency is as each run logged it, under that "
          "run's own concurrency and serving; it is not a load test.", ""]
    (OUT / f"table_{split}_registered.md").write_text("\n".join(L) + "\n")
    (OUT / f"table_{split}_registered.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
