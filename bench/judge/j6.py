"""J6 · S4, the choice of the SLM (SPEC §4 S4, D12), written as the `choice` fact.

Reads the `zeroshot` executions (a `replay` of the teacher's `calib` inputs to each base candidate,
`--engine slm:<candidate>`, F1) and their per-call evaluations (F2), plus the teacher's.

The pre-registered tie-break: **higher per-call-site accuracy on calib; tie: smaller footprint**.
Made operational as:
- a call site's score is J2's: execution accuracy where there is gold (generation, repair), and
  agreement with the teacher elsewhere;
- a candidate's score is the **mean over call sites**, each call site weighing the same, so the
  column filter (dozens of calls per question) does not decide alone; call sites the teacher never
  reached on calib are left out, and every candidate is scored on the same ones;
- a tie is a score within `selection.tie_tolerance` of the best; among tied candidates, the smallest
  `selection.footprint_gb` wins, and a footprint that is not recorded, or equal, leaves the choice
  undecided, which is an error rather than a silent pick.
"""
from typing import Any, Dict, Optional, Tuple

from bench.contracts.facts import write_fact
from bench.judge import j2
from bench.judge.base import JudgmentError, write_result

JUDGMENT = "J6"
# the configuration keys J6 reads, recorded in the result's `reads` (design §6.2)
CONFIG_KEYS = ("roles.slm_candidates", "selection.footprint_gb", "selection.tie_tolerance", "selection.triage")


def scores(per_call_site: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    return {site: j2.site_score(entry) for site, entry in per_call_site.items()}


def choose(candidates: Dict[str, Dict[str, Dict[str, Any]]], footprint_gb: Dict[str, Optional[float]],
           tie_tolerance: float) -> Tuple[str, Dict[str, Any]]:
    """The chosen candidate and the table behind it; `candidates` is {name: J2 per_call_site}."""
    if not candidates:
        raise JudgmentError("no zero-shot execution to choose from")
    by_candidate = {name: scores(table) for name, table in candidates.items()}
    sites = set.intersection(*({s for s, v in table.items() if v is not None} for table in by_candidate.values()))
    if not sites:
        raise JudgmentError("the candidates share no call site with a score")
    means = {name: sum(table[s] for s in sites) / len(sites) for name, table in by_candidate.items()}
    best = max(means.values())
    tied = sorted(name for name, mean in means.items() if best - mean <= tie_tolerance)
    decided_by = "score"
    if len(tied) > 1:
        missing = [name for name in tied if footprint_gb.get(name) is None]
        if missing:
            raise JudgmentError(f"tie between {tied}: selection.footprint_gb is not recorded for {missing}")
        smallest = min(footprint_gb[name] for name in tied)
        tied = [name for name in tied if footprint_gb[name] == smallest]
        if len(tied) > 1:
            raise JudgmentError(f"tie between {tied} in score and in footprint: the pre-registered tie-break does not decide")
        decided_by = "footprint"
    table = {"call_sites": sorted(sites), "by_candidate": by_candidate, "score": means,
             "footprint_gb": {name: footprint_gb.get(name) for name in candidates},
             "tie_tolerance": tie_tolerance, "decided_by": decided_by}
    return tied[0], table


def run(zeroshots: Dict[str, str], teacher_eval_run_id: str, config: Dict[str, Any]):
    """`zeroshots` is {replay run id: its per-call eval run id}. Returns (result path, fact path)."""
    names = {c["name"] for c in config["roles"].get("slm_candidates") or []}
    candidates, reads, sources = {}, {}, set()
    for replay_run_id, eval_run_id in sorted(zeroshots.items()):
        engine = j2.engine_of(replay_run_id)
        name = engine[len("slm:"):] if engine.startswith("slm:") and "+lora:" not in engine else None
        if name not in names:
            raise JudgmentError(f"{replay_run_id} ran {engine!r}, not the base of a candidate in roles.slm_candidates")
        if name in candidates:
            raise JudgmentError(f"two zero-shot executions of {name}")
        run_reads, result = j2.judge_replay(replay_run_id, eval_run_id, teacher_eval_run_id)
        if result["split"] != "calib":
            raise JudgmentError(f"{replay_run_id} is on {result['split']!r}: S4 selects on calib only")
        if result["call_sites"] is not None:
            raise JudgmentError(f"{replay_run_id} replayed only {result['call_sites']}: S4 needs a complete zero-shot replay")
        sources.add(run_reads["teacher"]["run_id"])
        candidates[name] = result["per_call_site"]
        reads[name] = run_reads
    if set(candidates) != names:
        raise JudgmentError(f"S4 compares exactly the configured candidates {sorted(names)}; "
                            f"zero-shot executions were given for {sorted(candidates)}")
    if len(sources) != 1:
        raise JudgmentError(f"the zero-shot executions replay different teacher runs: {sorted(sources)}")
    choice, table = choose(candidates, config["selection"]["footprint_gb"], config["selection"]["tie_tolerance"])
    fact = write_fact(JUDGMENT, "choice", {"slm": choice})
    result = {"choice": choice, "choice_fact": {"sha256": fact.parent.name}, **table,
              "per_call_site": candidates, "triage": config["selection"].get("triage") or []}
    return write_result(JUDGMENT, {**reads, "config": list(CONFIG_KEYS)}, result), fact
