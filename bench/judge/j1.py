"""J1 · execution accuracy (SPEC 6.3, D9), read from `eval` executions (`bench eval`), the only source of EX.

`ex_table` is one row per (arm, split, question); `ex_summary` is EX per arm and split, per
difficulty, and without the date-dependent golds and without the golds with LIMIT (the sensitivity
the design asks for). EX is a fraction of questions.
"""
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from bench.judge import JudgeError

DIFFICULTY_ORDER = ("simple", "moderate", "challenging")


def ex_table(eval_run_dirs: Iterable[Path]) -> List[Dict[str, Any]]:
    """{"arm", "split", "question_id", "difficulty", "correct", "date_dependent", "limit",
    "gold_error", "source_run_id"} for every result of every end-to-end eval execution given.
    An arm and split scored twice is refused: SPEC 6.1 evaluates each configuration once."""
    rows: List[Dict[str, Any]] = []
    seen: Dict[tuple, str] = {}
    for run_dir in map(Path, eval_run_dirs):
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("type") != "eval" or manifest.get("per_call"):
            raise JudgeError(f"{run_dir.name} is not an end-to-end eval execution")
        if not manifest.get("arm"):
            raise JudgeError(f"{run_dir.name} does not record the arm it scored")
        results = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
        if len(results) != manifest["n"]:
            raise JudgeError(f"{run_dir.name}: {len(results)} results, its manifest says {manifest['n']}")
        for result in results:
            key = (manifest["arm"], manifest["split"], result["question_id"])
            if key in seen:
                raise JudgeError(f"{key} is scored by both {seen[key]} and {run_dir.name}")
            seen[key] = run_dir.name
            rows.append({"arm": manifest["arm"], "split": manifest["split"], "question_id": result["question_id"],
                         "difficulty": result["difficulty"], "correct": bool(result["correct"]),
                         "date_dependent": bool(result["gold_date_substituted"]), "limit": bool(result["gold_has_limit"]),
                         "gold_error": result["gold_error"] is not None, "source_run_id": manifest["source_run_id"]})
    return rows


def _ex(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n, hits = len(rows), sum(r["correct"] for r in rows)
    return {"n": n, "correct": hits, "ex": hits / n if n else None}


def ex_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """EX per (arm, split), sorted by arm then split, each with its difficulties (in the Mini-Dev
    order, every one present reported) and the two sensitivity subsets."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["arm"], row["split"]), []).append(row)
    summary = []
    for (arm, split), group in sorted(groups.items()):
        difficulties = sorted({r["difficulty"] for r in group},
                              key=lambda d: (DIFFICULTY_ORDER.index(d) if d in DIFFICULTY_ORDER else len(DIFFICULTY_ORDER), str(d)))
        summary.append({
            "arm": arm, "split": split, **_ex(group),
            "by_difficulty": {d: _ex([r for r in group if r["difficulty"] == d]) for d in difficulties},
            "without_date_dependent": _ex([r for r in group if not r["date_dependent"]]),
            "without_limit": _ex([r for r in group if not r["limit"]]),
            "gold_errors": sum(r["gold_error"] for r in group),
        })
    return summary


def correct(rows: List[Dict[str, Any]], arm: str, split: str) -> Dict[str, bool]:
    """{question_id: correct} of one arm on one split: the input J4 pairs by question_id."""
    return {r["question_id"]: r["correct"] for r in rows if r["arm"] == arm and r["split"] == split}
