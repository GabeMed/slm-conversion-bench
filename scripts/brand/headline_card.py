"""The headline of the results as an image (1200 × 630, light and dark): the hybrid against the agent on the large LLM,
in three numbers: how many times cheaper per call, the accuracy difference, the answer time.

    python scripts/brand/headline_card.py --out .github/assets      # writes headline-light.svg, headline-dark.svg

Every number is read, not typed: cost and accuracy from `results/report/report.json` (the registered report: each
arm's cost per correct answer at the GPU's full and lowest utilization, its accuracy, its number of calls, and the
paired test B0|B5), the answer time from `results/tables/cost_latency_tables.md` (T2, the median time per question
as the test ran, a measured number outside the verdicts). Cost per call = the arm's cost per correct answer × its
correct answers ÷ its calls. It changes no file of the analysis code and no result.

Standard library only.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results" / "report" / "report.json"
TABLES = ROOT / "results" / "tables" / "cost_latency_tables.md"
W, H, X, R = 1200, 630, 72, 1128
CW = R - X
REF, HYB = "B0", "B5"
URL = "nest.technology/benchmarks"


def q_wall_p50(path: Path, arm: str) -> Optional[float]:
    """T2's median seconds per question for one arm (None if the table or the row is missing)."""
    if not path.exists():
        return None
    lines = path.read_text().split("\n")
    for i, line in enumerate(lines):
        if line.startswith("## T2."):
            head = None
            for row in lines[i + 1:]:
                if row.startswith("## "):
                    break
                if not row.startswith("|") or set(row.replace("|", "").strip()) <= set("-: "):
                    continue
                cells = [c.strip() for c in row.strip("|").split("|")]
                if head is None:
                    head = cells
                elif cells[0] == arm and "q wall p50" in head:
                    return float(cells[head.index("q wall p50")])
    return None


def numbers(report: Dict[str, Any], tables: Path = TABLES) -> Dict[str, Any]:
    a, b = report["arms"][REF], report["arms"][HYB]
    full, low = report["utilizations"][-1], report["utilizations"][0]          # "100%", "20%"

    def per_call(arm: Dict[str, Any], use: Optional[str]) -> float:
        c = arm["cost_per_correct"]
        c = c[use] if use in c else next(iter(c.values()))
        return c * arm["ex"] * arm["n"] / arm["calls"]

    ref_call = per_call(a, None)
    t = report["tests"][f"{REF}|{HYB}"]
    out = {"ref_call": ref_call, "hyb_call_full": per_call(b, full), "hyb_call_low": per_call(b, low),
           "ref_ex": a["ex"], "hyb_ex": b["ex"], "diff": t["diff"], "ci": (t["ci_low"], t["ci_high"]),
           "noninferior": t["noninferior"], "n": a["n"], "low": low,
           "slm_calls": (b.get("engine_shares") or {}).get("slm", {}).get("calls")}
    out["x_full"] = ref_call / out["hyb_call_full"]
    out["x_low"] = ref_call / out["hyb_call_low"]
    out["ref_s"], out["hyb_s"] = q_wall_p50(tables, REF), q_wall_p50(tables, HYB)
    return out


def _usd(x: float) -> str:
    return f"${x:.5f}".rstrip("0") if x < 0.001 else f"${x:.4f}".rstrip("0")


def draw(v: Dict[str, Any], theme: str) -> str:
    P = nest.PAL[theme]
    body: List[str] = []
    body.append(nest.lockup(X, 62, 28, P["ink"]))
    src = f"CHESS ON BIRD · {v['n']} TEST QUESTIONS"
    body.append(nest.text(R, 60, src, 15, P["dim"], 500, mono=True, anchor="end", tracking=1.4))
    # who against whom, in words
    body.append(nest.text(X, 128, "Fine-tuned Qwen3-8B on " + (f"{v['slm_calls'] * 100:.1f}%" if v["slm_calls"] else "most")
                          + " of the agent’s calls,", 34, P["ink"], 450, tracking=-1.1))
    body.append(nest.text(X, 170, "against Claude Sonnet 5.5 on every call", 34, P["dim"], 450, tracking=-1.1))
    # three tiles
    top, th, gap = 214, 330, 24
    tw = (CW - 2 * gap) / 3
    gid = f"hc-{theme}"
    body.append(f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{P["g"][0]}"/>'
                f'<stop offset="1" stop-color="{P["g"][1]}"/></linearGradient></defs>')
    pct = lambda f: f"{f * 100:.1f}%"
    tiles = [
        {"big": f"{v['x_full']:.0f}×", "label": "cheaper per call",
         "lines": [f"{_usd(v['ref_call'])} → {_usd(v['hyb_call_full'])} per call",
                   f"{v['x_low']:.1f}× with the GPU {v['low']} busy"]},
        {"big": f"{v['diff'] * 100:+.1f} pt", "label": "accuracy",
         "lines": [f"{pct(v['hyb_ex'])} against {pct(v['ref_ex'])}",
                   ("the same, within the 5-pt margin" if v["noninferior"] else "below the 5-pt margin")]},
    ]
    if v["ref_s"] and v["hyb_s"]:
        d = (v["hyb_s"] - v["ref_s"]) / v["ref_s"]
        tiles.append({"big": f"{d * 100:+.0f}%".replace("-", "−"), "label": "time to answer",
                      "lines": [f"{v['hyb_s']:.1f} s against {v['ref_s']:.1f} s", "median per question"]})
    for i, t in enumerate(tiles):
        x = X + i * (tw + gap)
        body.append(nest.rrect(x, top, tw, th, 22, f'fill="{P["card"]}"'))
        body.append(nest.rrect(x + 0.5, top + 0.5, tw - 1, th - 1, 21.5, f'fill="none" stroke="{P["line"]}" stroke-width="1"'))
        body.append(nest.rrect(x + 28, top + 30, 44, 8, 4, f'fill="url(#{gid})"'))     # the site's green, as a tab
        size = 104
        while size > 64 and nest.width(t["big"], size, weight=500, tracking=-size * 0.05) > tw - 56:
            size -= 4
        body.append(nest.text(x + 28, top + 60 + size * 0.78, t["big"], size, P["ink"], 500, tracking=round(-size * 0.05, 1)))
        body.append(nest.text(x + 28, top + 206, t["label"], 26, P["ink"], 450, tracking=-0.6))
        for k, line in enumerate(t["lines"]):
            body.append(nest.text(x + 28, top + 252 + k * 28, nest.fit(line, tw - 56, 18, tracking=-0.2), 18, P["dim"], 400,
                                  tracking=-0.2))
    foot = "Qwen3-8B (April 2025) · LoRA r16 · served on one A100 80GB PCIe, RunPod, $1.79/h"
    body.append(nest.text(X, 596, nest.fit(foot, CW - 330, 17, tracking=-0.2), 17, P["dim"], 400, tracking=-0.2))
    body.append(nest.text(R, 596, URL, 18, P["ink"], 500, mono=True, anchor="end", tracking=-0.2))
    label = ("Fine-tuned Qwen3-8B on most of the agent's calls against Claude Sonnet 5.5 on every call: "
             + "; ".join(f"{t['big']} {t['label']} ({', '.join(t['lines'])})" for t in tiles) + ".")
    return nest.svg(W, H, body, label)


def write(out: Path, v: Dict[str, Any]) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in nest.THEMES:
        p = out / f"headline-{theme}.svg"
        p.write_text(draw(v, theme))
        paths.append(p)
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", default=str(REPORT), help="report.json (default: results/report/report.json)")
    ap.add_argument("--tables", default=str(TABLES), help="the cost and latency tables (default: results/tables/…)")
    ap.add_argument("--out", default=".", help="where to write headline-{light,dark}.svg")
    args = ap.parse_args(argv)
    v = numbers(json.loads(Path(args.report).read_text()), Path(args.tables))
    for p in write(Path(args.out), v):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
