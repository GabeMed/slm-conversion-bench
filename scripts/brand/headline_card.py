"""The README's two charts under the bars card (light and dark), the hybrid against the agent on the large LLM: the
cost per call (cost-{theme}.svg) and the time to answer (time-{theme}.svg) as two bars, Qwen3-8B fine-tuned against
Claude Sonnet 5.5, with the makers' logos.

    python scripts/brand/headline_card.py --out .github/assets

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
W, X, R = 1200, 72, 1128
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



def _logo(theme: str, maker: str, x: float, y: float, s: float) -> str:
    uri = nest.logo(maker, theme)
    return (f'<image x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" height="{s:.1f}" preserveAspectRatio="xMidYMid meet" href="{uri}"/>'
            if uri else "")


def _defs(P: Dict[str, Any], theme: str, gid: str) -> str:
    so = (".28", ".2") if theme == "dark" else (".03", ".02")
    return (f'<defs><linearGradient id="{gid}-s" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{P["g"][0]}"/>'
            f'<stop offset="1" stop-color="{P["g"][1]}"/></linearGradient>'
            f'<linearGradient id="{gid}-b" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{P["k"][0]}"/>'
            f'<stop offset="1" stop-color="{P["k"][1]}"/></linearGradient>'
            f'<filter id="{gid}-sh" x="-4%" y="-4%" width="108%" height="114%">'
            f'<feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#000" flood-opacity="{so[0]}"/>'
            f'<feDropShadow dx="0" dy="9" stdDeviation="4" flood-color="#000" flood-opacity="{so[1]}"/></filter></defs>')


def _card(P: Dict[str, Any], theme: str, gid: str, x: float, y: float, w: float, h: float) -> str:
    """the site's card in relief: a .93 px ring and two soft shadows, no border"""
    ring = "rgba(0,0,0,.5)" if theme == "dark" else "rgba(0,0,0,.08)"
    return (nest.rrect(x, y, w, h, 24, f'fill="{P["card"]}" filter="url(#{gid}-sh)"')
            + nest.rrect(x + 0.47, y + 0.47, w - 0.93, h - 0.93, 23.5, f'fill="none" stroke="{ring}" stroke-width=".93"'))


def draw_chart(v: Dict[str, Any], theme: str, which: str) -> Optional[str]:
    """one comparison as two bars: cost per call or time to answer (median per question)"""
    if which == "cost":
        metric, q, s_, qt, st, mx = ("Cost per call", v["hyb_call_full"], v["ref_call"], _usd(v["hyb_call_full"]),
                                     _usd(v["ref_call"]), v["ref_call"])
        note = f"GPU in full use; {_usd(v['hyb_call_low'])} with it {v['low']} busy"
    else:
        if not (v["ref_s"] and v["hyb_s"]):
            return None
        metric, q, s_, qt, st, mx = ("Time to answer", v["hyb_s"], v["ref_s"], f"{v['hyb_s']:.1f} s", f"{v['ref_s']:.1f} s",
                                     max(v["hyb_s"], v["ref_s"]))
        note = "median per question, as the test ran"
    from results_card import _chip
    P = nest.PAL[theme]
    gid = f"ch-{which}-{theme}"
    body: List[str] = [_defs(P, theme, gid)]
    body.append(nest.text(X, 56, metric, 36, P["ink"], 450, tracking=-1.2))
    body.append(nest.text(X, 86, note, 18, P["dim"], 400, tracking=-0.2))
    top, ph = 112, 360
    body.append(_card(P, theme, gid, X, top, CW, ph))
    base, hmax, bw, cs = top + 250, 190, 132, 46
    body.append(f'<rect x="{X + 36}" y="{base:.1f}" width="{CW - 72}" height="1.5" fill="{P["line"]}"/>')
    for k, (val, txt, green, maker, name) in enumerate(((q, qt, True, "qwen", "Qwen3-8B fine-tuned"),
                                                        (s_, st, False, "anthropic", "Claude Sonnet 5.5"))):
        cx = X + CW * (0.36 if k == 0 else 0.64)
        h = max(5.0, hmax * val / mx)
        body.append(nest.rrect(cx - bw / 2, base - h, bw, h, min(10, h / 2), f'fill="url(#{gid}-{"s" if green else "b"})"'))
        body.append(nest.rrect(cx - bw / 2 + 0.5, base - h + 0.5, bw - 1, h - 1, min(9.5, h / 2),
                               f'fill="none" stroke="{P["gEdge"] if green else P["kEdge"]}" stroke-width="1"'))
        body.append(nest.text(cx, base - h - 14, txt, 30, P["ink"] if green else P["muted"], 500, mono=True,
                              anchor="middle", tracking=-0.8))
        body.append(_chip(P, theme, maker, name[0], cx - cs / 2, base + 16, cs))
        body.append(nest.text(cx, base + 16 + cs + 26, name, 18, P["ink"] if green else P["muted"], 500, anchor="middle",
                              tracking=-0.3))
    label = f"{metric}, Qwen3-8B fine-tuned {qt} against Claude Sonnet 5.5 {st} ({note})."
    return nest.svg(W, top + ph + 34, body, label)


def write(out: Path, v: Dict[str, Any]) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in nest.THEMES:
        for name, svg in (("cost", draw_chart(v, theme, "cost")), ("time", draw_chart(v, theme, "time"))):
            if svg:
                p = out / f"{name}-{theme}.svg"
                p.write_text(svg)
                paths.append(p)
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", default=str(REPORT), help="report.json (default: results/report/report.json)")
    ap.add_argument("--tables", default=str(TABLES), help="the cost and latency tables (default: results/tables/…)")
    ap.add_argument("--out", default=".", help="where to write cost- and time-{light,dark}.svg")
    args = ap.parse_args(argv)
    v = numbers(json.loads(Path(args.report).read_text()), Path(args.tables))
    for p in write(Path(args.out), v):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
