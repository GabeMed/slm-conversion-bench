"""The headline of the results as an animated image (1200 × 630, light and dark): the hybrid against the agent on the
large LLM, shown one comparison at a time as two bars (accuracy, cost per call, time to answer), then as three cards
with how much better the fine-tuned Qwen3-8B did than Claude Sonnet 5.5 (relative %, and × for cost). CSS animation
only, so it plays in an <img> (GitHub included); with reduced motion, the three cards alone.

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


CYCLE = 16  # seconds: accuracy, cost per call, time to answer (3.2 s each), then the three cards (6.4 s)

CSS = """
.sc,.fc{opacity:0}
.s1{animation:s1 %(c)ds infinite}.s2{animation:s2 %(c)ds infinite}.s3{animation:s3 %(c)ds infinite}
@keyframes s1{0%%{opacity:0}2%%{opacity:1}18%%{opacity:1}20%%{opacity:0}100%%{opacity:0}}
@keyframes s2{0%%,20%%{opacity:0}22%%{opacity:1}38%%{opacity:1}40%%{opacity:0}100%%{opacity:0}}
@keyframes s3{0%%,40%%{opacity:0}42%%{opacity:1}58%%{opacity:1}60%%{opacity:0}100%%{opacity:0}}
.bar{transform-box:fill-box;transform-origin:50%% 100%%}
.g1{animation:g1 %(c)ds infinite}.g2{animation:g2 %(c)ds infinite}.g3{animation:g3 %(c)ds infinite}
@keyframes g1{0%%,3%%{transform:scaleY(0)}9%%,100%%{transform:scaleY(1)}}
@keyframes g2{0%%,23%%{transform:scaleY(0)}29%%,100%%{transform:scaleY(1)}}
@keyframes g3{0%%,43%%{transform:scaleY(0)}49%%,100%%{transform:scaleY(1)}}
.v1{animation:v1 %(c)ds infinite}.v2{animation:v2 %(c)ds infinite}.v3{animation:v3 %(c)ds infinite}
@keyframes v1{0%%,8%%{opacity:0}11%%,100%%{opacity:1}}
@keyframes v2{0%%,28%%{opacity:0}31%%,100%%{opacity:1}}
@keyframes v3{0%%,48%%{opacity:0}51%%,100%%{opacity:1}}
.f1{animation:f1 %(c)ds infinite}.f2{animation:f2 %(c)ds infinite}.f3{animation:f3 %(c)ds infinite}
@keyframes f1{0%%,60%%{opacity:0;transform:translateY(14px)}63%%,97%%{opacity:1;transform:none}100%%{opacity:0}}
@keyframes f2{0%%,64%%{opacity:0;transform:translateY(14px)}67%%,97%%{opacity:1;transform:none}100%%{opacity:0}}
@keyframes f3{0%%,68%%{opacity:0;transform:translateY(14px)}71%%,97%%{opacity:1;transform:none}100%%{opacity:0}}
@media (prefers-reduced-motion:reduce){.sc{animation:none;opacity:0}.fc{animation:none;opacity:1;transform:none}}
""" % {"c": CYCLE}

ICONS = {  # 24 x 24, stroked, as the site's line icons
    "accuracy": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/>',
    "cost": '<circle cx="12" cy="12" r="9"/><path d="M15 8.6c-.6-.9-1.7-1.4-3-1.4-1.8 0-3 .9-3 2.3 0 3.1 6 1.6 6 4.8 0 1.4-1.3 2.4-3.1 2.4-1.4 0-2.6-.6-3.2-1.6M12 5.6v1.6M12 16.7v1.7"/>',
    "time": '<path d="M13 2.5 4 13.5h7l-1 8 9-11h-7l1-8z"/>',
}


def _logo(theme: str, maker: str, x: float, y: float, s: float) -> str:
    uri = nest.logo(maker, theme)
    return (f'<image x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" height="{s:.1f}" preserveAspectRatio="xMidYMid meet" href="{uri}"/>'
            if uri else "")


def draw(v: Dict[str, Any], theme: str) -> str:
    """an animated SVG: the three comparisons one by one as bar charts (accuracy, cost per call, time to answer),
    fine-tuned Qwen3-8B against Claude Sonnet 5.5, then three cards with how much better the small model did. CSS
    animations only (they play in an <img>, GitHub included); with reduced motion, only the three cards"""
    from results_card import _chip
    P = nest.PAL[theme]
    body: List[str] = []
    gid = f"hc-{theme}"
    body.append(f'<defs><linearGradient id="{gid}-s" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{P["g"][0]}"/>'
                f'<stop offset="1" stop-color="{P["g"][1]}"/></linearGradient>'
                f'<linearGradient id="{gid}-b" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{P["k"][0]}"/>'
                f'<stop offset="1" stop-color="{P["k"][1]}"/></linearGradient></defs>')
    body.append(nest.lockup(X, 62, 28, P["ink"]))
    body.append(nest.text(R, 62, URL, 18, P["dim"], 500, mono=True, anchor="end", tracking=-0.2))
    # the title, with the makers' logos in the line
    ty, ts, ls = 132, 40, 40
    x = X
    body.append(_logo(theme, "qwen", x, ty - ls + 6, ls)); x += ls + 12
    t1 = "Qwen3-8B fine-tuned"
    body.append(nest.text(x, ty, t1, ts, P["ink"], 450, tracking=-1.2)); x += nest.width(t1, ts, weight=450, tracking=-1.2) + 18
    body.append(nest.text(x, ty, "vs", ts, P["dim"], 400, tracking=-1.2)); x += nest.width("vs", ts, tracking=-1.2) + 18
    body.append(_logo(theme, "anthropic", x, ty - ls + 6, ls)); x += ls + 12
    body.append(nest.text(x, ty, "Claude Sonnet 5.5", ts, P["ink"], 450, tracking=-1.2))

    acc = (v["hyb_ex"] - v["ref_ex"]) / v["ref_ex"]
    pct = lambda f: f"{f * 100:.1f}%"
    charts = [
        {"metric": "Accuracy", "pill": f"{acc * 100:+.1f}% accuracy", "q": v["hyb_ex"], "s": v["ref_ex"],
         "qt": pct(v["hyb_ex"]), "st": pct(v["ref_ex"]), "max": 1.0,
         "card": (f"{acc * 100:+.1f}%", "more accurate", "accuracy")},
        {"metric": "Cost per call", "pill": f"{v['x_full']:.0f}× cheaper", "q": v["hyb_call_full"], "s": v["ref_call"],
         "qt": _usd(v["hyb_call_full"]), "st": _usd(v["ref_call"]), "max": v["ref_call"],
         "card": (f"{v['x_full']:.0f}×", "cheaper per call", "cost")},
    ]
    if v["ref_s"] and v["hyb_s"]:
        lat = (v["ref_s"] - v["hyb_s"]) / v["ref_s"]
        charts.append({"metric": "Time to answer", "pill": f"{lat * 100:.1f}% faster", "q": v["hyb_s"], "s": v["ref_s"],
                       "qt": f"{v['hyb_s']:.1f} s", "st": f"{v['ref_s']:.1f} s", "max": max(v["hyb_s"], v["ref_s"]),
                       "card": (f"{lat * 100:.1f}%", "faster answers", "time")})
    top, ph = 176, 430
    # the scenes: one big chart each
    for i, c in enumerate(charts, 1):
        g = [f'<g class="sc s{i}">']
        g.append(nest.rrect(X, top, CW, ph, 24, f'fill="{P["card"]}"'))
        g.append(nest.rrect(X + 0.5, top + 0.5, CW - 1, ph - 1, 23.5, f'fill="none" stroke="{P["line"]}" stroke-width="1"'))
        g.append(nest.text(X + 36, top + 50, c["metric"].upper(), 16, P["dim"], 500, mono=True, tracking=1.5))
        pw = nest.width(c["pill"], 24, weight=500, tracking=-0.4) + 34
        g.append(f'<g class="v{i}">' + nest.rrect(R - 36 - pw, top + 24, pw, 42, 21, f'fill="{P["greenBg"]}"')
                 + nest.text(R - 36 - pw + 17, top + 53, c["pill"], 24, P["green"], 500, tracking=-0.4) + "</g>")
        base, hmax, bw, cs = top + 326, 210, 132, 48
        g.append(f'<rect x="{X + 36}" y="{base:.1f}" width="{CW - 72}" height="1.5" fill="{P["line"]}"/>')
        for k, (val, txt, green, maker, name) in enumerate(((c["q"], c["qt"], True, "qwen", "Qwen3-8B fine-tuned"),
                                                            (c["s"], c["st"], False, "anthropic", "Claude Sonnet 5.5"))):
            cx = X + CW * (0.36 if k == 0 else 0.64)
            h = max(5.0, hmax * val / c["max"])
            g.append(f'<g class="bar g{i}">'
                     + nest.rrect(cx - bw / 2, base - h, bw, h, min(10, h / 2), f'fill="url(#{gid}-{"s" if green else "b"})"')
                     + nest.rrect(cx - bw / 2 + 0.5, base - h + 0.5, bw - 1, h - 1, min(9.5, h / 2),
                                  f'fill="none" stroke="{P["gEdge"] if green else P["kEdge"]}" stroke-width="1"') + "</g>")
            g.append(f'<g class="v{i}">' + nest.text(cx, base - h - 14, txt, 30, P["ink"] if green else P["muted"], 500,
                                                       mono=True, anchor="middle", tracking=-0.8) + "</g>")
            g.append(_chip(P, theme, maker, name[0], cx - cs / 2, base + 16, cs))
            g.append(nest.text(cx, base + 16 + cs + 26, name, 18, P["ink"] if green else P["muted"], 500, anchor="middle",
                               tracking=-0.3))
        g.append("</g>")
        body.append("".join(g))
    # the end: three cards, how much better the fine-tuned Qwen3-8B did than Claude Sonnet 5.5
    gap = 24
    tw = (CW - gap * (len(charts) - 1)) / len(charts)
    for i, c in enumerate(charts, 1):
        big, label, icon = c["card"]
        x = X + (i - 1) * (tw + gap)
        g = [f'<g class="fc f{i}">']
        g.append(nest.rrect(x, top, tw, ph, 24, f'fill="{P["card"]}"'))
        g.append(nest.rrect(x + 0.5, top + 0.5, tw - 1, ph - 1, 23.5, f'fill="none" stroke="{P["line"]}" stroke-width="1"'))
        g.append(f'<circle cx="{x + 36 + 32:.1f}" cy="{top + 36 + 32:.1f}" r="32" fill="{P["greenBg"]}"/>')
        g.append(f'<g transform="translate({x + 36 + 14:.1f} {top + 36 + 14:.1f}) scale(1.5)" fill="none" stroke="{P["green"]}" '
                 f'color="{P["green"]}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONS[icon]}</g>')
        size = 104
        while size > 64 and nest.width(big, size, weight=500, tracking=-size * 0.05) > tw - 72:
            size -= 4
        g.append(nest.text(x + 36, top + 214 + size * 0.2, big, size, P["ink"], 500, tracking=round(-size * 0.05, 1)))
        g.append(nest.text(x + 36, top + 292, label, 28, P["ink"], 450, tracking=-0.7))
        g.append(nest.text(x + 36, top + 336, "than", 18, P["dim"], 400))
        lx = x + 36 + nest.width("than", 18) + 10
        g.append(_logo(theme, "anthropic", lx, top + 320, 20))
        g.append(nest.text(lx + 28, top + 336, "Claude Sonnet 5.5", 18, P["dim"], 400))
        g.append(nest.text(x + 36, top + 388, f"{c['qt']} vs {c['st']}", 18, P["faint"], 500, mono=True, tracking=-0.4))
        g.append("</g>")
        body.append("".join(g))
    label = ("Fine-tuned Qwen3-8B vs Claude Sonnet 5.5, shown one by one, then as three cards: "
             + "; ".join(f"{c['card'][0]} {c['card'][1]} ({c['metric'].lower()} {c['qt']} against {c['st']})" for c in charts)
             + ".")
    return nest.svg(W, H, body, label, css=CSS)


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
