"""The results as an image in the look of nest.technology's share images (1200 × 675, light and dark).

    python scripts/brand/results_card.py reports/<sha256>              # writes results-light.svg, results-dark.svg there
    python scripts/brand/results_card.py reports/<sha256> --out .github/assets --title "…"
    python scripts/brand/results_card.py --pending --out .github/assets  # before any result: the arms, no numbers

It reads `report.json` (what `bench report` wrote) and the configuration only to name the models, and draws one bar per
arm: its execution accuracy (EX), and under it the cost per correct query at standard prices (an SLM arm's from its
highest to its lowest utilization). The SLM arms (B3, B4, B5) are green, the others grey, as the site's small and large
models. It changes no file of the analysis code and no result: the image is a picture of report.json.

Standard library and PyYAML only (the `core` environment).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("B0", "B1", "B2-production", "B2-cheap", "B3", "B4", "B5")
SLM_ARMS = ("B3", "B4", "B5")
W, H, X, R = 1200, 675, 72, 1128
CW = R - X
TITLE = "Which LLM calls of a real agent can a fine-tuned SLM take over?"
SOURCE = "CHESS on BIRD · arXiv 2506.02153"
FOOT = "Pre-registered. Every number from report.json."
URL = "nest.technology/benchmarks"


# ---------------------------------------------------------------- names

def pretty(model: Optional[str]) -> str:
    """claude-sonnet-5-5 → Claude Sonnet 5.5; openai/gpt-oss-120b → gpt-oss-120b."""
    if not model:
        return "?"
    name = model.split("/")[-1]
    if name.startswith("claude-"):
        words = name.split("-")[1:]
        version = [w for w in words if w.isdigit()]
        family = [w for w in words if not w.isdigit()]
        return "Claude " + " ".join(w.capitalize() for w in family) + (" " + ".".join(version) if version else "")
    return name


def maker(model: Optional[str]) -> Optional[str]:
    m = (model or "").lower()
    for key, name in (("claude", "anthropic"), ("anthropic", "anthropic"), ("gpt", "openai"), ("openai", "openai"),
                      ("qwen", "qwen"), ("nemotron", "nvidia"), ("nvidia", "nvidia")):
        if key in m:
            return name
    return None


def slm_of(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """The SLM the arms use (S4's choice, `arms.B4.choice`), as (display name, maker); (None, None) before it is chosen."""
    choice = ((config.get("arms") or {}).get("B4") or {}).get("choice") or \
             ((config.get("arms") or {}).get("B3") or {}).get("choice")
    if not choice:
        return None, None
    if choice.endswith(".json") and (ROOT / choice).is_file():   # the registered form: J6's judgment names the SLM
        choice = json.loads((ROOT / choice).read_text()).get("slm") or choice
    roles = config.get("roles") or {}
    for c in roles.get("slm_candidates") or []:
        if choice in (c.get("name"), c.get("base")):
            hf = (c.get("hf") or {}).get("repo")
            return c.get("base") or c.get("name"), maker(hf or c.get("name"))
    return choice, maker(choice)


def arm_names(config: Dict[str, Any]) -> Dict[str, Tuple[str, str, Optional[str]]]:
    """Each arm's model, how the agent uses it, and the model's maker (for the chip)."""
    roles = config.get("roles") or {}
    prod = (roles.get("production_llm") or {}).get("model")
    cheap = (roles.get("cheap_alt") or {}).get("model")
    slm, slm_maker = slm_of(config)
    s = slm or "SLM"
    return {
        "B0": (pretty(prod), "the agent's LLM", maker(prod)),
        "B1": (pretty(cheap), "few-shot", maker(cheap)),
        "B2-production": (pretty(prod), "one call", maker(prod)),
        "B2-cheap": (pretty(cheap), "one call", maker(cheap)),
        "B3": (s, "zero-shot", slm_maker),
        "B4": (s, "fine-tuned", slm_maker),
        "B5": (f"{s} + LLM", "routed per call", slm_maker),
    }


# ---------------------------------------------------------------- numbers

def _usd(x: float) -> str:
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:.3g}" if x >= 0.001 else f"${x:.2e}"


def cost_line(costs: Dict[str, Optional[float]]) -> Optional[str]:
    """The cost per correct query: one figure for an API arm, highest → lowest utilization for an SLM one."""
    found = [(k, v) for k, v in (costs or {}).items() if v]
    if not found:
        return None
    if len(found) == 1:
        return _usd(found[0][1])
    by_u = sorted(found, key=lambda kv: -float(kv[0].rstrip("%") or 0))
    return f"{_usd(by_u[0][1])}–{_usd(by_u[-1][1]).lstrip('$')}"


def bars_of(report: Dict[str, Any], config: Dict[str, Any], only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    names = arm_names(config)
    out = []
    for arm in only or ARMS:
        found = (report.get("arms") or {}).get(arm)
        if not found or found.get("ex") is None:
            continue
        model, variant, mk = names.get(arm, (arm, "", None))
        out.append({"arm": arm, "name": model, "variant": variant, "maker": mk, "f": float(found["ex"]),
                    "txt": f"{100 * found['ex']:.1f}%",
                    "cost": cost_line(found.get("cost_per_correct") or {}), "green": arm in SLM_ARMS})
    return out


def pending_bars(config: Dict[str, Any], only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    names = arm_names(config)
    return [{"arm": a, "name": names[a][0], "variant": names[a][1], "maker": names[a][2], "f": None, "txt": "—",
             "cost": None, "green": a in SLM_ARMS} for a in (only or ARMS)]


# ---------------------------------------------------------------- drawing (bench.js NestShare.draw, as SVG)

def _chip(P: Dict[str, Any], theme: str, mk: Optional[str], letter: str, x: float, y: float, s: float) -> str:
    r = round(s * 0.26)
    out = [nest.rrect(x, y, s, s, r, f'fill="{P["chip"]}"')]
    uri = nest.logo(mk, theme)
    if uri:
        p = round(s * 0.2)
        out.append(f'<image x="{x + p:.1f}" y="{y + p:.1f}" width="{s - 2 * p:.1f}" height="{s - 2 * p:.1f}" '
                   f'preserveAspectRatio="xMidYMid meet" href="{uri}"/>')
    else:
        out.append(nest.text(x + s / 2, y + s / 2 + s * 0.17, letter, round(s * 0.46), P["dim"], 600, anchor="middle"))
    out.append(nest.rrect(x + 0.5, y + 0.5, s - 1, s - 1, r, f'fill="none" stroke="{P["chipLine"]}" stroke-width="1"'))
    return "".join(out)


def draw(bars: List[Dict[str, Any]], theme: str, title: str = "", metric: str = "Execution accuracy",
         source: str = SOURCE, foot: str = FOOT, url: str = URL, pending: Optional[str] = None,
         lowest_use: str = "20%") -> str:
    P = nest.PAL[theme]
    body = []  # no background: the card stands on the page it is shown on
    gid = f"g-{theme}"
    body.append(f'<defs><linearGradient id="{gid}-s" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{P["g"][0]}"/>'
                f'<stop offset="1" stop-color="{P["g"][1]}"/></linearGradient>'
                f'<linearGradient id="{gid}-b" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{P["k"][0]}"/>'
                f'<stop offset="1" stop-color="{P["k"][1]}"/></linearGradient></defs>')
    # top: the brand as in the site's header, the source at the right
    body.append(nest.lockup(X, 64, 28, P["ink"]))
    body.append(nest.text(R, 62, nest.fit(source.upper(), CW - 280, 15, weight=500, mono=True, tracking=1.4), 15, P["dim"],
                          500, mono=True, anchor="end", tracking=1.4))
    # the title (the finding, when there is one): up to 2 lines, smaller if it does not fit; none, the chart moves up
    if title:
        ts, lines = 40, []
        while ts >= 30:
            lines = nest.wrap(title, CW, ts, 3, weight=450, tracking=-ts * 0.032)
            if len(lines) <= 2:
                break
            ts -= 2
        if len(lines) > 2:
            lines = nest.wrap(title, CW, ts, 2, weight=450, tracking=-ts * 0.032)
        lh, ty = round(ts * 1.14), 112 + round(ts * 0.8)
        for i, line in enumerate(lines):
            body.append(nest.text(X, ty + i * lh, line, ts, P["ink"], 450, tracking=round(-ts * 0.032, 2)))
        top = ty + (len(lines) - 1) * lh + 46
    else:
        top = 118
    # the chart card
    cy = top + 8
    ch = 618 - cy
    body.append(nest.rrect(X - 16, cy - 24, CW + 32, ch + 30, 18, f'fill="{P["card"]}"'))
    body.append(nest.rrect(X - 15.5, cy - 23.5, CW + 31, ch + 29, 18, f'fill="none" stroke="{P["line"]}" stroke-width="1"'))
    body.append(nest.text(X + 4, cy + 4, nest.fit(metric.upper(), CW - 200, 14, weight=500, mono=True, tracking=1.3), 14,
                          P["dim"], 500, mono=True, tracking=1.3))
    body.append(nest.text(R - 4, cy + 4, "Higher is better", 14, P["faint"], 400, anchor="end"))
    n = max(len(bars), 1)
    has_cost = any(b["cost"] for b in bars) or pending is not None
    base = cy + ch - (104 if n > 4 else 112) - (24 if has_cost else 0)
    plot_top = cy + 44
    ph = base - plot_top - 40
    body.append(f'<rect x="{X}" y="{base:.1f}" width="{CW}" height="1.5" fill="{P["line"]}"/>')
    gap = 22 if n > 4 else 34
    col = (CW - gap * (n - 1)) / n
    bw = min(col * 0.62, 128)
    if pending:
        body.append(nest.text(X + CW / 2, plot_top + ph / 2 + 8, pending, 18, P["dim"], 400, anchor="middle"))
    for i, b in enumerate(bars):
        x = X + i * (col + gap)
        cx = x + col / 2
        ink = P["ink"] if b["green"] else P["muted"]
        if b["f"] is not None:
            h = max(3 if b["f"] > 0 else 2, ph * b["f"])
            r = min(8, h / 2)
            body.append(nest.rrect(cx - bw / 2, base - h, bw, h, r, f'fill="url(#{gid}-{"s" if b["green"] else "b"})"'))
            body.append(nest.rrect(cx - bw / 2 + 0.5, base - h + 0.5, bw - 1, h - 1, min(7.5, h / 2),
                                   f'fill="none" stroke="{P["gEdge"] if b["green"] else P["kEdge"]}" stroke-width="1"'))
            vs = 26 if n > 4 else 30 if n > 3 else 34
            body.append(nest.text(cx, base - h - 12, b["txt"], vs, ink, 500, mono=True, anchor="middle", tracking=-0.8))
            body.append(nest.text(cx, base - h - 12 - vs - 2, b["arm"], 13, P["dim"], 500, mono=True, anchor="middle",
                                  tracking=1))
        s = 36 if n > 4 else 42
        body.append(_chip(P, theme, b["maker"], (b["name"] or "?")[0].upper(), cx - s / 2, base + 14, s))
        fs = 18 if n > 4 else 21
        wide = col + gap * 0.6
        y = base + s + 42
        nf = fs  # the model's name: smaller before it is cut
        while nf > 15 and nest.width(b["name"], nf, weight=500, tracking=-0.3) > wide:
            nf -= 1
        body.append(nest.text(cx, y, nest.fit(b["name"], wide, nf, weight=500, tracking=-0.3), nf, ink, 500,
                              anchor="middle", tracking=-0.3))
        y += fs + 3
        body.append(nest.text(cx, y, nest.fit(b["variant"], wide, fs - 2, tracking=-0.2), fs - 2, P["dim"], 400,
                              anchor="middle", tracking=-0.2))
        sub = b["cost"] if b["cost"] else (b["arm"] if b["f"] is None else None)
        if sub:
            body.append(nest.text(cx, y + 26, nest.fit(sub, wide, 14, weight=500, mono=True), 14, P["dim"], 500,
                                  mono=True, anchor="middle"))
    # the foot
    note = foot
    if has_cost and not pending:
        note = f"Pre-registered. Under each model: cost per correct query (SLMs: full to {lowest_use} use)."
    body.append(nest.text(X, 652, nest.fit(note, CW - 330, 18, tracking=-0.2), 18, P["dim"], 400, tracking=-0.2))
    body.append(nest.text(R, 652, url, 18, P["ink"], 500, mono=True, anchor="end", tracking=-0.2))
    label = (title + ". " if title else "Results. ") + ", ".join(f"{b['arm']} {b['name']}, {b['variant']}: {b['txt']}" + (f" ({b['cost']} per correct query)" if b["cost"] else "")
                                     for b in bars)
    return nest.svg(W, H, body, label)


# ---------------------------------------------------------------- CLI

def load_config(path: Optional[str]) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(Path(path or ROOT / "config.yaml").read_text())


def write(out: Path, bars: List[Dict[str, Any]], **kw) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for theme in nest.THEMES:
        p = out / f"results-{theme}.svg"
        p.write_text(draw(bars, theme, **kw))
        written.append(p)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report", nargs="?", help="a report directory (reports/<sha256>) or its report.json")
    ap.add_argument("--pending", action="store_true", help="no result yet: the arms, with no numbers")
    ap.add_argument("--config", help="the configuration that names the models (default: config.yaml)")
    ap.add_argument("--out", help="where to write results-{light,dark}.svg (default: the report's directory)")
    ap.add_argument("--title", default="", help="a headline over the chart (the finding, once there is one); none by default")
    ap.add_argument("--arms", nargs="+", choices=ARMS, help="the arms to draw, in order (default: every arm reported)")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    if args.pending:
        bars = pending_bars(config, args.arms or ["B0", "B1", "B3", "B4", "B5"])
        out = Path(args.out or ".")
        kw = {"pending": "Pending: no paid run yet. The protocol is registered at Stage 0."}
    else:
        if not args.report:
            ap.error("a report directory, or --pending")
        path = Path(args.report)
        path = path / "report.json" if path.is_dir() else path
        report = json.loads(path.read_text())
        bars = bars_of(report, config, args.arms)
        if not bars:
            ap.error(f"{path} reports no arm with an EX")
        n = {(report["arms"][b["arm"]] or {}).get("n") for b in bars}
        metric = f"Execution accuracy · {report.get('split', '')} split" + (f" · {n.pop()} questions" if len(n) == 1 else "")
        out = Path(args.out) if args.out else path.parent
        kw = {"metric": metric, "lowest_use": (report.get("utilizations") or ["20%"])[0]}
    for p in write(out, bars, title=args.title, **kw):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
