""""How it works" for the README, as the cards of nest.technology's own "How it works" (light and dark).

    python scripts/brand/diagram.py --out .github/assets      # writes how-light.svg, how-dark.svg

Six cards, the paper's S1–S6 as the harness runs them: a title in two tones, the step's number, and a small picture of
what the step does (the calls it logs, what curation keeps, the clusters, the candidates, the adapters, the router).
The pictures show the mechanism, never a result: no number in them is measured. The steps, the call sites and the
configuration values are written here by hand; change them with the harness. S5 and S6 are the steps whose
contribution the core measures (SPEC §4–5: B3 × B4 and B4 × B5); S1–S4 are run and described.
"""
import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nest  # noqa: E402

W, PAD, GAP = 960, 32, 20
CW = (W - 2 * PAD - GAP) / 2
TOP_H, ART_H = 128, 262
CH = TOP_H + ART_H

# the site's .hw-* tokens (index.html), light and dark
HW = {
    "light": {"tile": ("#efefeb", "#ebebe6"), "rim": "rgba(255,255,255,.75)", "edge": "rgba(38,35,35,.075)",
              "grey": "rgba(38,35,35,.56)", "paper": "#fbfbf8", "soft": "rgba(38,35,35,.7)", "chip": "#ecece6",
              "led": "#5d9f6c", "ledRing": "rgba(80,105,89,.16)", "raised": "#fbfbf8", "control": "rgba(32,32,32,.1)",
              "ring": "rgba(0,0,0,.08)", "shadow": "rgb(38,35,35)", "shadowA": ".16", "del": "#b4594f",
              "pill": "#e7e7e1"},
    "dark": {"tile": ("#201f1d", "#1c1b19"), "rim": "rgba(255,255,255,.035)", "edge": "rgba(255,255,255,.07)",
             "grey": "rgba(242,240,234,.54)", "paper": "#2a2926", "soft": "rgba(242,240,234,.62)", "chip": "#35342f",
             "led": "#8fd09b", "ledRing": "rgba(166,211,173,.16)", "raised": "#252421", "control": "rgba(255,255,255,.1)",
             "ring": "rgba(255,255,255,.07)", "shadow": "#000", "shadowA": ".55", "del": "#d98a80",
             "pill": "#2c2b28"},
}

STEPS = (
    ("S1", "Log every LLM call", " the agent makes. Each one becomes a line of calls.jsonl.", False, "log"),
    ("S2", "Keep what worked:", " the teacher's answers, masked, filtered and de-duplicated.", False, "curate"),
    ("S3", "Find the tasks", " by embedding each call with its answer and clustering the calls.", False, "cluster"),
    ("S4", "Pick the small model:", " a desk triage of the candidates, then zero-shot on calibration.", False, "select"),
    ("S5", "Fine-tune with LoRA,", " one adapter per cluster on the chosen SLM, on Modal's GPUs.", True, "train"),
    ("S6", "Route each task", " to the cheapest engine that is not worse than the LLM.", True, "route"),
)
CALL_SITES = ("extract_keywords", "select_tables", "select_columns", "generate_candidate", "revise")
CHECK = "m5 12.5 4.5 4.5L19 7"
CROSS = "M7 7l10 10M17 7 7 17"


def chip_w(label: str, icon: bool = False, led: bool = False, mono: bool = False, size: float = 12.5) -> float:
    """The width of a chip as Draw.chip draws it."""
    lead = 20 if icon else 13 if led else 0
    return 20 + lead + nest.width(label, size, 500 if mono else 450, mono=mono)


class Draw:
    def __init__(self, theme: str):
        self.theme = theme
        self.P = dict(nest.PAL[theme])
        self.H = HW[theme]
        self.out: List[str] = []

    # ---------------------------------------------------------------- primitives
    def add(self, *parts: str) -> None:
        self.out.extend(parts)

    def text(self, x, y, s, size, fill, weight=400, mono=False, anchor="start", tracking=0.0, extra=""):
        self.add(nest.text(x, y, s, size, fill, weight, mono, anchor, tracking, extra))

    def icon(self, path: str, x: float, y: float, s: float, stroke: str, sw: float = 2.0) -> None:
        k = s / 24
        self.add(f'<path transform="translate({x:.1f} {y:.1f}) scale({k:.4f})" d="{path}" fill="none" stroke="{stroke}" '
                 f'stroke-width="{sw / k:.2f}" stroke-linecap="round" stroke-linejoin="round"/>')

    def logo(self, maker: Optional[str], x: float, y: float, s: float, letter: str = "?") -> None:
        uri = nest.logo(maker, self.theme)
        if uri:
            self.add(f'<image x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" height="{s:.1f}" preserveAspectRatio="xMidYMid meet" href="{uri}"/>')
        else:
            self.add(nest.rrect(x, y, s, s, s * 0.28, f'fill="{self.H["chip"]}"'))
            self.text(x + s / 2, y + s * 0.74, letter, s * 0.62, self.P["dim"], 600, anchor="middle")

    def chip(self, x: float, y: float, label: str, maker: Optional[str] = None, led: bool = False, mono: bool = False,
             fill: Optional[str] = None, color: Optional[str] = None, h: float = 24, size: float = 12.5,
             letter: str = "?") -> float:
        """A .hw-chip; returns its width."""
        pad, gap = 10, 6
        lead = 0.0
        if maker is not None or letter != "?":
            lead = 14 + gap
        elif led:
            lead = 7 + gap
        w = pad * 2 + lead + nest.width(label, size, 500 if mono else 450, mono=mono)
        self.add(nest.rrect(x, y, w, h, h / 2, f'fill="{fill or self.H["chip"]}"'))
        cx = x + pad
        if maker is not None or letter != "?":
            self.logo(maker, cx, y + (h - 14) / 2, 14, letter)
        elif led:
            self.add(f'<circle cx="{cx + 3.5:.1f}" cy="{y + h / 2:.1f}" r="6" fill="{self.H["ledRing"]}"/>'
                     f'<circle cx="{cx + 3.5:.1f}" cy="{y + h / 2:.1f}" r="3.2" fill="{self.H["led"]}"/>')
        self.text(cx + lead, y + h / 2 + size * 0.36, label, size, color or self.P["ink"], 500 if mono else 450, mono=mono)
        return w

    def note(self, cx: float, y: float, label: str) -> None:
        """The pill under a picture: a green LED and one line (the site's 'Your current model keeps answering')."""
        size = 12.5
        w = 20 + 7 + 7 + nest.width(label, size, 450)
        x = cx - w / 2
        self.add(nest.rrect(x, y, w, 26, 13, f'fill="{self.H["raised"]}" stroke="{self.H["ring"]}" stroke-width="1"'))
        self.add(f'<circle cx="{x + 13.5:.1f}" cy="{y + 13:.1f}" r="6" fill="{self.H["ledRing"]}"/>'
                 f'<circle cx="{x + 13.5:.1f}" cy="{y + 13:.1f}" r="3.2" fill="{self.H["led"]}"/>')
        self.text(x + 27, y + 17.5, label, size, self.P["ink"], 450, tracking=-0.1)

    def window(self, x: float, y: float, w: float, h: float, tab: str, right: str = "") -> None:
        """A .hw-win: the paper panel, a tab with its name, a mono note at the right."""
        self.add(nest.rrect(x, y, w, h, 12, f'fill="{self.H["paper"]}" stroke="{self.H["ring"]}" stroke-width="1" filter="url(#lift-{self.theme})"'))
        tw = nest.width(tab, 11.5, 500, mono=True) + 16
        self.add(nest.rrect(x + 8, y + 8, tw, 22, 6, f'fill="{self.H["chip"]}"'))
        self.text(x + 16, y + 23, tab, 11.5, self.H["soft"], 500, mono=True)
        if right:
            self.text(x + w - 12, y + 23, right, 11.5, self.P["faint"], 500, mono=True, anchor="end")
        self.add(f'<rect x="{x:.1f}" y="{y + 38:.1f}" width="{w:.1f}" height="1" fill="{self.H["ring"]}"/>')

    def piece(self, x: float, y: float, w: float, fill: str) -> None:
        """The Nest piece (the specialist's glyph), `w` wide."""
        k = w / 72
        self.add(f'<path transform="translate({x:.1f} {y:.1f}) scale({k:.4f})" d="{nest.MARK}" fill="{fill}"/>')

    # ---------------------------------------------------------------- the six pictures (art area: x, y, w, h)
    def art_log(self, x, y, w, h):
        cx = x + w / 2
        # the agent calls its LLM
        a = chip_w("CHESS agent", icon=True)
        b_label = "Claude Sonnet 5.5"
        b = chip_w(b_label, icon=True)
        total = a + 30 + b
        x0 = cx - total / 2
        self.chip(x0, y + 4, "CHESS agent", letter="C")
        self.icon("M5 12h14m-5-5 5 5-5 5", x0 + a + 7, y + 8, 16, self.P["dim"], 1.6)
        self.chip(x0 + a + 30, y + 4, b_label, maker="anthropic")
        ww, wh = 352, 166
        wx, wy = cx - ww / 2, y + 40
        self.window(wx, wy, ww, wh, "calls.jsonl", "bench run")
        for i, site in enumerate(CALL_SITES):
            ry = wy + 50 + i * 22
            self.logo("anthropic", wx + 14, ry - 10, 12)
            self.text(wx + 34, ry, f'{{"call_site": "{site}", …', 11.5, self.H["soft"], 500, mono=True)
            self.icon(CHECK, wx + ww - 26, ry - 11, 13, self.H["led"], 2)
        self.note(cx, y + h - 34, "The agent runs as in production")

    def art_curate(self, x, y, w, h):
        cx = x + w / 2
        ww, wh = 330, 176
        wx, wy = cx - ww / 2, y + 8
        self.window(wx, wy, ww, wh, "teacher's train logs", "bench curate")
        rows = (("generate_candidate", "SQL runs, returns rows", True), ("generate_candidate", "SQL error", False),
                ("select_columns", "parsed", True), ("select_tables", "duplicate", False),
                ("revise", "SQL runs, returns rows", True))
        for i, (site, why, keep) in enumerate(rows):
            ry = wy + 56 + i * 24
            col = self.P["ink"] if keep else self.P["faint"]
            self.text(wx + 14, ry, site, 11.5, col, 500, mono=True)
            if not keep:
                tw = nest.width(site, 11.5, 500, mono=True)
                self.add(f'<rect x="{wx + 14:.1f}" y="{ry - 4:.1f}" width="{tw:.1f}" height="1.2" fill="{self.P["faint"]}"/>')
            self.text(wx + ww - 34, ry, why, 11.5, self.H["soft"] if keep else self.P["faint"], 400, anchor="end")
            if keep:
                self.icon(CHECK, wx + ww - 26, ry - 11, 13, self.H["led"], 2)
            else:
                self.icon(CROSS, wx + ww - 26, ry - 11, 13, self.H["del"], 2)
        self.note(cx, y + h - 34, "A production signal, never the gold answer")

    def art_cluster(self, x, y, w, h):
        cx = x + w / 2
        ww, wh = 330, 176
        wx, wy = cx - ww / 2, y + 8
        self.window(wx, wy, ww, wh, "embed · j5", "bench embed")
        rng = random.Random(7)
        groups = (("keywords", 0.17, 0.4), ("tables", 0.5, 0.3), ("columns", 0.83, 0.4), ("SQL", 0.32, 0.82),
                  ("repair", 0.68, 0.82))
        px, py, pw, ph = wx + 14, wy + 46, ww - 28, wh - 56
        for name, gx, gy in groups:
            hx, hy = px + gx * pw, py + gy * ph
            rw, rh = 74, 36
            self.add(nest.rrect(hx - rw / 2, hy - rh / 2, rw, rh, rh / 2,
                                f'fill="{self.H["chip"]}" stroke="{self.P["line"]}" stroke-dasharray="3 3"'))
            for _ in range(9):
                dx, dy = rng.gauss(0, 13), rng.gauss(0, 5.5)
                self.add(f'<circle cx="{hx + dx:.1f}" cy="{hy + dy:.1f}" r="2.6" fill="{self.P["muted"]}"/>')
            self.text(hx, hy - rh / 2 - 5, name, 10.5, self.H["soft"], 500, mono=True, anchor="middle")
        self.note(cx, y + h - 34, "The router sees only the prompt")

    def art_select(self, x, y, w, h):
        cx = x + w / 2
        ww, wh = 330, 176
        wx, wy = cx - ww / 2, y + 8
        self.window(wx, wy, ww, wh, "SLM candidates", "bench judge j6")
        opts = (("Qwen3-8B", "qwen", "Q", ""), ("Granite-4.2-8B", None, "G", ""), ("Nemotron-3-Nano-4B", "nvidia", "N", "reserve"))
        for i, (name, mk, letter, tag) in enumerate(opts):
            ry = wy + 50 + i * 36
            self.add(nest.rrect(wx + 12, ry, ww - 24, 28, 8, f'fill="{self.H["paper"]}" stroke="{self.H["control"]}" stroke-width="1"'))
            self.logo(mk, wx + 22, ry + 7, 14, letter)
            self.text(wx + 44, ry + 18.5, name, 12.5, self.P["ink"], 450)
            if tag:
                self.text(wx + ww - 22, ry + 18.5, tag, 11, self.P["faint"], 500, mono=True, anchor="end")
            else:
                self.add(f'<circle cx="{wx + ww - 28:.1f}" cy="{ry + 14:.1f}" r="6" fill="none" stroke="{self.P["faint"]}" stroke-width="1.3"/>')
        self.note(cx, y + h - 34, "Chosen on calibration, before any training")

    def art_train(self, x, y, w, h):
        cx = x + w / 2
        ww, wh = 330, 176
        wx, wy = cx - ww / 2, y + 8
        self.window(wx, wy, ww, wh, "config.yaml", "bench train")
        lines = (("train:", 0), ("gpu: L40S", 1), ("lora: {r: 16, alpha: 32}", 1), ("precision: bf16", 1))
        for i, (s, ind) in enumerate(lines):
            ly = wy + 58 + i * 19
            key, _, val = s.partition(":")
            tx = wx + 16 + ind * 16
            self.text(tx, ly, key + ":", 11.5, self.H["soft"], 500, mono=True)
            if val:
                self.text(tx + nest.width(key + ": ", 11.5, 500, mono=True), ly, val.strip(), 11.5, self.P["ink"], 500, mono=True)
        # the base and one adapter per cluster: the specialist's piece
        by = wy + wh - 30
        bw = nest.width("base SLM", 11, 500, mono=True) + 16
        self.add(nest.rrect(wx + 14, by - 4, bw, 22, 6, f'fill="{self.H["chip"]}"'))
        self.text(wx + 22, by + 11, "base SLM", 11, self.H["soft"], 500, mono=True)
        self.text(wx + 22 + bw, by + 11, "+", 13, self.P["dim"], 500, mono=True)
        for i in range(5):
            self.piece(wx + 36 + bw + i * 34, by - 1, 27, self.P["green"])
        self.note(cx, y + h - 34, "TRL + PEFT, served with vLLM")

    def art_route(self, x, y, w, h):
        cx = x + w / 2
        ww, wh = 340, 176
        wx, wy = cx - ww / 2, y + 8
        self.window(wx, wy, ww, wh, "router · j7", "bench judge j7")
        my = wy + 38 + (wh - 38) / 2
        cw = self.chip(wx + 14, my - 12, "a call", fill=self.H["chip"])
        nx = wx + 14 + cw + 32
        self.add(f'<line x1="{wx + 14 + cw:.1f}" y1="{my:.1f}" x2="{nx - 13:.1f}" y2="{my:.1f}" stroke="{self.P["faint"]}" stroke-width="1.4"/>')
        self.add(f'<circle cx="{nx:.1f}" cy="{my:.1f}" r="13" fill="{self.H["paper"]}" stroke="{self.H["control"]}" stroke-width="1.2"/>')
        self.piece(nx - 8, my - 5.3, 16, self.P["ink"])
        ex = nx + 46
        up, dn = my - 34, my + 34
        for ty, green in ((up, True), (dn, False)):
            self.add(f'<path d="M{nx + 13:.1f} {my:.1f} C{nx + 30:.1f} {my:.1f} {nx + 26:.1f} {ty:.1f} {ex:.1f} {ty:.1f}" fill="none" '
                     f'stroke="{self.H["led"] if green else self.P["faint"]}" stroke-width="1.4"/>')
        self.chip(ex, up - 12, "SLM + LoRA", maker=None, led=True, color=self.P["ink"])
        self.chip(ex, dn - 12, "Claude Sonnet 5.5", maker="anthropic")
        self.text(ex + 4, up + 26, "not worse than B0", 10.5, self.H["soft"], 500, mono=True)
        self.text(ex + 4, dn - 18, "otherwise", 10.5, self.P["faint"], 500, mono=True)
        self.note(cx, y + h - 34, "Decided per cluster, before the test")

    # ---------------------------------------------------------------- the card
    def card(self, i: int, x: float, y: float) -> None:
        sid, lead, rest, measured, kind = STEPS[i]
        t, H = self.theme, self.H
        self.add(f'<g clip-path="url(#card-{t}-{i})">',
                 nest.rrect(x, y, CW, CH, 22, f'fill="url(#tile-{t})"'),
                 f'<rect x="{x:.1f}" y="{y + 0.5:.1f}" width="{CW:.1f}" height="1" fill="{H["rim"]}"/>', "</g>",
                 nest.rrect(x + 0.5, y + 0.5, CW - 1, CH - 1, 21.5, f'fill="none" stroke="{H["edge"]}" stroke-width="1"'))
        # the number, as the site's .hw-no; S5 and S6 say they are measured
        nx = x + CW - 22 - 36
        self.add(nest.rrect(nx, y + 22, 36, 36, 10, f'fill="{H["raised"]}" stroke="{H["control"]}" stroke-width="1"'))
        self.text(nx + 18, y + 44, sid, 11.5, self.P["ink"], 500, mono=True, anchor="middle", tracking=0.2)
        if measured:
            mw = nest.width("measured", 11, 500, mono=True) + 16
            self.add(nest.rrect(nx - 8 - mw, y + 29, mw, 22, 11, f'fill="{self.P["greenBg"]}"'))
            self.text(nx - 8 - mw / 2, y + 44, "measured", 11, self.P["green"], 500, mono=True, anchor="middle")
        # the title in two tones: the lead in ink, the rest in grey (.hw-k inside .hw-tt)
        size, lh = 22, 27
        lines = nest.wrap(lead + rest, CW - 56 - 70, size, 3, weight=450, tracking=-0.66)
        done = 0
        for j, line in enumerate(lines):
            k = max(0, min(len(line), len(lead) - done))
            ink, grey = line[:k], line[k:]
            from xml.sax.saxutils import escape
            spans = (f'<tspan fill="{self.P["ink"]}">{escape(ink)}</tspan>' if ink else "") + \
                    (f'<tspan fill="{H["grey"]}">{escape(grey)}</tspan>' if grey else "")
            self.add(f'<text x="{x + 28:.1f}" y="{y + 50 + j * lh:.1f}" font-family="{nest.SANS}" font-size="{size}" '
                     f'font-weight="450" letter-spacing="-0.66">{spans}</text>')
            done += len(line) + 1
        getattr(self, f"art_{kind}")(x + 20, y + TOP_H, CW - 40, ART_H - 18)


def draw(theme: str) -> str:
    d = Draw(theme)
    P, H = d.P, d.H
    head = 150
    h = PAD + head + 3 * CH + 2 * GAP + PAD
    defs = [f'<linearGradient id="tile-{theme}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{H["tile"][0]}"/>'
            f'<stop offset="1" stop-color="{H["tile"][1]}"/></linearGradient>',
            f'<filter id="lift-{theme}" x="-20%" y="-20%" width="140%" height="170%">'
            f'<feDropShadow dx="0" dy="12" stdDeviation="8" flood-color="{H["shadow"]}" flood-opacity="{H["shadowA"]}"/>'
            f'<feDropShadow dx="0" dy="4" stdDeviation="3" flood-color="{H["shadow"]}" flood-opacity=".05"/></filter>']
    for i in range(6):
        x = PAD + (i % 2) * (CW + GAP)
        y = PAD + head + (i // 2) * (CH + GAP)
        defs.append(f'<clipPath id="card-{theme}-{i}">{nest.rrect(x, y, CW, CH, 22, "")}</clipPath>')
    d.add("<defs>" + "".join(defs) + "</defs>", nest.rrect(0, 0, W, h, 22, f'fill="{P["bg"]}"'))
    # the section's head, as the site's: the pill, the title in two tones, one line under it
    pw = nest.width("How it works", 13, 400, tracking=0.26) + 30
    d.add(nest.rrect(PAD, PAD, pw, 34, 17, f'fill="{P["tile"]}" stroke="{P["line"]}" stroke-width="1"'))
    d.text(PAD + 15, PAD + 22, "How it works", 13, P["muted"], 400, tracking=0.26)
    d.text(PAD, PAD + 84, "The paper proposes six steps.", 34, P["ink"], 450, tracking=-1.36)
    d.text(PAD, PAD + 122, "This harness runs every one of them.", 34, H["grey"], 450, tracking=-1.36)
    for i in range(6):
        d.card(i, PAD + (i % 2) * (CW + GAP), PAD + head + (i // 2) * (CH + GAP))
    label = ("How it works: the paper proposes six steps and this harness runs every one. " +
             " ".join(f"{s}: {lead}{rest}" for s, lead, rest, _, _ in STEPS) +
             " S5 and S6 are the steps whose contribution is measured.")
    return nest.svg(W, h, d.out, label)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=".")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for theme in nest.THEMES:
        p = out / f"how-{theme}.svg"
        p.write_text(draw(theme))
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
