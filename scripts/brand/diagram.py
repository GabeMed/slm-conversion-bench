""""How it works" for the README, as the row of cards of nest.technology's own "How it works" (light and dark).

    python scripts/brand/diagram.py --out .github/assets      # writes how-light.svg, how-dark.svg

Six cards, the paper's S1–S6 as the harness runs them, in a row that drifts slowly to the left and loops, its ends
fading, as on the site (it stands still where the reader asks for reduced motion). Each card is the site's: a title in
two tones, the step's number, and a small window of what the step does. The windows show the mechanism, never a
result: no number in them is measured. The steps, call sites and configuration values are written here by hand; change
them with the harness. S5 and S6 are the steps whose contribution the core measures (SPEC §4–5: B3 × B4, B4 × B5).
"""
import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nest  # noqa: E402

W = 960  # GitHub draws it a little under 1×: about two and a half cards in view
CARD_W, CARD_H, GAP = 420, 458, 20
HEAD = 8  # no head of its own: the README's "## How it works" is the title
H = HEAD + CARD_H + 6
DRIFT_S = 170  # seconds for the row to move by one set of six cards: slow, to be read

# the site's .hw-* tokens (index.html), light and dark
HW = {
    "light": {"tile": ("#efefeb", "#ebebe6"), "rim": "rgba(255,255,255,.75)", "edge": "rgba(38,35,35,.075)",
              "grey": "rgba(38,35,35,.56)", "paper": "#fbfbf8", "soft": "rgba(38,35,35,.7)", "chip": "#ecece6",
              "led": "#5d9f6c", "ledRing": "rgba(80,105,89,.16)", "raised": "#fbfbf8", "control": "rgba(32,32,32,.1)",
              "ring": "rgba(0,0,0,.08)", "shadow": "rgb(38,35,35)", "shadowA": ".16", "del": "#b4594f",
              "hl": "rgba(80,105,89,.12)", "hlEdge": "#5d9f6c"},
    "dark": {"tile": ("#201f1d", "#1c1b19"), "rim": "rgba(255,255,255,.035)", "edge": "rgba(255,255,255,.07)",
             "grey": "rgba(242,240,234,.54)", "paper": "#2a2926", "soft": "rgba(242,240,234,.62)", "chip": "#35342f",
             "led": "#8fd09b", "ledRing": "rgba(166,211,173,.16)", "raised": "#252421", "control": "rgba(255,255,255,.1)",
             "ring": "rgba(255,255,255,.07)", "shadow": "#000", "shadowA": ".55", "del": "#d98a80",
             "hl": "rgba(166,211,173,.12)", "hlEdge": "#8fd09b"},
}

STEPS = (
    ("S1", "Log every LLM call", " the agent makes, as it runs in production.", False, "log"),
    ("S2", "Keep what worked:", " the teacher's answers, masked, filtered and de-duplicated.", False, "curate"),
    ("S3", "Find the tasks", " the calls share, by clustering them.", False, "cluster"),
    ("S4", "Pick the small model", " by a desk triage, then zero-shot on calibration.", False, "select"),
    ("S5", "Fine-tune with LoRA", " on Tinker: one adapter per task on the chosen SLM.", True, "train"),
    ("S6", "Route each task", " to the cheapest engine not worse than the LLM.", True, "route"),
)
# CHESS's call sites, in words and as calls.jsonl names them
CALLS = (("Extract the keywords", "extract_keywords"), ("Pick the tables", "select_tables"),
         ("Pick the columns", "select_columns"), ("Write the SQL", "generate_candidate"), ("Fix the SQL", "revise"))
CHECK = "m5 12.5 4.5 4.5L19 7"
CROSS = "M7 7l10 10M17 7 7 17"
SHIELD = "M12 3.5 19 6v5.5c0 4.3-3 7.6-7 9-4-1.4-7-4.7-7-9V6Z M9 12l2.2 2.2L15.5 10"
CIRCLE = "M12 4.5a7.5 7.5 0 1 0 0 15 7.5 7.5 0 0 0 0-15Z"


class Draw:
    def __init__(self, theme: str):
        self.theme = theme
        self.P = dict(nest.PAL[theme])
        self.H = HW[theme]
        self.out: List[str] = []

    def add(self, *parts: str) -> None:
        self.out.extend(parts)

    def text(self, x, y, s, size, fill, weight=400, mono=False, anchor="start", tracking=0.0):
        self.add(nest.text(x, y, s, size, fill, weight, mono, anchor, tracking))

    def icon(self, path: str, x: float, y: float, s: float, stroke: str, sw: float = 1.7) -> None:
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

    def led(self, cx: float, cy: float, on: bool = True) -> None:
        if on:
            self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{self.H["ledRing"]}"/>'
                     f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{self.H["led"]}"/>')
        else:
            self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.4" fill="none" stroke="{self.P["faint"]}" stroke-width="1.2"/>')

    def mono_chip(self, right: float, cy: float, label: str) -> None:
        """A call site, as the site's tool names (`track_order`): a mono chip, right-aligned."""
        w = nest.width(label, 11, 500, mono=True) + 14
        self.add(nest.rrect(right - w, cy - 10.5, w, 21, 6, f'fill="{self.H["chip"]}"'))
        self.text(right - w + 7, cy + 4, label, 11, self.H["soft"], 500, mono=True)

    def pill(self, x: float, y: float, label: str, kind: str = "led") -> float:
        """The pill under a window (the site's "Your current model keeps answering"); returns its width."""
        size = 12
        lead = 20 if kind else 0
        w = 22 + lead + nest.width(label, size, 450, tracking=-0.1)
        self.add(nest.rrect(x, y, w, 28, 14, f'fill="{self.H["raised"]}" stroke="{self.H["ring"]}" stroke-width="1"'))
        if kind == "led":
            self.led(x + 16, y + 14)
        elif kind == "shield":
            self.icon(SHIELD, x + 9, y + 7, 14, self.H["soft"], 1.6)
        self.text(x + 11 + lead, y + 18.3, label, size, self.P["ink"], 450, tracking=-0.1)
        return w

    def window(self, x: float, y: float, w: float, h: float, name: str, section: str, right: str = "",
               maker: Optional[str] = None) -> float:
        """A .hw-win: the paper panel with its header ("support-agent · Provider"); returns the body's top."""
        self.add(nest.rrect(x, y, w, h, 12, f'fill="{self.H["paper"]}" stroke="{self.H["ring"]}" stroke-width="1" '
                                            f'filter="url(#lift-{self.theme})"'))
        self.add(f'<text x="{x + 14:.1f}" y="{y + 21:.1f}" font-family="{nest.SANS}" font-size="12" letter-spacing="-0.1">'
                 f'<tspan fill="{self.P["ink"]}" font-weight="500">{escape(name)}</tspan>'
                 f'<tspan fill="{self.H["soft"]}" font-weight="400"> · {escape(section)}</tspan></text>')
        if right:
            rx = x + w - 14
            self.text(rx, y + 21, right, 11.5, self.H["soft"], 450, anchor="end")
            if maker:
                self.logo(maker, rx - nest.width(right, 11.5, 450) - 18, y + 10, 12)
        self.add(f'<rect x="{x:.1f}" y="{y + 33:.1f}" width="{w:.1f}" height="1" fill="{self.H["ring"]}"/>')
        return y + 33

    # ---------------------------------------------------------------- the six windows (x, y: the window's corner)
    def win_log(self, x, y):
        w, h = 318, 202
        top = self.window(x, y, w, h, "CHESS", "Calls", "Claude Sonnet 5.5", "anthropic")
        for i, (say, site) in enumerate(CALLS):
            cy = top + 23 + i * 31
            self.led(x + 20, cy)
            self.text(x + 34, cy + 4.3, say, 12.5, self.P["ink"], 450, tracking=-0.1)
            self.mono_chip(x + w - 12, cy, site)
        self.pill(x, y + h + 14, "Each call is a line of calls.jsonl")

    def win_curate(self, x, y):
        w, h = 318, 202
        top = self.window(x, y, w, h, "Teacher's logs", "Curate")
        rows = (("Write the SQL", "runs, returns rows", True), ("Write the SQL", "SQL error", False),
                ("Pick the columns", "parsed", True), ("Pick the tables", "duplicate", False),
                ("Fix the SQL", "runs, returns rows", True))
        for i, (say, why, keep) in enumerate(rows):
            cy = top + 23 + i * 31
            box = self.H["hl"] if keep else self.H["chip"]
            self.add(nest.rrect(x + 12, cy - 10, 20, 20, 6, f'fill="{box}"'))
            if keep:
                self.icon(CHECK, x + 15, cy - 7, 14, self.H["led"], 2)
            else:
                self.icon(CROSS, x + 15, cy - 7, 14, self.H["del"], 2)
            col = self.P["ink"] if keep else self.P["faint"]
            self.text(x + 42, cy + 4.3, say, 12.5, col, 450, tracking=-0.1)
            if not keep:
                tw = nest.width(say, 12.5, 450, tracking=-0.1)
                self.add(f'<rect x="{x + 42:.1f}" y="{cy - 0.3:.1f}" width="{tw:.1f}" height="1.1" fill="{self.P["faint"]}"/>')
            self.text(x + w - 14, cy + 4.3, why, 11.5, self.H["soft"] if keep else self.P["faint"], 400, anchor="end")
        self.pill(x, y + h + 14, "A production signal, never the gold", "shield")

    def win_cluster(self, x, y):
        """Each call becomes a vector (its prompt with the teacher's answer); calls with near vectors are one task."""
        w, h = 318, 202
        top = self.window(x, y, w, h, "Calls", "Tasks", "prompt + answer → vector")
        rng = random.Random(4)
        base = {t: [rng.random() for _ in range(12)] for t in (1, 2, 3)}
        rows = (("Pick the tables", 1), ("Pick the columns", 1), ("Write the SQL", 2), ("Write the SQL", 2),
                ("Fix the SQL", 3))
        prev = None
        for i, (say, task) in enumerate(rows):
            cy = top + 23 + i * 31
            if prev is not None and task != prev:
                self.add(f'<rect x="{x + 12:.1f}" y="{cy - 16:.1f}" width="{w - 24:.1f}" height="1" fill="{self.P["line"]}"/>')
            prev = task
            self.text(x + 14, cy + 4.3, say, 12.5, self.P["ink"], 450, tracking=-0.1)
            for j, v in enumerate(base[task]):  # the call's vector: near its task's, never equal
                o = max(0.06, min(0.85, 0.08 + 0.75 * v + rng.uniform(-0.12, 0.12)))
                self.add(nest.rrect(x + 126 + j * 9, cy - 6.5, 7.5, 13, 2, f'fill="{self.P["ink"]}" fill-opacity="{o:.2f}"'))
            self.mono_chip(x + w - 12, cy, f"task {task}")
        self.pill(x, y + h + 14, "The router sees only the prompt")

    def win_select(self, x, y):
        w, h = 318, 192
        top = self.window(x, y, w, h, "SLM", "Candidates")
        opts = (("Qwen3-8B", "qwen", "Q", ""), ("Granite-4.2-8B", None, "G", ""),
                ("Nemotron-3-Nano-4B", "nvidia", "N", "reserve"))
        cw = (w - 24 - 8) / 2
        for i, (name, mk, letter, tag) in enumerate(opts):
            ox = x + 12 + (i % 2) * (cw + 8) if i < 2 else x + 12
            oy = top + 12 + (i // 2) * 36
            ow = cw if i < 2 else w - 24
            self.add(nest.rrect(ox, oy, ow, 28, 8, f'fill="{self.H["paper"]}" stroke="{self.H["control"]}" stroke-width="1"'))
            self.logo(mk, ox + 9, oy + 7, 14, letter)
            self.text(ox + 30, oy + 18.5, name, 12, self.P["ink"], 450, tracking=-0.1)
            if tag:
                self.text(ox + ow - 10, oy + 18.5, tag, 10.5, self.P["faint"], 500, mono=True, anchor="end")
        by = top + 96
        self.text(x + 14, by, "Zero-shot on calibration", 12, self.P["ink"], 500, tracking=-0.1)
        for j, mk in enumerate(("qwen", None)):
            ly = by + 12 + j * 22
            self.add(nest.rrect(x + 14, ly, w - 70, 12, 4, f'fill="{self.H["chip"]}"'))
            self.text(x + w - 14, ly + 10, "—", 11.5, self.P["faint"], 500, mono=True, anchor="end")
        self.pill(x, y + h + 14, "Chosen on Day 1, before any training", "")

    def win_train(self, x, y):
        w, h = 318, 192
        self.add(nest.rrect(x, y, w, h, 12, f'fill="{self.H["paper"]}" stroke="{self.H["ring"]}" stroke-width="1" '
                                            f'filter="url(#lift-{self.theme})"'))
        tw = nest.width("config.yaml", 11.5, 500, mono=True) + 18
        self.add(nest.rrect(x + 8, y + 8, tw, 22, 6, f'fill="{self.H["chip"]}"'))
        self.text(x + 17, y + 23, "config.yaml", 11.5, self.H["soft"], 500, mono=True)
        self.text(x + w - 14, y + 23, "bench train", 11.5, self.P["faint"], 500, mono=True, anchor="end")
        self.add(f'<rect x="{x:.1f}" y="{y + 38:.1f}" width="{w:.1f}" height="1" fill="{self.H["ring"]}"/>')
        lines = (("train:", 0, False), ("precision: bf16", 1, False), ("lora: {r: 16, alpha: 32}", 1, True),
                 ("seed: 20260930", 1, False))
        for i, (s, ind, hl) in enumerate(lines):
            ly = y + 64 + i * 25
            if hl:
                self.add(f'<rect x="{x:.1f}" y="{ly - 16:.1f}" width="{w:.1f}" height="25" fill="{self.H["hl"]}"/>'
                         f'<rect x="{x:.1f}" y="{ly - 16:.1f}" width="2.5" height="25" fill="{self.H["hlEdge"]}"/>')
            key, _, val = s.partition(":")
            tx = x + 16 + ind * 18
            self.text(tx, ly, key + ":", 11.5, self.H["soft"], 500, mono=True)
            if val:
                self.text(tx + nest.width(key + ": ", 11.5, 500, mono=True), ly, val.strip(), 11.5, self.P["ink"], 500, mono=True)
        self.pill(x, y + h + 14, "Trained on Tinker, served by vLLM")

    def win_route(self, x, y):
        w, h = 318, 192
        top = self.window(x, y, w, h, "Router", "Per task")
        for i, (say, _) in enumerate(CALLS[:4]):
            cy = top + 24 + i * 34
            self.text(x + 14, cy + 4.3, say, 12.5, self.P["ink"], 450, tracking=-0.1)
            sx = x + w - 12 - 104
            self.add(nest.rrect(sx, cy - 12, 104, 24, 7, f'fill="{self.H["chip"]}"'))
            self.text(sx + 26, cy + 4, "SLM", 11, self.H["soft"], 500, mono=True, anchor="middle")
            self.text(sx + 78, cy + 4, "LLM", 11, self.H["soft"], 500, mono=True, anchor="middle")
            self.add(f'<rect x="{sx + 52:.1f}" y="{cy - 7:.1f}" width="1" height="14" fill="{self.P["line"]}"/>')
        px = x
        for label in ("not worse than B0", "then the cheapest"):
            pw = nest.width(label, 12, 450) + 24
            self.add(nest.rrect(px, y + h + 14, pw, 28, 14, f'fill="{self.H["raised"]}" stroke="{self.H["ring"]}" stroke-width="1"'))
            self.text(px + 12, y + h + 32.3, label, 12, self.P["ink"], 450)
            px += pw + 8

    # ---------------------------------------------------------------- one card (the site's .hw-card), at (x, 0)
    def card(self, i: int, x: float, tag: str) -> None:
        sid, lead, rest, measured, kind = STEPS[i]
        t, H = self.theme, self.H
        cid = f"c-{t}-{tag}"
        self.add(f'<clipPath id="{cid}">{nest.rrect(x, 0, CARD_W, CARD_H, 22, "")}</clipPath>',
                 f'<g clip-path="url(#{cid})">', nest.rrect(x, 0, CARD_W, CARD_H, 22, f'fill="url(#tile-{t})"'),
                 f'<rect x="{x:.1f}" y="0.5" width="{CARD_W}" height="1" fill="{H["rim"]}"/>')
        getattr(self, f"win_{kind}")(x + 50, 132)
        self.add("</g>", nest.rrect(x + 0.5, 0.5, CARD_W - 1, CARD_H - 1, 21.5,
                                    f'fill="none" stroke="{H["edge"]}" stroke-width="1"'))
        nx = x + CARD_W - 22 - 36
        self.add(nest.rrect(nx, 22, 36, 36, 10, f'fill="{H["raised"]}" stroke="{H["control"]}" stroke-width="1"'))
        self.text(nx + 18, 44, sid, 11.5, self.P["ink"], 500, mono=True, anchor="middle", tracking=0.2)
        size, lh = 24, 29
        lines = nest.wrap(lead + rest, CARD_W - 56 - 50, size, 3, weight=450, tracking=-0.72)
        done = 0
        for j, line in enumerate(lines):
            k = max(0, min(len(line), len(lead) - done))
            ink, grey = line[:k], line[k:]
            spans = (f'<tspan fill="{self.P["ink"]}">{escape(ink)}</tspan>' if ink else "") + \
                    (f'<tspan fill="{H["grey"]}">{escape(grey)}</tspan>' if grey else "")
            self.add(f'<text x="{x + 28:.1f}" y="{50 + j * lh:.1f}" font-family="{nest.SANS}" font-size="{size}" '
                     f'font-weight="450" letter-spacing="-0.72">{spans}</text>')
            done += len(line) + 1
        if measured:  # S5 and S6: the steps whose contribution the core measures
            mw = nest.width("measured", 11, 500, mono=True) + 16
            self.add(nest.rrect(x + 28, CARD_H - 40, mw, 22, 11, f'fill="{self.P["greenBg"]}"'))
            self.text(x + 28 + mw / 2, CARD_H - 25, "measured", 11, self.P["green"], 500, mono=True, anchor="middle")


def draw(theme: str) -> str:
    d = Draw(theme)
    P, Hw = d.P, d.H
    step = CARD_W + GAP
    span = 6 * step
    defs = [f'<linearGradient id="tile-{theme}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{Hw["tile"][0]}"/>'
            f'<stop offset="1" stop-color="{Hw["tile"][1]}"/></linearGradient>',
            f'<filter id="lift-{theme}" x="-20%" y="-20%" width="140%" height="170%">'
            f'<feDropShadow dx="0" dy="12" stdDeviation="8" flood-color="{Hw["shadow"]}" flood-opacity="{Hw["shadowA"]}"/>'
            f'<feDropShadow dx="0" dy="4" stdDeviation="3" flood-color="{Hw["shadow"]}" flood-opacity=".05"/></filter>',
            # the row's ends fade, as on the site
            f'<linearGradient id="fade-{theme}" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/>'
            f'<stop offset=".06" stop-color="#fff" stop-opacity="1"/><stop offset=".94" stop-color="#fff" stop-opacity="1"/>'
            f'<stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>',
            f'<mask id="ends-{theme}" maskUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{CARD_H + 6}">'
            f'<rect width="{W}" height="{CARD_H + 6}" fill="url(#fade-{theme})"/></mask>']
    d.add("<defs>" + "".join(defs) + "</defs>")
    # the section's head, as the site's: the pill, the title in two tones, one line under it
    d.add(f'<g transform="translate(0 {HEAD})" mask="url(#ends-{theme})"><g class="row">')
    for k in range(12):  # two sets, so the loop has no seam
        d.card(k % 6, 24 + k * step, f"{k}")
    d.add("</g></g>")
    css = (f".row{{animation:drift {DRIFT_S}s linear infinite}}"
           f"@keyframes drift{{from{{transform:translateX(0)}}to{{transform:translateX(-{span}px)}}}}"
           "@media (prefers-reduced-motion:reduce){.row{animation:none}}")
    label = ("How it works: the paper proposes six steps; this harness runs every one. " +
             " ".join(f"{s}: {lead}{rest}" for s, lead, rest, _, _ in STEPS) +
             " S5 and S6 are the steps whose contribution is measured.")
    return nest.svg(W, H, d.out, label, css=css)


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
