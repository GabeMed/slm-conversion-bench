"""The README's hero, as a page of nest.technology: the pixel-art scene full width, the brand and a button at the top,
the headline over the dark wall (light: the scene by day; dark: by night).

    python scripts/brand/hero.py --out .github/assets            # writes hero-light.svg, hero-dark.svg
    python scripts/brand/hero.py --out .github/assets --status "Results: B4 matches B0 at a third of the cost"

The scene is `.github/assets/hero-art-{day,night}.png` (the art at 2×, its foot dissolving into pixels of its own
floor; made from the approved art by Nest's pixel pipeline). It is embedded, with Geist, so the image needs nothing
else. Only the words are written here.
"""
import argparse
import base64
import json
import random
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / ".github" / "assets"
W, H = 2048, 864
X = 112
WALL = 940  # the dark wall ends about here (2× art px): the words stay on it

EYEBROW = "arXiv 2506.02153"
TITLE = "Which LLM calls of a real agent can a fine-tuned SLM take over?"
SUB = ("NVIDIA's LLM-to-SLM conversion, run end to end on a real agent: CHESS answering BIRD text-to-SQL, "
       "with the protocol fixed before any result.")
STATUS = ""  # none until there is a finding
BUTTON = "nest.technology/benchmarks"
WHITE = "#ffffff"


def art_uri(phase: str) -> str:
    return "data:image/png;base64," + base64.b64encode((ASSETS / f"hero-art-{phase}.png").read_bytes()).decode("ascii")


def runs(cells):
    """Pixels as rectangles: the runs of each row."""
    rows = {}
    for x, y in sorted(cells, key=lambda c: (c[1], c[0])):
        rows.setdefault(y, []).append(x)
    out = []
    for y, xs in rows.items():
        start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            out.append((start, y, prev - start + 1))
            if x is not None:
                start = prev = x
    return out


def motion(phase: str) -> (List[str], str):
    """What moves (hero-anim.json, found in the art): the robot blinks; stars and the river's highlights go out and
    back; at night lit windows dim now and then; dust drifts in the light. Art px are 2 units here."""
    path = ASSETS / "hero-anim.json"
    if not path.exists():
        return [], ""
    a = json.loads(path.read_text())[phase]
    k = W / json.loads(path.read_text())["w"]
    rng = random.Random(9)
    out = [f'<g class="blink" fill="{a["lid"]}">']
    out += [f'<rect x="{x * k:g}" y="{y * k:g}" width="{n * k:g}" height="{k:g}"/>' for x, y, n in runs(a["eyes"])]
    out.append("</g>")
    for x, y, c in a["twinkle"]:
        out.append(f'<rect class="tw" x="{x * k:g}" y="{y * k:g}" width="{k:g}" height="{k:g}" fill="{c}" '
                   f'style="animation-duration:{rng.uniform(2.2, 4.8):.2f}s;animation-delay:-{rng.uniform(0, 5):.2f}s"/>')
    for x, y, c in a["flicker"]:
        out.append(f'<rect class="fl" x="{x * k:g}" y="{y * k:g}" width="{k:g}" height="{k:g}" fill="{c}" '
                   f'style="animation-duration:{rng.uniform(5, 11):.2f}s;animation-delay:-{rng.uniform(0, 11):.2f}s"/>')
    # the robot pushes the drawer in and pulls it out again: HALF and CLOSED drawn over the open one, in steps
    box = (898, 214, 101, 102)  # art px, the same for every state (Nest's pixel pipeline, drawer.json)
    for state in ("half", "closed"):
        f = ASSETS / f"hero-drawer-{phase}-{state}.png"
        if f.exists():
            uri = "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode("ascii")
            out.append(f'<image class="dr-{state}" x="{box[0] * k:g}" y="{box[1] * k:g}" width="{box[2] * k:g}" '
                       f'height="{box[3] * k:g}" href="{uri}" image-rendering="optimizeSpeed" style="image-rendering:pixelated"/>')
    mote = "rgba(255,246,214,.8)" if phase == "day" else "rgba(196,218,255,.55)"
    for i, (x, y) in enumerate(a["motes"]):
        out.append(f'<rect class="m{i % 3}" x="{x * k:g}" y="{y * k:g}" width="{2 * k:g}" height="{2 * k:g}" fill="{mote}" '
                   f'style="animation-duration:{rng.uniform(9, 15):.2f}s;animation-delay:-{rng.uniform(0, 15):.2f}s"/>')
    u = k  # one art px
    css = ("@keyframes blink{0%,93%,100%{opacity:0}94%,96.5%{opacity:1}}"
           ".blink{opacity:0;animation:blink 5.4s infinite}"
           "@keyframes tw{0%,62%,100%{opacity:0}74%,86%{opacity:1}}.tw{opacity:0;animation:tw 3s infinite}"
           "@keyframes fl{0%,80%,100%{opacity:0}84%,92%{opacity:.85}}.fl{opacity:0;animation:fl 7s infinite}"
           + "".join(f"@keyframes m{j}{{0%{{transform:translate(0,0);opacity:0}}15%,80%{{opacity:1}}"
                     f"100%{{transform:translate({dx * u:g}px,{dy * u:g}px);opacity:0}}}}"
                     f".m{j}{{opacity:0;animation:m{j} 12s steps({steps}) infinite}}"
                     for j, (dx, dy, steps) in enumerate(((10, -26, 26), (-8, -20, 20), (14, -14, 14))))
           + "@keyframes drh{0%,38%{opacity:0}38.01%,44%{opacity:1}44.01%,72%{opacity:0}72.01%,78%{opacity:1}78.01%,100%{opacity:0}}"
           ".dr-half{opacity:0;animation:drh 9s infinite}"
           "@keyframes drc{0%,44%{opacity:0}44.01%,72%{opacity:1}72.01%,100%{opacity:0}}"
           ".dr-closed{opacity:0;animation:drc 9s infinite}"
           + "@media (prefers-reduced-motion:reduce){.blink,.tw,.fl,.m0,.m1,.m2,.dr-half,.dr-closed{animation:none;opacity:0}}")
    return out, css


def glass(x: float, y: float, w: float, h: float, r: float) -> str:
    """The site's .glass-pill: a light veil, an inner edge."""
    return (nest.rrect(x, y, w, h, r, 'fill="rgba(255,255,255,.12)"') +
            nest.rrect(x + 0.75, y + 0.75, w - 1.5, h - 1.5, r - 0.75, 'fill="none" stroke="rgba(255,255,255,.22)" stroke-width="1.5"'))


def draw(theme: str, status: str = STATUS, title: str = TITLE, sub: str = SUB, uri: str = None) -> str:
    uri = uri or art_uri("night" if theme == "dark" else "day")
    out: List[str] = [f'<image x="0" y="0" width="{W}" height="{H}" preserveAspectRatio="none" href="{uri}" '
                      f'image-rendering="optimizeSpeed" style="image-rendering:pixelated"/>']
    moving, css = motion("night" if theme == "dark" else "day")
    out += moving
    # the top, as the site's header: the brand at the left, the button at the right
    out.append(nest.lockup(X, 112, 50, WHITE))
    bs = 28
    bw = nest.width(BUTTON, bs, 450, tracking=-0.3) + 48
    out.append(nest.rrect(W - X - bw, 66, bw, 70, 14, 'fill="#fbfbf8"'))
    out.append(nest.text(W - X - bw / 2, 66 + 35 + bs * 0.36, BUTTON, bs, "#262323", 450, anchor="middle", tracking=-0.3))
    # the words, on the wall
    ew = nest.width(EYEBROW, 22, 500, mono=True, tracking=1.2) + 44
    y = 250
    out.append(glass(X, y, ew, 50, 25))
    out.append(nest.text(X + 22, y + 32.5, EYEBROW, 22, "rgba(255,255,255,.9)", 500, mono=True, tracking=1.2))
    size = 104
    while True:
        lines = nest.wrap(title, WALL - X, size, 4, weight=400, tracking=-size * 0.04)
        if len(lines) <= 3 or size <= 72:
            break
        size -= 4
    ty = y + 50 + 34 + size * 0.86
    lh = round(size * 1.0)
    for i, line in enumerate(lines):
        out.append(nest.text(X - 4, ty + i * lh, line, size, WHITE, 400, tracking=round(-size * 0.04, 2)))
    sy = ty + (len(lines) - 1) * lh + 62
    for i, line in enumerate(nest.wrap(sub, WALL - X - 60, 30, 3, tracking=-0.4)):
        out.append(nest.text(X, sy + i * 42, line, 30, "rgba(255,255,255,.86)", 400, tracking=-0.4))
    if status:
        n_sub = len(nest.wrap(sub, WALL - X - 60, 30, 3, tracking=-0.4))
        by = sy + (n_sub - 1) * 42 + 44
        sw = nest.width(status, 28, 450, tracking=-0.3) + 48 + 30
        out.append(nest.rrect(X, by, sw, 70, 14, 'fill="#fbfbf8"'))
        out.append(f'<circle cx="{X + 32:.1f}" cy="{by + 35:.1f}" r="12" fill="rgba(80,105,89,.18)"/>'
                   f'<circle cx="{X + 32:.1f}" cy="{by + 35:.1f}" r="6.5" fill="#5d9f6c"/>')
        out.append(nest.text(X + 54, by + 35 + 28 * 0.36, status, 28, "#262323", 450, tracking=-0.3))
    label = f"Nest. {title} {sub}" + (f" {status}." if status else "")
    return nest.svg(W, H, out, label, css=css)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ASSETS))
    ap.add_argument("--status", default=STATUS, help="the line in the white button (the finding, once there is one)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for theme in nest.THEMES:
        p = out / f"hero-{theme}.svg"
        p.write_text(draw(theme, args.status))
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
