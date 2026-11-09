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

EYEBROW = "arXiv 2506.02153 · pre-registered"
TITLE = "Which LLM calls of a real agent can a fine-tuned SLM take over?"
SUB = ("NVIDIA's LLM-to-SLM conversion, run end to end on a real agent: CHESS answering BIRD text-to-SQL, "
       "with the protocol fixed before any result.")
STATUS = "Stage 0 registered · results pending"
BUTTON = "nest.technology/benchmarks"
WHITE = "#ffffff"


def art_uri(phase: str) -> str:
    return "data:image/png;base64," + base64.b64encode((ASSETS / f"hero-art-{phase}.png").read_bytes()).decode("ascii")


def glass(x: float, y: float, w: float, h: float, r: float) -> str:
    """The site's .glass-pill: a light veil, an inner edge."""
    return (nest.rrect(x, y, w, h, r, 'fill="rgba(255,255,255,.12)"') +
            nest.rrect(x + 0.75, y + 0.75, w - 1.5, h - 1.5, r - 0.75, 'fill="none" stroke="rgba(255,255,255,.22)" stroke-width="1.5"'))


def draw(theme: str, status: str = STATUS, title: str = TITLE, sub: str = SUB, uri: str = None) -> str:
    uri = uri or art_uri("night" if theme == "dark" else "day")
    out: List[str] = [f'<image x="0" y="0" width="{W}" height="{H}" preserveAspectRatio="none" href="{uri}" '
                      f'image-rendering="optimizeSpeed" style="image-rendering:pixelated"/>']
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
        lines = nest.wrap(title, WALL - X, size, 4, weight=500, tracking=-size * 0.045)
        if len(lines) <= 3 or size <= 72:
            break
        size -= 4
    ty = y + 50 + 34 + size * 0.86
    lh = round(size * 1.0)
    for i, line in enumerate(lines):
        out.append(nest.text(X - 4, ty + i * lh, line, size, WHITE, 500, tracking=round(-size * 0.045, 2)))
    sy = ty + (len(lines) - 1) * lh + 62
    for i, line in enumerate(nest.wrap(sub, WALL - X - 60, 30, 3, tracking=-0.4)):
        out.append(nest.text(X, sy + i * 42, line, 30, "rgba(255,255,255,.86)", 400, tracking=-0.4))
    n_sub = len(nest.wrap(sub, WALL - X - 60, 30, 3, tracking=-0.4))
    by = sy + (n_sub - 1) * 42 + 44
    sw = nest.width(status, 28, 450, tracking=-0.3) + 48 + 30
    out.append(nest.rrect(X, by, sw, 70, 14, 'fill="#fbfbf8"'))
    out.append(f'<circle cx="{X + 32:.1f}" cy="{by + 35:.1f}" r="12" fill="rgba(80,105,89,.18)"/>'
               f'<circle cx="{X + 32:.1f}" cy="{by + 35:.1f}" r="6.5" fill="#5d9f6c"/>')
    out.append(nest.text(X + 54, by + 35 + 28 * 0.36, status, 28, "#262323", 450, tracking=-0.3))
    label = f"Nest. {title} {sub} {status}."
    return nest.svg(W, H, out, label)


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
