"""The README's footer: the site's footer scene, its blimps flying smoothly across (light: day; dark: night).

    python scripts/brand/footer.py --out .github/assets      # writes footer-light.svg, footer-dark.svg

The scene is `.github/assets/footer/`: a background APNG (water, wind, train, boats, birds, stars; a 40.8 s seamless
loop captured from the site's own runtime, its top dissolving into pixels of its sky) and the blimps as sprites with
their routes (sprites.json). The blimps move by CSS inside the SVG, at the browser's frame rate, each wrapping exactly
once per loop so it stays in step with the background. Made by Nest's pixel pipeline.
"""
import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / ".github" / "assets" / "footer"


def uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def draw(theme: str) -> str:
    phase = "night" if theme == "dark" else "day"
    spec = json.loads((SRC / "sprites.json").read_text())
    fr = spec.get("frame") or {}
    w, h = (fr.get("w", 1536), fr.get("h", 512)) if isinstance(fr, dict) else tuple(fr)
    pix = 'image-rendering="optimizeSpeed" style="image-rendering:pixelated"'
    out = [f'<image x="0" y="0" width="{w}" height="{h}" href="{uri(SRC / f"footer-bg-{phase}.png")}" {pix}/>']
    css = []
    for s in spec["sprites"]:
        name = s["name"]
        period = s["wrap_px"] / s["speed"]
        delay = (s["enter_x"] - s["x0"]) / s["speed"]
        css.append(f"@keyframes f-{name}{{from{{transform:translateX(0)}}to{{transform:translateX(-{s['wrap_px']}px)}}}}"
                   f".f-{name}{{animation:f-{name} {period:.3f}s linear infinite;animation-delay:-{delay:.3f}s}}")
        out.append(f'<image class="f-{name}" x="{s["enter_x"]}" y="{s["y"]}" width="{s["w"]}" height="{s["h"]}" '
                   f'href="{uri(SRC / f"{name}-{phase}.png")}" {pix}/>')
    css.append("@media (prefers-reduced-motion:reduce){" + ",".join(f".f-{s['name']}" for s in spec["sprites"]) +
               "{animation-play-state:paused}}")
    return nest.svg(w, h, out, "Pixel art of the Nest city: white towers and domes among green hills, a river with "
                    "sailboats and a bridge, snowy mountains, and airships, one towing a NEST banner.",
                    fonts=False, css="".join(css))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / ".github" / "assets"))
    out = Path(ap.parse_args(argv).out)
    for theme in nest.THEMES:
        p = out / f"footer-{theme}.svg"
        p.write_text(draw(theme))
        print(p, p.stat().st_size // 1024, "KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
