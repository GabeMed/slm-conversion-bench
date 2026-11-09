"""Nest's look for the repository's images, as nest.technology draws it.

- The palette of the site in its two themes (the one its share images use, design/hero/benchmarks/bench.js).
- Geist and Geist Mono, embedded in each SVG as data: URIs (OFL-1.1, `fonts/OFL.txt`), so the image looks the same
  wherever it is shown, GitHub included, with no request for a font.
- Their advance widths (`fonts/metrics.json`, from the fonts at each weight used) to measure, fit and wrap text with no
  font library: the scripts need only the standard library.
- The mark (the piece) and the word "Nest" outlined from Geist 500, as in the site's header.

Nothing here reads or changes a result: the images are drawn from what `bench report` already wrote.
"""
import base64
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent

PAL: Dict[str, Dict[str, object]] = {
    "light": {"bg": "#f5f5f2", "ink": "#262323", "muted": "rgba(38,35,35,.74)", "dim": "rgba(38,35,35,.54)",
              "faint": "rgba(38,35,35,.36)", "line": "rgba(38,35,35,.14)",
              "g": ("rgb(181,233,153)", "rgb(153,233,157)"), "gEdge": "rgba(96,160,84,.6)",
              "k": ("#e2e1da", "#d8d7cf"), "kEdge": "rgba(38,35,35,.16)",
              "green": "#506959", "chip": "#ffffff", "chipLine": "rgba(38,35,35,.13)", "card": "#fbfbf8",
              "tile": "#f3f3f0", "greenBg": "#e5efe5"},
    "dark": {"bg": "#161514", "ink": "#f2f0ea", "muted": "rgba(242,240,234,.76)", "dim": "rgba(242,240,234,.58)",
             "faint": "rgba(242,240,234,.4)", "line": "rgba(242,240,234,.14)",
             "g": ("rgb(146,201,120)", "rgb(120,199,126)"), "gEdge": "rgba(0,0,0,.35)",
             "k": ("#5e5c56", "#54524d"), "kEdge": "rgba(0,0,0,.3)",
             "green": "#a6d3ad", "chip": "#2a2926", "chipLine": "rgba(242,240,234,.13)", "card": "#1f1e1c",
             "tile": "#242321", "greenBg": "#2b352c"},
}
THEMES = ("light", "dark")

# the mark: the piece, 72 × 48 (the site's assets/nest.svg)
MARK_VB = (72, 48)
MARK = "M24,0 H72 V24 H48 V48 H0 V24 H24 Z"
# "Nest" in Geist 500 with the header's tracking (-1.3 px at 28 px), in font units (1000 per em), on the baseline y = 0
WORDMARK = "M86 0V-710H223.2L551.4 -130.8V-710H659.4V0H517.4L194 -565.6V0Z M992.4 12Q915 12 858.4 -22.2Q801.8 -56.4 771 -119.1Q740.2 -181.8 740.2 -266Q740.2 -350.2 771 -412.4Q801.8 -474.6 857.6 -509.3Q913.4 -544 989.2 -544Q1061.8 -544 1116.9 -510.7Q1172 -477.4 1202.4 -414.4Q1232.8 -351.4 1232.8 -262V-235.2H850.6Q855 -157.4 892.5 -118.3Q930 -79.2 993.2 -79.2Q1040.6 -79.2 1071.5 -101Q1102.4 -122.8 1114.6 -160.4L1225 -152.8Q1203.8 -78.6 1142.2 -33.3Q1080.6 12 992.4 12ZM850.6 -315.2H1120.4Q1115.4 -386.2 1079.8 -419.9Q1044.2 -453.6 989.2 -453.6Q933 -453.6 896.5 -418.4Q860 -383.2 850.6 -315.2Z M1508.3 12Q1430.9 12 1379.4 -11.1Q1327.9 -34.2 1300.8 -74.8Q1273.7 -115.4 1269.7 -167.6L1379.1 -173.4Q1386.7 -128.6 1415.8 -102.7Q1444.9 -76.8 1508.5 -76.8Q1557.9 -76.8 1584.6 -92.9Q1611.3 -109 1611.3 -144Q1611.3 -163.6 1602.3 -176.6Q1593.3 -189.6 1567 -199.4Q1540.7 -209.2 1489.9 -218.4Q1412.5 -232.6 1368.6 -253.4Q1324.7 -274.2 1306.9 -304.7Q1289.1 -335.2 1289.1 -378.8Q1289.1 -451.8 1344.1 -497.9Q1399.1 -544 1502.7 -544Q1576.3 -544 1623.2 -519.6Q1670.1 -495.2 1694.3 -455.3Q1718.5 -415.4 1723.1 -369.2L1614.3 -363.2Q1612.7 -388.2 1600.6 -409.2Q1588.5 -430.2 1564.5 -442.7Q1540.5 -455.2 1500.7 -455.2Q1449.5 -455.2 1424.4 -435.1Q1399.3 -415 1399.3 -383.2Q1399.3 -359.4 1409.2 -344.2Q1419.1 -329 1443.9 -319.4Q1468.7 -309.8 1511.5 -302.6Q1591.7 -290 1637.7 -269.6Q1683.7 -249.2 1702.6 -218.7Q1721.5 -188.2 1721.5 -144.8Q1721.5 -70 1662.6 -29Q1603.7 12 1508.3 12Z M1997.5 0Q1921.7 0 1885.7 -34.9Q1849.7 -69.8 1849.7 -144.4V-656.6H1955.7V-151.6Q1955.7 -114.8 1971.8 -100.7Q1987.9 -86.6 2021.1 -86.6H2096.7V0ZM1768.9 -445.4V-532H2096.7V-445.4Z"
WORDMARK_W = 2128.7
CAP = 0.71  # Geist's cap height, in em


@lru_cache(maxsize=None)
def _metrics() -> Dict[str, Dict[str, float]]:
    return json.loads((HERE / "fonts" / "metrics.json").read_text())


def width(text: str, size: float, weight: int = 400, mono: bool = False, tracking: float = 0.0) -> float:
    """The advance of `text` in px at `size`, with `tracking` px after each character but the last."""
    family = "GeistMono" if mono else "Geist"
    table = _metrics().get(f"{family}-{weight}") or _metrics()[f"{family}-500"]
    fallback = table.get("n", 0.6)
    return sum(table.get(ch, fallback) for ch in text) * size + tracking * max(len(text) - 1, 0)


def fit(text: str, max_w: float, size: float, **kw) -> str:
    """`text`, cut with an ellipsis where it would be wider than `max_w`."""
    if width(text, size, **kw) <= max_w:
        return text
    while len(text) > 1 and width(text + "…", size, **kw) > max_w:
        text = text[:-1]
    return text.rstrip(" ,;:·(") + "…"


def wrap(text: str, max_w: float, size: float, max_lines: int, **kw) -> List[str]:
    """Greedy lines of `text` no wider than `max_w`; the last one is fitted if words are left over."""
    words, lines, line, i = str(text).split(" "), [], "", 0
    while i < len(words):
        trial = f"{line} {words[i]}" if line else words[i]
        if width(trial, size, **kw) <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = words[i]
            if len(lines) == max_lines:
                break
        i += 1
    if len(lines) < max_lines:
        if line:
            lines.append(line)
    elif i < len(words):
        lines[-1] = fit(lines[-1] + " " + " ".join(words[i:]), max_w, size, **kw)
    return lines


@lru_cache(maxsize=None)
def _data_uri(path: Path, kind: str) -> str:
    return f"data:{kind};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def fonts_css() -> str:
    """The two families as @font-face rules, the variable fonts (weights 100–900) inline."""
    rules = []
    for family, name in (("Geist", "Geist.woff2"), ("Geist Mono", "GeistMono.woff2")):
        rules.append(f"@font-face{{font-family:'{family}';font-weight:100 900;font-style:normal;"
                     f"src:url({_data_uri(HERE / 'fonts' / name, 'font/woff2')}) format('woff2')}}")
    return "".join(rules)


SANS = "Geist, Arial, sans-serif"
MONO = "'Geist Mono', ui-monospace, SFMono-Regular, Menlo, monospace"


def text(x: float, y: float, s: str, size: float, fill: str, weight: int = 400, mono: bool = False,
         anchor: str = "start", tracking: float = 0.0, extra: str = "") -> str:
    """One line of text: `tracking` in px, as the site's canvas letterSpacing."""
    ls = f' letter-spacing="{tracking:g}"' if tracking else ""
    an = f' text-anchor="{anchor}"' if anchor != "start" else ""
    family = MONO if mono else SANS
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{escape(family, {chr(39): "&#39;"})}" font-size="{size:g}" '
            f'font-weight="{weight}" fill="{fill}"{ls}{an}{extra}>{escape(s)}</text>')


def rrect(x: float, y: float, w: float, h: float, r: float, attrs: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r:.1f}" {attrs}/>'


def mark(x: float, base: float, w: float, fill: str) -> str:
    """The piece, `w` px wide, standing on the baseline `base`."""
    k = w / MARK_VB[0]
    return f'<path transform="translate({x:.2f} {base - MARK_VB[1] * k:.2f}) scale({k:.5f})" d="{MARK}" fill="{fill}"/>'


def wordmark(x: float, base: float, size: float, fill: str) -> str:
    """"Nest" outlined, at font size `size`, its baseline at `base`."""
    k = size / 1000
    return f'<path transform="translate({x:.2f} {base:.2f}) scale({k:.5f})" d="{WORDMARK}" fill="{fill}"/>'


def lockup(x: float, base: float, size: float, fill: str) -> str:
    """The header's brand: the mark as tall as the capitals, then "Nest" (31 px and 10 px at 28 px)."""
    w = size * 31 / 28
    return mark(x, base, w, fill) + wordmark(x + w + size * 10 / 28, base, size, fill)


def lockup_width(size: float) -> float:
    return size * 31 / 28 + size * 10 / 28 + WORDMARK_W * size / 1000


def logo(name: Optional[str], theme: str) -> Optional[str]:
    """A model maker's symbol for a chip, as a data: URI, or None (the chip shows an initial)."""
    if not name:
        return None
    for candidate in (f"{name}-{'light' if theme == 'dark' else 'dark'}.svg", f"{name}.svg"):
        path = HERE / "logos" / candidate
        if path.exists():
            return _data_uri(path, "image/svg+xml")
    return None


def svg(w: float, h: float, body: List[str], label: str, fonts: bool = True) -> str:
    style = f"<style>{fonts_css()}</style>" if fonts else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" width="{w:g}" height="{h:g}" '
            f'role="img" aria-label="{escape(label, {chr(34): "&quot;"})}"><title>{escape(label)}</title>{style}'
            + "".join(body) + "</svg>\n")


def lockup_svg(theme: str, size: float = 28) -> str:
    """The brand alone (README header): the mark and "Nest", in the theme's ink, on no background."""
    w, h = lockup_width(size), size * 1.0
    return svg(round(w + 1, 1), round(h, 1), [lockup(0.5, size * CAP + (h - size * CAP) / 2, size, PAL[theme]["ink"])],
               "Nest", fonts=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="write the brand's lockup as nest-{light,dark}.svg")
    ap.add_argument("--out", default=".")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (out / f"nest-{theme}.svg").write_text(lockup_svg(theme))
        print(out / f"nest-{theme}.svg")
