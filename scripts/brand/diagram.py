""""How it works" for the README: the paper's S1–S6 as the harness runs them, then the measurement (light and dark).

    python scripts/brand/diagram.py --out .github/assets      # writes how-light.svg, how-dark.svg

The steps and their commands are written here, by hand: change them with the harness. S5 and S6 are the steps whose
contribution the core measures (SPEC §4–5: B3 × B4 and B4 × B5); S1–S4 are run and described.
"""
import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nest  # noqa: E402

STEPS = (
    ("S1", "Collect", "Log every LLM call", "CHESS answers BIRD with the production LLM; every call is a line of calls.jsonl.", "bench run", False),
    ("S2", "Curate", "Keep what worked", "The teacher's train logs, masked, filtered on a production signal and de-duplicated.", "bench curate", False),
    ("S3", "Cluster", "Find the tasks", "Embed each prompt with its answer and cluster the calls into tasks.", "bench embed · j5", False),
    ("S4", "Select", "Pick the small model", "A desk triage of the candidates, then zero-shot on the calibration split.", "bench judge j6", False),
    ("S5", "Specialise", "Fine-tune with LoRA", "One adapter per cluster on the chosen SLM, trained on Modal's GPUs.", "bench train", True),
    ("S6", "Allocate", "Route each task", "Each cluster to the cheapest engine that is not worse than the LLM.", "bench judge j7", True),
)
MEASURE = ("Measure", "Execution accuracy and cost per correct query",
           "Seven arms on 498 held-out Arcwise-Plat-SQL questions. Every threshold, seed and split was registered "
           "before any result; the test split stays locked until then.", ("bench eval", "bench judge j3", "bench report"))
W, PAD, GAP = 960, 24, 16


def pill(x: float, y: float, label: str, on: bool, P) -> List[str]:
    w = nest.width(label, 11, 500, mono=True, tracking=1.1) + 16
    return [nest.rrect(x - w, y, w, 22, 11, f'fill="{P["greenBg"] if on else P["tile"]}"'),
            nest.text(x - w / 2, y + 15, label, 11, P["green"] if on else P["dim"], 500, mono=True, anchor="middle",
                      tracking=1.1)]


def chip(x: float, y: float, label: str, P) -> (List[str], float):
    w = nest.width(label, 13, 500, mono=True) + 20
    return [nest.rrect(x, y, w, 28, 8, f'fill="{P["tile"]}"'),
            nest.text(x + 10, y + 18.5, label, 13, P["body"] if "body" in P else P["ink"], 500, mono=True)], w


def draw(theme: str) -> str:
    P = dict(nest.PAL[theme], body=nest.PAL[theme]["muted"])
    cw = (W - 2 * PAD - 2 * GAP) / 3
    chh = 176
    top = PAD + 34
    mh = 172
    h = top + 2 * chh + GAP + GAP + mh + PAD
    out = [nest.rrect(0, 0, W, h, 18, f'fill="{P["bg"]}"')]
    out.append(nest.text(PAD + 4, PAD + 12, "THE PAPER'S ALGORITHM, RUN END TO END", 12, P["dim"], 500, mono=True, tracking=1.2))
    out.append(nest.text(W - PAD - 4, PAD + 12, "ARXIV 2506.02153 · §6", 12, P["faint"], 500, mono=True, anchor="end",
                         tracking=1.2))
    for i, (sid, verb, title, body, cmd, measured) in enumerate(STEPS):
        x = PAD + (i % 3) * (cw + GAP)
        y = top + (i // 3) * (chh + GAP)
        out.append(nest.rrect(x, y, cw, chh, 14, f'fill="{P["card"]}" stroke="{P["line"]}" stroke-width="1"'))
        out.append(nest.text(x + 20, y + 33, f"{sid} · {verb.upper()}", 12, P["dim"], 500, mono=True, tracking=1.2))
        out += pill(x + cw - 16, y + 17, "MEASURED" if measured else "RUN", measured, P)
        out.append(nest.text(x + 20, y + 66, title, 22, P["ink"], 450, tracking=-0.5))
        for j, line in enumerate(nest.wrap(body, cw - 40, 14.5, 2, tracking=-0.1)):
            out.append(nest.text(x + 20, y + 92 + j * 20, line, 14.5, P["muted"], 400, tracking=-0.1))
        c, _ = chip(x + 20, y + chh - 20 - 28, cmd, P)
        out += c
    y = top + 2 * chh + 2 * GAP
    verb, title, body, cmds = MEASURE
    out.append(nest.rrect(PAD, y, W - 2 * PAD, mh, 14, f'fill="{P["card"]}" stroke="{P["line"]}" stroke-width="1"'))
    out.append(nest.text(PAD + 20, y + 33, "MEASURE · B0–B5", 12, P["dim"], 500, mono=True, tracking=1.2))
    out.append(nest.text(PAD + 20, y + 66, title, 22, P["ink"], 450, tracking=-0.5))
    for j, line in enumerate(nest.wrap(body, W - 2 * PAD - 40, 14.5, 2, tracking=-0.1)):
        out.append(nest.text(PAD + 20, y + 92 + j * 20, line, 14.5, P["muted"], 400, tracking=-0.1))
    x = PAD + 20
    for cmd in cmds:
        c, w = chip(x, y + mh - 20 - 28, cmd, P)
        out += c
        x += w + 8
    label = ("How it works: the paper's algorithm run end to end. " +
             " ".join(f"{s} {v}: {t}, {b} ({c})." for s, v, t, b, c, _ in STEPS) + f" Then: {title}. {body}")
    return nest.svg(W, h, out, label)


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
