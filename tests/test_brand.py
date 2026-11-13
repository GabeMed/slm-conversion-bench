"""The README's images (scripts/brand): drawn from report.json alone, outside the registered analysis code."""
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "scripts" / "brand"


def _load(name):
    sys.path.insert(0, str(BRAND))
    spec = importlib.util.spec_from_file_location(f"brand_{name}", BRAND / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


card = _load("results_card")
headline = _load("headline_card")
diagram = _load("diagram")
hero = _load("hero")
footer = _load("footer")
nest = _load("nest")

CONFIG = {
    "roles": {"production_llm": {"model": "claude-sonnet-5-5"}, "cheap_alt": {"model": "openai/gpt-oss-120b"},
              "slm_candidates": [{"name": "qwen3-8b", "base": "Qwen3-8B", "hf": {"repo": "Qwen/Qwen3-8B"}}]},
    "arms": {"B4": {"choice": "qwen3-8b"}},
}
REPORT = {"split": "test", "utilizations": ["20%", "50%", "100%"], "arms": {
    "B0": {"n": 498, "ex": 0.6, "cost_per_correct": {"": 0.04}},
    "B1": {"n": 498, "ex": 0.5, "cost_per_correct": {"": 0.006}},
    "B4": {"n": 498, "ex": 0.59, "cost_per_correct": {"20%": 0.012, "50%": 0.006, "100%": 0.004}},
    "B5": {"n": 498, "ex": None, "cost_per_correct": {}},
}}


def _svg(text):
    root = ET.fromstring(text)
    assert root.tag.endswith("svg")
    return root


def test_names_come_from_the_configuration():
    names = card.arm_names(CONFIG)
    assert names["B0"][0] == "Claude Sonnet 5.5" and names["B0"][2] == "anthropic"
    assert names["B1"][0] == "gpt-oss-120b" and names["B1"][2] == "openai"
    assert names["B4"] == ("Qwen3-8B", "fine-tuned", "qwen")
    unchosen = card.arm_names({"roles": CONFIG["roles"]})
    assert unchosen["B4"][0] == "SLM" and unchosen["B4"][2] is None


def test_slm_named_from_the_registered_judgment():
    """config.yaml names the SLM through J6's judgment file, not by name: the card reads it (Qwen3-8B, Qwen's logo)"""
    import yaml
    config = yaml.safe_load((card.ROOT / "config.yaml").read_text())
    name, mk = card.slm_of(config)
    assert (name, mk) == ("Qwen3-8B", "qwen")
    assert card.arm_names(config)["B4"] == ("Qwen3-8B", "fine-tuned", "qwen")


def test_bars_are_the_report_numbers():
    bars = card.bars_of(REPORT, CONFIG)
    assert [b["arm"] for b in bars] == ["B0", "B1", "B4"]  # an arm with no EX is not drawn
    b0, _, b4 = bars
    assert (b0["txt"], b0["cost"], b0["green"]) == ("60.0%", "$0.04", False)
    assert (b4["txt"], b4["green"]) == ("59.0%", True)
    assert b4["cost"] == "$0.004–0.012"  # full use first, the lowest last


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_card_and_pending_card_are_svg(theme):
    root = _svg(card.draw(card.bars_of(REPORT, CONFIG), theme, title="A title"))
    label = root.get("aria-label")
    assert "B0 Claude Sonnet 5.5" in label and "60.0%" in label
    pending = _svg(card.draw(card.pending_bars(CONFIG), theme, pending="Pending."))
    assert "%" not in "".join(t.text or "" for t in pending.iter() if t.tag.endswith("text"))


def test_cli_writes_both_themes(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps(REPORT))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(json.dumps(CONFIG))  # JSON is YAML
    assert card.main([str(tmp_path), "--config", str(cfg)]) == 0
    for theme in ("light", "dark"):
        _svg((tmp_path / f"results-{theme}.svg").read_text())


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_diagram_and_lockup_are_svg(theme):
    _svg(diagram.draw(theme))
    _svg(nest.lockup_svg(theme))


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_hero_is_svg_with_its_words(theme):
    root = _svg(hero.draw(theme, status="A finding", uri="data:image/png;base64,"))
    assert "A finding" in root.get("aria-label")
    words = " ".join(t.text or "" for t in root.iter() if t.tag.endswith("text"))
    assert "A finding" in words and "fine-tuned SLM" in words


def test_text_fits():
    line = nest.fit("x" * 200, 100, 16)
    assert line.endswith("…") and nest.width(line, 16) <= 100
    assert all(nest.width(t, 20) <= 300 for t in nest.wrap("word " * 40, 300, 20, 2))


def test_outside_the_registered_analysis_code():
    from bench.prereg import ANALYSIS_CODE
    assert not any("scripts/brand".startswith(p) or p.startswith("scripts") for p in ANALYSIS_CODE)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_footer_is_svg_with_moving_blimps(theme):
    text = footer.draw(theme)
    root = _svg(text)
    assert "@keyframes f-banner" in text and len([e for e in root.iter() if e.tag.endswith("image")]) >= 2


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_headline_card_reads_the_report(theme):
    """the animated headline reads report.json and T2: B5 against B0, accuracy, cost per call, answer time"""
    report = json.loads(headline.REPORT.read_text())
    v = headline.numbers(report)
    b0, b5 = report["arms"]["B0"], report["arms"]["B5"]
    assert v["ref_call"] == pytest.approx(b0["cost_per_correct"][next(iter(b0["cost_per_correct"]))] * b0["ex"] * b0["n"] / b0["calls"])
    assert v["x_full"] == pytest.approx(v["ref_call"] / v["hyb_call_full"]) and v["x_full"] > v["x_low"] > 1
    assert v["diff"] == pytest.approx(b5["ex"] - b0["ex"], abs=1e-9)
    assert v["ref_s"] and v["hyb_s"]
    out = headline.draw(v, theme)
    assert "@keyframes" in out                         # animated: the three charts, then the three cards
    root = _svg(out)
    words = "".join(t.text or "" for t in root.iter() if t.tag.endswith("text"))
    for w in ("ACCURACY", "COST PER CALL", "TIME TO ANSWER", "more accurate", "cheaper per call", "faster answers",
              "Qwen3-8B fine-tuned", "Claude Sonnet 5.5"):
        assert w in words
