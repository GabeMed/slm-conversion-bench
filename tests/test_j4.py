"""J4 (SPEC 6.4): the margin Δ = (z0.95 + z0.80)·√(d/n) capped at 5 p.p., the one-sided paired
bootstrap with the question as the unit, the non-inferiority verdict and the power, against values
computed by hand. Every number below is written out, not recomputed with the code under test.

Constants: z0.95 = 1.6448536, z0.80 = 0.8416212, so z0.95 + z0.80 = 2.4864749."""
import pytest

from bench.judge import JudgeError
from bench.judge.j4 import margin, noninferiority


def arms(both_right, a_only, b_only, both_wrong):
    """Two arms as {question_id: correct}, built from the four cells of the paired table."""
    cells = [(True, True)] * both_right + [(True, False)] * a_only + [(False, True)] * b_only + [(False, False)] * both_wrong
    return ({str(i): a for i, (a, _) in enumerate(cells)}, {str(i): b for i, (_, b) in enumerate(cells)})


def test_margin_at_n_498():
    # d = 0.10: sqrt(0.10 / 498) = 0.0141705, x 2.4864749 = 0.0352346 -> 3.52 p.p., under the cap
    assert margin(0.10, 498, 5) == {"delta": pytest.approx(0.0352346, abs=1e-6), "testable": True}
    # d = 0.25: sqrt(0.25 / 498) = 0.0224055, x 2.4864749 = 0.0557108 -> 5.57 p.p., above the cap
    assert margin(0.25, 498, 5) == {"delta": pytest.approx(0.0557108, abs=1e-6), "testable": False}
    # delta is exactly 5 p.p. at d = 498 * (0.05 / 2.4864749)^2 = 0.2013730
    assert margin(0.20137, 498, 5)["testable"] is True
    assert margin(0.20138, 498, 5)["testable"] is False


def test_discordance_difference_margin_and_power_from_the_paired_table():
    # n = 500: 300 both right, 40 only A, 60 only B, 100 both wrong
    a, b = arms(300, 40, 60, 100)
    result = noninferiority(a, b, 5, seed=1, n_boot=2000, d_pilot=0.10)
    assert result["n"] == 500 and (result["a_only"], result["b_only"]) == (40, 60)
    assert result["d"] == pytest.approx(0.20)                      # (40 + 60) / 500
    assert result["diff"] == pytest.approx(-0.04)                  # (340 - 360) / 500
    assert result["delta"] == pytest.approx(0.0351640, abs=1e-6)   # 2.4864749 * sqrt(0.10 / 500)
    # power at a true difference of 0 with the observed d: Phi(0.0351640 / sqrt(0.20 / 500) - 1.6448536)
    # = Phi(1.7582007 - 1.6448536) = Phi(0.1133471) = 0.5451
    assert result["power"] == pytest.approx(0.5451, abs=1e-4)
    # the bootstrap lower bound is near the normal one: -0.04 - 1.6448536 * sqrt((0.20 - 0.04^2) / 500) = -0.0728
    assert result["ci_low"] == pytest.approx(-0.0728, abs=0.005)
    assert result["testable"] is True and result["noninferior"] is False


def test_without_a_pilot_the_margin_comes_from_the_pairs_and_power_is_the_design_power():
    a, b = arms(300, 60, 40, 100)  # A ahead by 0.04
    result = noninferiority(a, b, 5, seed=1, n_boot=2000)
    assert result["d_pilot"] is None
    assert result["delta"] == pytest.approx(0.0497295, abs=1e-6)  # 2.4864749 * sqrt(0.20 / 500)
    assert result["power"] == pytest.approx(0.80, abs=1e-6)       # Phi(z0.80), by construction
    assert result["ci_low"] == pytest.approx(0.04 - 0.0328, abs=0.005) and result["noninferior"] is True


def test_the_bootstrap_quantile_against_the_exact_binomial():
    # n = 20, A misses one question B gets, all else equal: the resampled difference is -K/20 with
    # K ~ Binomial(20, 1/20). P(K >= 4) = 0.0159 < 0.05 <= P(K >= 3) = 0.0755, so the 5% quantile is -3/20.
    a, b = arms(19, 0, 1, 0)
    result = noninferiority(a, b, 5, seed=7, n_boot=20000, d_pilot=0.10)
    assert result["ci_low"] == pytest.approx(-0.15)
    assert noninferiority(a, b, 5, seed=7, n_boot=20000, d_pilot=0.10) == result  # seeded


def test_identical_arms_resample_to_zero_only_when_paired():
    # were the arms resampled independently, the bound would fall below 0
    a, b = arms(300, 0, 0, 200)  # n = 500: delta = 3.52 p.p. with d_pilot = 0.10, testable
    result = noninferiority(a, b, 5, seed=3, n_boot=500, d_pilot=0.10)
    assert (result["d"], result["diff"], result["ci_low"]) == (0.0, 0.0, 0.0)
    assert result["noninferior"] is True and result["power"] == 1.0


def test_above_the_cap_the_verdict_is_not_testable_but_the_numbers_are_reported():
    a, b = arms(300, 60, 40, 100)
    result = noninferiority(a, b, 5, seed=1, n_boot=500, d_pilot=0.30)
    assert result["delta"] == pytest.approx(0.0609061, abs=1e-6)  # 2.4864749 * sqrt(0.30 / 500)
    assert (result["testable"], result["noninferior"]) == (False, None)
    assert result["diff"] == pytest.approx(0.04) and result["ci_low"] is not None


def test_pairs_are_matched_by_question_id_not_by_order():
    a, b = arms(10, 3, 1, 6)
    shuffled = {q: b[q] for q in reversed(list(b))}
    assert noninferiority(a, shuffled, 5, seed=2, n_boot=300) == noninferiority(a, b, 5, seed=2, n_boot=300)
    with pytest.raises(JudgeError, match="same questions"):
        noninferiority(a, {**b, "extra": True}, 5, seed=2, n_boot=300)
    with pytest.raises(JudgeError, match="same questions"):
        noninferiority(a, {q: v for q, v in b.items() if q != "0"}, 5, seed=2, n_boot=300)
