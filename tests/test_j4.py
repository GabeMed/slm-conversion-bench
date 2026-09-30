"""J4 (SPEC 6.4): the margin Δ = (z0.95 + z0.80)·√(d/n) capped at 5 p.p., the one-sided paired
bootstrap with the question as the unit, the non-inferiority verdict and the power, against values
computed by hand. Every number below is written out, not recomputed with the code under test.

Constants: z0.95 = 1.6448536, z0.80 = 0.8416212, so z0.95 + z0.80 = 2.4864749."""
import pytest

from bench.judge import JudgeError
from bench.judge.j4 import lower_quantile, margin, noninferiority


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


def test_without_a_pilot_there_is_no_preregistered_margin_and_no_verdict():
    # SPEC 6.4 fixes the margin from the pilot before the test: a margin taken from the pairs being
    # judged is descriptive only
    a, b = arms(300, 60, 40, 100)  # A ahead by 0.04
    result = noninferiority(a, b, 5, seed=1, n_boot=2000)
    assert (result["d_pilot"], result["margin_from"], result["noninferior"]) == (None, "pairs", None)
    assert result["delta"] == pytest.approx(0.0497295, abs=1e-6)  # 2.4864749 * sqrt(0.20 / 500)
    assert result["power"] == pytest.approx(0.80, abs=1e-6)       # Phi(z0.80), by construction
    assert result["ci_low"] == pytest.approx(0.04 - 0.0328, abs=0.005)
    with_pilot = noninferiority(a, b, 5, seed=1, n_boot=2000, d_pilot=0.20)
    assert (with_pilot["margin_from"], with_pilot["noninferior"]) == ("pilot", True)


def test_a_bootstrap_needs_resamples():
    a, b = arms(3, 1, 0, 1)
    with pytest.raises(JudgeError, match="n_boot"):
        noninferiority(a, b, 5, seed=1, n_boot=0, d_pilot=0.1)


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


def test_the_lower_bound_is_the_500th_smallest_of_10000():
    # (1 - 0.95) * 10000 is 500.00000000000045 in floating point, whose ceiling would take the 501st
    values = [i / 10000 for i in reversed(range(10000))]  # 500th smallest 0.0499, 501st 0.05
    assert lower_quantile(values) == 0.0499
    assert lower_quantile([i / 20000 for i in range(20000)]) == 0.04995  # the 1000th, not the 1001st
    assert lower_quantile([3.0, 1.0, 2.0]) == 1.0                       # ceil(0.15) = 1: the smallest
    assert lower_quantile(list(range(30))) == 1                         # ceil(1.5) = 2: the 2nd, not the 1st


def test_the_bootstrap_bound_is_that_quantile(monkeypatch):
    from bench.judge import j4
    monkeypatch.setattr(j4, "lower_quantile", lambda values: -0.123)
    a, b = arms(30, 6, 4, 10)
    assert noninferiority(a, b, 5, seed=1, n_boot=40, d_pilot=0.1)["ci_low"] == -0.123


def test_a_given_margin_is_used_as_is():
    # J7 passes half the pre-registered margin: 0.0352346 / 2 = 0.0176173
    a, b = arms(300, 60, 40, 100)  # diff 0.04, d 0.20
    result = noninferiority(a, b, 5, seed=1, n_boot=2000, margin=0.0176173)
    assert (result["delta"], result["margin_from"], result["testable"]) == (0.0176173, "given", True)
    # Phi(0.0176173 / sqrt(0.20 / 500) - 1.6448536) = Phi(0.8808650 - 1.6448536) = Phi(-0.7639886) = 0.2224
    assert result["power"] == pytest.approx(0.2224, abs=1e-4)
    assert result["noninferior"] is True  # ci_low about 0.0072 > -0.0176
    assert noninferiority(a, b, 5, seed=1, n_boot=200, margin=0.05)["delta"] == 0.05  # the cap itself is allowed


@pytest.mark.parametrize("kwargs", [{"margin": -0.001}, {"margin": 0.0501}, {"d_pilot": -0.01}, {"d_pilot": 1.01},
                                    {"margin": 0.01, "d_pilot": 0.1}])
def test_margins_and_pilot_discordances_are_validated(kwargs):
    a, b = arms(30, 6, 4, 10)
    with pytest.raises(JudgeError):
        noninferiority(a, b, 5, seed=1, n_boot=100, **kwargs)


def test_a_pilot_that_never_disagreed_gives_a_zero_margin():
    # kept as is until the author decides a floor: the test becomes one of superiority
    a, b = arms(300, 0, 0, 200)
    result = noninferiority(a, b, 5, seed=1, n_boot=200, d_pilot=0.0)
    assert (result["delta"], result["testable"], result["ci_low"], result["noninferior"]) == (0.0, True, 0.0, False)
