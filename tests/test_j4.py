"""J4 (SPEC 6.4): a fixed margin Δ given by the caller (design §6.3 T3), the one-sided paired bootstrap
with the question as the unit, the non-inferiority verdict and the power, against values computed by
hand. Every number below is written out, not recomputed with the code under test.

Constant: z0.95 = 1.6448536."""
import pytest

from bench.judge import JudgeError
from bench.judge.j4 import lower_quantile, noninferiority, power, upper_quantile


def arms(both_right, a_only, b_only, both_wrong):
    """Two arms as {question_id: correct}, built from the four cells of the paired table."""
    cells = [(True, True)] * both_right + [(True, False)] * a_only + [(False, True)] * b_only + [(False, False)] * both_wrong
    return ({str(i): a for i, (a, _) in enumerate(cells)}, {str(i): b for i, (_, b) in enumerate(cells)})


def test_planned_power_at_n_498_with_a_5_pp_margin():
    """The pilot's d gives only the planned power at the test's n (T3): Φ(0.05/√(d/498) − z0.95)."""
    # d = 0.10: sqrt(0.10 / 498) = 0.0141705; 0.05 / 0.0141705 = 3.528456; - 1.6448536 = 1.883602; Phi = 0.9702
    assert power(0.10, 498, 0.05) == pytest.approx(0.9702, abs=1e-4)
    # d = 0.15: sqrt(0.15 / 498) = 0.0173553; 2.880972 - 1.6448536 = 1.236118; Phi = 0.8918
    assert power(0.15, 498, 0.05) == pytest.approx(0.8918, abs=1e-4)
    # d = 0.20: sqrt(0.20 / 498) = 0.0200401; 2.494995 - 1.6448536 = 0.850141; Phi = 0.8024
    assert power(0.20, 498, 0.05) == pytest.approx(0.8024, abs=1e-4)
    # d = 0.30: sqrt(0.30 / 498) = 0.0245440; 2.037155 - 1.6448536 = 0.392301; Phi = 0.6526
    assert power(0.30, 498, 0.05) == pytest.approx(0.6526, abs=1e-4)
    # arms that never disagree: the bound is 0, above -margin exactly when the margin is positive
    assert (power(0.0, 498, 0.05), power(0.0, 498, 0.0)) == (1.0, 0.0)


def test_discordance_difference_and_power_from_the_paired_table():
    # n = 500: 300 both right, 40 only A, 60 only B, 100 both wrong
    a, b = arms(300, 40, 60, 100)
    result = noninferiority(a, b, seed=1, n_boot=2000, margin=0.05)
    assert result["n"] == 500 and (result["a_only"], result["b_only"]) == (40, 60)
    assert result["d"] == pytest.approx(0.20)                      # (40 + 60) / 500
    assert result["diff"] == pytest.approx(-0.04)                  # (340 - 360) / 500
    assert result["delta"] == 0.05                                 # the margin given, as is
    # power at the observed d: Phi(0.05 / sqrt(0.20 / 500) - 1.6448536) = Phi(2.5 - 1.6448536) = Phi(0.8551464) = 0.8038
    assert result["power"] == pytest.approx(0.8038, abs=1e-4)
    # the bootstrap lower bound is near the normal one: -0.04 - 1.6448536 * sqrt((0.20 - 0.04^2) / 500) = -0.0728
    assert result["ci_low"] == pytest.approx(-0.0728, abs=0.005)
    # and the upper one: -0.04 + 1.6448536 * sqrt((0.20 - 0.04^2) / 500) = -0.0072
    assert result["ci_high"] == pytest.approx(-0.0072, abs=0.005)
    assert result["noninferior"] is False                          # -0.0728 is below -0.05
    assert set(result) == {"n", "a_only", "b_only", "d", "delta", "diff", "ci_low", "ci_high", "noninferior", "power"}


@pytest.mark.parametrize("ci_low, verdict", [(-0.0499, True), (-0.0501, False), (-0.05, False)])
def test_the_verdict_on_each_side_of_minus_5_pp(monkeypatch, ci_low, verdict):
    """Non-inferior exactly when the one-sided lower bound is above −Δ, whatever the discordance."""
    from bench.judge import j4
    monkeypatch.setattr(j4, "_bounds", lambda diffs, seed, n_boot: (ci_low, ci_low + 0.04))
    a, b = arms(300, 40, 60, 100)
    assert noninferiority(a, b, seed=1, n_boot=10, margin=0.05)["noninferior"] is verdict


def test_the_margin_does_not_move_with_the_discordance():
    """The same 4 pp lead of A, at a low and a high discordance: Δ stays the margin given (T3)."""
    for cells in ((300, 20, 0, 180), (300, 80, 60, 60)):
        a, b = arms(*cells)
        result = noninferiority(a, b, seed=1, n_boot=500, margin=0.05)
        assert result["delta"] == 0.05 and result["noninferior"] is True and result["diff"] == pytest.approx(0.04)


def test_a_bootstrap_needs_resamples():
    a, b = arms(3, 1, 0, 1)
    with pytest.raises(JudgeError, match="n_boot"):
        noninferiority(a, b, seed=1, n_boot=0, margin=0.05)


def test_the_bootstrap_quantile_against_the_exact_binomial():
    # n = 20, A misses one question B gets, all else equal: the resampled difference is -K/20 with
    # K ~ Binomial(20, 1/20). P(K >= 4) = 0.0159 < 0.05 <= P(K >= 3) = 0.0755, so the 5% quantile is -3/20.
    a, b = arms(19, 0, 1, 0)
    result = noninferiority(a, b, seed=7, n_boot=20000, margin=0.05)
    assert result["ci_low"] == pytest.approx(-0.15)
    # P(K = 0) = 0.3585 > 0.05: the top 5% of resamples are all 0
    assert result["ci_high"] == 0.0
    assert noninferiority(a, b, seed=7, n_boot=20000, margin=0.05) == result  # seeded


def test_identical_arms_resample_to_zero_only_when_paired():
    # were the arms resampled independently, the bound would fall below 0
    a, b = arms(300, 0, 0, 200)  # n = 500
    result = noninferiority(a, b, seed=3, n_boot=500, margin=0.05)
    assert (result["d"], result["diff"], result["ci_low"], result["ci_high"]) == (0.0, 0.0, 0.0, 0.0)
    assert result["noninferior"] is True and result["power"] == 1.0


def test_pairs_are_matched_by_question_id_not_by_order():
    a, b = arms(10, 3, 1, 6)
    shuffled = {q: b[q] for q in reversed(list(b))}
    assert noninferiority(a, shuffled, seed=2, n_boot=300, margin=0.05) == noninferiority(a, b, seed=2, n_boot=300, margin=0.05)
    with pytest.raises(JudgeError, match="same questions"):
        noninferiority(a, {**b, "extra": True}, seed=2, n_boot=300, margin=0.05)
    with pytest.raises(JudgeError, match="same questions"):
        noninferiority(a, {q: v for q, v in b.items() if q != "0"}, seed=2, n_boot=300, margin=0.05)


def test_the_lower_bound_is_the_500th_smallest_of_10000():
    # (1 - 0.95) * 10000 is 500.00000000000045 in floating point, whose ceiling would take the 501st
    values = [i / 10000 for i in reversed(range(10000))]  # 500th smallest 0.0499, 501st 0.05
    assert lower_quantile(values) == 0.0499
    assert lower_quantile([i / 20000 for i in range(20000)]) == 0.04995  # the 1000th, not the 1001st
    assert lower_quantile([3.0, 1.0, 2.0]) == 1.0                       # ceil(0.15) = 1: the smallest
    assert lower_quantile(list(range(30))) == 1                         # ceil(1.5) = 2: the 2nd, not the 1st


def test_the_upper_bound_is_the_mirror_the_500th_largest_of_10000():
    values = [i / 10000 for i in range(10000)]  # 500th largest 0.95; the 9500th smallest would be 0.9499
    assert upper_quantile(list(reversed(values))) == 0.95
    assert upper_quantile(list(range(30))) == 28                        # ceil(1.5) = 2: the 2nd largest
    assert upper_quantile([3.0, 1.0, 2.0]) == 3.0                       # ceil(0.15) = 1: the largest


def test_the_bootstrap_bounds_are_those_quantiles_of_one_resampling(monkeypatch):
    from bench.judge import j4
    seen = []
    monkeypatch.setattr(j4, "lower_quantile", lambda values: seen.append(values) or -0.123)
    monkeypatch.setattr(j4, "upper_quantile", lambda values: seen.append(values) or 0.456)
    a, b = arms(30, 6, 4, 10)
    result = noninferiority(a, b, seed=1, n_boot=40, margin=0.05)
    assert (result["ci_low"], result["ci_high"]) == (-0.123, 0.456)
    assert len(seen) == 2 and seen[0] == seen[1] and len(seen[0]) == 40  # the same 40 resamples


def test_the_selection_margin_is_used_as_is():
    # J7 passes thresholds.selection_delta_pp: 2.5 p.p.
    a, b = arms(300, 60, 40, 100)  # diff 0.04, d 0.20
    result = noninferiority(a, b, seed=1, n_boot=2000, margin=0.025)
    assert result["delta"] == 0.025
    # Phi(0.025 / sqrt(0.20 / 500) - 1.6448536) = Phi(1.25 - 1.6448536) = Phi(-0.3948536) = 0.3465
    assert result["power"] == pytest.approx(0.3465, abs=1e-4)
    assert result["noninferior"] is True  # ci_low about 0.0072 > -0.025


@pytest.mark.parametrize("margin", [-0.001, 1.01, 5])  # 5: percentage points where EX units are due
def test_margins_are_validated(margin):
    a, b = arms(30, 6, 4, 10)
    with pytest.raises(JudgeError, match="margin"):
        noninferiority(a, b, seed=1, n_boot=100, margin=margin)


def test_the_margin_is_required():
    a, b = arms(30, 6, 4, 10)
    with pytest.raises(TypeError):
        noninferiority(a, b, seed=1, n_boot=100)  # no margin derived from the pairs, ever
