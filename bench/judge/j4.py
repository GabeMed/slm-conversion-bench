"""J4 · statistics (SPEC 6.4): is arm A non-inferior to arm B on the same questions?

- Δ, the margin, is **fixed before any data** and given by the caller (`margin`, in EX units): the
  report passes `thresholds.delta_pp`, J7 `thresholds.selection_delta_pp` (design §6.3 T3). It never
  depends on the discordance: "the choice of margin should be independent of considerations of power"
  (EMA 2005, EMEA/CPMP/EWP/2158/99);
- d: the paired discordance, the fraction of questions exactly one of the two arms gets right;
- diff = EX_A − EX_B, and its one-sided 95% lower bound (`ci_low`) and upper bound (`ci_high`) from
  one paired bootstrap with the question as the unit (resample questions, keep both arms' answers
  together), so confirming and refuting read one standard;
- non-inferior when that lower bound is above −Δ;
- power: `power(d, n, Δ)` = Φ(Δ/√(d/n) − z0.95), of this test at a true difference of 0 with a
  discordance d over n paired questions. The result carries it at the discordance observed here; the
  report states the **planned** power, at the pilot's d and the test's n. A low power makes a result
  inconclusive, never untestable.

Δ, diff and ci_low are in EX units (fractions of questions). The bootstrap uses
`random.Random(seed).random()`, whose sequence is fixed across Python versions. `ci_low` is the
inverted-CDF quantile (the smallest resampled value with at least 5% of the resamples at or below it:
the 500th smallest of 10000) and `ci_high` its mirror (the 500th largest), both indices computed in
integers.
"""
import math
import random
from fractions import Fraction
from statistics import NormalDist
from typing import Any, Dict, Mapping, Tuple

from bench.judge import JudgeError

CONFIDENCE = 0.95  # one-sided: z0.95
_ALPHA = 1 - Fraction(str(CONFIDENCE))  # 1/20 exactly: (1 - 0.95) * 10000 is not 500 in floating point
_Z = NormalDist()


def power(d: float, n: int, margin: float) -> float:
    """Φ(Δ/√(d/n) − z0.95): the power at a true difference of 0, discordance d over n paired questions.
    With d = 0 the arms never disagree: the bound is 0, above −Δ exactly when Δ > 0."""
    if d > 0:
        return _Z.cdf(margin / math.sqrt(d / n) - _Z.inv_cdf(CONFIDENCE))
    return 1.0 if margin > 0 else 0.0


def discordance(correct_a: Mapping[str, bool], correct_b: Mapping[str, bool]) -> float:
    """d: the fraction of the paired questions exactly one of the two arms gets right."""
    if set(correct_a) != set(correct_b) or not correct_a:
        raise JudgeError("the two arms must be scored on the same questions, and on at least one")
    return sum(bool(correct_a[q]) != bool(correct_b[q]) for q in correct_a) / len(correct_a)


def lower_quantile(values: list) -> float:
    """The inverted-CDF quantile at 1 − CONFIDENCE: the k-th smallest value, k = ⌈len·α⌉ in integers."""
    ordered = sorted(values)
    k = -(-len(ordered) * _ALPHA.numerator // _ALPHA.denominator)
    return ordered[k - 1]


def upper_quantile(values: list) -> float:
    """The mirror of `lower_quantile`: the k-th largest value, k = ⌈len·α⌉ in integers."""
    ordered = sorted(values)
    k = -(-len(ordered) * _ALPHA.numerator // _ALPHA.denominator)
    return ordered[len(ordered) - k]


def _bounds(diffs: list, seed: int, n_boot: int) -> Tuple[float, float]:
    """The one-sided lower and upper bounds of the mean difference, from the same resamples."""
    rng = random.Random(seed)
    n = len(diffs)
    means = [sum(diffs[int(rng.random() * n)] for _ in range(n)) / n for _ in range(n_boot)]
    return lower_quantile(means), upper_quantile(means)


def noninferiority(correct_a: Mapping[str, bool], correct_b: Mapping[str, bool], seed: int, n_boot: int, *,
                   margin: float) -> Dict[str, Any]:
    """A (the candidate) against B (the reference), paired by question_id: both must cover the same
    questions. `margin` is the fixed Δ, in EX units within [0, 1]."""
    d = discordance(correct_a, correct_b)
    if n_boot < 1:
        raise JudgeError(f"n_boot must be at least 1, not {n_boot}")
    if not 0 <= margin <= 1:
        raise JudgeError(f"margin is in EX units, within [0, 1], not {margin}")
    ids = sorted(correct_a)
    diffs = [int(bool(correct_a[q])) - int(bool(correct_b[q])) for q in ids]
    n = len(ids)
    a_only, b_only = diffs.count(1), diffs.count(-1)
    ci_low, ci_high = _bounds(diffs, seed, n_boot)
    return {"n": n, "a_only": a_only, "b_only": b_only, "d": d, "delta": margin, "diff": sum(diffs) / n,
            "ci_low": ci_low, "ci_high": ci_high, "noninferior": ci_low > -margin, "power": power(d, n, margin)}
