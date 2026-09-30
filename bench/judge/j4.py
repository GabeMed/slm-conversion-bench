"""J4 · statistics (SPEC 6.4): is arm A non-inferior to arm B on the same questions?

- d: the paired discordance, the fraction of questions exactly one of the two arms gets right;
- Δ = (z0.95 + z0.80)·√(d/n), with d from the pilot (`d_pilot`: the SPEC fixes the margin before
  the test, from the pilot calibration ids, `bench.data.pilot_sample`) and n the number of paired
  questions; above `delta_cap_pp` the comparison is "not testable with this n" (only descriptive).
  A `margin` given explicitly (J7 passes Δ(498)/2 from `margin`) is used as is. Without either, Δ
  is computed from the pairs being judged, reported as such (`margin_from: "pairs"`), and gives no
  verdict: a margin chosen after seeing the data is not the pre-registered one;
- diff = EX_A − EX_B, and its one-sided 95% lower bound (`ci_low`) and upper bound (`ci_high`) from
  one paired bootstrap with the question as the unit (resample questions, keep both arms' answers
  together), so confirming and refuting read one standard;
- non-inferior when that bound is above −Δ (a pilot or given margin within the cap, else None);
- power: of this test at a true difference of 0, with the discordance observed here,
  Φ(Δ/√(d/n) − z0.95). Without a pilot it is 0.80 by construction.

All of Δ, diff and ci_low are in EX units (fractions of questions); `delta_cap_pp` is in percentage
points. The bootstrap uses `random.Random(seed).random()`, whose sequence is fixed across Python
versions. `ci_low` is the inverted-CDF quantile (the smallest resampled value with at least 5% of
the resamples at or below it: the 500th smallest of 10000) and `ci_high` its mirror (the 500th
largest), both indices computed in integers.
"""
import math
import random
from fractions import Fraction
from statistics import NormalDist
from typing import Any, Dict, Mapping, Optional, Tuple

from bench.judge import JudgeError

CONFIDENCE = 0.95  # one-sided: z0.95
POWER = 0.80       # z0.80
_ALPHA = 1 - Fraction(str(CONFIDENCE))  # 1/20 exactly: (1 - 0.95) * 10000 is not 500 in floating point
_Z = NormalDist()


def margin(d: float, n: int, delta_cap_pp: float) -> Dict[str, Any]:
    """Δ for a discordance d over n paired questions, and whether it is within the cap."""
    delta = (_Z.inv_cdf(CONFIDENCE) + _Z.inv_cdf(POWER)) * math.sqrt(d / n)
    return {"delta": delta, "testable": delta * 100 <= delta_cap_pp}


_margin = margin  # `noninferiority` takes a keyword of the same name


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


def noninferiority(correct_a: Mapping[str, bool], correct_b: Mapping[str, bool], delta_cap_pp: float,
                   seed: int, n_boot: int, *, d_pilot: Optional[float] = None,
                   margin: Optional[float] = None) -> Dict[str, Any]:
    """A (the candidate) against B (the reference), paired by question_id: both must cover the
    same questions. `d_pilot` is the discordance measured on the pilot, in [0, 1]; `margin`, in EX
    units within [0, delta_cap_pp / 100], is used as is instead, and is testable by construction: a
    caller deriving it (J7's Δ(498)/2) checks first that the Δ it came from is within the cap
    (`margin(...)["testable"]`), or the comparison is not testable (SPEC 6.4). With neither, the
    margin uses the discordance of these pairs and there is no verdict (`noninferior` is None)."""
    if set(correct_a) != set(correct_b) or not correct_a:
        raise JudgeError("the two arms must be scored on the same questions, and on at least one")
    if n_boot < 1:
        raise JudgeError(f"n_boot must be at least 1, not {n_boot}")
    if d_pilot is not None and margin is not None:
        raise JudgeError("give the pilot discordance or the margin, not both")
    if d_pilot is not None and not 0 <= d_pilot <= 1:
        raise JudgeError(f"d_pilot is a fraction of questions, in [0, 1], not {d_pilot}")
    if margin is not None and not 0 <= margin <= delta_cap_pp / 100:
        raise JudgeError(f"margin must be within [0, {delta_cap_pp / 100}] (the cap, in EX units), not {margin}")
    ids = sorted(correct_a)
    diffs = [int(bool(correct_a[q])) - int(bool(correct_b[q])) for q in ids]
    n = len(ids)
    a_only, b_only = diffs.count(1), diffs.count(-1)
    d = (a_only + b_only) / n
    if margin is not None:
        rule, margin_from = {"delta": margin, "testable": True}, "given"
    else:
        rule = _margin(d if d_pilot is None else d_pilot, n, delta_cap_pp)
        margin_from = "pairs" if d_pilot is None else "pilot"
    delta = rule["delta"]
    ci_low, ci_high = _bounds(diffs, seed, n_boot)
    if d > 0:
        power = _Z.cdf(delta / math.sqrt(d / n) - _Z.inv_cdf(CONFIDENCE))
    else:  # the arms never disagree: the bound is 0, above -Δ exactly when Δ > 0
        power = 1.0 if delta > 0 else 0.0
    verdict = rule["testable"] and margin_from != "pairs"
    return {"n": n, "a_only": a_only, "b_only": b_only, "d": d, "d_pilot": d_pilot, "margin_from": margin_from,
            "delta": delta, "testable": rule["testable"], "diff": sum(diffs) / n, "ci_low": ci_low, "ci_high": ci_high,
            "noninferior": ci_low > -delta if verdict else None, "power": power}

