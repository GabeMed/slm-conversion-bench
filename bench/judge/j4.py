"""J4 · statistics (SPEC 6.4): is arm A non-inferior to arm B on the same questions?

- d: the paired discordance, the fraction of questions exactly one of the two arms gets right;
- Δ = (z0.95 + z0.80)·√(d/n), with d from the pilot when given (the SPEC fixes the margin before
  the test, from ~50 calibration questions) and n the number of paired questions; above
  `delta_cap_pp` the comparison is "not testable with this n" (only descriptive);
- diff = EX_A − EX_B, and its one-sided 95% lower bound by a paired bootstrap with the question as
  the unit (resample questions, keep both arms' answers together);
- non-inferior when that bound is above −Δ;
- power: of this test at a true difference of 0, with the discordance observed here,
  Φ(Δ/√(d/n) − z0.95). Without a pilot it is 0.80 by construction.

All of Δ, diff and ci_low are in EX units (fractions of questions); `delta_cap_pp` is in percentage
points. The bootstrap uses `random.Random(seed).random()`, whose sequence is fixed across Python
versions, and the inverted-CDF quantile (the smallest resampled value with at least 5% of the
resamples at or below it).
"""
import math
import random
from statistics import NormalDist
from typing import Any, Dict, Mapping, Optional

from bench.judge import JudgeError

CONFIDENCE = 0.95  # one-sided: z0.95
POWER = 0.80       # z0.80
_Z = NormalDist()


def margin(d: float, n: int, delta_cap_pp: float) -> Dict[str, Any]:
    """Δ for a discordance d over n paired questions, and whether it is within the cap."""
    delta = (_Z.inv_cdf(CONFIDENCE) + _Z.inv_cdf(POWER)) * math.sqrt(d / n)
    return {"delta": delta, "testable": delta * 100 <= delta_cap_pp}


def _lower_bound(diffs: list, seed: int, n_boot: int) -> float:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[int(rng.random() * n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[math.ceil((1 - CONFIDENCE) * n_boot) - 1]


def noninferiority(correct_a: Mapping[str, bool], correct_b: Mapping[str, bool], delta_cap_pp: float,
                   seed: int, n_boot: int, *, d_pilot: Optional[float] = None) -> Dict[str, Any]:
    """A (the candidate) against B (the reference), paired by question_id: both must cover the
    same questions. `d_pilot` is the discordance measured on the pilot; without it the margin uses
    the discordance of these pairs."""
    if set(correct_a) != set(correct_b) or not correct_a:
        raise JudgeError("the two arms must be scored on the same questions, and on at least one")
    ids = sorted(correct_a)
    diffs = [int(bool(correct_a[q])) - int(bool(correct_b[q])) for q in ids]
    n = len(ids)
    a_only, b_only = diffs.count(1), diffs.count(-1)
    d = (a_only + b_only) / n
    rule = margin(d if d_pilot is None else d_pilot, n, delta_cap_pp)
    delta = rule["delta"]
    ci_low = _lower_bound(diffs, seed, n_boot)
    if d > 0:
        power = _Z.cdf(delta / math.sqrt(d / n) - _Z.inv_cdf(CONFIDENCE))
    else:  # the arms never disagree: the bound is 0, above -Δ exactly when Δ > 0
        power = 1.0 if delta > 0 else 0.0
    return {"n": n, "a_only": a_only, "b_only": b_only, "d": d, "d_pilot": d_pilot,
            "delta": delta, "testable": rule["testable"], "diff": sum(diffs) / n, "ci_low": ci_low,
            "noninferior": ci_low > -delta if rule["testable"] else None, "power": power}
