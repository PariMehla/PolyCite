"""Bootstrap CIs, McNemar's test, Holm-Bonferroni correction.

Belebele is parallel (same question ID across languages), so every
comparison here should be paired: resample question indices, not
independent per-language samples, and use McNemar rather than a two-sample
test when comparing binary correctness between two conditions on the same
questions.
"""
from __future__ import annotations

import math
import random
from typing import Sequence


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 10_000, ci: float = 0.95, seed: int = 42
) -> dict:
    """Percentile bootstrap CI on the mean of `values`."""
    values = list(values)
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    lo_idx = int(alpha * n_resamples)
    hi_idx = int((1 - alpha) * n_resamples) - 1
    return {
        "mean": sum(values) / n,
        "low": means[lo_idx],
        "high": means[hi_idx],
        "n": n,
    }


def paired_bootstrap_diff(
    values_a: Sequence[float], values_b: Sequence[float], n_resamples: int = 10_000, ci: float = 0.95, seed: int = 42
) -> dict:
    """CI on mean(a) - mean(b), resampling question indices jointly (paired)."""
    assert len(values_a) == len(values_b), "paired comparison requires equal-length, aligned sequences"
    n = len(values_a)
    if n == 0:
        return {"diff": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        a = sum(values_a[i] for i in idx) / n
        b = sum(values_b[i] for i in idx) / n
        diffs.append(a - b)
    diffs.sort()
    alpha = (1 - ci) / 2
    lo_idx = int(alpha * n_resamples)
    hi_idx = int((1 - alpha) * n_resamples) - 1
    point = sum(values_a) / n - sum(values_b) / n
    return {"diff": point, "low": diffs[lo_idx], "high": diffs[hi_idx], "n": n}


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Exact-ish McNemar (chi-square with continuity correction) for paired
    binary correctness on the same questions under two conditions."""
    assert len(correct_a) == len(correct_b)
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)  # a correct, b wrong
    c = sum(1 for a, bb in zip(correct_a, correct_b) if bb and not a)  # b correct, a wrong
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = math.exp(-statistic / 2)  # chi-sq(1) upper tail approx, adequate for reporting
    return {"statistic": statistic, "p_value": min(p_value, 1.0), "b": b, "c": c}


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Returns adjusted p-values keyed the same way as the input."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running_max = 0.0
    for i, (key, p) in enumerate(items):
        adj = min((m - i) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted[key] = running_max
    return adjusted
