from polycite.analysis.stats import bootstrap_ci, holm_bonferroni, mcnemar_test, paired_bootstrap_diff


def test_bootstrap_ci_contains_true_mean_for_constant_values():
    ci = bootstrap_ci([1.0] * 20, n_resamples=500)
    assert ci["mean"] == 1.0
    assert ci["low"] == ci["high"] == 1.0


def test_bootstrap_ci_widens_with_more_variance():
    tight = bootstrap_ci([0.5, 0.5, 0.5, 0.5], n_resamples=500)
    wide = bootstrap_ci([0.0, 1.0, 0.0, 1.0], n_resamples=500)
    assert (wide["high"] - wide["low"]) > (tight["high"] - tight["low"])


def test_bootstrap_ci_empty():
    ci = bootstrap_ci([])
    assert ci["n"] == 0


def test_paired_bootstrap_diff_matches_point_estimate():
    a = [1.0, 1.0, 0.0, 1.0]
    b = [0.0, 0.0, 0.0, 1.0]
    result = paired_bootstrap_diff(a, b, n_resamples=500)
    assert abs(result["diff"] - 0.5) < 1e-9


def test_mcnemar_no_discordant_pairs():
    result = mcnemar_test([True, True], [True, True])
    assert result["p_value"] == 1.0


def test_mcnemar_detects_asymmetric_disagreement():
    a = [True] * 8 + [False] * 2
    b = [False] * 8 + [False] * 2
    result = mcnemar_test(a, b)
    assert result["b"] == 8
    assert result["c"] == 0
    assert result["p_value"] < 0.05


def test_holm_bonferroni_orders_by_significance():
    raw = {"x": 0.01, "y": 0.04, "z": 0.20}
    adjusted = holm_bonferroni(raw)
    assert adjusted["x"] <= adjusted["y"] <= adjusted["z"]
    assert all(v >= raw[k] for k, v in adjusted.items())
