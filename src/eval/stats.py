"""Statistical testing utilities for experiment evaluation.

Implements:
- Paired Wilcoxon signed-rank test (primary)
- Bootstrap confidence intervals
- Multi-seed result aggregation
"""

import numpy as np
from scipy import stats


def paired_wilcoxon(scores_a, scores_b, alternative="two-sided"):
    """Paired Wilcoxon signed-rank test.

    Args:
        scores_a: list/array of scores for method A (one per seed)
        scores_b: list/array of scores for method B (one per seed)
        alternative: "two-sided", "greater", or "less"

    Returns:
        dict with statistic, p_value, significant (at p<0.05)
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    assert len(a) == len(b) >= 5, f"Need ≥5 paired observations, got {len(a)}"

    stat, p = stats.wilcoxon(a, b, alternative=alternative)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant_005": p < 0.05,
        "significant_001": p < 0.01,
        "n_pairs": len(a),
        "mean_diff": float(np.mean(a - b)),
        "alternative": alternative,
    }


def bonferroni_correct(p_values):
    """Apply Bonferroni correction to a list of p-values."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean.

    Args:
        scores: list/array of scores (one per seed)
        n_bootstrap: number of bootstrap resamples
        ci: confidence level (default 0.95)

    Returns:
        dict with mean, std, median, ci_low, ci_high
    """
    rng = np.random.RandomState(seed)
    scores = np.array(scores)
    n = len(scores)

    means = []
    for _ in range(n_bootstrap):
        sample = scores[rng.randint(0, n, size=n)]
        means.append(np.mean(sample))
    means = np.array(means)

    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "median": float(np.median(scores)),
        "ci_low": float(np.percentile(means, 100 * alpha)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha))),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
    }


def summarize_seeds(seed_results):
    """Summarize accuracy across seeds.

    Args:
        seed_results: list of dicts, each with at least "accuracy" or "best_test_accuracy"

    Returns:
        dict with mean, std, median, min, max, all_seeds
    """
    accs = []
    for r in seed_results:
        acc = r.get("accuracy") or r.get("best_test_accuracy") or r.get("test_accuracy")
        if acc is not None:
            accs.append(float(acc))

    if not accs:
        return {"error": "no accuracy values found"}

    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "median": float(np.median(accs)),
        "min": float(np.min(accs)),
        "max": float(np.max(accs)),
        "n_seeds": len(accs),
        "all_seeds": accs,
    }
