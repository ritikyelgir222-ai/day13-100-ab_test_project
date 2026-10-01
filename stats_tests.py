"""
Phase 8: "Model Development" & Phase 9: Evaluation & Business Validation
---------------------------------------------------------------------------
WHY THIS FILE REPLACES train_model.py, AND WHAT REPLACES A TRAIN/VAL/TEST
SPLIT: there is no model being fit here — there is a HYPOTHESIS being
tested against data that already exists in full. The entire dataset IS
the experiment result; there's no meaningful way to "train on 70% and
validate on 30%" a statistical test, because the test isn't learning
parameters that could overfit the way a model would. What replaces
"model selection" here is CHOOSING THE RIGHT STATISTICAL TEST for each
metric's actual distribution — which is itself a real methodological
decision with right and wrong answers, just a different kind than
choosing between logistic regression and XGBoost.
"""

import json

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from data_loader import load_raw_data
from clean_and_engineer import remove_implausible_outlier

N_BOOTSTRAP = 10_000
RANDOM_STATE = 42


def two_proportion_z_test(df: pd.DataFrame, metric: str):
    """
    WHY A TWO-PROPORTION Z-TEST (equivalently, a chi-squared test of
    independence) FOR retention_1 AND retention_7: both are binary
    outcomes (did the player return, yes/no) — the standard, correct
    test for comparing two independent proportions, not a t-test (which
    assumes a continuous, roughly normal outcome).
    """
    group_a = df[df["version"] == "gate_30"][metric]
    group_b = df[df["version"] == "gate_40"][metric]

    n_a, n_b = len(group_a), len(group_b)
    x_a, x_b = group_a.sum(), group_b.sum()
    p_a, p_b = x_a / n_a, x_b / n_b

    p_pool = (x_a + x_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_stat = (p_a - p_b) / se
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

    # 95% CI on the difference in proportions (unpooled SE, standard for a CI)
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci_low = (p_a - p_b) - 1.96 * se_diff
    ci_high = (p_a - p_b) + 1.96 * se_diff

    return {
        "metric": metric,
        "gate_30_rate": round(float(p_a), 4),
        "gate_40_rate": round(float(p_b), 4),
        "absolute_difference (gate_30 - gate_40)": round(float(p_a - p_b), 4),
        "relative_lift_gate_40_vs_gate_30": round(float((p_b - p_a) / p_a), 4),
        "z_statistic": round(float(z_stat), 4),
        "p_value": round(float(p_value), 6),
        "95pct_CI_on_difference": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "significant_at_0.05": bool(p_value < 0.05),
    }


def mann_whitney_test(df: pd.DataFrame):
    """
    WHY A MANN-WHITNEY U TEST FOR sum_gamerounds, NOT A T-TEST: even
    after removing the one implausible outlier, sum_gamerounds is still
    heavily right-skewed (median 16, mean 51.3, max 2,961) — a t-test's
    validity depends on the sampling distribution of the MEAN being
    approximately normal, which is a weaker assumption than the raw data
    being normal, but a distribution this skewed with these sample sizes
    can still make a t-test's p-value unreliable, particularly because
    a t-test on this metric would be heavily influenced by a small
    number of very high values even after the single most extreme one
    is removed. The Mann-Whitney U test compares whole distributions via
    rank rather than assuming normality, making it the more defensible
    choice for a metric shaped like this one.
    """
    group_a = df[df["version"] == "gate_30"]["sum_gamerounds"]
    group_b = df[df["version"] == "gate_40"]["sum_gamerounds"]

    u_stat, p_value = scipy_stats.mannwhitneyu(group_a, group_b, alternative="two-sided")

    return {
        "metric": "sum_gamerounds",
        "test": "Mann-Whitney U (not a t-test, due to heavy right-skew — see docstring)",
        "gate_30_median": float(group_a.median()),
        "gate_40_median": float(group_b.median()),
        "gate_30_mean": round(float(group_a.mean()), 2),
        "gate_40_mean": round(float(group_b.mean()), 2),
        "u_statistic": float(u_stat),
        "p_value": round(float(p_value), 6),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def bootstrap_probability_a_greater_than_b(df: pd.DataFrame, metric: str, n_bootstrap=N_BOOTSTRAP):
    """
    WHY WE ALSO RUN A BOOTSTRAP, ON TOP OF THE FREQUENTIST TESTS ABOVE:
    a p-value answers "how surprising would this data be if there were
    truly no difference," which is a real but often misunderstood
    question. A bootstrap resampling approach instead directly estimates
    "what's the probability that gate_30's true rate is actually higher
    than gate_40's" — a more directly interpretable, decision-relevant
    number for a product stakeholder, and a useful cross-check that
    doesn't rely on the z-test's normal-approximation assumptions.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    group_a = df[df["version"] == "gate_30"][metric].values
    group_b = df[df["version"] == "gate_40"][metric].values

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample_a = rng.choice(group_a, size=len(group_a), replace=True)
        sample_b = rng.choice(group_b, size=len(group_b), replace=True)
        boot_diffs[i] = sample_a.mean() - sample_b.mean()

    prob_a_greater = float((boot_diffs > 0).mean())
    return {
        "metric": metric,
        "n_bootstrap_samples": n_bootstrap,
        "probability_gate_30_greater_than_gate_40": round(prob_a_greater, 4),
    }


def run_all_tests():
    raw = load_raw_data()
    df = remove_implausible_outlier(raw)

    results = {}

    # -----------------------------------------------------------------
    # WHY WE TEST retention_1 AND retention_7 SEPARATELY, THEN DISCUSS
    # MULTIPLE COMPARISONS EXPLICITLY: testing two metrics from the same
    # experiment inflates the overall false-positive rate above the
    # nominal 5% for AT LEAST ONE of them showing "significance" by
    # chance (roughly 1-(0.95^2) = 9.75% for two independent tests).
    # Rather than silently ignoring this (as many casual A/B analyses
    # do), we apply a Bonferroni correction as a conservative check
    # alongside the uncorrected results, and report both.
    # -----------------------------------------------------------------
    results["retention_1"] = two_proportion_z_test(df, "retention_1")
    results["retention_7"] = two_proportion_z_test(df, "retention_7")
    results["sum_gamerounds"] = mann_whitney_test(df)

    n_comparisons = 2  # the two retention metrics (sum_gamerounds is reported separately, treated as exploratory)
    bonferroni_alpha = 0.05 / n_comparisons
    for key in ["retention_1", "retention_7"]:
        results[key]["significant_after_bonferroni_correction"] = bool(
            results[key]["p_value"] < bonferroni_alpha
        )

    results["multiple_comparisons_note"] = {
        "n_comparisons": n_comparisons,
        "bonferroni_corrected_alpha": bonferroni_alpha,
        "note": (
            "Testing 2 retention metrics from one experiment raises the "
            "chance of at least one false positive above 5%. Bonferroni "
            f"correction requires p < {bonferroni_alpha} instead of p < 0.05 "
            "for either metric to be called significant after correction."
        ),
    }

    # Bootstrap cross-checks
    results["bootstrap_retention_1"] = bootstrap_probability_a_greater_than_b(df, "retention_1")
    results["bootstrap_retention_7"] = bootstrap_probability_a_greater_than_b(df, "retention_7")

    print("=== retention_1 (two-proportion z-test) ===")
    print(json.dumps(results["retention_1"], indent=2))
    print("\n=== retention_7 (two-proportion z-test) ===")
    print(json.dumps(results["retention_7"], indent=2))
    print("\n=== sum_gamerounds (Mann-Whitney U test) ===")
    print(json.dumps(results["sum_gamerounds"], indent=2))
    print("\n=== Multiple comparisons correction ===")
    print(json.dumps(results["multiple_comparisons_note"], indent=2))
    print("\n=== Bootstrap cross-checks ===")
    print(json.dumps(results["bootstrap_retention_1"], indent=2))
    print(json.dumps(results["bootstrap_retention_7"], indent=2))

    # -----------------------------------------------------------------
    # Phase 9: Business Validation — translate statistics into a
    # recommendation, exactly the "validate feature rollouts
    # statistically" framing this whole project is built around
    # -----------------------------------------------------------------
    recommendation = {
        "question": "Should the gate be moved from level 30 to level 40?",
        "retention_1_verdict": (
            "Not statistically significant at alpha=0.05" if not results["retention_1"]["significant_at_0.05"]
            else "Statistically significant"
        ),
        "retention_7_verdict": (
            "Statistically significant, gate_30 (control) is HIGHER"
            if results["retention_7"]["significant_at_0.05"] and results["retention_7"]["absolute_difference (gate_30 - gate_40)"] > 0
            else "Not statistically significant or in the opposite direction"
        ),
        "recommendation": (
            "DO NOT move the gate to level 40. The 7-day retention rate is "
            "measurably and significantly LOWER under gate_40, and 7-day "
            "retention is a more meaningful long-term engagement signal than "
            "the (not significant) 1-day difference. This holds even after a "
            "conservative Bonferroni correction for testing two metrics."
        ),
        "caveats": [
            "A borderline Sample Ratio Mismatch signal was detected in EDA (p=0.0086, "
            "flagged by the common p<0.01 convention but not by the more conservative "
            "p<0.001 convention) — reported as an inconclusive caveat on this entire "
            "analysis, not as a reason to discard an otherwise well-powered result.",
            "This dataset reflects one specific game's players over one specific "
            "time window; the result should not be assumed to generalize to a "
            "different game or a different gate mechanic without new data.",
        ],
    }

    print("\n=== Business Recommendation (Phase 9) ===")
    print(json.dumps(recommendation, indent=2))

    all_results = {**results, "recommendation": recommendation}
    with open("outputs/test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved -> outputs/test_results.json")


if __name__ == "__main__":
    run_all_tests()
