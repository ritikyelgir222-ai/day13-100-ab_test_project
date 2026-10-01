"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY THIS IS COMPLETELY DIFFERENT FROM EVERY PRIOR DAY'S MONITORING:
there is no deployed model whose performance can decay over time here.
What actually needs monitoring for an A/B test is the EXPERIMENT ITSELF
WHILE IT'S RUNNING — specifically, the "peeking problem": checking
results every day and stopping as soon as p < 0.05 first appears
inflates the true false-positive rate far above the nominal 5%, because
you're effectively running many sequential tests and taking the best-
looking one. This script demonstrates why, using this project's own
real data as a worked example, and computes the minimum sample size the
experiment should have been pre-registered to reach.
"""

import numpy as np
from scipy import stats as scipy_stats

from data_loader import load_raw_data
from clean_and_engineer import remove_implausible_outlier


def simulate_peeking_inflation(true_effect=0.0, n_per_look=500, n_looks=10, n_simulations=2000, random_state=42):
    """
    WHY WE SIMULATE THIS RATHER THAN JUST ASSERTING IT: demonstrating the
    peeking problem concretely, with actual simulated numbers, is far
    more convincing (and far more useful to include in a real dashboard
    tool) than simply stating "don't peek early" as an unsupported rule.
    Here we simulate MANY repeated A/A tests (true_effect=0, i.e. the two
    groups are actually identical) and check how often AT LEAST ONE of
    several sequential looks falsely shows p < 0.05, versus how often a
    single look at the final sample size would.
    """
    rng = np.random.default_rng(random_state)
    false_positive_at_any_look = 0
    false_positive_at_final_look_only = 0

    base_rate = 0.20  # arbitrary baseline conversion rate for the simulation
    for _ in range(n_simulations):
        group_a = rng.binomial(1, base_rate, size=n_per_look * n_looks)
        group_b = rng.binomial(1, base_rate + true_effect, size=n_per_look * n_looks)

        flagged_early = False
        for look in range(1, n_looks + 1):
            n = n_per_look * look
            p_a, p_b = group_a[:n].mean(), group_b[:n].mean()
            p_pool = (group_a[:n].sum() + group_b[:n].sum()) / (2 * n)
            se = np.sqrt(p_pool * (1 - p_pool) * (2 / n)) if p_pool not in (0, 1) else 1e-9
            z = (p_a - p_b) / se
            p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
            if p_val < 0.05:
                flagged_early = True
                if look == n_looks:
                    false_positive_at_final_look_only += 1
        if flagged_early:
            false_positive_at_any_look += 1

    return {
        "n_simulations": n_simulations,
        "n_sequential_looks": n_looks,
        "false_positive_rate_if_checking_only_at_the_end": round(false_positive_at_final_look_only / n_simulations, 4),
        "false_positive_rate_if_peeking_at_every_look": round(false_positive_at_any_look / n_simulations, 4),
    }


def minimum_sample_size(baseline_rate, minimum_detectable_effect, alpha=0.05, power=0.80):
    """
    Standard two-proportion sample size formula. WHY THIS MATTERS FOR
    "MONITORING": this is the check that should happen BEFORE an
    experiment starts, so that a decision to stop early or extend the
    test later can be judged against a pre-committed target, rather than
    an ad-hoc "let's just see how it looks" cutoff decided in hindsight.
    """
    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
    z_power = scipy_stats.norm.ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    p_bar = (p1 + p2) / 2

    n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_power * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (minimum_detectable_effect ** 2)
    return int(np.ceil(n))


if __name__ == "__main__":
    print("=== Peeking Problem Simulation ===")
    print("Simulating repeated A/A tests (two IDENTICAL groups) checked at 10 points")
    print("during data collection, to measure how much 'peeking' inflates false positives:\n")
    result = simulate_peeking_inflation()
    print(f"False-positive rate checking ONLY at the final sample size: "
          f"{result['false_positive_rate_if_checking_only_at_the_end']:.1%} (should be ~5%, the nominal alpha)")
    print(f"False-positive rate if stopping as soon as ANY of 10 sequential looks hits p<0.05: "
          f"{result['false_positive_rate_if_peeking_at_every_look']:.1%}")
    print("\nThis is why this project's stats_tests.py analyzes the COMPLETE, already-finished")
    print("dataset in one look, rather than simulating a sequential 'monitor as it runs' check —")
    print("the single biggest monitoring risk for an A/B test is doing that incorrectly, not")
    print("failing to do it at all.")

    print("\n=== Minimum Sample Size Check (what SHOULD have been planned upfront) ===")
    df = remove_implausible_outlier(load_raw_data())
    baseline = df[df["version"] == "gate_30"]["retention_7"].mean()
    # WHY A 1-PERCENTAGE-POINT MDE: this is roughly the size of effect
    # this experiment actually found (0.82 percentage points on
    # retention_7) — checking what sample size WOULD have been required
    # to reliably detect an effect of this size validates whether this
    # experiment was adequately powered, after the fact.
    mde = 0.01
    required_n_per_group = minimum_sample_size(baseline, mde)
    actual_n_per_group = min((df["version"] == "gate_30").sum(), (df["version"] == "gate_40").sum())

    print(f"Baseline retention_7 rate (gate_30): {baseline:.4f}")
    print(f"Minimum detectable effect used for this check: {mde:.2%} (roughly the effect size actually observed)")
    print(f"Required sample size per group (alpha=0.05, power=0.80): {required_n_per_group:,}")
    print(f"Actual smaller group size in this dataset: {actual_n_per_group:,}")
    if actual_n_per_group >= required_n_per_group:
        print("✅ This experiment WAS adequately powered to detect an effect of this size.")
    else:
        print("⚠️  This experiment was UNDER-powered for an effect of this size — a significant")
        print("   result would still be meaningful, but a non-significant one would be inconclusive")
        print("   rather than good evidence of no effect.")
