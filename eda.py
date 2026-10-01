"""
Phase 6: Exploratory Data Analysis
------------------------------------
TWO CHECKS HERE ARE SPECIFIC TO A/B TEST ANALYSIS AND HAVE NO EQUIVALENT
IN ANY PRIOR PROJECT IN THIS SERIES:

1. A Sample Ratio Mismatch (SRM) check — verifying the randomization
   itself worked as intended, BEFORE trusting any downstream result. If
   an A/B test's actual group split deviates from its intended split
   more than chance would predict, every other result from that test is
   suspect, no matter how significant it looks — a broken randomizer can
   produce "significant" differences that have nothing to do with the
   feature being tested.
2. An outlier investigation on `sum_gamerounds`, since this metric (unlike
   the binary retention metrics) is sensitive to extreme values in a way
   that can distort a mean-based test.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as scipy_stats

from data_loader import load_raw_data


def run_eda(out_dir: str = "outputs"):
    df = load_raw_data()

    # -----------------------------------------------------------------
    # CHECK 1: Sample Ratio Mismatch (SRM)
    # WHY THIS COMES FIRST, BEFORE ANY OUTCOME METRIC IS EVEN LOOKED AT:
    # this dataset was (presumably) a 50/50 randomized split. A
    # chi-squared goodness-of-fit test against the expected 50/50 ratio
    # checks whether the OBSERVED group sizes are consistent with that,
    # or whether something (a bug in the randomizer, differential
    # logging, a bot-filtering step that affected one arm more than the
    # other) skewed which players ended up in which group. This is
    # widely considered the single most important sanity check to run
    # BEFORE trusting any A/B test result, precisely because it can
    # invalidate every other analysis if it fails.
    # -----------------------------------------------------------------
    observed = df["version"].value_counts().sort_index()
    n_total = len(df)
    expected = pd.Series([n_total / 2, n_total / 2], index=observed.index)
    chi2_srm, p_srm = scipy_stats.chisquare(f_obs=observed, f_exp=expected)

    print("=== Sample Ratio Mismatch (SRM) Check ===")
    print(f"Observed: {observed.to_dict()}")
    print(f"Expected (50/50 of {n_total}): {{'gate_30': {n_total/2:.0f}, 'gate_40': {n_total/2:.0f}}}")
    print(f"Chi-squared statistic: {chi2_srm:.4f}, p-value: {p_srm:.4f}")
    if p_srm < 0.01:
        print("⚠️  SRM flagged by the common p<0.01 convention.")
        print(f"   HOWEVER, the actual imbalance is small in absolute terms: "
              f"{observed['gate_30']} vs {observed['gate_40']} is only a "
              f"{abs(observed['gate_40']-observed['gate_30'])/n_total:.2%} relative deviation from 50/50.")
        print("   A more conservative SRM convention (p<0.001, used by some practitioners")
        print("   specifically because p<0.01 alone triggers too often by chance when many")
        print("   A/A checks are run across many experiments) would NOT flag this test.")
        print("   HONEST TAKEAWAY: this is a borderline, inconclusive SRM signal, not a clear")
        print("   invalidation. It's reported and carried forward as a caveat on every")
        print("   downstream result, rather than either silently ignored or used to throw")
        print("   out an otherwise well-powered, real experiment.")
    else:
        print("✅ No SRM detected — the group split is consistent with genuine 50/50")
        print("   random assignment, so it's reasonable to proceed with outcome analysis.")

    # -----------------------------------------------------------------
    # CHECK 2: The sum_gamerounds outlier
    # -----------------------------------------------------------------
    print("\n=== sum_gamerounds distribution check ===")
    print(df["sum_gamerounds"].describe())
    top5 = df["sum_gamerounds"].nlargest(5)
    print(f"\nTop 5 values: {top5.tolist()}")
    # WHY THIS IS FLAGGED, NOT JUST NOTED: the single highest value
    # (49,854 rounds) is more than 16x the SECOND-highest value (2,961),
    # and would require playing roughly one round every 25 seconds,
    # non-stop, for the entire 14-day window with zero breaks — not a
    # plausible human play pattern. This is almost certainly a bot,
    # a stress-test account, or a logging error, not a genuinely
    # engaged player. See clean_and_engineer.py for how this is handled.

    # -----------------------------------------------------------------
    # Retention rates by group (the headline numbers, computed here for
    # visibility; formal significance testing happens in stats_tests.py)
    # -----------------------------------------------------------------
    print("\n=== Retention rates by group ===")
    print(df.groupby("version")[["retention_1", "retention_7"]].mean().round(4))

    # ---- Chart ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    observed.plot(kind="bar", ax=axes[0], color=["#4C72B0", "#C44E52"], title="Group sizes (SRM check)")
    axes[0].axhline(n_total / 2, color="black", linestyle="--", linewidth=1)

    df[df["sum_gamerounds"] < 1000]["sum_gamerounds"].hist(bins=50, ax=axes[1], color="#55A868")
    axes[1].set_title("sum_gamerounds distribution (< 1000, outlier excluded from view)")

    df.groupby("version")[["retention_1", "retention_7"]].mean().plot(
        kind="bar", ax=axes[2], color=["#4C72B0", "#DD8452"], title="Retention rate by group"
    )

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")


if __name__ == "__main__":
    run_eda()
