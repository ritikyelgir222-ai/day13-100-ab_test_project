"""
Phases 5 (data quality) & 7: preparing metrics for testing
------------------------------------------------------------
UNLIKE EVERY PRIOR PROJECT: there is no "feature engineering" for a
predictive model here, because there is no predictive model. What
Phase 7 means for an A/B test analysis is preparing the OUTCOME METRICS
themselves for valid statistical testing — which mostly means deciding
how to handle the one real data quality issue this dataset has.
"""

import pandas as pd

OUTLIER_THRESHOLD = 49000  # see reasoning below


def remove_implausible_outlier(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY WE REMOVE THIS ONE ROW, AND ONLY THIS ONE ROW: the single
    highest sum_gamerounds value (49,854) is over 16x the second-highest
    value (2,961) in a dataset of 90,189 players. Playing 49,854 rounds
    in 14 days requires roughly one completed round every 25 seconds,
    continuously, for the entire two-week window, with no sleep, meals,
    or breaks — not a plausible pattern for a genuine human player. This
    is almost certainly a bot, an internal test/QA account, or a
    telemetry logging error, not real player behavior the experiment is
    meant to measure.

    WHY WE DON'T REMOVE MORE THAN THIS ONE ROW: the second-highest value
    (2,961 rounds, still roughly 211 rounds/day) is high but not
    obviously implausible for a genuinely engaged player during a
    two-week span, and there's no similarly stark gap anywhere else in
    the distribution. Removing more rows than the data actually
    justifies would be "cleaning" the result toward what we expect to
    see, not toward what the data quality evidence actually supports —
    exactly the kind of unprincipled trimming a rigorous analysis has to
    avoid. A single, clearly-justified exclusion is a defensible choice;
    an arbitrary top-percentile cutoff would not be.
    """
    before = len(df)
    cleaned = df[df["sum_gamerounds"] < OUTLIER_THRESHOLD].copy()
    after = len(cleaned)
    print(f"Removed {before - after} row(s) with sum_gamerounds >= {OUTLIER_THRESHOLD} "
          f"({before} -> {after} rows)")
    return cleaned


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = remove_implausible_outlier(raw)

    print("\nsum_gamerounds summary AFTER removing the outlier:")
    print(cleaned["sum_gamerounds"].describe())

    cleaned.to_csv("outputs/cleaned_data.csv", index=False)
    print("\nSaved -> outputs/cleaned_data.csv")
