"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the "Cookie Cats" mobile game A/B test dataset,
originally shared by data scientist Rasmus Baath and widely used as a
canonical real A/B testing case study:
    https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats

90,189 real players who installed the mobile puzzle game Cookie Cats
while a genuine, live A/B test was running. Players were randomly
assigned to one of two versions: `gate_30` (a progress-blocking gate
placed at level 30 — the control/existing experience) or `gate_40` (the
gate moved to level 40 — the proposed new experience). The dataset
records each player's total game rounds played in the first 14 days,
plus whether they returned 1 day and 7 days after installing.

WHY THIS DATASET FOR DAY 13 (A/B Test Analysis Dashboard, Product)
--------------------------------------------------------------------
- It's a REAL, completed, live A/B test with genuine random assignment
  — not a synthetic dataset built to demonstrate a statistical test.
- It directly matches the "validate feature rollouts statistically"
  framing: a real product team genuinely needed to decide whether to
  ship a gate-placement change based on this exact data.
- It contains a well-documented real data quality issue (one player
  logged 49,854 game rounds — over 16x the next-highest value) that
  provides a genuine case study in why blindly running a t-test on raw,
  unexamined data can produce a misleading result (see clean_and_engineer.py).
- Unlike every prior project in this series, THIS is the first project
  with NO predictive model at its center. There's nothing to train on
  one sample and validate on another (see the module docstring in
  stats_tests.py for what replaces the usual train/val/test split here).

If you're following along on Kaggle: download "cookie_cats.csv" from
the link above and place it at `data/cookie_cats.csv` — the schema is
identical to the file already included in this project.
"""

import pandas as pd

RAW_DATA_PATH = "data/cookie_cats.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} players, {len(df.columns)} columns from {RAW_DATA_PATH}")
    print(f"Unique player IDs: {df['userid'].nunique()} (should equal row count -- one row per player)")
    print(f"\nGroup sizes:")
    print(df["version"].value_counts())
    print(f"\nOverall retention_1 rate: {df['retention_1'].mean():.2%}")
    print(f"Overall retention_7 rate: {df['retention_7'].mean():.2%}")
