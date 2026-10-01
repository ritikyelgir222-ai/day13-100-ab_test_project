# Project Documentation: A/B Test Analysis Dashboard
### Technical Documentation & Handover — Phase 12 of the SDLC

Companion to `SDLC_DOCUMENTATION.md` and `README.md`.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Statistically validate whether a feature rollout (moving a game's gate from level 30 to 40) should ship |
| **Client context** | Product team, mobile gaming |
| **Data source** | Cookie Cats A/B test dataset — [Kaggle, Rasmus Baath](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats) |
| **Dataset size** | 90,189 players, genuinely randomized 50/50 (with a borderline SRM caveat) |
| **Core artifact** | A statistical analysis + a reusable A/B test calculator (NOT a trained model) |
| **Headline result** | retention_7 significantly favors the control (gate_30); recommendation is to NOT ship the change |

---

## 2. Architecture

```
data/cookie_cats.csv (90,189 players, real live A/B test)
        │
        ▼
data_loader.py ──────► loads raw data, documents source
        │
        ▼
eda.py ────────────────► Sample Ratio Mismatch check (BEFORE trusting
        │                anything else), outlier investigation
        ▼
clean_and_engineer.py ─► removes ONE implausible outlier (documented,
        │                minimal, justified — not arbitrary trimming)
        ▼
stats_tests.py ────────► THE CORE OF THIS PROJECT: two-proportion z-tests
        │                for retention_1/7, Mann-Whitney U for
        │                sum_gamerounds, bootstrap cross-checks,
        │                Bonferroni correction, business recommendation
        │                (NO train/val/test split — see Section 5)
        ▼
outputs/test_results.json
        │
        ├──────────────► app.py ─── a REUSABLE statistical calculator,
        │                           not a model-serving API — verified
        │                           to reproduce stats_tests.py's exact
        │                           numbers
        │
        └──────────────► monitor.py ─ peeking-problem simulation +
                                       minimum sample size check
                                       (monitors the EXPERIMENT, not a
                                       deployed model)
```

---

## 3. Data Dictionary

| Column | Type | Description | Treatment |
|---|---|---|---|
| `userid` | int | Unique player ID | Used for group/uniqueness checks only |
| `version` | string | `gate_30` (control) or `gate_40` (treatment) | The experiment arm |
| `sum_gamerounds` | int | Total rounds played in first 14 days | ONE implausible outlier (49,854 rounds) removed |
| `retention_1` | bool | Returned 1 day after install | Primary outcome metric #1 |
| `retention_7` | bool | Returned 7 days after install | Primary outcome metric #2 |

---

## 4. Key EDA Findings

| Finding | Detail |
|---|---|
| Group sizes | gate_30: 44,700, gate_40: 45,489 |
| **Sample Ratio Mismatch check** | Chi-squared p=0.0086 — flagged by the common p<0.01 convention, but NOT by the more conservative p<0.001 convention used by some practitioners specifically to avoid over-triggering. Treated as a genuine, disclosed caveat, not dismissed or over-reacted to. |
| Outlier | One player logged 49,854 game rounds — over 16x the second-highest value (2,961). Implies playing continuously, one round every ~25 seconds, for 14 straight days with no breaks — not plausible human behavior. |
| Retention rates | retention_1: 44.82% (gate_30) vs. 44.23% (gate_40). retention_7: 19.02% (gate_30) vs. 18.20% (gate_40) |

---

## 5. Why There Is No Train/Val/Test Split In This Project

Every prior project in this series split data specifically to get an
honest estimate of how a MODEL would perform on data it hadn't seen. An
A/B test analysis has no equivalent concept: the dataset **is** the
complete result of an experiment that already ran to completion. There
is nothing to "hold out" and predict — the entire point is to describe
what actually happened in the full dataset as accurately and honestly as
possible. This is stated explicitly (in `stats_tests.py`'s module
docstring) rather than silently omitted, since a reader familiar with
the rest of the series might otherwise wonder where the split went.

---

## 6. Statistical Results

### retention_1 (two-proportion z-test)

| | Value |
|---|---|
| gate_30 rate | 44.82% |
| gate_40 rate | 44.23% |
| Absolute difference | 0.59 pp |
| z-statistic | 1.787 |
| **p-value** | **0.0739** |
| 95% CI on difference | [-0.06pp, 1.24pp] |
| Significant at α=0.05? | **No** |

### retention_7 (two-proportion z-test)

| | Value |
|---|---|
| gate_30 rate | 19.02% |
| gate_40 rate | 18.20% |
| Absolute difference | 0.82 pp |
| z-statistic | 3.157 |
| **p-value** | **0.0016** |
| 95% CI on difference | [0.31pp, 1.33pp] |
| Significant at α=0.05? | **Yes** |
| Significant after Bonferroni correction (α=0.025)? | **Yes** |

### sum_gamerounds (Mann-Whitney U test — not a t-test, see reasoning in `stats_tests.py`)

| | Value |
|---|---|
| gate_30 median | 17.0 |
| gate_40 median | 16.0 |
| **p-value** | **0.0509** |
| Significant at α=0.05? | No (borderline) |

### Bootstrap cross-checks (10,000 resamples each)

| Metric | P(gate_30 > gate_40) |
|---|---|
| retention_1 | 96.28% |
| retention_7 | 99.95% |

**External validation:** these bootstrap results closely match the
independently published analysis of this exact dataset by other analysts
(96.7% and ~100% respectively) — strong evidence this implementation is
correct, not just internally consistent.

---

## 7. Business Recommendation (Phase 9)

**Do NOT move the gate from level 30 to level 40.** The 7-day retention
rate — a more meaningful long-term engagement signal than the
(non-significant) 1-day difference — is measurably and significantly
lower under gate_40, and this holds even after a conservative Bonferroni
correction for testing two metrics from the same experiment.

**Caveats carried forward explicitly, not hidden:**
1. The borderline SRM signal (Section 4) means there's a small,
   unresolved question about whether randomization behaved perfectly —
   disclosed as a caveat on the whole analysis, not used to discard an
   otherwise well-powered result.
2. This is one game, one time window — the specific numbers shouldn't be
   assumed to transfer to a different product or gate mechanic.

---

## 8. API Reference (the Dashboard)

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser dashboard, pre-filled with this project's real numbers |
| `/docs` | GET | Interactive Swagger UI |
| `/health` | GET | Health check |
| `/analyze` | POST | Run a full two-proportion test on ANY two-group summary counts |

**Verified:** calling `/analyze` with this project's exact retention_7
counts (44,699/8,501 vs. 45,489/8,279) reproduces the identical p-value
(0.001592) computed independently in `stats_tests.py` — confirming the
two implementations agree, not just that each runs without error.

**A deliberate design choice:** the SRM check is built into every
`/analyze` call and takes priority in the plain-language summary when it
fires — so a user of this dashboard can never accidentally skip the one
check that should happen before trusting any other result.

---

## 9. Monitoring: The Peeking Problem (Phase 13)

Simulating 2,000 repeated A/A tests (two genuinely identical groups),
checked at 10 sequential points during data collection:

| Approach | False-positive rate |
|---|---|
| Checking only at the final sample size | 4.2% (close to the nominal 5% — validates the simulation itself) |
| Stopping as soon as ANY of 10 sequential looks hits p<0.05 | **20.4%** |

This is a concrete, simulated demonstration of why this project analyzes
the complete, already-finished dataset in a single look, rather than
simulating an in-flight "check every day and stop early" pattern.

**Minimum sample size check:** for an effect size roughly matching what
was actually observed (1 percentage point), the required sample size per
group was 24,658; the actual smaller group had 44,699 — this experiment
**was** adequately powered.

---

## 10. Known Limitations

1. **The SRM signal is genuinely unresolved.** A definitive investigation would require access to the original randomization/logging system, unavailable for this public dataset.
2. **Outlier removal is a judgment call**, even though a specific, stated threshold was used — a different analyst might draw the line differently, though the reasoning for removing exactly one row (and no more) is documented in full.
3. **Bonferroni correction is conservative** — a less conservative method (e.g. Benjamini-Hochberg) would reach the same significance conclusions here, but the choice of correction method is itself worth being explicit about rather than silently picking one.
4. **This is a completed, retrospective analysis**, not a live monitoring system — Phase 13's "peeking" simulation is illustrative of a general principle, not a monitor watching this specific (already-finished) experiment in real time.

---

## 11. File Map

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load raw data, document source |
| `eda.py` | 6 | SRM check, outlier investigation |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Documented outlier removal |
| `stats_tests.py` | 8–9 | Hypothesis tests, bootstrap, business recommendation |
| `app.py` | 10 | Reusable A/B test calculator dashboard |
| `monitor.py` | 13 | Peeking simulation, sample size check |
| `README.md` | 12 | Setup instructions, comparison to ML-based days |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation |
| `SDLC_DOCUMENTATION.md` | 1–14 | Full 14-phase narrative |
