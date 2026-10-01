# SDLC Documentation: A/B Test Analysis Dashboard
### Product Industry | Statistics (No ML Model) | Day 13 of the 100-Day Series
### Filled against the master 14-phase SDLC template, using the real project we built

Every number below comes from the actual analysis run on the real
Cookie Cats A/B test dataset. This is the first project in the series
with no predictive model at its center — several phases below explain
what replaces the usual ML-project content for each one.

---

## Phase 1: Discovery & Stakeholder Requirement Gathering
**Owner:** Business Analyst / Data Scientist | **Output:** Meeting notes, stakeholder map

- **Stakeholders identified:** product manager (problem owner, decides whether to ship), game design lead (proposed the gate change), data science team (runs the analysis)
- **Discovery findings:**
  - Current process: a live A/B test was already run (randomizing new installs into gate_30 vs. gate_40) — Phase 1 here is about correctly ANALYZING a completed experiment, not designing a new one from scratch
  - Decision this project informs: should the gate-placement change ship to 100% of players?
  - Critical framing question established immediately: which metric matters most if retention_1 and retention_7 disagree? (Answered explicitly in Phase 9: 7-day retention is treated as the more meaningful long-term signal)
- **Deliverable — Problem Statement:** "Determine, with appropriate statistical rigor, whether moving the game's first gate from level 30 to level 40 improves or harms player retention, and provide a clear ship/no-ship recommendation."

---

## Phase 2: Business Requirement Document (BRD)
**Owner:** Business Analyst | **Output:** BRD

- **Business objective:** a statistically defensible ship/no-ship recommendation, not just a directional "looks better/worse"
- **Scope:** in-scope: retention_1, retention_7, and sum_gamerounds as the three outcome metrics; out-of-scope: monetization metrics, which this dataset doesn't include
- **Success metrics / KPIs:** statistical significance at a pre-specified alpha (0.05, with a Bonferroni-corrected 0.025 also reported for the two retention metrics), not just an eyeballed difference
- **Assumptions & constraints:** single completed dataset, no ability to extend the experiment or collect more data; a borderline SRM signal is a real constraint on how confidently the result can be stated (see Phase 6)
- **Sign-off:** N/A — self-directed portfolio project

---

## Phase 3: Functional & Technical Requirement Document (FRD/TRD)
**Owner:** Data Scientist / Tech Lead | **Output:** FRD/TRD

- **Functional requirements:** correctly test each outcome metric with the statistically appropriate test for its distribution; produce a reusable calculator any FUTURE experiment (not just this one) could use
- **Non-functional requirements:** the dashboard tool must never let a user skip the SRM check — enforced by building it into every `/analyze` call automatically
- **Data sources:** a single static CSV in this exercise; a real system would pull from an experimentation platform's event logs
- **Integration points:** none in this version — a production version would integrate with a company's experimentation platform (e.g., Optimizely, a custom internal system)

---

## Phase 4: Project Planning
**Owner:** Project/Delivery Manager | **Output:** Project plan, risk register

- **Build sequence used:** data loading → EDA (SRM + outlier checks FIRST, before any outcome analysis) → cleaning → hypothesis testing → dashboard API → peeking/monitoring simulation → documentation
- **Real risks encountered during the build:**
  - Risk: the SRM check came back borderline (p=0.0086) rather than clearly passing or failing — resolved by researching and reporting BOTH common SRM thresholds (p<0.01 and the more conservative p<0.001) rather than picking whichever one gave a cleaner-looking story
  - Risk: initial instinct to just run a t-test on all three metrics uniformly — resolved by checking each metric's actual distribution shape first and choosing z-tests for the two binary metrics and a Mann-Whitney U test for the skewed continuous one

---

## Phase 5: Data Collection & Data Understanding
**Owner:** Data Engineer / Data Scientist | **Output:** Data dictionary, data quality report

- **Data inventory:** 90,189 players, 5 columns, real live A/B test data (Rasmus Baath / Kaggle)
- **Data quality report:** one clear implausible outlier (a single player logged 49,854 game rounds, over 16x the next-highest value) — documented with the specific reasoning for removing exactly this one row and no more
- **Access & governance:** public, already-anonymized dataset (numeric userid only); no additional governance required for this exercise

---

## Phase 6: Exploratory Data Analysis (EDA)
**Owner:** Data Scientist | **Output:** `eda.py`, `outputs/eda_summary.png`

**Two checks here have no equivalent in any prior day's project:**

1. **Sample Ratio Mismatch (SRM) check** — verifies the randomization itself worked, BEFORE trusting any outcome result. Actual result: chi-squared p=0.0086 (44,700 vs. 45,489, a 0.87% relative deviation from 50/50) — flagged by the common p<0.01 convention but NOT by the more conservative p<0.001 convention. Reported as a genuine, unresolved caveat rather than either dismissed or treated as fully invalidating.
2. **Outlier investigation** on `sum_gamerounds` — the single highest value (49,854) is far outside plausible human play patterns for the 14-day window.

**Retention rates by group:** retention_1: 44.82% (gate_30) vs. 44.23% (gate_40); retention_7: 19.02% (gate_30) vs. 18.20% (gate_40).

---

## Phase 7: Data Preprocessing & Feature Engineering
**Owner:** Data Scientist / ML Engineer | **Output:** `clean_and_engineer.py`

**What replaces "feature engineering" here:** there is no predictive
model, so there are no features to engineer. Phase 7's job in this
project is instead preparing the OUTCOME METRICS for valid testing —
which means deciding how to handle the one real data quality issue
(the outlier) with a specific, principled threshold, and explicitly
declining to remove anything beyond that one row, since the evidence
doesn't support a broader cleanup.

---

## Phase 8: Model Development
**Owner:** Data Scientist / ML Engineer | **Output:** `stats_tests.py`

**What replaces "model development" here:** choosing the correct
statistical test for each metric's actual distribution — a real
methodological decision with right and wrong answers, just a different
kind than choosing between logistic regression and XGBoost:

- `retention_1`, `retention_7` (binary outcomes) → two-proportion z-test
- `sum_gamerounds` (heavily right-skewed continuous) → Mann-Whitney U test, explicitly NOT a t-test, with the skew reasoning stated in the docstring

**What replaces "baseline vs. final model":** a bootstrap resampling
cross-check run alongside the frequentist tests, as an independent
verification using a different methodology — analogous in spirit to
comparing model families, but comparing INFERENCE APPROACHES instead.

---

## Phase 9: Model Evaluation & Business Validation
**Owner:** Data Scientist + Business Stakeholder | **Output:** `outputs/test_results.json`

**Actual results:**

| Metric | p-value | Significant at 0.05? | Significant after Bonferroni? |
|---|---|---|---|
| retention_1 | 0.0739 | No | No |
| retention_7 | 0.0016 | **Yes** | **Yes** |
| sum_gamerounds (Mann-Whitney) | 0.0509 | No (borderline) | N/A (not part of the correction) |

**Bootstrap cross-check (external validation):** P(gate_30 > gate_40) =
96.28% for retention_1, 99.95% for retention_7 — closely matching an
independently published analysis of this exact dataset (96.7%, ~100%),
which is strong evidence the implementation here is correct, not merely
internally consistent.

**Business recommendation:** do NOT move the gate to level 40. Every
caveat (the borderline SRM signal, the single-experiment/single-game
scope) is carried forward explicitly rather than smoothed over.

**Multiple comparisons:** since two retention metrics were tested from
one experiment, a Bonferroni correction (α=0.025 instead of 0.05) was
applied and reported alongside the uncorrected result — the retention_7
finding survives this more conservative bar.

---

## Phase 10: MLOps & Deployment
**Owner:** ML Engineer | **Output:** `app.py`

**What replaces "deploying a model" here:** there is no trained model
artifact. What gets deployed is a REUSABLE STATISTICAL CALCULATOR any
future experiment (not just this one) can use — a genuinely different
kind of "MLOps" artifact: operationalizing a repeatable ANALYSIS METHOD.

- **Verified integration:** calling the deployed `/analyze` endpoint with this project's exact retention_7 counts reproduces the identical p-value (0.001592) computed independently in `stats_tests.py` — confirming both implementations agree
- **A deliberate design choice:** the SRM check is built into every API call and takes priority in the response's plain-language summary when triggered, so a user of this tool can never accidentally skip the one check that should happen first

---

## Phase 11: Testing
**Owner:** QA / Data Scientist | **Output:** manual test log

- **What was actually tested:** full analysis pipeline run end-to-end; the dashboard API tested against the exact same real counts used in `stats_tests.py`, confirming both paths produce identical statistics — a genuine cross-implementation integration test, different in kind from the "does the API return a sensible-looking number" checks in prior days
- **What was NOT done:** no test of the dashboard against a genuinely different, second real dataset (only this project's own numbers were used to verify it); no automated test suite

---

## Phase 12: Documentation & Handover
**Owner:** Data Scientist | **Output:** `README.md`, `PROJECT_DOCUMENTATION.md`, this file

- `README.md` leads with an explicit comparison table against every ML-based day in the series, since the differences here are structural, not incremental — a reader needs to understand upfront that "train/val/test split," "model selection," and "deployment" all mean something different (or don't apply at all) in this project

---

## Phase 13: Monitoring & Maintenance
**Owner:** MLOps / Data Scientist | **Output:** `monitor.py`

**What replaces "model drift monitoring" here:** there is no deployed
model whose performance can decay. What actually needs monitoring for
an A/B test is the EXPERIMENT ITSELF while it's running — specifically
the "peeking problem."

**Actual simulation result:** checking 10 sequential points during data
collection and stopping as soon as any shows p<0.05 inflates the
false-positive rate from ~5% (checking once, at the end) to **20.4%** —
a concrete, simulated demonstration rather than an asserted rule.

**Minimum sample size check:** for an effect size matching what was
actually observed, 24,658 per group was required; the actual smaller
group had 44,699 — confirming this real experiment was adequately powered.

---

## Phase 14: Project Closure & Delivery
**Owner:** N/A (self-directed project) | **Output:** this document, `PROJECT_DOCUMENTATION.md`

- **Closure against original objective:** a clear, statistically defensible ship/no-ship recommendation was produced (do not ship), with every material caveat (borderline SRM, single-experiment scope) disclosed rather than smoothed over, and cross-validated against an independently published analysis of the same real dataset
- **Retrospective:**
  - What went well: checking the SRM and outlier issues BEFORE running any significance test, rather than after, meant those caveats could inform the interpretation of the main result from the start rather than being bolted on as an afterthought
  - What to improve next time: investigate the SRM signal's root cause directly (not possible with only the public dataset available here); consider a less conservative multiple-comparisons correction (Benjamini-Hochberg) as a documented alternative to Bonferroni, rather than reporting only the more conservative choice
- **Handover:** packaged as a complete, runnable project (code + real data + a reusable dashboard tool + documentation), consistent with every prior day, despite this project's fundamentally different technical content

---

## Mapping back to the Day 13 LinkedIn post

| LinkedIn section | Pulled from phases | Real number used |
|---|---|---|
| Client & Problem | 1–2 | Product team, gate-placement feature rollout decision |
| Requirements & Data | 3, 5 | Real Cookie Cats dataset, 90,189 players, the 49,854-round outlier |
| Approach | 6–8 | SRM + outlier checks → z-test/Mann-Whitney U, not a blanket t-test |
| Result & Business Impact | 9 | retention_7 significant (p=0.0016), recommendation: do not ship |
| Path to Production | 10–13 | Reusable A/B test dashboard + the peeking-problem simulation |
