# A/B Test Analysis Dashboard — Day 13 (Product, Statistics)

Full project code for Day 13 of the 100-day series. Uses the real
**Cookie Cats** mobile game A/B test dataset (Kaggle: Rasmus Baath):
90,189 real players, genuinely randomized into two gate-placement
versions (`gate_30` vs. `gate_40`), during a live A/B test.

## Why this project is fundamentally different from every prior day

Days 1, 2, 5, 7, 9, and 11 all built and evaluated a **predictive
model**. This project has **no model at all**. It tests a **hypothesis**
against data that already fully exists. This changes almost every phase:

| Aspect | ML projects (Days 1/2/5/7/9/11) | Day 13 (A/B test) |
|---|---|---|
| Core artifact | A trained model | A statistical test + a reusable calculator tool |
| "Train/val/test split" | Essential, for an honest accuracy estimate | **Doesn't apply** — the whole dataset IS the experiment result, analyzed in full |
| "Model selection" | Compare model families (LogReg vs. XGBoost, etc.) | Compare which **statistical test** fits each metric's actual distribution (z-test for proportions, Mann-Whitney U for skewed continuous data) |
| Evaluation metric | AUC, precision/recall, silhouette, cost | p-values, confidence intervals, effect sizes |
| "Deployment" | Serve a trained model via API | Serve a **reusable significance-testing tool** any future experiment can use |
| "Monitoring" | Watch for model drift/decay over time | Watch for **peeking** (checking results too early) and **Sample Ratio Mismatch** *during* the experiment |

## Real, honest findings from this exact dataset

- **retention_1**: no significant difference (p=0.074)
- **retention_7**: gate_30 (control) is significantly HIGHER (p=0.0016), even after a conservative Bonferroni correction for testing two metrics
- **Recommendation: do NOT move the gate to level 40** — this matches the real, published conclusion other analysts have reached on this exact dataset
- **A borderline Sample Ratio Mismatch signal** was found (p=0.0086) — flagged honestly as inconclusive rather than either ignored or used to throw out an otherwise well-powered result
- **One clear data quality issue**: a single player logged 49,854 game rounds (16x the next-highest value) — removed with documented reasoning, and reasoning for NOT removing anything beyond that one row
- **A concrete demonstration of the "peeking problem"**: simulating 10 sequential looks at identical (A/A) groups inflates the false-positive rate from ~5% to over 20%

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py      # Phase 5 — loads real data, group sizes
python eda.py                # Phase 6 — SRM check, outlier investigation
python clean_and_engineer.py # Phase 5 (fixes) + 7 — documented outlier removal
python stats_tests.py       # Phase 8-9 — the actual hypothesis tests + business recommendation
python monitor.py           # Phase 13 — peeking-problem simulation, sample size check
```

## Serve the dashboard (Phase 10)

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000/** — pre-filled with this project's own
retention_7 numbers. Try changing the group sizes to see the built-in
SRM check fire.

**Testing via curl:**
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"group_a_name":"gate_30","group_a_visitors":44699,"group_a_conversions":8501,"group_b_name":"gate_40","group_b_visitors":45489,"group_b_conversions":8279}'
```
This reproduces the exact retention_7 result from `stats_tests.py` — verified to match precisely (p=0.001592 both ways).

## File map

| File | SDLC Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Loads real data, documents source |
| `eda.py` | 6 | SRM check, outlier investigation — both unique to A/B test analysis |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Documented, minimal outlier removal |
| `stats_tests.py` | 8–9 | The actual hypothesis tests (replaces `train_model.py` — no model here) |
| `app.py` | 10 | Reusable A/B test calculator dashboard, not a model-serving API |
| `monitor.py` | 13 | Peeking-problem simulation + minimum sample size check |

## Known limitations (stated honestly)

- The borderline SRM signal (p=0.0086) is genuinely inconclusive with the tools used here — a real investigation would need access to the original randomization/logging pipeline, which isn't available for this public dataset.
- Only one outlier was removed, on a specific, stated justification — a different analyst might reasonably draw the implausibility line differently.
- The Bonferroni correction used for the two retention metrics is a conservative choice; less conservative corrections (e.g. Benjamini-Hochberg) exist and would flag the same results as significant here, but the choice itself is a judgment call worth being explicit about.
- This is one experiment, one game, one time window — the specific numeric results should not be assumed to generalize to other games or gate mechanics.
