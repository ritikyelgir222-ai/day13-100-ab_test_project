"""
Phase 10: MLOps & Deployment
--------------------------------
WHY THIS IS FUNDAMENTALLY DIFFERENT FROM EVERY PRIOR DAY'S "DEPLOYMENT":
there is no trained model artifact to load and serve here. What gets
deployed is a REUSABLE STATISTICAL CALCULATOR — a product team could
paste in the summary counts from ANY future A/B test (not just this
Cookie Cats data) and get back a properly-conducted significance test,
confidence interval, and plain-language recommendation. This is Day 13's
version of "MLOps": operationalizing a repeatable ANALYSIS METHOD, not a
trained model.

Run with:  uvicorn app:app --reload
"""

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from scipy import stats as scipy_stats

app = FastAPI(title="A/B Test Analysis Dashboard", version="1.0")


class ABTestInput(BaseModel):
    group_a_name: str = "Control"
    group_a_visitors: int
    group_a_conversions: int
    group_b_name: str = "Treatment"
    group_b_visitors: int
    group_b_conversions: int
    alpha: float = 0.05


class ABTestResult(BaseModel):
    group_a_rate: float
    group_b_rate: float
    absolute_difference: float
    relative_lift_pct: float
    z_statistic: float
    p_value: float
    confidence_interval_95pct: list[float]
    significant: bool
    srm_check_p_value: float
    srm_warning: bool
    plain_language_summary: str


def _run_ab_test(payload: ABTestInput) -> ABTestResult:
    n_a, x_a = payload.group_a_visitors, payload.group_a_conversions
    n_b, x_b = payload.group_b_visitors, payload.group_b_conversions

    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_stat = (p_a - p_b) / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci_low = (p_a - p_b) - 1.96 * se_diff
    ci_high = (p_a - p_b) + 1.96 * se_diff

    significant = p_value < payload.alpha
    relative_lift = (p_b - p_a) / p_a if p_a > 0 else 0.0

    # WHY THE SRM CHECK IS BUILT INTO THE SAME ENDPOINT, NOT A SEPARATE
    # TOOL: this project's own EDA showed the value of running an SRM
    # check BEFORE trusting a significance result — baking it into every
    # call means a user of this dashboard can never accidentally skip it.
    n_total = n_a + n_b
    chi2_srm, p_srm = scipy_stats.chisquare(f_obs=[n_a, n_b], f_exp=[n_total / 2, n_total / 2])
    srm_warning = p_srm < 0.01

    if srm_warning:
        summary = (
            f"WARNING - SAMPLE RATIO MISMATCH (p={p_srm:.4f}): group sizes deviate from the "
            f"expected 50/50 split more than chance would predict. Investigate your randomization "
            f"or logging pipeline before trusting the result below."
        )
    elif significant:
        direction = payload.group_a_name if p_a > p_b else payload.group_b_name
        summary = (
            f"Statistically significant at alpha={payload.alpha} (p={p_value:.4f}). "
            f"{direction} has the higher rate ({max(p_a,p_b):.2%} vs {min(p_a,p_b):.2%})."
        )
    else:
        summary = (
            f"NOT statistically significant at alpha={payload.alpha} (p={p_value:.4f}). "
            f"Cannot conclude the two groups differ based on this data."
        )

    return ABTestResult(
        group_a_rate=round(p_a, 4),
        group_b_rate=round(p_b, 4),
        absolute_difference=round(p_a - p_b, 4),
        relative_lift_pct=round(relative_lift * 100, 2),
        z_statistic=round(float(z_stat), 4),
        p_value=round(float(p_value), 6),
        confidence_interval_95pct=[round(float(ci_low), 4), round(float(ci_high), 4)],
        significant=bool(significant),
        srm_check_p_value=round(float(p_srm), 6),
        srm_warning=bool(srm_warning),
        plain_language_summary=summary,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=ABTestResult)
def analyze(payload: ABTestInput):
    return _run_ab_test(payload)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>A/B Test Analysis Dashboard</title></head>
    <body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
        <h2>A/B Test Analysis Dashboard</h2>
        <p>Paste in the summary counts from any two-group experiment. Pre-filled
           with this project's own Cookie Cats retention_7 result. Full docs at
           <a href="/docs">/docs</a>.</p>
        <form id="abForm">
            <label>Group A name: <input name="group_a_name" value="gate_30"></label><br><br>
            <label>Group A visitors: <input name="group_a_visitors" type="number" value="44699"></label><br><br>
            <label>Group A conversions: <input name="group_a_conversions" type="number" value="8501"></label><br><br>
            <label>Group B name: <input name="group_b_name" value="gate_40"></label><br><br>
            <label>Group B visitors: <input name="group_b_visitors" type="number" value="45489"></label><br><br>
            <label>Group B conversions: <input name="group_b_conversions" type="number" value="8279"></label><br><br>
            <button type="submit">Run analysis</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("abForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const form = new FormData(e.target);
            const payload = {
                group_a_name: form.get("group_a_name"),
                group_a_visitors: parseInt(form.get("group_a_visitors")),
                group_a_conversions: parseInt(form.get("group_a_conversions")),
                group_b_name: form.get("group_b_name"),
                group_b_visitors: parseInt(form.get("group_b_visitors")),
                group_b_conversions: parseInt(form.get("group_b_conversions"))
            };
            const res = await fetch("/analyze", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            document.getElementById("result").innerText = data.plain_language_summary +
                " | p=" + data.p_value + " | 95% CI on difference: [" +
                data.confidence_interval_95pct.join(", ") + "]";
        });
        </script>
    </body>
    </html>
    """
