"""Yield Curve Intelligence Platform -- Gradio application.

Thin UI wiring only: every quantitative computation lives in src/ and is
independently tested (see tests/). This file's job is presentation, not
math -- per the project's "keep UI code separate from business logic"
testing requirement.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from data.loader import load_curve_data, load_historical  # noqa: E402
from features.yield_changes import daily_changes, to_wide_panel  # noqa: E402
from interpolation import METHODS  # noqa: E402
from models.anomaly import (  # noqa: E402
    current_percentile,
    fit_anomaly_detector,
    most_unusual_dates,
)
from models.loo_validation import rank_methods, validate_all_methods  # noqa: E402
from models.pca_analysis import fit_pca, label_components  # noqa: E402
from utils.curve_classification import classify_curve, compute_spreads  # noqa: E402
from visualization.curve_plots import (  # noqa: E402
    plot_anomaly_scores,
    plot_loo_errors,
    plot_method_comparison,
    plot_pca_loadings,
    plot_pca_scores,
)

# ---------------------------------------------------------------------------
# Data loaded once at startup. Historical data is bundled (fast, ~1s);
# the latest curve is fetched live with graceful fallback (see loader.py).
# PCA and anomaly detection are fit once here, not per Gradio callback --
# they depend on fixed history, not on anything the user selects.
# ---------------------------------------------------------------------------

_CURRENT_YEAR = date.today().year
_latest = load_curve_data(_CURRENT_YEAR - 1, _CURRENT_YEAR)
_historical_tidy = load_historical()
_panel = to_wide_panel(_historical_tidy)
_changes = daily_changes(_panel)
_pca_result = fit_pca(_changes, n_components=3)
_pca_labels = label_components(_pca_result.loadings)
_anomaly_result = fit_anomaly_detector(_panel)

# Dates offered in the Interpolation Lab / Model Validation tabs: the most
# recent 60 available trading days, plus two dates independently confirmed
# unusual by the anomaly detector (Phase 6) -- included deliberately so a
# user can see what an inverted (2023) and panic-onset (2020) curve looks
# like, not just recent normal ones.
_recent_dates = sorted(_panel.index[-60:], reverse=True)
_notable_dates = [pd.Timestamp("2023-10-19"), pd.Timestamp("2020-03-06")]
_available_dates = sorted(set(_recent_dates) | set(_notable_dates), reverse=True)
_date_choices = [d.strftime("%Y-%m-%d") for d in _available_dates]


def _curve_for_date(date_str: str) -> tuple[pd.Series, pd.Timestamp]:
    ts = pd.Timestamp(date_str)
    if ts in _panel.index:
        return _panel.loc[ts].dropna(), ts
    # Fall back to the live/latest curve's own tidy table if not in the
    # bundled historical panel (shouldn't happen given _date_choices, but
    # defensive rather than a hard crash on an unexpected input).
    latest_curve = _latest.table[_latest.table["date"] == _latest.table["date"].max()]
    return latest_curve.set_index("maturity_years")["yield_pct"], _latest.as_of


# ---------------------------------------------------------------------------
# Section 1: Market Overview
# ---------------------------------------------------------------------------


def market_overview_content() -> str:
    curve = _latest.table[_latest.table["date"] == _latest.as_of]
    curve_dict = curve.set_index("maturity_years")["yield_pct"].to_dict()
    spreads = compute_spreads(curve_dict)
    classification = classify_curve(*spreads)

    source_note = (
        f"⚠️ Live data unavailable ({_latest.error}); showing bundled sample data instead."
        if _latest.source == "sample"
        else "Live data from Treasury.gov."
    )

    maturities_str = ", ".join(
        f"{row.maturity_label} = {row.yield_pct:.2f}%"
        for row in curve.sort_values("maturity_years").itertuples()
    )

    return f"""
### Latest available Treasury yield curve

**Observation date:** {_latest.as_of.strftime("%Y-%m-%d")}  ({source_note})

**Available maturities:** {maturities_str}

**Curve classification: {classification}**

Rule: *Normal* if both the 10Y-2Y and 10Y-3M spreads exceed +0.10 percentage points;
*Inverted* if both are below -0.10; *Flat* if both are within +/-0.10 of zero;
*Mixed* if the two spreads disagree.

**Key spreads:** 10Y-2Y = {spreads[0]:+.2f} pp,  10Y-3M = {spreads[1]:+.2f} pp
"""


# ---------------------------------------------------------------------------
# Section 2: Interpolation Laboratory
# ---------------------------------------------------------------------------


def run_interpolation_lab(date_str: str, target_maturity: float):
    curve, _ = _curve_for_date(date_str)
    x_nodes = curve.index.to_numpy(dtype=float)
    y_nodes = curve.to_numpy(dtype=float)

    rows = []
    for name, method in METHODS.items():
        try:
            estimate = method(x_nodes, y_nodes, float(target_maturity))
        except ValueError as exc:
            rows.append({"Method": name, "Estimated yield (%)": None, "Note": str(exc)})
            continue
        rows.append({"Method": name, "Estimated yield (%)": round(estimate, 4), "Note": ""})
    table = pd.DataFrame(rows)

    fig = plot_method_comparison(x_nodes, y_nodes, target_maturity=float(target_maturity))
    return table, fig


LAGRANGE_EXPLANATION = r"""
### The Lagrange formula

$$P(x) = \sum_{j=0}^{n} y_j \, L_j(x), \qquad L_j(x) = \prod_{k \ne j} \frac{x - x_k}{x_j - x_k}$$

**The "switch" idea:** each basis polynomial $L_j(x)$ equals **1** at its own node and **0** at
every other node -- like a switch that's "on" only for its own point. Summing $y_j L_j(x)$ over
all nodes therefore reproduces every observed $y_j$ exactly at its own node, and blends smoothly
between them elsewhere. See `docs/lagrange_derivation.md` for a full worked example.

**Warning demonstrated by this lab:** on the real, full Treasury grid, Lagrange can diverge by
*thousands of percentage points* in the wide 10Y-20Y gap (verified: `DECISIONS.md`). This is not
a bug -- it is what a single global polynomial genuinely does with unevenly spaced nodes. Compare
it against Linear/Cubic Spline/PCHIP in the chart above.
"""


# ---------------------------------------------------------------------------
# Section 3: Model Validation
# ---------------------------------------------------------------------------


def run_model_validation(date_str: str):
    curve, _ = _curve_for_date(date_str)
    x_nodes = curve.index.to_numpy(dtype=float)
    y_nodes = curve.to_numpy(dtype=float)

    results = validate_all_methods(x_nodes, y_nodes)
    ranked = rank_methods(results, by="rmse")

    table = pd.DataFrame(
        [
            {
                "Method": r.method,
                "MAE": round(r.mae, 4),
                "RMSE": round(r.rmse, 4),
                "Max abs error": round(r.max_abs_error, 4),
            }
            for r in ranked
        ]
    )
    fig = plot_loo_errors(results)
    best = ranked[0].method
    summary = (
        f"**Best method on this date (lowest RMSE): {best}.** "
        f"Based on only {len(x_nodes) - 2} interior maturities -- a small sample from a single "
        f"day's curve. This ranks methods *on this date*, not in general; see LEARNING_LOG.md "
        f"for why the same ranking should not be assumed to hold on every date."
    )
    return table, fig, summary


# ---------------------------------------------------------------------------
# Section 4: Yield-Curve Dynamics (PCA)
# ---------------------------------------------------------------------------

_pca_columns = _pca_result.loadings.columns
_pca_ratios = _pca_result.explained_variance_ratio
_pca_variance_md = "\n".join(
    f"- **{name}** ({_pca_labels[name]}): {ratio:.1%} of variance"
    for name, ratio in zip(_pca_columns, _pca_ratios, strict=True)
)
_panel_start = _panel.index.min().strftime("%Y-%m-%d")
_panel_end = _panel.index.max().strftime("%Y-%m-%d")

PCA_CONTENT = f"""
### PCA on daily yield changes, {_panel_start} to {_panel_end}

{_pca_variance_md}

Together, the first three components explain **{sum(_pca_result.explained_variance_ratio):.1%}**
of daily yield-curve variation.

Labels above were assigned from the **shape of the fitted loadings**, not assumed from component
order: a component is called "Level" only if its loadings are the same sign across every
maturity, "Slope" only if they change sign exactly once (short vs. long), and "Curvature" only
if they change sign twice (a hump/U-shape). See `models/pca_analysis.describe_loadings`.
Component *sign* is arbitrary -- a component and its negation represent the identical factor.

PCA here runs on **raw, unscaled** yield changes (not standardized), matching standard
fixed-income practice (Litterman & Scheinkman, 1991): scaling would artificially equalize the
naturally higher day-to-day volatility of short-end yields against the long end, distorting the
level factor's interpretation.
"""


# ---------------------------------------------------------------------------
# Section 5: Anomaly Analysis
# ---------------------------------------------------------------------------

_current_pct = current_percentile(_anomaly_result, _panel.index.max())
_top_unusual = most_unusual_dates(_anomaly_result, n=15)
_panel_start_year = _panel.index.min().strftime("%Y")
_panel_end_year = _panel.index.max().strftime("%Y")

ANOMALY_CONTENT = f"""
### Historical anomaly analysis (curve *shape*, {_panel_start_year}-{_panel_end_year})

**Today's curve ({_panel_end}) ranks in the {_current_pct:.1f}th
percentile of historical unusualness** (0 = most unusual ever observed in this window, 100 =
most typical).

Isolation Forest fit on standardized daily yield **levels** (not changes -- a separate question
from the PCA tab above) across {len(_panel.columns)} maturities and {len(_panel)} trading days.

> **A statistical anomaly is not automatically a crisis, and not a trading signal.** It only
> means this curve shape is rare relative to the last ~10 years of history. See the
> Responsible-Use tab.
"""


# ---------------------------------------------------------------------------
# Section 6: Educational Panel
# ---------------------------------------------------------------------------

EDUCATIONAL_CONTENT = """
### What is a yield curve?

A plot of interest rates (yields) on comparable debt (here, U.S. Treasuries) across different
maturities on a single date. Its shape reflects market expectations about future interest rates,
growth, and inflation.

### What does interpolation mean?

Estimating a value *between* two or more observed data points. It says nothing about values
*outside* the observed range (see "Interpolation vs. extrapolation" below), and nothing about
*future* dates (see "Interpolation vs. forecasting" below).

### How does Lagrange interpolation work?

It builds the one polynomial that passes exactly through every observed point, using basis
polynomials that act as "switches" -- see the Interpolation Lab tab and
`docs/lagrange_derivation.md`.

### Interpolation vs. extrapolation

Interpolation estimates a value *inside* the range of observed maturities (e.g., 4Y when you
have 3Y and 5Y). Extrapolation estimates a value *outside* that range (e.g., 40Y when your
longest observed maturity is 30Y) and is far less reliable -- this app warns explicitly whenever
a request would require extrapolation.

### Interpolation vs. forecasting

Interpolation never uses time. It estimates an unobserved *maturity* on a date that already
happened, using data *from that same date*. It makes no claim whatsoever about what any yield
will be tomorrow, next week, or next year. Forecasting is a distinct problem this app does not
attempt.

### Why can Lagrange become unstable?

A single polynomial through many unevenly-spaced points has only one job: pass through every
point exactly. Where points are far apart (e.g., the real 10Y-20Y gap), nothing constrains the
curve's shape in between, and a high-degree polynomial can swing wildly to satisfy every
constraint at once (Runge's phenomenon) -- demonstrated with real numbers in this app's
Interpolation Lab and Model Validation tabs.

### How does PCA connect mathematics with machine learning?

PCA finds the directions (linear combinations of the original variables) along which the data
varies the most. Applied to daily yield changes, the first few directions turn out to correspond
to interpretable, economically meaningful patterns (level, slope, curvature) -- an example of an
unsupervised ML technique recovering real structure, not just abstract math.

### How could this support risk analysis?

Level/slope/curvature factors are a standard way fixed-income risk systems decompose interest
rate risk (e.g., duration/key-rate-duration hedging). Anomaly detection offers a systematic,
reproducible way to flag unusual market days for further (human) review -- a starting point for
investigation, not a conclusion.
"""


# ---------------------------------------------------------------------------
# Section 7: Responsible-Use Statement
# ---------------------------------------------------------------------------

RESPONSIBLE_USE_CONTENT = """
### Responsible use

- **This application is for education and research.** It is not investment advice.
- **Estimates have model risk.** Every interpolation method here embeds assumptions (e.g.,
  smoothness, shape-preservation) that may not hold on any given date.
- **Market data may be delayed or revised.** Treasury.gov publishes end-of-day data; this is not
  real-time.
- **Anomaly scores are not trading signals.** A statistically unusual curve shape is not a
  prediction, a recommendation, or evidence of an impending market event.
- **Interpolation is not forecasting.** Nothing in this application estimates a future yield.
"""


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------

with gr.Blocks(title="Yield Curve Intelligence Platform") as demo:  # noqa: SIM117
    gr.Markdown("# Yield Curve Intelligence Platform")
    gr.Markdown(
        "Multi-method interpolation, leave-one-out validation, PCA factor analysis, and "
        "anomaly detection on the U.S. Treasury yield curve. Educational/research use only "
        "-- see the Responsible Use tab."
    )

    with gr.Tabs():
        with gr.Tab("Market Overview"):
            gr.Markdown(market_overview_content())

        with gr.Tab("Interpolation Laboratory"):
            gr.Markdown(
                "Select an observation date and target maturity to compare all four methods."
            )
            with gr.Row():
                lab_date = gr.Dropdown(
                    _date_choices, label="Observation date", value=_date_choices[0]
                )
                lab_maturity = gr.Slider(
                    label="Target maturity (years)",
                    minimum=0.25,
                    maximum=30.0,
                    value=4.0,
                    step=0.25,
                )
                lab_button = gr.Button("Estimate", variant="primary")
            lab_table = gr.Dataframe(label="Estimated yield by method")
            lab_plot = gr.Plot(label="Method comparison")
            gr.Markdown(LAGRANGE_EXPLANATION)
            lab_inputs = [lab_date, lab_maturity]
            lab_outputs = [lab_table, lab_plot]
            lab_button.click(fn=run_interpolation_lab, inputs=lab_inputs, outputs=lab_outputs)
            demo.load(fn=run_interpolation_lab, inputs=lab_inputs, outputs=lab_outputs)

        with gr.Tab("Model Validation"):
            gr.Markdown(
                "Leave-one-maturity-out validation: each interior maturity is removed, "
                "re-estimated from the rest, and compared to its real observed value."
            )
            val_date = gr.Dropdown(_date_choices, label="Observation date", value=_date_choices[0])
            val_button = gr.Button("Run validation", variant="primary")
            val_table = gr.Dataframe(label="Error metrics by method")
            val_plot = gr.Plot(label="Mean absolute error by method (log scale)")
            val_summary = gr.Markdown()
            val_outputs = [val_table, val_plot, val_summary]
            val_button.click(fn=run_model_validation, inputs=[val_date], outputs=val_outputs)
            demo.load(fn=run_model_validation, inputs=[val_date], outputs=val_outputs)

        with gr.Tab("Yield-Curve Dynamics"):
            gr.Markdown(PCA_CONTENT)
            gr.Plot(value=plot_pca_loadings(_pca_result.loadings), label="Loadings by maturity")
            gr.Plot(value=plot_pca_scores(_pca_result.scores), label="Factor scores over time")

        with gr.Tab("Anomaly Analysis"):
            gr.Markdown(ANOMALY_CONTENT)
            gr.Plot(
                value=plot_anomaly_scores(_anomaly_result.scores, _anomaly_result.is_anomaly),
                label="Anomaly score over time",
            )
            _unusual_columns = {
                "index": "date",
                "anomaly_score": "Anomaly score",
                "is_anomaly": "Flagged",
            }
            gr.Dataframe(
                value=_top_unusual.reset_index().rename(columns=_unusual_columns),
                label="15 most historically unusual dates",
            )

        with gr.Tab("Educational Panel"):
            gr.Markdown(EDUCATIONAL_CONTENT)

        with gr.Tab("Responsible Use"):
            gr.Markdown(RESPONSIBLE_USE_CONTENT)

if __name__ == "__main__":
    demo.launch()
