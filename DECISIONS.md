# Decisions

Architectural and modeling decisions for the Yield Curve Intelligence Platform, in the order made.

## 2026-09-09 — Separate repository, not nested in the portfolio repo

New sibling repo at `Desktop/yield-curve-intelligence/`, not a subfolder of
`ai-ml-quantitative-research-portfolio`. The proposed structure is a full
standalone repo skeleton (own README/LICENSE/.gitignore/CI), and nesting it
would collide with the portfolio repo's own root files and git history.

## 2026-09-09 — Python 3.13, not 3.11

Spec allowed "3.11 or another justified stable version." Only 3.13.15 is
installed on this machine (no 3.11 available), and nothing planned in the
stack (numpy/pandas/scipy/scikit-learn/plotly/gradio/pytest/ruff) requires
3.11 specifically. Using 3.13.

## 2026-09-09 — Primary data source: Treasury.gov Daily Par Yield Curve Rates

Nominal par yields (not TIPS/real yields), with FRED as a documented fallback
if the Treasury.gov endpoint is unavailable. To be verified against the
actual live endpoint in Phase 2, not assumed.

## 2026-09-09 — Historical depth target: ~10 years live, smaller bundled sample

PCA and anomaly detection need a real multi-year series to be meaningful (a
handful of days can't estimate level/slope/curvature factors credibly). Live
fetches will target ~10 years; the bundled `data/sample/` fallback is trimmed
to a smaller recent slice to keep the repository light.

## 2026-09-09 — Same lamsofttech/nyakslamo accounts for GitHub/HF

Reusing the same accounts as the portfolio project, accepting known
constraints already hit there: GitHub Actions billing-locked, Hugging Face
free tier requires ZeroGPU (with a genuine `@spaces.GPU` function) since
plain CPU Gradio hosting needs a paid plan on this account. Revisit in
Phase 10 if these haven't been resolved.

## 2026-09-09 — Verified live Treasury.gov endpoint; maturity grid is not fixed

Confirmed real endpoint:
`https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv`
— one request per calendar year, CSV, most-recent-row-first, dates as
`MM/DD/YYYY`.

Column set is **not stable across years** — Treasury has added shorter
bill tenors over time (2 Mo in ~2020, 4 Mo in ~2022, "1.5 Month" in 2026).
Decision: the data layer parses maturity labels generically (extract
number + unit, months divided by 12) rather than using a fixed lookup
table, and preserves a variable/NaN-containing column set per year rather
than forcing one fixed schema. Downstream code (interpolation, PCA) must
each decide how to handle a maturity that's missing for a given date.

## 2026-09-09 — Lagrange: classical direct form, not barycentric

Implemented the direct O(n^2)-per-point formula (sum of y_j * basis
product), not the numerically superior barycentric reformulation, because
the goal is a visible, directly-derivation-traceable mechanism, not
production efficiency. Cross-checked against scipy's BarycentricInterpolator
(recommended) and the now-deprecated scipy.interpolate.lagrange (both
agree to 1e-8).

## 2026-09-09 — Linear/spline/PCHIP wrap trusted libraries, not hand-derived

Only Lagrange is implemented from scratch, per the project's own emphasis.
Cubic spline and PCHIP derivations (tridiagonal systems, shape-preserving
slope constraints) are substantial topics on their own; the pedagogical
value for these three is in comparing behavior/tradeoffs, not re-deriving
each. Shared validation/extrapolation-warning logic factored into
src/interpolation/base.py so all four methods behave consistently.

## 2026-09-09 — Empirical confirmation: Lagrange diverges catastrophically
## on the real, full 11-node grid

At 15Y (the 10Y-20Y gap), degree-10 Lagrange through all real nodes gives
approximately -19,627% -- confirming the Phase 0 reasoning task's
prediction, but far more extreme than the earlier 3-node worked example
suggested. Linear/spline/PCHIP all agree near 5.03-5.08% at the same
point. This is the core empirical result the Interpolation Lab (Phase 7)
and Model Validation (Phase 5) sections exist to make visible.

## 2026-09-09 — requirements.txt / requirements-dev.txt split

Runtime deps separate from dev-only tools (pytest, ruff), so the deployed
app's Docker image doesn't carry test/lint tooling it never uses.
