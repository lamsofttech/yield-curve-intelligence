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

## 2026-09-09 — Historical dataset bundled/committed, not fetched live at startup

data/historical/treasury_2016_2026.csv (real, ~920KB, 32,715 tidy rows)
is committed, distinct from data/sample/ (small offline fallback for the
single-curve views). Reason: a live 11-year fetch takes ~3.5 minutes
(measured, not estimated) -- unacceptable as an app cold-start delay, and
PCA/anomaly detection are backward-looking analyses that don't need
data from the last few minutes to be meaningful. Market Overview and the
Interpolation Lab still use the live loader with graceful fallback.

## 2026-09-09 — Real PCA result: reproduces Litterman-Scheinkman ordering

On 2016-09-09's cut of history (2016-01-04 to 2026-09-08, 2,670 daily
changes, 11 common maturities): PC1 explains 72.6% of variance (same sign
across all maturities -> Level), PC2 10.6% (one sign change, short vs
long -> Slope), PC3 9.1% (two sign changes, U-shape -> Curvature).
Together, 92.25%. This matches the classic finding (Litterman &
Scheinkman, 1991) both in ordering and in "three factors explain the
overwhelming majority of variance" -- reproduced from real data, not
assumed. label_components() assigns labels from the observed loading
shape, never from component order or a hardcoded assumption.

## 2026-09-09 — Real anomaly detection surfaces genuine historical events

Isolation Forest on standardized daily yield levels (2016-2026, 11
maturities) independently flags October 2023 (the real 10Y Treasury yield
spike to ~5%, highest since 2007) and early March 2020 (COVID market
panic onset) as among the most unusual dates in the entire window -- both
well-known genuine market events, not arbitrary. The current date
(2026-09-08) ranks in the 1.3rd percentile of historical unusualness --
a striking real result, surfaced in the app with the required "not a
trading signal" caveat front and center.

## 2026-09-09 — Known transitive-dependency vulnerabilities, deferred to Dependabot

pip-audit flags CVEs in pillow (11.3.0) and starlette (0.52.1) -- both
transitive dependencies pulled in by gradio 5.50.0, not pinned directly.
A newer gradio (6.x) is available but is a major version bump with
untested API changes; blindly upgrading right before deployment risks
breaking the already-verified, tested app for marginal security benefit
on a no-auth, no-user-data educational app. Deferred to Dependabot
(configured in .github/dependabot.yml) to propose incremental, reviewable
version bumps rather than a risky big-bang upgrade under time pressure --
same judgment call made on the ai-ml-quantitative-research-portfolio repo.

## 2026-09-10 — GPU feature: batched historical stress test, not a no-op

Same ZeroGPU constraint hit on the portfolio project applies here too
(this account's free Space tier requires it; plain CPU hosting needs
PRO). Added a genuine feature rather than a fake @spaces.GPU function:
Lagrange vs. linear interpolation evaluated on a dense grid across the
*entire* bundled history at once. Because every day shares the same
maturity grid, this reduces to one (days x nodes) @ (nodes x grid) matrix
multiply -- verified exactly matching the trusted per-day implementation
before trusting it (see LEARNING_LOG.md).

Measured, not assumed: the naive per-day Python loop (our existing
implementation) would take ~23 minutes for the full 2,671-day x
2,000-point computation; the vectorized batch form takes ~0.24s even on
CPU. Documented honestly in the app itself that the real speedup is from
*vectorization*, not GPU hardware specifically -- this problem is small
enough that GPU's marginal benefit over a already-vectorized CPU
computation is modest. The tab exists to give the mandatory ZeroGPU
allocation genuine work, not to claim GPU was strictly necessary.

torch/spaces are NOT added to the base requirements.txt (kept out of
Docker/CI/local dev, which don't need them) -- app.py imports them in a
try/except and only shows the GPU tab if available, so the same single
app.py file serves both the lightweight base deployment and the HF Space
deployment (which will get its own requirements.txt with torch/spaces
added) without maintaining two copies of the UI wiring.

## 2026-09-10 — GitHub push succeeded; Actions blocked by known account billing lock

Push succeeded once the user granted the `workflow` OAuth scope. CI then
failed in ~2s on every job with "your account is locked due to a billing
issue" -- the same lamsofttech account constraint already documented from
the portfolio project, not a bug in this repo's config. Needs the user to
resolve at github.com/settings/billing; not something fixable here.
Deployment continues to the HF Space, which is unaffected (separate
account/system).
