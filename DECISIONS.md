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

## 2026-09-09 — requirements.txt / requirements-dev.txt split

Runtime deps separate from dev-only tools (pytest, ruff), so the deployed
app's Docker image doesn't carry test/lint tooling it never uses.
