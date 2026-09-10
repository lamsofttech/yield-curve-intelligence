# Demo script (~4 minutes)

For walking someone through the live app: https://huggingface.co/spaces/nyakslamo/yield-curve-intelligence

## 1. Market Overview (30s)

"This pulls the latest Treasury par yield curve live, classifies its shape — normal, inverted,
flat, or mixed — using an explicit, stated rule on the 10Y-2Y and 10Y-3M spreads, and shows the
data source and 'as of' date so it's never ambiguous whether you're looking at live or fallback
data."

## 2. Interpolation Laboratory (60s)

"Treasury only publishes yields at specific maturities. Say I need an estimate at 15 years, which
sits in a 10-year gap between the 10Y and 20Y points. Watch what happens." *(select 15Y target)*

"Lagrange — the classical method, which I implemented from scratch, not from a library — gives an
estimate in the **millions of percent**. That's not a bug. It's Runge's phenomenon: a single
polynomial forced through all 11 real, unevenly-spaced nodes has nothing constraining its shape in
that wide gap, so it swings wildly. Linear, cubic spline, and PCHIP all stay sane, within a
fraction of a point of each other."

"This is the whole point of the project: don't trust one interpolation method blindly — measure
and compare."

## 3. Model Validation (45s)

"This backs that up quantitatively: leave-one-maturity-out validation, removing each interior
point, re-estimating it from the rest, and comparing to the real value. Lagrange's error here is
six orders of magnitude larger than the other three methods. PCHIP wins on this date — with the
honest caveat, stated right in the app, that nine data points from one day isn't strong evidence
it wins on every date."

## 4. Yield-Curve Dynamics / PCA (45s)

"This is the ML side: PCA on ten years of daily yield changes. Three components explain over 92%
of the variation, and — importantly — I didn't just assume which component means what. I inspected
the actual loadings. The first is same-signed across every maturity, a parallel shift: level. The
second flips sign from short to long: slope. The third has a hump shape: curvature. That's the
classic Litterman-Scheinkman result from 1991, reproduced from real data."

## 5. Anomaly Analysis (30s)

"Isolation Forest flags the most unusual curve shapes in the last decade. It independently found
March 2020 — COVID — and October 2023 — the real Treasury yield spike to almost 5%, without me
telling it what to look for. Today's curve, incidentally, ranks in the bottom 2% of typicality —
worth a look, not a trading signal."

## 6. GPU Historical Stress Test (30s, if asked about deployment/engineering)

"This account's free Hugging Face tier requires a GPU allocation to even start the Space, so
rather than fake that requirement, I built something real: the Lagrange-vs-linear divergence,
computed across all 2,671 days in the dataset simultaneously, as a single batched matrix multiply.
I measured this honestly — the real speedup is from vectorization, not the GPU itself, at this
problem size. I'd rather say that than oversell it."

## Closing line

"Everything here — the math, the code, the deployment — is in the repo, including a running log
of every decision I made and why, and the problems I actually hit along the way."
