# Yield Curve Intelligence Platform

An interactive tool for analyzing the U.S. Treasury yield curve: multi-method
interpolation (Lagrange, linear, cubic spline, PCHIP) at any maturity,
leave-one-maturity-out validation of those methods, PCA-based level/slope/
curvature analysis of daily yield changes, and anomaly detection against
historical curve shapes.

**Status:** early development (Phase 1 of a planned build — see
`DECISIONS.md` and `LEARNING_LOG.md`). Not yet deployed.

## What this is not

- Not a forecast: interpolation estimates yields *between* observed
  maturities, never in the future.
- Not investment advice or a trading signal generator.
- Not a full term-structure model (v1 works with observed par yields only —
  zero-coupon bootstrapping, discount factors, and forward rates are
  distinct, more advanced concepts this version does not implement).

## Repository layout

```text
src/
  data/            Fetching, caching, validation of Treasury yield data
  interpolation/   Lagrange, linear, cubic spline, PCHIP (common interface)
  features/        Daily yield-change feature engineering
  models/          PCA, anomaly detection, leave-one-out validation
  visualization/   Plotly figure builders (UI-agnostic)
  utils/           Shared helpers
tests/             Mirrors src/
data/sample/       Bundled offline fallback dataset
notebooks/         Exploratory analysis (not production code)
docs/              Methodology notes
```

## Local setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
python -m pip install -r requirements-dev.txt
pytest
```

## License

MIT — see `LICENSE`.
