# Learning Log

## Phase 0 — Discovery and planning

**Concepts introduced:**
- Interpolation estimates values *between* observed points; it is not a
  forecast and says nothing about the future.
- Treasury maturities are unevenly spaced (dense at the short end: 1mo,
  2mo, 3mo, 4mo, 6mo, 1yr; then 2yr, 3yr, 5yr, 7yr, 10yr; then a big jump to
  20yr, 30yr). Node spacing, not just node count, drives interpolation
  stability.

**Reasoning task:** given the maturity grid above, which is riskier to trust
from a single global Lagrange polynomial — the estimate at 4yr (between the
closely-spaced 3yr/5yr nodes) or at 15yr (in the wide 10yr–20yr gap)?

**Answer given (by the assistant — the user chose to move faster rather than
attempt this personally; open item to revisit in your own words):** the 15yr
estimate is riskier. A Lagrange polynomial's only constraint is passing
exactly through every node; where nodes are close together (4yr, between
3yr/5yr) that pins the curve's local shape tightly. Where nodes are far
apart (15yr, in the 10–20yr gap), nothing constrains the shape in between —
the single global polynomial (degree ~12, one less than the node count) is
free to swing however it needs to in order to satisfy all 13 point
constraints at once, which is exactly the setup for Runge-phenomenon-style
overshoot. This is the concrete motivation for also building linear/spline/
PCHIP methods, which only look at local neighbors and degrade gracefully in
sparse regions instead of oscillating.

**Open question to revisit:** try re-deriving this answer in your own words
before Phase 3 (where we derive Lagrange formally) — does your intuition
match, and can you say *why* in terms of the basis polynomials themselves,
not just "gaps are risky"?

**Small task (maturity-label → years mapping):** not attempted by the user;
answered by the assistant instead (user chose to keep moving). Correct
mapping and the key trap (months need `/12`, years don't — a naive parser
that just extracts the leading number would treat "4 Mo" as 4.0 years
instead of 0.33) recorded in the Phase 2 entry below. Open item to revisit:
write this mapping yourself from scratch before Phase 3.

## Phase 1 — Environment and repository setup

**Decisions made:** see `DECISIONS.md` for the full list (repo location,
Python version, data source, historical depth, accounts, requirements
split).

**Problems encountered:**

- First `import numpy.random` in the fresh `.venv` failed with `ImportError:
  DLL load failed ... An Application Control policy has blocked this file.`
  This is Windows' own security layer (not a Python/numpy bug) doing an
  async reputation scan the first time a brand-new native `.pyd`/DLL loads
  from a given location — it blocks that first load, then clears and caches
  the verdict. Confirmed by retrying the identical import immediately after:
  it succeeded, unchanged. Diagnosed by comparing three venvs (`%TEMP%`,
  inside the established `MATHEMATICAL` folder, inside the new
  `yield-curve-intelligence` folder) before concluding it was transient
  rather than a path-based block — worth remembering not to jump to a
  conclusion from the first data point alone.
  **Takeaway:** retry once before treating a fresh-install import error as
  a real bug on this machine.

## Phase 2 — Data acquisition and validation

**Concepts learned:**
- Par yield vs. zero-coupon/spot yield vs. forward rate are distinct
  concepts; this project interpolates observed **par yields** only.
- Real-world financial data schemas evolve — the Treasury maturity grid
  itself grew from 11 columns (2016) to 14 (2026) as new bill tenors were
  introduced. Code that hard-codes "the" set of columns breaks the moment
  the data source adds one. A generic label parser (number + unit) instead
  of a fixed lookup table is more robust to this.
- Separating "how to fetch" (`treasury_client.py`, raises on failure) from
  "how to be graceful when fetching fails" (`loader.py`, catches and falls
  back) makes each piece independently testable — no test needs to fake a
  network failure just to check the CSV-parsing logic, and no test needs
  real CSV content just to check the fallback triggers correctly.
- `pytest.mark.xfail` as a teaching device: a test for not-yet-implemented
  code can exist and run (rather than being commented out or absent),
  reporting XFAIL until the implementation is correct, at which point it
  flips to XPASS — a concrete, visible "you got it" signal.

**Decisions made:** see `DECISIONS.md` (live endpoint verified, evolving
schema, sample data bundling approach).

**`validate_maturity_ordering`:** left for the user; not attempted, so the
assistant implemented it. Caught a real subtlety while doing so: pandas'
`Series.is_monotonic_increasing` actually means *non-decreasing* (it
accepts equal neighbors), not *strictly* increasing — using it directly
would have silently accepted a table with a duplicate maturity on the same
date. Fixed by checking `(maturities.diff().dropna() > 0).all()` instead,
which requires every consecutive step to be strictly positive. Added a
regression test (`test_validate_maturity_ordering_rejects_equal_neighbors`)
specifically for that case. **Open item:** re-implement this function
yourself from the docstring alone (now cleaned of hints) before Phase 4,
and see if you land on the same fix without being told about the
`is_monotonic_increasing` trap first.

**Open questions to revisit:** should the bundled sample data (currently
2025–2026 only) be extended with a longer historical slice specifically
for the PCA/anomaly-detection work in Phase 6, given that phase wants
~10 years of live history? Deferred until Phase 6.
