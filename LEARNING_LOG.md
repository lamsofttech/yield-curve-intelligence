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
