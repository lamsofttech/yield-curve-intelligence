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

## Phase 3 — Lagrange interpolation from first principles

Derived the formula and the "switch" property of each basis polynomial
L_j(x), worked a real 3-node example by hand (1Yr/5Yr/10Yr -> estimate at
3Yr), verified the arithmetic by computation before presenting it, and
compared the real observed 3Yr yield (4.44%) against the 3-node estimate
(4.386%) as a concrete illustration of estimation error. Implemented the
classical direct-form algorithm from scratch; cross-checked against both
scipy.interpolate.BarycentricInterpolator and the (now-deprecated, scipy
itself flagged this live) scipy.interpolate.lagrange.

## Phase 4 — Linear, cubic spline, PCHIP + common interface

Added the other three methods as thin wrappers around scipy, behind a
shared `METHODS` registry (dict of name -> callable, all with the same
signature) so validation code doesn't need an if/elif per method.
Extracted validate_nodes/warn_if_extrapolating into base.py (previously
duplicated only inside lagrange.py -- caught and fixed before it could be
duplicated three more times).

Verified numerically before writing any test/claim about it:
- numpy.interp clamps rather than extrapolates beyond the node range
  (flat, not extended) -- different from the other three methods.
- Natural cubic spline overshoots to [-0.25, 1.25] on a step-like dataset
  bounded in [0, 1]; PCHIP stays exactly within [0, 1] on the same data.
  This is *the* reason PCHIP exists as a distinct method.
- On the real, full 11-node Treasury grid, Lagrange at 15Y gives
  approximately -19,627%, while linear/spline/PCHIP all agree near
  5.03-5.08%. Not a bug -- a real, dramatic demonstration of Runge
  phenomenon, far more extreme than the tame 3-node Phase 3 example.

**Open question to revisit:** given how extreme the full-grid Lagrange
divergence is, should the Interpolation Lab (Phase 7) default to showing
Lagrange fit on the *full* grid, or default to a smaller/nearby subset of
nodes around the target maturity? Showing the full-grid catastrophe is
more honest and more interesting pedagogically; defaulting to it might
also make Lagrange look unfairly useless if not explained well in the UI.
Decision deferred to Phase 7.

## Phase 5 — Leave-one-maturity-out validation

Key design decision: only *interior* maturities are "eligible" -- removing
an endpoint would make the held-out estimate an extrapolation, not an
interpolation, contaminating the measurement. On the real full grid,
removing the 20Y node (interior once 30Y anchors the endpoint) and
refitting Lagrange gives an error of about -1,228,248 percentage points --
even more extreme than Phase 4's -19,627% single-point example. PCHIP wins
clearly on this date (MAE 0.047, RMSE 0.068, max error 0.18), with linear
and cubic spline close behind -- real, evidence-based grounds to disqualify
Lagrange on the full grid and to prefer PCHIP *on this date*, while still
not over-generalizing from 9 interior points to "PCHIP is always best."

## Phase 7 — Gradio application

Assembled all 7 required sections into app.py, kept as thin UI wiring
only -- every computation (interpolation, validation, PCA, anomaly
detection, classification) lives in src/ and is independently tested;
app.py's job is presentation. Data (live latest curve + bundled 10-year
history) and the PCA/anomaly fits happen once at module load, not per
Gradio callback, since they don't depend on user input.

Verified live: the app actually served over HTTP (curl -> 200), and every
callback function was tested directly with real inputs before trusting
the UI wiring (interpolation lab, model validation, market overview).
Both the live 2026-09-09 curve (classified "Normal") and the historical
2023-10-19 curve (classified "Inverted") produce sensible results end to
end.

**Problem encountered:** a stale/orphaned background process from an
earlier session had bound port 7860 before (recorded from the portfolio
project) -- checked the port was free before launching this time, avoided
repeating that debugging detour.

## Phase 8 — Engineering quality

- pip-audit found real CVEs in pillow/starlette (transitive deps of
  gradio). Deliberately did not force a gradio major-version bump (5->6)
  under time pressure right before deployment -- deferred to Dependabot
  (configured now) for incremental, reviewable bumps instead. Documented
  the reasoning in DECISIONS.md rather than silently ignoring the finding.
- Secrets scan and hardcoded-path scan: clean (nothing in project code;
  all hits were inside .venv/ third-party libraries).
- Added structured logging (utils/logging_config.py) -- previously zero
  logging anywhere. Wired into loader.py's fallback path (now logs
  whether live or sample data was used, and why on fallback) and app.py's
  startup sequence (data loading, PCA fit, anomaly fit, each timestamped).
- Confirmed (again) no Docker available in this environment -- same
  constraint hit on the portfolio project. Dockerfile can be written and
  reviewed but not locally built; Phase 9's CI docker job is the first
  actual build verification.
