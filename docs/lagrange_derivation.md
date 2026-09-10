# Lagrange interpolation: derivation and worked example

## The formula

Given $n+1$ nodes $(x_0,y_0), \dots, (x_n,y_n)$ with distinct $x_j$, the Lagrange
interpolating polynomial is:

$$P(x) = \sum_{j=0}^{n} y_j \, L_j(x), \qquad L_j(x) = \prod_{k \ne j} \frac{x - x_k}{x_j - x_k}$$

## The "switch" idea

Each basis polynomial $L_j(x)$ is built so that:

- At $x = x_j$ (its own node), every factor becomes $\frac{x_j-x_k}{x_j-x_k}=1$, so $L_j(x_j) = 1$.
- At $x = x_k$ for *any other* node, the numerator contains a factor $(x_k - x_k) = 0$, so $L_j(x_k) = 0$.

So $L_j$ is "on" (=1) at its own node and "off" (=0) at every other node — like a
switch. When you sum $y_j L_j(x)$ over all $j$ and evaluate at some observed
node $x_i$, every term except $y_i L_i(x_i) = y_i \cdot 1$ vanishes. That is the
entire mechanism that forces the polynomial through every observed point exactly.

## Worked example (real data, 2026-09-08)

Treasury observed 1Yr = 4.15%, 5Yr = 4.57%, 10Yr = 4.80% on 2026-09-08.
Estimate the 3-year yield using only these three nodes:

$$L_0(3)=\frac{(3-5)(3-10)}{(1-5)(1-10)}=0.3889,\quad
L_1(3)=\frac{(3-1)(3-10)}{(5-1)(5-10)}=0.7,\quad
L_2(3)=\frac{(3-1)(3-5)}{(10-1)(10-5)}=-0.0889$$

(Check: these three sum to exactly 1.0 — required, since $L_j$ are a partition of unity.)

$$P(3) = 4.15(0.3889) + 4.57(0.7) + 4.80(-0.0889) = 4.386\%$$

Treasury actually quotes a 3Yr yield directly too: **4.44%** that day. The
3-node estimate (4.386%) is close but not exact — real estimation error,
not an abstract textbook error term. It differs because the 2Yr node
(which would pull the local curve shape) was excluded from this small
example.

## Why the full grid is more dangerous than this example

The example above uses only 3 well-chosen nodes. The real Treasury curve
has 11+ nodes, unevenly spaced (dense at the short end, then a 10-year gap
between the 10Y and 20Y nodes). A single global polynomial through all of
them is degree 10 — free to swing however necessary to satisfy every
constraint simultaneously, and empirically (see `tests/interpolation/
test_method_comparison.py` and `tests/models/test_loo_validation.py`) it
does: on the real, full grid, Lagrange's estimate at 15Y is approximately
**-19,627%**, and its leave-one-out error after removing the 20Y node is
approximately **-1,228,248 percentage points**. Linear, cubic spline, and
PCHIP all remain within a fraction of a percentage point in the same
region. This is Runge's phenomenon, demonstrated on real financial data
rather than a synthetic textbook function.

## Why the classical (not barycentric) form was implemented here

`src/interpolation/lagrange.py` implements the direct/classical formula
above — evaluating each basis polynomial by its product definition — which
is $O(n^2)$ per evaluation point. Production numerical libraries (and
`scipy.interpolate.BarycentricInterpolator`) use a numerically superior
"barycentric" reformulation instead. The classical form was chosen here
deliberately: the goal of this module is that the mechanism stays visible
and directly traceable to the formula above, not maximum efficiency.
`tests/interpolation/test_lagrange.py` cross-checks this implementation
against both `BarycentricInterpolator` and the (now scipy-deprecated)
`scipy.interpolate.lagrange`, agreeing to within `1e-8`.
