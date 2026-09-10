# Résumé bullets

- Built and deployed an interactive U.S. Treasury yield-curve analysis platform (Python, Gradio,
  Docker, GitHub Actions) comparing four interpolation methods; implemented Lagrange interpolation
  from its defining formula and cross-validated it against SciPy to 1e-8 precision.
- Designed a leave-one-maturity-out validation framework quantifying interpolation error on real
  Treasury data, revealing a six-order-of-magnitude error gap between global-polynomial and local
  interpolation methods on the real, unevenly-spaced maturity grid.
- Applied PCA to 10 years of daily Treasury yield changes, reproducing the classical
  level/slope/curvature factor structure (Litterman & Scheinkman, 1991) from real data, with
  factor labels assigned by inspecting fitted loadings rather than assumed from component order.
- Built an Isolation Forest anomaly-detection model on historical yield-curve shapes that
  independently flagged known market stress events (2020 COVID onset, 2023 Treasury yield spike)
  without being told what to look for.
- Implemented a GPU-batched (PyTorch) historical stress-test feature reducing a measured
  ~23-minute sequential computation to sub-second execution via a single vectorized matrix
  multiply; deployed to Hugging Face Spaces with end-to-end verification via the live Gradio API.
- Engineered a resilient data pipeline handling Treasury.gov's evolving CSV schema (11→14 columns
  since 2016) with graceful live-source fallback, 119+ automated tests, and full CI/CD.

## LinkedIn post draft

---

I just shipped a project I'm genuinely proud of: an interactive U.S. Treasury yield-curve
analysis platform, built end to end — math, ML, and deployment.

The starting question was simple: Treasury only publishes yields at specific maturities. If you
need an estimate in between, which method do you trust?

So I implemented four of them — including hand-deriving Lagrange interpolation from its defining
formula rather than calling a library — and then did the thing that actually mattered: measured
where they disagree.

The answer, on real data, was more dramatic than I expected. In the 10-to-20-year maturity gap, a
single global polynomial through all the real Treasury nodes doesn't just get a *little* wrong —
it diverges to **millions of percent**. Local methods (linear, cubic spline, PCHIP) stay sane in
the exact same region. Leave-one-out validation confirms it quantitatively: a six-order-of-
magnitude error gap.

On the ML side, I ran PCA on ten years of daily yield changes and reproduced the classic
level/slope/curvature factor structure from fixed-income research (Litterman & Scheinkman, 1991)
— without assuming it going in; I inspected the actual fitted loadings before labeling anything.
An Isolation Forest anomaly detector, given no hints, independently flagged the 2020 COVID onset
and the 2023 Treasury yield spike as the most unusual periods in the dataset.

It's deployed and live — and I mean genuinely verified live, not just "the code looks right": I
called every interactive endpoint through the real API against the running deployment before
writing a single word claiming it worked.

Live app: https://huggingface.co/spaces/nyakslamo/yield-curve-intelligence
Source: https://github.com/lamsofttech/yield-curve-intelligence

This is a research and education tool, not investment advice — that distinction matters to me, and
it's stated clearly in the app itself.

#QuantitativeFinance #MachineLearning #Python #DataScience

---
