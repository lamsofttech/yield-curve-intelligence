---
title: Yield Curve Intelligence Platform
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# Yield Curve Intelligence Platform

Interactive U.S. Treasury yield curve analysis: Lagrange/linear/cubic-spline/PCHIP interpolation
compared honestly (including where Lagrange catastrophically diverges on the real grid),
leave-one-maturity-out validation, PCA factor analysis (level/slope/curvature, verified from real
data), and anomaly detection against ~10 years of history.

Full source, tests, and documentation: https://github.com/lamsofttech/yield-curve-intelligence

This Space's `requirements.txt` adds `torch`/`spaces` to the base app's dependencies (not present
in the GitHub repo's own `requirements.txt`, which stays lightweight for Docker/CI/local dev) --
this account's free Hugging Face tier requires ZeroGPU hardware for Gradio Spaces, and ZeroGPU
requires at least one genuine `@spaces.GPU`-decorated function to start. See the app's "GPU
Historical Stress Test" tab and `app.py`'s module docstring for what that function actually does
(a genuine, measured computation, not a no-op).

Educational/research use only -- see the Responsible Use tab in the app itself.
