#!/usr/bin/env python3
"""Shared Beta-Binomial utilities for BSW analysis."""

import numpy as np
from scipy.special import betaln


def estimate_rho(pred, g):
    """Estimate overdispersion rho from CV residuals."""
    p = np.clip(
        pred["BSW_pred"].values/100, 1e-6, 1-1e-6)
    r2 = (pred["BSW_resid"].values/100)**2
    n = np.maximum(g, 1); obs = np.mean(r2)
    base = np.mean(p*(1-p)/n)
    ext = np.mean(p*(1-p)*(n-1)/n)
    return min(max((obs-base)/ext, 1e-6), 0.5)


def bb_p0(n, p, rho):
    """Beta-Binomial P(X=0) via log-Beta fn."""
    phi = max(1/rho-1, 1e-6)
    a = np.maximum(p*phi, 1e-10)
    b = np.maximum((1-p)*phi, 1e-10)
    lp = betaln(a, b+n) - betaln(a, b)
    return np.exp(np.clip(lp, -700, 0))
