#!/usr/bin/env python3
"""Single generative model for BSW undercounting.
Replaces additive mechanism stacking with a latent
variable model that prevents double-counting."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from wahlbezirk_lr import (load_2025_wbz, LAND_CODE,
                            validate_totals)

DATA = Path("data")
SEP = "=" * 60
BD = "BÜNDNIS DEUTSCHLAND"
DEFICIT = 9_529
N_SIMS = 50_000


def load_all():
    """Load raw 2025 data + LR predictions."""
    print("Loading data...")
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen",
              "Wahlberechtigte (A)",
              "Wählende (B)", "Bezirksart"]:
        df[c] = pd.to_numeric(
            df[c], errors="coerce").fillna(0)
    df = df[df["Gültige - Zweitstimmen"] >= 1].copy()
    df = df.reset_index(drop=True)
    pred = pd.read_csv(
        DATA / "wahlbezirk_lr_predictions.csv",
        low_memory=False)
    assert len(df) == len(pred)
    validate_totals(df)
    print(f"  {len(df)} precincts loaded")
    return df, pred


def _v(df, party):
    c = f"{party} - Zweitstimmen"
    return pd.to_numeric(df[c], errors="coerce").fillna(0)


def _g(df):
    return df["Gültige - Zweitstimmen"].values.astype(float)


def estimate_error_rates(df, pred):
    """Stage 1: Estimate error channel rates from data.

    Channels:
    - swap: BSW votes misattributed to BD
    - zeroout: entire precinct BSW zeroed (data entry)

    Calibration points:
    - Delmenhorst recount: +24 BSW in one precinct
    - 50 recounts: +15 total (0.3/precinct)
    - Observed corrections: +4,277 prelim→final
    """
    print(f"\n{SEP}")
    print("STAGE 1: Estimate error rates")
    print(SEP)
    g = _g(df)
    bsw = _v(df, "BSW").values
    bd = _v(df, BD).values
    bp = pred["BSW_pred"].values / 100
    n = len(df)

    # Zero-out rate: fraction of precincts where
    # BSW=0 but expected > threshold
    lam = np.maximum(bp * g, 1e-6)
    p0 = np.exp(-lam)
    susp_zero = (bsw == 0) & (p0 < 0.01)
    # Expected zeros from Poisson
    exp_zeros = p0.sum()
    obs_zeros = (bsw == 0).sum()
    excess = max(obs_zeros - exp_zeros, 0)
    pi_zero = excess / n  # rate of "zeroout" errors
    # Average votes lost per zeroout
    mu_zero = lam[susp_zero].mean() if susp_zero.any() \
        else 0
    print(f"  Zero-out: pi={pi_zero:.5f}"
          f" ({excess:.0f}/{n})"
          f" mu={mu_zero:.1f} votes/event")

    # Swap rate from recount data:
    # 50 recounts → +15 BSW (0.3/precinct)
    # This is a per-precinct misattribution rate
    # Gamma posterior: shape=15, rate=50
    swap_shape = 15  # observed gains
    swap_rate_param = 50  # number of recounts
    theta_swap = swap_shape / swap_rate_param
    print(f"  Swap: θ={theta_swap:.3f} votes/precinct"
          f" (from recount data)")

    return dict(pi_zero=pi_zero, mu_zero=mu_zero,
                swap_shape=swap_shape,
                swap_rate=swap_rate_param,
                n=n)


def mc_posterior(params):
    """Stage 2: MC for total missing votes.
    swap_i ~ Poisson(θ), zeroout_i ~ Bern(π)×Pois(μ)
    Total = Σ(swap_i + zeroout_i)"""
    rng = np.random.RandomState(42)
    n=params["n"]; pi_z=params["pi_zero"]
    mu_z=params["mu_zero"]
    s_a=params["swap_shape"]; s_r=params["swap_rate"]
    totals = np.empty(N_SIMS)
    for i in range(N_SIMS):
        theta = rng.gamma(s_a, 1/s_r)
        swap = rng.poisson(theta * n)
        n_z = rng.binomial(n, pi_z)
        zero_v = rng.poisson(mu_z, n_z).sum()
        totals[i] = swap + zero_v
    return totals


def mc_conservative(params):
    """Conservative: only zero-out channel, no swap."""
    rng = np.random.RandomState(42)
    n=params["n"]; pi_z=params["pi_zero"]
    mu_z=params["mu_zero"]
    totals = np.empty(N_SIMS)
    for i in range(N_SIMS):
        n_z = rng.binomial(n, pi_z)
        totals[i] = rng.poisson(mu_z, n_z).sum()
    return totals


def mc_bias_adjusted(params):
    """Bias-adjusted: swap rate applies only to
    a fraction π_problem of precincts."""
    rng = np.random.RandomState(42)
    n=params["n"]; pi_z=params["pi_zero"]
    mu_z=params["mu_zero"]
    s_a=params["swap_shape"]; s_r=params["swap_rate"]
    totals = np.empty(N_SIMS)
    for i in range(N_SIMS):
        pi_prob = rng.beta(1, 9)  # ~10% problem
        n_hi = rng.binomial(n, pi_prob)
        th_hi = rng.gamma(s_a, 1/s_r)
        th_lo = rng.gamma(0.5, 1/10)
        swap = (rng.poisson(th_hi*n_hi)
                + rng.poisson(th_lo*(n-n_hi)))
        n_z = rng.binomial(n, pi_z)
        zero_v = rng.poisson(mu_z, n_z).sum()
        totals[i] = swap + zero_v
    return totals


def _stats(totals, label):
    p5,p50,p95 = np.percentile(totals,[5,50,95])
    pc = (totals >= DEFICIT).mean()
    print(f"  {label:<20} med={p50:>7,.0f}"
          f" [{p5:,.0f}-{p95:,.0f}]"
          f" P={pc:.1%}")
    return dict(model=label, median=p50,
        ci5=p5, ci95=p95, p_cross=pc)


def main():
    df, pred = load_all()
    params = estimate_error_rates(df, pred)
    print(f"\n{SEP}")
    print("STAGE 2: MC Posterior")
    print(SEP)
    rows = []
    t1 = mc_conservative(params)
    rows.append(_stats(t1, "conservative"))
    t2 = mc_posterior(params)
    rows.append(_stats(t2, "full_uniform"))
    t3 = mc_bias_adjusted(params)
    rows.append(_stats(t3, "bias_adjusted"))
    out = pd.DataFrame(rows)
    out.to_csv(DATA/"generative_model_posterior.csv",
               index=False)
    print(f"\n  Saved → generative_model_posterior.csv")


if __name__ == "__main__":
    main()
