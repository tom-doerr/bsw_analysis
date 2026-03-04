#!/usr/bin/env python3
"""Latent-class model to infer π (problem precinct
fraction). Uses continuous likelihood (not binary
cutoffs) for identifiability. BB-calibrated."""

import numpy as np
import pandas as pd
from pathlib import Path
from bb_utils import estimate_rho, bb_p0

DATA = Path("data")
SEP = "=" * 60
DEFICIT = 9_529
N_SIMS = 50_000


def load_all():
    from wahlbezirk_lr import (load_2025_wbz,
        LAND_CODE, validate_totals)
    print("Loading data...")
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen"]:
        df[c] = pd.to_numeric(
            df[c], errors="coerce").fillna(0)
    df = df[df["Gültige - Zweitstimmen"]>=1]
    df = df.copy().reset_index(drop=True)
    pred = pd.read_csv(
        DATA/"wahlbezirk_lr_predictions.csv",
        low_memory=False)
    assert len(df) == len(pred)
    validate_totals(df)
    return df, pred


def compute_continuous(df, pred):
    """Compute continuous indicators for EM."""
    g = df["Gültige - Zweitstimmen"].values.astype(float)
    bsw = pd.to_numeric(
        df["BSW - Zweitstimmen"],
        errors="coerce").fillna(0).values
    bp = np.clip(pred["BSW_pred"].values/100,
                 1e-8, 1-1e-8)
    rho = estimate_rho(pred, g)
    p0 = bb_p0(g, bp, rho)
    # Continuous x1: -log10(p0), higher=more suspicious
    x1 = -np.log10(np.clip(p0, 1e-300, 1))
    # Continuous x2: -resid_z, higher=more negative
    rz = (pred["BSW_resid"].values
          - pred["BSW_resid"].mean())
    rz /= max(pred["BSW_resid"].std(), 1e-6)
    x2 = -rz
    return x1, x2, g, bsw, bp, rho


def _log_norm(x, mu, var):
    """Log-pdf of N(mu, var)."""
    return -0.5*(np.log(2*np.pi*var)+(x-mu)**2/var)


def _em_mstep(gam, x1, x2, n):
    """M-step: update means and variances."""
    sg=gam.sum(); ng=n-sg
    m1_1=(gam*x1).sum()/max(sg,1e-10)
    m2_1=(gam*x2).sum()/max(sg,1e-10)
    m1_0=((1-gam)*x1).sum()/max(ng,1e-10)
    m2_0=((1-gam)*x2).sum()/max(ng,1e-10)
    _s,_n=max(sg,1),max(ng,1)
    v1_1=max((gam*(x1-m1_1)**2).sum()/_s,1e-6)
    v2_1=max((gam*(x2-m2_1)**2).sum()/_s,1e-6)
    v1_0=max(((1-gam)*(x1-m1_0)**2).sum()/_n,1e-6)
    v2_0=max(((1-gam)*(x2-m2_0)**2).sum()/_n,1e-6)
    return (m1_1,m1_0,m2_1,m2_0,
            v1_1,v1_0,v2_1,v2_0)


def em_continuous(x1, x2, n, max_it=500):
    """EM: continuous Gaussian mixture."""
    pi=0.05
    m=np.percentile(x1,95),np.mean(x1)
    m2=np.percentile(x2,95),np.mean(x2)
    v=np.var(x1),np.var(x1)
    v2=np.var(x2),np.var(x2)
    for it in range(max_it):
        pi_o=pi
        ll1=_log_norm(x1,m[0],v[0])+_log_norm(x2,m2[0],v2[0])
        ll0=_log_norm(x1,m[1],v[1])+_log_norm(x2,m2[1],v2[1])
        ln1=np.log(pi+1e-30)+ll1
        ld=np.logaddexp(ln1,np.log(1-pi+1e-30)+ll0)
        gam=np.exp(ln1-ld); pi=gam.sum()/n
        r=_em_mstep(gam,x1,x2,n)
        m=(r[0],r[1]); m2=(r[2],r[3])
        v=(r[4],r[5]); v2=(r[6],r[7])
        if abs(pi-pi_o)<1e-8: break
    return dict(pi=pi,gamma=gam,iters=it+1,
        mu1=m,mu2=m2,var1=v,var2=v2)


def bootstrap_pi(x1, x2, n, n_boot=1000):
    """Bootstrap CI for π."""
    rng = np.random.RandomState(42)
    pis = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r = em_continuous(x1[idx], x2[idx], n)
        pis.append(r["pi"])
    return np.percentile(pis, [2.5, 50, 97.5])


def mc_with_inferred_pi(pi, n, n_sims=N_SIMS):
    """Run generative model with data-inferred π."""
    rng = np.random.RandomState(42)
    s_a, s_r = 15, 50  # recount Gamma params
    pi_z = 0.0035  # zeroout rate from calibration
    mu_z = 12.0  # mean votes per zeroout
    totals = np.empty(n_sims)
    for i in range(n_sims):
        n_hi = rng.binomial(n, pi)
        th_hi = rng.gamma(s_a, 1/s_r)
        th_lo = rng.gamma(0.5, 1/10)
        swap = (rng.poisson(th_hi*n_hi)
                + rng.poisson(th_lo*(n-n_hi)))
        n_z = rng.binomial(n, pi_z)
        zv = rng.poisson(mu_z, n_z).sum()
        totals[i] = swap + zv
    return totals


def main():
    df, pred = load_all()
    x1, x2, g, bsw, bp, rho = compute_continuous(df, pred)
    n = len(df)
    print(f"\n{SEP}\nLATENT CLASS MODEL (continuous)\n{SEP}")
    print(f"  n={n}, BB rho={rho:.6f}")
    r = em_continuous(x1, x2, n)
    print(f"\n  EM results ({r['iters']} iters):")
    print(f"    π = {r['pi']:.4f}"
          f" ({r['pi']*100:.2f}% problem)")
    print(f"    μ(-log10 p0): prob={r['mu1'][0]:.2f}"
          f" norm={r['mu1'][1]:.2f}")
    print(f"    μ(-resid_z): prob={r['mu2'][0]:.2f}"
          f" norm={r['mu2'][1]:.2f}")
    print(f"    E[problem] = {r['pi']*n:.0f}")
    # Bootstrap CI
    print(f"\n  Bootstrapping π (1000 reps)...")
    ci = bootstrap_pi(x1, x2, n, 1000)
    print(f"    π 95% CI: [{ci[0]:.4f},"
          f" {ci[1]:.4f}, {ci[2]:.4f}]")
    # MC with inferred π
    print(f"\n{SEP}\nGENERATIVE (π from data)\n{SEP}")
    t = mc_with_inferred_pi(r["pi"], n)
    p5,p50,p95 = np.percentile(t,[5,50,95])
    pc = (t >= DEFICIT).mean()
    print(f"  π={r['pi']:.4f}: med={p50:,.0f}"
          f" [{p5:,.0f}-{p95:,.0f}] P={pc:.1%}")
    # CI bounds
    for lb,pv in [("lo",ci[0]),("hi",ci[2])]:
        tv=mc_with_inferred_pi(pv,n)
        m=np.median(tv);pv2=(tv>=DEFICIT).mean()
        print(f"  π={pv:.4f}({lb}): med={m:,.0f}"
              f" P={pv2:.1%}")
    # Save
    rows=[dict(param="pi_mle",value=r["pi"]),
          dict(param="pi_ci_lo",value=ci[0]),
          dict(param="pi_ci_hi",value=ci[2]),
          dict(param="mu_logp0_prob",value=r["mu1"][0]),
          dict(param="mu_logp0_norm",value=r["mu1"][1]),
          dict(param="p_cross",value=pc),
          dict(param="median_missing",value=p50)]
    pd.DataFrame(rows).to_csv(
        DATA/"latent_class_pi.csv",index=False)
    print(f"\n  Saved → latent_class_pi.csv")


if __name__=="__main__":
    main()
