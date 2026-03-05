#!/usr/bin/env python3
"""Null simulation: BB(p,ρ) no-fraud calibration."""

import numpy as np, pandas as pd
from pathlib import Path
from bb_utils import estimate_rho, bb_p0

DATA = Path("data")
SEP = "=" * 60
N_NULL = 200


def load_all():
    from wahlbezirk_lr import (load_2025_wbz,
        LAND_CODE, validate_totals)
    print("Loading data...")
    df = load_2025_wbz()
    df["Gültige - Zweitstimmen"] = pd.to_numeric(
        df["Gültige - Zweitstimmen"],
        errors="coerce").fillna(0)
    df = df[df["Gültige - Zweitstimmen"]>=1]
    df = df.copy().reset_index(drop=True)
    pred = pd.read_csv(
        DATA/"wahlbezirk_lr_predictions.csv",
        low_memory=False)
    assert len(df)==len(pred)
    validate_totals(df)
    return df, pred


def sim_bb(g, p, rho, rng):
    """Simulate BSW counts from BB(n,p,ρ)."""
    phi = max(1/rho-1, 1e-6)
    a = np.maximum(p*phi, 1e-10)
    b = np.maximum((1-p)*phi, 1e-10)
    pi = rng.beta(a, b)
    return rng.binomial(g.astype(int), pi)


def count_flags(bsw, g, p, rho):
    """Count suspicious_zero (BSW=0 & p0<0.01)."""
    p0 = bb_p0(g, p, rho)
    return int(((bsw==0) & (p0<0.01)).sum())


def _log_norm(x, mu, var):
    return -0.5*(np.log(2*np.pi*var)+(x-mu)**2/var)


def _em_mstep(gam, x1, x2, n):
    sg=gam.sum(); ng=n-sg
    m1=(gam*x1).sum()/max(sg,1e-10)
    m2=(gam*x2).sum()/max(sg,1e-10)
    n1=((1-gam)*x1).sum()/max(ng,1e-10)
    n2=((1-gam)*x2).sum()/max(ng,1e-10)
    _s,_n=max(sg,1),max(ng,1)
    v1=max((gam*(x1-m1)**2).sum()/_s,1e-6)
    v2=max((gam*(x2-m2)**2).sum()/_s,1e-6)
    w1=max(((1-gam)*(x1-n1)**2).sum()/_n,1e-6)
    w2=max(((1-gam)*(x2-n2)**2).sum()/_n,1e-6)
    return m1,n1,m2,n2,v1,w1,v2,w2


def em_pi(x1, x2, n, max_it=200):
    """Fit 2-component Gaussian mixture, return π."""
    pi=0.05
    m=np.percentile(x1,95),np.mean(x1)
    m2=np.percentile(x2,95),np.mean(x2)
    v=np.var(x1),np.var(x1)
    v2=np.var(x2),np.var(x2)
    for it in range(max_it):
        po=pi
        l1=_log_norm(x1,m[0],v[0])+_log_norm(x2,m2[0],v2[0])
        l0=_log_norm(x1,m[1],v[1])+_log_norm(x2,m2[1],v2[1])
        ln1=np.log(pi+1e-30)+l1
        ld=np.logaddexp(ln1,np.log(1-pi+1e-30)+l0)
        gam=np.exp(ln1-ld); pi=gam.sum()/n
        r=_em_mstep(gam,x1,x2,n)
        m=(r[0],r[1]); m2=(r[2],r[3])
        v=(r[4],r[5]); v2=(r[6],r[7])
        if abs(pi-po)<1e-8: break
    return pi


def compute_features(bsw, g, p, rho):
    """Compute x1,x2 from (possibly simulated) counts.
    x1=-log10(BB p0), x2=-(resid_z)."""
    p0 = bb_p0(g, p, rho)
    x1 = -np.log10(np.clip(p0, 1e-300, 1))
    resid = (bsw/np.maximum(g,1) - p) * 100
    mu, sd = resid.mean(), max(resid.std(), 1e-6)
    x2 = -(resid - mu) / sd
    return x1, x2


def main():
    df, pred = load_all()
    n = len(df)
    g = df["Gültige - Zweitstimmen"].values.astype(float)
    bp = np.clip(pred["BSW_pred"].values/100,1e-8,1-1e-8)
    rho = estimate_rho(pred, g)
    print(f"  n={n}, rho={rho:.6f}")
    bsw_obs = pd.to_numeric(
        df["BSW - Zweitstimmen"],
        errors="coerce").fillna(0).values
    obs_flags = count_flags(bsw_obs, g, bp, rho)
    x1, x2 = compute_features(bsw_obs, g, bp, rho)
    obs_pi = em_pi(x1, x2, n)
    print(f"\n{SEP}\nNULL CALIBRATION\n{SEP}")
    print(f"  Observed flags: {obs_flags}")
    print(f"  Observed π: {obs_pi:.4f}")
    _run_sims(g, bp, rho, obs_flags, obs_pi)


def _run_sims(g, bp, rho, obs_f, obs_pi):
    n = len(g)
    print(f"\n  Simulating {N_NULL} null worlds...")
    rng = np.random.RandomState(42)
    nf, npi = [], []
    for i in range(N_NULL):
        bsw_s = sim_bb(g, bp, rho, rng)
        nf.append(count_flags(bsw_s, g, bp, rho))
        x1, x2 = compute_features(bsw_s, g, bp, rho)
        npi.append(em_pi(x1, x2, n))
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{N_NULL}")
    nf = np.array(nf); npi = np.array(npi)
    _report(nf, npi, obs_f, obs_pi)


def _report(nf, npi, obs_f, obs_pi):
    pf = (nf >= obs_f).mean()
    fp = np.percentile(nf, [5, 50, 95])
    print(f"\n{SEP}\nFLAG RATE CALIBRATION\n{SEP}")
    print(f"  Null: med={fp[1]:.0f}"
          f" [{fp[0]:.0f}, {fp[2]:.0f}]")
    print(f"  Observed: {obs_f}")
    print(f"  P(null>=obs): {pf:.4f}")
    print(f"  Excess: {obs_f-fp[1]:.0f}")

    pp = (npi >= obs_pi).mean()
    pip = np.percentile(npi, [5, 50, 95])
    print(f"\n{SEP}\nLATENT π CALIBRATION\n{SEP}")
    print(f"  Null π: med={pip[1]:.4f}"
          f" [{pip[0]:.4f}, {pip[2]:.4f}]")
    print(f"  Observed π: {obs_pi:.4f}")
    print(f"  P(null>=obs): {pp:.4f}")
    _save(obs_f, fp, pf, obs_pi, pip, pp)


def _save(of, fp, pf, opi, pip, pp):
    rows = [
        dict(metric="flags_obs", value=of),
        dict(metric="flags_null_med", value=fp[1]),
        dict(metric="flags_p", value=pf),
        dict(metric="pi_obs", value=opi),
        dict(metric="pi_null_med", value=pip[1]),
        dict(metric="pi_p", value=pp)]
    pd.DataFrame(rows).to_csv(
        DATA/"null_calibration.csv", index=False)
    print(f"\n  Saved → null_calibration.csv")


if __name__ == "__main__":
    main()
