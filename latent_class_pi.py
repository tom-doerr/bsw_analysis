#!/usr/bin/env python3
"""Latent-class model to infer π (problem precinct
fraction) from registry flags instead of choosing
it by hand. Anchors the generative model."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import betaln

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


def compute_flags(df, pred):
    """Compute per-precinct flag indicators."""
    g = df["Gültige - Zweitstimmen"].values.astype(float)
    bsw = pd.to_numeric(
        df["BSW - Zweitstimmen"],
        errors="coerce").fillna(0).values
    bp = np.clip(pred["BSW_pred"].values/100,
                 1e-8, 1-1e-8)
    p0 = np.power(1-bp, g)
    resid_z = (pred["BSW_resid"].values
               - pred["BSW_resid"].mean())
    resid_z /= max(pred["BSW_resid"].std(), 1e-6)
    # Flags as noisy indicators of latent problem
    f_zero = (bsw == 0) & (p0 < 0.01)
    f_resid = resid_z < -2
    # At least one flag
    any_flag = f_zero | f_resid
    return f_zero, f_resid, any_flag, g, bsw, bp


def _em_step(f1,f2,pi,s1,s2,fp1,fp2,n):
    l1=f1*np.log(s1+1e-30)+(1-f1)*np.log(1-s1+1e-30)
    l0=f1*np.log(fp1+1e-30)+(1-f1)*np.log(1-fp1+1e-30)
    l1+=f2*np.log(s2+1e-30)+(1-f2)*np.log(1-s2+1e-30)
    l0+=f2*np.log(fp2+1e-30)+(1-f2)*np.log(1-fp2+1e-30)
    ln=np.log(pi+1e-30)+l1
    ld=np.logaddexp(ln,np.log(1-pi+1e-30)+l0)
    g=np.exp(ln-ld); sg=g.sum()
    pi=sg/n; s1=(g*f1).sum()/max(sg,1e-10)
    s2=(g*f2).sum()/max(sg,1e-10); ng=n-sg
    fp1=((1-g)*f1).sum()/max(ng,1e-10)
    fp2=((1-g)*f2).sum()/max(ng,1e-10)
    return pi,s1,s2,fp1,fp2,g


def em_latent_class(f_zero, f_resid, n):
    """EM for latent class: z_i∈{0,1}."""
    pi=0.05;s1,s2=0.5,0.3;fp1,fp2=0.005,0.02
    f1=f_zero.astype(float);f2=f_resid.astype(float)
    for it in range(200):
        pi_o=pi
        pi,s1,s2,fp1,fp2,gam=_em_step(
            f1,f2,pi,s1,s2,fp1,fp2,n)
        if abs(pi-pi_o)<1e-8: break
    return dict(pi=pi,s_zero=s1,s_resid=s2,
        fp_zero=fp1,fp_resid=fp2,
        gamma=gam,iters=it+1)


def bootstrap_pi(f_zero, f_resid, n, n_boot=1000):
    """Bootstrap CI for π."""
    rng = np.random.RandomState(42)
    pis = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r = em_latent_class(
            f_zero[idx], f_resid[idx], n)
        pis.append(r["pi"])
    pis = np.array(pis)
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
    f0, fr, af, g, bsw, bp = compute_flags(df, pred)
    n = len(df)
    print(f"\n{SEP}\nLATENT CLASS MODEL\n{SEP}")
    print(f"  n={n}, flags: zero={f0.sum()},"
          f" resid={fr.sum()}, any={af.sum()}")
    r = em_latent_class(f0, fr, n)
    print(f"\n  EM results ({r['iters']} iters):")
    print(f"    π = {r['pi']:.4f}"
          f" ({r['pi']*100:.2f}% problem)")
    print(f"    sens(zero) = {r['s_zero']:.3f}")
    print(f"    sens(resid) = {r['s_resid']:.3f}")
    print(f"    fp(zero) = {r['fp_zero']:.4f}")
    print(f"    fp(resid) = {r['fp_resid']:.4f}")
    print(f"    E[problem] = {r['pi']*n:.0f} precincts")
    # Bootstrap CI
    print(f"\n  Bootstrapping π (1000 reps)...")
    ci = bootstrap_pi(f0, fr, n, 1000)
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
          dict(param="sens_zero",value=r["s_zero"]),
          dict(param="sens_resid",value=r["s_resid"]),
          dict(param="p_cross",value=pc),
          dict(param="median_missing",value=p50)]
    pd.DataFrame(rows).to_csv(
        DATA/"latent_class_pi.csv",index=False)
    print(f"\n  Saved → latent_class_pi.csv")


if __name__=="__main__":
    main()
