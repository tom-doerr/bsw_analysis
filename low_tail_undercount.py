#!/usr/bin/env python3
"""Low-tail undercount analysis.
Computes P(BSW <= obs) under BB for ALL precincts,
not just BSW=0. Null-calibrates the excess."""

import numpy as np, pandas as pd
from pathlib import Path
from scipy.special import betaln
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
    return df, pred, LAND_CODE


def bb_cdf(k, n, p, rho):
    """P(X<=k) under BB(n,p,rho). Sums PMF 0..k[i]."""
    phi=max(1/rho-1,1e-6)
    a=np.maximum(p*phi,1e-10)
    b=np.maximum((1-p)*phi,1e-10)
    ni=n.astype(int); ki=k.astype(int)
    mx=int(ki.max())+1
    lp=betaln(a,b+ni)-betaln(a,b)
    cum=np.exp(np.clip(lp,-700,0))
    res=np.zeros(len(n))
    done=ki==0; res[done]=cum[done]
    if done.all(): return res
    for j in range(1,mx):
        lp+=(np.log(np.maximum(ni-j+1,1e-300))-np.log(j)
            +np.log(a+j-1)-np.log(np.maximum(b+ni-j,1e-300)))
        cum+=np.exp(np.clip(lp,-700,0))
        nw=(~done)&(ki==j)
        res[nw]=cum[nw]; done|=nw
        if done.all(): break
    res[~done]=cum[~done]
    return np.clip(res,0,1)


def compute_tail(g, bsw, bp, rho, alpha=0.01):
    """Find low-tail precincts + expected missing."""
    lam = bp*g
    # Only compute CDF where BSW < lambda (rest can't be low-tail)
    cand = bsw < lam
    pcdf = np.ones(len(g))
    if cand.any():
        pcdf[cand] = bb_cdf(
            bsw[cand], g[cand], bp[cand], rho)
    low = pcdf < alpha
    miss = np.where(low, lam - bsw, 0)
    miss = np.maximum(miss, 0)
    return pcdf, low, miss


def sim_bb(g, p, rho, rng):
    """Simulate BSW from BB(n,p,rho)."""
    phi=max(1/rho-1,1e-6)
    a=np.maximum(p*phi,1e-10)
    b=np.maximum((1-p)*phi,1e-10)
    pi=rng.beta(a,b)
    return rng.binomial(g.astype(int),pi)


def null_calibrate(g, bp, rho, obs_n, obs_miss):
    """Simulate null worlds and compare."""
    rng = np.random.RandomState(42)
    nn, nm = [], []
    print(f"\n  Simulating {N_NULL} null worlds...")
    for i in range(N_NULL):
        bs = sim_bb(g, bp, rho, rng)
        _, low, miss = compute_tail(g, bs, bp, rho)
        nn.append(int(low.sum()))
        nm.append(float(miss.sum()))
        if (i+1)%50==0: print(f"    {i+1}/{N_NULL}")
    nn=np.array(nn); nm=np.array(nm)
    pn=((nn>=obs_n).sum()+1)/(N_NULL+1)
    pm=((nm>=obs_miss).sum()+1)/(N_NULL+1)
    return nn, nm, pn, pm


def _report(g,bsw,bp,low,miss):
    nl=int(low.sum()); np_=int((low&(bsw>0)).sum())
    print(f"\n{SEP}\nLOW-TAIL UNDERCOUNT\n{SEP}")
    print(f"  Low-tail (p<0.01): {nl}")
    print(f"    BSW=0: {nl-np_}, BSW>0: {np_}")
    tm=float(miss.sum())
    m0=float(miss[bsw==0].sum())
    mp=float(miss[bsw>0].sum())
    print(f"  Expected missing: {tm:,.0f}")
    print(f"    from BSW=0: {m0:,.0f}")
    print(f"    from BSW>0: {mp:,.0f}")


def _bins(g,bp,low,miss):
    lam=bp*g; bins=[0,5,10,20,50,100,np.inf]
    lb=np.digitize(lam,bins)
    print(f"\n  {'λ bin':<10}{'n':>5}{'miss':>7}")
    for bi in range(1,len(bins)):
        lo,hi=bins[bi-1],bins[bi]
        m=lb==bi; nl=int(low[m].sum())
        mm=float(miss[m].sum())
        lab=f"{lo:.0f}-{hi:.0f}" if hi<np.inf else f"{lo:.0f}+"
        print(f"  {lab:<10}{nl:>5}{mm:>7.0f}")


def _null_report(nn, nm, pn, pm, obs_n, obs_m):
    fp=np.percentile(nn,[5,50,95])
    print(f"\n{SEP}\nNULL CALIBRATION\n{SEP}")
    print(f"  Null n: med={fp[1]:.0f} [{fp[0]:.0f},{fp[2]:.0f}]")
    print(f"  Observed n: {obs_n}")
    print(f"  P(null>=obs): {pn:.4f}")
    print(f"  Excess: {obs_n-fp[1]:.0f}")
    mp=np.percentile(nm,[5,50,95])
    print(f"\n  Null miss: med={mp[1]:,.0f}"
          f" [{mp[0]:,.0f},{mp[2]:,.0f}]")
    print(f"  Observed miss: {obs_m:,.0f}")
    print(f"  P(null>=obs): {pm:.4f}")
    print(f"  Excess miss: {obs_m-mp[1]:,.0f}")


def _save(on,om,nn,nm,pn,pm):
    f=np.percentile(nn,[5,50,95])
    m=np.percentile(nm,[5,50,95])
    rows=[dict(m="n_obs",v=on),dict(m="n_med",v=f[1]),
        dict(m="n_p5",v=f[0]),dict(m="n_p95",v=f[2]),
        dict(m="n_p",v=round(pn,4)),
        dict(m="miss_obs",v=round(om,1)),
        dict(m="miss_med",v=round(m[1],1)),
        dict(m="miss_p5",v=round(m[0],1)),
        dict(m="miss_p95",v=round(m[2],1)),
        dict(m="miss_p",v=round(pm,4)),
        dict(m="excess_n",v=round(on-f[1])),
        dict(m="excess_miss",v=round(om-m[1],1))]
    pd.DataFrame(rows).to_csv(
        DATA/"low_tail_calibration.csv",index=False)
    print(f"\n  Saved → low_tail_calibration.csv")


def main():
    df,pred,lc = load_all()
    n=len(df)
    g=df["Gültige - Zweitstimmen"].values.astype(float)
    bsw=pd.to_numeric(df["BSW - Zweitstimmen"],
        errors="coerce").fillna(0).values
    bp=np.clip(pred["BSW_pred"].values/100,1e-8,1-1e-8)
    rho=estimate_rho(pred,g)
    print(f"  n={n}, rho={rho:.6f}")
    pcdf,low,miss = compute_tail(g,bsw,bp,rho)
    _report(g,bsw,bp,low,miss)
    _bins(g,bp,low,miss)
    obs_n=int(low.sum()); obs_m=float(miss.sum())
    nn,nm,pn,pm = null_calibrate(g,bp,rho,obs_n,obs_m)
    _null_report(nn,nm,pn,pm,obs_n,obs_m)
    _save(obs_n,obs_m,nn,nm,pn,pm)


if __name__=="__main__":
    main()
