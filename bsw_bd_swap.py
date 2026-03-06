#!/usr/bin/env python3
"""BSW->BD swap misallocation model.
Estimates fraction q of BSW votes mis-attributed to BD,
using Saarland (non-adjacent ballot) as control."""

import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar
from bb_utils import estimate_rho, bb_p0

DATA = Path("data")
SEP = "=" * 60
BD = "BÜNDNIS DEUTSCHLAND"
N_BOOT = 1000


def load_all():
    from wahlbezirk_lr import (load_2025_wbz,
        LAND_CODE, validate_totals)
    print("Loading data...")
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen","Bezirksart"]:
        df[c] = pd.to_numeric(
            df[c], errors="coerce").fillna(0)
    df = df[df["Gültige - Zweitstimmen"]>=1]
    df = df.copy().reset_index(drop=True)
    pred = pd.read_csv(
        DATA/"wahlbezirk_lr_predictions.csv",
        low_memory=False)
    assert len(df)==len(pred)
    validate_totals(df)
    return df, pred, LAND_CODE


def get_cols(df, pred, lc):
    g=df["Gültige - Zweitstimmen"].values.astype(float)
    bsw=pd.to_numeric(df["BSW - Zweitstimmen"],
        errors="coerce").fillna(0).values
    bd=pd.to_numeric(df[f"{BD} - Zweitstimmen"],
        errors="coerce").fillna(0).values
    bp=np.clip(pred["BSW_pred"].values/100,1e-8,1-1e-8)
    land=pd.to_numeric(df["Land"],errors="coerce")
    ln=land.map(lc).fillna("").values
    ba=df["Bezirksart"].values.astype(int)
    return g,bsw,bd,bp,ln,ba


def estimate_q(bsw, bd, bp, g, adj):
    """Estimate swap fraction q via excess BD in adjacent states.
    Model: BD_obs = BD_true + q*BSW_true in adj states.
    Control: SL (non-adjacent) gives BD_true baseline."""
    # BD share: adjacent vs non-adjacent
    bd_adj = bd[adj]/np.maximum(g[adj],1)
    bd_ctl = bd[~adj]/np.maximum(g[~adj],1)
    # Weighted means
    w_adj = g[adj]/g[adj].sum()
    w_ctl = g[~adj]/g[~adj].sum()
    bd_rate_adj = (bd_adj*w_adj).sum()
    bd_rate_ctl = (bd_ctl*w_ctl).sum()
    # Excess BD in adjacent = q * BSW_pred
    bsw_rate_adj = (bp[adj]*w_adj).sum()
    excess = bd_rate_adj - bd_rate_ctl
    q = max(excess / max(bsw_rate_adj, 1e-10), 0)
    return q, bd_rate_adj, bd_rate_ctl, excess


def bootstrap_q(bsw,bd,bp,g,adj,n_boot=N_BOOT):
    """Bootstrap CI for q."""
    rng=np.random.RandomState(42)
    n=len(g); qs=[]
    for _ in range(n_boot):
        idx=rng.choice(n,n,replace=True)
        q,_,_,_ = estimate_q(
            bsw[idx],bd[idx],bp[idx],g[idx],adj[idx])
        qs.append(q)
    qs=np.array(qs)
    return qs, np.percentile(qs,[2.5,50,97.5])


def national_swap(q, bp, g, adj):
    """Estimate total swapped votes nationally."""
    lam = bp * g
    swap_adj = q * lam[adj]
    return float(swap_adj.sum())


def placebo(df, pred, lc, party, col):
    """Run same analysis for a control party."""
    g,_,_,_,ln,ba = get_cols(df, pred, lc)
    pv = pd.to_numeric(df[col],
        errors="coerce").fillna(0).values
    bp = np.clip(pred["BSW_pred"].values/100,1e-8,1-1e-8)
    adj = ln!="SL"
    pr = pv/np.maximum(g,1)
    w_a=g[adj]/g[adj].sum()
    w_c=g[~adj]/g[~adj].sum()
    ra=(pr[adj]*w_a).sum()
    rc=(pr[~adj]*w_c).sum()
    return party, ra*100, rc*100, (ra-rc)*100


def main():
    df,pred,lc = load_all()
    g,bsw,bd,bp,ln,ba = get_cols(df,pred,lc)
    adj = ln!="SL"
    print(f"  Adjacent: {adj.sum():,}, Control(SL): {(~adj).sum():,}")
    q,ba_,bc_,exc = estimate_q(bsw,bd,bp,g,adj)
    print(f"\n{SEP}\nBSW→BD SWAP MODEL\n{SEP}")
    print(f"  BD rate adj: {ba_*100:.4f}%")
    print(f"  BD rate ctl: {bc_*100:.4f}%")
    print(f"  Excess: {exc*100:.4f}pp")
    print(f"  q: {q:.4f}")
    tot = national_swap(q, bp, g, adj)
    print(f"  Estimated swapped: {tot:,.0f}")
    _boot(bsw,bd,bp,g,adj)
    _placebos(df,pred,lc,g,ln)
    _per_land(g,bsw,bd,bp,ln)
    _save_results(q,tot,bsw,bd,bp,g,adj)


def _boot(bsw,bd,bp,g,adj):
    print("\n  Bootstrapping q...")
    qs,ci = bootstrap_q(bsw,bd,bp,g,adj)
    print(f"  q: med={ci[1]:.4f} [{ci[0]:.4f},{ci[2]:.4f}]")
    tots=[national_swap(qi,bp,g,adj) for qi in ci]
    print(f"  Votes: [{tots[0]:,.0f},{tots[1]:,.0f},{tots[2]:,.0f}]")


def _placebos(df,pred,lc,g,ln):
    """Run placebo tests with other small parties."""
    print(f"\n  Placebo (adj vs SL control):")
    for name,col in [("FDP","FDP - Zweitstimmen"),
        ("Linke","Die Linke - Zweitstimmen"),
        ("Volt","Volt - Zweitstimmen")]:
        pv=pd.to_numeric(df[col],errors="coerce").fillna(0).values
        adj=ln!="SL"
        pr=pv/np.maximum(g,1)
        wa=g[adj]/g[adj].sum(); wc=g[~adj]/g[~adj].sum()
        ra=(pr[adj]*wa).sum()*100
        rc=(pr[~adj]*wc).sum()*100
        print(f"    {name}: adj={ra:.3f}% ctl={rc:.3f}%"
              f" gap={ra-rc:+.3f}pp")


def _per_land(g,bsw,bd,bp,ln):
    print(f"\n  Per-Land BD rate:")
    print(f"  {'Land':<4}{'BD%':>7}{'BSW%':>7}{'n':>7}")
    for l in sorted(set(ln)):
        if not l: continue
        m=ln==l
        w=g[m]/g[m].sum()
        br=(bd[m]/np.maximum(g[m],1)*w).sum()*100
        bs=(bp[m]*w).sum()*100
        print(f"  {l:<4}{br:>7.3f}{bs:>7.3f}{int(m.sum()):>7}")


def _save_results(q,tot,bsw,bd,bp,g,adj):
    qs,ci = bootstrap_q(bsw,bd,bp,g,adj,n_boot=200)
    tots=[national_swap(qi,bp,g,adj) for qi in ci]
    rows=[dict(m="q",v=round(q,6)),
        dict(m="q_lo",v=round(ci[0],6)),
        dict(m="q_hi",v=round(ci[2],6)),
        dict(m="swap_votes",v=round(tot)),
        dict(m="swap_lo",v=round(tots[0])),
        dict(m="swap_hi",v=round(tots[2])),
        dict(m="bd_total",v=int(bd.sum())),
        dict(m="bsw_total",v=int(bsw.sum()))]
    pd.DataFrame(rows).to_csv(
        DATA/"bsw_bd_swap.csv",index=False)
    print(f"\n  Saved → bsw_bd_swap.csv")


if __name__=="__main__":
    main()
