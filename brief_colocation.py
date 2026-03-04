#!/usr/bin/env python3
"""Briefwahl gap co-location analysis.
Cross-references brief/urne gaps with registry
flags and affidavit cases to test whether the
gap is confounding vs miscount."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from bb_utils import estimate_rho, bb_p0

DATA = Path("data")
SEP = "=" * 60
CONTROLS = ["FDP", "Die Linke"]
BD = "BÜNDNIS DEUTSCHLAND"


def load_all():
    from wahlbezirk_lr import (load_2025_wbz,
        LAND_CODE, validate_totals)
    print("Loading data...")
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen",
              "Bezirksart"]:
        df[c] = pd.to_numeric(
            df[c], errors="coerce").fillna(0)
    df = df[df["Gültige - Zweitstimmen"]>=1]
    df = df.copy().reset_index(drop=True)
    pred = pd.read_csv(
        DATA/"wahlbezirk_lr_predictions.csv",
        low_memory=False)
    assert len(df)==len(pred)
    validate_totals(df)
    return df, pred


def _v(df,p):
    c=f"{p} - Zweitstimmen"
    return pd.to_numeric(df[c],errors="coerce").fillna(0)


def _g(df):
    return df["Gültige - Zweitstimmen"].values.astype(float)


def _share(df,p):
    v=_v(df,p).values; g=_g(df)
    return np.where(g>0,v/g*100,0)


def flag_registry(df, pred):
    """Flag suspicious precincts (same as registry)."""
    g=_g(df); bsw=_v(df,"BSW").values
    bp=np.clip(pred["BSW_pred"].values/100,1e-8,1-1e-8)
    rho=estimate_rho(pred, g)
    p0=bb_p0(g, bp, rho)
    susp=(bsw==0)&(p0<0.01)
    bd=_v(df,BD).values
    hi_bd=(bsw==0)&(bd>5)
    return susp, hi_bd, susp|hi_bd


def wkr_brief_gaps(df):
    """Per-WKR Urne-Brief share gaps for BSW+ctrls."""
    g=_g(df)
    wkr=pd.to_numeric(df["Wahlkreis"],errors="coerce")
    ba=df["Bezirksart"].values;u=ba==0;b=ba==5
    ps=["BSW"]+CONTROLS
    gaps={p:[] for p in ps};ids=[];nf=[]
    for w in sorted(wkr.dropna().unique()):
        m=wkr==w;mu=m&u;mb=m&b
        if mu.sum()<5 or mb.sum()<5: continue
        ids.append(w);nf.append(mu.sum())
        for p in ps:
            s=_share(df,p)
            gaps[p].append(
                np.average(s[mu],weights=g[mu])
                -np.average(s[mb],weights=g[mb]))
    return {p:np.array(v) for p,v in gaps.items()},ids,nf


def colocation_test(df, pred):
    """Test co-location of brief gap with registry."""
    print(f"\n{SEP}\nBRIEF GAP CO-LOCATION\n{SEP}")
    susp,hi_bd,any_f=flag_registry(df,pred)
    wkr=pd.to_numeric(df["Wahlkreis"],errors="coerce")
    gaps,ids,_=wkr_brief_gaps(df)
    bg=gaps["BSW"]
    ca=np.mean([gaps[c] for c in CONTROLS],axis=0)
    anomaly=bg-ca
    wkr_flags=[]
    for w in ids:
        m=(wkr==w).values
        wkr_flags.append(any_f[m].sum())
    wkr_flags=np.array(wkr_flags)
    r,p=stats.spearmanr(anomaly,wkr_flags)
    print(f"  Spearman(anomaly,flags): ρ={r:.3f}"
          f" p={p:.3e}")
    return r,p,anomaly,wkr_flags,ids


def demographic_control(df):
    """Compare brief gap across parties."""
    print(f"\n{SEP}\nDEMOGRAPHIC COMPARISON\n{SEP}")
    gaps,ids,_=wkr_brief_gaps(df)
    bg=gaps["BSW"]
    print(f"  BSW gap: {bg.mean():+.3f}pp")
    for p in CONTROLS:
        print(f"  {p} gap: {gaps[p].mean():+.3f}pp")
    ca=np.mean([gaps[c] for c in CONTROLS],axis=0)
    anom=bg-ca
    t,p=stats.ttest_1samp(anom,0)
    print(f"  Anomaly: {anom.mean():+.3f}pp"
          f" t={t:.2f} p={p:.3e}")
    pos=(anom>0).sum()
    print(f"  Positive: {pos}/{len(anom)}")
    return anom


def main():
    df,pred=load_all()
    r,p,anom,wf,ids=colocation_test(df,pred)
    demo_anom=demographic_control(df)
    # High vs low flag split
    med=np.median(wf); hi=wf>med
    if hi.sum()>0 and (~hi).sum()>0:
        print(f"\n  High-flag WKR gap: "
              f"{anom[hi].mean():+.3f}pp")
        print(f"  Low-flag WKR gap: "
              f"{anom[~hi].mean():+.3f}pp")
    # Save
    out=pd.DataFrame(dict(wkr=ids,
        anomaly=np.round(anom,3),
        n_flags=wf))
    out.to_csv(DATA/"brief_colocation.csv",
               index=False)
    print(f"\n  Saved → brief_colocation.csv")


if __name__=="__main__":
    main()
