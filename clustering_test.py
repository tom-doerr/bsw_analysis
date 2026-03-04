#!/usr/bin/env python3
"""Geographic clustering of suspicious precincts.
Tests whether anomalies cluster by Gemeinde/Wahlkreis
beyond chance (permutation test)."""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

DATA = Path("data")
SEP = "=" * 60


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
    assert len(df)==len(pred)
    validate_totals(df)
    return df, pred


def flag_suspicious(df, pred):
    """Flag BSW=0 where Binomial P<0.01."""
    g = df["Gültige - Zweitstimmen"].values.astype(float)
    bsw = pd.to_numeric(
        df["BSW - Zweitstimmen"],
        errors="coerce").fillna(0).values
    bp = np.clip(pred["BSW_pred"].values/100,
                 1e-8, 1-1e-8)
    p0 = np.power(1-bp, g)
    susp = (bsw == 0) & (p0 < 0.01)
    miss = np.where(bsw == 0, bp*g, 0)
    return susp, miss


def get_geo(df):
    """Extract geographic keys."""
    wkr = pd.to_numeric(
        df.get("Wahlkreis", df.iloc[:,0]),
        errors="coerce").fillna(0).astype(int).values
    gem = df.get("Gemeindename",
        df.get("Gemeinde",
        pd.Series([""]*len(df)))).values
    from wahlbezirk_lr import LAND_CODE
    land = pd.to_numeric(
        df["Land"], errors="coerce")
    ln = land.map(LAND_CODE).fillna("").values
    ba = pd.to_numeric(
        df["Bezirksart"],errors="coerce").fillna(0)
    return wkr, gem, ln, ba.values.astype(int)


def top_gemeinden(susp, miss, gem, ln, k=20):
    """Top k Gemeinden by total missing_votes."""
    print(f"\n{SEP}\nTOP {k} GEMEINDEN by missing votes\n{SEP}")
    gkey = np.array([f"{l}:{g}" for l,g in zip(ln,gem)])
    df = pd.DataFrame({"key":gkey,"susp":susp,"miss":miss})
    agg = df.groupby("key").agg(
        n_susp=("susp","sum"),
        total_miss=("miss","sum"),
        n_total=("susp","count")).reset_index()
    agg = agg[agg.n_susp>0].sort_values(
        "total_miss",ascending=False).head(k)
    print(f"  {'Gemeinde':<35}{'n':>4}{'miss':>7}")
    for _,r in agg.iterrows():
        print(f"  {r.key:<35}{r.n_susp:>4}"
              f"{r.total_miss:>7.0f}")
    return agg


def perm_test(susp, groups, n_perm=5000):
    """Permutation test: is max(count/group) larger
    than chance? Returns obs_max, p-value."""
    rng=np.random.RandomState(42)
    ug=np.unique(groups); ns=int(susp.sum()); n=len(susp)
    obs_max=max(susp[groups==g].sum() for g in ug)
    exceed=0
    for _ in range(n_perm):
        p=np.zeros(n,dtype=bool)
        p[rng.choice(n,ns,replace=False)]=True
        mx=max(p[groups==g].sum() for g in ug)
        if mx>=obs_max: exceed+=1
    return obs_max, exceed/n_perm


def top_wahlkreise(susp, miss, wkr, k=20):
    """Top k Wahlkreise by suspicious_zero count."""
    print(f"\n{SEP}\nTOP {k} WAHLKREISE by susp zeros\n{SEP}")
    df = pd.DataFrame({"wkr":wkr,"susp":susp,"miss":miss})
    agg = df.groupby("wkr").agg(
        n_susp=("susp","sum"),
        total_miss=("miss","sum"),
        n_total=("susp","count")).reset_index()
    agg = agg[agg.n_susp>0].sort_values(
        "n_susp",ascending=False).head(k)
    print(f"  {'WKR':>4}{'n_susp':>7}{'miss':>8}"
          f"{'total':>7}")
    for _,r in agg.iterrows():
        print(f"  {r.wkr:>4}{r.n_susp:>7}"
              f"{r.total_miss:>8.0f}{r.n_total:>7}")
    return agg


def by_land(susp, ln):
    """Breakdown by Land."""
    print(f"\n{SEP}\nBY LAND\n{SEP}")
    for l in sorted(set(ln)):
        if not l: continue
        m=ln==l; ns=susp[m].sum(); nt=m.sum()
        r=ns/max(nt,1)*100
        print(f"  {l:<4}{ns:>5}/{nt:>6} ({r:.2f}%)")


def by_bezirksart(susp, ba):
    """Breakdown by Bezirksart."""
    print(f"\n{SEP}\nBY BEZIRKSART\n{SEP}")
    for b in sorted(set(ba)):
        m=ba==b;ns=susp[m].sum();nt=m.sum()
        r=ns/max(nt,1)*100
        nm={0:"Urne",5:"Brief",6:"Sonder",8:"Kombi"}
        print(f"  {nm.get(b,str(b)):<8}"
              f"{ns:>5}/{nt:>6} ({r:.2f}%)")


def main():
    df,pred=load_all()
    susp,miss=flag_suspicious(df,pred)
    wkr,gem,ln,ba=get_geo(df)
    print(f"\n  Suspicious: {susp.sum()}/{len(df)}")
    tg=top_gemeinden(susp,miss,gem,ln)
    tw=top_wahlkreise(susp,miss,wkr)
    by_land(susp,ln); by_bezirksart(susp,ba)
    # Permutation tests
    print(f"\n{SEP}\nPERMUTATION TESTS\n{SEP}")
    print("  WKR clustering (5000 perms)...")
    mx_w,p_w=perm_test(susp,wkr)
    print(f"  WKR: max={mx_w}, p={p_w:.4f}")
    print("  Land clustering...")
    mx_l,p_l=perm_test(susp,ln)
    print(f"  Land: max={mx_l}, p={p_l:.4f}")
    _save(tg,tw,mx_w,p_w,mx_l,p_l)


def _save(tg,tw,mx_w,p_w,mx_l,p_l):
    rows=[]
    for _,r in tg.iterrows():
        rows.append(dict(level="gemeinde",key=r.key,
            n_susp=r.n_susp,
            total_miss=round(r.total_miss,1)))
    for _,r in tw.iterrows():
        rows.append(dict(level="wahlkreis",
            key=str(int(r.wkr)),n_susp=r.n_susp,
            total_miss=round(r.total_miss,1)))
    pd.DataFrame(rows).to_csv(
        DATA/"clustering_results.csv",index=False)
    pd.DataFrame([
        dict(test="wkr",max_count=mx_w,p=p_w),
        dict(test="land",max_count=mx_l,p=p_l),
    ]).to_csv(DATA/"clustering_perm.csv",index=False)
    print(f"\n  Saved → clustering_results.csv")


if __name__=="__main__":
    main()
