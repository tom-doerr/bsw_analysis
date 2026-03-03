#!/usr/bin/env python3
"""Ballot adjacency natural experiment: test whether
BSW/BD confusion causes BSW undercounting."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr

from wahlbezirk_lr import (load_2025_wbz, LAND_CODE,
                            validate_totals)

DATA = Path("data")
SEP = "=" * 60
BD = "BÜNDNIS DEUTSCHLAND"


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


def _ve(df, party):
    c = f"{party} - Erststimmen"
    return pd.to_numeric(df[c], errors="coerce").fillna(0)


def _g(df):
    return df["Gültige - Zweitstimmen"].values.astype(float)


def erst_presence_did(df, pred):
    """Treatment A: BSW Erst presence → familiarity.
    Compare anomaly rates in WKRs with vs without BSW
    Erst candidates."""
    print(f"\n{SEP}")
    print("TREATMENT A: Erststimme Presence DiD")
    print(SEP)
    g = _g(df)
    bsw = _v(df, "BSW").values
    bd = _v(df, BD).values
    bsw_erst = _ve(df, "BSW").values
    wkr = pd.to_numeric(df["Wahlkreis"],
                        errors="coerce")
    land = pd.to_numeric(df["Land"], errors="coerce")

    # Per-WKR: does BSW have Erst candidate?
    wkr_erst = df.groupby(wkr).apply(
        lambda x: _ve(x, "BSW").sum() > 0)
    has_erst = wkr.map(wkr_erst).fillna(False).values
    bsw_z = bsw; bd_z = bd
    is_zero = bsw_z == 0
    zbd = is_zero & (bd_z > 5)
    bp = pred["BSW_pred"].values / 100
    lam = np.maximum(bp * g, 1e-6)
    susp = is_zero & (np.exp(-lam) < 0.01)
    for lab, m in [("BSW=0", is_zero),
                   ("BSW=0&BD>5", zbd),
                   ("Susp_zero", susp)]:
        re = m[has_erst].mean()
        rn = m[~has_erst].mean()
        rat = re/rn if rn > 0 else float("inf")
        print(f"\n  {lab}: erst={re:.4%}"
              f" no={rn:.4%} ratio={rat:.2f}")
    return has_erst


def urne_brief_did(df, pred):
    """Treatment B: Urne vs Brief counting."""
    print(f"\n{SEP}")
    print("TREATMENT B: Urne vs Brief")
    print(SEP)
    g=_g(df); bsw=_v(df,"BSW").values
    bd=_v(df,BD).values
    ba=pd.to_numeric(df["Bezirksart"],errors="coerce")
    urne=(ba==0).values; brief=(ba==5).values
    bp=pred["BSW_pred"].values/100
    lam=np.maximum(bp*g,1e-6)
    z=bsw==0; zbd=z&(bd>5)
    s=z&(np.exp(-lam)<0.01)
    for lab,m in [("BSW=0",z),("BSW=0&BD>5",zbd),
                  ("Susp",s)]:
        ru=m[urne].mean(); rb=m[brief].mean()
        rat=ru/rb if rb>0 else float("inf")
        print(f"  {lab}: urne={ru:.4%}"
              f" brief={rb:.4%} ratio={rat:.2f}")


def placebo_pairs(df):
    """Placebo: adjacent ballot pairs.
    ~25=BüSo, 26=BD, 27=BSW, 28=MERA25"""
    print(f"\n{SEP}")
    print("PLACEBO: Adjacent party pairs")
    print(SEP)
    g = _g(df)
    pairs = [("BSW", BD, "BSW↔BD"),
             ("MERA25", "BSW", "MERA25↔BSW"),
             ("FDP", "FREIE WÄHLER", "FDP↔FW")]
    for p1, p2, lab in pairs:
        v1 = _v(df,p1).values
        v2 = _v(df,p2).values
        z = (v1==0) & (v2>5)
        r,_ = pearsonr(
            np.where(g>0,v1/g*100,0),
            np.where(g>0,v2/g*100,0))
        print(f"  {lab}: {p1}=0&{p2}>5:"
              f" {z.sum()} ({z.mean():.4%})"
              f" r={r:+.4f}")


def size_stratified(df, pred):
    """Anomaly rate by precinct size quintile."""
    print(f"\n{SEP}")
    print("SIZE STRATIFICATION")
    print(SEP)
    g = _g(df)
    bsw = _v(df,"BSW").values
    bd = _v(df,BD).values
    bp = pred["BSW_pred"].values/100
    lam = np.maximum(bp*g, 1e-6)
    z = bsw==0; s = z&(np.exp(-lam)<0.01)
    q = pd.qcut(g, 5, labels=False,
                duplicates="drop")
    print(f"  {'Q':>2} {'size':>8} {'n':>6}"
          f" {'BSW=0':>6} {'susp':>5}")
    for qi in range(5):
        m = q == qi
        sz = g[m].mean()
        nz = z[m].sum()
        ns = s[m].sum()
        print(f"  {qi:>2} {sz:>8.0f} {m.sum():>6}"
              f" {nz:>6} {ns:>5}")


def bsw_resid_by_erst(df, pred):
    """BSW residuals: Erst-present vs absent WKRs."""
    print(f"\n{SEP}")
    print("RESIDUAL DiD: BSW residuals by Erst")
    print(SEP)
    wkr = pd.to_numeric(df["Wahlkreis"],
                        errors="coerce")
    wkr_erst = df.groupby(wkr).apply(
        lambda x: _ve(x,"BSW").sum() > 0)
    has = wkr.map(wkr_erst).fillna(False).values
    r = pred["BSW_resid"].values
    re = r[has]; rn = r[~has]
    t, p = stats.ttest_ind(re, rn)
    print(f"  Erst:    mean={re.mean():+.3f}"
          f" sd={re.std():.3f} n={len(re)}")
    print(f"  No-Erst: mean={rn.mean():+.3f}"
          f" sd={rn.std():.3f} n={len(rn)}")
    print(f"  t={t:.3f} p={p:.4f}")


def main():
    df, pred = load_all()
    erst_presence_did(df, pred)
    urne_brief_did(df, pred)
    placebo_pairs(df)
    size_stratified(df, pred)
    bsw_resid_by_erst(df, pred)
    # Save summary
    bsw=_v(df,"BSW").values; z=bsw==0
    ba=pd.to_numeric(df["Bezirksart"],
                     errors="coerce")
    wkr=pd.to_numeric(df["Wahlkreis"],
                      errors="coerce")
    we=df.groupby(wkr).apply(
        lambda x:_ve(x,"BSW").sum()>0)
    h=wkr.map(we).fillna(False).values
    rows=[]
    for l,m in [("erst",h),("no_erst",~h),
                ("urne",(ba==0).values),
                ("brief",(ba==5).values)]:
        rows.append(dict(group=l,n=m.sum(),
            bsw0=z[m].sum(),
            bsw0_rate=z[m].mean()))
    pd.DataFrame(rows).to_csv(
        DATA/"adjacency_did_results.csv",
        index=False)
    print(f"\n  Saved → adjacency_did_results.csv")


if __name__ == "__main__":
    main()
