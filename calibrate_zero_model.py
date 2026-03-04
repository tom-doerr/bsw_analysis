#!/usr/bin/env python3
"""Calibrate BSW zero-vote model: bin by λ,
compare obs vs exp zero rates."""

import numpy as np
import pandas as pd
from pathlib import Path
from wahlbezirk_lr import (load_2025_wbz,
    LAND_CODE, validate_totals)

DATA = Path("data")
SEP = "=" * 60


def load_all():
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


def _v(df, p):
    c = f"{p} - Zweitstimmen"
    return pd.to_numeric(
        df[c], errors="coerce").fillna(0)


def _g(df):
    return df["Gültige - Zweitstimmen"
              ].values.astype(float)


def _prep(df, pred):
    g=_g(df); bsw=_v(df,"BSW").values
    bp=np.clip(pred["BSW_pred"].values/100,
               1e-8,1-1e-8)
    return g,bsw,bp,bp*g,np.power(1-bp,g)


def calibrate(df, pred):
    """Bin by λ, compare obs vs exp."""
    print(f"\n{SEP}\nZERO-VOTE CALIBRATION\n{SEP}")
    g,bsw,bp,lam,p0=_prep(df,pred)
    iz=bsw==0
    bs=[0,1,3,5,10,20,50,np.inf]
    lb=["0-1","1-3","3-5","5-10",
        "10-20","20-50","50+"]
    ix=np.clip(np.digitize(lam,bs)-1,0,6)
    rows=[]
    for i,l in enumerate(lb):
        m=ix==i; n=m.sum()
        if n==0: continue
        o=iz[m].sum(); e=p0[m].sum()
        rows.append(dict(bin=l,n=n,obs=o,
            exp=round(e,1),
            excess=round(o-e,1)))
        print(f"  {l:<6}{n:>6}{o:>5}"
              f" {e:>7.1f}{o-e:>+7.1f}")
    return pd.DataFrame(rows)


def per_land(df, pred):
    """Per-Land calibration."""
    print(f"\n{SEP}\nPER-LAND CALIBRATION\n{SEP}")
    g,bsw,bp,lam,p0=_prep(df,pred)
    iz=bsw==0
    land=pd.to_numeric(df["Land"],
                       errors="coerce")
    for lv in sorted(land.dropna().unique()):
        m=(land==lv).values
        nm=LAND_CODE.get(int(lv),str(lv))
        o=iz[m].sum(); e=p0[m].sum()
        print(f"  {nm:<4}{m.sum():>6}"
              f" {o:>5}{e:>7.1f}{o-e:>+7.1f}")


def main():
    df,pred=load_all()
    cal=calibrate(df,pred)
    per_land(df,pred)
    cal.to_csv(DATA/"zero_calibration.csv",
               index=False)
    print(f"\n  Saved → zero_calibration.csv")


if __name__=="__main__":
    main()
