#!/usr/bin/env python3
"""Top 200 surviving BB anomalies case file.
Focuses on high-lambda BSW=0 precincts that
survive Beta-Binomial calibration."""

import numpy as np
import pandas as pd
from pathlib import Path
from bb_utils import estimate_rho, bb_p0

DATA = Path("data")
SEP = "=" * 60


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
    return df, pred, LAND_CODE


def compute(df, pred, lc):
    """Compute BB p0 and flag BSW=0."""
    g=df["Gültige - Zweitstimmen"].values.astype(float)
    bsw=pd.to_numeric(df["BSW - Zweitstimmen"],
        errors="coerce").fillna(0).values
    bp=np.clip(pred["BSW_pred"].values/100,1e-8,1-1e-8)
    rho=estimate_rho(pred,g)
    p0=bb_p0(g,bp,rho); lam=bp*g
    land=pd.to_numeric(df["Land"],errors="coerce")
    ln=land.map(lc).fillna("").values
    gem=df.get("Gemeindename",
        df.get("Gemeinde",pd.Series([""]*len(df)))).values
    wkr=pd.to_numeric(df.get("Wahlkreis",
        df.iloc[:,0]),errors="coerce").fillna(0)
    ba=df["Bezirksart"].values.astype(int)
    return g,bsw,bp,rho,p0,lam,ln,gem,wkr.values.astype(int),ba


def build_table(g,bsw,bp,rho,p0,lam,ln,gem,wkr,ba,k=200):
    """Build top-k anomaly table."""
    iz=bsw==0; miss=np.where(iz,lam,0)
    # Only BSW=0 with BB p0 < 0.01
    susp=iz&(p0<0.01)
    idx=np.where(susp)[0]
    order=np.argsort(miss[idx])[::-1][:k]
    sel=idx[order]
    rows=[]
    for i in sel:
        rows.append(dict(
            land=ln[i],wahlkreis=int(wkr[i]),
            gemeinde=gem[i],
            bezirksart={0:"Urne",5:"Brief"}.get(ba[i],str(ba[i])),
            valid_total=int(g[i]),
            bsw_pred_pct=round(bp[i]*100,3),
            lambda_=round(lam[i],2),
            p0_bb=f"{p0[i]:.2e}",
            expected_missing=round(miss[i],1)))
    return pd.DataFrame(rows)


def main():
    df,pred,lc=load_all()
    vals=compute(df,pred,lc)
    g,bsw,bp,rho,p0,lam,ln,gem,wkr,ba=vals
    print(f"\n{SEP}\nTOP ANOMALIES (BB)\n{SEP}")
    print(f"  rho={rho:.6f}")
    susp=(bsw==0)&(p0<0.01)
    print(f"  BB-suspicious: {susp.sum()}")
    tbl=build_table(*vals)
    print(f"  Top {len(tbl)}, total missing:"
          f" {tbl['expected_missing'].sum():,.0f}")
    tbl.to_csv(DATA/"top_anomalies_bb.csv",index=False)
    print(f"  Saved → top_anomalies_bb.csv")


if __name__=="__main__":
    main()
