#!/usr/bin/env python3
"""Beta-Binomial zero calibration: accounts for model
uncertainty when computing expected zeros."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import betaln

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
    df = df[df["Gültige - Zweitstimmen"] >= 1]
    df = df.copy().reset_index(drop=True)
    pred = pd.read_csv(
        DATA / "wahlbezirk_lr_predictions.csv",
        low_memory=False)
    assert len(df) == len(pred)
    validate_totals(df)
    return df, pred


def _v(df, p):
    c = f"{p} - Zweitstimmen"
    return pd.to_numeric(
        df[c], errors="coerce").fillna(0)


def _g(df):
    return df["Gültige - Zweitstimmen"
              ].values.astype(float)


def estimate_rho(pred, g):
    """Estimate ρ from CV residuals."""
    p=np.clip(pred["BSW_pred"].values/100,1e-6,1-1e-6)
    r2=(pred["BSW_resid"].values/100)**2
    n=np.maximum(g,1); obs=np.mean(r2)
    base=np.mean(p*(1-p)/n)
    ext=np.mean(p*(1-p)*(n-1)/n)
    rho=min(max((obs-base)/ext,1e-6),0.5) if ext>0 else 1e-6
    print(f"  ρ={rho:.6f} overdispersion={obs/base:.1f}x")
    return rho


def bb_p0(n, p, rho):
    """Beta-Binomial P(X=0). Uses log Beta fn."""
    phi=max(1/rho-1,1e-6)
    a=np.maximum(p*phi,1e-10)
    b=np.maximum((1-p)*phi,1e-10)
    lp=betaln(a,b+n)-betaln(a,b)
    return np.exp(np.clip(lp,-700,0))


def bn_p0(n, p):
    """Binomial P(X=0) = (1-p)^n."""
    return np.power(1-p, n)


def _prep(df,pred):
    g=_g(df);bsw=_v(df,"BSW").values
    bp=np.clip(pred["BSW_pred"].values/100,1e-8,1-1e-8)
    return g,bsw,bp,bp*g


def calibrate(df,pred,rho):
    """Bin by λ, compare Binom vs BetaBinom."""
    print(f"\n{SEP}\nZERO CALIB (BetaBinom)\n{SEP}")
    g,bsw,bp,lam=_prep(df,pred)
    iz=bsw==0;p0b=bn_p0(g,bp);p0bb=bb_p0(g,bp,rho)
    bs=[0,1,3,5,10,20,50,np.inf]
    lb=["0-1","1-3","3-5","5-10","10-20","20-50","50+"]
    ix=np.clip(np.digitize(lam,bs)-1,0,6);rows=[]
    for i,l in enumerate(lb):
        m=ix==i;n=m.sum()
        if n==0: continue
        o=iz[m].sum();eb=p0b[m].sum();ebb=p0bb[m].sum()
        rows.append(dict(bin=l,n=n,obs=o,
            exp_bn=round(eb,1),exp_bb=round(ebb,1),
            exc_bn=round(o-eb,1),exc_bb=round(o-ebb,1)))
    return pd.DataFrame(rows)


def controls(df,pred,rho):
    """Same calibration for FDP/Linke as controls."""
    print(f"\n{SEP}\nCONTROL COMPARISON\n{SEP}")
    g=_g(df)
    for party in ["BSW","FDP","Die Linke"]:
        v=_v(df,party).values;iz=v==0
        bp=np.clip(pred[f"{party}_pred"].values/100,
                   1e-8,1-1e-8)
        lam=bp*g;p0bb=bb_p0(g,bp,rho)
        bs=[0,1,3,5,10,20,50,np.inf]
        lb=["0-1","1-3","3-5","5-10",
            "10-20","20-50","50+"]
        ix=np.clip(np.digitize(lam,bs)-1,0,6)
        print(f"\n  {party}:")
        for i,l in enumerate(lb):
            m=ix==i
            if m.sum()==0: continue
            o=iz[m].sum();e=p0bb[m].sum()
            print(f"  {l:<6}{o:>5}"
                  f" {e:>7.1f} {o-e:>+7.1f}")


def per_land(df,pred,rho):
    """Per-Land calibration."""
    print(f"\n{SEP}\nPER-LAND CALIBRATION\n{SEP}")
    g,bsw,bp,_=_prep(df,pred)
    iz=bsw==0;p0b=bn_p0(g,bp);p0bb=bb_p0(g,bp,rho)
    land=pd.to_numeric(df["Land"],errors="coerce")
    rows=[]
    for lv in sorted(land.dropna().unique()):
        m=(land==lv).values
        nm=LAND_CODE.get(int(lv),str(lv))
        o=iz[m].sum();eb=p0b[m].sum();ebb=p0bb[m].sum()
        rows.append(dict(land=nm,n=m.sum(),obs=o,
            exp_bn=round(eb,1),exp_bb=round(ebb,1),
            exc_bn=round(o-eb,1),exc_bb=round(o-ebb,1)))
    return pd.DataFrame(rows)


def main():
    df,pred=load_all()
    g=_g(df);rho=estimate_rho(pred,g)
    cal=calibrate(df,pred,rho)
    lcal=per_land(df,pred,rho)
    controls(df,pred,rho)
    cal.to_csv(DATA/"zero_calibration_betabinom.csv",
               index=False)
    lcal.to_csv(DATA/"zero_calib_betabinom_land.csv",
                index=False)
    tb=cal["exc_bn"].sum();tbb=cal["exc_bb"].sum()
    print(f"\n  Excess Binom: {tb:+.0f}")
    print(f"  Excess BetaBinom: {tbb:+.0f}")
    r=tbb/max(tb,1)*100
    print(f"  Survival: {r:.0f}% survives uncertainty")
    print(f"  Saved → zero_calibration_betabinom.csv")


if __name__=="__main__":
    main()
