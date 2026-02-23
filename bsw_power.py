#!/usr/bin/env python3
"""Power analysis: can forensic tests detect
a 9,529-vote BSW miscount?"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chisquare, skew, kurtosis

DATA = Path("data")
SEP = "=" * 60
DEFICIT = 9_529

SCENARIOS = [
    (9529, 1, "9529x1"),
    (1906, 5, "1906x5"),
    (953, 10, "953x10"),
    (95, 100, "95x100"),
]
N_REPS = 20
BENFORD_EXP = np.array([.120,.114,.109,.104,
    .100,.097,.093,.090,.088,.085])


def load_data():
    """Load predictions + raw data."""
    from wahlbezirk_lr import load_2025_wbz
    print("Loading data...")
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen","Bezirksart"]:
        df[c] = pd.to_numeric(
            df[c], errors="coerce").fillna(0)
    df = df[df["Gültige - Zweitstimmen"]>=1].copy()
    df = df.reset_index(drop=True)
    pred = pd.read_csv(
        DATA/"wahlbezirk_lr_predictions.csv")
    assert len(df)==len(pred)
    print(f"  {len(df)} precincts")
    return df, pred


def inject(pred, df, n_aff, vpre, rng):
    """Adjust residuals to simulate miscount."""
    p = pred.copy()
    gz = df["Gültige - Zweitstimmen"].values
    bsw = pd.to_numeric(
        df["BSW - Zweitstimmen"],
        errors="coerce").fillna(0).values
    elig = np.where((df["Bezirksart"]==0)
                    & (bsw >= vpre))[0]
    n = min(n_aff, len(elig))
    idx = rng.choice(elig, n, replace=False)
    delta = np.zeros(len(p))
    delta[idx] = vpre / np.maximum(gz[idx], 1) * 100
    p["BSW_resid"] = p["BSW_resid"] - delta
    return p


def _benford2(votes):
    v = votes[votes >= 10].astype(float)
    lg = np.floor(np.log10(np.maximum(v, 1)))
    d2 = (v//10**(np.maximum(lg-1,0))).astype(int)%10
    return np.array([(d2==i).sum() for i in range(10)])


def test_battery(df, pred):
    """Run 5 key forensic tests."""
    r=pred["BSW_resid"].values
    gz=df["Gültige - Zweitstimmen"].values
    bsw_v=(r/100*gz+pred["BSW_pred"].values/100*gz)
    bsw_v=np.maximum(bsw_v,0).astype(int)
    obs=_benford2(bsw_v)
    chi2,pv=chisquare(obs,BENFORD_EXP*obs.sum())
    sk=skew(r); ku=kurtosis(r,fisher=False)
    wkr=pd.to_numeric(df["Wahlkreis"],
        errors="coerce")
    wm=pd.Series(r).groupby(wkr.values).mean()
    nf=(((wm-wm.mean())/wm.std())<-2).sum()
    nfrac=(r<-1).mean()
    return {"benford_chi2":chi2,"benford_p":pv,
        "skew":sk,"kurtosis":ku,
        "geo_flagged":nf,"neg1pp_frac":nfrac}


def _detect(base,t):
    d={}
    d["skew"]=t["skew"]<base["skew"]-0.05
    d["benford"]=t["benford_chi2"]>base["benford_chi2"]*1.5
    d["geo"]=t["geo_flagged"]>base["geo_flagged"]+2
    d["neg_frac"]=t["neg1pp_frac"]>base["neg1pp_frac"]*1.1
    return d

TESTS=["skew","benford","geo","neg_frac"]

def power_analysis(df,pred):
    base=test_battery(df,pred)
    print(f"\nBase: chi2={base['benford_chi2']:.1f}"
          f" skew={base['skew']:+.3f}")
    rows=[]
    for na,vp,lab in SCENARIOS:
        print(f"\n  {lab}:")
        ct={k:0 for k in TESTS}
        for rep in range(N_REPS):
            rng=np.random.RandomState(rep)
            pm=inject(pred,df,na,vp,rng)
            d=_detect(base,test_battery(df,pm))
            for k in TESTS:
                if d[k]:ct[k]+=1
        for k in TESTS:
            r=ct[k]/N_REPS
            rows.append(dict(scenario=lab,
                test=k,rate=r))
            print(f"    {k:<10} {r:.0%}")
    return pd.DataFrame(rows)


def main():
    df,pred=load_data()
    print(f"\n{SEP}\nPOWER ANALYSIS: Detection of "
          f"{DEFICIT:,}-vote miscount\n{SEP}")
    r=power_analysis(df,pred)
    r.to_csv(DATA/"power_analysis.csv",index=False)
    print(f"\nSaved power_analysis.csv")


if __name__=="__main__":
    main()
