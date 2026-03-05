#!/usr/bin/env python3
"""RWS brief/urne decomposition.
Uses Repräsentative Wahlstatistik to separate
demographic from counting explanations."""

import numpy as np, pandas as pd
from pathlib import Path

DATA = Path("data")
SEP = "=" * 60
AGES = ["2001-2007","1991-2000","1981-1990",
        "1966-1980","1956-1965","<=1955"]
SEXES = ["m|d|o", "w"]


def load_rws():
    """Load RWS Bezirksart data."""
    return pd.read_csv(
        DATA/"btw25_rws_bst2-ba.csv",
        sep=";", skiprows=12, encoding="utf-8-sig")


def _g(z,ba,sx,ag):
    return z[(z["Bezirksart"]==ba)
        &(z["Geschlecht"]==sx)
        &(z["Geburtsjahresgruppe"]==ag)]


def decompose(df,col="dar. BSW",stim=2):
    z=df[(df["Erst-/Zweitstimme"]==stim)&(df["Land"]=="Bund")]
    u=_g(z,"Urne","Summe","Summe").iloc[0]
    b=_g(z,"Brief","Summe","Summe").iloc[0]
    up=u[col]/u["Summe"]*100; bp=b[col]/b["Summe"]*100
    ns=lambda s:s!="Summe"
    uc=z[(z["Bezirksart"]=="Urne")&z["Geschlecht"].apply(ns)&z["Geburtsjahresgruppe"].apply(ns)]
    bc=z[(z["Bezirksart"]=="Brief")&z["Geschlecht"].apply(ns)&z["Geburtsjahresgruppe"].apply(ns)]
    uw=uc["Summe"].values
    br=bc[col].values/np.maximum(bc["Summe"].values,1)
    exp=(uw*br).sum()/uw.sum()*100
    return up,bp,exp


def cell_gaps(df,col="dar. BSW",stim=2):
    """Per age×sex cell gaps."""
    z=df[(df["Erst-/Zweitstimme"]==stim)&(df["Land"]=="Bund")]
    rows=[]
    for age in AGES:
        for sex in SEXES:
            u=_g(z,"Urne",sex,age)
            b=_g(z,"Brief",sex,age)
            if len(u)==0 or len(b)==0: continue
            u=u.iloc[0]; b=b.iloc[0]
            up=u[col]/u["Summe"]*100
            bp=b[col]/b["Summe"]*100
            rows.append(dict(age=age,sex=sex,
                urne_pct=round(up,3),
                brief_pct=round(bp,3),
                gap=round(up-bp,3)))
    return pd.DataFrame(rows)


def main():
    df = load_rws()
    print(f"\n{SEP}\nRWS BRIEF/URNE DECOMPOSITION\n{SEP}")
    parties = {"BSW":"dar. BSW","FDP":"FDP",
               "Die Linke":"Die Linke","AfD":"AfD"}
    rows = []
    for name,col in parties.items():
        up,bp,exp = decompose(df,col)
        raw=up-bp; demog=up-exp; resid=exp-bp
        print(f"\n  {name}: U={up:.3f} B={bp:.3f}"
              f" raw={raw:+.3f} resid={resid:+.3f}")
        rows.append(dict(party=name,urne=round(up,3),
            brief=round(bp,3),raw=round(raw,3),
            demog=round(demog,3),resid=round(resid,3)))
    pd.DataFrame(rows).to_csv(
        DATA/"rws_decomposition.csv",index=False)
    _cells(df)
    print(f"\n  Saved → rws_decomposition.csv")


def _cells(df):
    cg = cell_gaps(df)
    print(f"\n  BSW per age×sex cell:")
    for _,r in cg.iterrows():
        print(f"    {r.age:12}{r.sex:6}"
              f" U={r.urne_pct:5.2f} B={r.brief_pct:5.2f}"
              f" {r.gap:+.2f}pp")
    cg.to_csv(DATA/"rws_bsw_cells.csv",index=False)


if __name__=="__main__":
    main()
