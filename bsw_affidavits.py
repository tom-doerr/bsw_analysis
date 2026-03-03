#!/usr/bin/env python3
"""Cross-reference sworn statements with evidence
registry and statistical model predictions."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson

from wahlbezirk_lr import (load_2025_wbz, LAND_CODE,
                            validate_totals)

DATA = Path("data")
SEP = "=" * 60
BD = "BÜNDNIS DEUTSCHLAND"

# Known affidavit-supported cases from BSW filings.
# 8 eidesstattliche Versicherungen filed with BVerfG.
# Specific precinct IDs not yet public; these are
# from the Lipinski Erwiderung where BSW=0 & BD high.
# Update this list as details become available.
AFFIDAVIT_CASES = [
    {"id": 1, "land": "SH", "wahlkreis": 7,
     "gemeinde": "Wedel", "bsw": 0, "bd": 31,
     "source": "Lipinski Erwiderung"},
    {"id": 2, "land": "NI", "wahlkreis": 52,
     "gemeinde": "Katlenburg", "bsw": 0, "bd": 16,
     "source": "Lipinski Erwiderung"},
    {"id": 3, "land": "SN", "wahlkreis": 157,
     "gemeinde": "Reinhardtsdorf", "bsw": 0, "bd": 12,
     "source": "Lipinski Erwiderung"},
]


def load_all():
    print("Loading data...")
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen",
              "Wahlberechtigte (A)",
              "Bezirksart"]:
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


def _g(df):
    return df["Gültige - Zweitstimmen"].values.astype(float)


def _match_one(c, df, bsw, bd, wkr, gem, bp, g, p0, pred):
    m = (wkr==c["wahlkreis"]) & (bsw==0)
    m &= gem.str.contains(c["gemeinde"],
                          case=False, na=False)
    if c.get("bd"):
        m &= bd == c["bd"]
    idx = df.index[m]
    if len(idx)==0:
        print(f"  #{c['id']}: NO MATCH")
        return None
    i = idx[0]
    print(f"  #{c['id']}: {c['land']}"
          f" BD={int(bd[i])} P(0)={p0[i]:.2e}")
    return dict(id=c["id"], land=c["land"],
        wkr=c["wahlkreis"], gem=gem.iloc[i],
        bsw=int(bsw[i]), bd=int(bd[i]),
        valid=int(g[i]), pred=bp[i]*100,
        p0=p0[i], rz=pred["BSW_resid"].iloc[i])


def match_to_precincts(cases, df, pred):
    """Match affidavit cases to actual precincts."""
    print(f"\n{SEP}")
    print("AFFIDAVIT MATCHING")
    print(SEP)
    g=_g(df); bsw=_v(df,"BSW").values
    bd=_v(df,BD).values
    wkr=pd.to_numeric(df["Wahlkreis"],
                       errors="coerce")
    gem=df.get("Gemeindename",
               pd.Series([""]*len(df)))
    bp=pred["BSW_pred"].values/100
    lam=np.maximum(bp*g, 1e-6)
    p0=np.exp(-lam)
    matched = []
    for c in cases:
        r = _match_one(c, df, bsw, bd, wkr,
                       gem, bp, g, p0, pred)
        if r: matched.append(r)
    return pd.DataFrame(matched)


def extremeness(matched, df, pred):
    """How extreme are affidavit precincts?"""
    print(f"\n{SEP}")
    print("STATISTICAL EXTREMENESS")
    print(SEP)
    g=_g(df); bd=_v(df,BD).values
    land=pd.to_numeric(df["Land"],errors="coerce")
    bd_share = np.where(g>0, bd/g*100, 0)
    for _, r in matched.iterrows():
        lv = [k for k,v in LAND_CODE.items()
              if v==r.land]
        if not lv: continue
        lm = (land==lv[0]).values
        pct = (bd_share[lm] <= r.bd/r.valid*100
               ).mean() * 100
        print(f"  #{r.id} {r.land}: BD={r.bd}"
              f" → {pct:.1f}th percentile"
              f" P(BSW=0)={r.p0:.2e}")


def registry_overlap(matched):
    """Cross-reference with evidence registry."""
    print(f"\n{SEP}")
    print("REGISTRY OVERLAP")
    print(SEP)
    reg_path = DATA / "evidence_registry.csv"
    if not reg_path.exists():
        print("  No registry found — run"
              " evidence_registry.py first")
        return
    reg = pd.read_csv(reg_path)
    for _, r in matched.iterrows():
        m = (reg["wahlkreis"]==r.wkr) & \
            (reg["bsw_votes"]==0) & \
            (reg["bd_votes"]==r.bd)
        if m.any():
            flags = reg.loc[m.idxmax(), "flags"]
            print(f"  #{r.id}: FOUND in registry"
                  f" flags={flags}")
        else:
            print(f"  #{r.id}: NOT in registry")


def population_compare(matched, df, pred):
    """Compare affidavit precincts vs all BSW=0."""
    print(f"\n{SEP}")
    print("POPULATION COMPARISON")
    print(SEP)
    g=_g(df); bsw=_v(df,"BSW").values
    bd=_v(df,BD).values
    bp=pred["BSW_pred"].values/100
    all_z = bsw==0
    nz = all_z.sum()
    bd_in_z = bd[all_z].mean()
    print(f"  All BSW=0: n={nz}"
          f" mean_BD={bd_in_z:.1f}")
    if len(matched)>0:
        a_bd = matched["bd"].mean()
        a_p0 = matched["p0"].mean()
        print(f"  Affidavit: n={len(matched)}"
              f" mean_BD={a_bd:.1f}"
              f" mean_P(0)={a_p0:.2e}")
        print(f"  → Affidavit BD {a_bd/bd_in_z:.1f}x"
              f" higher than avg BSW=0")


def main():
    df, pred = load_all()
    matched = match_to_precincts(
        AFFIDAVIT_CASES, df, pred)
    if len(matched) == 0:
        print("  No matches found")
        return
    extremeness(matched, df, pred)
    registry_overlap(matched)
    population_compare(matched, df, pred)
    matched.to_csv(
        DATA/"affidavit_analysis.csv", index=False)
    print(f"\n  Saved → affidavit_analysis.csv")


if __name__ == "__main__":
    main()
