#!/usr/bin/env python3
"""Recount triage funnel: rank registry precincts by
conservative missing-vote lower bound for targeted recount.

Funnel: 3,578 flagged -> 300 by missing-vote LB ->
100 with overlay scoring -> top 20 actionable.

Output: data/recount_targets.csv (top 100)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import betaln

from wahlbezirk_lr import (load_2025_wbz,
    LAND_CODE, validate_totals)
from bb_utils import estimate_rho, bb_p0

DATA = Path("data")
SEP = "=" * 60
BD = "BÜNDNIS DEUTSCHLAND"
TIER1 = 300
TIER2 = 100


def load_all():
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


def bb_cdf_k(k, n, p, rho):
    """P(X<=k) under BB(n,p,rho)."""
    phi=max(1/rho-1,1e-6)
    a=np.maximum(p*phi,1e-10)
    b=np.maximum((1-p)*phi,1e-10)
    ni=n.astype(int); ki=k.astype(int)
    mx=int(ki.max())+1
    lp=betaln(a,b+ni)-betaln(a,b)
    cum=np.exp(np.clip(lp,-700,0))
    res=np.zeros(len(n))
    done=ki==0; res[done]=cum[done]
    if done.all(): return res
    for j in range(1,mx):
        lp+=(np.log(np.maximum(ni-j+1,1e-300))
            -np.log(j)+np.log(a+j-1)
            -np.log(np.maximum(b+ni-j,1e-300)))
        cum+=np.exp(np.clip(lp,-700,0))
        nw=(~done)&(ki==j)
        res[nw]=cum[nw]; done|=nw
        if done.all(): break
    res[~done]=cum[~done]
    return np.clip(res,0,1)


def conservative_missing_lb(bsw, g, bp, rho):
    """Conservative lower bound on missing votes.
    BSW=0: lambda. BSW>0: lambda-bsw if P(<=obs)<1%."""
    lam = bp * g
    miss = np.zeros(len(g))
    z = bsw == 0
    miss[z] = lam[z]
    pos = (~z) & (bsw < lam)
    if pos.any():
        pcdf = bb_cdf_k(
            bsw[pos], g[pos], bp[pos], rho)
        sig = pcdf < 0.01
        idx = np.where(pos)[0][sig]
        miss[idx] = lam[idx] - bsw[idx]
    return np.maximum(miss, 0)


def _votes(df, party):
    col = f"{party} - Zweitstimmen"
    return pd.to_numeric(
        df[col], errors="coerce").fillna(0)


def build_base(df, pred):
    """Build base table with conservative metrics."""
    g = df["Gültige - Zweitstimmen"].values.astype(float)
    bsw = _votes(df, "BSW").values.astype(float)
    bd = _votes(df, BD).values.astype(float)
    bp = np.clip(pred["BSW_pred"].values/100, 1e-8, 1-1e-8)
    rho = estimate_rho(pred, g)
    p0 = bb_p0(g, bp, rho)
    lam = bp * g
    miss_lb = conservative_missing_lb(bsw, g, bp, rho)
    pcdf = np.ones(len(g))
    cand = bsw < lam
    if cand.any():
        pcdf[cand] = bb_cdf_k(
            bsw[cand].astype(int), g[cand],
            bp[cand], rho)
    land = pd.to_numeric(df["Land"], errors="coerce")
    ln = land.map(LAND_CODE).fillna("").values
    wkr = pd.to_numeric(df.get("Wahlkreis",
        df.iloc[:, 0]), errors="coerce"
        ).fillna(0).astype(int).values
    gem = df.get("Gemeindename", df.get(
        "Gemeinde", pd.Series([""] * len(df)))).values
    ba = df["Bezirksart"].values.astype(int)
    wbz = df.get("Wahlbezirk",
        pd.Series([""] * len(df))).values
    bd_share = np.where(g > 0, bd/g*100, 0)
    bd_pctile = np.zeros(len(df))
    for lv in land.dropna().unique():
        m = (land == lv).values
        if m.sum() < 10: continue
        ranks = pd.Series(bd_share[m]).rank(pct=True)
        bd_pctile[m] = ranks.values
    return pd.DataFrame({
        "land": ln, "wahlkreis": wkr,
        "gemeinde": gem, "wbz": wbz,
        "bezirksart": ba,
        "valid_total": g.astype(int),
        "bsw_votes": bsw.astype(int),
        "bd_votes": bd.astype(int),
        "bsw_pred_pct": np.round(bp*100, 3),
        "lambda": np.round(lam, 2),
        "p0_bb": p0,
        "p_tail": np.round(pcdf, 6),
        "miss_lb": np.round(miss_lb, 1),
        "bd_pctile": np.round(bd_pctile, 4),
    }), rho


def load_overlays():
    """Load registry, affidavit, corrections."""
    reg = aff = oc = None
    rp = DATA / "evidence_registry.csv"
    if rp.exists():
        reg = pd.read_csv(rp, low_memory=False)
    ap = DATA / "affidavit_analysis.csv"
    if ap.exists():
        aff = pd.read_csv(ap, low_memory=False)
    op = DATA / "official_corrections.csv"
    if op.exists():
        oc = pd.read_csv(op, low_memory=False)
    return reg, aff, oc


def overlay_score(row, reg, aff):
    """Score from flag/affidavit/BD overlap."""
    s = 0.0
    if reg is not None:
        m = ((reg["wahlkreis"] == row["wahlkreis"])
             & (reg["bsw_votes"] == row["bsw_votes"])
             & (reg["valid_total"] == row["valid_total"]))
        if m.any():
            flags = str(reg.loc[m.idxmax(), "flags"])
            s += flags.count("|") + 1
    if aff is not None:
        m = ((aff["wkr"] == row["wahlkreis"])
             & (aff["bsw"] == row["bsw_votes"])
             & (aff["bd"] == row["bd_votes"]))
        if m.any(): s += 5
    if row["bd_pctile"] > 0.95: s += 1
    return s


def funnel(base, reg, aff):
    """Apply triage funnel."""
    cand = base[base["miss_lb"] > 0].copy()
    print(f"  Candidates (miss_lb > 0): {len(cand)}")
    t1 = cand.nlargest(TIER1, "miss_lb").copy()
    print(f"  Tier 1 (top {TIER1}): {len(t1)},"
          f" miss_lb={t1['miss_lb'].sum():,.0f}")
    t1["overlay_score"] = t1.apply(
        lambda r: overlay_score(r, reg, aff), axis=1)
    t1["composite"] = (t1["miss_lb"]
        * (1 + 0.5 * t1["overlay_score"]))
    t2 = t1.nlargest(TIER2, "composite").copy()
    print(f"  Tier 2 (top {TIER2}): {len(t2)},"
          f" miss_lb={t2['miss_lb'].sum():,.0f}")
    return t2


def _summary(targets):
    print(f"\n{SEP}")
    print("RECOUNT TARGET SUMMARY")
    print(SEP)
    print(f"  Targets: {len(targets)}")
    print(f"  Total miss_lb:"
          f" {targets['miss_lb'].sum():,.0f}")
    nz = (targets["bsw_votes"] == 0).sum()
    print(f"  BSW=0: {nz}, BSW>0: {len(targets)-nz}")
    print(f"\n  {'Land':<4}{'n':>4}{'miss_lb':>8}"
          f"{'BSW=0':>6}")
    for land in sorted(targets["land"].unique()):
        m = targets["land"] == land
        n = m.sum()
        ml = targets.loc[m, "miss_lb"].sum()
        z = ((targets["bsw_votes"]==0) & m).sum()
        print(f"  {land:<4}{n:>4}{ml:>8.0f}{z:>6}")
    print(f"\n  Top 20:")
    print(f"  {'Land':<4}{'WKR':>4}{'BSW':>4}"
          f"{'BD':>4}{'Valid':>5}{'miss':>7}"
          f"{'p':>8}{'ov':>3} Gemeinde")
    for _, r in targets.head(20).iterrows():
        print(f"  {r.land:<4}{r.wahlkreis:>4}"
              f"{r.bsw_votes:>4}{r.bd_votes:>4}"
              f"{r.valid_total:>5}{r.miss_lb:>7.1f}"
              f"{r.p_tail:>8.4f}"
              f"{r.overlay_score:>3.0f}"
              f" {r.gemeinde}")


def main():
    df, pred = load_all()
    base, rho = build_base(df, pred)
    print(f"  rho={rho:.6f}")
    reg, aff, oc = load_overlays()
    targets = funnel(base, reg, aff)
    targets = targets.sort_values(
        "composite", ascending=False)
    _summary(targets)
    out = DATA / "recount_targets.csv"
    targets.to_csv(out, index=False)
    print(f"\n  Saved {len(targets)} → {out}")


if __name__ == "__main__":
    main()
