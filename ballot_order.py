#!/usr/bin/env python3
"""Ballot order analysis: BSW/BD adjacency effect.
Uses official Parteireihenfolge per Land."""

import numpy as np, pandas as pd
from pathlib import Path
from bb_utils import estimate_rho, bb_p0

DATA = Path("data")
SEP = "=" * 60
BD = "BÜNDNIS DEUTSCHLAND"
LAND_COLS = ["SH","MV","HH","NI","HB","BB",
    "ST","BE","NW","SN","HE","TH","RP","BY","BW","SL"]
# Land code → abbreviation
LC = {1:"SH",2:"HH",3:"NI",4:"HB",5:"NW",6:"HE",
      7:"RP",8:"BW",9:"BY",10:"SL",11:"BE",12:"BB",
      13:"MV",14:"SN",15:"ST",16:"TH"}


def load_order():
    """Load ballot order, return {land_abbr: {party: rank}}."""
    df = pd.read_csv(
        DATA/"btw25_parteireihenfolge_laender.csv",
        sep=";", skiprows=7, encoding="utf-8-sig")
    order = {}
    for l in LAND_COLS:
        ranks = {}
        for _, r in df.iterrows():
            v = r.get(l)
            if pd.notna(v):
                try:
                    ranks[r["GruppennameKurz"]] = int(v)
                except (ValueError, TypeError):
                    pass
        order[l] = ranks
    return order


def load_btw25():
    from wahlbezirk_lr import (load_2025_wbz,
        LAND_CODE, validate_totals)
    print("Loading BTW25...")
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen","Bezirksart"]:
        df[c] = pd.to_numeric(
            df[c],errors="coerce").fillna(0)
    df = df[df["Gültige - Zweitstimmen"]>=1]
    df = df.copy().reset_index(drop=True)
    pred = pd.read_csv(
        DATA/"wahlbezirk_lr_predictions.csv",
        low_memory=False)
    assert len(df)==len(pred)
    validate_totals(df)
    return df, pred


def _get_cols(df, pred, order):
    g = df["Gültige - Zweitstimmen"].values.astype(float)
    bsw = pd.to_numeric(df["BSW - Zweitstimmen"],
        errors="coerce").fillna(0).values
    bd = pd.to_numeric(df[f"{BD} - Zweitstimmen"],
        errors="coerce").fillna(0).values
    bp = np.clip(pred["BSW_pred"].values/100,1e-8,1-1e-8)
    rho = estimate_rho(pred, g)
    p0 = bb_p0(g, bp, rho)
    land = pd.to_numeric(df["Land"],errors="coerce")
    ln = land.map(LC).fillna("").values
    return g,bsw,bd,bp,rho,p0,ln


def _order_feats(ln, order):
    n = len(ln)
    dist = np.full(n, 99)
    adj = np.zeros(n, dtype=bool)
    for i in range(n):
        o = order.get(ln[i], {})
        rb = o.get("BSW", 0)
        rd = o.get(BD, 0)
        if rb and rd:
            dist[i] = abs(rb-rd)
            adj[i] = dist[i] == 1
    return dist, adj


def analyze(df, pred, order):
    g,bsw,bd,bp,rho,p0,ln = _get_cols(df,pred,order)
    dist,adj = _order_feats(ln, order)
    susp = (bsw==0) & (p0<0.01)
    lam = bp*g
    print(f"\n{SEP}\nBALLOT ORDER ANALYSIS\n{SEP}")
    print(f"  Adjacent (dist=1): {adj.sum():,}"
          f" ({adj.mean():.1%})")
    print(f"  Non-adjacent: {(~adj).sum():,}")
    # SL is only non-adjacent state
    sl = ln=="SL"
    print(f"  Saarland (dist=2): {sl.sum():,}")
    _susp_by_adj(susp,adj,lam,bsw,bd,g,ln)
    _bd_analysis(bsw,bd,g,adj,susp,lam)
    _save(susp,adj,lam,bd,g,ln)
    return susp,adj,lam


def _susp_by_adj(susp,adj,lam,bsw,bd,g,ln):
    """Suspicious rate: adjacent vs not."""
    print(f"\n  Suspicious zeros:")
    for label,mask in [("Adjacent",adj),
                       ("Non-adj",~adj)]:
        n = mask.sum()
        ns = (susp & mask).sum()
        r = ns/max(n,1)*100
        print(f"    {label}: {ns}/{n} ({r:.3f}%)")


def _bd_analysis(bsw,bd,g,adj,susp,lam):
    """BD share in suspicious vs normal precincts."""
    print(f"\n  BD share (where BSW=0 & susp):")
    bd_s = bd[susp]/np.maximum(g[susp],1)*100
    bd_all = bd/np.maximum(g,1)*100
    print(f"    Susp: med={np.median(bd_s):.2f}%"
          f" mean={bd_s.mean():.2f}%")
    print(f"    All:  med={np.median(bd_all):.2f}%"
          f" mean={bd_all.mean():.2f}%")
    # BD share in BSW=0 vs BSW>0
    z = bsw==0; nz = bsw>0
    bd_z = bd[z]/np.maximum(g[z],1)*100
    bd_nz = bd[nz]/np.maximum(g[nz],1)*100
    print(f"    BSW=0: BD={bd_z.mean():.2f}%")
    print(f"    BSW>0: BD={bd_nz.mean():.2f}%")


def _save(susp,adj,lam,bd,g,ln):
    print(f"\n  Per-Land adjacency:")
    print(f"  {'Land':<4}{'adj':>5}{'susp':>6}"
          f"{'rate%':>7}{'BD_s%':>6}")
    rows = []
    for l in sorted(set(ln)):
        if not l: continue
        m=ln==l; a=adj[m].all()
        ns=susp[m].sum(); nt=m.sum()
        r=ns/max(nt,1)*100
        bds=bd[m&susp]/np.maximum(g[m&susp],1)*100
        bm=bds.mean() if len(bds)>0 else 0
        print(f"  {l:<4}{'Y' if a else 'N':>5}"
              f"{ns:>6}{r:>7.3f}{bm:>6.2f}")
        rows.append(dict(land=l,adj=a,n_susp=ns,
            n_total=nt,rate=round(r,4)))
    pd.DataFrame(rows).to_csv(
        DATA/"ballot_order_analysis.csv",index=False)
    print(f"\n  Saved → ballot_order_analysis.csv")


def main():
    order = load_order()
    print(f"\n{SEP}\nBALLOT ORDER\n{SEP}")
    for l in LAND_COLS:
        rb = order[l].get("BSW", 0)
        rd = order[l].get(BD, 0)
        d = abs(rb-rd) if rb and rd else 99
        print(f"  {l}: BD={rd} BSW={rb} d={d}")
    df, pred = load_btw25()
    analyze(df, pred, order)


if __name__ == "__main__":
    main()
