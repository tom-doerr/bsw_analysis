#!/usr/bin/env python3
"""Neighborhood credibility for top BB anomalies.
Compares each anomaly vs Gemeinde/WKR neighbors
and EW24 BSW baseline."""

import numpy as np, pandas as pd
from pathlib import Path
from zipfile import ZipFile
from bb_utils import estimate_rho, bb_p0

DATA = Path("data")
SEP = "=" * 60


def load_btw25():
    from wahlbezirk_lr import (load_2025_wbz,
        LAND_CODE, validate_totals)
    print("Loading BTW25...")
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen", "Bezirksart"]:
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


def load_ew24_gem():
    """Load EW24 BSW share per Gemeinde."""
    zp = DATA / "ew24_wbz.zip"
    with ZipFile(zp) as zf:
        with zf.open("Wbz_EW24_Ergebnisse.csv") as fh:
            df = pd.read_csv(fh, sep=";", low_memory=False)
    for c in ["Land","Kreis","Gemeinde"]:
        w = 2 if c != "Gemeinde" else 3
        df[c] = df[c].astype(str).str.zfill(w)
    df["gem_key"] = df["Land"]+"_"+df["Kreis"]+"_"+df["Gemeinde"]
    v = "gültig"
    df[v] = pd.to_numeric(df[v],errors="coerce").fillna(0)
    df["BSW"] = pd.to_numeric(df["BSW"],errors="coerce").fillna(0)
    gem = df.groupby("gem_key")[[v,"BSW"]].sum().reset_index()
    s = np.where(gem[v]>0, gem[v], 1)
    gem["ew24_bsw"] = gem["BSW"]/s*100
    return gem[["gem_key","ew24_bsw"]]


def build_gem_key(df):
    """Build Gemeinde key for BTW25 precincts."""
    def z(s, w):
        return (pd.to_numeric(s, errors="coerce")
                .fillna(0).astype(int)
                .astype(str).str.zfill(w))
    l = z(df["Land"], 2)
    k = z(df["Kreis"], 2)
    g = z(df["Gemeinde"], 3)
    return l + "_" + k + "_" + g


def _prep(df, pred, lc):
    g = df["Gültige - Zweitstimmen"].values.astype(float)
    bsw = pd.to_numeric(df["BSW - Zweitstimmen"],
        errors="coerce").fillna(0).values
    bp = np.clip(pred["BSW_pred"].values/100,1e-8,1-1e-8)
    rho = estimate_rho(pred, g)
    p0 = bb_p0(g, bp, rho)
    susp = (bsw==0) & (p0<0.01)
    gk = build_gem_key(df).values
    land = pd.to_numeric(df["Land"],errors="coerce")
    ln = land.map(lc).fillna("").values
    gem = df.get("Gemeindename",
        df.get("Gemeinde",pd.Series([""]*len(df)))).values
    wkr = pd.to_numeric(df.get("Wahlkreis",
        df.iloc[:,0]),errors="coerce").fillna(0).astype(int).values
    return g,bsw,bp,rho,p0,susp,gk,ln,gem,wkr


def _nbr_stats(idx, gk, g, bsw, bp, susp):
    """Neighborhood stats for a precinct's Gemeinde."""
    my_gk = gk[idx]
    nbr = np.where((gk==my_gk))[0]
    nbr = nbr[nbr != idx]  # exclude self
    if len(nbr) == 0:
        return 0, 0, 0, np.nan
    n_nbr = len(nbr)
    n_bsw_gt0 = int((bsw[nbr] > 0).sum())
    n_susp = int(susp[nbr].sum())
    # Median BSW share among neighbors with BSW>0
    shares = bsw[nbr] / np.maximum(g[nbr], 1) * 100
    med = float(np.median(shares[bsw[nbr]>0])) if n_bsw_gt0 else 0
    return n_nbr, n_bsw_gt0, n_susp, med


def analyze(df, pred, lc, ew24):
    g,bsw,bp,rho,p0,susp,gk,ln,gem,wkr=_prep(df,pred,lc)
    ew_map=dict(zip(ew24["gem_key"],ew24["ew24_bsw"]))
    lam=bp*g; idx=np.where(susp)[0]
    idx=idx[np.argsort(lam[idx])[::-1]]
    rows=[]
    for i in idx:
        nn,nb,ns,med=_nbr_stats(i,gk,g,bsw,bp,susp)
        ew=ew_map.get(gk[i],np.nan)
        rows.append(dict(
            land=ln[i],wkr=int(wkr[i]),
            gemeinde=gem[i],lam=round(lam[i],1),
            p0=f"{p0[i]:.2e}",n_nbr=nn,
            nbr_bsw_gt0=nb,nbr_susp=ns,
            nbr_med=round(med,2) if med==med else "",
            ew24=round(ew,2) if ew==ew else "",
            pred=round(bp[i]*100,2)))
    return pd.DataFrame(rows)


def matched_ctrl(g,bsw,bp,susp,gk,ln,n_d=200):
    """Matched non-anomalous control: neighbor BSW>0 frac."""
    rng=np.random.RandomState(42)
    lam=bp*g; bins=[0,5,10,20,50,np.inf]
    lb=np.digitize(lam,bins)
    st=np.array([f"{l}_{b}" for l,b in zip(ln,lb)])
    idx_s=np.where(susp)[0]
    pool=np.where(~susp&(bsw>0))[0]
    fracs=[]
    for _ in range(n_d):
        sel=[]
        for i in idx_s:
            c=pool[st[pool]==st[i]]
            if len(c)>0: sel.append(rng.choice(c))
        if not sel: continue
        tn=tg=0
        for j in sel:
            nb=np.where(gk==gk[j])[0]; nb=nb[nb!=j]
            tn+=len(nb); tg+=(bsw[nb]>0).sum()
        fracs.append(tg/max(tn,1))
    return np.array(fracs)


def main():
    df,pred,lc = load_btw25()
    ew24 = load_ew24_gem()
    print(f"  EW24: {len(ew24)} Gemeinden")
    tbl = analyze(df, pred, lc, ew24)
    print(f"\n{SEP}\nNEIGHBORHOOD CREDIBILITY\n{SEP}")
    print(f"  Anomalies: {len(tbl)}")
    has_nbr = tbl["n_nbr"] > 0
    print(f"  With neighbors: {has_nbr.sum()}")
    gt0 = tbl.loc[has_nbr, "nbr_bsw_gt0"]
    nn = tbl.loc[has_nbr, "n_nbr"]
    frac = (gt0/nn).mean()
    wfrac = gt0.sum()/nn.sum()
    print(f"  Avg frac BSW>0: {frac:.1%}")
    print(f"  Weighted frac: {wfrac:.1%}")
    _ctrl(df, pred, lc)
    _show_top(tbl)
    tbl.to_csv(DATA/"neighborhood_credibility.csv",
               index=False)
    print(f"\n  Saved → neighborhood_credibility.csv")


def _ctrl(df, pred, lc):
    """Run matched control comparison."""
    vals = _prep(df, pred, lc)
    g,bsw,bp,rho,p0,susp,gk,ln,gem,wkr = vals
    cf = matched_ctrl(g,bsw,bp,susp,gk,ln)
    cm = np.median(cf)
    lo,hi = np.percentile(cf,[5,95])
    print(f"  Control frac: med={cm:.1%}"
          f" [{lo:.1%}, {hi:.1%}]")


def _show_top(tbl, k=20):
    """Print top k anomalies with context."""
    print(f"\n  Top {k} by λ:")
    print(f"  {'Land':<4}{'WKR':>4}{'λ':>6}"
          f"{'nbr':>5}{'>0':>4}{'EW24':>6}"
          f"{'pred':>6} Gemeinde")
    for _,r in tbl.head(k).iterrows():
        ew = f"{r.ew24:5.1f}" if r.ew24!="" else "  N/A"
        print(f"  {r.land:<4}{r.wkr:>4}{r.lam:>6.1f}"
              f"{r.n_nbr:>5}{r.nbr_bsw_gt0:>4}"
              f"{ew:>6}{r.pred:>6.2f}"
              f" {r.gemeinde}")


if __name__ == "__main__":
    main()
