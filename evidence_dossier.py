#!/usr/bin/env python3
"""Per-precinct evidence dossier for BB anomalies.
Merges: registry, neighborhood, ballot order,
RWS, EW24 baseline."""

import json
import numpy as np, pandas as pd
from pathlib import Path
from zipfile import ZipFile
from bb_utils import estimate_rho, bb_p0

DATA = Path("data")
SEP = "=" * 60
BD = "BÜNDNIS DEUTSCHLAND"


def load_btw25():
    from wahlbezirk_lr import (load_2025_wbz,
        LAND_CODE, validate_totals)
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen",
              "Bezirksart"]:
        df[c] = pd.to_numeric(
            df[c], errors="coerce").fillna(0)
    df = df[df["Gültige - Zweitstimmen"] >= 1]
    df = df.copy().reset_index(drop=True)
    pred = pd.read_csv(
        DATA/"wahlbezirk_lr_predictions.csv",
        low_memory=False)
    assert len(df) == len(pred)
    validate_totals(df)
    return df, pred, LAND_CODE


def _gem_key(df):
    def z(s, w):
        return (pd.to_numeric(s, errors="coerce")
                .fillna(0).astype(int)
                .astype(str).str.zfill(w))
    return (z(df["Land"], 2) + "_"
        + z(df["Kreis"], 2) + "_"
        + z(df["Gemeinde"], 3))


def load_ew24_gem():
    zp = DATA / "ew24_wbz.zip"
    with ZipFile(zp) as zf:
        with zf.open("Wbz_EW24_Ergebnisse.csv") as fh:
            d = pd.read_csv(fh,sep=";",low_memory=False)
    for c in ["Land","Kreis","Gemeinde"]:
        w = 2 if c != "Gemeinde" else 3
        d[c] = d[c].astype(str).str.zfill(w)
    d["gk"] = d["Land"]+"_"+d["Kreis"]+"_"+d["Gemeinde"]
    v = "gültig"
    d[v] = pd.to_numeric(d[v],errors="coerce").fillna(0)
    d["BSW"] = pd.to_numeric(d["BSW"],errors="coerce").fillna(0)
    g = d.groupby("gk")[[v,"BSW"]].sum().reset_index()
    s = np.where(g[v]>0, g[v], 1)
    g["pct"] = g["BSW"]/s*100
    # Land-level fallback for city-states
    dl = d.groupby("Land")[[v,"BSW"]].sum().reset_index()
    sl = np.where(dl[v]>0, dl[v], 1)
    dl["pct"] = dl["BSW"]/sl*100
    lf = dict(zip(dl["Land"],dl["pct"]))
    lv = dict(zip(dl["Land"],dl["BSW"]))
    lt = dict(zip(dl["Land"],dl[v]))
    return (dict(zip(g["gk"],g["pct"])),
        dict(zip(g["gk"],g["BSW"])),
        dict(zip(g["gk"],g[v])),lf,lv,lt)


def load_ballot_order():
    from ballot_order import load_order, LC
    return load_order(), LC


def ballot_ctx(land, order):
    o = order.get(land, {})
    rb, rd = o.get("BSW",0), o.get(BD,0)
    if rb and rd:
        d = abs(rb-rd)
        return rb, rd, d, d==1
    return rb, rd, None, None


def _merge_reg(r, reg):
    """Merge registry fields into dossier row."""
    r["flags"]=""; r["recount_status"]=""
    r["source"]=""; r["claim"]=""
    if reg is None: return
    m = (reg["land"]==r["land"])&(reg["wahlkreis"]==r["wkr"])
    m = m&(reg["bsw_votes"]==r["bsw"])
    m = m&(reg["valid_total"]==r["valid"])
    hits = reg[m]
    if len(hits)==0: return
    h = hits.iloc[0]
    r["flags"]=h.get("flags","")
    r["recount_status"]=h.get("recount_status","")
    r["source"]=h.get("source","")
    r["claim"]=h.get("claim","")


def _load_registry():
    p = DATA/"evidence_registry.csv"
    if not p.exists(): return None
    return pd.read_csv(p,low_memory=False)


def build_core(df, pred, lc):
    g=df["Gültige - Zweitstimmen"].values.astype(float)
    bsw=pd.to_numeric(df["BSW - Zweitstimmen"],
        errors="coerce").fillna(0).values
    bd=pd.to_numeric(df[f"{BD} - Zweitstimmen"],
        errors="coerce").fillna(0).values
    bp=np.clip(pred["BSW_pred"].values/100,1e-8,1-1e-8)
    rho=estimate_rho(pred,g); p0=bb_p0(g,bp,rho)
    susp=(bsw==0)&(p0<0.01)
    land=pd.to_numeric(df["Land"],errors="coerce")
    ln=land.map(lc).fillna("").values
    gem=df.get("Gemeindename",
        df.get("Gemeinde",pd.Series([""]*len(df)))).values
    wkr=pd.to_numeric(df.get("Wahlkreis",
        df.iloc[:,0]),errors="coerce").fillna(0).astype(int).values
    ba=df["Bezirksart"].values.astype(int)
    gk=_gem_key(df).values
    wbz=df.get("Wahlbezirk",pd.Series([""]*len(df))).values
    lc_raw=pd.to_numeric(df["Land"],errors="coerce").fillna(0).astype(int).values
    kc=pd.to_numeric(df["Kreis"],errors="coerce").fillna(0).astype(int).values
    gc=pd.to_numeric(df["Gemeinde"],errors="coerce").fillna(0).astype(int).values
    return dict(g=g,bsw=bsw,bd=bd,bp=bp,rho=rho,
        p0=p0,lam=bp*g,susp=susp,ln=ln,gem=gem,
        wkr=wkr,ba=ba,gk=gk,wbz=wbz,
        lc_raw=lc_raw,kc=kc,gc=gc)


def nbr_ctx(idx, c):
    gk,g,bsw = c["gk"],c["g"],c["bsw"]
    nb = np.where(gk==gk[idx])[0]
    nb = nb[nb!=idx]
    if len(nb)==0:
        return 0,0,None,None
    sh = bsw[nb]/np.maximum(g[nb],1)*100
    gt = bsw[nb]>0
    med = round(float(np.median(sh[gt])),2) if gt.any() else 0.0
    return int(len(nb)),int(gt.sum()),med,int(bsw[nb].max())


def build_dossier(df, pred, lc):
    print("Building dossier...")
    c = build_core(df, pred, lc)
    ew_pct,ew_v,ew_t,ew_lp,ew_lv,ew_lt = load_ew24_gem()
    order, _ = load_ballot_order()
    reg = _load_registry()
    rws_p = DATA/"rws_decomposition.csv"
    rws = pd.read_csv(rws_p) if rws_p.exists() else None
    bsw_rws = None
    if rws is not None:
        row = rws[rws["party"]=="BSW"]
        if len(row): bsw_rws = row.iloc[0]
    idx = np.where(c["susp"])[0]
    idx = idx[np.argsort(c["lam"][idx])[::-1]]
    print(f"  {len(idx)} BB-suspicious precincts")
    rows = []
    for i in idx:
        rows.append(_one_row(i, c, ew_pct, ew_v,
            ew_t, ew_lp, ew_lv, ew_lt,
            order, bsw_rws, reg))
    return pd.DataFrame(rows)


def _one_row(i, c, ew_pct, ew_v, ew_t,
             ew_lp, ew_lv, ew_lt,
             order, rws, reg):
    gk = c["gk"][i]
    ba = {0:"Urne",5:"Brief"}.get(c["ba"][i],str(c["ba"][i]))
    lk = str(c["lc_raw"][i]).zfill(2)
    r = dict(land=c["ln"][i], wkr=int(c["wkr"][i]),
        land_code=int(c["lc_raw"][i]),
        kreis_code=int(c["kc"][i]),
        gemeinde_code=int(c["gc"][i]),
        wbz=str(c["wbz"][i]),
        gemeinde=c["gem"][i], bezirksart=ba,
        valid=int(c["g"][i]), bsw=int(c["bsw"][i]),
        bd=int(c["bd"][i]),
        bsw_pred=round(c["bp"][i]*100,3),
        lam=round(c["lam"][i],2),
        p0_bb=float(c["p0"][i]),
        log10_p0=round(float(np.log10(
            max(c["p0"][i],1e-300))),2),
        bd_pct=round(c["bd"][i]/max(c["g"][i],1)*100,3))
    # Neighbors
    nn,ng,med,mx = nbr_ctx(i, c)
    r.update(n_nbr=nn, nbr_gt0=ng,
        nbr_med_pct=med, nbr_max_bsw=mx)
    # EW24 (Land-level fallback for city-states)
    if gk in ew_pct:
        r["ew24_pct"] = round(ew_pct[gk],2)
        r["ew24_votes"] = int(ew_v[gk])
        r["ew24_valid"] = int(ew_t[gk])
        r["ew24_level"] = "gemeinde"
    elif lk in ew_lp:
        r["ew24_pct"] = round(ew_lp[lk],2)
        r["ew24_votes"] = int(ew_lv[lk])
        r["ew24_valid"] = int(ew_lt[lk])
        r["ew24_level"] = "land"
    else:
        r["ew24_pct"] = None
        r["ew24_votes"] = None
        r["ew24_valid"] = None
        r["ew24_level"] = None
    # Ballot order
    rb,rd,d,adj = ballot_ctx(c["ln"][i], order)
    r.update(bsw_pos=rb, bd_pos=rd,
        ballot_dist=d, ballot_adj=adj)
    # RWS national context
    if rws is not None:
        r["rws_urne"]=rws.get("urne")
        r["rws_brief"]=rws.get("brief")
        r["rws_resid"]=rws.get("resid")
    # Registry fields
    _merge_reg(r, reg)
    return r


def _summary(tbl):
    print(f"\n{SEP}\nEVIDENCE DOSSIER\n{SEP}")
    print(f"  Precincts: {len(tbl)}")
    tot = tbl["lam"].sum()
    print(f"  Total expected missing: {tot:,.0f}")
    lg = tbl.groupby("land")
    print(f"\n  {'Land':<4}{'n':>4}{'lam':>7}"
          f"{'adj':>5}{'EW24':>6}")
    for land, grp in sorted(lg):
        n=len(grp); ls=grp["lam"].sum()
        na=grp["ballot_adj"].sum()
        ew=grp["ew24_pct"].dropna().median()
        es=f"{ew:5.1f}" if pd.notna(ew) else "  N/A"
        print(f"  {land:<4}{n:>4}{ls:>7.0f}"
              f"{int(na):>5}{es:>6}")
    # Bezirksart
    print(f"\n  By Bezirksart:")
    for ba, grp in tbl.groupby("bezirksart"):
        print(f"    {ba}: {len(grp)} ({grp['lam'].sum():.0f} exp)")
    # Neighbors
    has = tbl["n_nbr"]>0
    print(f"\n  With neighbors: {has.sum()}/{len(tbl)}")
    if has.any():
        wf = tbl.loc[has,"nbr_gt0"].sum()/tbl.loc[has,"n_nbr"].sum()
        print(f"  Weighted nbr BSW>0: {wf:.1%}")
    adj = tbl["ballot_adj"]==True
    print(f"  Ballot adjacent: {adj.sum()}/{len(tbl)}")


def _top(tbl, k=15):
    print(f"\n  Top {k} by λ:")
    print(f"  {'Land':<4}{'WKR':>4}{'λ':>6}"
          f"{'BD':>4}{'nbr':>5}"
          f"{'EW24':>6}{'BA':>6} Gemeinde")
    for _, r in tbl.head(k).iterrows():
        ew = f"{r.ew24_pct:5.1f}" if pd.notna(
            r.ew24_pct) else "  N/A"
        print(f"  {r.land:<4}{r.wkr:>4}"
              f"{r.lam:>6.1f}{r.bd:>4}"
              f"{r.n_nbr:>5}{ew:>6}"
              f"{r.bezirksart:>6} {r.gemeinde}")


def main():
    print("Loading BTW25...")
    df, pred, lc = load_btw25()
    tbl = build_dossier(df, pred, lc)
    _summary(tbl); _top(tbl)
    tbl.to_csv(DATA/"evidence_dossier.csv",index=False)
    print(f"\n  Saved → evidence_dossier.csv")
    recs = tbl.to_dict(orient="records")
    for rec in recs:
        for k,v in rec.items():
            if isinstance(v,float) and np.isnan(v):
                rec[k] = None
    with open(DATA/"evidence_dossier.json","w") as fh:
        json.dump(recs,fh,indent=2,
            default=str,ensure_ascii=False)
    print(f"  Saved → evidence_dossier.json")


if __name__ == "__main__":
    main()
