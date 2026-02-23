#!/usr/bin/env python3
"""Gemeinde-level panel: track Die Linke→BSW flow
across 2013/2017/2021/2025 elections."""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", "invalid value")

from wahlbezirk_lr import (load_2025_wbz, load_2021_wbz,
    load_2013_wbz, load_2017_wbz, LAND_CODE)

DATA = Path("data")
SEP = "=" * 60


def _gem_key(df, land="Land", kreis="Kreis",
             gem="Gemeinde"):
    """Build Gemeinde key from geo columns."""
    def _fmt(s, w):
        return (pd.to_numeric(s, errors="coerce")
                .fillna(0).astype(int)
                .astype(str).str.zfill(w))
    l = _fmt(df[land], 2)
    k = _fmt(df[kreis], 2)
    g = _fmt(df[gem], 3)
    return l + "_" + k + "_" + g


def _agg_gem(df, valid_col, party_cols, label):
    """Aggregate df to Gemeinde, return shares."""
    df = df.copy()
    df["gem_key"] = _gem_key(df)
    df[valid_col] = pd.to_numeric(
        df[valid_col], errors="coerce").fillna(0)
    cols = [valid_col] + party_cols
    for c in party_cols:
        df[c] = pd.to_numeric(
            df[c], errors="coerce").fillna(0)
    g = df.groupby("gem_key")[cols].sum().reset_index()
    out = pd.DataFrame({"gem_key": g["gem_key"],
        f"{label}_valid": g[valid_col]})
    for c in party_cols:
        v = g[valid_col].values
        out[f"{label}_{c}"] = np.where(
            v > 0, g[c].values / v * 100, 0)
    return out


def _agg_bare(df, valid_col, ps, label):
    """Aggregate with bare party column names."""
    df = df.copy()
    df["gem_key"] = _gem_key(df)
    df[valid_col] = pd.to_numeric(
        df[valid_col], errors="coerce").fillna(0)
    for c in ps:
        df[c] = pd.to_numeric(
            df[c], errors="coerce").fillna(0)
    g = df.groupby("gem_key")[
        [valid_col]+ps].sum().reset_index()
    out = pd.DataFrame({"gem_key": g["gem_key"],
        f"{label}_valid": g[valid_col]})
    for c in ps:
        v = g[valid_col].values
        cn = "Die Linke" if c == "DIE LINKE" else c
        out[f"{label}_{cn}"] = np.where(
            v > 0, g[c].values / v * 100, 0)
    return out


def agg_25():
    print("  Loading 2025...")
    df = load_2025_wbz()
    ps = ["BSW","Die Linke","AfD","SPD",
          "CDU","FDP","GRÜNE"]
    cols = [f"{p} - Zweitstimmen" for p in ps]
    out = _agg_gem(df,"Gültige - Zweitstimmen",
                   cols, "z25")
    for p in ps:
        old = f"z25_{p} - Zweitstimmen"
        out.rename(columns={old:f"z25_{p}"},
                   inplace=True)
    return out


def agg_21():
    print("  Loading 2021...")
    df = load_2021_wbz()
    df = df.dropna(subset=[df.columns[0]])
    ps = ["Z_DIE LINKE","Z_AfD","Z_SPD",
          "Z_CDU","Z_FDP","Z_GRÜNE"]
    out = _agg_gem(df, "Z_Gültige", ps, "z21")
    for p in ps:
        cn = p.replace("Z_","")
        cn = "Die Linke" if cn=="DIE LINKE" else cn
        out.rename(columns={f"z21_{p}":f"z21_{cn}"},
                   inplace=True)
    return out


def agg_17():
    print("  Loading 2017...")
    _, z = load_2017_wbz()
    ps = ["DIE LINKE","AfD","SPD","CDU","FDP","GRÜNE"]
    return _agg_bare(z, "Gültige", ps, "z17")


def agg_13():
    print("  Loading 2013...")
    _, z = load_2013_wbz()
    ps = ["DIE LINKE","AfD","SPD","CDU","FDP","GRÜNE"]
    return _agg_bare(z, "Gültig", ps, "z13")


def build_panel():
    """Join all 4 elections on gem_key."""
    print("Building Gemeinde panel...")
    g25 = agg_25()
    g21 = agg_21()
    g17 = agg_17()
    g13 = agg_13()
    panel = g25.merge(g21, on="gem_key", how="inner")
    panel = panel.merge(g17, on="gem_key", how="inner")
    panel = panel.merge(g13, on="gem_key", how="inner")
    panel["land"] = panel["gem_key"].str[:2]
    print(f"  Panel: {len(panel)} Gemeinden "
          f"(4-way match)")
    return panel


def linke_flow(panel):
    """Track Die Linke decline → BSW emergence."""
    print(f"\n{SEP}")
    print("Die Linke → BSW Flow Analysis")
    print(SEP)
    p = panel
    # Die Linke trajectory
    for yr in ["z13","z17","z21","z25"]:
        col = f"{yr}_Die Linke"
        v = p[col].values
        w = p[f"{yr}_valid"].values
        wm = np.average(v, weights=w)
        print(f"  {yr} Die Linke: {wm:.2f}% "
              f"(weighted mean)")
    # BSW 2025
    bsw = p["z25_BSW"].values
    w25 = p["z25_valid"].values
    print(f"  z25 BSW: {np.average(bsw,weights=w25):.2f}%")
    # Linke drop vs BSW gain
    drop = p["z17_Die Linke"].values-p["z25_Die Linke"].values
    r,pv = pearsonr(drop, bsw)
    print(f"  r(Linke_drop, BSW): {r:.3f} (p={pv:.2e})")
    # Regression
    from numpy.polynomial.polynomial import polyfit
    c = polyfit(drop, bsw, 1, w=w25)
    pred_bsw = c[0] + c[1]*drop
    resid = bsw - pred_bsw
    deficit = resid/100*w25
    neg = deficit[deficit<0].sum()
    print(f"  BSW = {c[0]:.2f}+{c[1]:.3f}×drop")
    print(f"  Deficit Gemeinden: {(deficit<0).sum()}")
    print(f"  Total deficit: {neg:,.0f} votes")
    return resid, deficit


def per_land(panel, deficit):
    """Per-Land breakdown of deficit."""
    print(f"\n{SEP}")
    print("Per-Land Deficit")
    print(SEP)
    panel = panel.copy()
    panel["deficit"] = deficit
    print(f"  {'Land':<4} {'n':>5} {'deficit':>10}")
    for land in sorted(panel["land"].unique()):
        m = panel["land"] == land
        d = panel.loc[m, "deficit"].sum()
        nm = LAND_CODE.get(int(land), land)
        print(f"  {nm:<4} {m.sum():>5} {d:>+10,.0f}")
    return panel


def top_anomalies(panel, resid, deficit):
    """Top 20 Gemeinden by negative deficit."""
    print(f"\n{SEP}")
    print("Top 20 Anomalous Gemeinden")
    print(SEP)
    p = panel.copy()
    p["resid"] = resid; p["deficit_v"] = deficit
    top = p.nsmallest(20, "deficit_v")
    print(f"  {'gem_key':<12} {'Ld':<3} {'BSW%':>5}"
          f" {'exp':>5} {'deficit':>8}")
    for _, r in top.iterrows():
        nm = LAND_CODE.get(int(r["land"]),r["land"])
        exp = r["z25_BSW"] - r["resid"]
        print(f"  {r['gem_key']:<12} {nm:<3}"
              f" {r['z25_BSW']:>5.1f} {exp:>5.1f}"
              f" {r['deficit_v']:>+8.0f}")


def main():
    panel = build_panel()
    resid, deficit = linke_flow(panel)
    panel = per_land(panel, deficit)
    top_anomalies(panel, resid, deficit)
    # Save panel
    panel.to_csv(DATA / "panel_analysis.csv",
                 index=False)
    print(f"\nSaved {DATA/'panel_analysis.csv'}")


if __name__ == "__main__":
    main()
