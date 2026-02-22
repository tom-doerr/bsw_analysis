#!/usr/bin/env python3
"""EW24 → BTW25 swing analysis for BSW at Gemeinde level."""

import numpy as np
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from scipy.stats import spearmanr, pearsonr

DATA = Path("data")

# Party name mapping: EW24 → BTW25
EW_TO_BTW = {
    "DIE LINKE": "Die Linke",
    "DieBasis": "dieBasis",
}

# Parties to analyze swing
PARTIES = ["BSW", "AfD", "FDP", "Die Linke", "SPD",
           "CDU", "GRÜNE", "CSU", "FREIE WÄHLER"]

LAND_NAME = {
    "01": "SH", "02": "HH", "03": "NI", "04": "HB",
    "05": "NW", "06": "HE", "07": "RP", "08": "BW",
    "09": "BY", "10": "SL", "11": "BE", "12": "BB",
    "13": "MV", "14": "SN", "15": "ST", "16": "TH",
}


def load_ew24_gem():
    """Load EW24 aggregated to Gemeinde, return shares."""
    zp = DATA / "ew24_wbz.zip"
    with ZipFile(zp) as zf:
        with zf.open("Wbz_EW24_Ergebnisse.csv") as fh:
            df = pd.read_csv(fh, sep=";", low_memory=False)
    for c in ["Land", "Kreis", "Gemeinde"]:
        df[c] = df[c].astype(str).str.zfill(
            2 if c != "Gemeinde" else 3)
    df["gem_key"] = (df["Land"] + "_" + df["Kreis"]
                     + "_" + df["Gemeinde"])
    df["land"] = df["Land"]
    valid = "gültig"
    df[valid] = pd.to_numeric(df[valid], errors="coerce").fillna(0)
    pcols = ["BSW", "AfD", "FDP", "DIE LINKE", "SPD",
             "CDU", "GRÜNE", "CSU", "FREIE WÄHLER"]
    for c in pcols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    gem = df.groupby(["gem_key", "land"])[
        [valid] + pcols].sum().reset_index()
    out = pd.DataFrame({"gem_key": gem["gem_key"],
                         "land": gem["land"]})
    v = gem[valid].values
    safe = np.where(v > 0, v, 1)
    for c in pcols:
        name = EW_TO_BTW.get(c, c)
        out[f"ew_{name}"] = np.where(
            v > 0, gem[c].values / safe * 100, np.nan)
    out["ew_valid"] = v
    print(f"EW24: {len(out)} Gemeinden")
    return out


def load_btw25_gem():
    """Load BTW25 aggregated to Gemeinde, return shares."""
    zp = DATA / "btw25_wbz.zip"
    with ZipFile(zp) as zf:
        fn = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(fn) as fh:
            df = pd.read_csv(fh, sep=";", skiprows=4,
                             encoding="utf-8-sig", low_memory=False)
    for c in ["Land", "Kreis", "Gemeinde"]:
        df[c] = df[c].astype(str).str.zfill(
            2 if c != "Gemeinde" else 3)
    df["gem_key"] = (df["Land"] + "_" + df["Kreis"]
                     + "_" + df["Gemeinde"])
    valid = "Gültige - Zweitstimmen"
    df[valid] = pd.to_numeric(df[valid], errors="coerce").fillna(0)
    df = df[df[valid] >= 1]
    zcols = {}
    for p in PARTIES:
        col = f"{p} - Zweitstimmen"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            zcols[p] = col
    agg = {valid: "sum"}
    for p, col in zcols.items():
        agg[col] = "sum"
    gem = df.groupby("gem_key").agg(agg).reset_index()
    out = pd.DataFrame({"gem_key": gem["gem_key"]})
    v = gem[valid].values
    safe = np.where(v > 0, v, 1)
    for p, col in zcols.items():
        out[f"btw_{p}"] = gem[col].values / safe * 100
    out["btw_valid"] = v
    print(f"BTW25: {len(out)} Gemeinden")
    return out


def merge_and_swing():
    """Merge EW24+BTW25, compute swing per party."""
    ew = load_ew24_gem()
    btw = load_btw25_gem()
    df = ew.merge(btw, on="gem_key", how="inner")
    print(f"Matched: {len(df)} Gemeinden")
    # Compute swing = BTW25 - EW24
    for p in PARTIES:
        ec, bc = f"ew_{p}", f"btw_{p}"
        if ec in df.columns and bc in df.columns:
            df[f"swing_{p}"] = df[bc] - df[ec]
    return df


def step1_overall(df):
    """Overall swing distribution per party."""
    print(f"\n{'='*60}")
    print("1. Overall EW24→BTW25 swing (pp)")
    print(f"{'='*60}")
    hdr = f"{'Party':>15s}  {'EW24':>6s}  {'BTW25':>6s}  " \
          f"{'Swing':>7s}  {'Std':>6s}  {'Med':>7s}"
    print(hdr)
    for p in PARTIES:
        sc = f"swing_{p}"
        if sc not in df.columns:
            continue
        s = df[sc].dropna()
        ew = df[f"ew_{p}"].mean()
        bt = df[f"btw_{p}"].mean()
        print(f"{p:>15s}  {ew:6.2f}  {bt:6.2f}  "
              f"{s.mean():+7.2f}  {s.std():6.2f}  "
              f"{s.median():+7.2f}")


def step2_per_land(df):
    """BSW swing per Bundesland."""
    print(f"\n{'='*60}")
    print("2. BSW swing per Land (pp)")
    print(f"{'='*60}")
    hdr = f"{'Land':>4s}  {'n':>5s}  {'EW24':>6s}  {'BTW25':>6s}  " \
          f"{'Swing':>7s}  {'Std':>6s}"
    print(hdr)
    for lc in sorted(LAND_NAME.keys()):
        m = df[df["land"] == lc]
        if len(m) == 0:
            continue
        s = m["swing_BSW"].dropna()
        ln = LAND_NAME[lc]
        print(f"{ln:>4s}  {len(m):5d}  "
              f"{m['ew_BSW'].mean():6.2f}  "
              f"{m['btw_BSW'].mean():6.2f}  "
              f"{s.mean():+7.2f}  {s.std():6.2f}")


def step2b_swing_uniformity(df):
    """Compare swing variability across parties."""
    print(f"\n{'='*60}")
    print("2b. Swing uniformity (lower CV = more uniform)")
    print(f"{'='*60}")
    hdr = f"{'Party':>15s}  {'Mean':>7s}  {'Std':>6s}  " \
          f"{'CV':>6s}  {'IQR':>6s}"
    print(hdr)
    for p in PARTIES:
        sc = f"swing_{p}"
        if sc not in df.columns:
            continue
        s = df[sc].dropna()
        cv = s.std() / max(abs(s.mean()), 0.01)
        iqr = s.quantile(0.75) - s.quantile(0.25)
        print(f"{p:>15s}  {s.mean():+7.2f}  {s.std():6.2f}"
              f"  {cv:6.2f}  {iqr:6.2f}")


def step3_swing_correlates(df):
    """Correlate BSW swing with Gemeinde-level vars."""
    print(f"\n{'='*60}")
    print("3. BSW swing correlations")
    print(f"{'='*60}")
    sw = df["swing_BSW"].values
    pairs = [
        ("EW24 BSW share", df["ew_BSW"].values),
        ("BTW25 valid votes", np.log1p(df["btw_valid"].values)),
        ("EW24 valid votes", np.log1p(df["ew_valid"].values)),
    ]
    for name, x in pairs:
        mask = np.isfinite(x) & np.isfinite(sw)
        r, p = pearsonr(x[mask], sw[mask])
        rho, _ = spearmanr(x[mask], sw[mask])
        print(f"  {name:25s}: r={r:+.3f}  ρ={rho:+.3f}")


def step4_outliers(df):
    """Gemeinden with extreme BSW swing."""
    print(f"\n{'='*60}")
    print("4. BSW swing outliers (z-score)")
    print(f"{'='*60}")
    sw = df["swing_BSW"]
    z = (sw - sw.mean()) / sw.std()
    df["swing_z"] = z
    # Bottom 15 (BSW lost most vs EW24)
    print("\nBottom 15 (BSW lost most vs EW24):")
    bot = df.nsmallest(15, "swing_z")
    for _, r in bot.iterrows():
        ln = LAND_NAME.get(r["land"], "??")
        print(f"  {ln} {r['gem_key']}  "
              f"EW={r['ew_BSW']:5.1f}  "
              f"BTW={r['btw_BSW']:5.1f}  "
              f"Δ={r['swing_BSW']:+5.1f}pp  "
              f"z={r['swing_z']:+.1f}")
    n_neg = (z < -2).sum()
    n_pos = (z > 2).sum()
    print(f"\n  z<-2: {n_neg} Gemeinden  |  z>+2: {n_pos}")
    # Top 15 (BSW gained most vs EW24)
    print("\nTop 15 (BSW gained most vs EW24):")
    top = df.nlargest(15, "swing_z")
    for _, r in top.iterrows():
        ln = LAND_NAME.get(r["land"], "??")
        print(f"  {ln} {r['gem_key']}  "
              f"EW={r['ew_BSW']:5.1f}  "
              f"BTW={r['btw_BSW']:5.1f}  "
              f"Δ={r['swing_BSW']:+5.1f}pp  "
              f"z={r['swing_z']:+.1f}")


def step5_ew_btw_corr(df):
    """How well does EW24 share predict BTW25 share?"""
    print(f"\n{'='*60}")
    print("5. EW24→BTW25 correlation per party")
    print(f"{'='*60}")
    hdr = f"{'Party':>15s}  {'r':>6s}  {'ρ':>6s}  {'R²':>6s}"
    print(hdr)
    for p in PARTIES:
        ec, bc = f"ew_{p}", f"btw_{p}"
        if ec not in df.columns or bc not in df.columns:
            continue
        x, y = df[ec].values, df[bc].values
        mask = np.isfinite(x) & np.isfinite(y)
        r, _ = pearsonr(x[mask], y[mask])
        rho, _ = spearmanr(x[mask], y[mask])
        print(f"{p:>15s}  {r:6.3f}  {rho:6.3f}  "
              f"{r**2:6.3f}")


def step6_swing_vs_controls(df):
    """Per-Land BSW swing vs control party swings."""
    print(f"\n{'='*60}")
    print("6. Per-Land swing: BSW vs FDP vs Die Linke")
    print(f"{'='*60}")
    ctrls = ["FDP", "Die Linke"]
    hdr = f"{'Land':>4s}  {'BSW':>7s}  "
    hdr += "  ".join(f"{c:>7s}" for c in ctrls)
    print(hdr)
    for lc in sorted(LAND_NAME.keys()):
        m = df[df["land"] == lc]
        if len(m) == 0:
            continue
        ln = LAND_NAME[lc]
        vals = [f"{m['swing_BSW'].mean():+7.2f}"]
        for c in ctrls:
            sc = f"swing_{c}"
            vals.append(f"{m[sc].mean():+7.2f}")
        print(f"{ln:>4s}  {'  '.join(vals)}")


def main():
    df = merge_and_swing()
    step1_overall(df)
    step2_per_land(df)
    step2b_swing_uniformity(df)
    step3_swing_correlates(df)
    step4_outliers(df)
    step5_ew_btw_corr(df)
    step6_swing_vs_controls(df)
    # Save
    out = DATA / "bsw_swing.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
