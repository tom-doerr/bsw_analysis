#!/usr/bin/env python3
"""Aggregate Gemeinde-level swing to Wahlkreis."""

import numpy as np
import pandas as pd
from pathlib import Path
from zipfile import ZipFile

DATA = Path("data")
PARTIES = ["BSW", "AfD", "FDP", "Die Linke", "SPD",
           "CDU", "GRÜNE", "CSU", "FREIE WÄHLER"]


def load_wkr_gem_map():
    """Build Gemeinde→WKR mapping from BTW25."""
    zp = DATA / "btw25_wbz.zip"
    with ZipFile(zp) as zf:
        fn = [n for n in zf.namelist()
              if n.endswith(".csv")][0]
        with zf.open(fn) as fh:
            df = pd.read_csv(
                fh, sep=";", skiprows=4,
                encoding="utf-8-sig",
                low_memory=False)
    for c in ["Land", "Kreis", "Gemeinde"]:
        df[c] = df[c].astype(str).str.zfill(
            2 if c != "Gemeinde" else 3)
    df["gem_key"] = (df["Land"] + "_"
        + df["Kreis"] + "_" + df["Gemeinde"])
    vc = "Gültige - Zweitstimmen"
    df[vc] = pd.to_numeric(
        df[vc], errors="coerce").fillna(0)
    df["Wahlkreis"] = pd.to_numeric(
        df["Wahlkreis"], errors="coerce")
    agg = df.groupby(["gem_key", "Wahlkreis"])[
        vc].sum().reset_index()
    agg.columns = ["gem_key", "wkr", "valid"]
    print(f"  WKR-Gem map: {len(agg)} rows")
    return agg


def main():
    print("Loading data...")
    swing = pd.read_csv(DATA / "bsw_swing.csv")
    wmap = load_wkr_gem_map()
    # Join swing to WKR via gem_key
    merged = wmap.merge(swing, on="gem_key",
                        how="left")
    print(f"  Merged: {len(merged)} rows, "
          f"{merged['wkr'].nunique()} WKR")
    # Weighted mean swing per WKR
    scols = [f"swing_{p}" for p in PARTIES
             if f"swing_{p}" in merged.columns]
    rows = []
    for wkr, g in merged.groupby("wkr"):
        w = g["valid"].values
        total = w.sum()
        row = {"Wahlkreis": int(wkr)}
        for c in scols:
            vals = g[c].fillna(0).values
            if total > 0:
                row[c] = (vals * w).sum() / total
            else:
                row[c] = 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    fp = DATA / "bsw_swing_wkr.csv"
    out.to_csv(fp, index=False)
    print(f"Wrote {fp}: {len(out)} WKR")
    bsw = out["swing_BSW"]
    print(f"  BSW swing: [{bsw.min():.2f}, "
          f"{bsw.max():.2f}]")


if __name__ == "__main__":
    main()
