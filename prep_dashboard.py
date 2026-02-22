#!/usr/bin/env python3
"""Prepare JSON data for the Three.js dashboard."""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
OUT = Path("docs/data")

PARTIES = ["BSW", "AfD", "CDU", "SPD", "GRÜNE", "CSU",
           "FDP", "Die Linke", "FREIE WÄHLER"]

LAND_NAME = {
    "SH": "Schleswig-Holstein", "HH": "Hamburg",
    "NI": "Niedersachsen", "HB": "Bremen",
    "NW": "Nordrhein-Westfalen", "HE": "Hessen",
    "RP": "Rheinland-Pfalz", "BW": "Baden-Württemberg",
    "BY": "Bayern", "SL": "Saarland", "BE": "Berlin",
    "BB": "Brandenburg", "MV": "Mecklenburg-Vorpommern",
    "SN": "Sachsen", "ST": "Sachsen-Anhalt",
    "TH": "Thüringen",
}


def load_wkr_votes():
    """Load BTW25 Wahlkreis-level vote shares."""
    df = pd.read_csv(DATA / "btw2025_brief_wkr.csv",
                      sep=";", skiprows=4, encoding="utf-8-sig")
    # Sum Urne + Brief per WKR
    valid_col = "Gültige - Zweitstimmen"
    voter_col = "Wahlberechtigte"
    turnout_col = "Wählende"
    df[valid_col] = pd.to_numeric(df[valid_col], errors="coerce")
    df[voter_col] = pd.to_numeric(df[voter_col], errors="coerce")
    df[turnout_col] = pd.to_numeric(df[turnout_col], errors="coerce")
    agg = {"Wahlkreisname": "first", "Land": "first"}
    agg[valid_col] = "sum"
    agg[voter_col] = "sum"
    agg[turnout_col] = "sum"
    for p in PARTIES:
        col = f"{p} - Zweitstimmen"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            agg[col] = "sum"
    wkr = df.groupby("Wahlkreis-Nummer").agg(agg).reset_index()
    return wkr, valid_col


def load_residuals_by_wkr():
    """Load mean prediction residual per WKR."""
    cols = ["Wahlkreis"]
    for p in PARTIES:
        cols.extend([f"{p}_resid"])
    use = [c for c in cols if c != "Wahlkreis"] + ["Wahlkreis"]
    pred = pd.read_csv(DATA / "wahlbezirk_lr_predictions.csv",
                        usecols=lambda c: c in use)
    agg = {}
    for p in PARTIES:
        rc = f"{p}_resid"
        if rc in pred.columns:
            agg[rc] = "mean"
    agg["Wahlkreis"] = "count"
    g = pred.groupby("Wahlkreis").agg(agg)
    g = g.rename(columns={"Wahlkreis": "n_precincts"})
    return g.reset_index()


def load_swing_by_wkr():
    """Load WKR-level swing data if available."""
    fp = DATA / "bsw_swing_wkr.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp)


def build_wkr_data():
    """Build wkr_data.json."""
    wkr, vcol = load_wkr_votes()
    resid = load_residuals_by_wkr()
    wkr = wkr.merge(resid, left_on="Wahlkreis-Nummer",
                     right_on="Wahlkreis", how="left")
    swing = load_swing_by_wkr()
    if swing is not None:
        wkr = wkr.merge(swing, left_on="Wahlkreis-Nummer",
                         right_on="Wahlkreis", how="left",
                         suffixes=("", "_sw"))
    rows = []
    for _, r in wkr.iterrows():
        v = r[vcol]
        safe = v if v > 0 else 1
        d = {
            "wkr": int(r["Wahlkreis-Nummer"]),
            "name": r["Wahlkreisname"],
            "land": r["Land"],
            "turnout": round(r["Wählende"] / max(r["Wahlberechtigte"], 1) * 100, 1),
            "valid": int(v),
        }
        # Party shares
        shares = {}
        for p in PARTIES:
            col = f"{p} - Zweitstimmen"
            if col in r.index:
                shares[p] = round(float(r[col]) / safe * 100, 2)
        d["shares"] = shares
        # Residuals per party
        resids = {}
        for p in PARTIES:
            rc = f"{p}_resid"
            if rc in r.index and pd.notna(r[rc]):
                resids[p] = round(float(r[rc]), 3)
        d["resid"] = resids
        # Swing per party
        swings = {}
        for p in PARTIES:
            sc = f"swing_{p}"
            if sc in r.index and pd.notna(r[sc]):
                swings[p] = round(float(r[sc]), 3)
        d["swing"] = swings
        d["n_precincts"] = int(r.get("n_precincts", 0))
        rows.append(d)
    return rows


def build_summary():
    """Build summary.json with model metrics."""
    lr = pd.read_csv(DATA / "wahlbezirk_lr_metrics.csv", index_col=0)
    xgb = pd.read_csv(DATA / "xgb_enhanced_metrics.csv", index_col=0)
    parties = {}
    for p in PARTIES:
        d = {}
        if p in lr.index:
            d["lr_r2"] = round(float(lr.loc[p, "R2"]), 4) if "R2" in lr.columns else None
        if p in xgb.index:
            d["xgb_r2"] = round(float(xgb.loc[p, "R2"]), 4)
            d["xgb_mae"] = round(float(xgb.loc[p, "MAE_pp"]), 3)
        parties[p] = d
    return {
        "n_precincts": 95046,
        "n_features_lr": 210,
        "n_features_xgb": 281,
        "parties": parties,
    }


def build_geojson(shp_path):
    """Convert shapefile to simplified GeoJSON."""
    import re
    import shapefile
    sf = shapefile.Reader(shp_path)
    fields = [f[0] for f in sf.fields[1:]]
    features = []
    for sr in sf.shapeRecords():
        props = dict(zip(fields, sr.record))
        feat = {
            "type": "Feature",
            "properties": {
                "wkr": int(props["WKR_NR"]),
                "name": props["WKR_NAME"],
            },
            "geometry": sr.shape.__geo_interface__,
        }
        features.append(feat)
    gc = {"type": "FeatureCollection", "features": features}
    raw = json.dumps(gc, ensure_ascii=False)
    # Reduce coordinate precision to 4 decimals (~11m)
    raw = re.sub(
        r"-?\d+\.\d{5,}",
        lambda m: str(round(float(m.group()), 4)),
        raw,
    )
    out = OUT / "wahlkreise.json"
    with open(out, "w") as f:
        f.write(raw)
    sz = out.stat().st_size
    print(f"  GeoJSON: {sz/1024:.0f}KB, {len(features)} features")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building wkr_data.json...")
    wkr = build_wkr_data()
    with open(OUT / "wkr_data.json", "w") as f:
        json.dump(wkr, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  Wrote {len(wkr)} Wahlkreise")

    print("Building summary.json...")
    summary = build_summary()
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("  Done")

    shp = Path("/tmp/btw25_geometrie_wahlkreise_shp_geo")
    if shp.with_suffix(".shp").exists():
        print("Building wahlkreise.json...")
        build_geojson(str(shp))
    else:
        print("  Shapefile not found, skipping GeoJSON")


if __name__ == "__main__":
    main()
