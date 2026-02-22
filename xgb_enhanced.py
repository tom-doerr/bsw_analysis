#!/usr/bin/env python3
"""XGBoost model with Europawahl 2024 + Strukturdaten.
Predicts 2025 Zweitstimme share per party at precinct level."""

import io
import json
import numpy as np
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor
import shap

from wahlbezirk_lr import (
    load_2025_wbz, load_2021_wbz, load_2017_wbz,
    prep_2025, agg_21_to_wkr, agg_17_to_wkr,
    build_X_base, SEED, LAND_CODE,
)

DATA = Path("data")


def load_ew24():
    """Load Europawahl 2024, aggregate to Gemeinde-level shares."""
    zp = DATA / "ew24_wbz.zip"
    with ZipFile(zp) as zf:
        with zf.open("Wbz_EW24_Ergebnisse.csv") as fh:
            df = pd.read_csv(fh, sep=";", low_memory=False)
    # Build Gemeinde key: Land_Kreis_Gemeinde
    for c in ["Land", "Kreis", "Gemeinde"]:
        df[c] = df[c].astype(str).str.zfill(2 if c != "Gemeinde" else 3)
    df["gem_key"] = df["Land"] + "_" + df["Kreis"] + "_" + df["Gemeinde"]
    # Party columns (17 onwards until col 51)
    party_cols = [c for c in df.columns[17:52] if c.strip()]
    valid_col = "gültig"
    for c in [valid_col] + party_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # Aggregate to Gemeinde
    agg_cols = [valid_col] + party_cols
    gem = df.groupby("gem_key")[agg_cols].sum().reset_index()
    # Convert to shares
    out = pd.DataFrame({"gem_key": gem["gem_key"]})
    for c in party_cols:
        v = gem[valid_col].values
        safe = np.where(v > 0, v, 1)
        out[f"ew24_{c}"] = np.where(v > 0, gem[c].values / safe * 100, 0)
    print(f"  EW24: {len(out)} Gemeinden, {len(party_cols)} parties")
    return out


def load_strukturdaten():
    """Load BTW2025 Strukturdaten per Wahlkreis."""
    sd = pd.read_csv(DATA / "btw2025_strukturdaten.csv",
                      sep=";", skiprows=8, encoding="utf-8-sig")
    # Row 0 has column numbers, row 1 has names, data from row 2
    sd.columns = sd.iloc[0]  # use name row
    sd = sd.iloc[1:].reset_index(drop=True)
    wkr_col = "Wahlkreis-Nr."
    sd[wkr_col] = pd.to_numeric(sd[wkr_col], errors="coerce")
    sd = sd.dropna(subset=[wkr_col])
    sd[wkr_col] = sd[wkr_col].astype(int)
    # Numeric columns (skip Land, WKR-Nr, WKR-Name, Fußnoten)
    skip = {"Land", wkr_col, "Wahlkreis-Name", "Fußnoten"}
    out = pd.DataFrame({"Wahlkreis": sd[wkr_col].values})
    for c in sd.columns:
        if c in skip or pd.isna(c):
            continue
        vals = sd[c].astype(str).str.replace(",", ".").str.strip()
        num = pd.to_numeric(vals, errors="coerce")
        if num.notna().sum() > 200:
            # Shorten column name
            short = c[:50].strip()
            out[f"sd_{short}"] = num.values
    print(f"  Strukturdaten: {len(out)} WKR, "
          f"{len(out.columns)-1} features")
    return out


def load_all():
    print("Loading base data...")
    df25 = load_2025_wbz()
    e_feat, z_map, struct25, wkr, land, meta = prep_2025(df25)
    n = len(wkr)
    print("Loading historical data...")
    df21 = load_2021_wbz()
    hist21 = agg_21_to_wkr(df21)
    erst17, zweit17 = load_2017_wbz()
    hist17 = agg_17_to_wkr(erst17, zweit17)
    print("Building base features...")
    base = build_X_base(e_feat, struct25, wkr, land, hist21, hist17)
    print(f"  Base: {base.shape[1]} features")
    # Add EW24 (Gemeinde-level join)
    print("Loading Europawahl 2024...")
    ew24 = load_ew24()
    # Build gem_key for BTW25 precincts
    df25_filt = df25[df25["Gültige - Zweitstimmen"].apply(
        pd.to_numeric, errors="coerce").fillna(0) >= 1].copy()
    df25_filt = df25_filt.reset_index(drop=True)
    for c in ["Land", "Kreis", "Gemeinde"]:
        df25_filt[c] = df25_filt[c].astype(str).str.zfill(
            2 if c != "Gemeinde" else 3)
    gk = (df25_filt["Land"] + "_" + df25_filt["Kreis"]
          + "_" + df25_filt["Gemeinde"])
    ew_merged = gk.to_frame("gem_key").merge(
        ew24, on="gem_key", how="left"
    ).drop(columns="gem_key").fillna(0).reset_index(drop=True)
    matched = (gk.isin(ew24["gem_key"])).sum()
    print(f"  EW24 matched: {matched}/{n} precincts")
    # Add Strukturdaten (Wahlkreis-level join)
    print("Loading Strukturdaten...")
    sd = load_strukturdaten()
    wkr_s = pd.Series(wkr, name="Wahlkreis")
    sd_merged = wkr_s.to_frame().merge(
        sd, on="Wahlkreis", how="left"
    ).drop(columns="Wahlkreis").fillna(0).reset_index(drop=True)
    print(f"  Strukturdaten: {sd_merged.shape[1]} features")
    # Combine all features
    X = pd.concat([base.reset_index(drop=True),
                    ew_merged, sd_merged], axis=1)
    print(f"  Total features: {X.shape[1]}")
    return X, z_map, meta


def train_party(X, y, cv):
    """Train XGBoost for one party with 10-fold CV."""
    xgb = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1,
        tree_method="hist",
    )
    return cross_val_predict(xgb, X, y, cv=cv)


def compute_metrics(y, yp):
    rho, p = spearmanr(y, yp)
    return {
        "R2": r2_score(y, yp),
        "Spearman": rho,
        "MAE_pp": mean_absolute_error(y, yp),
        "RMSE_pp": np.sqrt(mean_squared_error(y, yp)),
        "Mean_share": y.mean(),
    }


MAJOR = ["BSW", "AfD", "CDU", "SPD", "GRÜNE",
         "Die Linke", "FDP", "CSU", "FREIE WÄHLER"]


def _train_full(X, y):
    mdl = XGBRegressor(
        n_estimators=300, max_depth=6,
        learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED,
        n_jobs=-1, tree_method="hist")
    mdl.fit(X, y)
    return mdl


def _shap_values(mdl, X):
    ex = shap.TreeExplainer(mdl)
    return ex.shap_values(X)


def _top_features(sv, names, k=20):
    ma = np.abs(sv).mean(axis=0)
    top = np.argsort(ma)[::-1][:k]
    print(f"top={names[top[0]]} ({ma[top[0]]:.3f})",
          flush=True)
    return [{"feature": names[i],
             "shap": round(float(ma[i]), 4)}
            for i in top]


def compute_shap_summary(X, z_map, feature_names):
    """Train full models + SHAP for major parties."""
    print(f"\n{'='*70}")
    print("SHAP Analysis (major parties)")
    print(f"{'='*70}", flush=True)
    summary = {}
    for party in MAJOR:
        if party not in z_map:
            continue
        print(f"  {party}...", end=" ", flush=True)
        mdl = _train_full(X, z_map[party])
        sv = _shap_values(mdl, X)
        summary[party] = _top_features(
            sv, feature_names, 20)
    return summary


def main():
    X_df, z_map, meta = load_all()
    X = X_df.values.astype(np.float64)
    feature_names = list(X_df.columns)
    parties = sorted(z_map.keys())
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    results = {}
    print(f"\nTraining {len(parties)} XGBoost models "
          f"({X.shape[1]} features, {X.shape[0]} rows)...",
          flush=True)
    for i, party in enumerate(parties):
        y = z_map[party]
        yp = train_party(X, y, cv)
        results[party] = compute_metrics(y, yp)
        print(f"  [{i+1}/{len(parties)}] {party}: "
              f"R²={results[party]['R2']:.4f} "
              f"MAE={results[party]['MAE_pp']:.4f}pp",
              flush=True)
    # Results table
    res = pd.DataFrame(results).T
    res.index.name = "party"
    res = res.sort_values("R2", ascending=False)
    print(f"\n{'='*70}")
    print("Results (XGBoost + EW24 + Strukturdaten)")
    print(f"{'='*70}")
    print(res.to_string(float_format="{:.4f}".format))
    # Compare to LR baseline
    lr = pd.read_csv(DATA / "wahlbezirk_lr_metrics.csv",
                      index_col=0)
    print(f"\n{'='*70}")
    print("Comparison: LR baseline vs XGBoost+EW24+SD")
    print(f"{'='*70}")
    print(f"{'Party':>20s}  {'LR_R²':>7s}  {'XGB_R²':>7s}  {'Δ':>7s}")
    for p in res.index:
        if p in lr.index:
            lr_r2 = lr.loc[p, "R2"]
            xgb_r2 = res.loc[p, "R2"]
            d = xgb_r2 - lr_r2
            print(f"{p:>20s}  {lr_r2:7.4f}  {xgb_r2:7.4f}"
                  f"  {d:+7.4f}")
    # Save
    out = DATA / "xgb_enhanced_metrics.csv"
    res.to_csv(out)
    print(f"\nWrote {out}")
    # BSW feature importance (train on full data)
    print(f"\n{'='*70}")
    print("BSW Top-20 Feature Importances")
    print(f"{'='*70}")
    xgb_full = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, tree_method="hist",
    )
    xgb_full.fit(X, z_map["BSW"])
    imp = xgb_full.feature_importances_
    top = np.argsort(imp)[::-1][:20]
    for rank, idx in enumerate(top):
        print(f"  {rank+1:2d}. {feature_names[idx]:40s}"
              f" {imp[idx]:.4f}")
    # SHAP analysis
    summary = compute_shap_summary(
        X, z_map, np.array(feature_names))
    out = Path("docs/data/shap_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
