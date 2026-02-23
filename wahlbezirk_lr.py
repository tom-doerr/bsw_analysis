#!/usr/bin/env python3
"""Wahlbezirk-level linear regression: predict 2025 Zweitstimme
share per party using historical + current election data."""

import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
SEED = 42

LAND_CODE = {
    1: "SH", 2: "HH", 3: "NI", 4: "HB", 5: "NW", 6: "HE",
    7: "RP", 8: "BW", 9: "BY", 10: "SL", 11: "BE", 12: "BB",
    13: "MV", 14: "SN", 15: "ST", 16: "TH",
}

# Canonical party name mapping: {year_specific_name: canonical}
CANON = {
    "DIE LINKE": "Die Linke",
    "GRÜNE": "GRÜNE",
    "du.": "du",
    "du": "du",
}


def canon(name: str) -> str:
    return CANON.get(name, name)


def load_2025_wbz():
    """Load 2025 precinct data from zip."""
    zp = DATA / "btw25_wbz.zip"
    with ZipFile(zp) as zf:
        with zf.open("btw25_wbz_ergebnisse.csv") as fh:
            df = pd.read_csv(fh, sep=";", skiprows=4, low_memory=False)
    return df


def load_2021_wbz():
    """Load 2021 precinct data, aggregate to Wahlkreis shares."""
    zp = DATA / "btw21_wbz.zip"
    with ZipFile(zp) as zf:
        with zf.open("btw21_wbz_ergebnisse.csv") as fh:
            df = pd.read_csv(fh, sep=";", low_memory=False)
    df = df.dropna(subset=[df.columns[0]])
    df.rename(columns={df.columns[0]: "Wahlkreis"}, inplace=True)
    df["Wahlkreis"] = pd.to_numeric(df["Wahlkreis"], errors="coerce")
    df = df.dropna(subset=["Wahlkreis"])
    df["Wahlkreis"] = df["Wahlkreis"].astype(int)
    return df


def load_2017_csv(zf, name):
    """Load one 2017 CSV from zip (latin-1, skiprows=4)."""
    with zf.open(name) as fh:
        raw = fh.read().decode("latin-1")
    df = pd.read_csv(io.StringIO(raw), sep=";", skiprows=4,
                      quotechar='"', low_memory=False)
    df["Wahlkreis"] = pd.to_numeric(
        df["Wahlkreis"].astype(str).str.strip('"'), errors="coerce"
    )
    df = df.dropna(subset=["Wahlkreis"])
    df["Wahlkreis"] = df["Wahlkreis"].astype(int)
    return df


def load_2017_wbz():
    """Load 2017 erst+zweit from zip."""
    zp = DATA / "btw17_wbz.zip"
    with ZipFile(zp) as zf:
        erst = load_2017_csv(zf, "btw17_wbz_erststimmen.csv")
        zweit = load_2017_csv(zf, "btw17_wbz_zweitstimmen.csv")
    return erst, zweit


def load_2013_csv(zf, name):
    """Load one 2013 CSV (UTF-8-sig, skiprows=4)."""
    with zf.open(name) as fh:
        raw = fh.read().decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(raw), sep=";",
        skiprows=4, quotechar='"', low_memory=False)
    df["Wahlkreis"] = pd.to_numeric(
        df["Wahlkreis"].astype(str).str.strip('"'),
        errors="coerce")
    df = df.dropna(subset=["Wahlkreis"])
    df["Wahlkreis"] = df["Wahlkreis"].astype(int)
    return df


def load_2013_wbz():
    zp = DATA / "btw13_wbz.zip"
    with ZipFile(zp) as zf:
        e = load_2013_csv(zf,
            "BTW13_Erststimmen_Wahlbezirke.csv")
        z = load_2013_csv(zf,
            "BTW13_Zweitstimmen_Wahlbezirke.csv")
    return e, z


def agg_to_wkr(df, vote_cols, wkr_col="Wahlkreis"):
    """Aggregate vote columns to Wahlkreis level."""
    num = df[[wkr_col] + vote_cols].copy()
    for c in vote_cols:
        num[c] = pd.to_numeric(num[c], errors="coerce").fillna(0)
    return num.groupby(wkr_col)[vote_cols].sum().reset_index()


def shares(df, party_cols, valid_col):
    """Convert absolute votes to % shares."""
    valid = pd.to_numeric(df[valid_col], errors="coerce").fillna(0)
    out = pd.DataFrame(index=df.index)
    for c in party_cols:
        vals = pd.to_numeric(df[c], errors="coerce").fillna(0)
        out[c] = np.where(valid > 0, vals / valid * 100, 0)
    return out


def cols_25(df):
    """Extract 2025 party column lists."""
    erst = [c for c in df.columns if c.endswith(" - Erststimmen")
            and not c.startswith(("Ungültige", "Gültige"))]
    zweit = [c for c in df.columns if c.endswith(" - Zweitstimmen")
             and not c.startswith(("Ungültige", "Gültige"))]
    return erst, zweit


def cols_21(df):
    """Extract 2021 E_/Z_ party column lists."""
    erst = [c for c in df.columns if c.startswith("E_")
            and c not in ("E_Ungültige", "E_Gültige", "E_Übrige")]
    zweit = [c for c in df.columns if c.startswith("Z_")
             and c not in ("Z_Ungültige", "Z_Gültige")]
    return erst, zweit


def cols_17(df, kind):
    """Extract 2017 party columns (bare names after Gültige)."""
    start = list(df.columns).index("Gültige") + 1
    skip = {"Ungekürzte Wahlbezirksbezeichnung", "WGr/EB"}
    return [c for c in df.columns[start:] if c not in skip]


def agg_21_to_wkr(df21):
    """Aggregate 2021 precinct data to WKR-level shares."""
    e21, z21 = cols_21(df21)
    struct = ["Wahlberechtigte (A)", "Wählende (B)",
              "E_Ungültige", "E_Gültige", "Z_Ungültige", "Z_Gültige"]
    wkr = agg_to_wkr(df21, struct + e21 + z21)
    out = pd.DataFrame({"Wahlkreis": wkr["Wahlkreis"]})
    # shares
    es = shares(wkr, e21, "E_Gültige")
    zs = shares(wkr, z21, "Z_Gültige")
    for c in e21:
        p = canon(c[2:])  # strip E_ prefix
        out[f"e21_{p}"] = es[c].values
    for c in z21:
        p = canon(c[2:])
        out[f"z21_{p}"] = zs[c].values
    return _add_struct(out, wkr, "21", struct)


def _add_struct(out, wkr, yr, struct):
    """Add structural features: turnout, invalid rates."""
    wa = pd.to_numeric(wkr[struct[0]], errors="coerce").fillna(0)
    wb = pd.to_numeric(wkr[struct[1]], errors="coerce").fillna(0)
    eu = pd.to_numeric(wkr[struct[2]], errors="coerce").fillna(0)
    eg = pd.to_numeric(wkr[struct[3]], errors="coerce").fillna(0)
    zu = pd.to_numeric(wkr[struct[4]], errors="coerce").fillna(0)
    zg = pd.to_numeric(wkr[struct[5]], errors="coerce").fillna(0)
    out[f"s{yr}_turnout"] = np.where(wa > 0, wb/wa*100, 0)
    out[f"s{yr}_inv_e"] = np.where((eu+eg) > 0, eu/(eu+eg)*100, 0)
    out[f"s{yr}_inv_z"] = np.where((zu+zg) > 0, zu/(zu+zg)*100, 0)
    out[f"s{yr}_log_wa"] = np.log1p(wa)
    return out


def agg_17_to_wkr(erst17, zweit17):
    """Aggregate 2017 to WKR-level shares."""
    ep = cols_17(erst17, "erst")
    zp = cols_17(zweit17, "zweit")
    e_struct = ["Wahlberechtigte (A)", "Wähler (B)",
                "Ungültige", "Gültige"]
    we = agg_to_wkr(erst17, e_struct + ep)
    wz = agg_to_wkr(zweit17, e_struct + zp)
    out = pd.DataFrame({"Wahlkreis": we["Wahlkreis"]})
    es = shares(we, ep, "Gültige")
    for c in ep:
        out[f"e17_{canon(c)}"] = es[c].values
    zs = shares(wz, zp, "Gültige")
    for c in zp:
        out[f"z17_{canon(c)}"] = zs[c].values
    # structural from zweit file
    z_struct = ["Wahlberechtigte (A)", "Wähler (B)",
                "Ungültige", "Gültige"]
    # Rename for _add_struct compatibility
    wz_r = wz.rename(columns={"Wähler (B)": "W_B"})
    wa = pd.to_numeric(wz["Wahlberechtigte (A)"],
                        errors="coerce").fillna(0)
    wb = pd.to_numeric(wz["Wähler (B)"], errors="coerce").fillna(0)
    zu = pd.to_numeric(wz["Ungültige"], errors="coerce").fillna(0)
    zg = pd.to_numeric(wz["Gültige"], errors="coerce").fillna(0)
    eu = pd.to_numeric(we["Ungültige"], errors="coerce").fillna(0)
    eg = pd.to_numeric(we["Gültige"], errors="coerce").fillna(0)
    out["s17_turnout"] = np.where(wa > 0, wb / wa * 100, 0)
    out["s17_inv_e"] = np.where((eu+eg) > 0, eu/(eu+eg)*100, 0)
    out["s17_inv_z"] = np.where((zu+zg) > 0, zu/(zu+zg)*100, 0)
    out["s17_log_wa"] = np.log1p(wa)
    return out


def prep_2025(df25):
    """Prepare 2025 precinct features and targets."""
    erst_cols, zweit_cols = cols_25(df25)
    # Numeric conversion
    all_num = (erst_cols + zweit_cols +
               ["Wahlberechtigte (A)", "Wählende (B)",
                "Ungültige - Erststimmen", "Gültige - Erststimmen",
                "Ungültige - Zweitstimmen", "Gültige - Zweitstimmen"])
    for c in all_num:
        df25[c] = pd.to_numeric(df25[c], errors="coerce").fillna(0)
    # Filter: at least 1 valid Zweitstimme
    df25 = df25[df25["Gültige - Zweitstimmen"] >= 1].copy()
    # Erststimme shares
    e_sh = shares(df25, erst_cols, "Gültige - Erststimmen")
    e_feat = {}
    for c in erst_cols:
        p = c.replace(" - Erststimmen", "")
        e_feat[f"e25_{canon(p)}"] = e_sh[c].values
    # Zweitstimme shares (will be split into target + features)
    z_sh = shares(df25, zweit_cols, "Gültige - Zweitstimmen")
    z_map = {}
    for c in zweit_cols:
        p = c.replace(" - Zweitstimmen", "")
        z_map[canon(p)] = z_sh[c].values
    # Structural
    wa = df25["Wahlberechtigte (A)"].values.astype(float)
    wb = df25["Wählende (B)"].values.astype(float)
    eu = df25["Ungültige - Erststimmen"].values.astype(float)
    eg = df25["Gültige - Erststimmen"].values.astype(float)
    zu = df25["Ungültige - Zweitstimmen"].values.astype(float)
    zg = df25["Gültige - Zweitstimmen"].values.astype(float)
    wa_safe = np.where(wa > 0, wa, 1)
    ee_safe = np.where((eu+eg) > 0, eu+eg, 1)
    zz_safe = np.where((zu+zg) > 0, zu+zg, 1)
    struct25 = {
        "s25_turnout": np.where(wa > 0, wb/wa_safe*100, 0),
        "s25_inv_e": np.where((eu+eg) > 0, eu/ee_safe*100, 0),
        "s25_inv_z": np.where((zu+zg) > 0, zu/zz_safe*100, 0),
        "s25_log_wa": np.log1p(wa),
    }
    wkr = pd.to_numeric(df25["Wahlkreis"], errors="coerce")
    wkr = wkr.fillna(0).astype(int).values
    # Land for one-hot
    land = df25["Land"].values
    # Precinct metadata
    meta = pd.DataFrame({
        "Wahlkreis": wkr,
        "Land": land,
        "Gemeindename": df25.get("Gemeindename",
            df25.get("Gemeinde Name", "")).values,
        "Wahlbezirk": df25.get("Wahlbezirk", "").values,
    })
    return e_feat, z_map, struct25, wkr, land, meta


def build_X_base(e_feat, struct25, wkr, land, hist21, hist17):
    """Build base feature matrix (everything except z25)."""
    n = len(wkr)
    parts = [pd.DataFrame(e_feat, index=range(n)),
             pd.DataFrame(struct25, index=range(n))]
    wkr_s = pd.Series(wkr, name="Wahlkreis")
    for hist in [hist21, hist17]:
        m = wkr_s.to_frame().merge(hist, on="Wahlkreis", how="left")
        m = m.drop(columns="Wahlkreis").fillna(0)
        parts.append(m.reset_index(drop=True))
    land_df = pd.get_dummies(
        pd.Series(land, name="land"), prefix="land", drop_first=True
    ).reset_index(drop=True)
    parts.append(land_df)
    return pd.concat(parts, axis=1)


def train_party(X, y, cv):
    """Train LR for one party, return CV predictions."""
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    return cross_val_predict(pipe, X, y, cv=cv)


def compute_metrics(y_true, y_pred):
    """Compute all metrics for one party."""
    rho, p_val = spearmanr(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "Spearman": rho,
        "Spearman_p": p_val,
        "MAE_pp": mean_absolute_error(y_true, y_pred),
        "MSE_pp2": mean_squared_error(y_true, y_pred),
        "RMSE_pp": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Mean_share": y_true.mean(),
    }


def main():
    print("Loading 2025 precinct data...")
    df25 = load_2025_wbz()
    e_feat, z_map, struct25, wkr, land, meta = prep_2025(df25)
    n = len(wkr)
    print(f"  {n} precincts")

    print("Loading 2021 data...")
    df21 = load_2021_wbz()
    hist21 = agg_21_to_wkr(df21)
    print(f"  {len(hist21)} Wahlkreise")

    print("Loading 2017 data...")
    erst17, zweit17 = load_2017_wbz()
    hist17 = agg_17_to_wkr(erst17, zweit17)
    print(f"  {len(hist17)} Wahlkreise")

    print("Building base features...")
    base = build_X_base(e_feat, struct25, wkr, land, hist21, hist17)
    print(f"  Base features: {base.shape[1]}")

    parties = sorted(z_map.keys())
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    # No z25 shares as features (they sum to 100% → leakage)
    X = base.values.astype(np.float64)
    results = {}
    preds = {}
    print(f"\nTraining {len(parties)} models "
          f"({X.shape[1]} features, {X.shape[0]} rows)...",
          flush=True)
    for i, party in enumerate(parties):
        y = z_map[party]
        y_pred = train_party(X, y, cv)
        results[party] = compute_metrics(y, y_pred)
        preds[party] = {"actual": y, "predicted": y_pred}
        print(f"  [{i+1}/{len(parties)}] {party}: "
              f"R²={results[party]['R2']:.4f} "
              f"MAE={results[party]['MAE_pp']:.4f}pp",
              flush=True)

    # === Results table ===
    res_df = pd.DataFrame(results).T
    res_df.index.name = "party"
    res_df = res_df.sort_values("R2", ascending=False)
    print("\n" + "=" * 80)
    print("Per-Party Results (sorted by R²)")
    print("=" * 80)
    print(res_df.to_string(float_format="{:.4f}".format))

    # === BSW comparison ===
    print("\n" + "=" * 80)
    print("BSW Rank Among All Parties")
    print("=" * 80)
    for metric in ["R2", "Spearman", "MAE_pp", "MSE_pp2", "RMSE_pp"]:
        ascending = metric in ("MAE_pp", "MSE_pp2", "RMSE_pp")
        ranked = res_df[metric].sort_values(ascending=ascending)
        rank = list(ranked.index).index("BSW") + 1
        val = results["BSW"][metric]
        print(f"  {metric:12s}: {val:8.4f}  "
              f"(rank {rank}/{len(parties)})")

    # === Anomaly detection ===
    anomalies = _find_anomalies(parties, preds, meta)
    anom_df = pd.DataFrame(anomalies)
    if len(anom_df):
        print(f"\n{'='*70}")
        print("Anomaly Detection (|z-score| > 2.0)")
        print(f"{'='*70}")
        bc = anom_df[anom_df["party"] == "BSW"]
        print(f"  Total anomalies: {len(anom_df)}")
        print(f"  BSW anomalies: {len(bc)}")

    # === BSW top over/under predictions ===
    _print_bsw_top(preds, meta)

    # === Save CSVs ===
    _save_csvs(res_df, preds, anom_df, meta, parties)


def _find_anomalies(parties, preds, meta):
    """Find precincts with |z-score| > 2.0."""
    rows = []
    for party in parties:
        r = preds[party]["actual"] - preds[party]["predicted"]
        s = r.std()
        if s == 0:
            continue
        z = (r - r.mean()) / s
        for i in np.where(np.abs(z) > 2.0)[0]:
            rows.append({
                "party": party,
                "Wahlkreis": meta["Wahlkreis"].iloc[i],
                "Gemeinde": meta["Gemeindename"].iloc[i],
                "Land": meta["Land"].iloc[i],
                "actual": preds[party]["actual"][i],
                "predicted": preds[party]["predicted"][i],
                "residual": r[i], "z_score": z[i],
            })
    return rows


def _print_bsw_top(preds, meta):
    """Print BSW top 10 over/under-predicted precincts."""
    a = preds["BSW"]["actual"]
    p = preds["BSW"]["predicted"]
    resid = a - p
    order_under = np.argsort(-resid)[:10]
    order_over = np.argsort(resid)[:10]
    print(f"\n{'='*70}")
    print("BSW Top 10 Under-predicted (actual > predicted)")
    print(f"{'='*70}")
    fmt = "{:>5s} {:>6s} {:30s} {:>8s} {:>8s} {:>8s}"
    print(fmt.format("WKR", "Land", "Gemeinde",
                      "Actual", "Pred", "Resid"))
    for idx in order_under:
        print(fmt.format(
            str(meta["Wahlkreis"].iloc[idx]),
            str(meta["Land"].iloc[idx]),
            str(meta["Gemeindename"].iloc[idx])[:30],
            f"{a[idx]:.2f}", f"{p[idx]:.2f}",
            f"{resid[idx]:+.2f}"))
    print(f"\n{'='*70}")
    print("BSW Top 10 Over-predicted (predicted > actual)")
    print(f"{'='*70}")
    print(fmt.format("WKR", "Land", "Gemeinde",
                      "Actual", "Pred", "Resid"))
    for idx in order_over:
        print(fmt.format(
            str(meta["Wahlkreis"].iloc[idx]),
            str(meta["Land"].iloc[idx]),
            str(meta["Gemeindename"].iloc[idx])[:30],
            f"{a[idx]:.2f}", f"{p[idx]:.2f}",
            f"{resid[idx]:+.2f}"))


def _save_csvs(res_df, preds, anom_df, meta, parties):
    """Save results to CSV files."""
    out = DATA / "wahlbezirk_lr_metrics.csv"
    res_df.to_csv(out)
    print(f"\nWrote {out}")
    pred_df = meta.copy()
    for party in parties:
        a = preds[party]["actual"]
        p = preds[party]["predicted"]
        pred_df[f"{party}_actual"] = a
        pred_df[f"{party}_pred"] = p
        pred_df[f"{party}_resid"] = a - p
    out2 = DATA / "wahlbezirk_lr_predictions.csv"
    pred_df.to_csv(out2, index=False)
    print(f"Wrote {out2}")
    if len(anom_df):
        out3 = DATA / "wahlbezirk_lr_anomalies.csv"
        anom_df.to_csv(out3, index=False)
        print(f"Wrote {out3}")


if __name__ == "__main__":
    main()
