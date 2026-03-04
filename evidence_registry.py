#!/usr/bin/env python3
"""Build structured evidence registry of suspicious precincts
where BSW may have been undercounted."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson

from wahlbezirk_lr import load_2025_wbz, LAND_CODE, validate_totals

DATA = Path("data")
SEP = "=" * 60
BD = "BÜNDNIS DEUTSCHLAND"
BSW_DEFICIT = 9_529


def load_all():
    """Load raw 2025 data + LR predictions."""
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
        DATA / "wahlbezirk_lr_predictions.csv")
    assert len(df) == len(pred)
    validate_totals(df)
    print(f"  {len(df)} precincts loaded")
    return df, pred


def _votes(df, party):
    col = f"{party} - Zweitstimmen"
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def _valid(df):
    return df["Gültige - Zweitstimmen"].values


KNOWN_CASES = [
    {"source": "BSW Erwiderung (Lipinski)",
     "gemeinde_name": "Wedel, Stadt", "land_name": "SH",
     "wahlkreis_nr": 7, "claim": "BD=31, BSW=0",
     "recount_status": "not_recounted"},
    {"source": "BSW Erwiderung (Lipinski)",
     "gemeinde_name": "Katlenburg-Lindau",
     "land_name": "NI", "wahlkreis_nr": 52,
     "claim": "BD=16, BSW=0",
     "recount_status": "not_recounted"},
    {"source": "BSW Erwiderung (Lipinski)",
     "gemeinde_name": "Reinhardtsdorf-Schöna",
     "land_name": "SN", "wahlkreis_nr": 157,
     "claim": "Brief BSW=0/152, BD=12 (7.9%)",
     "recount_status": "protocol_check_only"},
]


def compute_scores(df, pred):
    """Compute per-precinct anomaly scores."""
    g = _valid(df).astype(float)
    bsw_v = _votes(df, "BSW").values.astype(float)
    bd_v = _votes(df, BD).values.astype(float)
    bsw_pred_pct = pred["BSW_pred"].values
    bsw_resid = pred["BSW_resid"].values
    bp = np.clip(bsw_pred_pct / 100, 1e-8, 1-1e-8)
    p_zero = np.power(1 - bp, g)  # Binomial exact
    mu = bsw_resid.mean()
    sd = max(bsw_resid.std(), 1e-6)
    bsw_resid_z = (bsw_resid - mu) / sd
    # BD rank percentile within Land
    land = pd.to_numeric(df["Land"], errors="coerce")
    bd_share = np.where(g > 0, bd_v / g * 100, 0)
    bd_pctile = np.zeros(len(df))
    for lv in land.dropna().unique():
        m = (land == lv).values
        if m.sum() < 10:
            continue
        ranks = pd.Series(bd_share[m]).rank(pct=True)
        bd_pctile[m] = ranks.values

    wkr = pd.to_numeric(
        df.get("Wahlkreis", df.iloc[:, 0]),
        errors="coerce").fillna(0).astype(int)
    gem = df.get("Gemeindename", df.get(
        "Gemeinde", pd.Series([""] * len(df))))
    bezart = df["Bezirksart"].values.astype(int)
    land_name = land.map(LAND_CODE).fillna("")

    scores = pd.DataFrame({
        "land": land_name.values,
        "wahlkreis": wkr.values,
        "gemeinde": gem.values,
        "bezirksart": bezart,
        "bsw_votes": bsw_v.astype(int),
        "bd_votes": bd_v.astype(int),
        "valid_total": g.astype(int),
        "bsw_pred_pct": np.round(bsw_pred_pct, 3),
        "bsw_resid": np.round(bsw_resid, 3),
        "bsw_resid_z": np.round(bsw_resid_z, 3),
        "p_bsw_zero": np.round(p_zero, 6),
        "bd_pctile_land": np.round(bd_pctile, 4),
    })
    return scores


def flag_suspicious(scores):
    """Flag precincts meeting anomaly criteria."""
    bv = scores["bsw_votes"].values
    pz = scores["p_bsw_zero"].values
    rz = scores["bsw_resid_z"].values
    bp = scores["bd_pctile_land"].values
    bd = scores["bd_votes"].values
    f1 = (bv == 0) & (pz < 0.01)
    f2 = rz < -2
    f3 = bp > 0.98  # top 2% BD within Land
    f4 = (bv == 0) & (bd > 5)
    mask = f1 | f2 | f3 | f4
    out = scores[mask].copy()
    flags = []
    for i in out.index:
        f = []
        if f1[i]: f.append("suspicious_zero")
        if f2[i]: f.append("large_negative_resid")
        if f3[i]: f.append("high_bd")
        if f4[i]: f.append("zero_bsw_nonzero_bd")
        flags.append(f)
    out["flags"] = flags
    return out


def match_known(reg, known):
    """Mark known cases in the registry."""
    reg["recount_status"] = "unknown"
    reg["source"] = ""
    reg["claim"] = ""
    for c in known:
        m = reg["land"] == c["land_name"]
        if "wahlkreis_nr" in c:
            m &= reg["wahlkreis"] == c["wahlkreis_nr"]
        for i in reg.index[m]:
            gn = str(reg.loc[i, "gemeinde"]).lower()
            cn = c["gemeinde_name"].lower()
            if cn in gn or gn in cn:
                reg.loc[i, "recount_status"] = c[
                    "recount_status"]
                reg.loc[i, "source"] = c["source"]
                reg.loc[i, "claim"] = c["claim"]
    return reg


def main():
    df, pred = load_all()
    print(f"\n{SEP}")
    print("EVIDENCE REGISTRY: Suspicious Precincts")
    print(SEP)
    scores = compute_scores(df, pred)
    reg = flag_suspicious(scores)
    reg = match_known(reg, KNOWN_CASES)
    print(f"\n  Total precincts: {len(df):,}")
    print(f"  Flagged: {len(reg):,}")

    # Flag breakdown
    all_flags = [f for fl in reg["flags"] for f in fl]
    from collections import Counter
    fc = Counter(all_flags)
    for flag, cnt in fc.most_common():
        print(f"    {flag}: {cnt}")

    # Summary stats
    n_zero = (reg["bsw_votes"] == 0).sum()
    print(f"\n  BSW=0 in flagged: {n_zero}")
    lam = reg["bsw_pred_pct"] / 100 * reg["valid_total"]
    missing = lam[reg["bsw_votes"] == 0].sum()
    print(f"  Expected votes in BSW=0: {missing:,.0f}")

    # Per-Land breakdown
    print(f"\n  {'Land':<4} {'n':>5} {'BSW=0':>6}"
          f" {'mean_BD_pct':>11}")
    for land in sorted(reg["land"].unique()):
        m = reg["land"] == land
        n = m.sum()
        nz = ((reg["bsw_votes"] == 0) & m).sum()
        bp = reg.loc[m, "bd_pctile_land"].mean()
        print(f"  {land:<4} {n:>5} {nz:>6}"
              f" {bp:>11.3f}")

    # Add missing_votes for BSW=0 precincts
    reg["missing_votes"] = np.where(
        reg["bsw_votes"] == 0,
        reg["bsw_pred_pct"]/100 * reg["valid_total"],
        0).round(1)

    # Top 20 by missing votes
    zeros = reg[reg["bsw_votes"] == 0]
    top = zeros.sort_values(
        "missing_votes", ascending=False).head(20)
    print(f"\n  Top 20 by missing votes:")
    print(f"  {'Land':<4} {'WKR':>4} {'BD':>4}"
          f" {'Valid':>5} {'Miss':>5} {'P(0)':>8}")
    for _, r in top.iterrows():
        print(f"  {r.land:<4} {r.wahlkreis:>4}"
              f" {r.bd_votes:>4} {r.valid_total:>5}"
              f" {r.missing_votes:>5.0f}"
              f" {r.p_bsw_zero:>8.2e}")

    # Calibration: obs vs exp zeros per Land
    all_sc = compute_scores(df, pred)
    all_land = all_sc["land"].values
    all_bsw = all_sc["bsw_votes"].values
    all_p0 = all_sc["p_bsw_zero"].values
    print(f"\n  Calibration (obs vs exp zeros):")
    print(f"  {'Land':<4} {'obs':>5} {'exp':>7}"
          f" {'excess':>7}")
    for land in sorted(set(all_land)):
        m = all_land == land
        obs = (all_bsw[m] == 0).sum()
        exp = all_p0[m].sum()
        print(f"  {land:<4} {obs:>5} {exp:>7.1f}"
              f" {obs-exp:>+7.1f}")

    # Known case matches
    matched = reg[reg["source"] != ""]
    print(f"\n  Known cases matched: {len(matched)}")
    for _, r in matched.iterrows():
        print(f"    {r.land} WKR={r.wahlkreis}"
              f" {r.gemeinde}: {r.claim}")

    # Save CSV (sorted by missing votes)
    out_csv = DATA / "evidence_registry.csv"
    reg = reg.sort_values(
        "missing_votes", ascending=False)
    save = reg.copy()
    save["flags"] = save["flags"].apply(
        lambda f: "|".join(f))
    save.to_csv(out_csv, index=False)
    print(f"\n  Saved {len(reg)} entries → {out_csv}")

    # Save JSON
    recs = reg.to_dict(orient="records")
    out_json = DATA / "evidence_registry.json"
    with open(out_json, "w") as fh:
        json.dump(recs, fh, indent=2,
                  default=str, ensure_ascii=False)
    print(f"  Saved JSON → {out_json}")


if __name__ == "__main__":
    main()
