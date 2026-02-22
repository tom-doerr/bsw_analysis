#!/usr/bin/env python3
"""Test BSW's specific claims about vote miscounting
in the 2025 Bundestagswahl. BSW got 4.981%, missing
the 5% threshold by 9,529 votes."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import (spearmanr, pearsonr,
    chi2_contingency, poisson, mannwhitneyu)

from wahlbezirk_lr import load_2025_wbz, LAND_CODE

DATA = Path("data")
SEP = "=" * 60
CONTROLS = ["FDP", "Die Linke"]
BSW_DEFICIT = 9_529
BD = "BÜNDNIS DEUTSCHLAND"
LAND = {1:"SH",2:"HH",3:"NI",4:"HB",5:"NW",6:"HE",
        7:"RP",8:"BW",9:"BY",10:"SL",11:"BE",12:"BB",
        13:"MV",14:"SN",15:"ST",16:"TH"}


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
    print(f"  {len(df)} precincts loaded")
    return df, pred


def _votes(df, party):
    col = f"{party} - Zweitstimmen"
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def _valid(df):
    return df["Gültige - Zweitstimmen"].values


def _share(df, party):
    v = _votes(df, party).values
    g = _valid(df)
    return np.where(g > 0, v / g * 100, 0)


def _urne(df):
    return df["Bezirksart"] == 0


def claim1a_joint_residuals(df, pred):
    """Contingency: do BSW and BD residuals anti-correlate?"""
    print(f"\n{SEP}")
    print("CLAIM 1a: Joint BSW↔BD Residual Analysis")
    print(SEP)
    bsw_r = pred["BSW_resid"].values
    bd_r = pred[f"{BD}_resid"].values
    r, p = pearsonr(bsw_r, bd_r)
    print(f"  Pearson r(BSW,BD resid): {r:+.4f} (p={p:.4f})")
    ct = pd.crosstab(bsw_r >= 0, bd_r >= 0)
    chi2, pv, _, _ = chi2_contingency(ct)
    print(f"  Chi²(sign): {chi2:.1f} (p={pv:.4f})")
    for cp in CONTROLS:
        rc, _ = pearsonr(pred[f"{cp}_resid"].values, bd_r)
        print(f"  Control r({cp},BD): {rc:+.4f}")
    v = "NO swap signature" if r > -0.05 else "POSSIBLE swap"
    print(f"  → {v}")


def claim1b_per_land(df, pred):
    """Per-Land BSW↔BD residual correlation."""
    print(f"\n{SEP}")
    print("CLAIM 1b: Per-Land BSW↔BD Correlation")
    print(SEP)
    pred["Land"] = df["Land"].values
    print(f"  {'Land':<4} {'n':>6} {'r(BSW,BD)':>10} {'p':>8}")
    sig = 0
    for land, g in pred.groupby("Land"):
        if len(g) < 30:
            continue
        r, p = pearsonr(g["BSW_resid"], g[f"{BD}_resid"])
        flag = " *" if p < 0.05 and r < 0 else ""
        nm = LAND.get(land, str(land))
        print(f"  {nm:<4} {len(g):>6} {r:>+10.4f} {p:>8.4f}{flag}")
        if p < 0.05 and r < -0.05:
            sig += 1
    print(f"  → {sig} Länder with significant negative r")


def claim1c_swap_sim(df, pred):
    """Move fraction f of BD→BSW, check if crosses 5%."""
    print(f"\n{SEP}\nCLAIM 1c: Vote-Swap Simulation\n{SEP}")
    bsw_tot = _votes(df, "BSW").sum()
    bd_col = "Bündnis C - Zweitstimmen"
    bd_tot = pd.to_numeric(df[bd_col], errors="coerce").sum()
    total = _valid(df).sum()
    thr = total * 0.05
    print(f"  BSW:{bsw_tot:,.0f} BD:{bd_tot:,.0f} 5%:{thr:,.0f}")
    for f in [0.05, 0.1, 0.125, 0.2, 0.5, 1.0]:
        new = bsw_tot + f * bd_tot
        c = "YES" if new >= thr else "no"
        print(f"  f={f:.3f}: {new:,.0f} ({new/total*100:.3f}%) {c}")
    print(f"  → Need {(thr-bsw_tot)/bd_tot:.1%} of BD for 5%")


def claim2a_zero_poisson(df, pred):
    """Expected vs observed BSW=0 Urne precincts."""
    print(f"\n{SEP}\nCLAIM 2a: Zero-Vote Poisson Analysis\n{SEP}")
    u = _urne(df)
    g = _valid(df)
    for party in ["BSW"] + CONTROLS:
        v = _votes(df, party).values
        zeros = ((v == 0) & u).sum()
        pr = pred[f"{party}_pred"].values / 100
        lam = np.maximum(pr * g, 0)
        exp_z = np.exp(-lam[u]).sum()
        ratio = zeros / max(exp_z, 1)
        print(f"  {party:<12} obs={zeros:>5}  "
              f"exp={exp_z:>7.0f}  ratio={ratio:.2f}x")
    bsw_lam = np.maximum(pred["BSW_pred"].values / 100 * g, 0)
    p0 = np.exp(-bsw_lam)
    susp = ((p0 < 0.01) & u & (_votes(df, "BSW").values == 0))
    print(f"  Suspicious BSW=0 (P<1%): {susp.sum()}")


def claim2b_bd_in_zeros(df, pred):
    """BD share in BSW=0 vs BSW>0, size-controlled."""
    print(f"\n{SEP}\nCLAIM 2b: BD Share in BSW=0 Precincts\n{SEP}")
    u = _urne(df)
    bsw0 = (_votes(df, "BSW").values == 0) & u
    bsw1 = (_votes(df, "BSW").values > 0) & u
    bd_s = _share(df, "Bündnis C")
    m0, m1 = bd_s[bsw0].mean(), bd_s[bsw1].mean()
    print(f"  BD: BSW=0 {m0:.3f}% vs BSW>0 {m1:.3f}%"
          f" ({m0/m1:.2f}x)")
    print("  Control ratios (BSW=0 / BSW>0):")
    for p in ["CSU", "FREIE WÄHLER", "AfD", "SPD"]:
        s = _share(df, p)
        r = s[bsw0].mean() / max(s[bsw1].mean(), 0.001)
        print(f"    {p:<15} {r:.2f}x")
    print("  → If BD ratio < other ratios: size effect, not swap")


def claim2c_zero_profile(df, pred):
    """Profile BSW=0 precincts: size, Land, max impact."""
    print(f"\n{SEP}\nCLAIM 2c: Zero-Vote Profile\n{SEP}")
    u = _urne(df); g = _valid(df)
    bsw0 = (_votes(df, "BSW").values == 0) & u
    n0 = bsw0.sum()
    print(f"  BSW=0 Urne: {n0}  mean size: {g[bsw0].mean():.0f}"
          f" vs {g[u & ~bsw0].mean():.0f}")
    lands = df.loc[bsw0, "Land"].value_counts().head(5)
    for land, cnt in lands.items():
        print(f"    {LAND.get(land,land)}: {cnt} ({cnt/n0*100:.1f}%)")
    mx = g[bsw0].sum() * 0.04981
    print(f"  Max impact (all→4.98%): +{mx:,.0f} vs {BSW_DEFICIT:,}"
          f" needed → {'YES' if mx >= BSW_DEFICIT else 'NO'}")


def claim3a_extrapolation(df, pred):
    """Extrapolation arithmetic and bias."""
    print(f"\n{SEP}\nCLAIM 3: Correction Extrapolation\n{SEP}")
    n = len(df); u_n = _urne(df).sum()
    bsw0 = ((_votes(df,"BSW").values==0) & _urne(df)).sum()
    g = _valid(df)
    lam = np.maximum(pred["BSW_pred"].values/100*g, 0)
    susp = ((np.exp(-lam)<0.01) & _urne(df)
            & (_votes(df,"BSW").values==0)).sum()
    print("  BSW: 0.3 votes/precinct from 50 recounts")
    for nm, c in [("All",n),("Urne",u_n),
                  ("BSW=0",bsw0),("Susp",susp)]:
        ex = c*0.3
        print(f"  {nm:<8} {c:>7,} ×0.3={ex:>8,.0f} "
              f"{'Y' if ex>=BSW_DEFICIT else 'N'}")
    print("  50 recounts NOT random (BSW-selected)")
    # Bootstrap: CI on 0.3 estimate
    se = np.sqrt(0.3 / 50)  # Poisson SE
    lo, hi = 0.3 - 1.96*se, 0.3 + 1.96*se
    print(f"  95% CI on 0.3/precinct: [{lo:.3f}, {hi:.3f}]")
    print(f"  Need per precinct: {BSW_DEFICIT/n:.4f}")


def claim4_corrections(df, pred):
    """Disproportionate corrections context."""
    print(f"\n{SEP}\nCLAIM 4: Disproportionate Corrections\n{SEP}")
    total = _valid(df).sum()
    bsw_share = _votes(df, "BSW").sum() / total
    print(f"  BSW national share: {bsw_share*100:.3f}%")
    print(f"  Corrections: 4,277 / 7,425 = 57.6% to BSW")
    exp = 7425 * bsw_share
    print(f"  Expected if random: {exp:.0f} ({bsw_share*100:.1f}%)")
    print(f"  But: recounted precincts selected BY BSW")
    print(f"  Selection bias makes this test uninformative")
    print(f"  Context: 4,277 = {4277/total*100:.4f}% of all votes")


def main():
    df, pred = load_all()
    total = _valid(df).sum()
    bsw_tot = _votes(df, "BSW").sum()
    print(f"\n{SEP}")
    print("BSW CLAIMS ANALYSIS — 2025 Bundestagswahl")
    print(SEP)
    print(f"Precincts: {len(df):,}  BSW: {bsw_tot:,} "
          f"({bsw_tot/total*100:.3f}%)")
    print(f"Deficit to 5%: {BSW_DEFICIT:,} votes")
    claim1a_joint_residuals(df, pred)
    claim1b_per_land(df, pred)
    claim1c_swap_sim(df, pred)
    claim2a_zero_poisson(df, pred)
    claim2b_bd_in_zeros(df, pred)
    claim2c_zero_profile(df, pred)
    claim3a_extrapolation(df, pred)
    claim4_corrections(df, pred)
    print(f"\n{SEP}\nDONE\n{SEP}")


if __name__ == "__main__":
    main()
