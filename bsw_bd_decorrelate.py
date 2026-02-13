#!/usr/bin/env python3
"""Decorrelation analysis: predict BSW+BD sum, then
examine within-sum split for vote swapping evidence."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wahlbezirk_lr import (
    load_2025_wbz, load_2021_wbz, load_2017_wbz,
    prep_2025, agg_21_to_wkr, agg_17_to_wkr,
    build_X_base, SEED,
)


def make_pipe():
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LinearRegression()),
    ])


def load_all():
    print("Loading data...")
    df25 = load_2025_wbz()
    e_feat, z_map, struct25, wkr, land, meta = prep_2025(df25)
    df21 = load_2021_wbz()
    hist21 = agg_21_to_wkr(df21)
    erst17, zweit17 = load_2017_wbz()
    hist17 = agg_17_to_wkr(erst17, zweit17)
    print("Building features...")
    base = build_X_base(e_feat, struct25, wkr, land, hist21, hist17)
    X = base.values.astype(np.float64)
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    return X, z_map, meta, cv


def step1_predictability(X, z_map, cv):
    bsw = z_map["BSW"]
    bd = z_map["BÜNDNIS DEUTSCHLAND"]
    combo = bsw + bd
    pb = cross_val_predict(make_pipe(), X, bsw, cv=cv)
    pd_ = cross_val_predict(make_pipe(), X, bd, cv=cv)
    pc = cross_val_predict(make_pipe(), X, combo, cv=cv)
    print(f"\n{'='*60}")
    print("Step 1: Predictability comparison")
    print(f"{'='*60}")
    for lbl, y, p in [("BSW", bsw, pb),
                       ("BD", bd, pd_),
                       ("BSW+BD", combo, pc)]:
        r2 = r2_score(y, p)
        rho = spearmanr(y, p)[0]
        mae = mean_absolute_error(y, p)
        print(f"  {lbl:10s} R²={r2:.6f}  "
              f"ρ={rho:.4f}  MAE={mae:.4f}pp")
    delta = r2_score(combo, pc) - r2_score(bsw, pb)
    print(f"\n  R²(sum) - R²(BSW) = {delta:+.6f}")
    print(f"  Positive = sum more predictable")
    return bsw, bd, combo, pb, pd_, pc


def step2_within_sum(X, bsw, bd, combo, cv):
    print(f"\n{'='*60}")
    print("Step 2: Within-sum split analysis")
    print(f"{'='*60}")
    mask = combo > 0
    frac = np.where(mask, bsw/np.where(mask, combo, 1), np.nan)
    print(f"  N(BSW+BD>0): {mask.sum()}")
    print(f"  Mean BSW/(BSW+BD): {np.nanmean(frac):.4f}")
    print(f"  Std  BSW/(BSW+BD): {np.nanstd(frac):.4f}")
    valid = mask & np.isfinite(frac)
    pf = cross_val_predict(make_pipe(), X[valid],
                            frac[valid], cv=cv)
    r2f = r2_score(frac[valid], pf)
    rhof = spearmanr(frac[valid], pf)[0]
    print(f"  Predicting BSW/(BSW+BD) fraction:")
    print(f"    R²={r2f:.6f}  ρ={rhof:.4f}")
    return frac, valid


def step3_residuals(bsw, bd, combo, pb, pd_, pc):
    print(f"\n{'='*60}")
    print("Step 3: Residual analysis after decorrelation")
    print(f"{'='*60}")
    rb = bsw - pb
    rd = bd - pd_
    rc = combo - pc
    r_raw = np.corrcoef(rb, rd)[0, 1]
    print(f"  Raw residual corr(BSW,BD): {r_raw:+.6f}")
    # Orthogonalize w.r.t. combo residual
    proj_b = rb - rc * (np.dot(rb, rc) / np.dot(rc, rc))
    proj_d = rd - rc * (np.dot(rd, rc) / np.dot(rc, rc))
    r_orth = np.corrcoef(proj_b, proj_d)[0, 1]
    print(f"  Orthogonalized corr:       {r_orth:+.6f}")
    print(f"  Vote-swap → expect strong negative")


def step4_deviation_dist(bsw, combo, frac, valid):
    print(f"\n{'='*60}")
    print("Step 4: Within-sum deviation distribution")
    print(f"{'='*60}")
    mf = np.nanmean(frac)
    dev = bsw - combo * mf
    print(f"  Mean fraction: {mf:.4f}")
    print(f"  Deviation (actual-expected BSW):")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        v = np.nanpercentile(dev[valid], p)
        print(f"    P{p:02d}: {v:+.4f}pp")


def step5_land_breakdown(X, bsw, combo, frac, valid, meta, cv):
    print(f"\n{'='*60}")
    print("Step 5: BSW fraction by Bundesland")
    print(f"{'='*60}")
    land = meta["Land"].values
    hdr = f"{'Land':>4s}  {'Mean':>8s}  {'Std':>8s}  {'N':>6s}"
    print(hdr)
    for l in sorted(set(land)):
        m = (land == l) & valid
        if m.sum() < 50:
            continue
        fr = frac[m]
        print(f"{str(l):>4s}  {np.nanmean(fr):8.4f}"
              f"  {np.nanstd(fr):8.4f}  {m.sum():6d}")


def step6_controls(X, z_map, cv):
    print(f"\n{'='*60}")
    print("Step 6: Control — other party pairs")
    print(f"{'='*60}")
    pairs = [
        ("BSW", "BÜNDNIS DEUTSCHLAND"),
        ("BSW", "Die Linke"),
        ("BSW", "FDP"),
        ("Die Linke", "MLPD"),
        ("CDU", "FREIE WÄHLER"),
        ("SPD", "Die Linke"),
    ]
    hdr = f"{'Pair':>30s}  {'R²_A':>7s}  {'R²_B':>7s}"
    hdr += f"  {'R²_sum':>7s}  {'Δ':>8s}"
    print(hdr)
    for pa, pb in pairs:
        if pa not in z_map or pb not in z_map:
            continue
        ya, yb = z_map[pa], z_map[pb]
        ys = ya + yb
        ppa = cross_val_predict(make_pipe(), X, ya, cv=cv)
        ppb = cross_val_predict(make_pipe(), X, yb, cv=cv)
        pps = cross_val_predict(make_pipe(), X, ys, cv=cv)
        r2a = r2_score(ya, ppa)
        r2b = r2_score(yb, ppb)
        r2s = r2_score(ys, pps)
        lbl = f"{pa}+{pb}"
        print(f"{lbl:>30s}  {r2a:7.4f}  {r2b:7.4f}"
              f"  {r2s:7.4f}  {r2s-r2a:+8.4f}")


def main():
    X, z_map, meta, cv = load_all()
    out = step1_predictability(X, z_map, cv)
    bsw, bd, combo, pb, pd_, pc = out
    frac, valid = step2_within_sum(X, bsw, bd, combo, cv)
    step3_residuals(bsw, bd, combo, pb, pd_, pc)
    step4_deviation_dist(bsw, combo, frac, valid)
    step5_land_breakdown(X, bsw, combo, frac, valid, meta, cv)
    step6_controls(X, z_map, cv)
    print(f"\n{'='*60}")
    print("Interpretation")
    print(f"{'='*60}")
    print("If BSW↔BD vote-swapping exists:")
    print("  1. R²(sum) >> R²(BSW)")
    print("  2. Orthogonalized corr << 0")
    print("  3. BSW+BD stands out vs controls")


if __name__ == "__main__":
    main()
