#!/usr/bin/env python3
"""Triangulate LR vs XGB: compare suspicious precinct
overlap to defuse 'model artifact' criticism."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import betaln

DATA = Path("data")
SEP = "=" * 60


def load_data():
    from wahlbezirk_lr import (load_2025_wbz,
        validate_totals)
    print("Loading data...")
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen"]:
        df[c] = pd.to_numeric(
            df[c], errors="coerce").fillna(0)
    df = df[df["Gültige - Zweitstimmen"]>=1]
    df = df.copy().reset_index(drop=True)
    validate_totals(df)
    return df


def get_lr_pred():
    """Load existing LR predictions."""
    return pd.read_csv(
        DATA/"wahlbezirk_lr_predictions.csv",
        low_memory=False)


def get_xgb_pred(df):
    """Generate XGB CV predictions for BSW only."""
    from xgb_enhanced import load_all, train_party
    from sklearn.model_selection import KFold
    from wahlbezirk_lr import SEED
    print("  Training XGB for BSW...")
    X_df, z_map, _ = load_all()
    X = X_df.values.astype(np.float64)
    cv = KFold(10, shuffle=True, random_state=SEED)
    yp = train_party(X, z_map["BSW"], cv)
    y = z_map["BSW"]
    return y, yp


def bb_p0(n, p, rho):
    """Beta-Binomial P(X=0)."""
    phi = max(1/rho - 1, 1e-6)
    a = np.maximum(p*phi, 1e-10)
    b = np.maximum((1-p)*phi, 1e-10)
    lp = betaln(a, b+n) - betaln(a, b)
    return np.exp(np.clip(lp, -700, 0))


def find_suspicious(g, bsw, pred_pct, rho=0.003):
    """Flag suspicious zeros: BSW=0, BetaBinom P<0.01."""
    bp = np.clip(pred_pct/100, 1e-8, 1-1e-8)
    p0 = bb_p0(g, bp, rho)
    susp = (bsw == 0) & (p0 < 0.01)
    miss = np.where(bsw == 0, bp * g, 0)
    return susp, miss, p0


def compare(df, lr_pred, xgb_pred):
    """Compare LR vs XGB suspicious sets."""
    print(f"\n{SEP}\nTRIANGULATION: LR vs XGB\n{SEP}")
    g = df["Gültige - Zweitstimmen"].values.astype(float)
    bsw = pd.to_numeric(
        df["BSW - Zweitstimmen"],
        errors="coerce").fillna(0).values
    s_lr, m_lr, _ = find_suspicious(g, bsw, lr_pred)
    s_xgb, m_xgb, _ = find_suspicious(g, bsw, xgb_pred)
    n_lr = s_lr.sum(); n_xgb = s_xgb.sum()
    both = (s_lr & s_xgb).sum()
    either = (s_lr | s_xgb).sum()
    jacc = both / max(either, 1)
    print(f"  LR suspicious: {n_lr}")
    print(f"  XGB suspicious: {n_xgb}")
    print(f"  Both: {both}")
    print(f"  Jaccard overlap: {jacc:.3f}")
    return s_lr, s_xgb, m_lr, m_xgb


def rank_corr(m_lr, m_xgb):
    """Correlation of missing_votes rankings."""
    from scipy.stats import spearmanr, pearsonr
    # Only where either model flags something
    mask = (m_lr > 0) | (m_xgb > 0)
    if mask.sum() < 10:
        print("  Too few flagged for correlation")
        return
    rho, p = spearmanr(m_lr[mask], m_xgb[mask])
    r, _ = pearsonr(m_lr[mask], m_xgb[mask])
    print(f"\n  Missing votes correlation:")
    print(f"    Spearman ρ = {rho:.3f} (p={p:.2e})")
    print(f"    Pearson r  = {r:.3f}")
    return rho


def top_n_overlap(m_lr, m_xgb, ns=[20,50,100]):
    """How often top-N precincts match."""
    print(f"\n  Top-N overlap:")
    rows = []
    for n in ns:
        top_lr = set(np.argsort(m_lr)[::-1][:n])
        top_xgb = set(np.argsort(m_xgb)[::-1][:n])
        ov = len(top_lr & top_xgb)
        pct = ov / n * 100
        print(f"    Top-{n}: {ov}/{n} ({pct:.0f}%)")
        rows.append(dict(top_n=n, overlap=ov, pct=pct))
    return rows


def main():
    df = load_data()
    lr = get_lr_pred()
    lr_bsw = lr["BSW_pred"].values
    _, xgb_bsw = get_xgb_pred(df)
    s_lr, s_xgb, m_lr, m_xgb = compare(
        df, lr_bsw, xgb_bsw)
    rho = rank_corr(m_lr, m_xgb)
    ov = top_n_overlap(m_lr, m_xgb)
    # Total missing comparison
    t_lr = m_lr[m_lr > 0].sum()
    t_xgb = m_xgb[m_xgb > 0].sum()
    print(f"\n  Total missing (LR): {t_lr:,.0f}")
    print(f"  Total missing (XGB): {t_xgb:,.0f}")
    # Save
    out = pd.DataFrame(ov)
    out.to_csv(DATA/"triangulation_overlap.csv",
               index=False)
    print(f"\n  Saved → triangulation_overlap.csv")


if __name__ == "__main__":
    main()
