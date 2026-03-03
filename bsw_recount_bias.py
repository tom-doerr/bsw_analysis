#!/usr/bin/env python3
"""Principled selection-bias analysis of BSW's 50-recount
sample (+15 BSW votes → 0.3/precinct)."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from wahlbezirk_lr import load_2025_wbz, LAND_CODE, validate_totals

DATA = Path("data")
SEP = "=" * 60
BD = "BÜNDNIS DEUTSCHLAND"
BSW_DEFICIT = 9_529
N_RECOUNT = 50
TOTAL_GAIN = 15
RATE = TOTAL_GAIN / N_RECOUNT  # 0.3
N_SIMS = 50_000


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
        DATA / "wahlbezirk_lr_predictions.csv",
        low_memory=False)
    assert len(df) == len(pred)
    validate_totals(df)
    print(f"  {len(df)} precincts loaded")
    return df, pred


def _votes(df, party):
    col = f"{party} - Zweitstimmen"
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def _valid(df):
    return df["Gültige - Zweitstimmen"].values


def bootstrap_rate(n_boot=10_000):
    """Bootstrap CI on per-precinct correction rate.
    Model each recount as Poisson(θ), observe sum=15."""
    print(f"\n{SEP}")
    print("BOOTSTRAP: Per-precinct correction rate")
    print(SEP)
    rng = np.random.RandomState(42)
    # Gamma posterior for Poisson rate
    # Prior: Gamma(0.5, 0.01) (weak)
    a = 0.5 + TOTAL_GAIN
    b = 0.01 + N_RECOUNT
    rates = rng.gamma(a, 1/b, n_boot)
    lo, med, hi = np.percentile(rates, [2.5, 50, 97.5])
    print(f"  Rate θ: {med:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"  Mean: {rates.mean():.4f}")
    return rates


def characterize_recounted(df, pred):
    """Profile the likely-recounted population."""
    print(f"\n{SEP}")
    print("POPULATION: Likely-recounted vs all")
    print(SEP)
    g = _valid(df).astype(float)
    bsw_v = _votes(df, "BSW").values
    bd_v = _votes(df, BD).values
    bsw_pred = pred["BSW_pred"].values / 100
    lam = np.maximum(bsw_pred * g, 1e-6)
    p_zero = np.exp(-lam)
    n = len(df)
    susp = (bsw_v == 0) & (p_zero < 0.01)
    zero_bd = (bsw_v == 0) & (bd_v > 0)
    print(f"  All: {n:,}")
    print(f"  BSW=0 & P<1%: {susp.sum():,}"
          f" ({susp.mean():.3%})")
    print(f"  BSW=0 & BD>0: {zero_bd.sum():,}"
          f" ({zero_bd.mean():.3%})")
    return {"n": n, "n_susp": susp.sum(),
            "n_zero_bd": zero_bd.sum()}


def sensitivity_curve(rates, n_total):
    """For each representativeness f, P(≥deficit)."""
    print(f"\n{SEP}")
    print("SENSITIVITY: Representativeness sweep")
    print(SEP)
    rng = np.random.RandomState(42)
    fracs = [0.01,0.02,0.05,0.1,0.15,
             0.2,0.3,0.5,0.75,1.0]
    rows = []
    for f in fracs:
        n_a = int(f * n_total)
        g = np.array([rng.poisson(r*n_a)
                       for r in rates])
        lo,med,hi = np.percentile(g,[5,50,95])
        pc = (g >= BSW_DEFICIT).mean()
        rows.append(dict(f=f, n_apply=n_a,
            median=med, p_cross=pc,
            ci5=lo, ci95=hi))
        print(f"  f={f:.0%} med={med:,.0f}"
              f" P={pc:.1%} [{lo:,.0f},{hi:,.0f}]")
    return pd.DataFrame(rows)


def population_compare(df, pred):
    """Compare recounted profile vs population."""
    print(f"\n{SEP}")
    print("PROFILE: Recounted vs population")
    print(SEP)
    g = _valid(df).astype(float)
    bsw_v = _votes(df, "BSW").values
    bd_v = _votes(df, BD).values
    bp = pred["BSW_pred"].values / 100
    lam = np.maximum(bp * g, 1e-6)
    p0 = np.exp(-lam)
    ba = pd.to_numeric(df["Bezirksart"],
                       errors="coerce").fillna(0)
    susp = (bsw_v == 0) & (p0 < 0.01)
    # Size distribution
    for label, mask in [("All", np.ones(len(df), bool)),
                        ("Suspicious", susp)]:
        sz = g[mask]
        print(f"  {label} (n={mask.sum():,}):"
              f" size med={np.median(sz):.0f}"
              f" mean={sz.mean():.0f}")
    # Urne vs Brief
    urne_all = (ba == 0).mean()
    urne_s = (ba[susp] == 0).mean() if susp.any() else 0
    print(f"  Urne fraction: all={urne_all:.1%}"
          f" susp={urne_s:.1%}")
    # BSW predicted share
    bp_all = pred["BSW_pred"].values.mean()
    bp_s = pred["BSW_pred"].values[susp].mean()
    print(f"  BSW pred%: all={bp_all:.2f}"
          f" susp={bp_s:.2f}")
    # Land distribution
    land = pd.to_numeric(df["Land"], errors="coerce")
    print(f"\n  Land distribution:")
    for lv in sorted(land.dropna().unique()):
        m_all = (land == lv).mean()
        m_s = ((land == lv) & susp).sum()
        nm = LAND_CODE.get(int(lv), str(lv))
        if m_s > 0:
            print(f"    {nm}: all={m_all:.1%}"
                  f" susp={m_s}")


def main():
    df, pred = load_all()
    n = len(df)
    rates = bootstrap_rate()
    pop = characterize_recounted(df, pred)
    sens = sensitivity_curve(rates, n)
    population_compare(df, pred)
    # Interpretation
    print(f"\n{SEP}\nINTERPRETATION\n{SEP}")
    n_s = pop["n_susp"]
    f_s = n_s / n
    row = sens[sens["f"] >= f_s].iloc[0]
    print(f"  Suspicious: {n_s:,} ({f_s:.3%})")
    print(f"  If rate applies to susp only:"
          f" med={row['median']:,.0f}"
          f" P={row['p_cross']:.1%}")
    sens.to_csv(DATA/"recount_bias_analysis.csv",
                index=False)
    print(f"\n  Saved → recount_bias_analysis.csv")


if __name__ == "__main__":
    main()
