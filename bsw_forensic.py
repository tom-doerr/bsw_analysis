#!/usr/bin/env python3
"""Comprehensive forensic analysis: search for evidence
of missing/undercounted BSW votes across 95k precincts."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, pearsonr, chi2_contingency
from sklearn.mixture import GaussianMixture

from wahlbezirk_lr import load_2025_wbz, LAND_CODE

DATA = Path("data")
CONTROLS = ["FDP", "Die Linke"]
SEP = "=" * 60


def load_all():
    """Load raw 2025 data + pre-computed LR predictions."""
    print("Loading data...")
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen", "Wahlberechtigte (A)",
              "Wählende (B)", "Ungültige - Zweitstimmen",
              "Bezirksart"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df[df["Gültige - Zweitstimmen"] >= 1].copy()
    df = df.reset_index(drop=True)
    pred = pd.read_csv(DATA / "wahlbezirk_lr_predictions.csv")
    assert len(df) == len(pred), f"{len(df)} != {len(pred)}"
    print(f"  {len(df)} precincts loaded")
    return df, pred


def _get_share(df, party):
    """Get Zweitstimme share for a party."""
    col = f"{party} - Zweitstimmen"
    v = pd.to_numeric(df[col], errors="coerce").fillna(0)
    g = df["Gültige - Zweitstimmen"]
    return np.where(g > 0, v / g * 100, 0)


def _turnout(df):
    wa = df["Wahlberechtigte (A)"].values
    wb = df["Wählende (B)"].values
    safe = np.where(wa > 0, wa, 1)
    return np.where(wa > 0, wb / safe * 100, np.nan)


def step01_turnout_corr(df, pred):
    print(f"\n{SEP}\nStep 1: Turnout–vote share correlation\n{SEP}")
    urne = df["Bezirksart"] == 0
    to = _turnout(df)
    parties = ["BSW"] + CONTROLS + ["AfD", "SPD", "CDU"]
    land = df["Land"].values
    # Overall
    hdr = f"{'Party':>12s}  {'Pearson':>8s}  {'Spearman':>8s}"
    print(hdr)
    for p in parties:
        s = _get_share(df, p)
        m = urne & np.isfinite(to)
        r_p = pearsonr(to[m], s[m])[0]
        r_s = spearmanr(to[m], s[m])[0]
        print(f"{p:>12s}  {r_p:+8.4f}  {r_s:+8.4f}")
    # Per Land for BSW
    bsw_s = _get_share(df, "BSW")
    print(f"\nBSW turnout correlation by Land:")
    print(f"{'Land':>4s}  {'r':>8s}  {'N':>6s}")
    for l in sorted(set(land)):
        m = urne & (land == l) & np.isfinite(to)
        if m.sum() < 50:
            continue
        r = pearsonr(to[m], bsw_s[m])[0]
        print(f"{LAND_CODE.get(l,l):>4s}  {r:+8.4f}  {m.sum():6d}")


def step02_brief_vs_urne(df, pred):
    print(f"\n{SEP}\nStep 2: Briefwahl vs Urne\n{SEP}")
    ba = df["Bezirksart"].values
    urne = ba == 0
    brief = ba == 5
    land = df["Land"].values
    parties = ["BSW"] + CONTROLS
    for p in parties:
        s = _get_share(df, p)
        r = pred[f"{p}_resid"].values
        print(f"\n  {p}:")
        hdr = f"  {'Land':>4s}  {'Urne':>7s}  {'Brief':>7s}"
        hdr += f"  {'Δ':>7s}  {'t':>7s}"
        print(hdr)
        for l in sorted(set(land)):
            mu = urne & (land == l)
            mb = brief & (land == l)
            if mu.sum() < 30 or mb.sum() < 30:
                continue
            su, sb = s[mu], s[mb]
            d = su.mean() - sb.mean()
            t, _ = stats.ttest_ind(su, sb)
            lc = LAND_CODE.get(l, str(l))
            print(f"  {lc:>4s}  {su.mean():7.2f}  {sb.mean():7.2f}"
                  f"  {d:+7.2f}  {t:+7.2f}")
    # Residual comparison
    print(f"\n  BSW residual mean: Urne vs Brief")
    rb = pred["BSW_resid"].values
    print(f"    Urne: {rb[urne].mean():+.4f}pp  "
          f"Brief: {rb[brief].mean():+.4f}pp")


def _benford_2nd(votes):
    """2nd-digit Benford test on vote counts >= 10."""
    v = votes[votes >= 10].astype(int)
    digits = (v // (10 ** (np.floor(np.log10(v)) - 1))) % 10
    obs = np.array([(digits == d).sum() for d in range(10)])
    # Benford 2nd digit expected probabilities
    exp_p = np.zeros(10)
    for d in range(10):
        for k in range(1, 10):
            exp_p[d] += np.log10(1 + 1 / (10*k + d))
    exp = exp_p * obs.sum()
    chi2 = ((obs - exp)**2 / exp).sum()
    p = 1 - stats.chi2.cdf(chi2, df=9)
    return obs, exp, chi2, p


def step03_benford_2nd(df):
    print(f"\n{SEP}\nStep 3: Second-digit Benford's Law\n{SEP}")
    parties = ["BSW"] + CONTROLS + ["AfD"]
    for p in parties:
        col = f"{p} - Zweitstimmen"
        v = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        obs, exp, chi2, pv = _benford_2nd(v)
        n = (v >= 10).sum()
        print(f"\n  {p} (n={n}):")
        print(f"    χ²={chi2:.2f}  p={pv:.4f}")
        print(f"    {'Dig':>4s} {'Obs':>6s} {'Exp':>8s} {'Δ':>7s}")
        for d in range(10):
            print(f"    {d:4d} {obs[d]:6d} {exp[d]:8.1f}"
                  f" {obs[d]-exp[d]:+7.1f}")


def step04_size_stratified(df, pred):
    print(f"\n{SEP}\nStep 4: Precinct-size stratification\n{SEP}")
    urne = df["Bezirksart"] == 0
    wa = df["Wahlberechtigte (A)"].values
    parties = ["BSW"] + CONTROLS
    for p in parties:
        s = _get_share(df, p)
        r = pred[f"{p}_resid"].values
        m = urne & (wa > 0)
        q = pd.qcut(wa[m], 5, labels=False, duplicates="drop")
        print(f"\n  {p} (Urne, quintiles of WA):")
        hdr = f"  {'Q':>2s}  {'WA_lo':>6s}  {'WA_hi':>6s}"
        hdr += f"  {'Share':>7s}  {'Resid':>7s}  {'N':>6s}"
        print(hdr)
        wa_m, s_m, r_m = wa[m], s[m], r[m]
        for qi in range(5):
            mask = q == qi
            lo = wa_m[mask].min()
            hi = wa_m[mask].max()
            print(f"  {qi:2d}  {lo:6.0f}  {hi:6.0f}"
                  f"  {s_m[mask].mean():7.2f}"
                  f"  {r_m[mask].mean():+7.3f}"
                  f"  {mask.sum():6d}")
        rho = spearmanr(q, r_m)[0]
        print(f"  Spearman(quintile, resid): {rho:+.4f}")


def step05_invalid_corr(df, pred):
    print(f"\n{SEP}\nStep 5: Invalid vote correlation\n{SEP}")
    zu = pd.to_numeric(df["Ungültige - Zweitstimmen"],
                        errors="coerce").fillna(0).values
    zg = df["Gültige - Zweitstimmen"].values
    inv_rate = np.where((zu+zg) > 0, zu/(zu+zg)*100, 0)
    land = df["Land"].values
    parties = ["BSW"] + CONTROLS
    print(f"  Overall correlations:")
    hdr = f"  {'Party':>12s}  {'r(inv,share)':>12s}"
    hdr += f"  {'r(inv,resid)':>12s}"
    print(hdr)
    for p in parties:
        s = _get_share(df, p)
        r = pred[f"{p}_resid"].values
        rs = pearsonr(inv_rate, s)[0]
        rr = pearsonr(inv_rate, r)[0]
        print(f"  {p:>12s}  {rs:+12.4f}  {rr:+12.4f}")
    # Per Land for BSW
    bsw_r = pred["BSW_resid"].values
    print(f"\n  BSW inv-resid correlation by Land:")
    for l in sorted(set(land)):
        m = land == l
        if m.sum() < 50:
            continue
        r = pearsonr(inv_rate[m], bsw_r[m])[0]
        lc = LAND_CODE.get(l, str(l))
        print(f"    {lc:>4s}: {r:+.4f} (n={m.sum()})")


def step06_multimodality(pred):
    print(f"\n{SEP}\nStep 6: Multimodality (KDE peak count)\n{SEP}")
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde
    parties = ["BSW"] + CONTROLS
    for p in parties:
        r = pred[f"{p}_resid"].values
        r = r[np.isfinite(r)]
        kde = gaussian_kde(r, bw_method="silverman")
        x = np.linspace(r.min(), r.max(), 1000)
        y = kde(x)
        peaks, _ = find_peaks(y, prominence=0.01 * y.max())
        print(f"  {p}: {len(peaks)} peak(s) at "
              f"{', '.join(f'{x[i]:.2f}' for i in peaks)}")


def step07_kurtosis_skew(pred):
    print(f"\n{SEP}\nStep 7: Kurtosis & skewness\n{SEP}")
    parties = sorted(set(
        c.replace("_resid", "") for c in pred.columns
        if c.endswith("_resid")
    ))
    rows = []
    for p in parties:
        r = pred[f"{p}_resid"].values
        s = pred[f"{p}_actual"].values
        rows.append({"party": p, "mean": s.mean(),
                      "skew": stats.skew(r),
                      "kurt": stats.kurtosis(r)})
    tbl = pd.DataFrame(rows).sort_values("skew")
    print(tbl.to_string(index=False, float_format="{:.3f}".format))
    bsw = tbl[tbl["party"] == "BSW"].iloc[0]
    rk_s = (tbl["skew"] <= bsw["skew"]).sum()
    rk_k = (tbl["kurt"] >= bsw["kurt"]).sum()
    print(f"\n  BSW skew rank: {rk_s}/{len(tbl)}")
    print(f"  BSW kurtosis rank: {rk_k}/{len(tbl)}")


def step08_geo_cluster(df, pred):
    print(f"\n{SEP}\nStep 8: Geographic clustering\n{SEP}")
    wkr = pd.to_numeric(df["Wahlkreis"], errors="coerce")
    wkr = wkr.fillna(0).astype(int).values
    land = df["Land"].values
    rb = pred["BSW_resid"].values
    # Per Wahlkreis stats
    rows = []
    for w in sorted(set(wkr)):
        if w == 0:
            continue
        m = wkr == w
        n = m.sum()
        if n < 5:
            continue
        rows.append({
            "WKR": w, "Land": LAND_CODE.get(land[m][0], "?"),
            "N": n, "mean_r": rb[m].mean(),
            "frac_neg": (rb[m] < 0).mean(),
        })
    wdf = pd.DataFrame(rows)
    mu = wdf["mean_r"].mean()
    sd = wdf["mean_r"].std()
    wdf["z"] = (wdf["mean_r"] - mu) / sd
    flagged = wdf[wdf["z"] < -2].sort_values("z")
    print(f"  {len(wdf)} Wahlkreise analyzed")
    print(f"  Flagged (z<-2): {len(flagged)}")
    if len(flagged):
        print(flagged[["WKR", "Land", "N", "mean_r", "z"]]
              .to_string(index=False, float_format="{:.3f}".format))
    # Check controls in flagged WKR
    for p in CONTROLS:
        rp = pred[f"{p}_resid"].values
        fmean = np.mean([rp[wkr == w].mean()
                         for w in flagged["WKR"]])
        print(f"  {p} mean resid in flagged WKR: {fmean:+.3f}")


def step09_zero_vote(df, pred):
    print(f"\n{SEP}\nStep 9: Zero-vote deep dive\n{SEP}")
    parties = ["BSW"] + CONTROLS
    zg = df["Gültige - Zweitstimmen"].values
    for p in parties:
        col = f"{p} - Zweitstimmen"
        v = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        pr = pred[f"{p}_pred"].values
        zeros = v == 0
        lam = pr / 100 * zg
        # Poisson P(0) = exp(-lambda)
        p0 = np.exp(-np.clip(lam, 0, 500))
        susp = zeros & (pr > 3) & (zg > 100) & (p0 < 0.01)
        print(f"\n  {p}:")
        print(f"    Total zeros: {zeros.sum()}")
        print(f"    Suspicious (pred>3%, valid>100,"
              f" P<0.01): {susp.sum()}")
        if susp.sum() > 0 and p == "BSW":
            land = df["Land"].values
            print(f"    By Land:")
            for l in sorted(set(land[susp])):
                n = ((land == l) & susp).sum()
                lc = LAND_CODE.get(l, str(l))
                print(f"      {lc}: {n}")


def step10_gmm(pred):
    print(f"\n{SEP}\nStep 10: Gaussian Mixture Model\n{SEP}")
    parties = ["BSW"] + CONTROLS
    for p in parties:
        r = pred[f"{p}_resid"].values.reshape(-1, 1)
        g1 = GaussianMixture(1, random_state=42).fit(r)
        g2 = GaussianMixture(2, random_state=42).fit(r)
        print(f"\n  {p}:")
        print(f"    BIC(1): {g1.bic(r):.0f}  "
              f"BIC(2): {g2.bic(r):.0f}  "
              f"ΔBIC: {g1.bic(r)-g2.bic(r):+.0f}")
        if g2.bic(r) < g1.bic(r):
            mu = g2.means_.flatten()
            sd = np.sqrt(g2.covariances_.flatten())
            w = g2.weights_
            o = np.argsort(mu)
            for i, j in enumerate(o):
                print(f"    Comp{i}: μ={mu[j]:+.3f} "
                      f"σ={sd[j]:.3f} w={w[j]:.3f}")


def step11_feature_importance(df, pred):
    print(f"\n{SEP}\nStep 11: Feature importance\n{SEP}")
    rb = pred["BSW_resid"].values
    feats = {}
    feats["turnout"] = _turnout(df)
    feats["Bezirksart"] = df["Bezirksart"].values
    wa = df["Wahlberechtigte (A)"].values
    feats["log_voters"] = np.log1p(wa)
    zu = pd.to_numeric(df["Ungültige - Zweitstimmen"],
                        errors="coerce").fillna(0).values
    zg = df["Gültige - Zweitstimmen"].values
    feats["inv_rate"] = np.where((zu+zg)>0, zu/(zu+zg)*100, 0)
    land = df["Land"].values
    for l in sorted(set(land)):
        feats[f"land_{LAND_CODE.get(l,l)}"] = (land == l) * 1.0
    rows = []
    for name, vals in feats.items():
        m = np.isfinite(vals) & np.isfinite(rb)
        if m.sum() < 100:
            continue
        r = pearsonr(vals[m], rb[m])[0]
        rows.append({"feature": name, "corr": r})
    tbl = pd.DataFrame(rows)
    tbl = tbl.reindex(tbl["corr"].abs().sort_values(
        ascending=False).index)
    print(tbl.head(15).to_string(
        index=False, float_format="{:.4f}".format))


def main():
    df, pred = load_all()
    step01_turnout_corr(df, pred)
    step02_brief_vs_urne(df, pred)
    step03_benford_2nd(df)
    step04_size_stratified(df, pred)
    step05_invalid_corr(df, pred)
    step06_multimodality(pred)
    step07_kurtosis_skew(pred)
    step08_geo_cluster(df, pred)
    step09_zero_vote(df, pred)
    step10_gmm(pred)
    step11_feature_importance(df, pred)
    print(f"\n{SEP}\nDone — 11 forensic tests completed\n{SEP}")


if __name__ == "__main__":
    main()
