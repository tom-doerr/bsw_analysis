#!/usr/bin/env python3
"""Build the strongest statistical case that BSW was
systematically undercounted in the 2025 Bundestagswahl.
BSW got 4.981%, missing 5% by 9,529 votes."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, poisson

from wahlbezirk_lr import (load_2025_wbz, LAND_CODE,
                           validate_totals)

DATA = Path("data")
SEP = "=" * 60
CONTROLS = ["FDP", "Die Linke"]
BSW_DEFICIT = 9_529
BD = "BÜNDNIS DEUTSCHLAND"


def load_all():
    """Load raw 2025 data + LR predictions."""
    print("Loading data...")
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen",
              "Gültige - Erststimmen",
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


def _votes(df, party, kind="Zweitstimmen"):
    col = f"{party} - {kind}"
    return pd.to_numeric(
        df[col], errors="coerce").fillna(0)


def _valid(df, kind="Zweitstimmen"):
    return df[f"Gültige - {kind}"].values


def _share(df, party, kind="Zweitstimmen"):
    v = _votes(df, party, kind).values
    g = _valid(df, kind)
    return np.where(g > 0, v / g * 100, 0)


def _urne(df):
    return df["Bezirksart"] == 0


def counterfactual_visibility(df):
    """COUNTERFACTUAL: Ballot visibility effect.
    NOT a counting error — estimates votes BSW would
    have gained with full Erststimme coverage."""
    print(f"\n{SEP}")
    print("COUNTERFACTUAL: Ballot Visibility Effect")
    print("(NOT a counting error — campaign effect)")
    print(SEP)
    df["Land"] = pd.to_numeric(df["Land"], errors="coerce")
    erst_bsw = _votes(df, "BSW", "Erststimmen").values
    zweit_bsw = _votes(df, "BSW").values
    ge = _valid(df, "Erststimmen")
    gz = _valid(df)
    has_erst = erst_bsw > 0
    lands_with = sorted(
        df.loc[has_erst, "Land"].unique())
    print(f"  BSW Erst coverage: {len(lands_with)}"
          f"/16 Länder")
    results = {}
    print(f"  {'Land':<4} {'n':>6} {'Erst%':>7}"
          f" {'Zweit%':>7} {'Gap':>6}")
    for land in lands_with:
        m = (df["Land"] == land) & (ge > 0) & (gz > 0)
        e_pct = erst_bsw[m].sum() / ge[m].sum() * 100
        z_pct = zweit_bsw[m].sum() / gz[m].sum() * 100
        gap = e_pct - z_pct
        nm = LAND_CODE.get(int(land), str(land))
        n = m.sum()
        results[nm] = {"gap": gap, "n": n,
                       "erst": e_pct, "zweit": z_pct}
        print(f"  {nm:<4} {n:>6} {e_pct:>7.2f}"
              f" {z_pct:>7.2f} {gap:>+6.2f}")
    # Within-Land: BSW Zweit in coverage vs non-coverage WK
    print("  Within-Land visibility effect:")
    total_lost = 0
    for land in lands_with:
        lm = df["Land"] == land
        cov = lm & (erst_bsw > 0) & (gz > 0)
        ncov = lm & (erst_bsw == 0) & (gz > 0)
        if ncov.sum() < 10:
            continue
        zc = zweit_bsw[cov].sum()/gz[cov].sum()*100
        zn = zweit_bsw[ncov].sum()/gz[ncov].sum()*100
        boost = zc - zn
        lost = max(boost, 0)/100*gz[ncov].sum()
        total_lost += lost
        nm = LAND_CODE.get(int(land), str(land))
        print(f"    {nm}: cov={zc:.2f}% no={zn:.2f}%"
              f" Δ={boost:+.2f}pp lost={lost:,.0f}")
    print(f"  Total visibility-lost: {total_lost:,.0f}")
    return total_lost


def mechanism1_recount_mc(df):
    """Recount extrapolation from BSW's 50 recounts
    that found +0.3 extra votes/precinct on average."""
    print(f"\n{SEP}")
    print("MECHANISM 1: Recount Monte Carlo")
    print(SEP)
    n_recounts = 50
    mean_gain = 0.3
    n_precincts = len(df)
    n_sims = 10_000
    rng = np.random.RandomState(42)
    # Gamma posterior: shape=n*mean, rate=n
    shape = n_recounts * mean_gain
    rate = n_recounts
    # Each sim: sample per-precinct gain, sum
    totals = []
    for _ in range(n_sims):
        lam = rng.gamma(shape, 1 / rate)
        gains = rng.poisson(lam, n_precincts)
        totals.append(gains.sum())
    totals = np.array(totals)
    p5, p50, p95 = np.percentile(totals, [5, 50, 95])
    p_cross = (totals >= BSW_DEFICIT).mean()
    print(f"  Raw (no bias discount):")
    print(f"    Median gain: {p50:,.0f}")
    print(f"    90% CI: [{p5:,.0f}, {p95:,.0f}]")
    print(f"    P(gain≥{BSW_DEFICIT:,}): {p_cross:.3f}")
    # With selection bias discounts
    results = {}
    for disc in [0.5, 0.25, 0.1]:
        adj = totals * disc
        pc = (adj >= BSW_DEFICIT).mean()
        med = np.median(adj)
        results[disc] = {"median": med, "p": pc}
        print(f"  Discount {disc:.0%}: median="
              f"{med:,.0f}, P(cross)={pc:.3f}")
    return {"raw_median": p50, "p_cross_raw": p_cross,
            "results": results}


def mechanism2_bd_adjacency(df, pred):
    """BD ballot-adjacency misattribution by Land.
    BSW(26) and BD(27) adjacent on ballot."""
    print(f"\n{SEP}")
    print("MECHANISM 2: BD Ballot-Adjacency by Land")
    print(SEP)
    bd_s = _share(df, BD)
    bsw_r = pred["BSW_resid"].values
    land = pd.to_numeric(df["Land"], errors="coerce")
    print(f"  {'Ld':<4} {'n':>6} {'r(resid,BD)':>12}"
          f" {'BD%':>6} {'BD/BSW':>7}")
    bd_total = _votes(df, BD).values.sum()
    for lv in sorted(land.dropna().unique()):
        m = (land == lv).values
        if m.sum() < 30:
            continue
        r, p = pearsonr(bsw_r[m], bd_s[m])
        bd_mean = bd_s[m].mean()
        bsw_mean = _share(df, "BSW")[m].mean()
        ratio = bd_mean / max(bsw_mean, 0.01)
        nm = LAND_CODE.get(int(lv), str(lv))
        flag = " *" if r < -0.05 and p < 0.05 else ""
        print(f"  {nm:<4} {m.sum():>6} {r:>+12.4f}"
              f" {bd_mean:>6.3f} {ratio:>7.3f}{flag}")
    # Estimate: if X% of BD were misattributed BSW
    for pct in [5, 10, 20]:
        gain = bd_total * pct / 100
        print(f"  If {pct}% of BD→BSW: +{gain:,.0f}"
              f" votes ({'enough' if gain >= BSW_DEFICIT else 'not enough'})")
    return bd_total


def mechanism3_excess_zeros(df, pred):
    """Missing votes from excess BSW=0 precincts
    beyond Poisson expectation."""
    print(f"\n{SEP}")
    print("MECHANISM 3: Excess Zero Impact")
    print(SEP)
    u = _urne(df)
    g = _valid(df)
    bsw_v = _votes(df, "BSW").values
    bsw_pred = pred["BSW_pred"].values / 100
    lam = np.maximum(bsw_pred * g, 0)
    # Expected zeros from Poisson
    p0 = np.exp(-lam)
    exp_zeros = p0[u].sum()
    obs_zeros = ((bsw_v == 0) & u).sum()
    excess = obs_zeros - exp_zeros
    print(f"  Observed BSW=0 Urne: {obs_zeros}")
    print(f"  Poisson expected: {exp_zeros:.0f}")
    print(f"  Excess zeros: {excess:.0f}")
    # Suspicious zeros: Poisson P < 1%
    is_zero = (bsw_v == 0) & u
    susp = is_zero & (p0 < 0.01)
    expected_v = lam[susp]
    total_missing = expected_v.sum()
    print(f"  Suspicious zeros (P<1%): {susp.sum()}")
    print(f"  Expected votes in susp: "
          f"{total_missing:,.0f}")
    # Top excess most-unlikely zeros
    idx = np.where(is_zero)[0]
    probs = p0[idx]
    order = np.argsort(probs)
    top_n = int(min(excess, len(idx)))
    top_idx = idx[order[:top_n]]
    missing_excess = lam[top_idx].sum()
    print(f"  Missing from {top_n} excess: "
          f"{missing_excess:,.0f}")
    return missing_excess


def mechanism4_zip_model(df, pred):
    """Zero-Inflated Poisson via EM algorithm.
    Estimate structural zero probability π."""
    print(f"\n{SEP}")
    print("MECHANISM 4: Zero-Inflated Poisson Model")
    print(SEP)
    u = _urne(df); g = _valid(df)
    results = {}
    for party in ["BSW"] + CONTROLS:
        v = _votes(df, party).values[u].astype(float)
        lam_hat, pi_hat = max(v.mean(), 0.01), 0.1
        for _ in range(100):
            p0 = np.exp(-lam_hat)
            w = np.where(v==0,
                pi_hat/(pi_hat+(1-pi_hat)*p0), 0)
            pi_new = w.mean()
            nz = 1 - w
            lam_new = (nz*v).sum()/max(nz.sum(),1)
            if abs(pi_new-pi_hat)+abs(lam_new-lam_hat)<1e-8:
                break
            pi_hat, lam_hat = pi_new, lam_new
        results[party] = (pi_hat, lam_hat)
        print(f"  {party:<12} π={pi_hat:.4f}"
              f" λ={lam_hat:.2f}")
    # Excess structural zeros for BSW
    pi_bsw, lam_bsw = results["BSW"]
    pi_ctrl = np.mean([results[c][0] for c in CONTROLS])
    excess_pi = max(pi_bsw - pi_ctrl, 0)
    n_urne = u.sum()
    excess_struct = excess_pi * n_urne
    missing = excess_struct * lam_bsw
    print(f"  BSW excess π: {excess_pi:.4f}")
    print(f"  Excess struct zeros: {excess_struct:.0f}")
    print(f"  Est missing votes: {missing:,.0f}")
    return missing


def mechanism5_small_precinct(df, pred):
    """Small-precinct systematic BSW residual bias."""
    print(f"\n{SEP}")
    print("MECHANISM 5: Small-Precinct Bias")
    print(SEP)
    g = _valid(df); bsw_r = pred["BSW_resid"].values
    qs = pd.qcut(g, 5, labels=False, duplicates="drop")
    print(f"  {'Q':>2} {'Size':>10} {'n':>6}"
          f" {'BSW':>8} {'FDP':>8} {'Linke':>8}")
    total_bias = 0
    for q in sorted(np.unique(qs)):
        m = qs == q
        sz = f"{g[m].min():.0f}-{g[m].max():.0f}"
        br = bsw_r[m].mean()
        fr = pred["FDP_resid"].values[m].mean()
        lr = pred["Die Linke_resid"].values[m].mean()
        bias_v = br / 100 * g[m].sum()
        total_bias += bias_v
        print(f"  {q:>2} {sz:>10} {m.sum():>6}"
              f" {br:>+8.4f} {fr:>+8.4f}"
              f" {lr:>+8.4f}")
    print(f"  Total BSW bias: {total_bias:+,.0f} votes")
    # Only count negative-residual quintiles
    m0 = qs == 0
    small_bias = bsw_r[m0].mean() / 100 * g[m0].sum()
    print(f"  Smallest quintile bias: "
          f"{small_bias:+,.0f} votes")
    # Return undercount from small precincts only
    under = abs(small_bias) if small_bias < 0 else 0
    return under


def _brief_gaps(df):
    """Per-WKR Urne-Brief share gaps for BSW+controls."""
    gz=_valid(df)
    wkr=pd.to_numeric(df["Wahlkreis"],errors="coerce")
    ba=df["Bezirksart"].values
    u=ba==0;b=ba==5
    ps=["BSW"]+CONTROLS
    gaps={p:[] for p in ps};ids=[];bv=[]
    for w in sorted(wkr.dropna().unique()):
        m=wkr==w;mu=m&u;mb=m&b
        if mu.sum()<5 or mb.sum()<5: continue
        ids.append(w);bv.append(gz[mb].sum())
        for p in ps:
            s=_share(df,p)
            gaps[p].append(
                np.average(s[mu],weights=gz[mu])
                -np.average(s[mb],weights=gz[mb]))
    return {p:np.array(v) for p,v in gaps.items()},np.array(bv),ids


def mechanism6_brief_urne(df, pred):
    """Briefwahl vs Urne gap anomaly."""
    print(f"\n{SEP}")
    print("MECHANISM 6: Briefwahl Deep Dive")
    print(SEP)
    gaps, bv, ids = _brief_gaps(df)
    bg = gaps["BSW"]
    print(f"  {len(ids)} WKR with Brief+Urne data")
    print(f"  BSW Urne-Brief gap: "
          f"mean={bg.mean():+.3f}pp")
    for p in CONTROLS:
        print(f"  {p} gap: "
              f"mean={gaps[p].mean():+.3f}pp")
    # Anomaly: BSW gap minus avg control gap
    ctrl_avg = np.mean([gaps[c] for c in CONTROLS], axis=0)
    anomaly = bg - ctrl_avg
    print(f"  BSW anomaly (gap-controls): "
          f"mean={anomaly.mean():+.3f}pp")
    # Missing votes: where BSW gap is anomalously positive
    # (Urne > Brief more than controls)
    pos = anomaly > 0
    missing = (anomaly[pos]/100 * bv[pos]).sum()
    print(f"  WKR with positive anomaly: "
          f"{pos.sum()}/{len(ids)}")
    print(f"  Est missing Brief votes: "
          f"{missing:,.0f}")
    return missing


def summary(cf_result, results):
    """Separate counterfactual from mechanisms."""
    print(f"\n{'#' * 60}")
    print("COUNTERFACTUAL (not counting errors)")
    print('#' * 60)
    print(f"  Ballot Visibility: {cf_result:>12,.0f}")
    print("  (Campaign effect, not fraud evidence)")
    print(f"\n{'#' * 60}")
    print("COUNTING-ERROR MECHANISMS")
    print('#' * 60)
    names = ["Recount Extrapolation",
             "BD Adjacency (10%)", "Excess Zeros",
             "ZIP Model", "Small-Precinct Bias",
             "Briefwahl Gap"]
    rows = []
    print(f"\n  {'Analysis':<25} {'Est. Votes':>12}")
    print(f"  {'-'*25} {'-'*12}")
    for name, val in zip(names, results):
        print(f"  {name:<25} {val:>12,.0f}")
        rows.append({"analysis": name, "votes": val})
    total = sum(results)
    print(f"  {'─'*37}")
    print(f"  {'TOTAL':<25} {total:>12,.0f}")
    print(f"  {'BSW deficit':<25} {BSW_DEFICIT:>12,}")
    print(f"  {'Surplus':<25} "
          f"{total - BSW_DEFICIT:>+12,.0f}")
    # Scenarios (counting-error only, no visibility)
    cons = results[2]+results[4]  # zeros+small
    cent = results[0]*0.25+results[2]+results[3]+results[5]*0.5
    opt = results[0]*0.5+results[1]+results[5]
    print(f"\n  Scenarios (non-additive):")
    for nm,v in [("Conservative",cons),
                 ("Central",cent),("Optimistic",opt)]:
        s = "YES" if v >= BSW_DEFICIT else "NO"
        print(f"    {nm}: {v:>10,.0f} ({s})")
    print(f"    Deficit: {BSW_DEFICIT:>10,}")
    rows.insert(0, {"analysis": "COUNTERFACTUAL: Visibility",
                     "votes": cf_result})
    out = pd.DataFrame(rows)
    out.to_csv(DATA/"bsw_evidence_summary.csv",
               index=False)
    print(f"\n  Saved bsw_evidence_summary.csv")
    return total


def main():
    df, pred = load_all()
    cf = counterfactual_visibility(df)
    r1d = mechanism1_recount_mc(df)
    r1 = r1d["raw_median"]
    r2 = mechanism2_bd_adjacency(df, pred)*0.10
    r3 = mechanism3_excess_zeros(df, pred)
    r4 = mechanism4_zip_model(df, pred)
    r5 = mechanism5_small_precinct(df, pred)
    r6 = mechanism6_brief_urne(df, pred)
    summary(cf, [r1, r2, r3, r4, r5, r6])


if __name__ == "__main__":
    main()
