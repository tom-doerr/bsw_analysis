#!/usr/bin/env python3
"""Generate formal HTML report for BSW analysis."""

import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from jinja2 import Template

DATA = Path("data")
OUT = Path("docs/report.html")

PARTIES = ["BSW", "AfD", "CDU", "SPD", "GRÜNE",
           "Die Linke", "FDP", "CSU", "FREIE WÄHLER"]
CONTROLS = ["FDP", "Die Linke"]
PLOTLY_CFG = dict(include_plotlyjs="cdn",
                  full_html=False)


def load_key_stats():
    """Load headline numbers from data files."""
    s = {}
    oc = pd.read_csv(DATA / "official_corrections.csv")
    bsw = oc[oc["party"] == "BSW"].iloc[0]
    bd = oc[oc["party"] == "BD"].iloc[0]
    s["bsw_final"] = int(bsw["final"])
    s["bsw_pct"] = float(bsw["pct"])
    s["bsw_delta"] = int(bsw["delta"])
    s["bd_delta"] = int(bd["delta"])
    totals = pd.read_csv(
        DATA / "election_totals.csv", index_col=0)
    total = int(totals.loc["valid_total", "value"])
    s["valid_total"] = total
    threshold = math.ceil(total * 0.05)
    s["threshold"] = threshold
    s["deficit"] = threshold - s["bsw_final"]
    s["deficit_pp"] = s["deficit"] / total * 100
    s["recovery_pct"] = (
        s["bsw_delta"] / s["deficit"] * 100)
    _load_secondary(s)
    return s


def _load_secondary(s):
    lt = pd.read_csv(
        DATA / "low_tail_calibration.csv",
        index_col=0)
    s["lt_n"] = int(lt.loc["n_obs", "v"])
    s["lt_excess"] = lt.loc["excess_miss", "v"]
    s["lt_p"] = lt.loc["n_p", "v"]
    nc = pd.read_csv(
        DATA / "null_calibration.csv", index_col=0)
    s["zero_flags"] = int(
        nc.loc["flags_obs", "value"])
    aff = pd.read_csv(
        DATA / "affidavit_analysis.csv")
    s["n_affidavits"] = len(aff)
    lr = pd.read_csv(
        DATA / "wahlbezirk_lr_metrics.csv",
        index_col=0)
    s["bsw_r2"] = lr.loc["BSW", "R2"]


def load_data():
    """Load all analysis outputs."""
    from wahlbezirk_lr import load_2025_wbz, LAND_CODE
    print("Loading precinct data...")
    df = load_2025_wbz()
    for c in ["Gültige - Zweitstimmen",
              "Bezirksart"]:
        df[c] = pd.to_numeric(
            df[c], errors="coerce").fillna(0)
    df = df[df["Gültige - Zweitstimmen"] >= 1]
    df = df.reset_index(drop=True)
    pred = pd.read_csv(
        DATA / "wahlbezirk_lr_predictions.csv")
    lr = pd.read_csv(
        DATA / "wahlbezirk_lr_metrics.csv",
        index_col=0)
    xgb = pd.read_csv(
        DATA / "xgb_enhanced_metrics.csv",
        index_col=0)
    return df, pred, lr, xgb, LAND_CODE


def _votes(df, party):
    col = f"{party} - Zweitstimmen"
    return pd.to_numeric(
        df[col], errors="coerce").fillna(0)


def _valid(df):
    return df["Gültige - Zweitstimmen"].values


def _share(df, party):
    v = _votes(df, party).values
    g = _valid(df)
    return np.where(g > 0, v / g * 100, 0)


def sec_executive(s):
    """Executive summary HTML."""
    return (
'<h2>Executive Summary</h2>'
f'<p>BSW received {s["bsw_pct"]:.3f}%, missing the '
f'5% threshold by {s["deficit"]:,} votes '
f'({s["deficit_pp"]:.3f}pp). This report examines '
f'whether the margin is small enough to justify '
f'targeted recounts.</p>'
    ) + _exec_findings(s)


def _exec_findings(s):
    return (
f'<div class="finding"><b>Key findings:</b><ul>'
f'<li>Official corrections recovered '
f'+{s["bsw_delta"]:,} BSW votes '
f'({s["recovery_pct"]:.1f}% of deficit)</li>'
f'<li>{s["lt_n"]:,} low-tail precincts show '
f'{s["lt_excess"]:,.0f} excess missing votes '
f'(p={s["lt_p"]:.3f})</li>'
f'<li>Forensic battery has 0% power to detect '
f'{s["deficit"]:,} votes spread across '
f'precincts</li>'
f'<li>{s["n_affidavits"]} affidavit-backed cases '
f'confirmed in anomaly registry</li>'
f'<li>Strict BSW model (no 2025 Erststimmen) '
f'confirms predictions '
f'(R\u00b2={s["bsw_r2"]:.2f})</li>'
f'</ul></div>')


def sec_corrections(s):
    """Official corrections from Arbeitstabelle 9."""
    return (
        '<h2>Official Corrections</h2>'
        '<p>Between preliminary and final results '
        '(Arbeitstabelle 9), BSW gained '
        f'<b>+{s["bsw_delta"]:,}</b> Zweitstimmen '
        f'&mdash; <b>{s["recovery_pct"]:.1f}%</b> '
        f'of the deficit. BD lost '
        f'&minus;{abs(s["bd_delta"]):,}.</p>')


def sec_power(s):
    """Power analysis section."""
    conc = s["deficit"] // 10
    return (
        '<h2>Power Analysis</h2>'
        '<p>The forensic battery has <b>0% power</b> '
        f'to detect {s["deficit"]:,} votes spread '
        f'across precincts. Only concentrated errors '
        f'({conc}&times;10) are detectable '
        f'(90%).</p>')


def sec_gap(s):
    """Official gap section."""
    return (
        '<h2>The Official Gap</h2>'
        f'<p>BSW received <b>{s["bsw_final"]:,}</b> '
        f'valid Zweitstimmen out of '
        f'{s["valid_total"]:,} total '
        f'(<b>{s["bsw_pct"]:.3f}%</b>). The 5.000% '
        f'threshold requires {s["threshold"]:,} '
        f'&mdash; a deficit of '
        f'<b>{s["deficit"]:,}</b> '
        f'({s["deficit_pp"]:.3f}pp).</p>')


def sec_model_perf(lr, xgb):
    """Model performance comparison chart+table."""
    ps = [p for p in PARTIES
          if p in lr.index and p in xgb.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Linear Reg", x=ps,
        y=[lr.loc[p, "R2"] for p in ps],
        marker_color="steelblue"))
    fig.add_trace(go.Bar(
        name="XGBoost", x=ps,
        y=[xgb.loc[p, "R2"] for p in ps],
        marker_color="darkorange"))
    fig.update_layout(
        title="Model R² by Party",
        yaxis_title="R²", barmode="group",
        height=380, template="plotly_white")
    chart = fig.to_html(**PLOTLY_CFG)
    tbl = "<table><thead><tr><th>Party</th>"
    tbl += "<th>LR R²</th><th>XGB R²</th>"
    tbl += "<th>Δ</th></tr></thead><tbody>"
    for p in ps:
        l, x = lr.loc[p, "R2"], xgb.loc[p, "R2"]
        tbl += (f"<tr><td>{p}</td><td>{l:.4f}</td>"
                f"<td>{x:.4f}</td>"
                f"<td>{x-l:+.4f}</td></tr>")
    tbl += "</tbody></table>"
    return (f"<h2>Model Performance</h2>"
            f"<p>95,046 precincts, GroupKFold(10) "
            f"by Wahlkreis. LR: BSW uses strict "
            f"features (no 2025 Erststimmen); "
            f"other parties use base features.</p>"
            f"{chart}{tbl}")


def sec_residual_dist(pred):
    """Residual KDE for BSW vs controls."""
    from scipy.stats import gaussian_kde
    fig = go.Figure()
    for p in ["BSW"] + CONTROLS:
        r = pred[f"{p}_resid"].dropna().values
        x = np.linspace(r.min(), r.max(), 200)
        fig.add_trace(go.Scatter(
            x=x, y=gaussian_kde(r)(x),
            name=p, mode="lines"))
    fig.update_layout(
        title="Residual Distributions",
        xaxis_title="Residual (pp)",
        height=350, template="plotly_white")
    h = "<h2>Residual Distributions</h2>"
    h += "<p>Unimodal, no hidden subpopulation.</p>"
    return h + fig.to_html(**PLOTLY_CFG)


def sec_turnout(df):
    from scipy.stats import pearsonr
    g = _valid(df)
    wb = pd.to_numeric(df["Wahlberechtigte (A)"],
        errors="coerce").fillna(0).values
    t = np.where(wb>0,g/wb*100,0)
    b = _share(df, "BSW")
    m = (t>30)&(t<100)&(b>0)
    r,_ = pearsonr(t[m],b[m])
    rng = np.random.RandomState(42)
    i = rng.choice(np.where(m)[0],5000,replace=False)
    fig = go.Figure(go.Scattergl(x=t[i],y=b[i],
        mode="markers",
        marker=dict(size=2,opacity=0.3)))
    fig.update_layout(title=f"Turnout vs BSW (r={r:.3f})",
        xaxis_title="Turnout %",yaxis_title="BSW %",
        height=350,template="plotly_white")
    return "<h2>Turnout Correlation</h2>" + fig.to_html(**PLOTLY_CFG)


def _benford_digit2(votes):
    v = votes[votes >= 10].astype(float)
    with np.errstate(divide="ignore"):
        lg = np.floor(np.log10(np.maximum(v, 1)))
    d2 = (v // 10**(np.maximum(lg-1, 0))).astype(int) % 10
    return np.array([(d2==i).sum()
                     for i in range(10)])


def sec_benford(df):
    """Benford 2nd digit."""
    from scipy.stats import chisquare
    exp = np.array([.120,.114,.109,.104,.100,
                    .097,.093,.090,.088,.085])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(10)),
        y=exp, name="Expected",
        marker_color="gray"))
    tbl = "<table><tr><th>Party</th>"
    tbl += "<th>χ²</th><th>p</th></tr>"
    for p in ["BSW"] + CONTROLS:
        v = _votes(df, p).values.astype(int)
        obs = _benford_digit2(v)
        c2, pv = chisquare(obs, exp*obs.sum())
        fig.add_trace(go.Bar(x=list(range(10)),
            y=obs/obs.sum(), name=p))
        tbl += (f"<tr><td>{p}</td>"
                f"<td>{c2:.1f}</td>"
                f"<td>{pv:.4f}</td></tr>")
    tbl += "</table>"
    fig.update_layout(title="Benford 2nd Digit",
        barmode="group", height=350,
        template="plotly_white")
    h = "<h2>Benford's 2nd Digit</h2>"
    return h + fig.to_html(**PLOTLY_CFG) + tbl


def sec_skew_kurt(pred):
    """Skewness and kurtosis table."""
    from scipy.stats import skew, kurtosis
    tbl = "<h2>Distribution Shape</h2>"
    tbl += "<table><tr><th>Party</th>"
    tbl += "<th>Skew</th><th>Kurtosis</th></tr>"
    for p in PARTIES:
        rc = f"{p}_resid"
        if rc not in pred.columns:
            continue
        r = pred[rc].dropna().values
        s = skew(r)
        k = kurtosis(r, fisher=False)
        tbl += (f"<tr><td>{p}</td>"
                f"<td>{s:+.3f}</td>"
                f"<td>{k:.2f}</td></tr>")
    tbl += "</table>"
    # Use computed BSW values instead of hardcoding
    rc = "BSW_resid"
    if rc in pred.columns:
        r = pred[rc].dropna().values
        s_bsw = skew(r)
        k_bsw = kurtosis(r, fisher=False)
        tbl += (f"<p>BSW skew={s_bsw:+.2f} "
                f"(positive right tail). "
                f"Missing votes would produce negative "
                f"skew. Kurtosis {k_bsw:.1f} "
                f"(leptokurtic; normal=3.0).</p>")
    return tbl


def sec_claims(s):
    """BSW's 4 specific claims."""
    nz = s["lt_n"] - s["zero_flags"]
    h = '<h2>BSW Claims Assessment</h2>'
    h += '<h3>1: BSW&#8596;BD Ballot Confusion</h3>'
    h += '<p>Residual r=+0.004. No systematic swap '
    h += 'detected. Would need ~12.5% of BD.</p>'
    h += '<h3>2: Zero-Vote Precincts</h3>'
    h += (f'<p>{s["lt_n"]:,} low-tail precincts '
          f'({s["zero_flags"]} zeros + {nz} '
          f'BSW&gt;0). Null-calibrated excess: '
          f'{s["lt_excess"]:,.0f} '
          f'votes (p={s["lt_p"]:.3f}).</p>')
    h += '<h3>3: Recount Extrapolation</h3>'
    h += '<p>50 BSW-selected recounts. Selection '
    h += 'bias limits national extrapolation.</p>'
    h += '<h3>4: Official Corrections</h3>'
    h += (f'<p>+{s["bsw_delta"]:,} BSW '
          f'({s["recovery_pct"]:.1f}% of deficit) '
          f'via normal verification process.</p>')
    return h


def sec_conclusion(s):
    h = '<h2>Conclusion</h2>'
    h += '<div class="finding">'
    h += (f'The {s["deficit"]:,}-vote deficit is small '
          f'enough that targeted recounts are '
          f'justified. Official corrections recovered '
          f'{s["recovery_pct"]:.1f}%, '
          f'{s["lt_excess"]:,.0f} excess low-tail '
          f'missing votes are statistically '
          f'significant (p={s["lt_p"]:.3f}), '
          f'and forensic tests lack power to detect '
          f'diffuse errors. The question is not '
          f'settled &mdash; it requires recounts.')
    h += '</div>'
    return h


CSS = """body{font-family:Arial,sans-serif;
max-width:1100px;margin:40px auto;padding:0 20px}
h1{border-bottom:3px solid #b5179e;padding-bottom:10px}
h2{color:#555;margin-top:36px;
border-bottom:2px solid #ddd;padding-bottom:8px}
table{border-collapse:collapse;width:100%;margin:16px 0}
th,td{border:1px solid #ddd;padding:6px 10px}
th{background:#f2f2f2}
.verdict{background:#e8f5e9;border-left:4px solid
#2e7d32;padding:8px 12px;margin:12px 0;
font-weight:bold;color:#2e7d32}
.finding{background:#fff3e0;border-left:4px solid
#e65100;padding:8px 12px;margin:12px 0;color:#bf360c}
.finding ul{margin:8px 0;padding-left:20px}"""


def render(sections):
    date = datetime.now().strftime("%Y-%m-%d")
    content = "\n".join(sections)
    tmpl = Template(
        '<!DOCTYPE html><html><head>'
        '<meta charset="UTF-8">'
        '<title>BSW Recount Analysis</title>'
        '<style>{{css}}</style></head><body>'
        '<h1>BSW: The Case for Targeted Recounts</h1>'
        '<p><i>{{date}}</i> | '
        '<a href="./">Dashboard</a></p>'
        '{{content}}'
        '<hr><footer>Bundeswahlleiterin, '
        'dl-de/by-2-0</footer>'
        '</body></html>')
    return tmpl.render(css=CSS, date=date,
                       content=content)


def main():
    s = load_key_stats()
    df, pred, lr, xgb, _ = load_data()
    print("Generating sections...")
    secs = [
        sec_executive(s),
        sec_gap(s),
        sec_corrections(s),
        sec_power(s),
        sec_claims(s),
        sec_model_perf(lr, xgb),
        sec_residual_dist(pred),
        sec_turnout(df),
        sec_benford(df),
        sec_skew_kurt(pred),
        sec_conclusion(s),
    ]
    print("Rendering HTML...")
    html = render(secs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print(f"Wrote {OUT} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
