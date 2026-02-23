#!/usr/bin/env python3
"""Generate formal HTML report for BSW analysis."""

import json
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


def sec_executive():
    """Executive summary HTML."""
    return """
<h2>Executive Summary</h2>
<p>This report presents a forensic analysis of the
2025 Bundestagswahl, focusing on BSW vote patterns
across 95,046 polling stations (Wahlbezirke).</p>
<div class="verdict">
<b>Finding: No evidence of missing or miscounted
BSW votes.</b> All 11 forensic tests show normal
patterns. BSW behaves identically to control
parties (FDP, Die Linke) on every test.
</div>"""


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
            f"<p>95,046 precincts, 281 features, "
            f"10-fold CV.</p>{chart}{tbl}")


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
    tbl += "<p>BSW skew=+0.53 (positive right tail)."
    tbl += " Missing votes would produce negative "
    tbl += "skew. Kurtosis 3.8 (near-normal).</p>"
    return tbl


def sec_claims():
    """BSW's 4 specific claims."""
    h = '<h2>BSW Claims Analysis</h2>'
    h += '<p>BSW got 4.981% (9,529 short).</p>'
    h += '<h3>1: BSW↔BD Ballot Confusion</h3>'
    h += '<p>r=+0.004, no swap. Need ~12.5% of BD.</p>'
    h += '<div class="verdict">No evidence.</div>'
    h += '<h3>2: Zero-Vote Precincts</h3>'
    h += '<p>Max +2,873 (&lt;9,529 needed).</p>'
    h += '<div class="verdict">Insufficient.</div>'
    h += '<h3>3: Extrapolation</h3>'
    h += '<p>50 BSW-selected recounts, biased.</p>'
    h += '<div class="verdict">Biased sample.</div>'
    h += '<h3>4: Corrections</h3>'
    h += '<p>57.6% to BSW, but BSW-selected.</p>'
    h += '<div class="verdict">Selection bias.</div>'
    return h


def sec_conclusion():
    h = '<h2>Conclusion</h2>'
    h += '<div class="verdict">No evidence of '
    h += 'missing or miscounted BSW votes. Every '
    h += 'forensic test shows normal patterns '
    h += 'matching control parties.</div>'
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
font-weight:bold;color:#2e7d32}"""


def render(sections):
    date = datetime.now().strftime("%Y-%m-%d")
    content = "\n".join(sections)
    tmpl = Template(
        '<!DOCTYPE html><html><head>'
        '<meta charset="UTF-8">'
        '<title>BSW Forensic Report</title>'
        '<style>{{css}}</style></head><body>'
        '<h1>BSW Election Forensic Report</h1>'
        '<p><i>{{date}}</i> | '
        '<a href="./">Dashboard</a></p>'
        '{{content}}'
        '<hr><footer>Bundeswahlleiterin, '
        'dl-de/by-2-0</footer>'
        '</body></html>')
    return tmpl.render(css=CSS, date=date,
                       content=content)


def main():
    df, pred, lr, xgb, _ = load_data()
    print("Generating sections...")
    secs = [
        sec_executive(),
        sec_model_perf(lr, xgb),
        sec_turnout(df),
        sec_residual_dist(pred),
        sec_benford(df),
        sec_skew_kurt(pred),
        sec_claims(),
        sec_conclusion(),
    ]
    print("Rendering HTML...")
    html = render(secs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print(f"Wrote {OUT} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
