# BSW Analysis

**[Live Dashboard](https://tom-doerr.github.io/bsw_analysis/)** |
**[Forensic Report](https://tom-doerr.github.io/bsw_analysis/report.html)**

Statistical analysis of the 2025 Bundestagswahl at precinct
level (95k Wahlbezirke). BSW received 4.981% — 9,529 votes
short of the 5% threshold. This repo examines whether the
margin justifies targeted recounts.

## Data

Included in `data/` (public government data):
- `btw25_wbz.zip` — 2025 precinct results (~95k precincts)
- `btw21_wbz.zip` — 2021 precinct results
- `btw17_wbz.zip` — 2017 precinct results
- `btw13_wbz.zip` — 2013 precinct results
- `ew24_wbz.zip` — Europawahl 2024 precinct results (BSW included)
- `btw2025_strukturdaten.csv` — Sociodemographic data per Wahlkreis
- `ew24_strukturdaten.csv` — EW24 Strukturdaten

Datenquelle: © Die Bundeswahlleiterin, Wiesbaden 2025 ([bundeswahlleiterin.de](https://www.bundeswahlleiterin.de)), dl-de/by-2-0

## Scripts

- `wahlbezirk_lr.py` — Ridge regression per party (GroupKFold by WKR)
- `ridge_party_cv.py` — Ridge regression (precinct-level, 2025 only)
- `bsw_bd_decorrelate.py` — BSW+BD sum decorrelation analysis
- `bsw_forensic.py` — 11-test forensic battery for missing votes
- `bsw_claims_test.py` — Tests BSW's 4 specific claims about miscounting
- `xgb_enhanced.py` — XGBoost + Europawahl 2024 + Strukturdaten
- `bsw_evidence.py` — scenario analysis for BSW crossing 5%
- `bsw_bayesian.py` — Bayesian posterior P(Δ≥9,529)
- `bsw_power.py` — Power analysis for forensic battery
- `panel_analysis.py` — Gemeinde-level 4-election panel
- `evidence_registry.py` — Suspicious precinct registry with anomaly scores
- `bsw_recount_bias.py` — Recount selection-bias sensitivity analysis
- `bsw_adjacency_did.py` — Ballot adjacency natural experiment
- `bsw_generative.py` — Latent-variable generative model (no double-counting)
- `bsw_affidavits.py` — Sworn statement cross-reference
- `calibrate_zero_model.py` — Zero-vote model calibration
- `calibrate_zero_betabinom.py` — BB zero calibration
- `triangulate_lr_xgb.py` — LR vs XGB triangulation
- `low_tail_undercount.py` — Low-tail BB undercount
- `bsw_bd_swap.py` — BSW→BD swap model
- `official_corrections.py` — Prelim→final corrections
- `generate_report.py` — HTML report generation

## Features

**Independence-first LR (253 features):**
- 2025 structural: turnout, invalid rates, log(voters)
- 2021/2017 Erst+Zweit shares (Wahlkreis-level)
- EW24 party shares (Gemeinde-level join)
- Strukturdaten demographics (Wahlkreis-level)
- Bundesland dummies (15 cols)
- No 2025 Erststimmen (avoids same-election leak)

**Base LR (210 features):** adds 2025 Erststimmen

## Prediction Results

95,046 precincts, 29 party models, GroupKFold(10) by Wahlkreis,
Ridge(alpha=5000). Independence-first model uses no 2025
Erststimmen.

| Party | LR R² | Notes |
|-------|-------|-------|
| CDU | 0.96 | |
| AfD | 0.96 | |
| Die Linke | 0.86 | |
| SPD | 0.85 | |
| **BSW** | **0.64** | **strict (no e25)** |
| **BSW** | **0.63** | **base (with e25)** |
| FDP | 0.59 | |

BSW R²=0.64 with the strict model confirms predictions do not
depend on same-election features. Leave-one-Land-out: R²=0.38
(strict model generalizes better than base R²=0.04).

### BSW Feature Importances (XGBoost)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | 2017 Die Linke Erststimme | 42.3% |
| 2 | EW24 BSW Zweitstimme | 16.8% |
| 3 | 2021 AfD Zweitstimme | 6.1% |
| 4 | Sachsen (Land dummy) | 3.2% |
| 5 | Foreigner population % | 0.9% |

BSW draws primarily from former Die Linke voters and
AfD-adjacent demographics in eastern Germany.

## Statistical Evidence

### Official Corrections (Arbeitstabelle 9)

BSW gained **+4,277** Zweitstimmen (prelim→final) —
**44.9%** of the 9,529 deficit. BD lost −2,640.

### Low-Tail Undercount

784 precincts show fewer BSW votes than predicted.
**Null-calibrated excess: 5,145 votes (p=0.005).**

### Power Analysis

Forensic battery has **0% power** for diffuse errors.
"No evidence" ≠ "no errors exist."

### BSW↔BD Decorrelation

Tested whether votes were swapped between them:

- Raw residual correlation: +0.004 (no anti-correlation)
- Would need ~12.5% of ALL BD votes for BSW to reach 5%
- BSW+BD pair does not stand out vs control pairs

### Forensic Battery (11 tests)

All tests pass — but have 0% power for diffuse errors.
BSW matches control parties (FDP, Die Linke).

**1. Turnout–BSW correlation** — Weak positive overall
(r=+0.22), similar to AfD. Per-Land breakdown shows negative
r in West (NI -0.26, HE -0.25) and positive in East — a
normal demographic pattern matching other parties.

**2. Briefwahl vs Urne** — Clear East-West split: BSW is
~0.5-1.5pp higher in Urne (West) but 2-3.6pp higher in Brief
(East). Die Linke shows the exact same pattern. BSW residual
means are near-zero in both channels (Urne -0.006, Brief
+0.013pp). No channel-specific manipulation.

**3. Second-digit Benford's Law** — BSW χ²=24.1 (p=0.004),
but FDP is worse (χ²=50.0, p≈0) and AfD far worse (χ²=385).
Die Linke passes cleanly (p=0.23). Not a BSW-specific anomaly.

**4. Precinct-size stratification** — BSW residuals have a
weak positive trend with precinct size (Spearman +0.066):
larger precincts have slightly *more* BSW than predicted.
Opposite of the fraud hypothesis (larger = easier to
manipulate). FDP and Die Linke show negligible patterns.

**5. Invalid vote correlation** — BSW residual–invalid rate
correlation is 0.0000 overall. Per-Land values are small and
mixed-sign. No evidence of BSW ballots being invalidated.

**6. Multimodality** — KDE of BSW residuals shows a single
peak at -0.21. FDP and Die Linke also unimodal. No hidden
"depleted" subpopulation.

**7. Kurtosis & skewness** — BSW skew = +0.53 (rank 6/29,
slightly positive right tail). Missing votes would produce
*negative* skew (left tail). Kurtosis = 3.8 (rank 27/29,
among the lowest). Completely normal distribution shape.

**8. Geographic clustering** — Only 5/299 Wahlkreise have
BSW mean residual z < -2. In those 5, FDP residuals are
positive (+0.08) and Die Linke slightly negative (-0.08).
Pattern reflects model limitations, not coordinated fraud.

**9. Zero-vote deep dive** — 60 suspicious BSW zeros (model
predicts >3%, >100 valid votes, Poisson P<0.01). Mostly in
BY (22) and NI (10). But FDP also has 10 suspicious zeros
despite being well-established. 60 out of 95k precincts is
within normal variance.

**10. Gaussian Mixture Model** — 2-component GMM fits better
for all three parties equally (BSW ΔBIC=9258, Die Linke
ΔBIC=12757). BSW components: 67% with μ=-0.2pp, 33% with
μ=+0.4pp. Die Linke nearly identical. Reflects demographic
heterogeneity, not a "depleted" fraud subpopulation.

**11. Feature importance** — BSW residuals correlate with
essentially nothing (all |r| < 0.02). The model already
captures all systematic variation; remaining errors are noise.

### XGBoost Triangulation

71% Jaccard overlap between LR and XGB suspicious sets.
Spearman ρ=0.898. Top-20 overlap: 80%, Top-50: 92%.
Both models flag the same precincts.

### BSW's Specific Claims

BSW got 4.981%, missing 5% by 9,529 votes (0.019pp).

**Claim 1: Ballot confusion** — r=+0.004, no systematic
swap detected. Would need ~12.5% of BD votes.

**Claim 2: Zero-vote precincts** — 784 low-tail precincts.
Null-calibrated excess: 5,145 votes (p=0.005).

**Claim 3: Recount extrapolation** — 50 BSW-selected
recounts. Selection bias limits extrapolation.

**Claim 4: Official corrections** — BSW +4,277 (44.9%
of deficit) through normal verification.

## Summary

The 9,529-vote deficit is small enough that targeted
recounts are justified:
- Official corrections already recovered 44.9%
- 5,145 excess missing votes (p=0.005)
- Forensic tests lack power for diffuse errors
- 3 affidavit-backed cases confirmed in registry
- Independence-first model (no e25) confirms R²=0.64

## Evidence Registry

3,578 flagged precincts by 4 criteria (BB P(0), BD rank).
Uses Beta-Binomial p0 via bb_utils.
BB-calibrated excess: HE +19.4, NI +18.0, BY +3.9.
All 3 affidavit cases matched.

## Recount Bias: Sensitivity Curve

Rate θ=0.304 [0.18, 0.48]. If recounts represent only the
81 BB-suspicious precincts: ~25 votes (P=0%). Need f≥20%
representativeness for any chance. f=30%: P=35.5%.

## Ballot Adjacency Natural Experiment

Anomalies **lower** where BSW has Erst (ratio 0.43).
Logistic regression with controls: has_erst OR=1.12,
**p=0.50** (not significant). FDP placebo: OR=1.54, p=0.04.
BSW=0 concentrates in small precincts (471/517 in Q0).

## Generative Model (speculative)

Swap + zero-out channels. Conservative: med=1,832, P=0%.
Bias-adjusted Beta(1,9): med=7,175, P=34%.
Results are highly sensitive to π prior — treat as
scenario exploration, not proof.

## Usage

```
python3 wahlbezirk_lr.py        # LR prediction models
python3 xgb_enhanced.py         # XGBoost + EW24 + Strukturdaten
python3 bsw_bd_decorrelate.py   # decorrelation analysis
python3 bsw_forensic.py         # forensic battery
python3 bsw_claims_test.py      # BSW's specific claims
python3 bsw_evidence.py         # scenario analysis
python3 bsw_bayesian.py         # Bayesian posterior
python3 bsw_power.py            # power analysis
python3 panel_analysis.py       # Gemeinde panel analysis
python3 evidence_registry.py   # build precinct registry
python3 bsw_recount_bias.py    # recount bias analysis
python3 bsw_adjacency_did.py   # adjacency DiD
python3 bsw_generative.py      # generative model
python3 bsw_affidavits.py      # affidavit cross-reference
python3 calibrate_zero_model.py # zero-vote calibration
```
