# BSW Analysis

**[Live Dashboard](https://tom-doerr.github.io/bsw_analysis/)** |
**[Forensic Report](https://tom-doerr.github.io/bsw_analysis/report.html)**

Predicting 2025 Bundestagswahl Zweitstimme shares at polling
station (Wahlbezirk) level using linear regression and XGBoost,
plus comprehensive forensic analysis searching for evidence of
missing or miscounted BSW votes.

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

- `wahlbezirk_lr.py` — Linear regression per party (10-fold CV)
- `ridge_party_cv.py` — Ridge regression (precinct-level, 2025 only)
- `bsw_bd_decorrelate.py` — BSW+BD sum decorrelation analysis
- `bsw_forensic.py` — 11-test forensic battery for missing votes
- `bsw_claims_test.py` — Tests BSW's 4 specific claims about miscounting
- `xgb_enhanced.py` — XGBoost + Europawahl 2024 + Strukturdaten
- `bsw_evidence.py` — 7-analysis case for BSW crossing 5%
- `bsw_bayesian.py` — Bayesian posterior P(Δ≥9,529)
- `bsw_power.py` — Power analysis for forensic battery
- `panel_analysis.py` — Gemeinde-level 4-election panel

## Features

**LR baseline (210 features):**
- 2025 Erststimme shares per party (28 cols)
- 2025 structural: turnout, invalid rates, log(voters)
- 2021 Erst+Zweit shares aggregated to Wahlkreis
- 2017 Erst+Zweit shares aggregated to Wahlkreis
- Bundesland one-hot encoding (15 cols, drop_first)

**XGBoost enhanced (+71 = 281 features):**
- Europawahl 2024 party shares (35 cols, Gemeinde-level join)
- Strukturdaten demographics (36 cols, Wahlkreis-level join)

## Prediction Results

95,046 precincts, 29 party models, 10-fold CV.

| Party | LR R² | XGB R² | Δ | MAE (pp) |
|-------|-------|--------|------|----------|
| CSU | 0.995 | 0.996 | +0.002 | 0.31 |
| AfD | 0.982 | 0.990 | +0.008 | 0.88 |
| CDU | 0.981 | 0.986 | +0.005 | 1.06 |
| GRÜNE | 0.939 | 0.962 | +0.022 | 1.01 |
| Die Linke | 0.915 | 0.945 | +0.031 | 1.01 |
| SPD | 0.893 | 0.928 | +0.035 | 1.26 |
| FREIE WÄHLER | 0.827 | 0.877 | +0.050 | 0.49 |
| **BSW** | **0.731** | **0.809** | **+0.078** | **0.98** |
| FDP | 0.652 | 0.701 | +0.049 | 0.77 |

BSW has the largest R² improvement of any major party (+0.078),
driven by Europawahl 2024 BSW data and XGBoost non-linearities.

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

## Forensic Analysis: BSW Vote Integrity

### Motivation

BSW (Bündnis Sahra Wagenknecht) is a new party that first
contested in the 2025 Bundestagswahl. We used the prediction
model residuals and raw precinct data to search for statistical
evidence of missing, miscounted, or suppressed BSW votes.

### BSW + BÜNDNIS DEUTSCHLAND Decorrelation

BSW and BÜNDNIS DEUTSCHLAND are adjacent on the ballot
(positions 26 and 27 in Zweitstimmen). We tested whether
votes were systematically swapped between them:

- **R²(BSW+BD sum) = 0.735** vs R²(BSW alone) = 0.731 —
  sum barely more predictable (+0.004)
- **Within-sum fraction** BSW/(BSW+BD) is ~96.7% on average
  and nearly unpredictable (R² = 0.038)
- **Orthogonalized residual correlation**: -1.0 (mathematical
  artifact, not evidence — two parts summing to a whole)
- **Raw residual correlation**: +0.004 (no anti-correlation)
- BSW+BD pair does not stand out vs control pairs

**Verdict: No evidence of BSW↔BD vote swapping.**

### 11-Test Forensic Battery

All tests compare BSW against control parties (FDP, Die Linke)
to distinguish BSW-specific anomalies from normal patterns.

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

### XGBoost Residual Confirmation

The improved XGBoost model (R²=0.81, 29% less unexplained
variance) confirms all findings. BSW residuals have positive
skew (+0.51), near-zero mean residuals in every Bundesland,
and identical distributional shape to FDP and Die Linke.
The lower noise floor makes the picture *cleaner*, not more
suspicious.

### Conclusion

**No evidence of missing or miscounted BSW votes.** Every
forensic test shows normal patterns across both LR and XGBoost
models. BSW behaves identically to control parties (FDP,
Die Linke) on all 11 tests.

### BSW's Specific Claims

BSW got 4.981%, missing the 5% threshold by 9,529 votes.
The party made four specific claims about vote miscounting:

**Claim 1: BSW↔BD ballot confusion** — Residual correlation
r=+0.004 (no anti-correlation). Would need ~12.5% of ALL BD
votes transferred to reach 5%. **No evidence.**

**Claim 2: Zero-vote precincts** — 481 BSW=0 Urne precincts
(1.41x expected). Max impact +2,873 votes, far short of
9,529 needed. **Insufficient magnitude.**

**Claim 3: Correction extrapolation** — 50 BSW-selected
recounts found 0.3 extra votes/precinct. Selection bias
invalidates national extrapolation. **Not representative.**

**Claim 4: Disproportionate corrections** — 57.6% of
corrections went to BSW, but precincts were BSW-selected.
4,277 = 0.009% of all votes. **Selection bias.**

## Evidence Analysis: Case for BSW Crossing 5%

Six counting-error mechanisms (visibility reported
separately as counterfactual). Deficit: 9,529.
- **Conservative** (zeros + small): 5,300
- **Central** (recount25% + zeros + ZIP + brief): 19,878
- **Optimistic** (recount50% + BD + brief): 36,218

### Bayesian Posterior

Model B (selection-bias mixture): **P(Δ≥9,529) ≈ 25%**.
Prior-insensitive across 3 priors. See `bsw_bayesian.py`.

### Power Analysis

Forensic battery **cannot detect** spread-thin miscounts
(9,529×1: 0% detection). Only concentrated patterns
(953×10) detected via skewness shift (90%).

## Usage

```
python3 wahlbezirk_lr.py        # LR prediction models
python3 xgb_enhanced.py         # XGBoost + EW24 + Strukturdaten
python3 bsw_bd_decorrelate.py   # decorrelation analysis
python3 bsw_forensic.py         # forensic battery
python3 bsw_claims_test.py      # BSW's specific claims
python3 bsw_evidence.py         # evidence for crossing 5%
python3 bsw_bayesian.py         # Bayesian posterior
python3 bsw_power.py            # power analysis
python3 panel_analysis.py       # Gemeinde panel analysis
```
