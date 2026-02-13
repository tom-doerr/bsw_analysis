# BSW Analysis

Predicting 2025 Bundestagswahl Zweitstimme shares at polling
station (Wahlbezirk) level using linear regression, plus
comprehensive forensic analysis searching for evidence of
missing or miscounted BSW votes.

## Data

Download into `data/`:
- `btw25_wbz.zip` — 2025 precinct results (~95k precincts)
- `btw21_wbz.zip` — 2021 precinct results
- `btw17_wbz.zip` — 2017 precinct results

## Scripts

- `wahlbezirk_lr.py` — Linear regression per party (10-fold CV)
- `ridge_party_cv.py` — Ridge regression (precinct-level, 2025 only)
- `bsw_bd_decorrelate.py` — BSW+BD sum decorrelation analysis
- `bsw_forensic.py` — 11-test forensic battery for missing votes

## Features (210 total)

- 2025 Erststimme shares per party (28 cols)
- 2025 structural: turnout, invalid rates, log(voters)
- 2021 Erst+Zweit shares per party (aggregated to Wahlkreis)
- 2017 Erst+Zweit shares per party (aggregated to Wahlkreis)
- Bundesland one-hot encoding (15 cols, drop_first)

## Prediction Results

95,046 precincts, 210 features, 29 party models.
Pipeline: StandardScaler + LinearRegression, 10-fold CV.

| Party | R² | Spearman | MAE (pp) | Mean share |
|-------|-----|----------|----------|------------|
| CSU | 0.995 | 0.684 | 0.58 | 7.2% |
| AfD | 0.982 | 0.989 | 1.17 | 22.4% |
| CDU | 0.981 | 0.985 | 1.30 | 21.4% |
| GRÜNE | 0.939 | 0.961 | 1.27 | 10.8% |
| Die Linke | 0.915 | 0.919 | 1.23 | 8.5% |
| SPD | 0.893 | 0.948 | 1.53 | 15.8% |
| FREIE WÄHLER | 0.827 | 0.828 | 0.60 | 1.7% |
| **BSW** | **0.731** | **0.781** | **1.17** | **5.0%** |
| FDP | 0.652 | 0.799 | 0.82 | 4.2% |

BSW ranks 9th/29 in R² — harder to predict since it's a new
party with no 2017/2021 Zweitstimme history.

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

### Conclusion

**No evidence of missing or miscounted BSW votes.** Every
forensic test either shows normal patterns or BSW behaves
identically to control parties (FDP, Die Linke). BSW's lower
R² (0.73 vs 0.89+ for major parties) is fully explained by
its novelty: no 2017/2021 Zweitstimme history.

## Usage

```
python3 wahlbezirk_lr.py        # prediction models
python3 bsw_bd_decorrelate.py   # decorrelation analysis
python3 bsw_forensic.py         # forensic battery
```
