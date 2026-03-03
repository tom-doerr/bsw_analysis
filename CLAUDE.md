# BSW Analysis

## Project
Bundestagswahl election analysis predicting Zweitstimme shares.
Repo: `~/git/bsw_analysis` (github.com/tom-doerr/bsw_analysis)

## Scripts
- `wahlbezirk_lr.py` — LR per party at Wahlbezirk level (95k rows)
- `ridge_party_cv.py` — Ridge regression (precinct, 2025 only)
- `bsw_bd_decorrelate.py` — BSW+BD sum decorrelation analysis
- `bsw_forensic.py` — 11-test forensic battery for missing BSW votes
- `bsw_claims_test.py` — Tests BSW's 4 specific claims about vote miscounting
- `xgb_enhanced.py` — XGBoost + EW24 + Strukturdaten + SHAP
- `bsw_swing.py` — EW24→BTW25 swing analysis (Gemeinde)
- `aggregate_swing_wkr.py` — Aggregate swing to Wahlkreis
- `prep_dashboard.py` — Build JSON for Three.js dashboard
- `generate_report.py` — HTML report with Plotly charts
- `bsw_evidence.py` — 7-analysis case for BSW crossing 5%
- `bsw_bayesian.py` — Bayesian posterior P(Δ≥9,529)
- `bsw_power.py` — Power analysis for forensic battery
- `panel_analysis.py` — 4-election Gemeinde panel
- `evidence_registry.py` — Suspicious precinct registry
- `bsw_recount_bias.py` — Recount selection-bias analysis
- `bsw_adjacency_did.py` — Ballot adjacency DiD
- `bsw_generative.py` — Latent-variable generative model
- `bsw_affidavits.py` — Sworn statement cross-reference

## Data (`data/`)
- `btw{25,21,17,13}_wbz.zip` — Precinct-level election results
- `ew24_wbz.zip` — Europawahl 2024 precinct results (BSW included)
- `btw2025_strukturdaten.csv` — Sociodemographic data per Wahlkreis
- `ew24_strukturdaten.csv` — EW24 Strukturdaten
- `btw2025_brief_wkr.csv` — 2025 Wahlkreis-level aggregated
- `btw2025kreis.csv` — 2025 Kreis-level aggregated

## Data Format Quirks
- 2025 CSVs: skiprows=4, semicolon-sep, UTF-8-BOM
- 2021 CSV: NO skiprows (header at row 0), BOM, E_/Z_ prefixed cols
- 2017 CSVs: skiprows=4, latin-1 encoding, quoted fields, bare names
- 2017 has separate erst/zweitstimmen files
- EW24: no skiprows, semicolon, UTF-8, party cols 17-51, col 45=BSW
- Strukturdaten: skiprows=8, semicolon, commas as decimal separators
- EW24 has no Wahlkreis col; join via Land+Kreis+Gemeinde key
- Party name diffs: DIE LINKE (17/21) vs Die Linke (25)

## Key Findings
- LR baseline: BSW R²=0.73 (rank 9/29), new party = hard to predict
- XGBoost+EW24+SD: BSW R²=0.81 (+0.08), biggest major party gain
- Top BSW features: 2017 Die Linke Erst (42%), EW24 BSW (17%), 2021 AfD (6%)
- Major parties (CDU, AfD, CSU, GRÜNE, Die Linke, SPD) all R²>0.89
- FDP also relatively hard (LR R²=0.65, XGB R²=0.70)
- BSW biggest anomalies in Brandenburg and Sachsen-Anhalt

## Forensic Results (no evidence of fraud)
- No BSW↔BD vote-swapping (decorrelation, r=+0.004)
- 11 forensic tests all PASS: turnout corr, Brief/Urne, 2nd-digit
  Benford, size strat, invalid corr, multimodality, kurtosis/skew,
  geo clustering, zero-vote Poisson, GMM, feature importance
- BSW patterns match controls (FDP, Die Linke) on all tests
- East-West Brief/Urne split matches Die Linke pattern exactly

## BSW Claims Test Results
- Claim 1 (BSW↔BD ballot swap): r=+0.004, no swap signature. Need ~12.5% of ALL BD votes for 5%.
- Claim 2 (zero-vote precincts): 481 zeros (1.41x expected), max impact +2,873 votes (< 9,529 needed)
- Claim 3 (extrapolation from 50 recounts): sample biased (BSW-selected), not representative
- Claim 4 (disproportionate corrections): selection bias makes analysis uninformative

## BSW Evidence Analysis (case for 5%)
- Counterfactual (visibility): 131k, reported separately
- 6 counting-error mechanisms: recount, BD, zeros, ZIP, small, brief
- Conservative: 5.3k, Central: 19.9k, Optimistic: 36.2k
- Bayesian posterior P(≥9,529): ~25% (mixture model)
- Power: forensic battery cannot detect 9,529×1 miscount (0%)

## Evidence Registry
- 5,400 flagged precincts (of 95k) by 4 criteria
- 239 BSW=0 in flagged, 104 suspicious (P<1%)
- 3 known cases matched with all 4 flags
- Output: data/evidence_registry.csv + .json

## Recount Bias Analysis
- Rate θ=0.304 [0.177, 0.481] from Gamma posterior
- 104 suspicious precincts only → ~289 votes
- Need f≥20% representativeness for crossing chance
- f=30%: P=35.6%, f=50%: P=93.9%

## Adjacency DiD
- Anomalies LOWER where BSW has Erst (ratio 0.43)
- BSW=0 5.76x more in Urne vs Brief (size effect)
- FDP↔FW has more zero-with-neighbor than BSW↔BD
- 471/517 BSW=0 in smallest quintile (Poisson noise)
- Residuals more negative in Erst WKRs (t=-21.4)

## Generative Model
- Swap+zeroout channels, no double-counting
- Conservative: med=1,832, P=0%
- Bias-adjusted (10% problem): med=7,175, P=34%

## Affidavit Analysis
- 3 matched: 99.9-100th pct BD within Land
- P(BSW=0) 10^-15 to 10^-5, all in registry
- Affidavit BD 17.3x higher than avg BSW=0

## Panel Analysis (4-election Gemeinde tracking)
- 7,766 Gemeinden matched across 2013/17/21/25
- r(Linke_drop, BSW) = 0.684, BSW = 5.00+0.48×drop
- East surplus, West deficit pattern in BSW vs prediction
- 2013 data: UTF-8-sig encoding, bare party names

## Data Columns (raw 2025)
- Bezirksart: 0=Urne (66.5k), 5=Brief (28.6k), 6/8 rare
- Briefwahl precincts have Wahlberechtigte=0
- 80 columns total: geo IDs, voter counts, Erst/Zweit per party

## Dashboard
- Live: https://tom-doerr.github.io/bsw_analysis/
- Report: https://tom-doerr.github.io/bsw_analysis/report.html
- Three.js r170 (CDN), docs/ served by GitHub Pages
- Metrics: vote share, residual, swing, turnout
- SHAP top features shown in info panel on click

## SHAP
- BSW top SHAP: ew24_BSW (1.18), e25_Die Linke (0.56), e25_AfD (0.30)
- shap_summary.json has top 20 features per party

## Gotchas
- Zweitstimme shares sum to 100% → including other parties' z25 shares leaks
- Write hook limits edits to 800 bytes; build files incrementally
- Use np.hstack not DataFrame.copy() for 95k-row feature matrices
