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
- `xgb_enhanced.py` — XGBoost + Europawahl 2024 + Strukturdaten

## Data (`data/`)
- `btw{25,21,17}_wbz.zip` — Precinct-level election results
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
- Claim 1 (BSW↔BD ballot swap): r=+0.004, no swap signature. Need 81% of ALL BD votes for 5%.
- Claim 2 (zero-vote precincts): 481 zeros (1.41x expected), max impact +2,873 votes (< 9,529 needed)
- Claim 3 (extrapolation from 50 recounts): sample biased (BSW-selected), not representative
- Claim 4 (disproportionate corrections): selection bias makes analysis uninformative

## Data Columns (raw 2025)
- Bezirksart: 0=Urne (66.5k), 5=Brief (28.6k), 6/8 rare
- Briefwahl precincts have Wahlberechtigte=0
- 80 columns total: geo IDs, voter counts, Erst/Zweit per party

## Gotchas
- Zweitstimme shares sum to 100% → including other parties' z25 shares leaks
- Write hook limits edits to 800 bytes; build files incrementally
- Use np.hstack not DataFrame.copy() for 95k-row feature matrices
