# BSW Analysis

## Project
Bundestagswahl election analysis predicting Zweitstimme shares.
Repo: `~/git/bsw_analysis` (github.com/tom-doerr/bsw_analysis)

## Scripts
- `wahlbezirk_lr.py` — LR per party at Wahlbezirk level (95k rows)
- `ridge_party_cv.py` — Ridge regression (precinct, 2025 only)

## Data (`data/`, gitignored)
- `btw{25,21,17}_wbz.zip` — Precinct-level election results
- `btw2025_brief_wkr.csv` — 2025 Wahlkreis-level aggregated
- `btw2025kreis.csv` — 2025 Kreis-level aggregated

## Data Format Quirks
- 2025 CSVs: skiprows=4, semicolon-sep, UTF-8-BOM
- 2021 CSV: NO skiprows (header at row 0), BOM, E_/Z_ prefixed cols
- 2017 CSVs: skiprows=4, latin-1 encoding, quoted fields, bare names
- 2017 has separate erst/zweitstimmen files
- Party name diffs: DIE LINKE (17/21) vs Die Linke (25)

## Key Findings
- BSW R²=0.73 (rank 9/29), hardest major party to predict (new, no history)
- Major parties (CDU, AfD, CSU, GRÜNE, Die Linke, SPD) all R²>0.89
- FDP also relatively hard (R²=0.65)
- BSW biggest anomalies in Brandenburg and Sachsen-Anhalt

## Gotchas
- Zweitstimme shares sum to 100% → including other parties' z25 shares leaks
- Write hook limits edits to 800 bytes; build files incrementally
- Use np.hstack not DataFrame.copy() for 95k-row feature matrices
