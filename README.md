# BSW Analysis

Predicting 2025 Bundestagswahl Zweitstimme shares at polling
station (Wahlbezirk) level using linear regression.

## Data

Download into `data/`:
- `btw25_wbz.zip` — 2025 precinct results (~95k precincts)
- `btw21_wbz.zip` — 2021 precinct results
- `btw17_wbz.zip` — 2017 precinct results

## Scripts

- `wahlbezirk_lr.py` — Linear regression per party (10-fold CV)
- `ridge_party_cv.py` — Ridge regression (precinct-level, 2025 only)

## Features (210 total)

- 2025 Erststimme shares per party (28 cols)
- 2025 structural: turnout, invalid rates, log(voters)
- 2021 Erst+Zweit shares per party (aggregated to Wahlkreis)
- 2017 Erst+Zweit shares per party (aggregated to Wahlkreis)
- Bundesland one-hot encoding (15 cols, drop_first)

## Key Results (BSW)

BSW ranks 9th/29 in R² (0.73) and 8th in Spearman (0.78).
Major parties (CDU, AfD, CSU, GRÜNE, Die Linke, SPD) all have
R² > 0.89. BSW is harder to predict since it's a new party with
no 2017/2021 historical Zweitstimme data.

```
python3 wahlbezirk_lr.py
```
