#!/usr/bin/env python3
"""Official corrections: preliminary → final result.
Source: Bundeswahlleiterin Arbeitstabelle 9
(btw25_arbtab9.pdf)."""

import pandas as pd
from pathlib import Path

DATA = Path("data")
SEP = "=" * 60
DEFICIT = 9_529

# Verified from btw25_arbtab9.pdf (Zweitstimmen)
# Columns: party, final, delta (final - preliminary)
CORRECTIONS = [
    ("SPD",      8_149_124,    +840),
    ("CDU",     11_196_374,  +1_674),
    ("GRÜNE",    5_762_380,    +904),
    ("FDP",      2_148_757,    -121),
    ("AfD",     10_328_780,  +1_632),
    ("CSU",      2_964_028,    +296),
    ("Die Linke",4_356_532,  +1_150),
    ("BSW",      2_472_947,  +4_277),
    ("BD",          76_372,  -2_640),
    ("SSW",         76_138,     +12),
    ("Volt",       355_262,    +116),
]
VALID_TOTAL = 49_649_512
VALID_DELTA = +7_425


def build_table():
    rows=[]
    for p,f,d in CORRECTIONS:
        rows.append(dict(party=p,final=f,
            delta=d,pct=round(f/VALID_TOTAL*100,3)))
    return pd.DataFrame(rows)


def main():
    tbl=build_table()
    b=tbl[tbl["party"]=="BSW"].iloc[0]
    bd=tbl[tbl["party"]=="BD"].iloc[0]
    print(f"\n{SEP}\nOFFICIAL CORRECTIONS\n{SEP}")
    print(f"  Source: Bundeswahlleiterin Arbeitstabelle 9")
    print(f"  Valid: {VALID_TOTAL:,} (Δ{VALID_DELTA:+,})")
    print(f"\n  BSW: {b.final:,} ({b.pct:.3f}%)")
    print(f"  BSW Δ: {b.delta:+,} ({b.delta/DEFICIT:.1%} of deficit)")
    print(f"  Remaining: {DEFICIT-b.delta:,} votes")
    gpp=(0.05-b.final/VALID_TOTAL)*100
    print(f"  Gap: {gpp:.3f}pp = {gpp/100*VALID_TOTAL:.0f} votes")
    print(f"  Per precinct: {(DEFICIT-b.delta)/95046:.2f}")
    # BD correction (suggestive of swap)
    print(f"\n  BD: {bd.final:,}, Δ{bd.delta:+,}")
    print(f"  BD lost votes while BSW gained")
    # All parties sorted by delta
    print(f"\n  All corrections (Zweitstimmen):")
    print(f"  {'Party':<16}{'Final':>12}{'Δ':>8}")
    for _,r in tbl.sort_values("delta",ascending=False).iterrows():
        print(f"  {r.party:<16}{r.final:>12,}{r.delta:>+8,}")
    tbl.to_csv(DATA/"official_corrections.csv",index=False)
    print(f"\n  Saved → official_corrections.csv")


if __name__=="__main__":
    main()
