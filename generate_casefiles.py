#!/usr/bin/env python3
"""Generate per-precinct case files for recount targets.

Each file: official result, predicted share, tail metrics,
registry/affidavit status, neighborhood, EW24 baseline,
ballot order, and an evidence checklist.

Output: casefiles/<land>_wkr<NNN>_<gemeinde>.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
CASEDIR = Path("casefiles")
SEP = "=" * 60


def load_targets():
    p = DATA / "recount_targets.csv"
    if not p.exists():
        raise FileNotFoundError(
            "Run recount_targets.py first")
    return pd.read_csv(p, low_memory=False)


def _load(name):
    p = DATA / name
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return None


def _find(r, tbl, keys):
    """Find matching row in tbl by key pairs."""
    if tbl is None: return None
    m = pd.Series(True, index=tbl.index)
    for tc, rc in keys:
        m &= tbl[tc] == r[rc]
    if m.any(): return tbl[m].iloc[0]
    return None


def _safe(val, fmt=".2f"):
    if pd.isna(val): return "N/A"
    return f"{val:{fmt}}"


def _location(r, L):
    ba = {0:"Urne",5:"Brief"}.get(
        int(r["bezirksart"]),str(r["bezirksart"]))
    L.append("LOCATION")
    L.append(f"  Land:           {r['land']}")
    L.append(f"  Wahlkreis:      {int(r['wahlkreis'])}")
    L.append(f"  Gemeinde:       {r['gemeinde']}")
    if pd.notna(r.get("wbz")):
        L.append(f"  Wahlbezirk:     {r['wbz']}")
    L.append(f"  Bezirksart:     {ba}")
    L.append("")


def _official(r, L):
    v = max(r["valid_total"], 1)
    L.append("OFFICIAL RESULT")
    L.append(f"  Valid votes:    {int(r['valid_total'])}")
    L.append(f"  BSW votes:      {int(r['bsw_votes'])}")
    L.append(f"  BD votes:       {int(r['bd_votes'])}")
    L.append(f"  BSW share:      "
             f"{r['bsw_votes']/v*100:.2f}%")
    L.append(f"  BD share:       "
             f"{r['bd_votes']/v*100:.2f}%")
    L.append("")


def _stats(r, L):
    L.append("STATISTICAL ASSESSMENT")
    L.append(f"  BSW predicted:  {r['bsw_pred_pct']:.3f}%")
    L.append(f"  Expected votes: {r['lambda']:.1f}")
    L.append(f"  Missing (LB):   {r['miss_lb']:.1f}")
    L.append(f"  P(BSW<=obs):    {r['p_tail']:.6f}")
    if r["bsw_votes"] == 0:
        L.append(f"  P(BSW=0):       {r['p0_bb']:.2e}")
    L.append(f"  BD pctile:      {r['bd_pctile']:.4f}")
    L.append(f"  Overlay score:  {r['overlay_score']:.0f}")
    L.append(f"  Composite:      {r['composite']:.1f}")
    L.append("")


def _registry(r, reg, L):
    rr = _find(r, reg, [
        ("wahlkreis","wahlkreis"),
        ("bsw_votes","bsw_votes"),
        ("valid_total","valid_total")])
    L.append("REGISTRY STATUS")
    if rr is not None:
        L.append(f"  In registry:    YES")
        L.append(f"  Flags:          {rr.get('flags','')}")
        L.append(f"  Recount:        "
                 f"{rr.get('recount_status','unknown')}")
        src = rr.get("source", "")
        if src: L.append(f"  Source:         {src}")
        cl = rr.get("claim", "")
        if cl: L.append(f"  Claim:          {cl}")
    else:
        L.append("  In registry:    NO")
    L.append("")


def _affidavit(r, aff, L):
    ar = _find(r, aff, [
        ("wkr","wahlkreis"),
        ("bsw","bsw_votes"),
        ("bd","bd_votes")])
    L.append("AFFIDAVIT STATUS")
    if ar is not None:
        L.append(f"  Matched:        YES")
        L.append(f"  P(BSW=0):       {ar['p0']:.2e}")
    else:
        L.append("  Matched:        NO")
    L.append("")


def _dossier_ctx(r, dos, L):
    dr = _find(r, dos, [
        ("wkr","wahlkreis"),
        ("bsw","bsw_votes"),
        ("valid","valid_total")])
    L.append("NEIGHBORHOOD CONTEXT")
    if dr is not None:
        L.append(f"  Neighbors:      {int(dr.get('n_nbr',0))}")
        L.append(f"  Nbr BSW>0:      {int(dr.get('nbr_gt0',0))}")
        L.append(f"  Nbr med BSW:    "
                 f"{_safe(dr.get('nbr_med_pct'))}%")
    else:
        L.append("  (no dossier data)")
    L.append("")
    L.append("EW24 BASELINE")
    if dr is not None and pd.notna(dr.get("ew24_pct")):
        L.append(f"  EW24 BSW:       {dr['ew24_pct']:.2f}%")
        L.append(f"  EW24 level:     {dr['ew24_level']}")
    else:
        L.append("  (no EW24 data)")
    L.append("")
    L.append("BALLOT ORDER")
    if dr is not None and pd.notna(dr.get("bsw_pos")):
        L.append(f"  BSW position:   {int(dr['bsw_pos'])}")
        L.append(f"  BD position:    {int(dr['bd_pos'])}")
        adj = dr.get("ballot_adj")
        L.append(f"  Adjacent:       "
                 f"{'YES' if adj else 'NO'}")
    else:
        L.append("  (no ballot data)")
    L.append("")


def _checklist(L):
    L.append("EVIDENCE TO OBTAIN")
    L.append("  [ ] Wahlniederschrift (tally sheet)")
    L.append("  [ ] Recount minutes / correction log")
    L.append("  [ ] Invalid-ballot review records")
    L.append("  [ ] Municipality correspondence")
    L.append("  [ ] Returning officer contact")
    L.append("  [ ] Local media coverage")
    L.append("  [ ] Witness statements")
    L.append("")


def generate_one(r, dos, reg, aff, rank):
    L = [SEP, f"RECOUNT CASE FILE — Rank #{rank}",
         SEP, ""]
    _location(r, L)
    _official(r, L)
    _stats(r, L)
    _registry(r, reg, L)
    _affidavit(r, aff, L)
    _dossier_ctx(r, dos, L)
    _checklist(L)
    L.append(SEP)
    return "\n".join(L)


def _slug(r):
    gem = str(r["gemeinde"]).replace(" ","_")
    gem = gem.replace(",","").replace("/","_")[:40]
    return (f"{r['land']}_wkr{int(r['wahlkreis']):03d}"
            f"_{gem}")


def main():
    targets = load_targets()
    dos = _load("evidence_dossier.csv")
    reg = _load("evidence_registry.csv")
    aff = _load("affidavit_analysis.csv")
    CASEDIR.mkdir(exist_ok=True)
    print(f"Generating {len(targets)} case files...")
    for rank, (_, r) in enumerate(
            targets.iterrows(), 1):
        text = generate_one(r, dos, reg, aff, rank)
        slug = _slug(r)
        out = CASEDIR / f"{slug}.txt"
        out.write_text(text, encoding="utf-8")
    print(f"  Wrote {len(targets)} files → {CASEDIR}/")
    _write_index(targets)


def _write_index(targets):
    hdr = (f"{'#':>3} {'Land':<4} {'WKR':>4}"
           f" {'BSW':>4} {'BD':>4} {'miss':>7}"
           f" {'ov':>3} Gemeinde")
    idx = ["RECOUNT TARGETS — INDEX",
           SEP, "", hdr, "-"*60]
    for rk, (_, r) in enumerate(
            targets.iterrows(), 1):
        idx.append(f"{rk:>3} {r['land']:<4}"
            f" {int(r['wahlkreis']):>4}"
            f" {int(r['bsw_votes']):>4}"
            f" {int(r['bd_votes']):>4}"
            f" {r['miss_lb']:>7.1f}"
            f" {r['overlay_score']:>3.0f}"
            f" {r['gemeinde']}")
    ip = CASEDIR / "INDEX.txt"
    ip.write_text("\n".join(idx))
    print(f"  Wrote index → {ip}")


if __name__ == "__main__":
    main()
