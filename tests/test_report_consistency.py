"""Regression tests: report numbers match data files."""

import math
import re
from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
REPORT = (Path(__file__).resolve().parent.parent
          / "docs" / "report.html")


@pytest.fixture(scope="module")
def oc():
    return pd.read_csv(DATA / "official_corrections.csv")


@pytest.fixture(scope="module")
def lt():
    return pd.read_csv(
        DATA / "low_tail_calibration.csv",
        index_col=0)


@pytest.fixture(scope="module")
def html():
    return REPORT.read_text()


def test_bsw_total_in_report(oc, html):
    bsw = int(oc.loc[oc["party"] == "BSW", "final"].iloc[0])
    assert f"{bsw:,}" in html


def test_deficit_in_report(oc, html):
    totals = pd.read_csv(
        DATA / "election_totals.csv", index_col=0)
    total = int(totals.loc["valid_total", "value"])
    bsw = int(oc.loc[oc["party"] == "BSW", "final"].iloc[0])
    threshold = math.ceil(total * 0.05)
    deficit = threshold - bsw
    assert f"{deficit:,}" in html


def test_bsw_delta_in_report(oc, html):
    delta = int(oc.loc[oc["party"] == "BSW", "delta"].iloc[0])
    assert f"{delta:,}" in html


def test_low_tail_excess_in_report(lt, html):
    excess = lt.loc["excess_miss", "v"]
    assert f"{excess:,.0f}" in html


def test_low_tail_pvalue_in_report(lt, html):
    p = lt.loc["n_p", "v"]
    assert f"{p:.3f}" in html


def test_metrics_csv_matches_report(html):
    lr = pd.read_csv(
        DATA / "wahlbezirk_lr_metrics.csv",
        index_col=0)
    r2 = lr.loc["BSW", "R2"]
    assert f"{r2:.4f}" in html


def test_no_stale_hardcoded_totals(html):
    """Old wrong numbers must not appear."""
    assert "2,410,553" not in html
    assert "2,420,082" not in html


# --- README consistency tests ---

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def readme():
    return (ROOT / "README.md").read_text()


def test_readme_triangulation(readme):
    tri = pd.read_csv(DATA / "triangulation_overlap.csv")
    t20 = int(tri[tri["top_n"]==20]["pct"].iloc[0])
    t50 = int(tri[tri["top_n"]==50]["pct"].iloc[0])
    assert f"Top-20 overlap: {t20}%" in readme
    assert f"Top-50: {t50}%" in readme


def test_readme_bb_excess(readme):
    bb = pd.read_csv(
        DATA / "zero_calib_betabinom_land.csv")
    for land in ["HE", "NI", "BY"]:
        row = bb[bb["land"] == land]
        if len(row):
            ex = row.iloc[0]["exc_bb"]
            assert f"{ex:+.1f}" in readme
