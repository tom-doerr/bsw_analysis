"""Tests for prep_dashboard.py data pipeline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prep_dashboard import (
    PARTIES, load_wkr_votes,
    load_residuals_by_wkr,
    build_wkr_data, build_summary,
)


@pytest.fixture(scope="module")
def wkr_votes():
    wkr, vcol = load_wkr_votes()
    return wkr, vcol


@pytest.fixture(scope="module")
def wkr_data():
    return build_wkr_data()


def test_load_wkr_votes_shape(wkr_votes):
    wkr, vcol = wkr_votes
    assert len(wkr) == 299, f"Expected 299 WKR, got {len(wkr)}"
    assert vcol in wkr.columns


def test_load_wkr_votes_parties(wkr_votes):
    wkr, _ = wkr_votes
    for p in PARTIES:
        col = f"{p} - Zweitstimmen"
        if p == "CSU":
            continue  # CSU only in Bavaria
        assert col in wkr.columns, f"Missing {col}"


def test_load_residuals_shape():
    resid = load_residuals_by_wkr()
    assert len(resid) >= 290, f"Expected ~299 rows, got {len(resid)}"
    assert "n_precincts" in resid.columns


def test_build_wkr_data_structure(wkr_data):
    assert len(wkr_data) == 299
    required = {"wkr", "name", "land", "turnout",
                "valid", "shares", "resid", "swing",
                "n_precincts"}
    for d in wkr_data:
        assert required <= set(d.keys()), f"Missing keys in WKR {d.get('wkr')}"


def test_wkr_shares_sum(wkr_data):
    for d in wkr_data:
        total = sum(d["shares"].values())
        assert 80 < total < 105, (
            f"WKR {d['wkr']} shares sum to {total:.1f}"
        )


def test_wkr_turnout_range(wkr_data):
    for d in wkr_data:
        assert 30 < d["turnout"] < 100, (
            f"WKR {d['wkr']} turnout={d['turnout']}"
        )


def test_build_summary_structure():
    s = build_summary()
    assert "parties" in s
    assert "n_precincts" in s
    for p in PARTIES:
        assert p in s["parties"], f"Missing party {p}"


def test_wkr_swing_present(wkr_data):
    has_swing = sum(1 for d in wkr_data
                    if d.get("swing"))
    assert has_swing >= 290, (
        f"Only {has_swing}/299 WKR have swing data")


def test_summary_r2_range():
    s = build_summary()
    for p, d in s["parties"].items():
        for k in ("lr_r2", "xgb_r2"):
            if k in d and d[k] is not None:
                assert 0 <= d[k] <= 1, f"{p} {k}={d[k]}"
