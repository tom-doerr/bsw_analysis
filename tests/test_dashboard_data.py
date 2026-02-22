"""Tests for generated dashboard JSON files."""

import json
import math
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs" / "data"


@pytest.fixture(scope="module")
def wkr_json():
    with open(DOCS / "wkr_data.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def geo_json():
    with open(DOCS / "wahlkreise.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def summary_json():
    with open(DOCS / "summary.json") as f:
        return json.load(f)


def test_wkr_data_json_valid(wkr_json):
    assert len(wkr_json) == 299
    required = {"wkr", "name", "land", "turnout",
                "shares", "resid", "swing"}
    for d in wkr_json:
        assert required <= set(d.keys())


def test_geojson_valid(geo_json):
    assert geo_json["type"] == "FeatureCollection"
    feats = geo_json["features"]
    assert len(feats) == 299
    for f in feats:
        assert "wkr" in f["properties"]
        assert "name" in f["properties"]
        assert f["geometry"]["type"] in (
            "Polygon", "MultiPolygon"
        )


def test_wkr_geo_alignment(wkr_json, geo_json):
    data_ids = {d["wkr"] for d in wkr_json}
    geo_ids = {f["properties"]["wkr"]
               for f in geo_json["features"]}
    assert data_ids == geo_ids, (
        f"Mismatch: data-only={data_ids - geo_ids}, "
        f"geo-only={geo_ids - data_ids}"
    )


def test_summary_json_valid(summary_json):
    assert "parties" in summary_json
    assert "n_precincts" in summary_json
    assert summary_json["n_precincts"] > 0
    assert len(summary_json["parties"]) == 9


def test_geo_coords_in_germany(geo_json):
    """All coordinates within Germany bounding box."""
    for feat in geo_json["features"]:
        geom = feat["geometry"]
        coords = geom["coordinates"]
        if geom["type"] == "Polygon":
            coords = [coords]
        for poly in coords:
            for ring in poly:
                for lon, lat in ring:
                    assert 5.5 < lon < 15.5, (
                        f"lon {lon} out of range")
                    assert 47 < lat < 55.5, (
                        f"lat {lat} out of range")


def test_projection_bounds(geo_json):
    """Projected coords fit within camera view."""
    CENTER = [10.4, 51.1]
    SCALE = 6
    cos_c = math.cos(CENTER[1] * math.pi / 180)
    max_x = max_z = 0
    for feat in geo_json["features"]:
        geom = feat["geometry"]
        coords = geom["coordinates"]
        if geom["type"] == "Polygon":
            coords = [coords]
        for poly in coords:
            for ring in poly:
                for lon, lat in ring:
                    x = abs((lon-CENTER[0])*SCALE*cos_c)
                    z = abs((lat-CENTER[1])*SCALE)
                    max_x = max(max_x, x)
                    max_z = max(max_z, z)
    assert max_x < 25, f"max_x={max_x:.1f} > 25"
    assert max_z < 30, f"max_z={max_z:.1f} > 30"
