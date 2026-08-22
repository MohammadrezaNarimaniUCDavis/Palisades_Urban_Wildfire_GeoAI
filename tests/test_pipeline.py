"""Sanity checks for critical pipeline assumptions (blueprint section 8.3).

Run with:  conda run -n gee python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.project import load_config, path  # noqa: E402
from analysis.predictor_sets import (IMPACT_VARIABLES, PRE_FIRE_PREDICTORS,  # noqa: E402
                                     MODEL_BLOCKS, OBSERVABILITY_BIASED)

cfg = load_config()


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return pd.read_parquet(path("data", "processed", "structure_features.parquet"))


@pytest.fixture(scope="module")
def dins() -> gpd.GeoDataFrame:
    return gpd.read_file(path("data", "interim", "dins_clean.gpkg"), layer="dins")


def test_no_impact_variables_in_predictors():
    """Post-fire/impact variables must never enter a susceptibility model."""
    for name, cols in MODEL_BLOCKS.items():
        leak = set(cols) & set(IMPACT_VARIABLES)
        assert not leak, f"{name} contains impact variables: {leak}"
        assert not any(c.startswith("dnbr") for c in cols), f"dNBR leaked into {name}"


def test_observability_biased_attributes_excluded():
    for cols in MODEL_BLOCKS.values():
        assert not set(cols) & set(OBSERVABILITY_BIASED)


def test_crs_analysis_layers():
    study = gpd.read_file(path("data", "processed", "study_area.gpkg"), layer="study_area")
    assert study.crs.to_epsg() == 26911


def test_dins_unique_ids(dins):
    assert dins["GLOBALID"].is_unique


def test_dins_coordinates_within_study(dins):
    study = gpd.read_file(path("data", "processed", "study_area.gpkg"),
                          layer="study_area").to_crs(dins.crs)
    inside = dins.within(study.geometry.iloc[0])
    assert inside.all(), f"{(~inside).sum()} structures outside study area"


def test_outcome_coding(dins):
    assert set(dins["destroyed"].unique()) <= {0, 1}
    assert "Inaccessible" not in set(dins["DAMAGE"].unique())


def test_index_value_ranges(features):
    for col in ["ndvi_pre_r30_100", "ndmi_pre_r30_100"]:
        v = features[col].dropna()
        assert v.between(-1, 1).all(), f"{col} outside [-1, 1]"


def test_yearbuilt_plausible(features):
    v = features["yearbuilt"].dropna()
    assert v.between(1800, 2025).all()


def test_predictor_temporal_provenance():
    """Every primary predictor must be a declared pre-fire variable."""
    prefire_ok = {
        # spectral composites end 2025-01-06 (config prefire_window)
        "ndvi", "ndmi", "lst", "canopy", "fuel", "dist_wildland",
        # static terrain
        "elevation", "slope", "northness", "eastness", "tpi", "tri",
        # pre-fire OSM snapshot (2025-01-01) / assessor attributes
        "nn_building", "footprint", "bld_count", "dist_road", "road_len",
        "dist_deadend", "yearbuilt", "assessed_value",
    }
    for c in PRE_FIRE_PREDICTORS:
        assert any(c.startswith(p) for p in prefire_ok), f"unclassified predictor: {c}"
    end = pd.Timestamp(cfg["temporal"]["prefire_window"][1])
    assert end < pd.Timestamp(cfg["study_area"]["event_start"])


def test_sample_sizes(features):
    assert len(features) > 10000
    res = features[features["residential"] == 1]
    assert len(res) > 9000
    assert 0.4 < res["destroyed"].mean() < 0.7
