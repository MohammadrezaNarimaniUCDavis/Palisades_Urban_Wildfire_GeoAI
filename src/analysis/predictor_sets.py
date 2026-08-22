"""Declared predictor sets and the primary modeling frame.

The variable taxonomy (decision D-006) is enforced here: only PRE-FIRE
variables may appear in predictor sets. IMPACT_VARIABLES are listed
explicitly so an automated test can verify none of them leak into models.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import load_config, path

# ---- variable groups ---------------------------------------------------------

M0_TERRAIN = [
    "elevation_pt", "slope_deg_pt", "northness_pt", "eastness_pt",
    "tpi300_pt", "tri_pt", "dist_wildland_m",
]

M1_VEGETATION = [
    # immediate pre-fire condition (three rings)
    "ndvi_pre_r0_30", "ndvi_pre_r30_100", "ndvi_pre_r100_300",
    "ndmi_pre_r0_30", "ndmi_pre_r30_100", "ndmi_pre_r100_300",
    # seasonal anomaly (pre-fire minus 2022-2024 Oct-Dec baseline)
    "ndvi_anom_r0_30", "ndvi_anom_r30_100", "ndvi_anom_r100_300",
    "ndmi_anom_r0_30", "ndmi_anom_r30_100", "ndmi_anom_r100_300",
    # thermal condition (30 m; innermost ring is a single pixel and omitted)
    "lst_pre_r30_100", "lst_pre_r100_300",
    # fuels (subgroup fractions shrub/grass/timber excluded: they sum exactly
    # to fuel_wildland -> perfect collinearity; composition kept descriptive)
    "canopy_cover_r0_30", "canopy_cover_r30_100", "canopy_cover_r100_300",
    "fuel_wildland_r0_30", "fuel_wildland_r30_100", "fuel_wildland_r100_300",
]

M2_BUILT = [
    "nn_building_dist_m", "footprint_area_m2",
    "bld_count_r30", "bld_count_r100", "bld_count_r300",
    "dist_road_m", "road_len_r300_m_per_km2", "dist_deadend_m",
    "yearbuilt", "assessed_value_log",
]

PRE_FIRE_PREDICTORS = M0_TERRAIN + M1_VEGETATION + M2_BUILT

MODEL_BLOCKS = {
    "M0": M0_TERRAIN,
    "M1": M0_TERRAIN + M1_VEGETATION,
    "M2": M0_TERRAIN + M2_BUILT,
    "M3": PRE_FIRE_PREDICTORS,
}

# Further pruning for the interpretable logistic model (|rho|>0.85 / VIF>10
# pairs resolved by domain choice; see outputs/qa/eda_notes.json):
#   tri_pt ~ slope (0.92)          -> keep slope
#   bld_count_r300 ~ r100 (0.90) and ~ road_len (0.89) -> keep r100 + road_len
#   lst_r100_300 ~ lst_r30_100 (0.89) -> keep r30_100
#   fuel_wildland_r30_100 ~ dist_wildland (-0.89) -> keep dist + r0_30/r100_300
LOGIT_EXCLUDE = [
    "tri_pt", "bld_count_r300", "lst_pre_r100_300", "fuel_wildland_r30_100",
    "canopy_cover_r100_300", "ndvi_pre_r100_300",
]
LOGIT_PREDICTORS = [c for c in PRE_FIRE_PREDICTORS if c not in LOGIT_EXCLUDE]

# Impact/event variables that must NEVER enter a susceptibility model
IMPACT_VARIABLES = [
    "dnbr_r0_30", "dnbr_r30_100", "dnbr_r100_300",
    "damage", "damage_ord", "destroyed",
]

# DINS construction attributes: excluded from primary models due to
# differential post-fire observability (outputs/qa/dins_attr_missingness.csv)
OBSERVABILITY_BIASED = [
    "roof_construction", "eaves", "vent_screen", "exterior_siding", "window_pane",
]


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(path("data", "processed", "structure_features.parquet"))
    import numpy as np
    df["assessed_value_log"] = np.log10(df["assessed_value"].clip(lower=1))
    df.loc[df["assessed_value"].isna(), "assessed_value_log"] = float("nan")
    return df


def primary_model_frame(residential_only: bool = True):
    """Return (df, X, y) for the primary analysis."""
    df = load_features()
    if residential_only:
        df = df[df["residential"] == 1].copy()
    # sanity: no impact variable in the predictor list
    leak = set(PRE_FIRE_PREDICTORS) & set(IMPACT_VARIABLES)
    assert not leak, f"impact variables leaked into predictors: {leak}"
    X = df[PRE_FIRE_PREDICTORS].astype(float)
    y = df["destroyed"].astype(int)
    return df, X, y
