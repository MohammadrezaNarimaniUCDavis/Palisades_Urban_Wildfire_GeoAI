"""Export publication tables from pipeline outputs (CSV + XLSX + LaTeX).

Table 1  Data sources and roles
Table 2  Predictor definitions
Table 3  Sample characteristics (DINS damage x category)
Table 4  Model performance (spatial vs random CV)
Table 5  Key effects (odds ratios + SHAP importance)
Table 6  Sensitivity analyses
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, load_config, path

log = get_logger("50_make_tables")


def export(df: pd.DataFrame, name: str, index: bool = False) -> None:
    df.to_csv(path("outputs", "tables", f"{name}.csv"), index=index)
    df.to_excel(path("outputs", "tables", f"{name}.xlsx"), index=index)
    with open(path("outputs", "tables", f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=index, escape=True))
    log.info("exported %s (%d rows)", name, len(df))


def table1_data_sources() -> None:
    rows = [
        ["CAL FIRE DINS", "CAL FIRE / California Open Data", "Point (structure)",
         "Post-fire inspections, 2025", "Damage class, structure category, year built",
         "Primary outcome"],
        ["WFIGS Interagency Fire Perimeters", "NIFC", "Polygon",
         "Final 2025 perimeter", "Fire perimeter geometry", "Study area definition"],
        ["Sentinel-2 L2A (via GEE)", "Copernicus/ESA", "10-20 m",
         "Oct 2024-Jan 2025 (pre); Oct-Dec 2022-2024 (baseline); Jan-Feb 2025 (post); monthly to Jul 2026",
         "NDVI, NDMI, NBR/dNBR", "Pre-fire vegetation predictors; severity & recovery"],
        ["Landsat 8/9 C2 L2 (via GEE)", "USGS", "30 m", "Oct 2024-Jan 2025",
         "Land surface temperature", "Pre-fire thermal predictor"],
        ["LANDFIRE LF2024", "USGS/USDA-FS", "30 m", "2024 update (pre-fire vintage)",
         "FBFM40 fuel models, canopy cover", "Fuel type/continuity predictors"],
        ["USGS 3DEP DEM (via GEE)", "USGS", "10 m", "Static",
         "Elevation, slope, aspect, TPI, TRI", "Terrain predictors"],
        ["OpenStreetMap (Overpass attic)", "OSM contributors (ODbL)", "Feature",
         "Snapshot 2025-01-01 (pre-fire)", "Buildings, roads, facilities",
         "Built-environment predictors; accessibility"],
        ["gridMET (via GEE)", "Climatology Lab", "~4 km daily",
         "Apr 2024-Jan 2025; Oct-Dec 1980-2024", "Precipitation, VPD, fuel moisture, ERC",
         "Antecedent climate context"],
        ["ERA5-Land (via GEE)", "ECMWF/Copernicus", "~9 km hourly", "Jan 5-15, 2025",
         "Wind, temperature, dewpoint", "Event weather context"],
        ["CDC/ATSDR SVI 2022", "CDC/ATSDR", "Census tract", "2022 release",
         "Vulnerability themes, vehicle access, age", "Community context (Track B)"],
        ["TIGER/Line 2023", "U.S. Census Bureau", "Tract/block group", "2023 vintage",
         "Census geometries", "Community units"],
    ]
    df = pd.DataFrame(rows, columns=["Dataset", "Provider", "Resolution",
                                     "Temporal window used", "Variables derived",
                                     "Analytical role"])
    export(df, "Table_01_data_sources")


def table2_predictors() -> None:
    from analysis.predictor_sets import M0_TERRAIN, M1_VEGETATION, M2_BUILT
    desc = {
        "elevation_pt": ("Elevation at structure (m)", "3DEP 10 m", "point"),
        "slope_deg_pt": ("Slope (degrees)", "3DEP 10 m", "point"),
        "northness_pt": ("cos(aspect)", "3DEP 10 m", "point"),
        "eastness_pt": ("sin(aspect)", "3DEP 10 m", "point"),
        "tpi300_pt": ("Topographic position index (300 m)", "3DEP 10 m", "point"),
        "tri_pt": ("Terrain ruggedness index", "3DEP 10 m", "point"),
        "dist_wildland_m": ("Distance to nearest wildland fuel pixel (m)", "LANDFIRE FBFM40", "point"),
    }
    rows = []
    for c in M0_TERRAIN:
        d = desc.get(c, (c, "", ""))
        rows.append(["Terrain (M0)", c, d[0], d[1], d[2]])
    for c in M1_VEGETATION:
        base = ("NDVI" if c.startswith("ndvi_pre") else
                "NDMI" if c.startswith("ndmi_pre") else
                "NDVI anomaly (pre minus 2022-2024 baseline)" if c.startswith("ndvi_anom") else
                "NDMI anomaly (pre minus 2022-2024 baseline)" if c.startswith("ndmi_anom") else
                "Land surface temperature (deg C)" if c.startswith("lst") else
                "Canopy cover (%)" if c.startswith("canopy") else
                "Wildland burnable fuel fraction")
        ring = c.split("_r")[-1].replace("_", "-") + " m ring"
        src = ("Sentinel-2 10 m" if "nd" in c[:4] else
               "Landsat 30 m" if c.startswith("lst") else "LANDFIRE 30 m")
        rows.append(["Vegetation & fuels (M1)", c, base, src, ring])
    desc2 = {
        "nn_building_dist_m": ("Edge distance to nearest neighboring building (m)", "OSM buildings 2025-01-01", "structure"),
        "footprint_area_m2": ("Matched building footprint area (m2)", "OSM buildings", "structure"),
        "bld_count_r30": ("Building count within 30 m", "OSM buildings", "30 m disk"),
        "bld_count_r100": ("Building count within 100 m", "OSM buildings", "100 m disk"),
        "bld_count_r300": ("Building count within 300 m", "OSM buildings", "300 m disk"),
        "dist_road_m": ("Distance to nearest drivable road (m)", "OSM roads 2025-01-01", "point"),
        "road_len_r300_m_per_km2": ("Road length density within 300 m", "OSM roads", "300 m disk"),
        "dist_deadend_m": ("Distance to nearest dead-end node (m)", "OSM roads", "point"),
        "yearbuilt": ("Year built (assessor)", "DINS/assessor", "structure"),
        "assessed_value_log": ("log10 assessed improved value", "DINS/assessor", "structure"),
    }
    for c in M2_BUILT:
        d = desc2.get(c, (c, "", ""))
        rows.append(["Built environment (M2)", c, d[0], d[1], d[2]])
    df = pd.DataFrame(rows, columns=["Block", "Variable", "Definition", "Source", "Support"])
    export(df, "Table_02_predictors")


def table3_sample() -> None:
    ct = pd.read_csv(path("outputs", "qa", "dins_class_counts.csv"), index_col=0)
    export(ct.reset_index(), "Table_03_sample")


def table4_performance() -> None:
    cv = pd.read_csv(path("outputs", "model_diagnostics", "cv_metrics.csv"))
    g = (cv.groupby(["cv_scheme", "model_block", "learner"])
         [["roc_auc", "pr_auc", "balanced_accuracy", "f1", "brier"]]
         .agg(["mean", "std"]))
    out = pd.DataFrame(index=g.index)
    for m in ["roc_auc", "pr_auc", "balanced_accuracy", "f1", "brier"]:
        out[m] = (g[(m, "mean")].round(3).astype(str) + " (" +
                  g[(m, "std")].round(3).astype(str) + ")")
    out = out.reset_index()
    export(out, "Table_04_performance")


def table5_effects() -> None:
    lc = pd.read_csv(path("outputs", "model_diagnostics", "logit_coefficients.csv"),
                     index_col=0).drop(index="const")
    shp = pd.read_csv(path("outputs", "model_diagnostics", "shap_importance.csv"))
    shp_rank = {p: i + 1 for i, p in enumerate(shp["predictor"])}
    shp_val = dict(zip(shp["predictor"], shp["mean_abs_shap"]))
    lc = lc.sort_values("or", ascending=False)
    df = pd.DataFrame({
        "Variable": lc.index,
        "OR per SD": lc["or"].round(3),
        "95% CI": ("(" + lc["or_lo"].round(3).astype(str) + ", "
                   + lc["or_hi"].round(3).astype(str) + ")"),
        "p": lc["p"].apply(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"),
        "Mean |SHAP| (XGB)": [round(shp_val.get(v, np.nan), 3) for v in lc.index],
        "SHAP rank": [shp_rank.get(v, "-") for v in lc.index],
    })
    export(df, "Table_05_effects")


def table6_sensitivity() -> None:
    sc = pd.read_csv(path("outputs", "model_diagnostics", "scale_sensitivity.csv"))
    bs = pd.read_csv(path("outputs", "model_diagnostics", "blocksize_sensitivity.csv"))
    oc = pd.read_csv(path("outputs", "model_diagnostics", "outcome_sensitivity.csv"))
    rows = []
    for _, r in sc.iterrows():
        rows.append(["Ring scale", r["scale"].replace("_", "-") + " m"
                     if r["scale"] != "all_scales" else "All scales",
                     f"{r['auc_mean']:.3f} ({r['auc_sd']:.3f})",
                     f"{r['prauc_mean']:.3f}", int(r["n_predictors"])])
    for _, r in bs.iterrows():
        rows.append(["Block size", f"{int(r['block_m'])} m",
                     f"{r['auc_mean']:.3f} ({r['auc_sd']:.3f})",
                     f"{r['prauc_mean']:.3f}", int(r["n_predictors"])])
    for _, r in oc.iterrows():
        lab = ("Destroyed+Major, residential" if "major" in r["variant"]
               else "Destroyed, all structures")
        rows.append(["Outcome/population", lab,
                     f"{r['auc_mean']:.3f} ({r['auc_sd']:.3f})",
                     f"{r['prauc_mean']:.3f}", int(r["n_predictors"])])
    df = pd.DataFrame(rows, columns=["Dimension", "Specification",
                                     "Spatial-CV AUC (SD)", "PR-AUC", "N predictors"])
    export(df, "Table_06_sensitivity")


if __name__ == "__main__":
    table1_data_sources()
    table2_predictors()
    table3_sample()
    table4_performance()
    table5_effects()
    table6_sensitivity()
    log.info("tables complete")
