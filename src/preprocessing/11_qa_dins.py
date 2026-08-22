"""DINS coverage and quality audit (blueprint Gate G3).

Produces outputs/qa/:
  - dins_audit.json            (counts, duplicates, geometry validity, coverage)
  - dins_class_counts.csv      (damage class x structure category)
  - dins_attr_missingness.csv  (attribute missingness BY damage class -> observability bias)
and data/interim/dins_clean.gpkg (deduplicated, classed, in analysis CRS).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, load_config, path

log = get_logger("11_qa_dins")

ATTR_FIELDS = [
    "YEARBUILT", "ROOFCONSTRUCTION", "EAVES", "VENTSCREEN", "EXTERIORSIDING",
    "WINDOWPANE", "DECKPORCHONGRADE", "DECKPORCHELEVATED", "PATIOCOVERCARPORT",
    "FENCEATTACHEDTOSTRUCTURE", "ASSESSEDIMPROVEDVALUE", "DEFENSIVEACTIONS",
]
UNKNOWN_TOKENS = {"", "unknown", "unknown - not applicable", None}


def is_unknown(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    s = str(v).strip().lower()
    return s in UNKNOWN_TOKENS or s.startswith("unknown")


def main() -> None:
    cfg = load_config()
    crs = cfg["crs"]["analysis"]

    dins = gpd.read_file(path("data", "raw", "dins", "dins_palisades_raw.geojson"))
    per = gpd.read_file(path("data", "processed", "study_area.gpkg"), layer="perimeter").to_crs(crs)
    study = gpd.read_file(path("data", "processed", "study_area.gpkg"), layer="study_area").to_crs(crs)

    audit: dict = {"n_raw": int(len(dins))}

    # --- geometry validity ----------------------------------------------------
    dins = dins.set_crs(4326, allow_override=True)
    missing_geom = int(dins.geometry.isna().sum() + (~dins.geometry.is_valid).sum())
    audit["missing_or_invalid_geometry"] = missing_geom
    dins = dins[dins.geometry.notna() & dins.geometry.is_valid].to_crs(crs)

    # coordinate sanity: LATITUDE/LONGITUDE fields vs geometry
    ll = dins.to_crs(4326)
    dev = np.sqrt(
        (ll.geometry.x - ll["LONGITUDE"].astype(float)) ** 2
        + (ll.geometry.y - ll["LATITUDE"].astype(float)) ** 2
    )
    audit["coord_field_mismatch_gt_100m"] = int((dev > 0.001).sum())

    # --- duplicates -------------------------------------------------------------
    audit["duplicate_globalid"] = int(dins["GLOBALID"].duplicated().sum())
    xy = np.round(np.c_[dins.geometry.x, dins.geometry.y], 1)
    dins["_xy"] = [f"{a}_{b}" for a, b in xy]
    dup_xy = dins["_xy"].duplicated(keep=False)
    audit["duplicate_location_0p1m"] = int(dup_xy.sum())
    # keep the most complete record per exact location (most non-null attrs)
    dins["_completeness"] = dins[ATTR_FIELDS].notna().sum(axis=1)
    dins = (
        dins.sort_values("_completeness", ascending=False)
        .drop_duplicates(subset="_xy", keep="first")
        .drop(columns=["_xy", "_completeness"])
    )
    audit["n_after_dedup"] = int(len(dins))

    # --- class distribution -------------------------------------------------------
    ct = pd.crosstab(dins["DAMAGE"], dins["STRUCTURECATEGORY"], margins=True)
    ct.to_csv(path("outputs", "qa", "dins_class_counts.csv"))
    audit["damage_counts"] = dins["DAMAGE"].value_counts().to_dict()
    audit["category_counts"] = dins["STRUCTURECATEGORY"].value_counts().to_dict()

    # --- spatial coverage -----------------------------------------------------------
    inside_per = dins.within(per.geometry.iloc[0])
    inside_study = dins.within(study.geometry.iloc[0])
    audit["inside_perimeter"] = int(inside_per.sum())
    audit["outside_perimeter_within_study"] = int((~inside_per & inside_study).sum())
    audit["outside_study_area"] = int((~inside_study).sum())

    # --- missingness by damage class (observability bias audit) ----------------------
    rows = []
    for dmg, grp in dins.groupby("DAMAGE"):
        for f in ATTR_FIELDS:
            unk = grp[f].apply(is_unknown).mean()
            rows.append({"damage": dmg, "field": f, "pct_unknown": round(100 * unk, 1),
                         "n": len(grp)})
    miss = pd.DataFrame(rows).pivot(index="field", columns="damage", values="pct_unknown")
    miss.to_csv(path("outputs", "qa", "dins_attr_missingness.csv"))
    log.info("attribute missingness by class written")

    # YEARBUILT validity
    yb = pd.to_numeric(dins["YEARBUILT"], errors="coerce")
    audit["yearbuilt_missing_or_zero"] = int(((yb.isna()) | (yb < 1800)).sum())
    audit["yearbuilt_range"] = [float(yb[yb >= 1800].min()), float(yb[yb >= 1800].max())]

    # inspection dates sanity
    if "INCIDENTSTARTDATE" in dins.columns:
        d = pd.to_datetime(dins["INCIDENTSTARTDATE"], unit="ms", errors="coerce")
        audit["incidentstart_range"] = [str(d.min()), str(d.max())]

    # --- outcome coding ------------------------------------------------------------
    order = cfg["dins"]["damage_order"]
    dins = dins[~dins["DAMAGE"].isin(cfg["dins"]["excluded_damage"])].copy()
    dins["damage_ord"] = dins["DAMAGE"].map({c: i for i, c in enumerate(order)})
    dins["destroyed"] = (dins["DAMAGE"] == cfg["model"]["outcome_positive"]).astype(int)
    dins["residential"] = dins["STRUCTURECATEGORY"].isin(
        cfg["dins"]["residential_categories"]).astype(int)
    audit["n_modeling_all"] = int(len(dins))
    audit["n_modeling_residential"] = int(dins["residential"].sum())
    audit["destroyed_rate_residential"] = float(
        dins.loc[dins["residential"] == 1, "destroyed"].mean())

    out = path("data", "interim", "dins_clean.gpkg")
    dins.to_file(out, layer="dins", driver="GPKG")
    log.info("wrote %s (%d records)", out, len(dins))

    audit_fp = path("outputs", "qa", "dins_audit.json")
    audit_fp.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    log.info("audit: %s", json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
