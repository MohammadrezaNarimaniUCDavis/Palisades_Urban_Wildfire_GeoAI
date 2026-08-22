"""Build the analysis study area: final Palisades perimeter + 2 km buffer (EPSG:26911).

Outputs data/processed/study_area.gpkg with layers:
  - perimeter      (final WFIGS polygon, analysis CRS)
  - study_area     (perimeter buffered 2 km, analysis CRS)
  - bbox_wgs84     (study-area bounding box in EPSG:4326, for API/GEE queries)
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, load_config, path

log = get_logger("10_build_study_area")


def main() -> None:
    cfg = load_config()
    crs = cfg["crs"]["analysis"]
    buffer_m = cfg["study_area"]["buffer_m"]

    raw = path("data", "raw", "perimeter", "wfigs_palisades_raw.geojson")
    gdf = gpd.read_file(raw)
    assert len(gdf) >= 1, "no perimeter features"
    if len(gdf) > 1:  # keep the largest polygon if multiple vintages exist
        gdf = gdf.loc[[gdf.to_crs(crs).area.idxmax()]]
    per = gdf.to_crs(crs)
    per["geometry"] = per.geometry.buffer(0)  # fix any invalid rings
    assert per.geometry.is_valid.all(), "invalid perimeter geometry"

    area_km2 = float(per.geometry.area.iloc[0]) / 1e6
    acres = area_km2 * 247.105
    log.info("perimeter area: %.1f km2 (%.0f acres; WFIGS GISAcres=%s)",
             area_km2, acres, per.get("poly_GISAcres", ["?"]).iloc[0])

    study = per.copy()
    study["geometry"] = study.geometry.buffer(buffer_m)
    study_km2 = float(study.geometry.area.iloc[0]) / 1e6
    log.info("study area (perimeter + %dm buffer): %.1f km2", buffer_m, study_km2)

    bbox_wgs = gpd.GeoDataFrame(
        {"name": ["study_bbox"]},
        geometry=[box(*study.to_crs(4326).total_bounds)],
        crs="EPSG:4326",
    )
    log.info("WGS84 bbox: %s", list(study.to_crs(4326).total_bounds))

    out = path("data", "processed", "study_area.gpkg")
    per.to_file(out, layer="perimeter", driver="GPKG")
    study.to_file(out, layer="study_area", driver="GPKG")
    bbox_wgs.to_file(out, layer="bbox_wgs84", driver="GPKG")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
