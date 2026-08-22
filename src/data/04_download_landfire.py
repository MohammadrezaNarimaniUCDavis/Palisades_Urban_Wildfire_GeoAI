"""Download LANDFIRE LF2024 fuel layers clipped to the study area.

Uses the official LANDFIRE ArcGIS ImageServer `exportImage` endpoint
(no registration required). LF2024 is the most recent PRE-FIRE vintage:
it incorporates disturbances only through 2024, so it cannot leak the
January 2025 fire scar (LF2025 would and is therefore not used).

Products:
  - FBFM40 (Scott & Burgan 40 fire behavior fuel models), 30 m
  - CC (forest canopy cover, %), 30 m
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import geopandas as gpd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import file_info, get_logger, load_config, path, record_manifest, utcnow

log = get_logger("04_download_landfire")

SERVICES = {
    "fbfm40": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024/LF2024_FBFM40_CONUS/ImageServer",
    "cc": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024/LF2024_CC_CONUS/ImageServer",
}
RES_M = 30


def study_bounds_26911(pad_m: float = 600.0) -> tuple[float, float, float, float]:
    """Padded UTM bounds matching GEE exports (avoids empty map-frame margins)."""
    cfg = load_config()
    gdf = gpd.read_file(path("data", "processed", "study_area.gpkg"), layer="study_area")
    minx, miny, maxx, maxy = (float(v) for v in gdf.to_crs(cfg["crs"]["analysis"]).total_bounds)
    return (minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)


def export_image(name: str, url: str, bounds: tuple[float, float, float, float],
                 force: bool = False) -> Path:
    out = path("data", "raw", "landfire", f"lf2024_{name}.tif")
    if out.exists() and not force:
        log.info("cached: %s", out.name)
        return out
    if out.exists() and force:
        out.unlink()
        log.info("re-exporting %s ...", name)
    minx, miny, maxx, maxy = bounds
    cols = math.ceil((maxx - minx) / RES_M)
    rows = math.ceil((maxy - miny) / RES_M)
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "26911",
        "imageSR": "26911",
        "size": f"{cols},{rows}",
        "format": "tiff",
        "pixelType": "U8" if name == "cc" else "S16",
        "noData": "",
        "interpolation": "RSP_NearestNeighbor",
        "f": "image",
    }
    r = requests.get(f"{url}/exportImage", params=params, timeout=600)
    r.raise_for_status()
    if not r.content.startswith(b"II") and not r.content.startswith(b"MM"):
        raise RuntimeError(f"{name}: response is not a TIFF ({r.content[:200]!r})")
    out.write_bytes(r.content)
    log.info("wrote %s (%.2f MB, %dx%d @ %dm)", out.name, len(r.content) / 1e6, cols, rows, RES_M)
    return out


def main() -> None:
    bounds = study_bounds_26911()
    log.info("study bounds (26911): %s", bounds)
    for name, url in SERVICES.items():
        # capture service metadata for provenance
        meta = requests.get(f"{url}?f=json", timeout=60).json()
        fp = export_image(name, url, bounds)
        record_manifest({
            "dataset_id": "D06",
            "dataset_name": f"LANDFIRE LF2024 {name.upper()} (study-area clip)",
            "provider": "LANDFIRE (USGS/USDA Forest Service)",
            "landing_page": "https://landfire.gov/fuel/fbfm40" if name == "fbfm40" else "https://landfire.gov/fuel/cc",
            "access_url": f"{url}/exportImage",
            "access_method": f"ArcGIS ImageServer exportImage, bbox EPSG:26911, {RES_M} m, nearest neighbor",
            "license": "Open U.S. government data",
            "temporal_coverage": "LF2024 update (disturbances through 2024; pre-fire vintage)",
            "spatial_coverage": "Palisades study area bbox",
            "resolution": "30 m",
            "native_crs": "EPSG:26911 (requested)",
            "variables": "FBFM40 fuel model codes" if name == "fbfm40" else "Canopy cover (%)",
            "analytical_role": "Pre-fire fuel type/continuity predictors",
            "retrieved_utc": utcnow(),
            "local_path": f"data/raw/landfire/lf2024_{name}.tif",
            "version": str(meta.get("description", ""))[:120],
            "limitations": "30 m product; vintage precedes fire but may miss recent local landscaping",
            "credentials_required": "No",
            **file_info(fp),
        })
    log.info("LANDFIRE acquisition complete")


if __name__ == "__main__":
    main()
