"""Download CAL FIRE DINS damage-inspection records for the 2025 Palisades Fire.

Queries the official ArcGIS REST layer with pagination, saves the raw
GeoJSON snapshot untouched, and records provenance in the data manifest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import file_info, get_logger, load_config, path, record_manifest, utcnow

log = get_logger("01_download_dins")


def fetch_all_features(base_url: str, where: str, out_fields: str = "*") -> dict:
    """Fetch all features from an ArcGIS REST layer with result pagination."""
    features: list = []
    offset = 0
    page = 2000
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page,
            "orderByFields": "OBJECTID",
        }
        r = requests.get(f"{base_url}/query", params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"ArcGIS error: {data['error']}")
        chunk = data.get("features", [])
        features.extend(chunk)
        log.info("fetched %d features (offset %d)", len(chunk), offset)
        if len(chunk) < page:
            break
        offset += page
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    cfg = load_config()
    base = cfg["dins"]["rest_url"]
    like = cfg["study_area"]["dins_incident_like"]
    where = (
        f"UPPER(INCIDENTNAME) LIKE '%{like}%' "
        "AND INCIDENTSTARTDATE >= TIMESTAMP '2025-01-01 00:00:00'"
    )

    # Layer metadata snapshot (fields, service edit date) for provenance
    meta = requests.get(f"{base}?f=json", timeout=60).json()
    meta_fp = path("data", "raw", "dins", "dins_layer_metadata.json")
    meta_fp.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    fc = fetch_all_features(base, where)
    log.info("total DINS features: %d", len(fc["features"]))

    out = path("data", "raw", "dins", "dins_palisades_raw.geojson")
    out.write_text(json.dumps(fc), encoding="utf-8")
    log.info("wrote %s", out)

    record_manifest({
        "dataset_id": "D01",
        "dataset_name": "CAL FIRE Damage Inspection (DINS) - Palisades 2025",
        "provider": "CAL FIRE / California Open Data",
        "landing_page": "https://data.ca.gov/dataset/cal-fire-damage-inspection-dins-data",
        "access_url": base,
        "access_method": f"ArcGIS REST paginated query; where={where}",
        "license": "Open government data (California Open Data terms)",
        "temporal_coverage": "Inspections following 2025-01-07 ignition",
        "spatial_coverage": "Palisades Fire, Los Angeles County, CA",
        "resolution": "Inspected structure (point)",
        "native_crs": "EPSG:4326 (requested outSR)",
        "variables": "DAMAGE; STRUCTURETYPE; STRUCTURECATEGORY; YEARBUILT; construction attributes; coordinates",
        "analytical_role": "Primary outcome (structure damage)",
        "retrieved_utc": utcnow(),
        "local_path": str(out.relative_to(out.parents[3])),
        "version": str(meta.get("editingInfo", {}).get("lastEditDate", "")),
        "limitations": "Inspected structures only; uninspected structures are unknown, not undamaged",
        "credentials_required": "No",
        **file_info(out),
    })
    log.info("manifest updated")


if __name__ == "__main__":
    main()
