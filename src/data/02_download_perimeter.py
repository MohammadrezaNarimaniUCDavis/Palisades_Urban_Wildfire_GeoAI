"""Download the final WFIGS interagency perimeter for the 2025 Palisades Fire.

Queries the NIFC WFIGS ArcGIS REST service, selects the Palisades (2025)
incident polygon, archives the raw response, and records provenance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import file_info, get_logger, load_config, path, record_manifest, utcnow

log = get_logger("02_download_perimeter")

# Candidate WFIGS perimeter services (field schemas share attr_* prefixes)
SERVICES = [
    # WFIGS Interagency Fire Perimeters (current + historic, all years)
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0",
    # Fallback: 2025 year-specific layer name variations
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_YearToDate/FeatureServer/0",
]


def query_service(url: str) -> dict | None:
    where = (
        "UPPER(attr_IncidentName) LIKE '%PALISADES%' "
        "AND attr_FireDiscoveryDateTime >= TIMESTAMP '2025-01-01 00:00:00' "
        "AND attr_FireDiscoveryDateTime < TIMESTAMP '2025-02-01 00:00:00'"
    )
    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    try:
        r = requests.get(f"{url}/query", params=params, timeout=180)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            log.warning("service error at %s: %s", url, data["error"])
            return None
        if data.get("features"):
            return data
        log.warning("no features from %s", url)
        return None
    except Exception as e:  # noqa: BLE001 - probing alternative services
        log.warning("failed %s: %s", url, e)
        return None


def main() -> None:
    fc = None
    used = None
    for svc in SERVICES:
        fc = query_service(svc)
        if fc:
            used = svc
            break
    if not fc:
        raise RuntimeError("No WFIGS service returned a Palisades 2025 perimeter")

    feats = fc["features"]
    log.info("retrieved %d perimeter feature(s) from %s", len(feats), used)
    for f in feats:
        a = f["properties"]
        log.info(
            "OBJECTID=%s name=%s acres=%s gis_acres=%s updated=%s",
            a.get("OBJECTID"), a.get("attr_IncidentName"),
            a.get("attr_CalculatedAcres"), a.get("poly_GISAcres"),
            a.get("attr_ModifiedOnDateTime_dt"),
        )

    out = path("data", "raw", "perimeter", "wfigs_palisades_raw.geojson")
    out.write_text(json.dumps(fc), encoding="utf-8")
    log.info("wrote %s", out)

    record_manifest({
        "dataset_id": "D02",
        "dataset_name": "WFIGS Interagency Fire Perimeter - Palisades 2025",
        "provider": "National Interagency Fire Center (NIFC)",
        "landing_page": "https://data-nifc.opendata.arcgis.com/datasets/nifc::wfigs-interagency-fire-perimeters/about",
        "access_url": used,
        "access_method": "ArcGIS REST query (incident name + discovery date filter)",
        "license": "Open government data",
        "temporal_coverage": "2025-01-07 to 2025-01-31 incident",
        "spatial_coverage": "Palisades Fire, Los Angeles County, CA",
        "resolution": "Incident polygon",
        "native_crs": "EPSG:4326 (requested outSR)",
        "variables": "Final perimeter geometry; incident attributes; GIS acres",
        "analytical_role": "Study area boundary",
        "retrieved_utc": utcnow(),
        "local_path": str(out.relative_to(out.parents[3])),
        "version": "",
        "limitations": "Single authoritative geometry archived with OBJECTID and timestamps",
        "credentials_required": "No",
        **file_info(out),
    })
    log.info("manifest updated")


if __name__ == "__main__":
    main()
