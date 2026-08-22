"""Download a pre-fire OpenStreetMap snapshot: buildings, road network, facilities.

Uses date-scoped Overpass queries (attic data at 2025-01-01, i.e. six days
before ignition) so that post-fire edits/deletions do not contaminate the
pre-fire built-environment layers (per acquisition notes section 3).

Outputs (data/raw/osm/):
  - osm_buildings_prefire.gpkg   (building polygons in study area)
  - osm_roads_prefire.gpkg       (drivable edges + nodes)
  - osm_facilities_prefire.gpkg  (fire stations + hospitals within 10 km)
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import osmnx as ox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import file_info, get_logger, load_config, path, record_manifest, utcnow

log = get_logger("05_download_osm")

SNAPSHOT = "2025-01-01T00:00:00Z"


def configure_osmnx() -> None:
    ox.settings.overpass_settings = f'[out:json][timeout:{{timeout}}][date:"{SNAPSHOT}"]'
    ox.settings.requests_timeout = 600
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(path("data", "raw", "osm", "cache"))


def manifest(fp: Path, name: str, variables: str, role: str) -> None:
    record_manifest({
        "dataset_id": "D17", "dataset_name": name,
        "provider": "OpenStreetMap contributors (ODbL)",
        "landing_page": "https://www.openstreetmap.org",
        "access_url": "https://overpass-api.de/api/interpreter",
        "access_method": f'Overpass attic query [date:"{SNAPSHOT}"] via osmnx',
        "license": "ODbL 1.0 (attribution required)",
        "temporal_coverage": f"OSM state at {SNAPSHOT} (pre-fire snapshot)",
        "spatial_coverage": "Palisades study area (+10 km for facilities)",
        "resolution": "feature", "native_crs": "EPSG:4326",
        "variables": variables, "analytical_role": role,
        "retrieved_utc": utcnow(),
        "local_path": str(fp.relative_to(fp.parents[3])),
        "version": SNAPSHOT,
        "limitations": "Volunteered data; completeness audited against DINS/Microsoft footprints",
        "credentials_required": "No", **file_info(fp),
    })


def main() -> None:
    configure_osmnx()
    study = gpd.read_file(path("data", "processed", "study_area.gpkg"), layer="study_area")
    poly_wgs = study.to_crs(4326).geometry.iloc[0]

    # --- buildings ----------------------------------------------------------
    b_fp = path("data", "raw", "osm", "osm_buildings_prefire.gpkg")
    if not b_fp.exists():
        log.info("querying buildings (attic %s) ...", SNAPSHOT)
        bld = ox.features_from_polygon(poly_wgs, tags={"building": True})
        bld = bld.reset_index()
        keep = [c for c in ["osmid", "element", "building", "height", "building:levels",
                            "start_date", "geometry"] if c in bld.columns]
        bld = bld[keep]
        bld = bld[bld.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        bld.to_file(b_fp, layer="buildings", driver="GPKG")
        log.info("buildings: %d polygons -> %s", len(bld), b_fp)
    manifest(b_fp, "OSM buildings (pre-fire snapshot)", "building polygons, type",
             "Building footprints for spacing/density (cross-checked)")

    # --- drivable road network ----------------------------------------------
    r_fp = path("data", "raw", "osm", "osm_roads_prefire.gpkg")
    if not r_fp.exists():
        log.info("querying drivable network (attic %s) ...", SNAPSHOT)
        G = ox.graph_from_polygon(poly_wgs, network_type="drive", simplify=True,
                                  retain_all=True, truncate_by_edge=True)
        nodes, edges = ox.graph_to_gdfs(G)
        edges = edges.reset_index()
        keep = [c for c in ["u", "v", "key", "osmid", "highway", "name", "oneway",
                            "length", "geometry"] if c in edges.columns]
        for c in ("osmid", "highway", "name"):
            if c in keep:
                edges[c] = edges[c].astype(str)
        edges[keep].to_file(r_fp, layer="edges", driver="GPKG")
        nodes.reset_index()[["osmid", "x", "y", "street_count", "geometry"]].to_file(
            r_fp, layer="nodes", driver="GPKG")
        log.info("roads: %d edges, %d nodes -> %s", len(edges), len(nodes), r_fp)
    manifest(r_fp, "OSM drivable road network (pre-fire snapshot)",
             "edges (class, length), nodes (street_count)",
             "Road density/access predictors; network accessibility")

    # --- facilities (10 km buffered search) ----------------------------------
    f_fp = path("data", "raw", "osm", "osm_facilities_prefire.gpkg")
    if not f_fp.exists():
        log.info("querying facilities (attic %s) ...", SNAPSHOT)
        search = study.buffer(10000).to_crs(4326).geometry.iloc[0]
        fac = ox.features_from_polygon(search, tags={"amenity": ["fire_station", "hospital"]})
        fac = fac.reset_index()
        fac["geometry"] = fac.geometry.centroid
        keep = [c for c in ["osmid", "element", "amenity", "name", "emergency", "geometry"]
                if c in fac.columns]
        fac[keep].to_file(f_fp, layer="facilities", driver="GPKG")
        log.info("facilities: %d -> %s", len(fac), f_fp)
    manifest(f_fp, "OSM fire stations & hospitals (pre-fire snapshot)",
             "amenity, name, location", "Critical-service accessibility (Track B)")

    log.info("OSM acquisition complete")


if __name__ == "__main__":
    main()
