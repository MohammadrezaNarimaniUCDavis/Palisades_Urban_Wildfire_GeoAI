"""Community context (Track B): tract-level damage, social vulnerability, access.

Joins DINS structure outcomes to census tracts, attaches CDC/ATSDR SVI 2022
indicators, and computes NETWORK travel distance from every inspected
structure to the nearest fire station and hospital (pre-fire OSM snapshot
network extended 10 km beyond the study area so external facilities are
reachable). Distances are baseline accessibility measures, not evacuation
performance (decision D-013).

Outputs:
  - data/processed/tract_context.gpkg      (tract polygons + indicators)
  - outputs/tables/tract_context.csv
  - data/processed/structure_access.parquet (per-structure network distances)
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, load_config, path

log = get_logger("34_community")
cfg = load_config()
SNAPSHOT = "2025-01-01T00:00:00Z"


def get_extended_network() -> "nx.MultiDiGraph":
    fp = path("data", "raw", "osm", "osm_network_extended.graphml")
    if fp.exists():
        log.info("loading cached extended network")
        return ox.load_graphml(fp)
    ox.settings.overpass_settings = f'[out:json][timeout:{{timeout}}][date:"{SNAPSHOT}"]'
    ox.settings.requests_timeout = 900
    study = gpd.read_file(path("data", "processed", "study_area.gpkg"), layer="study_area")
    poly = study.buffer(10000).to_crs(4326).geometry.iloc[0]
    log.info("downloading extended drive network (study + 10 km) ...")
    G = ox.graph_from_polygon(poly, network_type="drive", simplify=True,
                              retain_all=False, truncate_by_edge=True)
    ox.save_graphml(G, fp)
    log.info("network: %d nodes, %d edges", len(G.nodes), len(G.edges))
    return G


def main() -> None:
    crs = cfg["crs"]["analysis"]
    feats = pd.read_parquet(path("data", "processed", "structure_features.parquet"))
    pts = gpd.GeoDataFrame(feats[["globalid", "damage", "destroyed", "residential"]],
                           geometry=gpd.points_from_xy(feats["x"], feats["y"]),
                           crs=crs)

    # ---------------- network access -------------------------------------------
    acc_fp = path("data", "processed", "structure_access.parquet")
    if not acc_fp.exists():
        G = get_extended_network()
        fac = gpd.read_file(path("data", "raw", "osm", "osm_facilities_prefire.gpkg"),
                            layer="facilities")
        fac_wgs = fac.to_crs(4326)
        pts_wgs = pts.to_crs(4326)

        node_ids = ox.nearest_nodes(G, pts_wgs.geometry.x.values, pts_wgs.geometry.y.values)
        Gu = ox.convert.to_undirected(G)

        access = {}
        for kind in ("fire_station", "hospital"):
            f = fac_wgs[fac_wgs["amenity"] == kind]
            if not len(f):
                continue
            fnodes = set(ox.nearest_nodes(Gu, f.geometry.x.values, f.geometry.y.values))
            dist = nx.multi_source_dijkstra_path_length(Gu, fnodes, weight="length")
            access[f"netdist_{kind}_m"] = [dist.get(n, np.nan) for n in node_ids]
            log.info("%s: reachable %.1f%%", kind,
                     100 * np.mean(np.isfinite(access[f'netdist_{kind}_m'])))
        acc = pd.DataFrame({"globalid": feats["globalid"], **access})
        acc.to_parquet(acc_fp, index=False)
    acc = pd.read_parquet(acc_fp)

    # ---------------- tract aggregation -------------------------------------------
    tracts = gpd.read_file(path("data", "raw", "census", "tracts_study.gpkg"),
                           layer="tracts").to_crs(crs)
    joined = gpd.sjoin(pts, tracts[["GEOID", "geometry"]], how="left",
                       predicate="within")
    joined = joined.merge(acc, on="globalid", how="left")

    agg = (joined[joined["residential"] == 1]
           .groupby("GEOID")
           .agg(n_inspected=("globalid", "count"),
                n_destroyed=("destroyed", "sum"),
                destroyed_rate=("destroyed", "mean"),
                med_dist_fire_m=("netdist_fire_station_m", "median"),
                med_dist_hosp_m=("netdist_hospital_m", "median"))
           .reset_index())

    svi = pd.read_csv(path("data", "raw", "census", "svi2022_ca.csv"), dtype={"FIPS": str})
    svi_cols = {
        "FIPS": "GEOID", "RPL_THEMES": "svi_overall",
        "RPL_THEME1": "svi_ses", "RPL_THEME2": "svi_household",
        "RPL_THEME3": "svi_minority", "RPL_THEME4": "svi_housing_transport",
        "EP_NOVEH": "pct_no_vehicle", "EP_AGE65": "pct_age65",
        "EP_POV150": "pct_pov150", "E_TOTPOP": "pop_total",
    }
    svi = svi[list(svi_cols)].rename(columns=svi_cols)
    svi = svi.replace(-999, np.nan)

    ctx = tracts[["GEOID", "geometry"]].merge(agg, on="GEOID", how="left")
    ctx = ctx.merge(svi, on="GEOID", how="left")
    ctx = ctx[ctx["n_inspected"].fillna(0) > 0]

    ctx.to_file(path("data", "processed", "tract_context.gpkg"),
                layer="tracts", driver="GPKG")
    ctx.drop(columns="geometry").to_csv(
        path("outputs", "tables", "tract_context.csv"), index=False)
    log.info("tract context: %d tracts with inspected structures", len(ctx))
    log.info("SVI overall range: %.2f-%.2f; destroyed rate range %.2f-%.2f",
             ctx["svi_overall"].min(), ctx["svi_overall"].max(),
             ctx["destroyed_rate"].min(), ctx["destroyed_rate"].max())

    # rank correlation: vulnerability vs destruction (descriptive)
    from scipy.stats import spearmanr
    ok = ctx.dropna(subset=["svi_overall", "destroyed_rate"])
    rho, p = spearmanr(ok["svi_overall"], ok["destroyed_rate"])
    log.info("Spearman SVI vs destroyed rate: rho=%.3f p=%.3f (n=%d)", rho, p, len(ok))

    log.info("community context complete")


if __name__ == "__main__":
    main()
