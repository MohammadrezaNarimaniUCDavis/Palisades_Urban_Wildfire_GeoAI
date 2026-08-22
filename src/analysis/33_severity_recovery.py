"""Burn severity characterization and early vegetation recovery (Track B / impact).

1. dNBR distribution by DINS damage class (descriptive; impact indicators).
2. Burned-area fractions by severity class within the perimeter.
3. Monthly NDVI recovery trajectories (Feb 2025 - Jul 2026) by burn-severity
   class, computed in GEE with the same compositing rules as acquisition, and
   normalized against the month-matched 2022-2024 climatology per class
   (seasonally matched recovery, decision D-016).

Outputs (outputs/tables + data/processed):
  - dnbr_by_damage_class.csv
  - severity_area_fractions.csv
  - recovery_trajectories.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, load_config, path

log = get_logger("33_severity_recovery")
cfg = load_config()


def s2_masked(region, start: str, end: str) -> ee.ImageCollection:
    s2 = (ee.ImageCollection(cfg["gee"]["s2_collection"])
          .filterBounds(region).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))
    prob = (ee.ImageCollection(cfg["gee"]["s2_cloud_prob"])
            .filterBounds(region).filterDate(start, end))
    joined = ee.Join.saveFirst("cloud_prob").apply(
        primary=s2, secondary=prob,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"))

    def mask(img):
        img = ee.Image(img)
        cp = ee.Image(img.get("cloud_prob")).select("probability")
        scl = img.select("SCL")
        good = (cp.lt(cfg["gee"]["cloud_prob_threshold"])
                .And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9))
                .And(scl.neq(10)).And(scl.neq(11)))
        return img.updateMask(good).divide(10000).select(["B4", "B8", "B11", "B12"])

    return ee.ImageCollection(joined).map(mask)


def main() -> None:
    # ---------- 1. dNBR by damage class (local) --------------------------------
    feats = pd.read_parquet(path("data", "processed", "structure_features.parquet"))
    res = feats[feats["residential"] == 1]
    tab = (res.groupby("damage")["dnbr_r30_100"]
           .agg(["count", "mean", "median", "std"]).round(1))
    tab.to_csv(path("outputs", "tables", "dnbr_by_damage_class.csv"))
    log.info("dNBR (30-100 m ring) by damage class:\n%s", tab.to_string())

    # ---------- 2. severity area fractions within perimeter ---------------------
    per = gpd.read_file(path("data", "processed", "study_area.gpkg"),
                        layer="perimeter").to_crs(cfg["crs"]["analysis"])
    with rasterio.open(path("data", "interim", "severity", "severity_class.tif")) as src:
        from rasterio.mask import mask as rmask
        arr, _ = rmask(src, per.geometry, crop=True, nodata=np.nan)
    v = arr[0]
    v = v[np.isfinite(v)]
    labels = cfg["burn_severity"]["severity_labels"]
    fr = pd.DataFrame({
        "severity": labels,
        "fraction": [float((v == i).mean()) for i in range(len(labels))],
        "area_km2": [float((v == i).sum()) * 100 / 1e6 for i in range(len(labels))],
    })
    fr.to_csv(path("outputs", "tables", "severity_area_fractions.csv"), index=False)
    log.info("severity fractions within perimeter:\n%s", fr.to_string(index=False))

    # ---------- 3. recovery trajectories (GEE) ------------------------------------
    out_fp = path("outputs", "tables", "recovery_trajectories.csv")
    if out_fp.exists():
        log.info("recovery trajectories cached")
        return
    ee.Initialize()
    gdf = gpd.read_file(path("data", "processed", "study_area.gpkg"), layer="perimeter")
    geom = ee.Geometry(gdf.to_crs(4326).geometry.iloc[0].__geo_interface__)

    pre_start, pre_end = cfg["temporal"]["prefire_window"]
    post_start, post_end = cfg["temporal"]["postfire_window"]
    nbr_pre = s2_masked(geom, pre_start, pre_end).median().normalizedDifference(["B8", "B12"])
    nbr_post = s2_masked(geom, post_start, post_end).median().normalizedDifference(["B8", "B12"])
    dnbr = nbr_pre.subtract(nbr_post).multiply(1000)
    breaks = cfg["burn_severity"]["severity_breaks"]
    sev = (dnbr.gt(breaks[1]).add(dnbr.gt(breaks[2])).add(dnbr.gt(breaks[3]))
           .rename("severity").clip(geom))

    def monthly_ndvi_by_class(start: str, end: str, label: str) -> list[dict]:
        coll = s2_masked(geom, start, end)
        n = coll.size().getInfo()
        if n == 0:
            return []
        ndvi = coll.median().normalizedDifference(["B8", "B4"]).rename("ndvi")
        combo = ndvi.addBands(sev)
        stats = combo.reduceRegion(
            reducer=ee.Reducer.mean().group(groupField=1, groupName="severity"),
            geometry=geom, scale=20, maxPixels=1e9).getInfo()
        rows = []
        for g in stats.get("groups", []):
            rows.append({"period": label, "severity": int(g["severity"]),
                         "ndvi": g["mean"], "n_scenes": n})
        return rows

    rows: list[dict] = []
    months = pd.period_range(cfg["temporal"]["recovery_start"],
                             cfg["temporal"]["recovery_end"], freq="M")
    for p in months:
        start = f"{p.year}-{p.month:02d}-01"
        end = str((p + 1).to_timestamp().date())
        rows += monthly_ndvi_by_class(start, end, f"{p.year}-{p.month:02d}")
        log.info("recovery month %s done", p)

    # month-matched climatology 2022-2024 per class
    for m in range(1, 13):
        coll = None
        for yr in (2022, 2023, 2024):
            c = s2_masked(geom, f"{yr}-{m:02d}-01",
                          str((pd.Period(f"{yr}-{m:02d}") + 1).to_timestamp().date()))
            coll = c if coll is None else coll.merge(c)
        ndvi = coll.median().normalizedDifference(["B8", "B4"]).rename("ndvi")
        stats = ndvi.addBands(sev).reduceRegion(
            reducer=ee.Reducer.mean().group(groupField=1, groupName="severity"),
            geometry=geom, scale=20, maxPixels=1e9).getInfo()
        for g in stats.get("groups", []):
            rows.append({"period": f"clim-{m:02d}", "severity": int(g["severity"]),
                         "ndvi": g["mean"], "n_scenes": -1})
        log.info("climatology month %02d done", m)

    pd.DataFrame(rows).to_csv(out_fp, index=False)
    log.info("wrote %s", out_fp)


if __name__ == "__main__":
    main()
