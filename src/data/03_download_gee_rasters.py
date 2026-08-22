"""Acquire remote-sensing rasters and climate time series via Google Earth Engine.

Products (all clipped to the study bounding box, EPSG:26911):
  - Sentinel-2 L2A cloud-masked median composites:
      prefire  (2024-10-01..2025-01-06): NDVI, NDMI, NBR
      baseline (Oct-Dec 2022-2024):      NDVI, NDMI
      postfire (2025-01-28..2025-02-28): NBR
  - Landsat 8/9 C2 L2 surface temperature (prefire median, deg C)
  - USGS 3DEP 10 m DEM
  - gridMET daily fire-weather context (2024-04-01..2025-02-01) as CSV
  - gridMET Oct-Dec precipitation totals 1980-2024 (antecedent context) as CSV
  - ERA5-Land hourly wind/temperature/dewpoint 2025-01-05..2025-01-16 as CSV

A scene-selection table (ID, date, cloud %) is stored for every S2/Landsat window.
All exports are cached: existing files are not re-downloaded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import ee
import geemap
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import file_info, get_logger, load_config, path, record_manifest, utcnow

log = get_logger("03_download_gee")
cfg = load_config()

EXPORT_CRS = cfg["crs"]["analysis"]
SCALE = cfg["gee"]["export_scale_m"]
CLOUD_THR = cfg["gee"]["cloud_prob_threshold"]


def study_region(pad_m: float = 600.0) -> ee.Geometry:
    """UTM-aligned study rectangle (plus pad).

    A WGS84 lon/lat bbox clipped into EPSG:26911 leaves a nodata frame around
    the UTM export grid (empty top/side margins in map figures). Exporting a
    plane rectangle in the analysis CRS fills that frame continuously.
    """
    gdf = gpd.read_file(path("data", "processed", "study_area.gpkg"), layer="study_area")
    gdf = gdf.to_crs(EXPORT_CRS)
    minx, miny, maxx, maxy = (float(v) for v in gdf.total_bounds)
    return ee.Geometry.Rectangle(
        [minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m],
        proj=EXPORT_CRS,
        geodesic=False,
    )


def s2_masked(region: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    """Sentinel-2 SR harmonized, cloud-masked with s2cloudless probability."""
    s2 = (
        ee.ImageCollection(cfg["gee"]["s2_collection"])
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
    )
    prob = (
        ee.ImageCollection(cfg["gee"]["s2_cloud_prob"])
        .filterBounds(region)
        .filterDate(start, end)
    )
    joined = ee.Join.saveFirst("cloud_prob").apply(
        primary=s2,
        secondary=prob,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def mask(img):
        img = ee.Image(img)
        cp = ee.Image(img.get("cloud_prob")).select("probability")
        scl = img.select("SCL")
        good = (
            cp.lt(CLOUD_THR)
            .And(scl.neq(3))   # cloud shadow
            .And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))  # cloud med/high/cirrus
            .And(scl.neq(11))  # snow (spurious over ocean glint)
        )
        return img.updateMask(good).divide(10000).select(["B4", "B8", "B11", "B12"])

    return ee.ImageCollection(joined).map(mask)


def s2_scene_table(region: ee.Geometry, start: str, end: str) -> pd.DataFrame:
    s2 = (
        ee.ImageCollection(cfg["gee"]["s2_collection"])
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
    )
    info = s2.reduceColumns(
        ee.Reducer.toList(3),
        ["system:index", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE"],
    ).get("list").getInfo()
    df = pd.DataFrame(info, columns=["scene_id", "time_start_ms", "cloud_pct"])
    if len(df):
        df["date_utc"] = pd.to_datetime(df["time_start_ms"], unit="ms")
    return df


def indices(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")
    nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
    return ee.Image.cat([ndvi, ndmi, nbr])


def export(img: ee.Image, name: str, region: ee.Geometry, scale: int = SCALE,
           force: bool = False) -> Path:
    out = path("data", "raw", "gee", f"{name}.tif")
    if out.exists() and not force:
        log.info("cached: %s", out.name)
        return out
    if out.exists() and force:
        out.unlink()
        log.info("re-exporting %s at %dm ...", name, scale)
    else:
        log.info("exporting %s at %dm ...", name, scale)
    geemap.ee_export_image(
        img.clip(region), filename=str(out), scale=scale, region=region,
        crs=EXPORT_CRS, file_per_band=False,
    )
    if not out.exists():
        raise RuntimeError(f"export failed: {name}")
    log.info("wrote %s (%.1f MB)", out.name, out.stat().st_size / 1e6)
    return out


def manifest_raster(fp: Path, dataset_id: str, name: str, provider: str,
                    landing: str, temporal: str, resolution: str, variables: str,
                    role: str) -> None:
    record_manifest({
        "dataset_id": dataset_id, "dataset_name": name, "provider": provider,
        "landing_page": landing, "access_url": "Google Earth Engine",
        "access_method": "GEE compositing + export (see src/data/03_download_gee_rasters.py)",
        "license": "Provider open-data terms (Copernicus / USGS / NASA as applicable)",
        "temporal_coverage": temporal, "spatial_coverage": "Palisades study area bbox",
        "resolution": resolution, "native_crs": f"exported {EXPORT_CRS}",
        "variables": variables, "analytical_role": role,
        "retrieved_utc": utcnow(),
        "local_path": str(fp.relative_to(fp.parents[3])),
        "version": "", "limitations": "Cloud-masked median composite; see scene table",
        "credentials_required": "GEE account (authenticated locally)",
        **file_info(fp),
    })


def main() -> None:
    ee.Initialize()
    region = study_region()
    t = cfg["temporal"]

    pre_start, pre_end = t["prefire_window"]
    post_start, post_end = t["postfire_window"]

    # ---- scene selection tables -------------------------------------------
    scenes = {
        "prefire": s2_scene_table(region, pre_start, pre_end),
        "postfire": s2_scene_table(region, post_start, post_end),
    }
    for yr in t["baseline_years"]:
        scenes[f"baseline_{yr}"] = s2_scene_table(region, f"{yr}-10-01", f"{yr}-12-31")
    tab = pd.concat(scenes, names=["window"]).reset_index(level=0)
    tab_fp = path("data", "metadata", "s2_scene_selection.csv")
    tab.to_csv(tab_fp, index=False)
    log.info("scene table: %d scenes -> %s", len(tab), tab_fp)

    # ---- Sentinel-2 composites --------------------------------------------
    pre = indices(s2_masked(region, pre_start, pre_end).median())
    fp = export(pre.select("NDVI"), "s2_prefire_ndvi", region)
    manifest_raster(fp, "D04a", "Sentinel-2 prefire NDVI composite", "Copernicus/ESA via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
                    f"{pre_start}..{pre_end}", "10 m", "NDVI (median composite)",
                    "Pre-fire vegetation amount predictor")
    fp = export(pre.select("NDMI"), "s2_prefire_ndmi", region)
    manifest_raster(fp, "D04b", "Sentinel-2 prefire NDMI composite", "Copernicus/ESA via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
                    f"{pre_start}..{pre_end}", "10 m", "NDMI (median composite)",
                    "Pre-fire vegetation moisture predictor")
    fp = export(pre.select("NBR"), "s2_prefire_nbr", region)
    manifest_raster(fp, "D04c", "Sentinel-2 prefire NBR composite", "Copernicus/ESA via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
                    f"{pre_start}..{pre_end}", "10 m", "NBR (median composite)",
                    "Burn-severity baseline (dNBR input; NOT a predictor)")

    # baseline composite across Oct-Dec of baseline years
    coll = None
    for yr in t["baseline_years"]:
        c = s2_masked(region, f"{yr}-10-01", f"{yr}-12-31")
        coll = c if coll is None else coll.merge(c)
    base = indices(coll.median())
    fp = export(base.select("NDVI"), "s2_baseline_ndvi", region)
    manifest_raster(fp, "D04d", "Sentinel-2 baseline NDVI (Oct-Dec 2022-2024)", "Copernicus/ESA via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
                    "Oct-Dec 2022-2024", "10 m", "NDVI (median composite)",
                    "Seasonal baseline for anomaly computation")
    fp = export(base.select("NDMI"), "s2_baseline_ndmi", region)
    manifest_raster(fp, "D04e", "Sentinel-2 baseline NDMI (Oct-Dec 2022-2024)", "Copernicus/ESA via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
                    "Oct-Dec 2022-2024", "10 m", "NDMI (median composite)",
                    "Seasonal baseline for anomaly computation")

    post = indices(s2_masked(region, post_start, post_end).median())
    fp = export(post.select("NBR"), "s2_postfire_nbr", region)
    manifest_raster(fp, "D04f", "Sentinel-2 postfire NBR composite", "Copernicus/ESA via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
                    f"{post_start}..{post_end}", "10 m", "NBR (median composite)",
                    "Burn severity (dNBR); impact indicator only")
    fp = export(post.select("NDVI"), "s2_postfire_ndvi", region)
    manifest_raster(fp, "D04g", "Sentinel-2 postfire NDVI composite", "Copernicus/ESA via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
                    f"{post_start}..{post_end}", "10 m", "NDVI (median composite)",
                    "Post-fire vegetation amount")
    fp = export(post.select("NDMI"), "s2_postfire_ndmi", region)
    manifest_raster(fp, "D04h", "Sentinel-2 postfire NDMI composite", "Copernicus/ESA via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
                    f"{post_start}..{post_end}", "10 m", "NDMI (median composite)",
                    "Post-fire vegetation moisture")

    # True-color RGB (uint8 visualize stretch; for Figure 03_03 context only)
    def s2_rgb(start: str, end: str) -> ee.Image:
        s2 = (
            ee.ImageCollection(cfg["gee"]["s2_collection"])
            .filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        )
        prob = (
            ee.ImageCollection(cfg["gee"]["s2_cloud_prob"])
            .filterBounds(region).filterDate(start, end)
        )
        joined = ee.Join.saveFirst("cloud_prob").apply(
            primary=s2, secondary=prob,
            condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
        )

        def mask(img):
            img = ee.Image(img)
            cp = ee.Image(img.get("cloud_prob")).select("probability")
            scl = img.select("SCL")
            good = (
                cp.lt(CLOUD_THR)
                .And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9))
                .And(scl.neq(10)).And(scl.neq(11))
            )
            return img.updateMask(good).divide(10000).select(["B4", "B3", "B2"])

        return (
            ee.ImageCollection(joined).map(mask).median()
            .rename(["R", "G", "B"])
            .visualize(bands=["R", "G", "B"], min=0.02, max=0.28, gamma=1.05)
        )

    fp = export(s2_rgb(pre_start, pre_end), "s2_prefire_rgb", region)
    manifest_raster(fp, "D04i", "Sentinel-2 prefire true-color RGB", "Copernicus/ESA via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
                    f"{pre_start}..{pre_end}", "10 m", "B4/B3/B2 uint8 stretch",
                    "True-color context figure (not a predictor)")
    dur_start, dur_end = t["duringfire_window"]
    fp = export(s2_rgb(dur_start, dur_end), "s2_duringfire_rgb", region)
    manifest_raster(fp, "D04k", "Sentinel-2 during-fire true-color RGB", "Copernicus/ESA via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
                    f"{dur_start}..{dur_end}", "10 m", "B4/B3/B2 uint8 stretch",
                    "True-color context figure (not a predictor)")
    fp = export(s2_rgb(post_start, post_end), "s2_postfire_rgb", region)
    manifest_raster(fp, "D04j", "Sentinel-2 postfire true-color RGB", "Copernicus/ESA via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
                    f"{post_start}..{post_end}", "10 m", "B4/B3/B2 uint8 stretch",
                    "True-color context figure (not a predictor)")

    # ---- Landsat LST prefire ----------------------------------------------
    def lst_masked(cid: str, start: str, end: str) -> ee.ImageCollection:
        def f(img):
            qa = img.select("QA_PIXEL")
            clear = (
                qa.bitwiseAnd(1 << 3).eq(0)   # cloud
                .And(qa.bitwiseAnd(1 << 4).eq(0))  # cloud shadow
                .And(qa.bitwiseAnd(1 << 1).eq(0))  # dilated cloud
            )
            lst_c = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
            return lst_c.updateMask(clear).rename("LST")
        return (
            ee.ImageCollection(cid).filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", 60)).map(f)
        )

    lsts = lst_masked(cfg["gee"]["landsat_l2"][0], pre_start, pre_end).merge(
        lst_masked(cfg["gee"]["landsat_l2"][1], pre_start, pre_end))
    n_lst = lsts.size().getInfo()
    log.info("Landsat LST scenes in prefire window: %d", n_lst)
    fp = export(lsts.median(), "landsat_prefire_lst", region, scale=30)
    manifest_raster(fp, "D05", "Landsat 8/9 C2 L2 prefire LST composite", "USGS via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2",
                    f"{pre_start}..{pre_end}", "30 m", "Surface temperature (deg C, median)",
                    "Pre-fire thermal condition predictor")

    lsts_post = lst_masked(cfg["gee"]["landsat_l2"][0], post_start, post_end).merge(
        lst_masked(cfg["gee"]["landsat_l2"][1], post_start, post_end))
    n_lst_post = lsts_post.size().getInfo()
    log.info("Landsat LST scenes in postfire window: %d", n_lst_post)
    fp = export(lsts_post.median(), "landsat_postfire_lst", region, scale=30)
    manifest_raster(fp, "D05b", "Landsat 8/9 C2 L2 postfire LST composite", "USGS via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2",
                    f"{post_start}..{post_end}", "30 m", "Surface temperature (deg C, median)",
                    "Post-fire thermal condition")

    # ---- DEM ---------------------------------------------------------------
    dem = ee.Image(cfg["gee"]["dem"]).select("elevation")
    fp = export(dem, "dem_3dep_10m", region, scale=cfg["gee"]["dem_scale_m"])
    manifest_raster(fp, "D16", "USGS 3DEP 10 m DEM", "USGS via GEE",
                    "https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_10m",
                    "static (3DEP seamless)", "10 m", "Elevation (m)",
                    "Terrain predictors (slope, aspect, TPI, TRI derived locally)")

    # ---- gridMET daily context ---------------------------------------------
    gm_fp = path("data", "raw", "climate", "gridmet_daily.csv")
    if not gm_fp.exists():
        a_start, a_end = cfg["temporal"]["antecedent_climate_window"]
        gm = (
            ee.ImageCollection(cfg["gee"]["gridmet"])
            .filterDate(a_start, a_end)
            .select(["pr", "tmmx", "tmmn", "rmax", "rmin", "vs", "th", "erc", "bi", "fm100", "fm1000", "vpd"])
        )

        def daily(img):
            d = img.reduceRegion(ee.Reducer.mean(), region, 4000, maxPixels=1e9)
            return ee.Feature(None, d).set("date", img.date().format("YYYY-MM-dd"))

        feats = gm.map(daily).getInfo()["features"]
        rows = [{**f["properties"]} for f in feats]
        pd.DataFrame(rows).to_csv(gm_fp, index=False)
        log.info("wrote %s (%d days)", gm_fp, len(rows))
    record_manifest({
        "dataset_id": "D13", "dataset_name": "gridMET daily fire-weather context",
        "provider": "Climatology Lab / Univ. Idaho via GEE",
        "landing_page": "https://www.climatologylab.org/gridmet.html",
        "access_url": "GEE IDAHO_EPSCOR/GRIDMET", "access_method": "GEE reduceRegion (area mean)",
        "license": "Open research data", "temporal_coverage": "2024-04-01..2025-01-31",
        "spatial_coverage": "study-area mean", "resolution": "~4 km daily",
        "native_crs": "EPSG:4326", "variables": "pr,tmmx,tmmn,rmax,rmin,vs,th,erc,bi,fm100,fm1000,vpd",
        "analytical_role": "Antecedent/event climate context (not structure-level predictor)",
        "retrieved_utc": utcnow(), "local_path": str(gm_fp.relative_to(gm_fp.parents[3])),
        "version": "", "limitations": "Background climate only", "credentials_required": "GEE",
        **file_info(gm_fp),
    })

    # ---- gridMET Oct-Dec precipitation history (1980-2024) -----------------
    hist_fp = path("data", "raw", "climate", "gridmet_octdec_precip_1980_2024.csv")
    if not hist_fp.exists():
        rows = []
        for yr in range(1980, 2025):
            total = (
                ee.ImageCollection(cfg["gee"]["gridmet"])
                .filterDate(f"{yr}-10-01", f"{yr}-12-31").select("pr").sum()
                .reduceRegion(ee.Reducer.mean(), region, 4000, maxPixels=1e9)
                .get("pr")
            )
            rows.append({"year": yr, "octdec_pr_mm": total.getInfo()})
        pd.DataFrame(rows).to_csv(hist_fp, index=False)
        log.info("wrote %s", hist_fp)

    # ---- ERA5-Land hourly event window --------------------------------------
    e5_fp = path("data", "raw", "climate", "era5land_hourly_event.csv")
    if not e5_fp.exists():
        e5 = (
            ee.ImageCollection(cfg["gee"]["era5_land"])
            .filterDate("2025-01-05", "2025-01-16")
            .select(["u_component_of_wind_10m", "v_component_of_wind_10m",
                     "temperature_2m", "dewpoint_temperature_2m"])
        )

        def hourly(img):
            d = img.reduceRegion(ee.Reducer.mean(), region, 11000, maxPixels=1e9)
            return ee.Feature(None, d).set("time", img.date().format("YYYY-MM-dd HH:mm"))

        feats = e5.map(hourly).getInfo()["features"]
        rows = [{**f["properties"]} for f in feats]
        pd.DataFrame(rows).to_csv(e5_fp, index=False)
        log.info("wrote %s (%d hours)", e5_fp, len(rows))
    record_manifest({
        "dataset_id": "D12s", "dataset_name": "ERA5-Land hourly event weather (HRRR substitute)",
        "provider": "ECMWF/Copernicus via GEE",
        "landing_page": "https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY",
        "access_url": "GEE ECMWF/ERA5_LAND/HOURLY", "access_method": "GEE reduceRegion (area mean)",
        "license": "Copernicus open data", "temporal_coverage": "2025-01-05..2025-01-15",
        "spatial_coverage": "study-area mean", "resolution": "~9 km hourly",
        "native_crs": "EPSG:4326", "variables": "u10,v10,t2m,d2m",
        "analytical_role": "Event weather chronology context (documented HRRR substitution; not a structure-level predictor)",
        "retrieved_utc": utcnow(), "local_path": str(e5_fp.relative_to(e5_fp.parents[3])),
        "version": "", "limitations": "Coarse reanalysis; underestimates local gusts",
        "credentials_required": "GEE", **file_info(e5_fp),
    })

    log.info("GEE acquisition complete")


if __name__ == "__main__":
    main()
