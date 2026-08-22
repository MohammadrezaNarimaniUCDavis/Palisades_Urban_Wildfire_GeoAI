"""Extract structure-level predictors for every DINS-inspected structure.

Raster predictors use exact ring statistics (0-30, 30-100, 100-300 m) computed
by convolving value/valid grids with circular kernels and differencing disk
sums. Vector predictors use STRtree spatial indexing.

Variable groups (temporal provenance enforced downstream):
  PRE-FIRE:  NDVI/NDMI (immediate pre-fire + seasonal anomaly), LST, canopy
             cover, fuel-class fractions, distance to wildland fuel, terrain,
             building arrangement, road context, YEARBUILT, assessed value.
  IMPACT:    dNBR ring means (descriptive only; excluded from predictor list).

Output: data/processed/structure_features.parquet (+ .gpkg for mapping)
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, load_config, path

log = get_logger("20_extract_predictors")
cfg = load_config()
RINGS = [tuple(r) for r in cfg["features"]["rings_m"]]
RADII = sorted({r[1] for r in RINGS} | {r[0] for r in RINGS if r[0] > 0})


def circular_kernel(radius_px: float) -> np.ndarray:
    r = int(np.floor(radius_px))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return ((x * x + y * y) <= radius_px * radius_px).astype("float32")


class RingSampler:
    """Compute disk sums for a raster once, then sample ring means at points."""

    def __init__(self, fp: Path, radii_m: list[int]):
        with rasterio.open(fp) as src:
            arr = src.read(1).astype("float64")
            self.transform = src.transform
            self.nodata = src.nodata
            self.res = src.transform.a
        if self.nodata is not None:
            arr[arr == self.nodata] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        self.valid = np.isfinite(arr).astype("float64")
        vals = np.nan_to_num(arr, nan=0.0)
        self.disk_sum: dict[int, np.ndarray] = {}
        self.disk_n: dict[int, np.ndarray] = {}
        for rm in radii_m:
            k = circular_kernel(rm / self.res)
            self.disk_sum[rm] = ndimage.convolve(vals, k, mode="nearest")
            self.disk_n[rm] = ndimage.convolve(self.valid, k, mode="nearest")
        self.point_arr = arr

    def _rc(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols, rows = (~self.transform) * (xs, ys)
        return (np.clip(np.floor(rows).astype(int), 0, self.point_arr.shape[0] - 1),
                np.clip(np.floor(cols).astype(int), 0, self.point_arr.shape[1] - 1))

    def ring_mean(self, xs, ys, r_in: int, r_out: int) -> np.ndarray:
        rows, cols = self._rc(xs, ys)
        s_out = self.disk_sum[r_out][rows, cols]
        n_out = self.disk_n[r_out][rows, cols]
        if r_in > 0:
            s_in = self.disk_sum[r_in][rows, cols]
            n_in = self.disk_n[r_in][rows, cols]
        else:
            s_in = np.zeros_like(s_out)
            n_in = np.zeros_like(n_out)
        n = n_out - n_in
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(n > 0.5, (s_out - s_in) / np.maximum(n, 1e-9), np.nan)
        return out

    def point_value(self, xs, ys) -> np.ndarray:
        rows, cols = self._rc(xs, ys)
        return self.point_arr[rows, cols]


def add_ring_features(df: pd.DataFrame, name: str, fp: Path, xs, ys,
                      point_too: bool = False) -> None:
    rs = RingSampler(fp, RADII)
    for (r0, r1) in RINGS:
        df[f"{name}_r{r0}_{r1}"] = rs.ring_mean(xs, ys, r0, r1)
    if point_too:
        df[f"{name}_pt"] = rs.point_value(xs, ys)


def fuel_group_masks(fbfm_fp: Path) -> dict[str, tuple[Path, rasterio.Affine]]:
    """Write binary group masks for FBFM40 classes; return file paths."""
    groups = {
        "fuel_wildland": list(range(101, 110)) + list(range(121, 125))
                          + list(range(141, 150)) + list(range(161, 166))
                          + list(range(181, 190)) + list(range(201, 205)),
        "fuel_shrub": list(range(141, 150)),
        "fuel_grass": list(range(101, 110)) + list(range(121, 125)),
        "fuel_timber": list(range(161, 166)) + list(range(181, 190)),
        "fuel_urban_nb": [91],
    }
    with rasterio.open(fbfm_fp) as src:
        fb = src.read(1)
        prof = src.profile
    outdir = path("data", "interim", "fuels", "x").parent
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, codes in groups.items():
        mask = np.isin(fb, codes).astype("float32")
        # nodata region (outside CONUS grid) -> treat as valid 0 (ocean)
        fp = outdir / f"{name}.tif"
        p = dict(prof)
        p.update(dtype="float32", count=1, nodata=None, compress="deflate")
        with rasterio.open(fp, "w", **p) as dst:
            dst.write(mask, 1)
        outputs[name] = fp
    return outputs


def distance_to_class(mask_fp: Path, out_fp: Path) -> Path:
    """Distance (m) to nearest pixel of a binary class raster."""
    with rasterio.open(mask_fp) as src:
        m = src.read(1)
        prof = src.profile
        res = src.transform.a
    dist_px = ndimage.distance_transform_edt(m < 0.5)
    dist = (dist_px * res).astype("float32")
    p = dict(prof)
    p.update(dtype="float32", count=1, nodata=None, compress="deflate")
    with rasterio.open(out_fp, "w", **p) as dst:
        dst.write(dist, 1)
    return out_fp


def main() -> None:
    crs = cfg["crs"]["analysis"]
    dins = gpd.read_file(path("data", "interim", "dins_clean.gpkg"), layer="dins").to_crs(crs)
    xs = dins.geometry.x.values
    ys = dins.geometry.y.values
    df = pd.DataFrame({
        "globalid": dins["GLOBALID"].values,
        "x": xs, "y": ys,
        "damage": dins["DAMAGE"].values,
        "damage_ord": dins["damage_ord"].values,
        "destroyed": dins["destroyed"].values,
        "residential": dins["residential"].values,
        "structure_category": dins["STRUCTURECATEGORY"].values,
        "yearbuilt": pd.to_numeric(dins["YEARBUILT"], errors="coerce").values,
        "assessed_value": pd.to_numeric(dins["ASSESSEDIMPROVEDVALUE"], errors="coerce").values,
        "roof_construction": dins["ROOFCONSTRUCTION"].values,
        "eaves": dins["EAVES"].values,
        "vent_screen": dins["VENTSCREEN"].values,
        "exterior_siding": dins["EXTERIORSIDING"].values,
        "window_pane": dins["WINDOWPANE"].values,
    })
    df.loc[df["yearbuilt"] < 1800, "yearbuilt"] = np.nan
    df.loc[df["assessed_value"] <= 0, "assessed_value"] = np.nan

    # ---- vegetation rasters (pre-fire + baseline anomaly) --------------------
    gee = path("data", "raw", "gee", "x").parent
    log.info("ring stats: NDVI/NDMI pre-fire ...")
    add_ring_features(df, "ndvi_pre", gee / "s2_prefire_ndvi.tif", xs, ys)
    add_ring_features(df, "ndmi_pre", gee / "s2_prefire_ndmi.tif", xs, ys)
    log.info("ring stats: NDVI/NDMI baseline ...")
    add_ring_features(df, "ndvi_base", gee / "s2_baseline_ndvi.tif", xs, ys)
    add_ring_features(df, "ndmi_base", gee / "s2_baseline_ndmi.tif", xs, ys)
    for (r0, r1) in RINGS:
        df[f"ndvi_anom_r{r0}_{r1}"] = df[f"ndvi_pre_r{r0}_{r1}"] - df[f"ndvi_base_r{r0}_{r1}"]
        df[f"ndmi_anom_r{r0}_{r1}"] = df[f"ndmi_pre_r{r0}_{r1}"] - df[f"ndmi_base_r{r0}_{r1}"]

    log.info("ring stats: LST ...")
    add_ring_features(df, "lst_pre", gee / "landsat_prefire_lst.tif", xs, ys)

    # ---- impact (descriptive only) -------------------------------------------
    log.info("ring stats: dNBR (impact indicator, not predictor) ...")
    add_ring_features(df, "dnbr", path("data", "interim", "severity", "dnbr.tif"), xs, ys)

    # ---- fuels ---------------------------------------------------------------
    log.info("fuel group fractions ...")
    masks = fuel_group_masks(path("data", "raw", "landfire", "lf2024_fbfm40.tif"))
    for name, fp in masks.items():
        add_ring_features(df, name, fp, xs, ys)
    add_ring_features(df, "canopy_cover", path("data", "raw", "landfire", "lf2024_cc.tif"), xs, ys)

    log.info("distance to wildland fuel ...")
    dist_fp = distance_to_class(masks["fuel_wildland"],
                                path("data", "interim", "fuels", "dist_wildland.tif"))
    rs = RingSampler(dist_fp, [30])
    df["dist_wildland_m"] = rs.point_value(xs, ys)

    # ---- terrain ---------------------------------------------------------------
    log.info("terrain sampling ...")
    tdir = path("data", "interim", "terrain", "x").parent
    for name in ["elevation", "slope_deg", "northness", "eastness", "tpi300", "tri"]:
        rs = RingSampler(tdir / f"{name}.tif", [100])
        df[f"{name}_pt"] = rs.point_value(xs, ys)
        if name == "slope_deg":
            df["slope_r0_100"] = rs.ring_mean(xs, ys, 0, 100)

    # ---- building arrangement (pre-fire OSM) ------------------------------------
    log.info("building arrangement ...")
    bld = gpd.read_file(path("data", "raw", "osm", "osm_buildings_prefire.gpkg"),
                        layer="buildings").to_crs(crs)
    bld = bld[bld.geometry.area > 10]  # drop slivers <10 m2
    geoms = list(bld.geometry.values)
    tree = STRtree(geoms)
    pts = list(dins.geometry.values)

    # own footprint: containing or nearest polygon within 20 m
    own_idx = np.full(len(pts), -1)
    nearest = tree.nearest(pts)
    for i, (pt, j) in enumerate(zip(pts, nearest)):
        if geoms[j].distance(pt) <= 20:
            own_idx[i] = j
    df["footprint_area_m2"] = [geoms[j].area if j >= 0 else np.nan for j in own_idx]
    df["osm_footprint_matched"] = (own_idx >= 0).astype(int)

    # nearest NEIGHBOR building (edge-to-edge from own footprint or point)
    log.info("nearest-neighbor building distance ...")
    nn_dist = np.full(len(pts), np.nan)
    for i, pt in enumerate(pts):
        cand = tree.query(pt.buffer(500))
        best = np.inf
        for j in cand:
            if j == own_idx[i]:
                continue
            src_geom = geoms[own_idx[i]] if own_idx[i] >= 0 else pt
            d = src_geom.distance(geoms[j])
            if d < best:
                best = d
        nn_dist[i] = best if np.isfinite(best) else np.nan
    df["nn_building_dist_m"] = nn_dist

    # building counts within radii (centroids)
    cent = gpd.GeoDataFrame(geometry=bld.geometry.centroid, crs=crs)
    cent_tree = STRtree(list(cent.geometry.values))
    for r in cfg["features"]["density_radii_m"]:
        counts = [len(cent_tree.query(pt.buffer(r))) for pt in pts]
        df[f"bld_count_r{r}"] = counts

    # DINS-to-DINS nearest inspected-structure distance
    from scipy.spatial import cKDTree
    kd = cKDTree(np.c_[xs, ys])
    d2, _ = kd.query(np.c_[xs, ys], k=2)
    df["nn_dins_dist_m"] = d2[:, 1]

    # ---- roads --------------------------------------------------------------------
    log.info("road context ...")
    roads = gpd.read_file(path("data", "raw", "osm", "osm_roads_prefire.gpkg"),
                          layer="edges").to_crs(crs)
    rtree = STRtree(list(roads.geometry.values))
    rgeoms = list(roads.geometry.values)
    df["dist_road_m"] = [
        min((rgeoms[j].distance(pt) for j in rtree.query(pt.buffer(1000))), default=np.nan)
        for pt in pts
    ]
    dens = []
    for pt in pts:
        buf = pt.buffer(300)
        total = sum(rgeoms[j].intersection(buf).length for j in rtree.query(buf))
        dens.append(total / (np.pi * 300**2 / 1e6))   # m per km2
    df["road_len_r300_m_per_km2"] = dens

    # dead-end proximity
    nodes = gpd.read_file(path("data", "raw", "osm", "osm_roads_prefire.gpkg"),
                          layer="nodes").to_crs(crs)
    de = nodes[nodes["street_count"] == 1]
    if len(de):
        kd_de = cKDTree(np.c_[de.geometry.x, de.geometry.y])
        df["dist_deadend_m"] = kd_de.query(np.c_[xs, ys], k=1)[0]

    # ---- save -------------------------------------------------------------------
    out = path("data", "processed", "structure_features.parquet")
    df.to_parquet(out, index=False)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs=crs)
    gdf.to_file(path("data", "processed", "structure_features.gpkg"),
                layer="structures", driver="GPKG")
    log.info("wrote %s (%d rows x %d cols)", out, len(df), df.shape[1])

    na_rate = df.isna().mean().sort_values(ascending=False)
    log.info("top missingness:\n%s", na_rate.head(12).to_string())


if __name__ == "__main__":
    main()
