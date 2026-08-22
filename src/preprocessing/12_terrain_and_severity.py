"""Derive terrain variables from the 3DEP DEM and burn severity from NBR composites.

Terrain (10 m, EPSG:26911): slope (deg), northness, eastness, TPI-300 m, TRI.
Severity: dNBR = (NBR_pre - NBR_post) * 1000 (scaled), RdNBR, and a classified
severity raster using standard thresholds (config: burn_severity.severity_breaks).

Outputs -> data/interim/terrain/*.tif and data/interim/severity/*.tif
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, load_config, path

log = get_logger("12_terrain_severity")


def write_like(src_profile: dict, fp: Path, arr: np.ndarray, nodata=np.nan) -> None:
    prof = dict(src_profile)
    prof.update(dtype="float32", count=1, nodata=nodata, compress="deflate")
    with rasterio.open(fp, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)
    log.info("wrote %s", fp.name)


def circular_kernel(radius_px: int) -> np.ndarray:
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return (x * x + y * y <= radius_px * radius_px).astype("float32")


def main() -> None:
    cfg = load_config()

    # ---------------- terrain -------------------------------------------------
    dem_fp = path("data", "raw", "gee", "dem_3dep_10m.tif")
    with rasterio.open(dem_fp) as src:
        dem = src.read(1).astype("float64")
        prof = src.profile
        resx = src.transform.a
    dem[dem < -1000] = np.nan

    dy, dx = np.gradient(dem, resx)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    aspect = np.arctan2(-dx, dy)          # radians, 0 = north
    northness = np.cos(aspect)
    eastness = np.sin(aspect)

    # TPI at 300 m: elevation minus focal mean in 300 m disk
    k = circular_kernel(int(300 / resx))
    k /= k.sum()
    demf = np.nan_to_num(dem, nan=0.0)
    valid = np.isfinite(dem).astype("float64")
    num = ndimage.convolve(demf, k, mode="nearest")
    den = ndimage.convolve(valid, k, mode="nearest")
    focal_mean = np.where(den > 0, num / den, np.nan)
    tpi300 = dem - focal_mean

    # TRI: mean absolute difference to 8 neighbors
    tri = np.zeros_like(dem)
    cnt = np.zeros_like(dem)
    for sy in (-1, 0, 1):
        for sx in (-1, 0, 1):
            if sy == 0 and sx == 0:
                continue
            shifted = np.roll(np.roll(dem, sy, axis=0), sx, axis=1)
            d = np.abs(dem - shifted)
            m = np.isfinite(d)
            tri[m] += d[m]
            cnt[m] += 1
    tri = np.where(cnt > 0, tri / np.maximum(cnt, 1), np.nan)

    tdir = path("data", "interim", "terrain", "x").parent
    tdir.mkdir(parents=True, exist_ok=True)
    write_like(prof, tdir / "elevation.tif", np.where(np.isfinite(dem), dem, np.nan))
    write_like(prof, tdir / "slope_deg.tif", slope)
    write_like(prof, tdir / "northness.tif", northness)
    write_like(prof, tdir / "eastness.tif", eastness)
    write_like(prof, tdir / "tpi300.tif", tpi300)
    write_like(prof, tdir / "tri.tif", tri)

    # ---------------- burn severity ---------------------------------------------
    with rasterio.open(path("data", "raw", "gee", "s2_prefire_nbr.tif")) as s:
        nbr_pre = s.read(1).astype("float64")
        sev_prof = s.profile
    with rasterio.open(path("data", "raw", "gee", "s2_postfire_nbr.tif")) as s:
        nbr_post = s.read(1).astype("float64")

    dnbr = (nbr_pre - nbr_post) * cfg["burn_severity"]["dnbr_scale"]
    # RdNBR (Miller & Thode 2007): dNBR / sqrt(|NBR_pre|), NBR_pre in fractional units
    denom = np.sqrt(np.clip(np.abs(nbr_pre), 0.001, None))
    rdnbr = dnbr / denom

    breaks = cfg["burn_severity"]["severity_breaks"]
    sev = np.digitize(dnbr, breaks[1:-1])  # 0..3
    sev = sev.astype("float64")
    sev[~np.isfinite(dnbr)] = np.nan

    sdir = path("data", "interim", "severity", "x").parent
    sdir.mkdir(parents=True, exist_ok=True)
    write_like(sev_prof, sdir / "dnbr.tif", dnbr)
    write_like(sev_prof, sdir / "rdnbr.tif", rdnbr)
    write_like(sev_prof, sdir / "severity_class.tif", sev)

    log.info("terrain + severity derivation complete")


if __name__ == "__main__":
    main()
