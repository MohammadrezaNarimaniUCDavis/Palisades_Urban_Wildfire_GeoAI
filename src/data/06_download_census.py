"""Download census geographies and CDC/ATSDR SVI 2022 for the community track.

Note (decision D-019): the Census Bureau API now requires an API key, so the
planned direct ACS block-group pull is blocked without credentials. The
CDC/ATSDR SVI 2022 (tract level) is itself constructed from ACS 2018-2022
5-year estimates and contains the indicators this study needs (population,
poverty, age 65+, no-vehicle households, plus theme percentile rankings),
including estimate (E_*) and MOE (M_*) fields. It is used as the social data
source; this substitution is recorded in the manifest and decision log.

Products:
  - TIGER/Line 2023 census tracts + block groups (LA County, study-area subset)
  - CDC/ATSDR SVI 2022 California tract CSV
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import file_info, get_logger, load_config, path, record_manifest, utcnow

log = get_logger("06_download_census")
cfg = load_config()

TIGER = {
    "blockgroups": "https://www2.census.gov/geo/tiger/TIGER2023/BG/tl_2023_06_bg.zip",
    "tracts": "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_06_tract.zip",
}
SVI_URLS = [
    "https://svi.cdc.gov/Documents/Data/2022/csv/states/California.csv",
    "https://svi.cdc.gov/Documents/Data/2022/csv/states/SVI_2022_California.csv",
    "https://svi.cdc.gov/Documents/Data/2022/csv/SVI_2022_US.csv",
]


def tiger_layer(name: str, url: str, study_wgs: gpd.GeoDataFrame) -> None:
    out = path("data", "raw", "census", f"{name}_study.gpkg")
    if out.exists():
        log.info("cached: %s", out.name)
    else:
        log.info("downloading TIGER %s ...", name)
        r = requests.get(url, timeout=900)
        r.raise_for_status()
        zdir = path("data", "raw", "census", f"tiger_{name}")
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(zdir)
        shp = next(zdir.glob("*.shp"))
        g = gpd.read_file(shp)
        g = g[g["COUNTYFP"] == cfg["census"]["county_fips"]]
        sel = g[g.to_crs(4326).intersects(study_wgs.geometry.iloc[0])].copy()
        sel.to_file(out, layer=name, driver="GPKG")
        log.info("%s intersecting study area: %d -> %s", name, len(sel), out)
    record_manifest({
        "dataset_id": "D21g", "dataset_name": f"TIGER/Line 2023 {name} (study area)",
        "provider": "U.S. Census Bureau",
        "landing_page": "https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html",
        "access_url": url, "access_method": "Direct download + spatial filter",
        "license": "U.S. public domain", "temporal_coverage": "2023 vintage",
        "spatial_coverage": f"LA County {name} intersecting study area",
        "resolution": name[:-1] if name.endswith("s") else name, "native_crs": "EPSG:4269",
        "variables": "GEOID, geometry", "analytical_role": "Community-unit geometry (Track B)",
        "retrieved_utc": utcnow(), "local_path": f"data/raw/census/{name}_study.gpkg",
        "version": "2023", "limitations": "", "credentials_required": "No",
        **file_info(out),
    })


def main() -> None:
    study = gpd.read_file(path("data", "processed", "study_area.gpkg"), layer="study_area")
    study_wgs = study.to_crs(4326)

    for name, url in TIGER.items():
        tiger_layer(name, url, study_wgs)

    svi_fp = path("data", "raw", "census", "svi2022_ca.csv")
    u_used = "(cached)"
    if not svi_fp.exists():
        got = None
        for u in SVI_URLS:
            try:
                log.info("trying SVI url: %s", u)
                r = requests.get(u, timeout=900)
                if r.status_code == 200 and len(r.content) > 1_000_000:
                    got = (u, r.content)
                    break
                log.warning("HTTP %d / %d bytes from %s", r.status_code, len(r.content), u)
            except Exception as e:  # noqa: BLE001
                log.warning("SVI fetch failed %s: %s", u, e)
        if not got:
            raise RuntimeError("Could not download CDC SVI 2022 from candidate URLs")
        svi_fp.write_bytes(got[1])
        u_used = got[0]
        log.info("SVI downloaded from %s (%.1f MB)", u_used, len(got[1]) / 1e6)
    record_manifest({
        "dataset_id": "D22", "dataset_name": "CDC/ATSDR SVI 2022 (California tracts)",
        "provider": "CDC/ATSDR",
        "landing_page": "https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html",
        "access_url": u_used, "access_method": "Direct CSV download",
        "license": "Open U.S. government data",
        "temporal_coverage": "SVI 2022 release (ACS 2018-2022 inputs)",
        "spatial_coverage": "California (filtered to study tracts in processing)",
        "resolution": "census tract", "native_crs": "n/a (tabular; GEOID join)",
        "variables": "RPL_THEMES, theme percentiles, E_NOVEH, E_AGE65, EP_POV150, components + MOEs",
        "analytical_role": "Social vulnerability & capacity context (Track B); ACS substitute (D-019)",
        "retrieved_utc": utcnow(), "local_path": "data/raw/census/svi2022_ca.csv",
        "version": "2022", "limitations": "Relative percentile ranking; tract scale",
        "credentials_required": "No", **file_info(svi_fp),
    })
    log.info("census acquisition complete")


if __name__ == "__main__":
    main()
