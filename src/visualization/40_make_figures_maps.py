"""Publication figures 1-5 + S1: study area, event context, pre-fire environment,
built environment, predictor distributions, workflow diagram.

All figures use the shared journal style (config/figure_style.yaml).
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, load_config, path
from visualization.style import (
    COLORS, DAMAGE_COLORS, DAMAGE_SHORT, PALETTES, W_DOUBLE, CM, FS,
    add_cbar, add_north_arrow, add_scalebar, apply_style, clip_map_layers,
    cover_frame_overflow, full_frame, hillshade_background, map_axes, map_frame,
    ocean_overlay, panel_label, save_figure, set_extent,
)

log = get_logger("40_figures_maps")
cfg = load_config()
CRS = cfg["crs"]["analysis"]


def load_layers():
    study = gpd.read_file(path("data", "processed", "study_area.gpkg"), layer="study_area")
    per = gpd.read_file(path("data", "processed", "study_area.gpkg"), layer="perimeter")
    dins = gpd.read_file(path("data", "interim", "dins_clean.gpkg"), layer="dins").to_crs(CRS)
    roads = gpd.read_file(path("data", "raw", "osm", "osm_roads_prefire.gpkg"),
                          layer="edges").to_crs(CRS)
    return study, per, dins, roads


def get_states() -> gpd.GeoDataFrame:
    fp = path("data", "external", "tl_2023_us_state")
    shp = list(Path(fp).glob("*.shp"))
    if not shp:
        url = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(fp)
        shp = list(Path(fp).glob("*.shp"))
    return gpd.read_file(shp[0])


def raster_map(ax, fp: Path, cmap: str, vmin, vmax, per, study, label: str,
               ylabels: bool = True, xlabels: bool = True):
    with rasterio.open(fp) as src:
        arr = src.read(1).astype("float64")
        b = src.bounds
        nod = src.nodata
    if nod is not None:
        arr[arr == nod] = np.nan
    arr[arr < -1e5] = np.nan
    # Landsat LST exports can include 0 fill outside valid pixels
    if "lst" in Path(fp).name.lower():
        arr[(arr <= 0) | (arr > 70)] = np.nan
    ext = [b.left, b.right, b.bottom, b.top]
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=ext,
                   interpolation="bilinear", zorder=1, aspect="auto")
    ocean_overlay(ax)
    per.plot(ax=ax, facecolor="none", edgecolor=COLORS["perimeter"],
             linewidth=0.9, zorder=5)
    set_extent(ax, study, pad=400)
    ax.set_label("map")
    add_cbar(im, ax, label)
    map_frame(ax, crs=CRS, ylabels=ylabels, xlabels=xlabels, gis_frame=True)
    add_scalebar(ax, 4.0)
    return im


def rgb_map(ax, fp: Path, per, study, ylabels: bool = True, xlabels: bool = True):
    """True-color RGB panel matching raster_map framing (no colorbar)."""
    with rasterio.open(fp) as src:
        if src.count < 3:
            raise ValueError(f"expected 3-band RGB GeoTIFF, got {src.count}: {fp}")
        rgb = np.dstack([src.read(i) for i in (1, 2, 3)]).astype("float64")
        b = src.bounds
        nod = src.nodata
    if nod is not None:
        rgb[np.any(rgb == nod, axis=2)] = np.nan
    # uint8 stretch from GEE visualize(), or reflectance if ever float
    finite = np.isfinite(rgb)
    if finite.any() and np.nanmax(rgb) > 1.5:
        rgb = np.clip(rgb / 255.0, 0, 1)
    else:
        rgb = np.clip(rgb, 0, 1)
    # Transparent nodata so ocean overlay / frame stay clean
    alpha = np.where(np.all(np.isfinite(rgb), axis=2), 1.0, 0.0)
    rgba = np.dstack([np.nan_to_num(rgb, nan=0.0), alpha])
    ext = [b.left, b.right, b.bottom, b.top]
    ax.imshow(rgba, extent=ext, interpolation="bilinear", zorder=1, aspect="auto")
    ocean_overlay(ax)
    per.plot(ax=ax, facecolor="none", edgecolor=COLORS["perimeter"],
             linewidth=0.9, zorder=5)
    set_extent(ax, study, pad=400)
    ax.set_label("map")
    map_frame(ax, crs=CRS, ylabels=ylabels, xlabels=xlabels, gis_frame=True)
    add_scalebar(ax, 4.0)


# ----------------------------------------------------------------------------- Figure 1
DAMAGE_PLOT_ORDER = list(DAMAGE_COLORS.keys())


def _add_ca_inset(ax, per) -> None:
    """California locator inset inside the map frame (top-left)."""
    axin = ax.inset_axes([0.025, 0.63, 0.22, 0.34])
    axin.set_label("inset")
    axin.set_facecolor("white")
    axin.patch.set_alpha(0.96)

    states = get_states().to_crs("EPSG:3310")
    ca = states[states["STUSPS"] == "CA"]
    ca.plot(ax=axin, facecolor="#F4F4F4", edgecolor="#555555", linewidth=0.55, zorder=1)
    bx = ca.total_bounds
    pad_x, pad_top, pad_bot = 90000, 140000, 110000
    axin.set_xlim(bx[0] - pad_x, bx[2] + pad_x)
    axin.set_ylim(bx[1] - pad_x - pad_bot, bx[3] + pad_x + pad_top)
    axin.set_aspect("equal")

    cen = per.to_crs("EPSG:3310").centroid.iloc[0]
    axin.plot(cen.x, cen.y, marker="*", markersize=8, color=COLORS["destroyed"],
              markeredgecolor="black", markeredgewidth=0.3, zorder=3)

    axin.text(0.5, 0.94, "California", transform=axin.transAxes,
              ha="center", va="top", fontsize=FS["size_small"], fontweight="bold",
              color=COLORS["neutral_dark"], zorder=4)
    axin.text(0.08, 0.06, "Los Angeles", transform=axin.transAxes,
              ha="left", va="bottom", fontsize=FS["size_small"],
              color=COLORS["neutral_dark"], zorder=4)

    for s in axin.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color(COLORS["neutral_dark"])
    axin.set_xticks([])
    axin.set_yticks([])


def figure_01_study_area():
    study, per, dins, roads = load_layers()
    res = dins[dins["residential"] == 1]

    fig = plt.figure(figsize=(W_DOUBLE, 13.0 * CM))
    minx, miny, maxx, maxy = study.total_bounds
    ax = map_axes(fig, (maxx - minx + 800) / (maxy - miny + 800),
                  bottom=0.21, max_height=0.74)
    ax.set_label("map")
    set_extent(ax, study, pad=550, inset_m=25)
    ax.set_autoscale_on(False)
    # Match terrain so any sub-pixel frame seam is not bright white
    ax.set_facecolor("#D0D0D0")
    hillshade_background(ax)
    major = roads[roads["highway"].str.contains("motorway|trunk|primary|secondary", na=False)]
    roads.plot(ax=ax, color=COLORS["road"], linewidth=0.22, zorder=2, alpha=0.55)
    major.plot(ax=ax, color=COLORS["neutral_mid"], linewidth=0.65, zorder=3)

    study.plot(ax=ax, facecolor="none", edgecolor=COLORS["buffer"],
               linewidth=0.9, linestyle=(0, (4, 2)), zorder=4)
    per.plot(ax=ax, facecolor="none", edgecolor=COLORS["perimeter"],
             linewidth=1.15, zorder=5)

    for i, dmg in enumerate(DAMAGE_PLOT_ORDER):
        sub = res[res["DAMAGE"] == dmg]
        sub.plot(ax=ax, markersize=1.75, color=DAMAGE_COLORS[dmg], zorder=6 + i,
                 linewidth=0, alpha=0.90)

    map_frame(ax, crs=CRS, gis_frame=True)
    ax.set_facecolor("#D0D0D0")
    clip_map_layers(ax)
    add_scalebar(ax, 4.0, location="lower left")
    ax.set_aspect("auto")
    ax.set_autoscale_on(False)

    _add_ca_inset(ax, per)
    add_north_arrow(ax, style="arcgis_split", location="upper right", scale=1.05)

    handles = [Line2D([], [], marker="o", linestyle="", markersize=5,
                      color=DAMAGE_COLORS[d], label=s)
               for d, s in zip(DAMAGE_COLORS, DAMAGE_SHORT)]
    handles += [
        Line2D([], [], color=COLORS["perimeter"], linewidth=1.2, label="Fire perimeter"),
        Line2D([], [], color=COLORS["buffer"], linewidth=0.9,
               linestyle=(0, (4, 2)), label="Study area (+2 km)"),
    ]
    pos = ax.get_position()
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=FS["size_small"],
               frameon=True, edgecolor="#BDBDBD", fancybox=False,
               title="DINS damage class", title_fontsize=FS["size_label"],
               bbox_to_anchor=(pos.x0 + pos.width / 2, 0.03), columnspacing=0.9,
               handletextpad=0.35, borderpad=0.6)

    cover_frame_overflow(fig, ax)
    save_figure(fig, "Figure_01_study_area")


# ----------------------------------------------------------------------------- Figure 2
def figure_02_event_context():
    gm_hist = pd.read_csv(path("data", "raw", "climate",
                               "gridmet_octdec_precip_1980_2024.csv"))
    e5 = pd.read_csv(path("data", "raw", "climate", "era5land_hourly_event.csv"))
    gm = pd.read_csv(path("data", "raw", "climate", "gridmet_daily.csv"))

    # Distinct, colorblind-friendly series colors (Okabe–Ito inspired)
    c_hist = "#8EABBF"       # cool slate — prior years
    c_drought = "#D55E00"    # vermillion — extreme dry 2024
    c_median = "#1A1A1A"
    c_fuel = "#0072B2"       # blue — fuel moisture
    c_vpd = "#8B3A00"        # deep burnt orange — strong on white
    c_wind = "#332288"       # indigo — wind
    c_rh = "#009E73"         # teal — humidity
    c_ign = "#CC3311"        # red — ignition

    fig, axes = plt.subplots(3, 1, figsize=(W_DOUBLE, 16.5 * CM),
                             layout="constrained")

    ax = axes[0]
    bar_c = [c_drought if y == 2024 else c_hist for y in gm_hist["year"]]
    ax.bar(gm_hist["year"], gm_hist["octdec_pr_mm"], color=bar_c, width=0.85,
           edgecolor="none", zorder=2)
    med = gm_hist["octdec_pr_mm"].median()
    ax.axhline(med, color=c_median, linewidth=1.1, linestyle="--", zorder=3)
    v2024 = float(gm_hist.loc[gm_hist["year"] == 2024, "octdec_pr_mm"].iloc[0])
    # Vertical label above the near-zero 2024 bar, with arrow
    y_max = float(gm_hist["octdec_pr_mm"].max())
    ax.annotate(
        f"2024: {v2024:.0f} mm",
        xy=(2024, v2024),
        xytext=(2024, max(115, v2024 + 0.32 * y_max)),
        textcoords="data",
        fontsize=FS["size_small"],
        fontweight="bold",
        color=c_drought,
        ha="center",
        va="bottom",
        rotation=90,
        arrowprops=dict(arrowstyle="->", color=c_drought, lw=1.0),
        zorder=5,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Oct–Dec precipitation (mm)")
    ax.set_xlim(1979, 2025)
    handles = [
        Patch(facecolor=c_hist, edgecolor="none", label="1980–2023"),
        Patch(facecolor=c_drought, edgecolor="none",
              label=f"2024 ({v2024:.0f} mm)"),
        Line2D([], [], color=c_median, linestyle="--", linewidth=1.1,
               label=f"Median ({med:.0f} mm)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=FS["size_small"],
              frameon=True, edgecolor="#BDBDBD", fancybox=False, borderpad=0.35,
              labelspacing=0.3, handlelength=1.4, handleheight=0.9)
    full_frame(ax)
    panel_label(ax, "a", "Antecedent drought")

    ax = axes[1]
    gm["date"] = pd.to_datetime(gm["date"])
    ln1 = ax.plot(gm["date"], gm["fm100"], color=c_fuel, linewidth=1.35,
                  label="100-h fuel moisture")
    ax2 = ax.twinx()
    ln2 = ax2.plot(gm["date"], gm["vpd"], color=c_vpd, linewidth=1.4,
                   label="VPD")
    ign = ax.axvline(pd.Timestamp("2025-01-07"), color=c_ign,
                     linewidth=1.15, linestyle="--", label="Ignition")
    ax.set_ylabel("100-h fuel moisture (%)", color=c_fuel)
    ax2.set_ylabel("VPD (kPa)", color=c_vpd)
    ax.tick_params(axis="y", labelcolor=c_fuel, colors=c_fuel)
    ax2.tick_params(axis="y", labelcolor=c_vpd, colors=c_vpd)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4, 7, 10, 1]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_locator(NullLocator())
    lines = ln1 + ln2 + [ign]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right",
              fontsize=FS["size_small"], frameon=True, edgecolor="#BDBDBD",
              fancybox=False, borderpad=0.35, labelspacing=0.3, handlelength=1.5)
    full_frame(ax, twin=ax2)
    ax.spines["left"].set_color(c_fuel)
    ax2.spines["right"].set_color(c_vpd)
    panel_label(ax, "b", "Fuel moisture & VPD")

    ax = axes[2]
    e5["time"] = pd.to_datetime(e5["time"])
    ws = np.hypot(e5["u_component_of_wind_10m"], e5["v_component_of_wind_10m"])
    t = e5["temperature_2m"] - 273.15
    td = e5["dewpoint_temperature_2m"] - 273.15
    rh = 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
    ln1 = ax.plot(e5["time"], ws, color=c_wind, linewidth=1.35,
                  label="10 m wind speed")
    ax2 = ax.twinx()
    ln2 = ax2.plot(e5["time"], rh, color=c_rh, linewidth=1.25,
                   label="Relative humidity")
    ign = ax.axvline(pd.Timestamp("2025-01-07 18:00"), color=c_ign,
                     linewidth=1.15, linestyle="--", label="Ignition")
    ax.set_ylabel("10 m wind speed (m s$^{-1}$)", color=c_wind)
    ax2.set_ylabel("Relative humidity (%)", color=c_rh)
    ax.tick_params(axis="y", labelcolor=c_wind, colors=c_wind)
    ax2.tick_params(axis="y", labelcolor=c_rh, colors=c_rh)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
    ax.xaxis.set_minor_locator(NullLocator())
    lines = ln1 + ln2 + [ign]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right",
              fontsize=FS["size_small"], frameon=True, edgecolor="#BDBDBD",
              fancybox=False, borderpad=0.35, labelspacing=0.3, handlelength=1.5)
    full_frame(ax, twin=ax2)
    ax.spines["left"].set_color(c_wind)
    ax2.spines["right"].set_color(c_rh)
    panel_label(ax, "c", "Event wind & humidity")

    save_figure(fig, "Figure_02_event_context")


# ----------------------------------------------------------------------------- Figure 3
def _fuel_groups_panel(ax, per, study) -> None:
    with rasterio.open(path("data", "raw", "landfire", "lf2024_fbfm40.tif")) as src:
        fb = src.read(1)
        b = src.bounds
    groups = np.full(fb.shape, np.nan)
    groups[np.isin(fb, [91, 92, 93])] = 0
    groups[np.isin(fb, [98, 99])] = 1
    groups[np.isin(fb, list(range(101, 110)) + list(range(121, 125)))] = 2
    groups[np.isin(fb, list(range(141, 150)))] = 3
    groups[np.isin(fb, list(range(161, 166)) + list(range(181, 190))
           + list(range(201, 205)))] = 4
    cmap = ListedColormap(["#BDBDBD", "#D9E8F2", "#E8D48A", "#C0773F", "#3F6B38"])
    ext = [b.left, b.right, b.bottom, b.top]
    im = ax.imshow(groups, cmap=cmap, extent=ext, interpolation="nearest", zorder=1,
                   vmin=-0.5, vmax=4.5, aspect="auto")
    ocean_overlay(ax)
    per.plot(ax=ax, facecolor="none", edgecolor=COLORS["perimeter"], linewidth=0.9, zorder=5)
    set_extent(ax, study, pad=400)
    ax.set_label("map")
    cb = add_cbar(im, ax, "")
    map_frame(ax, crs=CRS, gis_frame=True)
    add_scalebar(ax, 4.0)
    cb.set_ticks([0, 1, 2, 3, 4])
    cb.set_ticklabels(["Developed", "Water", "Grass", "Shrub", "Timber"])
    cb.ax.tick_params(labelsize=FS["size_small"])


def _burn_severity_panel(ax, per, study) -> None:
    """Post-fire categorical companion to pre-fire fuel groups."""
    from visualization.style import _HS, _ensure_hillshade

    with rasterio.open(path("data", "interim", "severity", "severity_class.tif")) as src:
        sev = src.read(1).astype("float64")
        b = src.bounds
        nod = src.nodata
    if nod is not None:
        sev[sev == nod] = np.nan
    sev[~np.isfinite(sev)] = np.nan
    # Mask ocean so the sea matches other panels (blue), not Unburned gray
    _ensure_hillshade()
    ocean = _HS["ocean"]
    if ocean.shape == sev.shape:
        sev = np.where(np.isfinite(ocean), np.nan, sev)
    # Unburned / Low / Moderate / High — unburned must not look like empty frame
    cmap = ListedColormap(["#D9D9D9", "#FEE08B", "#FC8D59", "#D73027"])
    cmap = cmap.copy()
    cmap.set_bad(color="none")
    ext = [b.left, b.right, b.bottom, b.top]
    set_extent(ax, study, pad=400)
    ax.set_autoscale_on(False)
    hillshade_background(ax, alpha=0.45)
    im = ax.imshow(sev, cmap=cmap, extent=ext, interpolation="nearest", zorder=2,
                   vmin=-0.5, vmax=3.5, aspect="auto")
    ocean_overlay(ax, zorder=3)
    per.plot(ax=ax, facecolor="none", edgecolor=COLORS["perimeter"], linewidth=0.9, zorder=5)
    ax.set_label("map")
    cb = add_cbar(im, ax, "")
    map_frame(ax, crs=CRS, gis_frame=True)
    add_scalebar(ax, 4.0)
    cb.set_ticks([0, 1, 2, 3])
    cb.set_ticklabels(["Unburned", "Low", "Moderate", "High"])
    cb.ax.tick_params(labelsize=FS["size_small"])


def _figure_03_environment(period: str) -> None:
    """Shared 2x2 environment layout for pre- or post-fire."""
    assert period in {"prefire", "postfire"}
    study, per, _, _ = load_layers()
    per = per.to_crs(CRS)
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 13.0 * CM))

    tag = "Pre-fire" if period == "prefire" else "Post-fire"
    ndvi = path("data", "raw", "gee", f"s2_{period}_ndvi.tif")
    ndmi = path("data", "raw", "gee", f"s2_{period}_ndmi.tif")
    lst = path("data", "raw", "gee", f"landsat_{period}_lst.tif")

    raster_map(axes[0, 0], ndvi, PALETTES["vegetation_seq"], 0, 0.8, per, study, "NDVI",
               xlabels=False)
    panel_label(axes[0, 0], "a", f"{tag} NDVI")

    raster_map(axes[0, 1], ndmi, PALETTES["moisture_seq"], -0.3, 0.4, per, study, "NDMI",
               ylabels=False, xlabels=False)
    panel_label(axes[0, 1], "b", f"{tag} NDMI")

    if period == "prefire":
        _fuel_groups_panel(axes[1, 0], per, study)
        panel_label(axes[1, 0], "c", "Fuel groups")
    else:
        _burn_severity_panel(axes[1, 0], per, study)
        panel_label(axes[1, 0], "c", "Burn severity")

    raster_map(axes[1, 1], lst, PALETTES["lst_seq"], 15, 40, per, study, "LST (C)",
               ylabels=False)
    panel_label(axes[1, 1], "d", f"{tag} LST")

    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.06,
                       hspace=0.18, wspace=0.22)
    out = ("Figure_03_01_prefire_environment" if period == "prefire"
           else "Figure_03_02_postfire_environment")
    save_figure(fig, out)


def figure_03_prefire_environment():
    _figure_03_environment("prefire")


def figure_03_02_postfire_environment():
    _figure_03_environment("postfire")


def figure_03_03_rgb_context(active_date: str | None = "20250107", out_name: str | None = None):
    """2x3 true color + SWIR false color: pre | active | post.

    Row 1 (a–c): True Color (B4/B3/B2)
    Row 2 (d–f): False color (B12/B11/B4)
    Band details belong in the figure caption, not panel titles.
    Default active scene: ignition day 2025-01-07.
    """
    study, per, _, _ = load_layers()
    per = per.to_crs(CRS)
    if active_date is None:
        active_date = "20250107"

    gee_dir = path("data", "raw", "gee")
    pre_rgb = gee_dir / "s2_prefire_rgb.tif"
    pre_swir = gee_dir / "s2_prefire_swir.tif"
    post_rgb = gee_dir / "s2_postfire_rgb.tif"
    post_swir = gee_dir / "s2_postfire_swir.tif"
    act_rgb = gee_dir / f"s2_activefire_rgb_{active_date}.tif"
    act_swir = gee_dir / f"s2_activefire_swir_{active_date}.tif"

    for fp in (pre_rgb, pre_swir, post_rgb, post_swir, act_rgb, act_swir):
        if not fp.exists():
            raise FileNotFoundError(
                f"missing {fp.name}; run src/data/reexport_rgb_composites.py first"
            )

    fig, axes = plt.subplots(2, 3, figsize=(W_DOUBLE, 12.4 * CM))
    top = [
        ("a", "True Color · Pre-fire", pre_rgb),
        ("b", "True Color · Active fire", act_rgb),
        ("c", "True Color · Post-fire", post_rgb),
    ]
    bot = [
        ("d", "False color · Pre-fire", pre_swir),
        ("e", "False color · Active fire", act_swir),
        ("f", "False color · Post-fire", post_swir),
    ]
    for row, panels, xlabels in (
        (0, top, False),
        (1, bot, True),
    ):
        for col, (letter, title, fp) in enumerate(panels):
            ax = axes[row, col]
            rgb_map(ax, fp, per, study, xlabels=xlabels, ylabels=(col == 0))
            panel_label(ax, letter, title)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.07,
                        wspace=0.16, hspace=0.22)
    save_figure(fig, out_name or "Figure_03_03_TRUE_B4B3B2_FalseB12B11B4")


# ----------------------------------------------------------------------------- Figure 4
def figure_04_built_environment():
    from scipy.stats import gaussian_kde

    feats = pd.read_parquet(path("data", "processed", "structure_features.parquet"))
    res = feats[feats["residential"] == 1]
    study, per, _, _ = load_layers()
    per = per.to_crs(CRS)

    # Survived / destroyed — same narrative colors as Fig. 1 / style.yaml
    c_surv = COLORS["survived"]
    c_dest = COLORS["destroyed"]

    # Match Fig. 3 map panel aspect (wide, not tall/narrow): same figsize
    # margins/wspace as the 2x2 environment maps, shorter top row height.
    fig = plt.figure(figsize=(W_DOUBLE, 13.0 * CM))
    gs_top = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.95, bottom=0.55,
                              wspace=0.22)
    gs_bot = fig.add_gridspec(1, 3, left=0.07, right=0.97, top=0.46, bottom=0.08,
                              wspace=0.30)

    ax = fig.add_subplot(gs_top[0, 0])
    ax.set_label("map")
    set_extent(ax, study, pad=400)
    ax.set_autoscale_on(False)
    hillshade_background(ax, alpha=0.65)
    sc = ax.scatter(res["x"], res["y"], c=res["bld_count_r100"], s=1.15,
                    cmap=PALETTES["density_seq"], vmin=0, vmax=60, zorder=6,
                    linewidths=0)
    per.plot(ax=ax, facecolor="none", edgecolor=COLORS["perimeter"], linewidth=1.0, zorder=5)
    add_cbar(sc, ax, "Buildings within 100 m")
    map_frame(ax, crs=CRS, gis_frame=True)
    add_scalebar(ax, 4.0)
    panel_label(ax, "a", "Building density")

    ax = fig.add_subplot(gs_top[0, 1])
    ax.set_label("map")
    set_extent(ax, study, pad=400)
    ax.set_autoscale_on(False)
    hillshade_background(ax, alpha=0.65)
    sc = ax.scatter(res["x"], res["y"], c=res["yearbuilt"], s=1.15,
                    cmap=PALETTES["year_seq"], vmin=1930, vmax=2020, zorder=6,
                    linewidths=0)
    per.plot(ax=ax, facecolor="none", edgecolor=COLORS["perimeter"], linewidth=1.0, zorder=5)
    add_cbar(sc, ax, "Year built")
    map_frame(ax, crs=CRS, ylabels=False, gis_frame=True)
    add_scalebar(ax, 4.0)
    panel_label(ax, "b", "Year built")

    hist_specs = [
        ("nn_building_dist_m", "Nearest-building distance (m)", (0, 30),
         "Building spacing"),
        ("bld_count_r100", "Buildings within 100 m", (0, 80),
         "Local density"),
        ("yearbuilt", "Year built", (1920, 2025),
         "Construction era"),
    ]
    for i, (col, xlab, xlim, title) in enumerate(hist_specs):
        ax = fig.add_subplot(gs_bot[0, i])
        xs = np.linspace(xlim[0], xlim[1], 220)
        for v, c, lbl in [(0, c_surv, "Survived"), (1, c_dest, "Destroyed")]:
            d = res.loc[res["destroyed"] == v, col].dropna().to_numpy(dtype=float)
            d = d[(d >= xlim[0]) & (d <= xlim[1])]
            if len(d) < 8:
                continue
            # Light KDE only — keep peaks readable (not over-smoothed)
            kde = gaussian_kde(d, bw_method=0.05)
            dens = kde(xs)
            ax.fill_between(xs, dens, color=c, alpha=0.28, linewidth=0, zorder=2)
            ax.plot(xs, dens, color=c, linewidth=1.45, label=lbl, zorder=3)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Density" if i == 0 else "")
        if i == 0:
            ax.legend(fontsize=FS["size_small"], loc="upper right",
                      frameon=True, edgecolor="#BDBDBD", fancybox=False,
                      borderpad=0.35, labelspacing=0.3, handlelength=1.4)
        ax.set_xlim(xlim)
        ax.set_ylim(bottom=0)
        ax.margins(x=0, y=0.06)
        full_frame(ax)
        panel_label(ax, chr(99 + i), title)

    save_figure(fig, "Figure_04_built_environment")


# ----------------------------------------------------------------------------- Figure 5
# Raincloud distributions — 4×2 layout (panels as wide as Fig. 3).
FIG05_PANELS = [
    # (column, concise title, y-axis label, optional y-limits)
    ("ndvi_pre_r30_100", "Prefire NDVI", "NDVI", None),
    ("ndmi_pre_r30_100", "Prefire NDMI", "NDMI", None),
    ("fuel_wildland_r100_300", "Wildland fuel", "Fuel fraction", None),
    ("slope_deg_pt", "Slope", "Slope (°)", None),
    ("nn_building_dist_m", "Building spacing", "Distance (m)", (0, 40)),
    ("bld_count_r100", "Local density", "Buildings / 100 m", None),
    ("yearbuilt", "Year built", "Year", None),
    ("dist_wildland_m", "Wildland distance", "Distance (m)", (0, 550)),
]
DAMAGE_XTICK = ["None", "Aff.", "Min.", "Maj.", "Dest."]


def _fig05_load():
    feats = pd.read_parquet(path("data", "processed", "structure_features.parquet"))
    res = feats[feats["residential"] == 1].copy()
    order = list(DAMAGE_COLORS.keys())
    return res, order


def _fig05_series(res, order, col):
    return [res.loc[res["damage"] == d, col].dropna().to_numpy(dtype=float)
            for d in order]


def _fig05_ylim(ax, data, ylim):
    if ylim is not None:
        ax.set_ylim(*ylim)
        return
    vals = np.concatenate([d for d in data if len(d)])
    if len(vals) == 0:
        return
    lo, hi = np.nanpercentile(vals, [1, 99])
    pad = 0.06 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + pad)


def _fig05_axes_finish(ax, i, ylabel, title, order):
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(DAMAGE_XTICK, fontsize=FS["size_small"],
                       rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.4, 5.6)
    full_frame(ax)
    panel_label(ax, chr(97 + i), title)


def _fig05_legend(fig, res, order):
    """Fig. 1–style legend: color + abbreviation + full name + n."""
    ns = [int((res["damage"] == d).sum()) for d in order]
    handles = []
    for d, abbr, full, n in zip(order, DAMAGE_XTICK, DAMAGE_SHORT, ns):
        handles.append(
            Line2D([], [], marker="o", linestyle="", markersize=5.5,
                   color=DAMAGE_COLORS[d],
                   label=f"{abbr} — {full} (n={n:,})")
        )
    fig.legend(
        handles=handles, loc="lower center", ncol=3,
        fontsize=FS["size_small"], frameon=True, edgecolor="#BDBDBD",
        fancybox=False, title="DINS damage class",
        title_fontsize=FS["size_label"],
        bbox_to_anchor=(0.535, -0.002), columnspacing=1.05,
        handletextpad=0.35, borderpad=0.55, labelspacing=0.35,
    )


def _kde_1d(vals, grid, bw=0.12):
    from scipy.stats import gaussian_kde
    v = vals[np.isfinite(vals)]
    if len(v) < 8:
        return None
    kde = gaussian_kde(v, bw_method=bw)
    dens = kde(grid)
    dens = dens / (dens.max() + 1e-12)
    return dens


def _draw_raincloud(ax, data, order, rng):
    """Half-violin (left) + tonal box + jittered points (right)."""
    for i, (vals, d) in enumerate(zip(data, order), start=1):
        c = DAMAGE_COLORS[d]
        v = vals[np.isfinite(vals)]
        if len(v) < 8:
            continue
        lo, hi = np.nanpercentile(v, [1, 99])
        grid = np.linspace(lo, hi, 160)
        dens = _kde_1d(v, grid, bw=0.14)
        if dens is not None:
            w = 0.38 * dens
            ax.fill_betweenx(grid, i - w, i, color=c, alpha=0.38,
                             linewidth=0, zorder=2)
            ax.plot(i - w, grid, color=c, linewidth=0.85, alpha=0.85, zorder=3)

        n_pts = min(90, len(v))
        sample = rng.choice(v, size=n_pts, replace=False)
        jitter = rng.uniform(0.06, 0.34, size=n_pts)
        ax.scatter(i + jitter, sample, s=3.5, color=c, alpha=0.32,
                   linewidths=0, zorder=1, rasterized=True)

        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        whis_lo = max(v.min(), q1 - 1.5 * iqr)
        whis_hi = min(v.max(), q3 + 1.5 * iqr)
        ax.plot([i, i], [whis_lo, whis_hi], color=COLORS["neutral_dark"],
                linewidth=0.75, zorder=4, solid_capstyle="round")
        ax.add_patch(Rectangle(
            (i - 0.07, q1), 0.14, max(q3 - q1, 1e-9), facecolor=c,
            edgecolor=COLORS["neutral_dark"], linewidth=0.7, zorder=5,
            alpha=1.0,
        ))
        ax.plot([i - 0.07, i + 0.07], [med, med], color=COLORS["neutral_dark"],
                linewidth=1.2, zorder=6, solid_capstyle="butt")


def figure_05_distributions():
    """Raincloud distributions of key predictors by DINS damage class."""
    res, order = _fig05_load()
    # Extra bottom room for Fig. 1–style legend
    fig, axes = plt.subplots(4, 2, figsize=(W_DOUBLE, 19.6 * CM))
    rng = np.random.default_rng(42)

    for i, (ax, (col, title, ylabel, ylim)) in enumerate(
            zip(axes.flat, FIG05_PANELS)):
        data = _fig05_series(res, order, col)
        _draw_raincloud(ax, data, order, rng)
        _fig05_ylim(ax, data, ylim)
        _fig05_axes_finish(ax, i, ylabel, title, order)

    _fig05_legend(fig, res, order)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.975, bottom=0.105,
                        hspace=0.34, wspace=0.28)
    save_figure(fig, "Figure_05_distributions")


# ----------------------------------------------------------------------------- Figure S1

def _s1_style():
    """Shared professional palette (paper-consistent)."""
    return {
        "src_fc": "#F7F1E6", "src_ec": "#8B7355",
        "hz_fc": "#E8F1F8", "hz_ec": "#2F5F8A",
        "a_fc": "#E4F0E6", "a_ec": "#2F6B3A",
        "b_fc": "#F8E8E6", "b_ec": "#A33B3B",
        "cv_fc": "#EEF0F4", "cv_ec": "#4A5568",
        "arrow": "#4A5568",
        "rail": "#6B7C8F",
    }


def _s1_box(ax, x, y, w, h, lines, fc, ec, fs=None, round_pad=0.04):
    from matplotlib.patches import FancyBboxPatch
    fs = fs or FS["size_small"]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={round_pad},rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=1.15, zorder=2,
        mutation_aspect=0.55, clip_on=False,
    ))
    n = len(lines)
    for i, line in enumerate(lines):
        yy = y + h - (i + 0.72) * (h / (n + 0.45))
        weight = "bold" if i == 0 and n > 1 else "normal"
        size = (fs + 0.5) if weight == "bold" else fs
        ax.text(x + w / 2, yy, line, ha="center", va="center",
                fontsize=size, zorder=3, color=COLORS["neutral_dark"],
                fontweight=weight, clip_on=False)


def _s1_arrow(ax, x0, y0, x1, y1, color, lw=1.25, style="-|>", rad=0.0):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw, mutation_scale=12,
            connectionstyle=f"arc3,rad={rad}" if rad else "arc3,rad=0",
            shrinkA=0, shrinkB=0,
        ),
        zorder=1,
    )


def _s1_down(ax, x, y_bottom_of_upper, y_top_of_lower, color, lw=1.25):
    gap = y_bottom_of_upper - y_top_of_lower
    if gap < 0.12:
        return
    _s1_arrow(ax, x, y_bottom_of_upper - 0.01, x, y_top_of_lower + 0.01, color, lw=lw)


def _s1_vbar(ax, x, y0, y1, color):
    ax.plot([x, x], [y0, y1], color=color, lw=1.15, zorder=1, solid_capstyle="round")


def _s1_banner(ax, x, y, w, h, text, fc, ec):
    _s1_box(ax, x, y, w, h, [text], fc, ec, fs=FS["size_label"], round_pad=0.02)


def _s1_stage_title(ax, y, num, label, color, cx=6.0):
    """Centered stage header: circled number + label above each group."""
    r = 0.17
    # Center the [circle + gap + label] pair on cx
    circ_x = cx - 0.55
    ax.add_patch(plt.Circle((circ_x, y), r, facecolor=color, edgecolor="none", zorder=3))
    ax.text(circ_x, y, num, ha="center", va="center",
            fontsize=FS["size_label"], fontweight="bold", color="white", zorder=4)
    ax.text(circ_x + r + 0.12, y, label, ha="left", va="center",
            fontsize=FS["size_label"] + 0.4, fontweight="bold", color=color, zorder=3)


def _s1_common_content():
    srcs = [
        ["CAL FIRE DINS", "12,137 inspections"],
        ["Sentinel-2 L2A", "+ Landsat LST"],
        ["LANDFIRE 2024", "fuel models"],
        ["USGS 3DEP", "10 m DEM"],
        ["OpenStreetMap", "1 Jan 2025"],
        ["CDC SVI 2022", "+ TIGER tracts"],
    ]
    track_a = [
        ["Pre-fire predictors only",
         "NDVI / NDMI (3 rings), LST, fuels",
         "Terrain, buildings, roads, year built"],
        ["Damage models (destroyed vs survived)",
         "M0 → M1 vegetation → M2 built → M3",
         "L2 logistic + XGBoost"],
        ["Spatial validation",
         "1 km block CV (5 folds) vs random CV",
         "Calibration, scale, block-size tests"],
        ["Interpretation",
         "Odds ratios and SHAP values",
         "Partial effects and scale dependence"],
    ]
    track_b = [
        ["Impact indicators (not predictors)",
         "dNBR / RdNBR, NDVI recovery",
         "SVI, network access to services"],
        ["Impact and community context",
         "Severity vs damage, recovery",
         "Tract destruction and SVI"],
    ]
    return srcs, track_a, track_b


def figure_s1_workflow():
    """Four-stage workflow: Sources → Harmonize → Analyze (A/B) → Interpret."""
    S = _s1_style()
    srcs, track_a, track_b = _s1_common_content()
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 15.2 * CM))
    ax.set_xlim(0, 12)
    ax.set_ylim(0.25, 12.0)
    ax.axis("off")

    # ---- 1 · Sources (6 boxes centered inside frame; leave room for rounded pad) ----
    _s1_stage_title(ax, 11.65, "1", "Sources", S["src_ec"])
    n_src = len(srcs)
    # Match Track A/B outer edges (0.40 … 11.60); leave pad room on both sides
    row_left, row_right = 0.45, 11.55
    gap = 0.10
    avail = row_right - row_left
    sw = (avail - (n_src - 1) * gap) / n_src
    xs = []
    for i, lines in enumerate(srcs):
        x = row_left + i * (sw + gap)
        xs.append(x + sw / 2)
        _s1_box(ax, x, 10.15, sw, 1.20, lines, S["src_fc"], S["src_ec"],
                fs=FS["size_small"] - 0.3, round_pad=0.03)
        _s1_vbar(ax, x + sw / 2, 10.15, 9.70, S["rail"])
    ax.plot([xs[0], xs[-1]], [9.70, 9.70], color=S["rail"], lw=1.15, zorder=1)
    _s1_arrow(ax, 6.0, 9.70, 6.0, 9.15, S["arrow"])

    # ---- 2 · Harmonize ----
    _s1_stage_title(ax, 8.85, "2", "Harmonize", S["hz_ec"])
    _s1_box(ax, 0.55, 7.70, 10.9, 0.95,
            ["Harmonize to EPSG:26911",
             "QA gates · DINS audit · provenance · reproducible exports"],
            S["hz_fc"], S["hz_ec"])
    _s1_arrow(ax, 3.3, 7.70, 3.3, 7.30, S["a_ec"])
    _s1_arrow(ax, 8.7, 7.70, 8.7, 7.30, S["b_ec"])

    # ---- 3 · Analyze (title raised clear of track banners) ----
    _s1_stage_title(ax, 7.15, "3", "Analyze", "#555555")
    _s1_banner(ax, 0.40, 6.50, 5.4, 0.36, "Track A — prediction", S["a_fc"], S["a_ec"])
    _s1_banner(ax, 6.20, 6.50, 5.4, 0.36, "Track B — impact", S["b_fc"], S["b_ec"])

    a_stack = [
        (5.00, 1.25, track_a[0]),
        (3.45, 1.25, track_a[1]),
        (1.90, 1.25, track_a[2]),
    ]
    for i, (y, h, lines) in enumerate(a_stack):
        _s1_box(ax, 0.40, y, 5.4, h, lines, S["a_fc"], S["a_ec"])
        if i < len(a_stack) - 1:
            y2, h2, _ = a_stack[i + 1]
            _s1_down(ax, 3.1, y, y2 + h2, S["a_ec"])

    _s1_box(ax, 6.20, 4.60, 5.4, 1.65, track_b[0], S["b_fc"], S["b_ec"])
    _s1_down(ax, 8.9, 4.60, 3.50, S["b_ec"])
    _s1_box(ax, 6.20, 1.90, 5.4, 1.60, track_b[1], S["b_fc"], S["b_ec"])

    # ---- 4 · Interpret (title raised clear of the box) ----
    _s1_down(ax, 3.1, 1.90, 1.72, S["a_ec"])
    _s1_down(ax, 8.9, 1.90, 1.72, S["b_ec"])
    _s1_stage_title(ax, 1.58, "4", "Interpret", S["cv_ec"])
    _s1_box(ax, 0.55, 0.30, 10.9, 0.95,
            ["Interpretation · odds ratios & SHAP · community / recovery context",
             track_a[3][2]],
            S["cv_fc"], S["cv_ec"])

    save_figure(fig, "Figure_S1_workflow")


if __name__ == "__main__":
    apply_style()
    figure_01_study_area()
    figure_02_event_context()
    figure_03_prefire_environment()
    figure_03_02_postfire_environment()
    figure_03_03_rgb_context()
    figure_04_built_environment()
    figure_05_distributions()
    figure_s1_workflow()
    log.info("map figures complete")
