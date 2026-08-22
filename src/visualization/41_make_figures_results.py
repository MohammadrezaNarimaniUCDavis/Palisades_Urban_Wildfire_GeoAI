"""Publication figures 6-10: validation, interpretation, sensitivity,
severity/recovery, community context.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, load_config, path
from visualization.style import (
    COLORS, DAMAGE_COLORS, DAMAGE_SHORT, PALETTES, W_DOUBLE, CM, FS,
    add_cbar, add_scalebar, apply_style, clip_map_layers, cover_frame_overflow,
    full_frame, hillshade_background, map_frame, ocean_overlay,
    panel_label, save_figure, set_extent,
)

log = get_logger("41_figures_results")
cfg = load_config()
CRS = cfg["crs"]["analysis"]

NICE = {
    "bld_count_r300": "Buildings 300 m", "bld_count_r100": "Buildings 100 m",
    "bld_count_r30": "Buildings 30 m",
    "ndmi_pre_r100_300": "NDMI 100–300 m", "ndmi_pre_r30_100": "NDMI 30–100 m",
    "ndmi_pre_r0_30": "NDMI 0–30 m",
    "ndvi_pre_r100_300": "NDVI 100–300 m", "ndvi_pre_r30_100": "NDVI 30–100 m",
    "ndvi_pre_r0_30": "NDVI 0–30 m",
    "ndmi_anom_r100_300": "NDMI anom. 100–300 m", "ndmi_anom_r30_100": "NDMI anom. 30–100 m",
    "ndmi_anom_r0_30": "NDMI anom. 0–30 m",
    "ndvi_anom_r100_300": "NDVI anom. 100–300 m", "ndvi_anom_r30_100": "NDVI anom. 30–100 m",
    "ndvi_anom_r0_30": "NDVI anom. 0–30 m",
    "elevation_pt": "Elevation", "slope_deg_pt": "Slope", "slope_r0_100": "Slope 0–100 m",
    "northness_pt": "Northness", "eastness_pt": "Eastness",
    "tpi300_pt": "Topographic position", "tri_pt": "Ruggedness",
    "dist_wildland_m": "Dist. to wildland", "dist_road_m": "Dist. to road",
    "dist_deadend_m": "Dist. to dead-end", "road_len_r300_m_per_km2": "Road density 300 m",
    "nn_building_dist_m": "Nearest-building dist.", "footprint_area_m2": "Footprint area",
    "yearbuilt": "Year built", "assessed_value_log": "Assessed value (log)",
    "lst_pre_r30_100": "LST 30–100 m", "lst_pre_r100_300": "LST 100–300 m",
    "canopy_cover_r0_30": "Canopy 0–30 m", "canopy_cover_r30_100": "Canopy 30–100 m",
    "canopy_cover_r100_300": "Canopy 100–300 m",
    "fuel_wildland_r0_30": "Wildland fuel 0–30 m",
    "fuel_wildland_r30_100": "Wildland fuel 30–100 m",
    "fuel_wildland_r100_300": "Wildland fuel 100–300 m",
}

FOLD_COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]


def _study_per():
    study = gpd.read_file(path("data", "processed", "study_area.gpkg"), layer="study_area")
    per = gpd.read_file(path("data", "processed", "study_area.gpkg"),
                        layer="perimeter").to_crs(CRS)
    return study, per


# ------------------------------------------------------------------- Figure 6
def figure_06_validation():
    cv = pd.read_csv(path("outputs", "model_diagnostics", "cv_metrics.csv"))
    oof = pd.read_parquet(path("outputs", "model_diagnostics", "oof_predictions.parquet"))
    cal = pd.read_csv(path("outputs", "model_diagnostics", "calibration_data.csv"))
    study, per = _study_per()

    # Shared 2×2 columns so (a)/(c) and (b+cbar)/(d) align
    fig = plt.figure(figsize=(W_DOUBLE, 13.0 * CM))
    gs = fig.add_gridspec(
        2, 2, left=0.07, right=0.97, top=0.95, bottom=0.08,
        height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.26,
    )
    # Top-right: map + reserved colorbar strip (d uses full right cell width)
    gs_b = gs[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.048], wspace=0.12)

    # (a) Spatial CV folds
    ax = fig.add_subplot(gs[0, 0])
    ax.set_label("map")
    set_extent(ax, study, pad=400)
    ax.set_autoscale_on(False)
    hillshade_background(ax, alpha=0.65)
    fold_cmap = ListedColormap(FOLD_COLORS)
    ax.scatter(oof["x"], oof["y"], c=oof["spatial_fold"], cmap=fold_cmap,
               s=1.2, zorder=6, linewidths=0, vmin=-0.5, vmax=4.5)
    per.plot(ax=ax, facecolor="none", edgecolor=COLORS["perimeter"],
             linewidth=1.0, zorder=5)
    map_frame(ax, crs=CRS, gis_frame=True)
    add_scalebar(ax, 4.0)
    handles = [Line2D([], [], marker="s", linestyle="", color=FOLD_COLORS[i],
                      markersize=5.5, label=f"Fold {i + 1}") for i in range(5)]
    ax.legend(handles=handles, loc="upper left", fontsize=FS["size_small"],
              ncol=1, frameon=True, edgecolor="#BDBDBD", fancybox=False,
              borderpad=0.35, handletextpad=0.35, labelspacing=0.25)
    panel_label(ax, "a", "Spatial CV folds")

    # (b) Out-of-fold probability
    ax = fig.add_subplot(gs_b[0, 0])
    ax.set_label("map")
    set_extent(ax, study, pad=400)
    ax.set_autoscale_on(False)
    hillshade_background(ax, alpha=0.55)
    sc = ax.scatter(oof["x"], oof["y"], c=oof["spatial_M3_xgb"], cmap="RdBu_r",
                    vmin=0, vmax=1, s=1.2, zorder=6, linewidths=0)
    per.plot(ax=ax, facecolor="none", edgecolor=COLORS["perimeter"],
             linewidth=1.0, zorder=5)
    map_frame(ax, crs=CRS, ylabels=False, gis_frame=True)
    add_scalebar(ax, 4.0)
    panel_label(ax, "b", "Out-of-fold probability")

    cax = fig.add_subplot(gs_b[0, 1])
    cax.set_label("colorbar")
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("P(destroyed), out-of-fold", size=FS["size_label"])
    cb.ax.tick_params(labelsize=FS["size_small"], length=2.5, width=0.6,
                      direction="in")
    cb.outline.set_linewidth(0.7)
    cb.outline.set_edgecolor(COLORS["neutral_dark"])

    # (c) ROC-AUC — same column width as (a)
    ax = fig.add_subplot(gs[1, 0])
    xpos = np.arange(4)
    # Clearer scheme contrast (Okabe–Ito): cool blue = random, vermillion = spatial
    c_rand, c_spat = "#0072B2", "#D55E00"
    off = {"random": -0.18, "spatial": 0.18}
    mark = {"logit": "o", "xgb": "s"}
    col = {"random": c_rand, "spatial": c_spat}
    for scheme in ("random", "spatial"):
        for learner in ("logit", "xgb"):
            sub = (cv[(cv.cv_scheme == scheme) & (cv.learner == learner)]
                   .groupby("model_block")["roc_auc"].agg(["mean", "std"])
                   .reindex(["M0", "M1", "M2", "M3"]))
            dx = off[scheme] + (0.07 if learner == "xgb" else -0.07)
            ax.errorbar(
                xpos + dx, sub["mean"], yerr=sub["std"], fmt=mark[learner],
                color=col[scheme], markersize=5.6, capsize=2.4, linewidth=1.05,
                elinewidth=0.95, markerfacecolor="white" if learner == "logit"
                else col[scheme], markeredgewidth=1.05, zorder=3,
            )
    ax.set_xticks(xpos)
    ax.set_xticklabels(["M0", "M1", "M2", "M3"])
    ax.set_xlabel("Predictor block")
    ax.set_ylabel("ROC-AUC (mean ± SD)")
    ax.axhline(0.5, color="#BDBDBD", linewidth=0.75, linestyle=":", zorder=1)
    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor=c_rand, markeredgewidth=1.05,
               label="Random CV, logistic"),
        Line2D([], [], marker="s", linestyle="", color=c_rand,
               label="Random CV, XGBoost"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor=c_spat, markeredgewidth=1.05,
               label="Spatial CV, logistic"),
        Line2D([], [], marker="s", linestyle="", color=c_spat,
               label="Spatial CV, XGBoost"),
    ]
    # Inside frame, bottom-right (2×2); series sit near the top
    ax.legend(
        handles=handles, fontsize=FS["size_small"], loc="lower right",
        ncol=2, frameon=True, edgecolor="#BDBDBD", fancybox=False,
        borderpad=0.30, labelspacing=0.22, handletextpad=0.30,
        handlelength=1.15, columnspacing=0.85,
    )
    ax.set_ylim(0.45, 1.02)
    ax.set_xlim(-0.35, 3.35)
    full_frame(ax)
    panel_label(ax, "c", "Model discrimination")

    # (d) Calibration — full right cell, aligns with (b) map + colorbar
    ax = fig.add_subplot(gs[1, 1])
    c_logit, c_xgb = COLORS["survived"], "#D55E00"
    ax.plot([0, 1], [0, 1], color="#B0B0B0", linewidth=0.9, linestyle="--",
            label="1:1", zorder=1)
    for learner, c, lab, mk in [
        ("logit", c_logit, "Logistic", "o"),
        ("xgb", c_xgb, "XGBoost", "s"),
    ]:
        sub = cal[cal.learner == learner]
        ax.plot(sub["mean_pred"], sub["frac_pos"], marker=mk, markersize=5.0,
                color=c, linewidth=1.35, label=lab, markerfacecolor=c,
                markeredgecolor="white", markeredgewidth=0.55, zorder=3)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed destroyed fraction")
    ax.yaxis.labelpad = 6
    ax.legend(fontsize=FS["size_small"], loc="lower right",
              frameon=True, edgecolor="#BDBDBD", fancybox=False,
              borderpad=0.35, labelspacing=0.28, handlelength=1.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    full_frame(ax)
    panel_label(ax, "d", "Calibration")

    save_figure(fig, "Figure_06_validation")


# ------------------------------------------------------------------- Figure 7
def _fig07_draw_or_diverging(ax, sig):
    """Diverging effect bars from null + CI whiskers (boxplot-like)."""
    from matplotlib.colors import to_rgba

    ypos = np.arange(len(sig))
    colors = [COLORS["destroyed"] if o > 1 else COLORS["survived"] for o in sig["or"]]
    ax.set_xscale("log")
    ax.axvline(1, color="#999999", linewidth=0.7, zorder=1)
    for y in ypos:
        if y % 2 == 0:
            ax.axhspan(y - 0.48, y + 0.48, color="#F4F4F4", zorder=0, lw=0)

    for y, (_, row), c in zip(ypos, sig.iterrows(), colors):
        o, lo, hi = float(row["or"]), float(row["or_lo"]), float(row["or_hi"])
        x0, x1 = (min(1.0, o), max(1.0, o))
        # Faded bars so CI boxplots read clearly on top
        ax.barh(y, x1 - x0, left=x0, height=0.52, color=to_rgba(c, 0.38),
                edgecolor="none", zorder=2, align="center")
        ax.hlines(y, lo, hi, color=c, linewidth=1.25, zorder=4)
        ax.vlines([lo, hi], y - 0.18, y + 0.18, colors=c, linewidth=1.25, zorder=4)
        ax.scatter([o], [y], marker="s", s=14, color="white",
                   edgecolors=c, linewidths=1.0, zorder=5)

    ax.set_yticks(ypos)
    ax.set_yticklabels([NICE.get(i, i) for i in sig.index],
                       fontsize=FS["size_small"])
    ax.set_ylim(-0.7, len(sig) - 0.3)
    ax.set_xlabel("Odds ratio per SD (95% CI)")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(float(sig["or_lo"].min()) * 0.90,
                float(sig["or_hi"].max()) * 1.12)


def figure_07_interpretation():
    from matplotlib.colors import ListedColormap

    dep_cmap = PALETTES["year_seq"]  # cividis

    imp = pd.read_csv(path("outputs", "model_diagnostics", "shap_importance.csv"))
    sample = pd.read_parquet(path("outputs", "model_diagnostics", "shap_sample.parquet"))
    lc = pd.read_csv(path("outputs", "model_diagnostics", "logit_coefficients.csv"), index_col=0)

    fig = plt.figure(figsize=(W_DOUBLE, 14.8 * CM))
    gs = fig.add_gridspec(
        2, 3, left=0.11, right=0.97, top=0.94, bottom=0.07,
        width_ratios=[1.0, 0.28, 1.12], height_ratios=[1.22, 1.0],
        hspace=0.32, wspace=0.08,
    )

    def _with_cbar(spec):
        return spec.subgridspec(1, 2, width_ratios=[1.0, 0.065], wspace=0.08)

    def _v_cbar(mappable, cax, label, ticks=None, ticklabels=None):
        cax.set_label("colorbar")
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label(label, size=FS["size_small"], labelpad=3)
        cb.ax.tick_params(labelsize=FS["size_small"], length=2.2, width=0.55,
                          direction="in", pad=1.5)
        cb.outline.set_linewidth(0.7)
        cb.outline.set_edgecolor(COLORS["neutral_dark"])
        if ticks is not None:
            cb.set_ticks(ticks)
        if ticklabels is not None:
            cb.set_ticklabels(ticklabels)
        return cb

    # (a) SHAP summary
    gs_a = _with_cbar(gs[0, 0])
    ax = fig.add_subplot(gs_a[0, 0])
    top = imp.head(12)["predictor"].tolist()
    rng = np.random.default_rng(0)
    for i, p in enumerate(reversed(top)):
        sv = sample[f"shap_{p}"].values
        fv = sample[f"val_{p}"].values
        q5, q95 = np.nanpercentile(fv, [5, 95])
        norm = np.clip((fv - q5) / max(q95 - q5, 1e-9), 0, 1)
        jitter = rng.normal(0, 0.13, len(sv))
        ax.scatter(sv, i + jitter, c=norm, cmap=PALETTES["shap_div"], s=2.4,
                   alpha=0.55, linewidths=0)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([NICE.get(p, p) for p in reversed(top)],
                       fontsize=FS["size_small"])
    ax.axvline(0, color="#999999", linewidth=0.7)
    ax.set_xlabel("SHAP value (log-odds of destruction)")
    sm = plt.cm.ScalarMappable(cmap=PALETTES["shap_div"])
    sm.set_array(np.array([0.0, 1.0]))
    sm.set_clim(0, 1)
    _v_cbar(sm, fig.add_subplot(gs_a[0, 1]), "Feature value",
            ticks=[0, 1], ticklabels=["Low", "High"])
    full_frame(ax)
    panel_label(ax, "a", "SHAP summary")

    # (b) Building density
    gs_b = _with_cbar(gs[0, 2])
    ax = fig.add_subplot(gs_b[0, 0])
    p, color_by = "bld_count_r100", "nn_building_dist_m"
    fv = sample[f"val_{p}"].values
    sv = sample[f"shap_{p}"].values
    cv_ = sample[f"val_{color_by}"].values
    q5, q95 = np.nanpercentile(cv_, [5, 95])
    sc = ax.scatter(fv, sv, c=np.clip(cv_, q5, q95), cmap=dep_cmap,
                    s=3.2, alpha=0.55, linewidths=0)
    ax.axhline(0, color="#999999", linewidth=0.7)
    ax.set_xlabel("Buildings 100 m")
    ax.set_ylabel("SHAP value")
    ax.set_xlim(0, 100)
    _v_cbar(sc, fig.add_subplot(gs_b[0, 1]), "NN dist. (m)")
    full_frame(ax)
    panel_label(ax, "b", "Building density")

    # (c) Odds ratios — faded diverging bars + CI boxplots
    gs_c = _with_cbar(gs[1, 0])
    ax = fig.add_subplot(gs_c[0, 0])
    sig = lc.drop(index="const")
    sig = sig[sig["p"] < 0.05].sort_values("or")
    _fig07_draw_or_diverging(ax, sig)
    or_cmap = ListedColormap([COLORS["survived"], COLORS["destroyed"]])
    sm_or = plt.cm.ScalarMappable(cmap=or_cmap, norm=plt.Normalize(0, 1))
    sm_or.set_array(np.array([0.0, 1.0]))
    _v_cbar(sm_or, fig.add_subplot(gs_c[0, 1]), "Direction",
            ticks=[0.25, 0.75], ticklabels=["OR < 1", "OR > 1"])
    full_frame(ax)
    panel_label(ax, "c", "Odds ratios")

    # (d) Moisture
    gs_d = _with_cbar(gs[1, 2])
    ax = fig.add_subplot(gs_d[0, 0])
    p, color_by = "ndmi_pre_r100_300", "ndvi_pre_r30_100"
    fv = sample[f"val_{p}"].values
    sv = sample[f"shap_{p}"].values
    cv_ = sample[f"val_{color_by}"].values
    q5, q95 = np.nanpercentile(cv_, [5, 95])
    sc = ax.scatter(fv, sv, c=np.clip(cv_, q5, q95), cmap=dep_cmap,
                    s=3.2, alpha=0.55, linewidths=0)
    ax.axhline(0, color="#999999", linewidth=0.7)
    ax.set_xlabel("NDMI 100–300 m")
    ax.set_ylabel("SHAP value")
    _v_cbar(sc, fig.add_subplot(gs_d[0, 1]), "NDVI 30–100 m")
    full_frame(ax)
    panel_label(ax, "d", "Moisture")

    save_figure(fig, "Figure_07_interpretation")


# ------------------------------------------------------------------- Figure 8
def figure_08_sensitivity():
    """Blues ribbon palette; (a)/(c) use dotted band + solid dots (v04 style)."""
    from matplotlib.colors import to_rgba

    sc = pd.read_csv(path("outputs", "model_diagnostics", "scale_sensitivity.csv"))
    bs = pd.read_csv(path("outputs", "model_diagnostics", "blocksize_sensitivity.csv"))
    imp = pd.read_csv(path("outputs", "model_diagnostics", "shap_importance.csv"))

    c = COLORS["survived"]  # #2166AC
    ring_cols = ["#92C5DE", "#4393C3", "#2166AC"]

    fig = plt.figure(figsize=(W_DOUBLE, 7.0 * CM))
    gs = fig.add_gridspec(
        1, 3, left=0.065, right=0.985, top=0.86, bottom=0.16,
        wspace=0.48,
    )

    def _auc_dotband(ax, x, y, e, xticklabels, xlabel, marker="o"):
        ax.set_ylim(0.50, 0.90)
        ax.set_yticks(np.arange(0.50, 0.91, 0.10))
        ax.axhline(0.50, color=COLORS["neutral_light"], linewidth=0.85,
                   linestyle="--", zorder=1)
        ax.fill_between(x, y - e, y + e, color=c, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(x, y, color=c, linewidth=1.2, linestyle=":", zorder=3)
        for xi, yi, ei in zip(x, y, e):
            ax.plot([xi, xi], [yi - ei, yi + ei], color=c, linewidth=0.95,
                    alpha=0.85, zorder=3)
        ax.scatter(x, y, s=52, marker=marker, color=c, edgecolors="white",
                   linewidths=0.9, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=FS["size_small"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Spatial-CV ROC-AUC")
        ax.set_xlim(x[0] - 0.45, x[-1] + 0.45)
        full_frame(ax)

    # (a) Ring-scale AUC
    ax = fig.add_subplot(gs[0, 0])
    order = ["0_30", "30_100", "100_300", "all_scales"]
    sub = sc.set_index("scale").loc[order]
    x = np.arange(len(order), dtype=float)
    _auc_dotband(
        ax, x, sub["auc_mean"].to_numpy(), sub["auc_sd"].to_numpy(),
        ["0–30", "30–100", "100–300", "All"], "Ring scale (m)", marker="o",
    )
    panel_label(ax, "a", "Ring-scale AUC")

    # (b) SHAP by ring — horizontal blues bars (from v01)
    ax = fig.add_subplot(gs[0, 1])
    rings = [
        ("0–30 m", "_r0_30", "bld_count_r30"),
        ("30–100 m", "_r30_100", "bld_count_r100"),
        ("100–300 m", "_r100_300", "bld_count_r300"),
    ]
    vals, labs = [], []
    for lab, tag, bld in rings:
        mask = imp["predictor"].str.endswith(tag) | (imp["predictor"] == bld)
        vals.append(float(imp.loc[mask, "mean_abs_shap"].sum()))
        labs.append(lab)
    ypos = np.arange(len(vals))
    for y_i, v, col in zip(ypos, vals, ring_cols):
        ax.barh(y_i, v, height=0.58, color=to_rgba(col, 0.90), edgecolor="white",
                linewidth=0.7, zorder=2)
        ax.text(v + 0.06 * max(vals), y_i, f"{v:.2f}", va="center", ha="left",
                fontsize=FS["size_small"], color=COLORS["neutral_dark"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(labs, fontsize=FS["size_small"])
    ax.set_xlabel("Summed mean |SHAP|")
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_ylim(-0.55, len(vals) - 0.45)
    full_frame(ax)
    panel_label(ax, "b", "SHAP by ring")

    # (c) Block-size AUC
    ax = fig.add_subplot(gs[0, 2])
    bs = bs.sort_values("block_m")
    x = np.arange(len(bs), dtype=float)
    _auc_dotband(
        ax, x, bs["auc_mean"].to_numpy(), bs["auc_sd"].to_numpy(),
        [str(int(v)) for v in bs["block_m"]], "Spatial block size (m)",
        marker="s",
    )
    panel_label(ax, "c", "Block-size AUC")

    save_figure(fig, "Figure_08_sensitivity")


# ------------------------------------------------------------------- Figure 9
DAMAGE_XTICK = ["None", "Aff.", "Min.", "Maj.", "Dest."]


def _fig09_kde_1d(vals, grid, bw=0.14):
    from scipy.stats import gaussian_kde
    v = vals[np.isfinite(vals)]
    if len(v) < 8:
        return None
    dens = gaussian_kde(v, bw_method=bw)(grid)
    return dens / (dens.max() + 1e-12)


def _fig09_raincloud(ax, data, order, rng):
    """Fig. 05-style half-violin + tonal box + jittered points."""
    from matplotlib.patches import Rectangle

    for i, (vals, d) in enumerate(zip(data, order), start=1):
        c = DAMAGE_COLORS[d]
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 8:
            continue
        lo, hi = np.nanpercentile(v, [1, 99])
        grid = np.linspace(lo, hi, 160)
        dens = _fig09_kde_1d(v, grid, bw=0.14)
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
            edgecolor=COLORS["neutral_dark"], linewidth=0.7, zorder=5, alpha=1.0,
        ))
        ax.plot([i - 0.07, i + 0.07], [med, med], color=COLORS["neutral_dark"],
                linewidth=1.2, zorder=6, solid_capstyle="butt")


def figure_09_severity_recovery():
    study, per = _study_per()
    feats = pd.read_parquet(path("data", "processed", "structure_features.parquet"))
    res = feats[feats["residential"] == 1]
    rec = pd.read_csv(path("outputs", "tables", "recovery_trajectories.csv"))
    order = list(DAMAGE_COLORS.keys())

    fig = plt.figure(figsize=(W_DOUBLE, 13.6 * CM))
    gs = fig.add_gridspec(
        2, 2, left=0.08, right=0.97, top=0.94, bottom=0.08,
        height_ratios=[1.15, 1.0], hspace=0.34, wspace=0.42,
        width_ratios=[1.0, 1.08],
    )
    # Colorbar tight to map; extra column wspace keeps it clear of panel (b)
    gs_a = gs[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.048], wspace=0.03)

    # (a) dNBR map — same GIS frame language as Figs 01–06
    ax = fig.add_subplot(gs_a[0, 0])
    ax.set_label("map")
    hillshade_background(ax, alpha=0.55)
    with rasterio.open(path("data", "interim", "severity", "dnbr.tif")) as src:
        arr = src.read(1).astype("float64")
        b = src.bounds
        nod = src.nodata
    if nod is not None:
        arr[arr == nod] = np.nan
    arr[arr < -2000] = np.nan
    ext = [b.left, b.right, b.bottom, b.top]
    im = ax.imshow(arr, cmap=PALETTES["dnbr_div"], vmin=-100, vmax=800,
                   extent=ext, interpolation="bilinear", zorder=2, aspect="auto")
    ocean_overlay(ax)
    per.plot(ax=ax, facecolor="none", edgecolor=COLORS["perimeter"],
             linewidth=1.0, zorder=5)
    set_extent(ax, study, pad=400)
    clip_map_layers(ax)
    map_frame(ax, crs=CRS, gis_frame=True)
    add_scalebar(ax, 4.0)
    cover_frame_overflow(fig, ax)
    panel_label(ax, "a", "Burn severity (dNBR)")

    cax = fig.add_subplot(gs_a[0, 1])
    cax.set_label("colorbar")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("dNBR (×1000)", size=FS["size_label"], labelpad=2)
    cb.ax.tick_params(labelsize=FS["size_small"], length=2.5, width=0.6,
                      direction="in", pad=1.5)
    cb.outline.set_linewidth(0.7)
    cb.outline.set_edgecolor(COLORS["neutral_dark"])

    # (b) Raincloud dNBR by DINS damage (Fig. 05 style)
    ax = fig.add_subplot(gs[0, 1])
    data = [res.loc[res["damage"] == d, "dnbr_r30_100"].dropna().to_numpy()
            for d in order]
    rng = np.random.default_rng(42)
    _fig09_raincloud(ax, data, order, rng)
    vals = np.concatenate([d for d in data if len(d)])
    lo, hi = np.nanpercentile(vals, [1, 99])
    pad = 0.06 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(DAMAGE_SHORT, fontsize=FS["size_small"],
                       rotation=28, ha="right")
    ax.set_xlim(0.4, 5.6)
    ax.set_ylabel("dNBR (×1000), 30–100 m")
    full_frame(ax)
    panel_label(ax, "b", "dNBR by damage")

    # (c) Early NDVI recovery trajectories
    ax = fig.add_subplot(gs[1, :])
    sev_names = {1: "Low severity", 2: "Moderate severity", 3: "High severity"}
    sev_colors = {1: "#4C7F3F", 2: COLORS["accent"], 3: COLORS["destroyed"]}
    clim = rec[rec["period"].str.startswith("clim")].copy()
    clim["month"] = clim["period"].str[-2:].astype(int)
    mon = rec[~rec["period"].str.startswith("clim")].copy()
    mon["date"] = pd.to_datetime(mon["period"] + "-15")
    mon["month"] = mon["date"].dt.month
    for s, name in sev_names.items():
        m = mon[mon["severity"] == s].sort_values("date")
        c = clim[clim["severity"] == s].set_index("month")["ndvi"]
        ratio = m["ndvi"].values / m["month"].map(c).values
        ax.plot(m["date"], ratio, marker="o", markersize=4.2, linewidth=1.25,
                color=sev_colors[s], label=name, markerfacecolor="white",
                markeredgewidth=1.05, zorder=3)
    ax.axhline(1.0, color=COLORS["neutral_dark"], linewidth=0.95, linestyle="--",
               label="2022–2024 seasonal norm", zorder=2)
    ax.set_ylabel("NDVI relative to month-matched\n2022–2024 climatology")
    ax.set_xlabel("Date")
    ax.legend(fontsize=FS["size_small"], loc="lower right",
              frameon=True, edgecolor="#BDBDBD", fancybox=False, ncol=2,
              labelspacing=0.35, handlelength=1.8, borderpad=0.45)
    full_frame(ax)
    panel_label(ax, "c", "Early NDVI recovery")

    save_figure(fig, "Figure_09_severity_recovery")


# ------------------------------------------------------------------- Figure 10
def figure_10_community():
    from scipy.stats import spearmanr
    from shapely.ops import unary_union

    ctx = gpd.read_file(path("data", "processed", "tract_context.gpkg"),
                        layer="tracts").to_crs(CRS)
    study, per = _study_per()

    clip = unary_union(per.geometry)
    ctx_in = ctx.copy()
    ctx_in["geometry"] = ctx_in.geometry.intersection(clip)
    ctx_in = ctx_in[~ctx_in.geometry.is_empty].copy()
    covered = unary_union(ctx_in.geometry) if len(ctx_in) else None
    holes = clip.difference(covered) if covered is not None else clip
    hole_gdf = None
    if (holes is not None) and (not holes.is_empty):
        hole_gdf = gpd.GeoDataFrame(geometry=[holes], crs=CRS)

    fig = plt.figure(figsize=(W_DOUBLE, 13.8 * CM))
    gs = fig.add_gridspec(
        2, 2, left=0.08, right=0.97, top=0.94, bottom=0.08,
        height_ratios=[1.18, 1.0], hspace=0.32, wspace=0.22,
        width_ratios=[1.0, 1.0],
    )

    def _map_panel(spec, gdf, column, cmap, label, letter, title, ylabels=True,
                   vmin=0.0, vmax=1.0):
        gs_m = spec.subgridspec(1, 2, width_ratios=[1.0, 0.048], wspace=0.03)
        ax = fig.add_subplot(gs_m[0, 0])
        ax.set_label("map")
        hillshade_background(ax, alpha=0.60)
        ocean_overlay(ax)
        if hole_gdf is not None:
            hole_gdf.plot(ax=ax, facecolor="#F0F0F0", edgecolor="none", zorder=4)
        n0 = len(ax.collections)
        gdf.plot(
            ax=ax, column=column, cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolor="white", linewidth=0.35, legend=False, zorder=5,
            missing_kwds={"color": "#EEEEEE", "edgecolor": "white"},
        )
        mapped = ax.collections[n0]
        per.plot(ax=ax, facecolor="none", edgecolor=COLORS["perimeter"],
                 linewidth=1.05, zorder=6)
        set_extent(ax, study, pad=400)
        clip_map_layers(ax)
        map_frame(ax, crs=CRS, ylabels=ylabels, gis_frame=True)
        add_scalebar(ax, 4.0)
        cover_frame_overflow(fig, ax)
        panel_label(ax, letter, title)

        cax = fig.add_subplot(gs_m[0, 1])
        cax.set_label("colorbar")
        cb = fig.colorbar(mapped, cax=cax)
        cb.set_label(label, size=FS["size_label"], labelpad=2)
        cb.ax.tick_params(labelsize=FS["size_small"], length=2.5, width=0.6,
                          direction="in", pad=1.5)
        cb.outline.set_linewidth(0.7)
        cb.outline.set_edgecolor(COLORS["neutral_dark"])

    svi_vmax = float(np.nanpercentile(ctx_in["svi_overall"].dropna(), 98))
    svi_vmax = max(svi_vmax, 0.25)
    _map_panel(gs[0, 0], ctx_in, "destroyed_rate", PALETTES["destruction_seq"],
               "Destroyed fraction", "a", "Residential destruction", ylabels=True)
    _map_panel(gs[0, 1], ctx_in, "svi_overall", PALETTES["svi_seq"],
               "SVI percentile (2022)", "b", "Social vulnerability", ylabels=False,
               vmin=0.0, vmax=svi_vmax)

    # (c) Tract-level association — bubble: size = n, color = fire-station distance
    ax = fig.add_subplot(gs[1, :])
    ok = ctx.dropna(subset=["svi_overall", "destroyed_rate"]).copy()
    sc = ax.scatter(
        ok["svi_overall"], ok["destroyed_rate"],
        s=np.sqrt(ok["n_inspected"]) * 2.4,
        c=ok["med_dist_fire_m"] / 1000, cmap=PALETTES["access_seq"],
        edgecolor=COLORS["neutral_dark"], linewidth=0.45, alpha=0.92, zorder=3,
    )
    rho, pval = spearmanr(ok["svi_overall"], ok["destroyed_rate"])
    ax.set_xlabel("SVI overall percentile")
    ax.set_ylabel("Destroyed fraction")
    ax.text(
        0.02, 0.95, f"Spearman rho = {rho:.2f}  (p = {pval:.2f})",
        transform=ax.transAxes, fontsize=FS["size_small"], va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="#BDBDBD", boxstyle="square,pad=0.3",
                  linewidth=0.6), zorder=6,
    )
    for n, lab in ((100, "100"), (500, "500"), (2000, "2,000")):
        ax.scatter([], [], s=np.sqrt(n) * 2.4, c="#BDBDBD",
                   edgecolor=COLORS["neutral_dark"], linewidth=0.45, label=lab)
    ax.legend(title="Inspected structures", fontsize=FS["size_small"],
              title_fontsize=FS["size_small"], loc="upper right",
              frameon=True, edgecolor="#BDBDBD", fancybox=False,
              borderpad=0.45, labelspacing=0.45, handletextpad=0.5)
    ax.set_xlim(ok["svi_overall"].min() - 0.02, ok["svi_overall"].max() + 0.02)
    ax.set_ylim(-0.03, 1.03)
    add_cbar(sc, ax, "Dist. to fire station (km)")
    full_frame(ax)
    panel_label(ax, "c", "SVI vs destruction")

    save_figure(fig, "Figure_10_community")


if __name__ == "__main__":
    apply_style()
    figure_06_validation()
    figure_07_interpretation()
    figure_08_sensitivity()
    figure_09_severity_recovery()
    figure_10_community()
    log.info("results figures complete")
