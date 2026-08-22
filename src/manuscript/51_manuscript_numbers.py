"""Collect every quantitative value cited in the manuscript into one JSON file.

Ensures all reported numbers trace to pipeline outputs (no fabrication).
Output: manuscript/manuscript_numbers.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, path

log = get_logger("51_numbers")


def main() -> None:
    N: dict = {}

    audit = json.loads(Path(path("outputs", "qa", "dins_audit.json")).read_text())
    N["dins"] = audit

    res_summary = pd.read_parquet(path("data", "processed", "structure_features.parquet"))
    res = res_summary[res_summary["residential"] == 1]
    N["sample"] = {
        "n_residential": int(len(res)),
        "n_destroyed": int(res["destroyed"].sum()),
        "destroyed_rate": round(float(res["destroyed"].mean()), 4),
        "median_yearbuilt": float(res["yearbuilt"].median()),
        "median_nn_dist": round(float(res["nn_building_dist_m"].median()), 1),
        "median_bld_r100": float(res["bld_count_r100"].median()),
    }

    cv = pd.read_csv(path("outputs", "model_diagnostics", "cv_metrics.csv"))
    perf = {}
    for (scheme, block, learner), g in cv.groupby(["cv_scheme", "model_block", "learner"]):
        perf[f"{scheme}_{block}_{learner}"] = {
            "auc": round(g["roc_auc"].mean(), 3), "auc_sd": round(g["roc_auc"].std(), 3),
            "prauc": round(g["pr_auc"].mean(), 3),
            "bal_acc": round(g["balanced_accuracy"].mean(), 3),
            "f1": round(g["f1"].mean(), 3),
            "brier": round(g["brier"].mean(), 3),
        }
    N["performance"] = perf
    m3s = cv[(cv.model_block == "M3") & (cv.cv_scheme == "spatial")]
    N["calibration_m3_spatial"] = {
        ln: {"slope_mean": round(g["cal_slope"].mean(), 2),
             "slope_range": [round(g["cal_slope"].min(), 2), round(g["cal_slope"].max(), 2)],
             "intercept_mean": round(g["cal_intercept"].mean(), 2)}
        for ln, g in m3s.groupby("learner")
    }

    lc = pd.read_csv(path("outputs", "model_diagnostics", "logit_coefficients.csv"),
                     index_col=0).drop(index="const")
    ors = {}
    for v in ["bld_count_r100", "ndmi_pre_r100_300", "ndmi_pre_r30_100",
              "ndvi_pre_r30_100", "tpi300_pt", "yearbuilt", "dist_wildland_m",
              "dist_deadend_m", "elevation_pt", "eastness_pt", "bld_count_r30",
              "ndmi_anom_r100_300", "canopy_cover_r0_30", "assessed_value_log"]:
        r = lc.loc[v]
        ors[v] = {"or": round(r["or"], 2), "lo": round(r["or_lo"], 2),
                  "hi": round(r["or_hi"], 2), "p": float(r["p"])}
    N["odds_ratios"] = ors

    shp = pd.read_csv(path("outputs", "model_diagnostics", "shap_importance.csv"))
    N["shap_top10"] = shp.head(10).round(3).to_dict("records")

    for f, key in [("scale_sensitivity.csv", "scale"),
                   ("blocksize_sensitivity.csv", "blocksize"),
                   ("outcome_sensitivity.csv", "outcome"),
                   ("residual_morans.csv", "residual_morans")]:
        N[key] = pd.read_csv(path("outputs", "model_diagnostics", f)).round(3).to_dict("records")

    N["morans"] = pd.read_csv(path("outputs", "qa", "morans_i.csv")).round(3).to_dict("records")

    N["dnbr_by_class"] = pd.read_csv(
        path("outputs", "tables", "dnbr_by_damage_class.csv")).round(1).to_dict("records")
    N["severity_fractions"] = pd.read_csv(
        path("outputs", "tables", "severity_area_fractions.csv")).round(3).to_dict("records")

    rec = pd.read_csv(path("outputs", "tables", "recovery_trajectories.csv"))
    clim = rec[rec["period"].str.startswith("clim")].copy()
    clim["month"] = clim["period"].str[-2:].astype(int)
    mon = rec[~rec["period"].str.startswith("clim")].copy()
    mon["month"] = pd.to_datetime(mon["period"] + "-15").dt.month
    ratios = {}
    for s in (1, 2, 3):
        m = mon[mon["severity"] == s]
        c = clim[clim["severity"] == s].set_index("month")["ndvi"]
        r = m["ndvi"].values / m["month"].map(c).values
        ratios[str(s)] = {"min_ratio": round(float(np.nanmin(r)), 2),
                          "first_month_ratio": round(float(r[0]), 2),
                          "last_month_ratio": round(float(r[-1]), 2),
                          "max_ratio": round(float(np.nanmax(r)), 2)}
    N["recovery_ratios"] = ratios

    gm_hist = pd.read_csv(path("data", "raw", "climate", "gridmet_octdec_precip_1980_2024.csv"))
    N["climate"] = {
        "octdec_2024_mm": round(float(gm_hist.loc[gm_hist.year == 2024, "octdec_pr_mm"].iloc[0]), 1),
        "octdec_median_mm": round(float(gm_hist["octdec_pr_mm"].median()), 1),
        "octdec_2024_rank": int((gm_hist["octdec_pr_mm"] <
                                 gm_hist.loc[gm_hist.year == 2024, "octdec_pr_mm"].iloc[0]).sum() + 1),
        "n_years": int(len(gm_hist)),
    }
    e5 = pd.read_csv(path("data", "raw", "climate", "era5land_hourly_event.csv"))
    ws = np.hypot(e5["u_component_of_wind_10m"], e5["v_component_of_wind_10m"])
    t = e5["temperature_2m"] - 273.15
    td = e5["dewpoint_temperature_2m"] - 273.15
    rh = 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
    N["event_weather"] = {"peak_wind_ms": round(float(ws.max()), 1),
                          "min_rh_pct": round(float(rh.min()), 1)}

    ctx = pd.read_csv(path("outputs", "tables", "tract_context.csv"))
    from scipy.stats import spearmanr
    ok = ctx.dropna(subset=["svi_overall", "destroyed_rate"])
    rho, p = spearmanr(ok["svi_overall"], ok["destroyed_rate"])
    N["community"] = {
        "n_tracts": int(len(ctx)),
        "svi_max": round(float(ctx["svi_overall"].max()), 2),
        "spearman_svi_destroyed": {"rho": round(float(rho), 2), "p": round(float(p), 2),
                                   "n": int(len(ok))},
        "med_dist_fire_km_range": [round(float(ctx["med_dist_fire_m"].min() / 1000), 1),
                                   round(float(ctx["med_dist_fire_m"].max() / 1000), 1)],
    }

    scenes = pd.read_csv(path("data", "metadata", "s2_scene_selection.csv"))
    N["scenes"] = scenes.groupby("window")["scene_id"].count().to_dict()

    eda = json.loads(Path(path("outputs", "qa", "eda_notes.json")).read_text())
    N["eda"] = {"n_high_corr_pairs": len(eda["high_corr_pairs"]),
                "n_vif_over_10": len(eda["vif_over_10"])}

    out = path("manuscript", "manuscript_numbers.json")
    out.write_text(json.dumps(N, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
