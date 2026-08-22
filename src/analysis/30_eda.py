"""Exploratory analysis before modeling.

Outputs (outputs/qa + outputs/tables):
  - predictor_summary_by_class.csv  (means/medians by destroyed vs survived)
  - correlation_matrix.csv          (Spearman, all candidate predictors)
  - vif.csv                         (variance inflation factors, primary set)
  - morans_i.csv                    (outcome + key predictors)
  - eda_notes.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, load_config, path
from analysis.predictor_sets import PRE_FIRE_PREDICTORS, primary_model_frame

log = get_logger("30_eda")


def main() -> None:
    df, X, y = primary_model_frame()
    log.info("primary frame: %d residential structures, %d predictors, destroyed rate %.3f",
             len(df), X.shape[1], y.mean())

    # ---- summary by class ------------------------------------------------------
    summ = df.groupby("destroyed")[PRE_FIRE_PREDICTORS].agg(["mean", "median", "std"]).T
    summ.to_csv(path("outputs", "qa", "predictor_summary_by_class.csv"))

    # ---- Spearman correlations ---------------------------------------------------
    corr = X.corr(method="spearman")
    corr.to_csv(path("outputs", "qa", "correlation_matrix.csv"))
    hi = (
        corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
        .stack().rename("rho").reset_index()
    )
    hi = hi[hi["rho"].abs() > 0.85].sort_values("rho", key=abs, ascending=False)
    log.info("pairs with |rho|>0.85: %d", len(hi))

    # ---- VIF ------------------------------------------------------------------------
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xz = (X - X.mean()) / X.std()
    Xz = Xz.fillna(0.0)
    Xz.insert(0, "const", 1.0)
    vifs = []
    for i, c in enumerate(Xz.columns):
        if c == "const":
            continue
        vifs.append({"variable": c, "vif": variance_inflation_factor(Xz.values, i)})
    vif = pd.DataFrame(vifs).sort_values("vif", ascending=False)
    vif.to_csv(path("outputs", "qa", "vif.csv"), index=False)
    log.info("VIF>10: %d variables", (vif["vif"] > 10).sum())

    # ---- Moran's I -----------------------------------------------------------------
    from libpysal.weights import KNN
    from esda.moran import Moran
    coords = df[["x", "y"]].values
    w = KNN.from_array(coords, k=8)
    w.transform = "r"
    rows = []
    for col in ["destroyed", "ndvi_pre_r30_100", "ndmi_pre_r30_100",
                "bld_count_r100", "slope_deg_pt", "yearbuilt", "dist_wildland_m"]:
        v = df[col].astype(float).values
        v = np.where(np.isfinite(v), v, np.nanmedian(v))
        m = Moran(v, w, permutations=199)
        rows.append({"variable": col, "morans_i": m.I, "p_sim": m.p_sim})
        log.info("Moran's I %s: %.3f (p=%.3f)", col, m.I, m.p_sim)
    pd.DataFrame(rows).to_csv(path("outputs", "qa", "morans_i.csv"), index=False)

    notes = {
        "n_residential": int(len(df)),
        "destroyed_rate": float(y.mean()),
        "high_corr_pairs": hi.to_dict("records"),
        "vif_over_10": vif[vif["vif"] > 10].to_dict("records"),
    }
    path("outputs", "qa", "eda_notes.json").write_text(
        json.dumps(notes, indent=2, default=str), encoding="utf-8")
    log.info("EDA complete")


if __name__ == "__main__":
    main()
