"""Model interpretation and sensitivity analyses.

Outputs (outputs/model_diagnostics/):
  - logit_coefficients.csv     standardized coefficients, odds ratios, 95% CI
  - shap_importance.csv        mean |SHAP| per predictor (XGB M3, full refit)
  - shap_sample.parquet        SHAP values + feature values (sample for figures)
  - scale_sensitivity.csv      spatial-CV AUC with ring-restricted vegetation sets
  - blocksize_sensitivity.csv  spatial-CV AUC at 500/1000/2000 m blocks
  - outcome_sensitivity.csv    alternative outcome codings / populations
  - residual_morans.csv        Moran's I of out-of-fold residuals
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, load_config, path
from analysis.predictor_sets import (LOGIT_PREDICTORS, M0_TERRAIN, M2_BUILT,
                                     MODEL_BLOCKS, PRE_FIRE_PREDICTORS,
                                     load_features, primary_model_frame)

log = get_logger("32_interpret")
cfg = load_config()
SEED = cfg["project"]["random_seed"]

XGB_FIXED = dict(max_depth=4, learning_rate=0.1, n_estimators=600, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                 objective="binary:logistic", eval_metric="logloss",
                 random_state=SEED, n_jobs=8)


def spatial_blocks(df: pd.DataFrame, size_m: float) -> pd.Series:
    return (np.floor(df["x"] / size_m).astype(int).astype(str) + "_" +
            np.floor(df["y"] / size_m).astype(int).astype(str))


def blocks_to_folds(blocks: pd.Series, k: int, seed: int) -> pd.Series:
    ids = blocks.unique()
    perm = np.random.default_rng(seed).permutation(len(ids))
    assign = {b: int(perm[i] % k) for i, b in enumerate(ids)}
    return blocks.map(assign)


def spatial_cv_auc(df: pd.DataFrame, y: pd.Series, cols: list[str],
                   block_m: float, seed: int = SEED) -> dict:
    folds = blocks_to_folds(spatial_blocks(df, block_m), cfg["model"]["n_folds"], seed)
    X = df[cols].astype(float)
    aucs, praucs = [], []
    for f in sorted(folds.unique()):
        te = folds == f
        spw = float((y[~te] == 0).sum() / max((y[~te] == 1).sum(), 1))
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[~te])
        Xte = imp.transform(X[te])
        m = XGBClassifier(**XGB_FIXED, scale_pos_weight=spw)
        m.fit(Xtr, y[~te])
        p = m.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        praucs.append(average_precision_score(y[te], p))
    return {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
            "prauc_mean": float(np.mean(praucs)), "n_predictors": len(cols)}


def main() -> None:
    df, X, y = primary_model_frame()
    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # ---- logistic coefficients (statsmodels, standardized) ---------------------
    import statsmodels.api as sm
    Xl = df[LOGIT_PREDICTORS].astype(float)
    med = Xl.median()
    Xl = Xl.fillna(med)
    Xz = (Xl - Xl.mean()) / Xl.std()
    Xz = sm.add_constant(Xz)
    fit = sm.Logit(y, Xz).fit(disp=0, maxiter=200)
    coefs = pd.DataFrame({
        "coef": fit.params, "se": fit.bse, "p": fit.pvalues,
        "or": np.exp(fit.params),
        "or_lo": np.exp(fit.params - 1.96 * fit.bse),
        "or_hi": np.exp(fit.params + 1.96 * fit.bse),
    })
    coefs.to_csv(path("outputs", "model_diagnostics", "logit_coefficients.csv"))
    log.info("logit fit: n=%d, pseudo-R2=%.3f", int(fit.nobs), fit.prsquared)

    # ---- SHAP on full-data XGB M3 -------------------------------------------------
    import shap
    spw = float((y == 0).sum() / (y == 1).sum())
    imp = SimpleImputer(strategy="median")
    Xi = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    xgb = XGBClassifier(**XGB_FIXED, scale_pos_weight=spw)
    xgb.fit(Xi, y)
    explainer = shap.TreeExplainer(xgb)
    sv = explainer.shap_values(Xi)
    imp_df = pd.DataFrame({
        "predictor": X.columns,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(path("outputs", "model_diagnostics", "shap_importance.csv"), index=False)
    log.info("top SHAP:\n%s", imp_df.head(10).to_string(index=False))

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xi), size=min(3000, len(Xi)), replace=False)
    sample = pd.concat([
        Xi.iloc[idx].add_prefix("val_"),
        pd.DataFrame(sv[idx], columns=[f"shap_{c}" for c in X.columns]).set_index(Xi.iloc[idx].index),
    ], axis=1)
    sample["destroyed"] = y.iloc[idx].values
    sample.to_parquet(path("outputs", "model_diagnostics", "shap_sample.parquet"))

    # ---- scale sensitivity ----------------------------------------------------------
    log.info("scale sensitivity ...")
    ring_tags = {"0_30": "r0_30", "30_100": "r30_100", "100_300": "r100_300"}
    veg_prefixes = ("ndvi_", "ndmi_", "lst_", "canopy_", "fuel_", "bld_count")
    rows = []
    for label, tag in ring_tags.items():
        cols = [c for c in PRE_FIRE_PREDICTORS
                if not c.startswith(veg_prefixes)
                or tag in c or c.endswith(f"_r{label.split('_')[1]}")]
        # ensure bld_count matched properly (bld_count_r30 etc.)
        cols = [c for c in cols if not c.startswith("bld_count") or
                c == f"bld_count_r{label.split('_')[1]}"]
        r = spatial_cv_auc(df, y, sorted(set(cols)), cfg["model"]["spatial_block_size_m"])
        r["scale"] = label
        rows.append(r)
        log.info("scale %s: AUC=%.3f (sd %.3f)", label, r["auc_mean"], r["auc_sd"])
    r = spatial_cv_auc(df, y, PRE_FIRE_PREDICTORS, cfg["model"]["spatial_block_size_m"])
    r["scale"] = "all_scales"
    rows.append(r)
    pd.DataFrame(rows).to_csv(
        path("outputs", "model_diagnostics", "scale_sensitivity.csv"), index=False)

    # ---- block size sensitivity --------------------------------------------------------
    log.info("block-size sensitivity ...")
    rows = []
    for bs in (500, 1000, 2000):
        r = spatial_cv_auc(df, y, PRE_FIRE_PREDICTORS, bs)
        r["block_m"] = bs
        rows.append(r)
        log.info("block %dm: AUC=%.3f (sd %.3f)", bs, r["auc_mean"], r["auc_sd"])
    pd.DataFrame(rows).to_csv(
        path("outputs", "model_diagnostics", "blocksize_sensitivity.csv"), index=False)

    # ---- outcome / population sensitivity ------------------------------------------------
    log.info("outcome sensitivity ...")
    rows = []
    # destroyed + major vs rest (residential)
    y2 = ((df["damage"] == "Destroyed (>50%)") | (df["damage"] == "Major (25-50%)")).astype(int)
    r = spatial_cv_auc(df, y2, PRE_FIRE_PREDICTORS, cfg["model"]["spatial_block_size_m"])
    r["variant"] = "destroyed_or_major_residential"
    rows.append(r)
    # all structure categories
    df_all = load_features()
    df_all["assessed_value_log"] = np.log10(df_all["assessed_value"].clip(lower=1))
    df_all = df_all.reset_index(drop=True)
    y_all = df_all["destroyed"].astype(int)
    r = spatial_cv_auc(df_all, y_all, PRE_FIRE_PREDICTORS, cfg["model"]["spatial_block_size_m"])
    r["variant"] = "destroyed_all_structures"
    rows.append(r)
    pd.DataFrame(rows).to_csv(
        path("outputs", "model_diagnostics", "outcome_sensitivity.csv"), index=False)

    # ---- Moran's I of OOF residuals ----------------------------------------------------------
    log.info("residual spatial autocorrelation ...")
    from libpysal.weights import KNN
    from esda.moran import Moran
    oof = pd.read_parquet(path("outputs", "model_diagnostics", "oof_predictions.parquet"))
    w = KNN.from_array(oof[["x", "y"]].values, k=8)
    w.transform = "r"
    rows = []
    for col in ["spatial_M3_xgb", "spatial_M3_logit", "spatial_M0_xgb"]:
        resid = oof["destroyed"].values - oof[col].values
        m = Moran(np.nan_to_num(resid), w, permutations=199)
        rows.append({"model": col, "morans_i": m.I, "p_sim": m.p_sim})
        log.info("Moran's I resid %s: %.3f (p=%.3f)", col, m.I, m.p_sim)
    pd.DataFrame(rows).to_csv(
        path("outputs", "model_diagnostics", "residual_morans.csv"), index=False)

    log.info("interpretation complete")


if __name__ == "__main__":
    main()
