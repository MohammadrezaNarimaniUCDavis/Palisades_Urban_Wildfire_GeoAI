"""Fit and validate the structure-damage susceptibility models.

Design (decision D-011):
  - Nested predictor blocks M0 (terrain), M1 (+vegetation/fuels),
    M2 (+built environment), M3 (integrated).
  - Learners: L2 logistic regression (pruned predictor set) and XGBoost.
  - Validation: 5-fold SPATIAL BLOCK CV (1 km blocks randomly assigned to
    folds, seeded) vs. plain random 5-fold CV to quantify optimism.
  - All preprocessing (imputation, scaling) and hyperparameter tuning happen
    INSIDE training folds (inner 3-fold grouped CV).
  - Class weights for the mild imbalance (56/44).

Outputs (outputs/model_diagnostics/):
  - cv_metrics.csv           per model x block x CV-scheme x fold
  - oof_predictions.parquet  out-of-fold probabilities (spatial CV, all models)
  - best_params.json         chosen hyperparameters per fold
  - calibration_data.csv     reliability-curve bins (spatial CV, M3 models)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, load_config, path
from analysis.predictor_sets import LOGIT_PREDICTORS, MODEL_BLOCKS, primary_model_frame

log = get_logger("31_fit_models")
cfg = load_config()
SEED = cfg["project"]["random_seed"]
rng = np.random.default_rng(SEED)


def spatial_blocks(df: pd.DataFrame, size_m: float) -> pd.Series:
    bx = np.floor(df["x"] / size_m).astype(int)
    by = np.floor(df["y"] / size_m).astype(int)
    return bx.astype(str) + "_" + by.astype(str)


def blocks_to_folds(blocks: pd.Series, k: int, seed: int) -> pd.Series:
    ids = blocks.unique()
    r = np.random.default_rng(seed)
    perm = r.permutation(len(ids))
    assign = {b: int(perm[i] % k) for i, b in enumerate(ids)}
    return blocks.map(assign)


def make_logit() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                   solver="lbfgs")),
    ])


def make_xgb(scale_pos_weight: float) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=SEED, n_jobs=8,
            scale_pos_weight=scale_pos_weight,
        )),
    ])


def tune_and_fit(model_kind: str, Xtr: pd.DataFrame, ytr: pd.Series,
                 groups_tr: pd.Series) -> tuple[Pipeline, dict]:
    """Inner 3-fold grouped CV for hyperparameter selection; refit on full train."""
    inner = GroupKFold(n_splits=3)
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    if model_kind == "logit":
        grid = [{"clf__C": c} for c in cfg["model"]["logistic_C"]]
        base = make_logit()
    else:
        g = cfg["model"]["xgb_params_grid"]
        grid = [
            {"clf__max_depth": d, "clf__learning_rate": lr,
             "clf__n_estimators": n, "clf__subsample": g["subsample"][0],
             "clf__colsample_bytree": g["colsample_bytree"][0],
             "clf__min_child_weight": g["min_child_weight"][0]}
            for d in g["max_depth"] for lr in g["learning_rate"]
            for n in g["n_estimators"]
        ]
        base = make_xgb(spw)

    best_score, best_params = -np.inf, grid[0]
    for params in grid:
        scores = []
        for tr_i, va_i in inner.split(Xtr, ytr, groups_tr):
            m = base.set_params(**params)
            m.fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
            p = m.predict_proba(Xtr.iloc[va_i])[:, 1]
            scores.append(roc_auc_score(ytr.iloc[va_i], p))
        s = float(np.mean(scores))
        if s > best_score:
            best_score, best_params = s, params

    final = base.set_params(**best_params)
    final.fit(Xtr, ytr)
    return final, {**best_params, "inner_auc": best_score}


def metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    yhat = (p >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "balanced_accuracy": balanced_accuracy_score(y_true, yhat),
        "f1": f1_score(y_true, yhat),
        "brier": brier_score_loss(y_true, p),
        "n_test": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
    }


def calibration_stats(y_true: np.ndarray, p: np.ndarray) -> dict:
    """Calibration intercept and slope via logistic recalibration on logit(p)."""
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    import statsmodels.api as sm
    X = sm.add_constant(lp)
    fit = sm.GLM(y_true, X, family=sm.families.Binomial()).fit()
    return {"cal_intercept": float(fit.params[0]), "cal_slope": float(fit.params[1])}


def run_cv(df: pd.DataFrame, y: pd.Series, fold_series: pd.Series,
           scheme: str, results: list, oof_store: dict) -> None:
    blocks = spatial_blocks(df, cfg["model"]["spatial_block_size_m"])
    for block_name, cols in MODEL_BLOCKS.items():
        for model_kind in ("logit", "xgb"):
            use_cols = [c for c in cols if model_kind == "xgb" or c in LOGIT_PREDICTORS]
            X = df[use_cols].astype(float)
            oof = np.full(len(df), np.nan)
            for fold in sorted(fold_series.unique()):
                te = fold_series == fold
                tr = ~te
                model, params = tune_and_fit(model_kind, X[tr], y[tr], blocks[tr])
                p = model.predict_proba(X[te])[:, 1]
                oof[te.values] = p
                m = metrics(y[te].values, p)
                m.update({"model_block": block_name, "learner": model_kind,
                          "cv_scheme": scheme, "fold": int(fold),
                          "n_predictors": len(use_cols)})
                if block_name == "M3":
                    m.update(calibration_stats(y[te].values, p))
                results.append(m)
                log.info("%s %s %s fold %d: AUC=%.3f PR=%.3f Brier=%.3f",
                         scheme, block_name, model_kind, fold,
                         m["roc_auc"], m["pr_auc"], m["brier"])
            oof_store[f"{scheme}_{block_name}_{model_kind}"] = oof


def main() -> None:
    df, _, y = primary_model_frame()
    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)

    blocks = spatial_blocks(df, cfg["model"]["spatial_block_size_m"])
    log.info("n=%d structures in %d blocks of %dm", len(df), blocks.nunique(),
             cfg["model"]["spatial_block_size_m"])

    spatial_folds = blocks_to_folds(blocks, cfg["model"]["n_folds"], SEED)
    random_folds = pd.Series(
        np.random.default_rng(SEED).integers(0, cfg["model"]["n_folds"], len(df)),
        index=df.index,
    )

    results: list[dict] = []
    oof_store: dict[str, np.ndarray] = {}
    run_cv(df, y, spatial_folds, "spatial", results, oof_store)
    run_cv(df, y, random_folds, "random", results, oof_store)

    res = pd.DataFrame(results)
    res.to_csv(path("outputs", "model_diagnostics", "cv_metrics.csv"), index=False)

    oof = pd.DataFrame({"globalid": df["globalid"], "x": df["x"], "y": df["y"],
                        "destroyed": y, "spatial_fold": spatial_folds.values,
                        **oof_store})
    oof.to_parquet(path("outputs", "model_diagnostics", "oof_predictions.parquet"),
                   index=False)

    # summary
    summ = (res.groupby(["cv_scheme", "model_block", "learner"])
            [["roc_auc", "pr_auc", "balanced_accuracy", "f1", "brier"]]
            .agg(["mean", "std"]).round(4))
    summ.to_csv(path("outputs", "model_diagnostics", "cv_summary.csv"))
    log.info("summary:\n%s", summ.to_string())

    # calibration reliability bins for M3 spatial models
    from sklearn.calibration import calibration_curve
    rows = []
    for learner in ("logit", "xgb"):
        p = oof[f"spatial_M3_{learner}"].values
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
        for f_, m_ in zip(frac, mean_p):
            rows.append({"learner": learner, "mean_pred": m_, "frac_pos": f_})
    pd.DataFrame(rows).to_csv(
        path("outputs", "model_diagnostics", "calibration_data.csv"), index=False)

    log.info("modeling complete")


if __name__ == "__main__":
    main()
