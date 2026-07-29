!pip install optuna
import os
import warnings
from collections import Counter
from pathlib import Path
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns

from scipy.stats import linregress, friedmanchisquare, wilcoxon
from sklearn import set_config
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "42"
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Keep DataFrame column names in pipeline steps.
set_config(transform_output="pandas")

# Configuration
RANDOM_STATE = 42
VARIANCE_THRESHOLD = 1e-6
N_SPLITS = 5
N_REPEATS = 5
PERMUTATION_REPEATS = 20
BOOTSTRAP_REPEATS = 2000

MAX_STABLE_FEATURES = 45

OPTUNA_TRIALS_XGB = 100
OPTUNA_TRIALS_LGBM = 100
OPTUNA_TRIALS_CAT = 100
OPTUNA_TRIALS_ET = 100
OPTUNA_TRIALS_RF = 100
CORRELATION_PRUNE_THRESHOLD = 0.90

out_root = Path("output")
out_root.mkdir(parents=True, exist_ok=True)

DATA_PATH = out_root / "STEP3C_SELECTED_DESCRIPTOR_DATASET.csv"
MASTER_PATH = out_root / "MASTER_MPEA_DATASET.csv"
DIR_HARDNESS = out_root / "step4_hardness_regressor"
DIR_HARDNESS.mkdir(parents=True, exist_ok=True)

OPTUNA_DB_PATH = DIR_HARDNESS / "optuna_studies.db"

def get_or_create_study(study_name, n_trials_target):
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{OPTUNA_DB_PATH}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        load_if_exists=True,
    )
    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, n_trials_target - completed)
    if completed:
        print(f"  -> resumed '{study_name}': {completed} trial(s) already completed, {remaining} remaining")
    return study, remaining

TARGET_REG = "PROPERTY: HV"
COLORS = {
    "Dummy": "gray", "Extra Trees": "#ff7f0e", "Random Forest": "#1f77b4",
    "XGBoost": "#d62728", "LightGBM": "#9467bd", "CatBoost": "#bcbd22"
}

def set_style():
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    sns.set_style("whitegrid")

def load_data(path):
    return pd.read_csv(path)

def attach_groups(df, master_path):
    master = pd.read_csv(master_path)
    pull_cols = ["SAMPLE_ID"]
    if "COMPOSITION_SIGNATURE" not in df.columns and "COMPOSITION_SIGNATURE" in master.columns:
        pull_cols.append("COMPOSITION_SIGNATURE")
    if "PROCESS_PHASE_SIGNATURE" not in df.columns and "PROCESS_PHASE_SIGNATURE" in master.columns:
        pull_cols.append("PROCESS_PHASE_SIGNATURE")
    if len(pull_cols) > 1:
        group_map = master[pull_cols].drop_duplicates("SAMPLE_ID")
        df = df.merge(group_map, on="SAMPLE_ID", how="left", validate="one_to_one")
    return df

def extract_initial_descriptors(df):
    exclude_cols = [
        TARGET_REG, "SAMPLE_ID", "COMPOSITION_SIGNATURE", "SOURCE", "REFERENCE",
        "YEAR", "PHASE_VEC_TENDENCY", "PHASE_AGREEMENT", "DATASET_VERSION",
        "ALLOY_HASH", "PROCESS_PHASE_SIGNATURE"
    ]
    elem_cols = [c for c in df.columns if c.startswith("ELEM_")]
    raw_cols = ["FORMULA", "RAW_FORMULA", "RAW_PROCESSING", "RAW_MICROSTRUCTURE", "RAW_PHASE"]
    leakage_cols = ["PROPERTY_HV_MEAN", "PROPERTY_HV_MIN", "PROPERTY_HV_MAX", "PROPERTY_HV_STD", "PROPERTY_HV_COUNT", "PROPERTY_HV_RANGE", "HV_CV", "HV_CLASS"]

    exclude_cols.extend(elem_cols + raw_cols + leakage_cols)
    descriptor_cols = [c for c in df.columns if c not in exclude_cols]
    return df[descriptor_cols].select_dtypes(include=[np.number]).copy(), df["SAMPLE_ID"].values

# Feature selection
class ConsensusSelector(BaseEstimator, SelectorMixin):
    _name_check_announced = False

    def __init__(self, top_k=MAX_STABLE_FEATURES, random_state=RANDOM_STATE):
        self.top_k = top_k
        self.random_state = random_state
        self.support_ = None
        self.selected_features_ = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.values
            X_val = X.values
            if not ConsensusSelector._name_check_announced:
                preview = list(self.feature_names_in_[:3])
                print(f"[ConsensusSelector] descriptor columns available (e.g. {preview}).")
                ConsensusSelector._name_check_announced = True
        else:
            X_val = X
            if not hasattr(self, 'feature_names_in_'):
                print(
                    "[ConsensusSelector] WARNING: received a NumPy array; "
                    "using placeholder descriptor names."
                )
                self.feature_names_in_ = np.array([f"f_{i}" for i in range(X_val.shape[1])])
        self.n_features_in_ = X_val.shape[1]

        et = ExtraTreesRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1).fit(X_val, y)
        xgb = XGBRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1, tree_method="hist", verbosity=0).fit(X_val, y)

        et_rank = pd.Series(et.feature_importances_).rank(pct=True)
        xgb_rank = pd.Series(xgb.feature_importances_).rank(pct=True)
        consensus = (0.50 * et_rank.values + 0.50 * xgb_rank.values)

        k_actual = min(self.top_k, X_val.shape[1])
        candidate_pool = min(X_val.shape[1], max(k_actual * 2, k_actual + 10))
        ranked_indices = np.argsort(consensus)[::-1][:candidate_pool]

        ranked_cols = self.feature_names_in_[ranked_indices]
        X_df = pd.DataFrame(X_val, columns=self.feature_names_in_)
        corr_matrix = X_df[ranked_cols].corr().abs()

        kept_cols = []
        for col in ranked_cols:
            if any(corr_matrix.loc[col, kept] > CORRELATION_PRUNE_THRESHOLD for kept in kept_cols):
                continue
            kept_cols.append(col)
            if len(kept_cols) == k_actual:
                break

        if len(kept_cols) < k_actual:
            for col in ranked_cols:
                if col not in kept_cols:
                    kept_cols.append(col)
                if len(kept_cols) == k_actual:
                    break

        self.support_ = np.isin(self.feature_names_in_, kept_cols)
        self.selected_features_ = self.feature_names_in_[self.support_]
        return self

    def _get_support_mask(self):
        return self.support_

# Bootstrap confidence intervals
def compute_bootstrap_cis(y_true, y_pred, repeats=BOOTSTRAP_REPEATS):
    r2_dist, rmse_dist, mae_dist = [], [], []
    n = len(y_true)
    for _ in range(repeats):
        idx = np.random.choice(n, n, replace=True)
        y_t, y_p = y_true[idx], y_pred[idx]
        r2_dist.append(r2_score(y_t, y_p))
        rmse_dist.append(np.sqrt(mean_squared_error(y_t, y_p)))
        mae_dist.append(mean_absolute_error(y_t, y_p))

    return {
        "R2_95CI": (np.percentile(r2_dist, 2.5), np.percentile(r2_dist, 97.5)),
        "RMSE_95CI": (np.percentile(rmse_dist, 2.5), np.percentile(rmse_dist, 97.5)),
        "MAE_95CI": (np.percentile(mae_dist, 2.5), np.percentile(mae_dist, 97.5))
    }

def cv_objective_evaluation(model_pipeline, X, y, y_bins, groups):
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    preds = np.zeros_like(y, dtype=float)
    counts = np.zeros_like(y, dtype=float)

    for tr, te in cv.split(X, y_bins, groups):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        model = TransformedTargetRegressor(regressor=model_pipeline, func=np.log1p, inverse_func=np.expm1)
        model.fit(X_tr, y[tr])
        preds[te] += model.predict(X_te)
        counts[te] += 1

    return r2_score(y, preds / counts)

def tune_models(X_df, y, groups):
    y_bins = pd.qcut(y, q=N_SPLITS, labels=False, duplicates="drop")

    # XGBoost
    print("\n[1/7] Tuning XGBoost...")
    def obj_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "tree_method": "hist", "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0
        }
        # Constant imputation for tree models.
        pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)), ("vt", VarianceThreshold(threshold=VARIANCE_THRESHOLD)), ("selector", ConsensusSelector()), ("model", XGBRegressor(**params))])
        return cv_objective_evaluation(pipe, X_df, y, y_bins, groups)

    study_xgb, remaining_xgb = get_or_create_study("xgb_hardness", OPTUNA_TRIALS_XGB)
    if remaining_xgb > 0:
        study_xgb.optimize(obj_xgb, n_trials=remaining_xgb, show_progress_bar=True)
    best_xgb = study_xgb.best_params
    best_xgb.update({"tree_method": "hist", "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0})
    print("XGBoost complete")

    # LightGBM
    best_lgbm = {}
    if LIGHTGBM_AVAILABLE:
        print("\n[2/7] Tuning LightGBM...")
        def obj_lgbm(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 15, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1
            }
            # Constant imputation for tree models.
            pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)), ("vt", VarianceThreshold(threshold=VARIANCE_THRESHOLD)), ("selector", ConsensusSelector()), ("model", LGBMRegressor(**params))])
            return cv_objective_evaluation(pipe, X_df, y, y_bins, groups)

        study_lgbm, remaining_lgbm = get_or_create_study("lgbm_hardness", OPTUNA_TRIALS_LGBM)
        if remaining_lgbm > 0:
            study_lgbm.optimize(obj_lgbm, n_trials=remaining_lgbm, show_progress_bar=True)
        best_lgbm = study_lgbm.best_params
        best_lgbm.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1})
        print("LightGBM complete")

    # Extra Trees
    print("\n[3/7] Tuning Extra Trees...")
    def obj_et(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "random_state": RANDOM_STATE, "n_jobs": -1
        }
        # Constant imputation for tree models.
        pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)), ("vt", VarianceThreshold(threshold=VARIANCE_THRESHOLD)), ("selector", ConsensusSelector()), ("model", ExtraTreesRegressor(**params))])
        return cv_objective_evaluation(pipe, X_df, y, y_bins, groups)

    study_et, remaining_et = get_or_create_study("et_hardness", OPTUNA_TRIALS_ET)
    if remaining_et > 0:
        study_et.optimize(obj_et, n_trials=remaining_et, show_progress_bar=True)
    best_et = study_et.best_params
    best_et.update({"random_state": RANDOM_STATE, "n_jobs": -1})
    print("Extra Trees complete")

    # Random Forest
    print("\n[4/7] Tuning Random Forest...")
    def obj_rf(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "random_state": RANDOM_STATE, "n_jobs": -1
        }
        # Constant imputation for tree models.
        pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)), ("vt", VarianceThreshold(threshold=VARIANCE_THRESHOLD)), ("selector", ConsensusSelector()), ("model", RandomForestRegressor(**params))])
        return cv_objective_evaluation(pipe, X_df, y, y_bins, groups)

    study_rf, remaining_rf = get_or_create_study("rf_hardness", OPTUNA_TRIALS_RF)
    if remaining_rf > 0:
        study_rf.optimize(obj_rf, n_trials=remaining_rf, show_progress_bar=True)
    best_rf = study_rf.best_params
    best_rf.update({"random_state": RANDOM_STATE, "n_jobs": -1})
    print("Random Forest complete")

    # CatBoost
    best_cat = {}
    if CATBOOST_AVAILABLE:
        print("\n[5/7] Tuning CatBoost...")
        def obj_cat(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 200, 1200),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "random_seed": RANDOM_STATE, "verbose": 0
            }
            # Constant imputation for tree models.
            pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)), ("vt", VarianceThreshold(threshold=VARIANCE_THRESHOLD)), ("selector", ConsensusSelector()), ("model", CatBoostRegressor(**params))])
            return cv_objective_evaluation(pipe, X_df, y, y_bins, groups)

        study_cat, remaining_cat = get_or_create_study("cat_hardness", OPTUNA_TRIALS_CAT)
        if remaining_cat > 0:
            study_cat.optimize(obj_cat, n_trials=remaining_cat, show_progress_bar=True)
        best_cat = study_cat.best_params
        best_cat.update({"random_seed": RANDOM_STATE, "verbose": 0})
        print("CatBoost complete")

    return best_xgb, best_lgbm, best_et, best_rf, best_cat

def build_regressor_models(xgb_params, lgbm_params, et_params, rf_params, cat_params):
    raw_pipelines = {
        # Constant imputation for tree models.
        "Dummy": Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)), ("vt", VarianceThreshold(threshold=VARIANCE_THRESHOLD)), ("model", DummyRegressor(strategy="mean"))]),
        "Random Forest": Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)), ("vt", VarianceThreshold(threshold=VARIANCE_THRESHOLD)), ("selector", ConsensusSelector()), ("model", RandomForestRegressor(**rf_params))]),
        "Extra Trees": Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)), ("vt", VarianceThreshold(threshold=VARIANCE_THRESHOLD)), ("selector", ConsensusSelector()), ("model", ExtraTreesRegressor(**et_params))]),
        "XGBoost": Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)), ("vt", VarianceThreshold(threshold=VARIANCE_THRESHOLD)), ("selector", ConsensusSelector()), ("model", XGBRegressor(**xgb_params))])
    }
    if LIGHTGBM_AVAILABLE and lgbm_params:
        raw_pipelines["LightGBM"] = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)), ("vt", VarianceThreshold(threshold=VARIANCE_THRESHOLD)), ("selector", ConsensusSelector()), ("model", LGBMRegressor(**lgbm_params))])
    if CATBOOST_AVAILABLE and cat_params:
        raw_pipelines["CatBoost"] = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)), ("vt", VarianceThreshold(threshold=VARIANCE_THRESHOLD)), ("selector", ConsensusSelector()), ("model", CatBoostRegressor(**cat_params))])

    return {
        name: TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
        for name, pipe in raw_pipelines.items()
    }

# Applicability domain
def evaluate_oof_applicability(X_train_subset, X_test_subset):
    # Impute before PCA.
    imputer = SimpleImputer(strategy="median")
    X_train_clean = imputer.fit_transform(X_train_subset)
    X_test_clean = imputer.transform(X_test_subset)

    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    scaler = StandardScaler()

    X_tr_scaled = scaler.fit_transform(X_train_clean)
    X_tr_pca = pca.fit_transform(X_tr_scaled)

    mcd = MinCovDet(random_state=RANDOM_STATE).fit(X_tr_pca)
    threshold = np.percentile(mcd.mahalanobis(X_tr_pca), 97.5)

    X_te_scaled = scaler.transform(X_test_clean)
    X_te_pca = pca.transform(X_te_scaled)
    distances = mcd.mahalanobis(X_te_pca)

    return distances <= threshold

def evaluate_regressors(X_df, y, sample_ids, groups, models):
    y_bins = pd.qcut(y, q=N_SPLITS, labels=False, duplicates="drop")
    folds = []
    for repeat in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        folds.extend(list(cv.split(X_df, y_bins, groups)))

    total_folds = len(folds)
    cv_preds = {name: np.zeros_like(y, dtype=float) for name in models}
    cv_count = np.zeros_like(y, dtype=float)
    selection_counts = Counter()
    oof_records = []

    for fold_idx, (tr, te) in enumerate(folds):
        X_tr, X_te = X_df.iloc[tr], X_df.iloc[te]
        y_tr = y[tr]

        ad_masks = {}
        for name, wrapper in models.items():
            model = clone(wrapper)
            model.fit(X_tr, y_tr)

            if name != "Dummy":
                selected = model.regressor_.named_steps["selector"].selected_features_
                ad_masks[name] = evaluate_oof_applicability(X_tr[selected], X_te[selected])
                for feature in selected:
                    selection_counts[feature] += 1
            else:
                ad_masks[name] = np.ones(len(te), dtype=bool)

            preds = model.predict(X_te)
            cv_preds[name][te] += preds

            for i, idx in enumerate(te):
                oof_records.append({
                    "Sample_ID": sample_ids[idx], "Observed": y[idx], "Predicted": preds[i],
                    "Fold": fold_idx, "Model": name, "In_Applicability_Domain": ad_masks[name][i]
                })
        cv_count[te] += 1

    rows = []
    oof_df = pd.DataFrame(oof_records)

    for name in models:
        y_cv = cv_preds[name] / cv_count
        cv_r2 = r2_score(y, y_cv)
        rmse = np.sqrt(mean_squared_error(y, y_cv))
        mae = mean_absolute_error(y, y_cv)

        ci_dict = compute_bootstrap_cis(y, y_cv)

        rows.append({
            "Model": name, "CV_R2": cv_r2, "R2_95CI": f"{ci_dict['R2_95CI'][0]:.3f} - {ci_dict['R2_95CI'][1]:.3f}",
            "RMSE": rmse, "RMSE_95CI": f"{ci_dict['RMSE_95CI'][0]:.3f} - {ci_dict['RMSE_95CI'][1]:.3f}",
            "MAE": mae, "MAE_95CI": f"{ci_dict['MAE_95CI'][0]:.3f} - {ci_dict['MAE_95CI'][1]:.3f}"
        })

    results = pd.DataFrame(rows).sort_values("CV_R2", ascending=False)

    # Holm-Bonferroni tests
    model_errors = {name: np.abs(y - (cv_preds[name] / cv_count)) for name in models if name != "Dummy"}
    if len(model_errors) >= 3:
        stat, p_val = friedmanchisquare(*list(model_errors.values()))
        print(f"\nFriedman test p-value: {p_val:.4e}")

        if p_val < 0.05:
            best_name = results.iloc[0]["Model"]
            if best_name != "Dummy":
                p_values, comparisons = [], []
                for name, errors in model_errors.items():
                    if name != best_name:
                        _, wp_val = wilcoxon(model_errors[best_name], errors)
                        p_values.append(wp_val)
                        comparisons.append(name)

                # Holm step-down
                sorted_idx = np.argsort(p_values)
                m = len(p_values)
                prev_adj = 0.0

                print(f"Wilcoxon signed-rank against selected model ({best_name}) [Holm-Bonferroni adjusted]:")
                for k, idx in enumerate(sorted_idx):
                    adj_p = min(1.0, p_values[idx] * (m - k))
                    adj_p = max(prev_adj, adj_p)
                    prev_adj = adj_p
                    print(f"  vs {comparisons[idx]}: raw_p = {p_values[idx]:.4e}, adj_p = {adj_p:.4e}")

    return results, cv_preds, selection_counts, total_folds, oof_df

def plot_calibration(oof_df, best_model_name):
    df_best = (
        oof_df[oof_df["Model"] == best_model_name]
        .groupby("Sample_ID", as_index=False)[["Observed", "Predicted"]]
        .mean()
    )
    y_obs, y_pred = df_best["Observed"], df_best["Predicted"]
    residuals = y_obs - y_pred

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    sns.scatterplot(x=y_pred, y=y_obs, alpha=0.5, ax=axs[0])
    axs[0].plot([y_obs.min(), y_obs.max()], [y_obs.min(), y_obs.max()], 'r--')
    slope = linregress(y_pred, y_obs).slope
    axs[0].set_title(f"Observed vs Predicted\n(Slope: {slope:.3f})")
    axs[0].set_xlabel("Predicted HV")
    axs[0].set_ylabel("Observed HV")

    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=axs[1])
    axs[1].axhline(0, color='r', linestyle='--')
    axs[1].set_title("Residuals vs Predicted")
    axs[1].set_xlabel("Predicted HV")
    axs[1].set_ylabel("Residuals")

    sns.histplot(residuals, kde=True, ax=axs[2])
    axs[2].set_title("Residual Distribution")
    axs[2].set_xlabel("Error")

    plt.tight_layout()
    plt.savefig(DIR_HARDNESS / "calibration_analysis.png", dpi=300)
    plt.close()

def main():
    np.random.seed(RANDOM_STATE)
    set_style()
    print("Loading datasets...")

    df = load_data(DATA_PATH)
    df = attach_groups(df, MASTER_PATH)
    df = df.dropna(subset=[TARGET_REG]).reset_index(drop=True)

    y_reg = df[TARGET_REG].values
    groups = (df["PROCESS_PHASE_SIGNATURE"] if "PROCESS_PHASE_SIGNATURE" in df.columns else df["COMPOSITION_SIGNATURE"]).values
    X_reg, sample_ids = extract_initial_descriptors(df)

    print("\nStarting hardness regression...")
    # Tune models
    best_xgb, best_lgbm, best_et, best_rf, best_cat = tune_models(X_reg, y_reg, groups)
    models = build_regressor_models(best_xgb, best_lgbm, best_et, best_rf, best_cat)

    # Evaluate models
    print("\n[6/7] Running repeated cross-validation...")
    results, cv_preds, selection_counts, total_folds, oof_df = evaluate_regressors(X_reg, y_reg, sample_ids, groups, models)
    print("Cross-validation complete")

    best_model_name = results.iloc[0]["Model"]
    print(f"\nSelected model: {best_model_name}")

    # Export results
    results.to_csv(DIR_HARDNESS / "final_regression_results_with_CI.csv", index=False)
    oof_df.to_csv(DIR_HARDNESS / "OOF_predictions.csv", index=False)

    num_tuned_models = len(models) - 1
    freq_df = pd.DataFrame({
        "Descriptor": list(selection_counts.keys()),
        "Selection_Frequency": [c / (total_folds * num_tuned_models) for c in selection_counts.values()]
    }).sort_values("Selection_Frequency", ascending=False)
    freq_df.to_csv(DIR_HARDNESS / "feature_selection_frequency.csv", index=False)

    # Final model fit
    print("\n[7/7] Training final model...")
    best_model = clone(models[best_model_name])
    best_model.fit(X_reg, y_reg)

    final_selected_features = best_model.regressor_.named_steps["selector"].selected_features_

    # --- BUG FIX: Save the exact feature list ---
    pd.DataFrame(
        {"Feature": final_selected_features}
    ).to_csv(
        DIR_HARDNESS / "hardness_features.csv",
        index=False
    )
    print(f"Saved {len(final_selected_features)} selected descriptors.")

    # --- BUG FIX: Save the transformed descriptor matrix for SHAP ---
    selector = best_model.regressor_.named_steps["selector"]
    X_selected = selector.transform(
        best_model.regressor_.named_steps["vt"].transform(
            best_model.regressor_.named_steps["imputer"].transform(X_reg)
        )
    )

    X_selected = pd.DataFrame(
        X_selected,
        columns=final_selected_features
    )

    X_selected.to_csv(
        DIR_HARDNESS / "selected_descriptor_matrix.csv",
        index=False
    )
    print("Saved exact selected descriptor matrix for SHAP.")

    # Permutation importance
    print("Computing permutation importance...")
    r = permutation_importance(best_model, X_reg, y_reg, n_repeats=PERMUTATION_REPEATS, random_state=RANDOM_STATE)

    perm_df = pd.DataFrame({
        "Feature": X_reg.columns,
        "Importance": r.importances_mean
    })
    perm_df = perm_df[perm_df["Feature"].isin(final_selected_features)].sort_values("Importance", ascending=False)
    perm_df.to_csv(DIR_HARDNESS / "permutation_importance.csv", index=False)

    print("Saving outputs...")
    joblib.dump(best_model, DIR_HARDNESS / "best_hardness_model.pkl")

    # Plot calibration
    plot_calibration(oof_df, best_model_name)
    print("\nDone.")

if __name__ == "__main__":
    main()
