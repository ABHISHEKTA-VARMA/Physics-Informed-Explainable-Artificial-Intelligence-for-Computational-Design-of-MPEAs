import shutil
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import friedmanchisquare, pearsonr, spearmanr
from sklearn.base import clone
from sklearn.covariance import EmpiricalCovariance
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    from google.colab import files

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


warnings.filterwarnings("ignore")

# Configuration
RANDOM_STATE = 42
DATASET_VERSION = "step2_baseline_pure_composition_v8"

np.random.seed(RANDOM_STATE)

input_path = Path("output/MASTER_MPEA_DATASET.csv")
output_dir = Path("output/step2_baseline_ml")
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
if IN_COLAB and not input_path.exists():
    print("Upload MASTER_MPEA_DATASET.csv")
    files.upload()

    uploaded_master = Path("MASTER_MPEA_DATASET.csv")
    if uploaded_master.exists():
        input_path.parent.mkdir(exist_ok=True)
        shutil.copyfile(uploaded_master, input_path)

if not input_path.exists():
    raise FileNotFoundError(
        "Missing required input file: output/MASTER_MPEA_DATASET.csv"
    )

# Plot settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 13,
    }
)
sns.set_style("whitegrid")

data = pd.read_csv(input_path)
target = "PROPERTY: HV"

# Schema
id_col = next(
    (c for c in ["SAMPLE_ID", "REFERENCE_ID", "ID"] if c in data.columns),
    None,
)
group_col = next(
    (
        c
        for c in ["COMPOSITION_SIGNATURE", "COMPOSITION", "ALLOY_SYSTEM"]
        if c in data.columns
    ),
    None,
)
source_col = next(
    (c for c in ["SOURCE", "REFERENCE", "DATASET"] if c in data.columns),
    None,
)

if not id_col:
    raise ValueError(
        "Could not locate a valid ID column (e.g., SAMPLE_ID)."
    )

if not group_col:
    raise ValueError(
        "Could not locate a valid grouping column "
        "(e.g., COMPOSITION_SIGNATURE)."
    )

if not source_col:
    print(
        "[WARNING] No SOURCE column found. "
        "Residual analysis by source will be bypassed."
    )
    data["SOURCE"] = "Unknown"
    source_col = "SOURCE"

# Composition features
elem_cols = [col for col in data.columns if col.startswith("ELEM_")]
feature_cols = elem_cols

# Residual labels
phase_cols = [
    col for col in data.columns if col.startswith("PHASE_OBSERVED_")
]
process_cols = [
    col for col in data.columns if col.startswith("PROCESS_CONDITION_")
]

if "PHASE_CLASS" not in data.columns:
    if phase_cols:
        data["PHASE_CLASS"] = (
            data[phase_cols]
            .idxmax(axis=1)
            .str.replace("PHASE_OBSERVED_", "", regex=False)
        )
    else:
        data["PHASE_CLASS"] = "Unknown"

if "PROCESS_CLASS" not in data.columns:
    if process_cols:
        data["PROCESS_CLASS"] = (
            data[process_cols]
            .idxmax(axis=1)
            .str.replace("PROCESS_CONDITION_", "", regex=False)
        )
    else:
        data["PROCESS_CLASS"] = "Unknown"

if len(feature_cols) == 0:
    raise ValueError("No element features (ELEM_*) detected.")

X_df = data[feature_cols].copy().fillna(0)
X = X_df.values
y = data[target].values
groups = data[group_col].values

if np.isnan(X).any():
    raise ValueError("NaN detected in feature matrix X.")

if np.isnan(y).any():
    raise ValueError("NaN detected in target array y.")

# Data sanity
print("\n--- DATA SANITY CHECK ---")
print(f"Total Samples: {len(data)}")
print(f"Total Features (PURE COMPOSITION): {len(feature_cols)}")
print(f"Identifier Column: {id_col}")
print(f"Grouping Column: {group_col}")
print(f"Unique compositions: {data[group_col].nunique()}")
print(f"Target mean: {y.mean():.4f}")
print(f"Target std: {y.std():.4f}")
print("-------------------------\n")

# Cross-validation
y_bins = pd.qcut(
    y,
    q=4,
    labels=False,
    duplicates="drop",
)

oof_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)
metric_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE + 10,
)

# Models
models = {
    "Dummy": Pipeline(
        [
            ("m", DummyRegressor(strategy="mean")),
        ]
    ),
    "Ridge": Pipeline(
        [
            ("s", StandardScaler()),
            ("m", Ridge(alpha=1.0)),
        ]
    ),
    "ElasticNet": Pipeline(
        [
            ("s", StandardScaler()),
            (
                "m",
                ElasticNet(
                    alpha=0.001,
                    l1_ratio=0.5,
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "Extra Trees": Pipeline(
        [
            (
                "m",
                ExtraTreesRegressor(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "Random Forest": Pipeline(
        [
            (
                "m",
                RandomForestRegressor(
                    n_estimators=400,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "Gradient Boosting": Pipeline(
        [
            (
                "m",
                GradientBoostingRegressor(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=4,
                    subsample=0.85,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "XGBoost": Pipeline(
        [
            (
                "m",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.025,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.15,
                    reg_lambda=1.2,
                    min_child_weight=2,
                    verbosity=0,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}

stochastic_models = {
    "Extra Trees",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
}

colors = {
    "Dummy": "gray",
    "Ridge": "#9467bd",
    "ElasticNet": "#8c564b",
    "Extra Trees": "#ff7f0e",
    "Random Forest": "#1f77b4",
    "Gradient Boosting": "#2ca02c",
    "XGBoost": "#d62728",
}

rows = []
cv_preds = {}
cv_variability = {}
fold_tracking = []
fold_r2_dict = {}

# Train and evaluate models
for name, pipe in models.items():
    print(f"Fitting {name}...")

    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for tr, te in metric_cv.split(X, y_bins, groups):
        model = clone(pipe).fit(X[tr], y[tr])
        yp = model.predict(X[te])

        r2_scores.append(r2_score(y[te], yp))
        rmse_scores.append(
            np.sqrt(mean_squared_error(y[te], yp))
        )
        mae_scores.append(mean_absolute_error(y[te], yp))

    fold_r2_dict[name] = r2_scores

    y_oof = np.zeros_like(y, dtype=float)
    y_std = np.zeros_like(y, dtype=float)

    n_seeds = 10 if name in stochastic_models else 1

    for fold, (tr, te) in enumerate(
        oof_cv.split(X, y_bins, groups),
        start=1,
    ):
        local_preds = []

        for seed in range(n_seeds):
            model = clone(pipe)

            if (
                n_seeds > 1
                and hasattr(model.named_steps["m"], "random_state")
            ):
                model.named_steps["m"].set_params(
                    random_state=RANDOM_STATE + seed
                )

            model.fit(X[tr], y[tr])
            local_preds.append(model.predict(X[te]))

        local_preds = np.vstack(local_preds)
        y_oof[te] = local_preds.mean(axis=0)
        y_std[te] = local_preds.std(axis=0)

        for idx in te:
            fold_tracking.append(
                {
                    id_col: data.iloc[idx][id_col],
                    "Model": name,
                    "Fold": fold,
                }
            )

    cv_preds[name] = y_oof
    cv_variability[name] = y_std

    abs_error = np.abs(y - y_oof)

    pear_corr = (
        pearsonr(y_std, abs_error)[0]
        if np.std(y_std) > 0
        else np.nan
    )
    spear_corr = (
        spearmanr(y_std, abs_error)[0]
        if np.std(y_std) > 0
        else np.nan
    )

    rows.append(
        [
            name,
            np.mean(r2_scores),
            np.std(r2_scores),
            np.mean(rmse_scores),
            np.std(rmse_scores),
            np.mean(mae_scores),
            np.std(mae_scores),
            pear_corr,
            spear_corr,
        ]
    )

pd.DataFrame(fold_tracking).to_csv(
    output_dir / "cv_fold_assignments.csv",
    index=False,
)

results = pd.DataFrame(
    rows,
    columns=[
        "Model",
        "R2",
        "R2_STD",
        "RMSE",
        "RMSE_STD",
        "MAE",
        "MAE_STD",
        "VAR_PEARSON",
        "VAR_SPEARMAN",
    ],
)

# Model selection
results["Rank_R2"] = results["R2"].rank(ascending=False)
results["Rank_RMSE"] = results["RMSE"].rank(ascending=True)
results["Rank_STD"] = results["R2_STD"].rank(ascending=True)
results["Combined_Rank"] = (
    results["Rank_R2"]
    + results["Rank_RMSE"]
    + results["Rank_STD"]
)
results = results.sort_values(
    "Combined_Rank"
).reset_index(drop=True)

results["DATASET_VERSION"] = DATASET_VERSION
results.to_csv(
    output_dir / "baseline_results.csv",
    index=False,
)

selected_model_name = results.iloc[0]["Model"]
selected_preds = cv_preds[selected_model_name]

oof_r2 = r2_score(y, selected_preds)
oof_rmse = np.sqrt(
    mean_squared_error(y, selected_preds)
)
oof_mae = mean_absolute_error(y, selected_preds)

print("\n--- MODEL SELECTION ---")
print(f"Selected baseline model: {selected_model_name}")
print(f"OOF R2: {oof_r2:.4f}")
print(f"OOF RMSE: {oof_rmse:.4f}")
print(f"OOF MAE: {oof_mae:.4f}")

# Friedman test
non_dummy_models = [
    model_name
    for model_name in models
    if model_name != "Dummy"
]
fold_r2_table = [
    fold_r2_dict[model_name]
    for model_name in non_dummy_models
]

friedman_df = pd.DataFrame(
    fold_r2_dict
)[non_dummy_models]
friedman_df.index = [
    f"Fold_{i + 1}"
    for i in range(len(friedman_df))
]
friedman_df.to_csv(
    output_dir / "friedman_fold_r2_scores.csv",
    index=True,
)

stat, p_val = friedmanchisquare(*fold_r2_table)

print("\n--- STATISTICAL TESTING ---")
print(
    "Friedman Test across ML models "
    f"(excl. Dummy): p-value = {p_val:.4e}"
)

if p_val < 0.05:
    print(
        "-> Statistically significant differences "
        "exist between model performances."
    )

# Uncertainty calibration
variability = cv_variability[selected_model_name]
abs_error = np.abs(y - selected_preds)

var_threshold_top = np.percentile(variability, 90)
var_threshold_bottom = np.percentile(variability, 10)

top_10_mae = np.mean(
    abs_error[variability >= var_threshold_top]
)
bottom_10_mae = np.mean(
    abs_error[variability <= var_threshold_bottom]
)

print("\n--- UNCERTAINTY CALIBRATION ---")
print(
    "Top 10% most uncertain predictions MAE: "
    f"{top_10_mae:.2f} HV"
)
print(
    "Bottom 10% most confident predictions MAE: "
    f"{bottom_10_mae:.2f} HV"
)

metadata_lines = [
    f"DATASET_VERSION: {DATASET_VERSION}",
    f"SELECTED_MODEL: {selected_model_name}",
    f"OOF_R2: {oof_r2:.4f}",
    f"OOF_RMSE: {oof_rmse:.4f}",
    f"OOF_MAE: {oof_mae:.4f}",
    f"FRIEDMAN_P_VALUE: {p_val:.4e}",
    f"TOP_10_UNCERTAINTY_MAE: {top_10_mae:.4f}",
    f"BOTTOM_10_UNCERTAINTY_MAE: {bottom_10_mae:.4f}",
]
(output_dir / "step2_metadata.txt").write_text(
    "\n".join(metadata_lines),
    encoding="utf-8",
)

# Learning curve
print(f"\nGenerating Learning Curve for {selected_model_name}...")

lc_cv_splits = list(
    oof_cv.split(X, y_bins, groups)
)

train_sizes, train_scores, test_scores = learning_curve(
    clone(models[selected_model_name]),
    X,
    y,
    groups=groups,
    cv=lc_cv_splits,
    scoring="r2",
    n_jobs=-1,
    train_sizes=np.linspace(0.2, 1.0, 5),
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(7, 5))
plt.fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color=colors[selected_model_name],
)
plt.fill_between(
    train_sizes,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.1,
    color="black",
)
plt.plot(
    train_sizes,
    train_scores_mean,
    "o-",
    color=colors[selected_model_name],
    label="Training score",
)
plt.plot(
    train_sizes,
    test_scores_mean,
    "o-",
    color="black",
    label="Cross-validation (OOF) score",
)
plt.xlabel("Number of Training Samples")
plt.ylabel("R² Score")
plt.title(f"Learning Curve: {selected_model_name}")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    output_dir / "learning_curve.png",
    dpi=700,
)
plt.close()

# Train final model
selected_model = clone(models[selected_model_name])
final_model = selected_model.fit(X, y)

joblib.dump(
    final_model,
    output_dir / "best_baseline_model.pkl",
)

joblib.dump(
    feature_cols,
    output_dir / "baseline_feature_names.pkl",
)
pd.DataFrame(
    {"All_Features": feature_cols}
).to_csv(
    output_dir / "model_features.csv",
    index=False,
)

# Permutation importance
print(
    f"\nCalculating Permutation Importance across all "
    f"{len(feature_cols)} features..."
)

fold_importances = []
selected_pipe = clone(models[selected_model_name])

for tr, te in oof_cv.split(X, y_bins, groups):
    selected_pipe.fit(X[tr], y[tr])

    perm = permutation_importance(
        selected_pipe,
        X[te],
        y[te],
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    fold_importances.append(perm.importances_mean)

importance_table = pd.DataFrame(
    {
        "Feature": feature_cols,
        "Mean_Importance": np.mean(
            fold_importances,
            axis=0,
        ),
        "Importance_STD": np.std(
            fold_importances,
            axis=0,
        ),
    }
).sort_values(
    "Mean_Importance",
    ascending=False,
)
importance_table.to_csv(
    output_dir / "feature_importance_stability.csv",
    index=False,
)

# Residual plots
selected_residuals = y - selected_preds

for metric in ["R2", "RMSE", "MAE"]:
    vals = results.set_index("Model")[metric]
    vals = vals.sort_values(
        ascending=(metric != "R2")
    )

    plt.figure(figsize=(7, 4))
    plt.bar(
        vals.index,
        vals.values,
        color=[colors[model_name] for model_name in vals.index],
        edgecolor="black",
    )
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{metric}.png",
        dpi=700,
    )
    plt.close()

plt.figure(figsize=(7, 7))
plt.scatter(
    y,
    selected_preds,
    s=22,
    alpha=0.6,
    color=colors[selected_model_name],
)
lims = [y.min(), y.max()]
plt.plot(lims, lims, "k--", linewidth=1.5)
plt.xlabel("Experimental HV")
plt.ylabel("Predicted HV")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    output_dir / "best_model_parity.png",
    dpi=700,
)
plt.close()

plt.figure(figsize=(7, 4))
sns.kdeplot(
    selected_residuals,
    linewidth=2,
    color=colors[selected_model_name],
)
plt.xlabel("Residual (HV)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    output_dir / "residual_distribution.png",
    dpi=700,
)
plt.close()

plt.figure(figsize=(6, 5))
plt.scatter(
    selected_preds,
    selected_residuals,
    s=20,
    alpha=0.6,
    color=colors[selected_model_name],
)
plt.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1.5,
)
plt.xlabel("Predicted HV")
plt.ylabel("Residual (HV)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    output_dir / "residual_vs_predicted.png",
    dpi=700,
)
plt.close()

residual_df = pd.DataFrame(
    {
        "Residual": selected_residuals,
        "PHASE_CLASS": data["PHASE_CLASS"],
        "PROCESS_CLASS": data["PROCESS_CLASS"],
        "SOURCE": data[source_col],
    }
)

for meta_col in ["PHASE_CLASS", "PROCESS_CLASS", "SOURCE"]:
    plt.figure(figsize=(8, 4))
    sns.boxplot(
        data=residual_df,
        x=meta_col,
        y="Residual",
        hue=meta_col,
        legend=False,
    )
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"residuals_by_{meta_col}.png",
        dpi=700,
    )
    plt.close()

# Feature-space coverage
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cov = EmpiricalCovariance().fit(X_scaled)
mahal = cov.mahalanobis(X_scaled)

pd.DataFrame(
    {
        id_col: data[id_col],
        "MAHALANOBIS_DISTANCE": mahal,
    }
).to_csv(
    output_dir / "mahalanobis_distance.csv",
    index=False,
)

plt.figure(figsize=(6, 4))
plt.hist(
    mahal,
    bins=30,
    edgecolor="black",
)
plt.xlabel("Feature-Space Mahalanobis Distance")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    output_dir / "feature_space_coverage.png",
    dpi=700,
)
plt.close()

# Variability and error
plt.figure(figsize=(6, 5))
plt.scatter(
    variability,
    abs_error,
    s=20,
    alpha=0.6,
    color=colors[selected_model_name],
)
plt.xlabel("Ensemble Prediction Variability (HV std)")
plt.ylabel("Absolute Prediction Error (HV)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    output_dir / "variability_error_correlation.png",
    dpi=700,
)
plt.close()

# Prediction errors
prediction_error_df = pd.DataFrame(
    {
        id_col: data[id_col],
        "Experimental_HV": y,
        "Predicted_HV": selected_preds,
        "Absolute_Error": abs_error,
        "PHASE_CLASS": data["PHASE_CLASS"],
        "SOURCE": data[source_col],
    }
).sort_values(
    "Absolute_Error",
    ascending=False,
)
prediction_error_df.to_csv(
    output_dir / "prediction_error_ranking.csv",
    index=False,
)

pred_df = pd.DataFrame(
    {
        id_col: data[id_col],
        "Experimental_HV": y,
    }
)
for name, yp in cv_preds.items():
    pred_df[f"{name}_Pred"] = yp

pred_df.to_csv(
    output_dir / "crossval_predictions.csv",
    index=False,
)

var_df = pd.DataFrame(
    {
        id_col: data[id_col],
        "Experimental_HV": y,
    }
)
for name, var in cv_variability.items():
    var_df[f"{name}_VAR"] = var

var_df.to_csv(
    output_dir / "ensemble_prediction_variability.csv",
    index=False,
)

print("\nEvaluation complete.")

if IN_COLAB:
    zip_path = shutil.make_archive(
        "step2_baseline_ml_outputs",
        "zip",
        output_dir,
    )
    print(f"Output archive written: {zip_path}")
    files.download(zip_path)
