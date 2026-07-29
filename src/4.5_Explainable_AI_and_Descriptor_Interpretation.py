import json
import shutil
import warnings
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from scipy.stats import spearmanr, pearsonr
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_COL = "PROPERTY: HV"
GROUP_COL = "COMPOSITION_SIGNATURE"
CALCULATE_INTERACTIONS = True
N_FOLDS_STABILITY = 5
SHAP_IMPORTANCE_THRESHOLD = 0.01
BACKGROUND_SAMPLE_SIZE = 100
INTERACTION_SAMPLE_SIZE = 200

TREE_MODELS = (
    "RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor",
    "DecisionTreeRegressor", "XGBRegressor", "LGBMRegressor", "CatBoostRegressor",
    "HistGradientBoostingRegressor"
)

inp_data = Path("output/STEP3C_SELECTED_DESCRIPTOR_DATASET.csv")
inp_model = Path("output/step4_hardness_regressor/best_hardness_model.pkl")
inp_fem_alloys = Path("output/step6_validation_exports/FEM_VALIDATION_STACK.csv")
inp_metadata = Path("output/DESCRIPTOR_FAMILY_TABLE.csv")

out_dir = Path("output/shap_interpretability")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "shap_dependence").mkdir(exist_ok=True)
(out_dir / "shap_interactions").mkdir(exist_ok=True)
(out_dir / "local_explanations").mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 13})
sns.set_style("whitegrid")

np.random.seed(RANDOM_STATE)
rng = np.random.default_rng(RANDOM_STATE)


def compute_hash(filepath):
    if Path(filepath).exists():
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    return "FILE_NOT_FOUND"


def extract_base_model(pipeline):
    if hasattr(pipeline, "regressor_"):
        base_pipe = pipeline.regressor_
    elif hasattr(pipeline, "regressor"):
        base_pipe = pipeline.regressor
    else:
        base_pipe = pipeline

    if hasattr(base_pipe, "named_steps"):
        final_model = list(base_pipe.named_steps.values())[-1]
        try:
            preprocess_pipe = base_pipe[:-1]
        except Exception:
            preprocess_pipe = None
    else:
        final_model = base_pipe
        preprocess_pipe = None

    return final_model, preprocess_pipe


# Load data
df = pd.read_csv(inp_data)
df = df[df[TARGET_COL] > 0].copy()

model_pipeline = joblib.load(inp_model)

# Feature names used during training
if hasattr(model_pipeline, "feature_names_in_"):
    raw_feature_names = list(model_pipeline.feature_names_in_)
elif hasattr(model_pipeline, "regressor_") and hasattr(model_pipeline.regressor_, "feature_names_in_"):
    raw_feature_names = list(model_pipeline.regressor_.feature_names_in_)
elif hasattr(model_pipeline, "regressor_") and hasattr(model_pipeline.regressor_.named_steps.get("imputer", None), "feature_names_in_"):
    raw_feature_names = list(model_pipeline.regressor_.named_steps["imputer"].feature_names_in_)
else:
    raise RuntimeError("Cannot recover original training feature list.")

# Load descriptor metadata
if inp_metadata.exists():
    try:
        meta_df = pd.read_csv(inp_metadata)
        req_cols = ["Feature", "Descriptor_Family"]
        if not all(col in meta_df.columns for col in req_cols):
            print(f"Warning: {inp_metadata} missing required columns. Using default fallback.")
            raise ValueError("Schema mismatch")

        meta_df = meta_df.set_index("Feature")
        if "Physical_Meaning" not in meta_df.columns:
            meta_df["Physical_Meaning"] = "Unknown"

    except Exception:
        meta_df = pd.DataFrame(index=raw_feature_names, columns=["Descriptor_Family", "Physical_Meaning"])
        meta_df["Descriptor_Family"] = "Unassigned"
        meta_df["Physical_Meaning"] = "Unknown"
else:
    meta_df = pd.DataFrame(index=raw_feature_names, columns=["Descriptor_Family", "Physical_Meaning"])
    meta_df["Descriptor_Family"] = "Unassigned"
    meta_df["Physical_Meaning"] = "Unknown"

X = df[raw_feature_names].copy()
y = df[TARGET_COL].values

print(f"Dataset columns prepared: {len(X.columns)}")

# Verify feature compatibility
try:
    fitted_cols = None
    if hasattr(model_pipeline, "feature_names_in_"):
        fitted_cols = model_pipeline.feature_names_in_
    elif hasattr(model_pipeline, "regressor_") and hasattr(model_pipeline.regressor_, "feature_names_in_"):
        fitted_cols = model_pipeline.regressor_.feature_names_in_

    if fitted_cols is not None:
        print(f"Model expects: {len(fitted_cols)} columns")

        missing = sorted(set(fitted_cols) - set(X.columns))
        extra = sorted(set(X.columns) - set(fitted_cols))

        if missing or extra:
            print("\nFeature mismatch detected")
            print(f"Missing from dataset ({len(missing)}):")
            print(missing[:20])
            print(f"\nExtra in dataset ({len(extra)}):")
            print(extra[:20])
            print("\n Model pipeline and dataset inconsistent.")
    else:
        print("Automatic Validation")
except Exception as e:
    print(f"Skipping feature validation check: {e}")

samples_before = len(X)
z_scores = np.abs((y - y.mean()) / (y.std() + 1e-9))
mask = z_scores < 3
X = X.iloc[mask].reset_index(drop=True)
y = y[mask]
alloy_hashes = df["ALLOY_HASH"].iloc[mask].reset_index(drop=True) if "ALLOY_HASH" in df.columns else pd.Series([f"Alloy_{i}" for i in range(len(X))])
formulas = df["FORMULA"].iloc[mask].reset_index(drop=True) if "FORMULA" in df.columns else alloy_hashes

groups = df[GROUP_COL].iloc[mask].reset_index(drop=True).values if GROUP_COL in df.columns else None
n_groups = len(np.unique(groups)) if groups is not None else 0

final_model, preprocessor = extract_base_model(model_pipeline)

try:
    X_proc = preprocessor.transform(X) if preprocessor else X.values
    try:
        proc_feature_names = preprocessor.get_feature_names_out(raw_feature_names)
    except Exception:
        if X_proc.shape[1] == len(raw_feature_names):
            proc_feature_names = raw_feature_names
        else:
            proc_feature_names = [f"Feature_{i+1}" for i in range(X_proc.shape[1])]
except Exception:
    X_proc = X.values
    proc_feature_names = raw_feature_names

feature_names = list(proc_feature_names)
X_proc_df = pd.DataFrame(X_proc, columns=feature_names)

# SHAP values
if type(final_model).__name__ in TREE_MODELS:
    try:
        explainer = shap.TreeExplainer(final_model, feature_perturbation="tree_path_dependent")
        shap_backend = "TreeExplainer"
    except Exception:
        bg_idx = rng.choice(len(X_proc), size=min(BACKGROUND_SAMPLE_SIZE, len(X_proc)), replace=False)
        explainer = shap.Explainer(final_model, X_proc[bg_idx])
        shap_backend = "SHAP automatic explainer (with background)"
else:
    bg_idx = rng.choice(len(X_proc), size=min(BACKGROUND_SAMPLE_SIZE, len(X_proc)), replace=False)
    explainer = shap.Explainer(final_model, X_proc[bg_idx])
    shap_backend = "SHAP automatic explainer (with background)"

print(f"SHAP backend: {shap_backend}")
shap_exp = explainer(X_proc_df, check_additivity=False)
shap_values = shap_exp.values
base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (np.ndarray, list)) else explainer.expected_value
predictions = model_pipeline.predict(X)

# Global SHAP importance
print("Computing SHAP importance...")
mean_shap = np.abs(shap_values).mean(axis=0)

std_shap = np.nan * np.ones_like(mean_shap)
lower95 = np.nan * np.ones_like(mean_shap)
upper95 = np.nan * np.ones_like(mean_shap)

# SHAP summary plots
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_proc_df, show=False)
plt.tight_layout()
plt.savefig(out_dir / "shap_beeswarm.png", dpi=700, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_proc_df, plot_type="violin", show=False)
plt.tight_layout()
plt.savefig(out_dir / "shap_violin.png", dpi=700, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
plt.scatter(predictions, np.abs(shap_values).sum(axis=1), alpha=0.5, color="#1f77b4")
plt.xlabel("Predicted HV")
plt.ylabel("Total Absolute SHAP Contribution")
plt.title("SHAP Contribution vs Prediction")
plt.tight_layout()
plt.savefig(out_dir / "shap_vs_prediction.png", dpi=700)
plt.close()

# Hierarchical clustering
try:
    clustering = shap.utils.hclust(X_proc_df, y)
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_exp, clustering=clustering, clustering_cutoff=0.5, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_hierarchical_clustering.png", dpi=700, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Hierarchical clustering skipped: {e}")

# Cross-validation permutation importance
if groups is not None and n_groups >= 2:
    actual_folds = min(N_FOLDS_STABILITY, n_groups)
    cv = GroupKFold(n_splits=actual_folds)
    split_gen = cv.split(X, y, groups=groups)
else:
    actual_folds = max(2, min(N_FOLDS_STABILITY, len(X)))
    cv = KFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_STATE)
    split_gen = cv.split(X, y)

print("Computing permutation importance...")
perm_importances = []
for train_idx, val_idx in split_gen:
    fold_model = clone(model_pipeline).fit(X.iloc[train_idx], y[train_idx])
    perm = permutation_importance(fold_model, X.iloc[val_idx], y[val_idx], n_repeats=5, random_state=RANDOM_STATE)
    perm_importances.append(perm.importances_mean)
cv_perm_mean = np.mean(perm_importances, axis=0)

# Align permutation importance with processed features
perm_dict = dict(zip(X.columns, cv_perm_mean))
aligned_perm_cv = [perm_dict.get(feat, np.nan) for feat in feature_names]

# Correlation analysis
directions, p_corr_vals, p_pval_vals, s_corr_vals = [], [], [], []
for i, col in enumerate(feature_names):
    p_corr, p_pval = pearsonr(X_proc_df[col], y)
    s_corr, _ = spearmanr(X_proc_df[col], y)
    p_corr_vals.append(p_corr)
    p_pval_vals.append(p_pval)
    s_corr_vals.append(s_corr)

    frac_pos = (shap_values[:, i] > 0).mean()
    if frac_pos > 0.6:
        directions.append("Mostly Positive")
    elif frac_pos < 0.4:
        directions.append("Mostly Negative")
    else:
        directions.append("Mixed")

_, p_pval_adj, _, _ = multipletests(p_pval_vals, method="fdr_bh")

# Summary table
shap_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_SHAP": mean_shap,
    "Std_SHAP": std_shap,
    "Lower_95_CI": lower95,
    "Upper_95_CI": upper95,
    "Permutation_Importance_CV": aligned_perm_cv,
    "Pearson_r": p_corr_vals,
    "Pearson_pval": p_pval_vals,
    "Pearson_pval_FDR": p_pval_adj,
    "Spearman_rho": s_corr_vals,
    "Direction": directions
})

shap_df = shap_df.merge(meta_df[["Descriptor_Family", "Physical_Meaning"]], left_on="Feature", right_index=True, how="left")
shap_df["Descriptor_Family"] = shap_df["Descriptor_Family"].fillna("Other")
shap_df = shap_df.sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
shap_df["Rank"] = np.arange(1, len(shap_df) + 1)
shap_df.to_csv(out_dir / "comprehensive_shap_importance.csv", index=False)

top15_df = shap_df.head(15)
top15_features = top15_df["Feature"].tolist()
top15_df.to_csv(out_dir / "TOP15_SHAP_DESCRIPTORS.csv", index=False)

plt.figure(figsize=(9, 7))
plot_df = top15_df.sort_values("Mean_SHAP", ascending=True)
plt.barh(
    plot_df["Feature"],
    plot_df["Mean_SHAP"],
    color="#4C72B0",
    alpha=0.85
)
plt.xlabel("Mean |SHAP Value|")
plt.ylabel("Descriptor")
plt.title("Top 15 Global SHAP Importance")
plt.tight_layout()
plt.savefig(out_dir / "top15_shap_stability.png", dpi=700, bbox_inches="tight")
plt.close()

total_mean_shap = shap_df["Mean_SHAP"].sum()
sig_features = shap_df[shap_df["Mean_SHAP"] / (total_mean_shap + 1e-9) >= SHAP_IMPORTANCE_THRESHOLD]["Feature"].tolist()

for feature in sig_features:
    try:
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(feature, shap_values, X_proc_df, interaction_index="auto", show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_dependence" / f"dependence_{feature}.png", dpi=700, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Skipped dependence plot for {feature}: {e}")
        plt.close()

mech_df = shap_df[["Feature", "Descriptor_Family", "Physical_Meaning", "Mean_SHAP", "Rank"]]
mech_df.to_csv(out_dir / "descriptor_mechanism_summary.csv", index=False)

# Descriptor family contributions
family_agg = shap_df.groupby("Descriptor_Family")["Mean_SHAP"].sum().sort_values(ascending=False).reset_index()
family_agg["Contribution_Percentage"] = (family_agg["Mean_SHAP"] / (total_mean_shap + 1e-9)) * 100
family_agg.to_csv(out_dir / "descriptor_family_percentages.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=family_agg, x="Contribution_Percentage", y="Descriptor_Family", hue="Descriptor_Family", palette="viridis", legend=False)
plt.xlabel("Total SHAP Contribution (%)")
plt.ylabel("Descriptor Family")
plt.title("Relative Importance by Physical Descriptor Family")
plt.tight_layout()
plt.savefig(out_dir / "descriptor_family_plot.png", dpi=700)
plt.close()

# Local explanations
print("Generating local explanations")
abs_err = np.abs(predictions - y)
idx_best = np.argmin(abs_err)
idx_worst = np.argmax(abs_err)
idx_med = np.argsort(abs_err)[len(abs_err) // 2]

alloy_targets = {"Best_Predicted": idx_best, "Typical_Predicted": idx_med, "Worst_Predicted": idx_worst}

for label, idx in alloy_targets.items():
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_exp[idx], show=False)
    plt.title(f"Waterfall: {label} (Error: {abs_err[idx]:.1f} HV)")
    plt.tight_layout()
    plt.savefig(out_dir / "local_explanations" / f"waterfall_{label}.png", dpi=700)
    plt.close()

    if label != "Typical_Predicted":
        html = shap.plots.force(base_val, shap_values[idx, :], X_proc_df.iloc[idx, :])
        shap.save_html(str(out_dir / "local_explanations" / f"force_{label}.html"), html)

local_records = []
for i in range(len(X_proc)):
    top20_idx = np.argsort(np.abs(shap_values[i]))[-20:][::-1]
    for j in top20_idx:
        if abs(shap_values[i, j]) > 1e-5:
            local_records.append({
                "Alloy_Hash": alloy_hashes.iloc[i],
                "Formula": formulas.iloc[i],
                "True_HV": y[i],
                "Predicted_HV": predictions[i],
                "Prediction_Error": abs_err[i],
                "Residual": predictions[i] - y[i],
                "Base_Value": base_val,
                "Feature": feature_names[j],
                "SHAP_Contribution": shap_values[i, j]
            })

pd.DataFrame(local_records).to_csv(out_dir / "local_explanations" / "top20_local_contributions.csv", index=False)

# Decision plot
plt.figure(figsize=(10, 8))
sample_idx = rng.choice(len(X_proc_df), size=min(100, len(X_proc_df)), replace=False)
shap.decision_plot(base_val, shap_values[sample_idx], X_proc_df.iloc[sample_idx], show=False)
plt.tight_layout()
plt.savefig(out_dir / "shap_decision_plot.png", dpi=700)
plt.close()

# SHAP interactions
if CALCULATE_INTERACTIONS:
    print("Computing interactions")
    try:
        int_idx = rng.choice(len(X_proc), size=min(INTERACTION_SAMPLE_SIZE, len(X_proc)), replace=False)
        interaction_values = explainer.shap_interaction_values(X_proc[int_idx])

        abs_interaction_matrix = np.abs(interaction_values).mean(axis=0)
        np.fill_diagonal(abs_interaction_matrix, 0)
        int_df = pd.DataFrame(abs_interaction_matrix, index=feature_names, columns=feature_names)

        signed_interaction_matrix = interaction_values.mean(axis=0)
        np.fill_diagonal(signed_interaction_matrix, 0)
        signed_df = pd.DataFrame(signed_interaction_matrix, index=feature_names, columns=feature_names)

        top15_feats = top15_df["Feature"].tolist()
        ordered_int_df = int_df.loc[top15_feats, top15_feats]

        plt.figure(figsize=(12, 10))
        sns.heatmap(ordered_int_df, cmap="magma", xticklabels=True, yticklabels=True)
        plt.title("Top 15 Feature SHAP Interactions")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_interactions" / "interaction_heatmap_ordered.png", dpi=700)
        plt.close()

        int_df.to_csv(out_dir / "shap_interactions" / "absolute_interaction_matrix.csv")
        signed_df.to_csv(out_dir / "shap_interactions" / "signed_interaction_matrix.csv")

        interaction_pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                interaction_pairs.append({
                    "Feature_1": feature_names[i],
                    "Feature_2": feature_names[j],
                    "Absolute_Interaction_Strength": abs_interaction_matrix[i, j],
                    "Signed_Interaction_Strength": signed_interaction_matrix[i, j]
                })

        pairs_df = pd.DataFrame(interaction_pairs).sort_values("Absolute_Interaction_Strength", ascending=False)
        pairs_df.head(50).to_csv(out_dir / "shap_interactions" / "top50_interaction_pairs.csv", index=False)

    except Exception as exc:
        print(f"SHAP interaction extraction unavailable: {exc}")

# FEM linkage
try:
    fem_df = pd.read_csv(inp_fem_alloys)
    top_feats = top15_df["Feature"].head(5).tolist()

    linkage_data = []
    for _, row in fem_df.iterrows():
        a_hash = row.get("ALLOY_HASH", row.get("SAMPLE_ID", "Unknown"))
        entry = {"ALLOY_HASH": a_hash, "Predicted_HV": row.get("PREDICTED_HV", np.nan)}

        idx_match = alloy_hashes[alloy_hashes == a_hash].index
        if len(idx_match) > 0:
            idx = idx_match[0]
            for feature in top_feats:
                feat_idx = feature_names.index(feature)
                val = X_proc_df.at[idx, feature]
                entry[f"{feature}_Value"] = val
                entry[f"{feature}_Percentile"] = (X_proc_df[feature] < val).mean() * 100
                entry[f"{feature}_SHAP"] = shap_values[idx, feat_idx]

        linkage_data.append(entry)

    pd.DataFrame(linkage_data).to_csv(out_dir / "fem_descriptor_linkage.csv", index=False)
    fem_linkage_status = "computed"
except FileNotFoundError:
    fem_linkage_status = "skipped_missing_input"

# Correlation and VIF
if len(top15_features) >= 2:
    X_top = X_proc_df[top15_features]
    X_scaled = (X_top - X_top.mean()) / (X_top.std() + 1e-9)

    plt.figure(figsize=(12, 10))
    sns.heatmap(X_scaled.corr(), cmap="coolwarm", center=0, annot=False, square=True)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "top15_feature_correlation.png", dpi=700)
    plt.close()

    vif_data = []
    for i, f in enumerate(X_top.columns):
        if X_top[f].var() < 1e-9 or np.linalg.matrix_rank(X_scaled.values) < X_scaled.shape[1]:
            vif = np.nan
        else:
            vif = variance_inflation_factor(X_scaled.values, i)

        vif_data.append({"Feature": f, "VIF": vif})

    pd.DataFrame(vif_data).to_csv(out_dir / "top15_vif_analysis.csv", index=False)

# Metadata
import sklearn

metadata = {
    "module": "SHAP analysis",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "reproducibility": {
        "random_seed": RANDOM_STATE,
        "python_version": sys.version,
        "shap_version": shap.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pipeline_hash": compute_hash(inp_model),
        "dataset_hash": compute_hash(inp_data),
    },
    "data_diagnostics": {
        "samples_evaluated": len(X_proc_df),
        "outliers_removed": int(samples_before - len(X_proc)),
        "z_score_threshold": 3,
        "n_descriptors": len(feature_names),
        "n_composition_groups": n_groups
    },
    "model_diagnostics": {
        "model_class": final_model.__class__.__name__
    },
    "shap_diagnostics": {
        "backend": shap_backend,
        "importance_threshold_used": SHAP_IMPORTANCE_THRESHOLD,
        "background_sample_size": BACKGROUND_SAMPLE_SIZE,
        "interaction_sample_size": INTERACTION_SAMPLE_SIZE,
        "bootstrap_iterations": 0,
        "bootstrap_type": "None",
        "permutation_importance": "Cross-validation",
        "top_feature": str(top15_df.iloc[0]["Feature"]),
        "p_value_adjustment": "Benjamini-Hochberg (FDR)"
    },
    "fem_linkage_status": fem_linkage_status
}

with open(out_dir / "shap_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Creating archive")
zip_name = "shap_interpretability_results"
zip_output_path = Path("output") / zip_name

try:
    shutil.make_archive(base_name=str(zip_output_path), format="zip", root_dir=out_dir)
    print(f"Archive written: {zip_output_path}.zip")
    try:
        from google.colab import files
        files.download(f"{zip_output_path}.zip")
    except ImportError:
        pass
except Exception as exc:
    print(f"Archive generation failed: {exc}")

print("Analysis complete.")
