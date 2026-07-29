import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")

# Configuration and styling
DIR_HARDNESS = Path("output/step4_hardness_regressor")
DATA_PATH = Path("output/STEP3C_SELECTED_DESCRIPTOR_DATASET.csv")

COLORS = {
    "Dummy": "gray",
    "Extra Trees": "#ff7f0e",
    "Random Forest": "#1f77b4",
    "XGBoost": "#d62728",
    "LightGBM": "#9467bd",
    "CatBoost": "#bcbd22",
    "Gradient Boosting": "#8c564b",
    "ElasticNet": "#e377c2",
    "Ridge": "#7f7f7f",
    "SVR": "#17becf"
}

def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11
    })
    sns.set_style("whitegrid")

set_plot_style()

print("Generating visualizations...")


# plot and save metrics (R2, RMSE, MAE)

def plot_model_metric(df, metric_col, y_label, filename, ascending=False):
    if metric_col not in df.columns:
        print(f" Warning: Column '{metric_col}' not found. Skipping {filename} plot.")
        return

    df_sorted = df.sort_values(metric_col, ascending=ascending).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_sorted,
        x="Model",
        y=metric_col,
        palette=[COLORS.get(m, "gray") for m in df_sorted["Model"]],
        edgecolor="black",
        linewidth=0.7
    )
    ax.grid(axis='y', alpha=0.3)
    plt.ylabel(y_label)
    plt.xlabel("Algorithm")

    # Handle y-limits dynamically (important for negative Dummy R2)
    if "R2" in metric_col.upper() or "R²" in metric_col.upper():
        ymin = min(-0.10, df_sorted[metric_col].min() - 0.05)
        plt.ylim(ymin, 1.0)
    else:
        plt.ylim(0, df_sorted[metric_col].max() * 1.15)

    # Annotations handling both positive and negative bars
    for p in ax.patches:
        h = p.get_height()
        va_align = 'bottom' if h >= 0 else 'top'
        y_offset = 5 if h >= 0 else -5
        ax.annotate(f"{h:.3f}",
                    (p.get_x() + p.get_width() / 2., h),
                    ha='center', va=va_align, fontsize=11, xytext=(0, y_offset),
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig(DIR_HARDNESS / f"{filename}.png", dpi=600, bbox_inches="tight")
    plt.savefig(DIR_HARDNESS / f"{filename}.pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f" Saved: {filename}.png/.pdf")
# Model Comparison
res_file = DIR_HARDNESS / "final_regression_results_with_CI.csv"
if res_file.exists():
    df_res = pd.read_csv(res_file)
    print(f"\nFound columns in results file: {list(df_res.columns)}")

    if not df_res.empty:
        # Auto-detect column names for R2, RMSE, and MAE
        r2_col = next((c for c in df_res.columns if "R2" in c.upper() or "R²" in c.upper()), None)
        rmse_col = next((c for c in df_res.columns if "RMSE" in c.upper()), None)
        mae_col = next((c for c in df_res.columns if "MAE" in c.upper()), None)

        if r2_col:
            plot_model_metric(df_res, r2_col, f"Cross-Validated R²", "model_r2_comparison", ascending=False)
        else:
            print(" Warning: No R2 column found!")

        if rmse_col:
            plot_model_metric(df_res, rmse_col, f"Cross-Validated RMSE", "model_rmse_comparison", ascending=True)
        else:
            print(" Warning: No RMSE column found in CSV! RMSE plot skipped.")

        if mae_col:
            plot_model_metric(df_res, mae_col, f"Cross-Validated MAE", "model_mae_comparison", ascending=True)
        else:
            print(" Warning: No MAE column found in CSV! MAE plot skipped.")

# Feature selection frequency
freq_file = DIR_HARDNESS / "feature_selection_frequency.csv"
if freq_file.exists():
    df_freq = (
        pd.read_csv(freq_file)
        .sort_values("Selection_Frequency", ascending=False)
        .head(20)
        .iloc[::-1]
    )

    if not df_freq.empty:
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(
            data=df_freq,
            x="Selection_Frequency",
            y="Descriptor",
            color="#2ca02c",
            edgecolor="black",
            linewidth=0.7
        )
        ax.grid(axis='x', alpha=0.3)
        plt.xlabel("Selection Frequency (Across Folds)")
        plt.ylabel("Descriptor")
        plt.xlim(0, 1.05)

        plt.tight_layout()
        plt.savefig(DIR_HARDNESS / "feature_selection_frequency.png", dpi=600, bbox_inches="tight")
        plt.savefig(DIR_HARDNESS / "feature_selection_frequency.pdf", dpi=600, bbox_inches="tight")
        plt.close()
        print(" Saved: feature_selection_frequency.png/.pdf")

# Permutation importance
perm_file = DIR_HARDNESS / "permutation_importance.csv"
if perm_file.exists():
    df_perm = (
        pd.read_csv(perm_file)
        .sort_values("Importance", ascending=False)
        .head(20)
        .iloc[::-1]
    )

    if not df_perm.empty:
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(
            data=df_perm,
            x="Importance",
            y="Feature",
            color="#9467bd",
            edgecolor="black",
            linewidth=0.7
        )
        ax.grid(axis='x', alpha=0.3)
        plt.xlabel("Mean Decrease in Predictive Accuracy")
        plt.ylabel("Descriptor")

        plt.tight_layout()
        plt.savefig(DIR_HARDNESS / "permutation_importance.png", dpi=600, bbox_inches="tight")
        plt.savefig(DIR_HARDNESS / "permutation_importance.pdf", dpi=600, bbox_inches="tight")
        plt.close()
        print(" Saved: permutation_importance.png/.pdf")

# Calibration and residuals
oof_file = DIR_HARDNESS / "OOF_predictions.csv"
if oof_file.exists():
    df_oof = pd.read_csv(oof_file)

    if not df_oof.empty and res_file.exists():
        # Ensure r2_col is defined (fallback to standard if above block failed)
        r2_col = next((c for c in df_res.columns if "R2" in c.upper() or "R²" in c.upper()), "CV_R2")
        rmse_col = next((c for c in df_res.columns if "RMSE" in c.upper()), "CV_RMSE")
        mae_col = next((c for c in df_res.columns if "MAE" in c.upper()), "CV_MAE")

        # Robustly select the best model based on R2
        best_model_row = df_res.sort_values(r2_col, ascending=False).iloc[0]
        selected_model_name = best_model_row["Model"]
        r2_val = best_model_row[r2_col]

        # Safely fetch RMSE and MAE for textbox
        rmse_val = best_model_row.get(rmse_col, np.nan) if rmse_col in df_res.columns else np.nan
        mae_val = best_model_row.get(mae_col, np.nan) if mae_col in df_res.columns else np.nan

        df_selected = (
            df_oof[df_oof["Model"] == selected_model_name]
            .groupby("Sample_ID", as_index=False)[["Observed", "Predicted"]]
            .mean()
        )

        y_obs = df_selected["Observed"]
        y_pred = df_selected["Predicted"]
        residuals = y_obs - y_pred

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Observed vs Predicted
        sns.scatterplot(x=y_pred, y=y_obs, alpha=0.6, s=35, color=COLORS.get(selected_model_name, "blue"), edgecolor="black", ax=axs[0])
        lims = [
            min(y_obs.min(), y_pred.min()),
            max(y_obs.max(), y_pred.max())
        ]
        axs[0].plot(lims, lims, color="red", linestyle="--", linewidth=2)
        axs[0].set_xlim(lims)
        axs[0].set_ylim(lims)

        reg = linregress(y_pred, y_obs)

        # Regression stats moved to a clean textbox (Now with dynamically found RMSE and MAE)
        textstr = f'y = {reg.slope:.3f}x + {reg.intercept:.2f}\nR² = {r2_val:.3f}'
        if pd.notna(rmse_val): textstr += f'\nRMSE = {rmse_val:.3f}'
        if pd.notna(mae_val):  textstr += f'\nMAE = {mae_val:.3f}'

        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
        axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)

        axs[0].set_title("(a) Observed vs Predicted")
        axs[0].set_xlabel("Predicted Hardness (HV)")
        axs[0].set_ylabel("Observed Hardness (HV)")
        axs[0].grid(alpha=0.3)

        # 2. Residuals vs Predicted
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, s=35, color=COLORS.get(selected_model_name, "blue"), edgecolor="black", ax=axs[1])
        axs[1].axhline(0, color="red", linestyle="--", linewidth=2)
        axs[1].set_title("(b) Residuals vs Predicted")
        axs[1].set_xlabel("Predicted Hardness (HV)")
        axs[1].set_ylabel("Residual Error (HV)")
        axs[1].grid(alpha=0.3)

        # 3. Residual Distribution
        sns.histplot(residuals, bins=35, kde=True, color=COLORS.get(selected_model_name, "blue"), edgecolor="black", ax=axs[2])
        axs[2].set_title("(c) Residual Distribution")
        axs[2].set_xlabel("Residual Error (HV)")
        axs[2].set_ylabel("Frequency")
        axs[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(DIR_HARDNESS / "calibration_analysis.png", dpi=600, bbox_inches="tight")
        plt.savefig(DIR_HARDNESS / "calibration_analysis.pdf", dpi=600, bbox_inches="tight")
        plt.close()
        print(" Saved: calibration_analysis.png/.pdf")
