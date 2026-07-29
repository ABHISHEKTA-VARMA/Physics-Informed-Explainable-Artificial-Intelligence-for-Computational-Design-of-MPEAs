import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("ML_FEM_CONSISTENCY.csv")

labels = df["ALLOY_COMPOSITION"]
hv_ml = df["ML_PREDICTED_HV"]
hv_fem = df["FEM_ESTIMATED_HV"]

n_alloys = len(df)
r2 = r2_score(hv_ml, hv_fem)
pearson_r = np.corrcoef(hv_ml, hv_fem)[0, 1]
mae = mean_absolute_error(hv_ml, hv_fem)
rmse = np.sqrt(mean_squared_error(hv_ml, hv_fem))
mape = np.mean(np.abs((hv_ml - hv_fem) / hv_ml)) * 100

df["ERROR_%"] = (np.abs(hv_fem - hv_ml) / hv_ml) * 100

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    }
)

# ---------------------------------------------------------
# Figure 1: Parity Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))

limits = [
    min(hv_ml.min(), hv_fem.min()),
    max(hv_ml.max(), hv_fem.max()),
]
parity_line = np.linspace(limits[0] * 0.98, limits[1] * 1.02, 500)

ax.fill_between(
    parity_line,
    parity_line * 0.95,
    parity_line * 1.05,
    alpha=0.12,
    color="tab:orange",
    label="±5% Error Band",
)

ax.plot(
    parity_line,
    parity_line,
    "--",
    color="black",
    linewidth=2.5,
    label="Parity Line",
)

ax.scatter(
    hv_ml,
    hv_fem,
    s=120,
    marker="o",
    color="tab:blue",
    edgecolors="black",
    linewidth=0.8,
    label="Validation Data",
)

metrics_text = (
    "Regression Metrics\n\n"
    f"N = {n_alloys}\n"
    f"$R^2$ = {r2:.3f}\n"
    f"Pearson r = {pearson_r:.3f}\n"
    f"MAE = {mae:.2f} HV\n"
    f"RMSE = {rmse:.2f} HV\n"
    f"MAPE = {mape:.2f}%"
)

ax.text(
    0.05,
    0.95,
    metrics_text,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(
        facecolor="white",
        edgecolor="black",
        alpha=0.9,
    ),
)

ax.set_xlabel("ML Predicted Hardness (HV)")
ax.set_ylabel("FEM Estimated Hardness (HV)")
ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=True, loc="lower right")

plt.tight_layout()
plt.savefig("Figure_1_Parity_Plot.tiff", dpi=600, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------
# Figure 2: Error Analysis Bar Chart
# ---------------------------------------------------------
error_df = df[["ERROR_%"]].copy()
error_df["Label"] = labels
error_df = error_df.sort_values(by="ERROR_%", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(
    error_df["Label"],
    error_df["ERROR_%"],
    color="tab:blue",
)

ax.axvline(
    5,
    color="tab:red",
    linestyle="--",
    linewidth=2,
    label="5% error limit",
)

ax.set_xlabel("Absolute Percentage Error (%)")
ax.set_ylabel("Alloy Composition") # Updated axis label for clarity
ax.set_title("Prediction Error by Alloy")
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("Figure_2_Error_Analysis.tiff", dpi=600, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------
# Summary Output
# ---------------------------------------------------------
summary_df = pd.DataFrame(
    {
        "Metric": [
            "N",
            "R²",
            "Pearson r",
            "MAE (HV)",
            "RMSE (HV)",
            "MAPE (%)",
            "Maximum Error (%)",
        ],
        "Value": [
            n_alloys,
            round(r2, 3),
            round(pearson_r, 3),
            round(mae, 2),
            round(rmse, 2),
            round(mape, 2),
            round(df["ERROR_%"].max(), 2),
        ],
    }
)

print("\nML-FEM validation summary")
print(summary_df.to_string(index=False))
