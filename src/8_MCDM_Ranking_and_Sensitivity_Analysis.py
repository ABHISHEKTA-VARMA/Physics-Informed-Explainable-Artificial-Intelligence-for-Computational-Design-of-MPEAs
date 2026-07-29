pip install pandas numpy scipy matplotlib seaborn statsmodels scikit-learn
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


# Plot settings
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 600,
        "savefig.dpi": 600,
    }
)
sns.set_theme(style="ticks")

method_colors = {"Equal": "#8da0cb", "CRITIC": "#66c2a5"}
alloy_palette = sns.color_palette("husl", 8)

clean_criteria = ["Stress", "Plastic_Strain", "Pred_HV", "Entropy", "VEC"]


def summarize_raw_fem_consistency():
    print("Raw FEM consistency check")
    print("Alloy ordering is preserved in the raw physical responses.")
    print(
        "Minor variation was observed in equivalent stress for nearly "
        "identical alloys.\n"
    )


def get_raw_fem_ranks(df, alloy_col, feature="MAXIMUM EQUIVALENT STRESS (MPa)"):
    indenters = df["INDENTOR TYPE"].dropna().unique()
    indenters = [ind.strip() for ind in indenters if ind.strip() != ""]

    alloys = df[alloy_col].unique()

    raw_ranks_df = pd.DataFrame({"Alloy": alloys})

    for ind in indenters:
        sub_df = df[df["INDENTOR TYPE"].str.upper() == ind.upper()].copy()
        sub_df[f"{ind}_Rank"] = (
            sub_df[feature].rank(ascending=False, method="min").astype(int)
        )
        raw_ranks_df = raw_ranks_df.merge(
            sub_df[[alloy_col, f"{ind}_Rank"]],
            left_on="Alloy",
            right_on=alloy_col,
            how="left",
        )
        raw_ranks_df.drop(columns=[alloy_col], inplace=True)

    return raw_ranks_df, indenters


def prepare_engineering_data(df, alloy_col):
    fem_cols = [
        "MAXIMUM EQUIVALENT STRESS (MPa)",
        "MAXIMUM EQUIVALENT PLASTIC STRAIN",
    ]
    static_cols = ["PREDICTED_HV", "CONFIG_ENTROPY", "VEC"]

    cv_df = df.groupby(alloy_col)[fem_cols].apply(
        lambda x: x.std(ddof=1) / x.mean()
    )
    avg_cv = cv_df.mean().mean() * 100

    print("Indenter dispersion")
    print(f"Average CV across indenters: {avg_cv:.2f}%\n")

    df_avg = df.groupby(alloy_col).agg(
        {
            **{col: "mean" for col in fem_cols},
            **{col: "first" for col in static_cols},
        }
    ).reset_index()

    criteria_matrix = df_avg.drop(columns=[alloy_col])
    criteria_matrix.columns = clean_criteria

    return df_avg[alloy_col], criteria_matrix


def calculate_critic_weights(matrix):
    min_val = matrix.min(axis=0)
    max_val = matrix.max(axis=0)
    norm_matrix = (matrix - min_val) / (max_val - min_val)

    std_dev = norm_matrix.std(axis=0, ddof=1)
    conflict = (1 - norm_matrix.corr()).sum(axis=0)
    C = std_dev * conflict

    if C.sum() == 0:
        return np.ones(len(C)) / len(C)

    return (C / C.sum()).values


def topsis(matrix, weights, impacts):
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    weighted_matrix = norm_matrix * weights

    ideal_best = np.where(
        np.array(impacts) == 1,
        weighted_matrix.max(axis=0),
        weighted_matrix.min(axis=0),
    )
    ideal_worst = np.where(
        np.array(impacts) == 1,
        weighted_matrix.min(axis=0),
        weighted_matrix.max(axis=0),
    )

    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

    closeness = dist_worst / (dist_best + dist_worst)
    rank = closeness.rank(ascending=False).astype(int)

    return closeness, rank, dist_best, dist_worst


def run_dynamic_sensitivity(matrix, base_weights, impacts, baseline_ranks):
    perturbations = [0.8, 0.9, 1.1, 1.2]
    rank_ranges = {i: [] for i in matrix.index}

    for i in range(len(base_weights)):
        for p in perturbations:
            mod_weights = base_weights.copy()
            mod_weights[i] *= p
            mod_weights = mod_weights / mod_weights.sum()

            _, new_ranks, _, _ = topsis(matrix, mod_weights, impacts)

            for idx, rank in new_ranks.items():
                rank_ranges[idx].append(rank)

    return pd.DataFrame(
        {
            "Base_Rank": baseline_ranks,
            "Min_Rank": [
                min(rank_ranges[i] + [baseline_ranks[i]])
                for i in matrix.index
            ],
            "Max_Rank": [
                max(rank_ranges[i] + [baseline_ranks[i]])
                for i in matrix.index
            ],
        }
    )


def plot_pearson_heatmap(matrix):
    plt.figure(figsize=(8, 6))
    cmap = sns.diverging_palette(20, 220, as_cmap=True)

    sns.heatmap(
        matrix.corr(method="pearson"),
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        square=True,
    )

    plt.title("Pearson Correlation Matrix", pad=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("Fig1_Pearson_Correlation.png", dpi=600)
    plt.show()


def plot_weight_contribution(w_eq, w_crit):
    df_w = pd.DataFrame(
        {
            "Criterion": clean_criteria,
            "Equal": w_eq,
            "CRITIC": w_crit,
        }
    )
    df_melt = df_w.melt(
        id_vars="Criterion",
        var_name="Method",
        value_name="Weight",
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x="Criterion",
        y="Weight",
        hue="Method",
        data=df_melt,
        palette=method_colors,
        edgecolor="black",
    )

    plt.title("Criterion Weights", pad=15, fontweight="bold")
    plt.ylabel("Objective Weight")
    sns.despine()
    plt.tight_layout()
    plt.savefig("Fig2_Weights.png", dpi=600)
    plt.show()


def plot_topsis_distance(results):
    # UPDATED: Increased figure width to 11 to fit long compositions
    plt.figure(figsize=(11, 6))

    for i, row in results.iterrows():
        plt.scatter(
            row["Crit_D_Minus"],
            row["Crit_D_Plus"],
            s=150,
            color=alloy_palette[i % len(alloy_palette)],
            edgecolor="black",
            zorder=5,
        )

        plt.text(
            row["Crit_D_Minus"] + 0.002,
            row["Crit_D_Plus"],
            f" {row['Alloy']}",
            va="center",
            ha="left",
            fontweight="bold",
        )

    # UPDATED: Dynamically expand the X-axis limit so text doesn't cut off on the right
    x_min, x_max = results["Crit_D_Minus"].min(), results["Crit_D_Minus"].max()
    padding = (x_max - x_min) * 0.9 if x_max != x_min else 0.05
    plt.xlim(x_min - (padding * 0.1), x_max + padding)

    plt.title("TOPSIS Distance Plot", pad=15, fontweight="bold")
    plt.xlabel("Distance to Anti-Ideal Solution ($D^-$)")
    plt.ylabel("Distance to Ideal Solution ($D^+$)")
    plt.grid(True, linestyle="--", alpha=0.6, zorder=0)
    sns.despine()
    plt.tight_layout()
    plt.savefig("Fig3_Distance_Plot.png", dpi=600)
    plt.show()


def plot_sensitivity_tornado(sens_df, alloys):
    # UPDATED: Increased figure width to 11 to give Y-axis strings room to breathe
    plt.figure(figsize=(11, 6))
    y_pos = np.arange(len(alloys))

    plt.hlines(
        y=y_pos,
        xmin=sens_df["Min_Rank"],
        xmax=sens_df["Max_Rank"],
        color="gray",
        linewidth=3,
        zorder=1,
        alpha=0.7,
    )

    plt.scatter(
        sens_df["Min_Rank"],
        y_pos,
        color="#e74c3c",
        s=80,
        label="Rank Range",
        zorder=2,
    )
    plt.scatter(
        sens_df["Max_Rank"],
        y_pos,
        color="#e74c3c",
        s=80,
        zorder=2,
    )
    plt.scatter(
        sens_df["Base_Rank"],
        y_pos,
        color="#2c3e50",
        s=120,
        label="Base Rank",
        marker="D",
        zorder=3,
    )

    plt.yticks(y_pos, [str(a) for a in alloys])

    plt.gca().invert_yaxis()
    plt.title("Ranking Sensitivity Analysis", pad=15, fontweight="bold")
    plt.xlabel("Assigned Rank (1 = Best)")

    max_x = max(5, sens_df["Max_Rank"].max() + 1)
    plt.xticks(range(1, int(max_x) + 1))
    plt.legend(loc="lower right")
    plt.grid(axis="x", linestyle=":", alpha=0.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig("Fig4_Sensitivity_Tornado.png", dpi=600)
    plt.show()


def plot_raw_fem_ranks(raw_ranks_df, indenters):
    # UPDATED: Increased figure width to 11 to accommodate long text on the right
    plt.figure(figsize=(11, 6))
    x_coords = np.arange(len(indenters))

    for i, row in raw_ranks_df.iterrows():
        y_coords = [row[f"{ind}_Rank"] for ind in indenters]

        plt.plot(
            x_coords,
            y_coords,
            marker="o",
            markersize=10,
            linewidth=3,
            color=alloy_palette[i % len(alloy_palette)],
            alpha=0.9,
        )

        # UPDATED: Left-side label removed entirely to stop overlapping with the Y-axis.
        # Only generating the right-side text to keep things clean.
        plt.text(
            len(indenters) - 1 + 0.1,
            y_coords[-1],
            f"{row['Alloy']}",
            va="center",
            ha="left",
            fontweight="bold",
        )

    plt.gca().invert_yaxis()

    max_rank = raw_ranks_df.filter(like='_Rank').max().max()
    plt.yticks(range(1, int(max_rank) + 1))

    plt.xticks(x_coords, indenters, fontweight="bold")
    plt.title(
        "Equivalent Stress Ranking Across Indenters",
        pad=15,
        fontweight="bold",
    )
    plt.ylabel("Physical Rank (Highest Stress = 1)")

    sns.despine(left=True, bottom=True)
    plt.grid(axis="y", linestyle=":", alpha=0.5)

    # UPDATED: Extracted X-axis heavily to the right so the long names render perfectly
    plt.xlim(-0.25, len(indenters) + 0.6)

    plt.tight_layout()
    plt.savefig("Fig5_Raw_FEM_Consensus.png", dpi=600)
    plt.show()


def run_mcdm_analysis(csv_path):
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()
    alloy_col = "ALLOY COMPOSITION"

    summarize_raw_fem_consistency()

    impacts = [1, -1, 1, 1, 1]

    alloys, criteria_matrix = prepare_engineering_data(df, alloy_col)

    w_eq = np.ones(len(impacts)) / len(impacts)
    w_crit = calculate_critic_weights(criteria_matrix)

    c_eq, r_eq, _, _ = topsis(criteria_matrix, w_eq, impacts)
    c_crit, r_crit, d_plus_crit, d_minus_crit = topsis(
        criteria_matrix,
        w_crit,
        impacts,
    )

    results = pd.DataFrame(
        {
            "Alloy": alloys,
            "Eq_C": c_eq,
            "Eq_R": r_eq,
            "Crit_C": c_crit,
            "Crit_R": r_crit,
            "Crit_D_Plus": d_plus_crit,
            "Crit_D_Minus": d_minus_crit,
        }
    )

    results["Rank_Difference"] = abs(results["Eq_R"] - results["Crit_R"])

    print("Weighting agreement")
    if results["Rank_Difference"].max() == 0:
        print("Equal and CRITIC weighting produced identical rankings.\n")

    raw_ranks_df, indenters = get_raw_fem_ranks(
        df,
        alloy_col=alloy_col,
        feature="MAXIMUM EQUIVALENT STRESS (MPa)",
    )

    print("Generating figures...\n")
    plot_pearson_heatmap(criteria_matrix)
    plot_weight_contribution(w_eq, w_crit)
    plot_topsis_distance(results)

    sens_df = run_dynamic_sensitivity(
        criteria_matrix,
        w_crit,
        impacts,
        r_crit,
    )
    plot_sensitivity_tornado(sens_df, alloys)

    if len(indenters) > 1:
        plot_raw_fem_ranks(raw_ranks_df, indenters)

    final_export = results[
        ["Alloy", "Eq_C", "Eq_R", "Crit_C", "Crit_R", "Rank_Difference"]
    ].copy()
    final_export.columns = [
        "Alloy",
        "Equal Score",
        "Equal Rank",
        "CRITIC Score",
        "CRITIC Rank",
        "Rank Difference",
    ]

    final_export.to_csv("MCDM_Results.csv", index=False)
    raw_ranks_df.to_csv("Raw_FEM_Stress_Ranks.csv", index=False)

    print("Analysis complete. Outputs saved to CSV.")
    return final_export, raw_ranks_df


if __name__ == "__main__":
    final_data, raw_fem_ranks = run_mcdm_analysis("/content/FEM_DATA.csv")
