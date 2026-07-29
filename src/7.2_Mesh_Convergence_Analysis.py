from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator


file_path = Path("/content/MESH_SENSITIVITY_INDENTORS.csv")

if not file_path.is_file():
    print(f"Missing input file: {file_path}")
else:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.5,
            "xtick.major.size": 6,
            "xtick.minor.size": 3,
            "xtick.major.width": 1.2,
            "xtick.minor.width": 1.0,
            "ytick.major.size": 6,
            "ytick.minor.size": 3,
            "ytick.major.width": 1.2,
            "ytick.minor.width": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 600,
        }
    )

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    mesh_col = df.columns[0]
    force_cols = [
        "VICKERS - Force Reaction Maximum Total [N]",
        "BERKOVICH - Force Reaction Maximum Total [N]",
        "KNOOP - Force Reaction Maximum Total [N]",
    ]

    panel_labels = [
        "(a) Vickers",
        "(b) Berkovich",
        "(c) Knoop",
    ]

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
    ]

    label_offsets = [
        (-25, -28),
        (18, 28),
        (-25, -28),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    axes = axes.flatten()

    selected_mesh = 0.20

    for idx, force_col in enumerate(force_cols):
        ax = axes[idx]

        df_plot = (
            df[[mesh_col, force_col]]
            .dropna()
            .sort_values(mesh_col, ascending=False)
            .reset_index(drop=True)
        )

        ax.plot(
            df_plot[mesh_col],
            df_plot[force_col],
            marker="o",
            color=colors[idx],
            markersize=7,
            mfc="white",
            mew=2.2,
            linewidth=2.3,
            zorder=3,
        )

        selected_force = df_plot.loc[
            (df_plot[mesh_col] - selected_mesh).abs().idxmin(),
            force_col,
        ]

        ax.scatter(
            selected_mesh,
            selected_force,
            s=120,
            zorder=5,
            marker="s",
            color="black",
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.15,
            linewidth=0.5,
            zorder=0,
        )

        ax.set_xlabel("Mesh size (mm)")
        ax.set_ylabel("Maximum reaction force (N)")
        ax.invert_xaxis()

        ax.margins(y=0.05)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        fine = df_plot[df_plot[mesh_col] <= selected_mesh]
        variation = (
            (fine[force_col].max() - fine[force_col].min())
            / fine[force_col].mean()
            * 100
        )

        ax.text(
            0.95,
            0.95,
            f"Variation ($\\leq$0.20 mm)\n{variation:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9,
            ),
            zorder=6,
        )

        ax.annotate(
            "Adopted mesh\n0.20 mm",
            xy=(selected_mesh, selected_force),
            xytext=label_offsets[idx],
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=10,
            ha="center",
            va="center",
            zorder=6,
        )

        ax.text(
            0.03,
            0.95,
            panel_labels[idx],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            zorder=6,
        )

    plt.tight_layout(w_pad=2.0)

    output_file = "Fig_Mesh_Convergence_Final.png"
    plt.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"Saved: {output_file}")
