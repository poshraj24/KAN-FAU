import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms as mtransforms


def improved_mean_plot_test_only(
    test_csv,
    output_prefix,
    log2fc_threshold=0.5,
    x_range_mode="auto",
    x_manual_range=None,
):
    """
    Test-only version for mean data that accommodates all points and shows accurate counts.
    Each row is a gene with single mean values (not arrays).

    Parameters:
    - x_range_mode: "auto" (fit all data), "manual" (use x_manual_range), "threshold" (±3*threshold)
    - x_manual_range: tuple (x_min, x_max) if x_range_mode="manual"
    """
    # --- Load & sort ---
    dt = pd.read_csv(test_csv).sort_values("log2_fold_change").reset_index(drop=True)

    if len(dt) == 0:
        raise RuntimeError(f"No valid data in {test_csv}")

    # ---- Determine x-axis range ----
    all_x = dt["log2_fold_change"]
    data_min = float(all_x.min())
    data_max = float(all_x.max())

    if x_range_mode == "auto":
        # Use 5th and 95th percentiles to avoid extreme outliers
        x_min, x_max = np.percentile(all_x, [5, 95])
        # Add some padding
        x_range = data_max - data_min
        padding = x_range * 0.05 if x_range > 0 else 0.1
        x_min, x_max = data_min - padding, data_max + padding
    elif x_range_mode == "manual" and x_manual_range:
        x_min, x_max = x_manual_range
    elif x_range_mode == "threshold":
        # Show ±3 times the threshold (or at least ±1.5)
        x_range = max(3 * log2fc_threshold, 1.5)
        x_min, x_max = -x_range, x_range
    else:
        # Fallback to full range
        x_min, x_max = data_min, data_max

    print(f"Data range: [{data_min:.2f}, {data_max:.2f}]")
    print(f"Plot range: [{x_min:.2f}, {x_max:.2f}]")

    # ---- Categorize genes by threshold ----
    def categorize_genes(df, thr):
        under = df["log2_fold_change"] <= -thr
        acc = (df["log2_fold_change"] > -thr) & (df["log2_fold_change"] < thr)
        over = df["log2_fold_change"] >= thr

        total_counts = (int(under.sum()), int(acc.sum()), int(over.sum()))
        return under, acc, over, total_counts

    t_under, t_acc, t_over, t_total = categorize_genes(dt, log2fc_threshold)

    COLORS = {"under": "#1976D2", "accurate": "#388E3C", "over": "#D32F2F"}

    # Y-limits
    y_all = dt["predicted_value"]
    y_min, y_max = y_all.min(), y_all.max()
    pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    y_lo, y_hi = y_min - pad, y_max + pad

    # ---- Create single figure for test only ----
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150, constrained_layout=True)
    plt.subplots_adjust(bottom=0.22)

    # ---- Helper function to create labels ----
    def create_label(category, total_count):
        return f"{category} (n={total_count})"

    # ---------- Test Plot ----------
    ax.scatter(
        dt.loc[t_under, "log2_fold_change"],
        dt.loc[t_under, "predicted_value"],
        color=COLORS["under"],
        s=10,
        alpha=0.7,
        linewidth=0.5,
        label=create_label("Under", t_total[0]),
    )
    ax.scatter(
        dt.loc[t_acc, "log2_fold_change"],
        dt.loc[t_acc, "predicted_value"],
        color=COLORS["accurate"],
        s=10,
        alpha=0.7,
        linewidth=0.5,
        label=create_label("Accurate", t_total[1]),
    )
    ax.scatter(
        dt.loc[t_over, "log2_fold_change"],
        dt.loc[t_over, "predicted_value"],
        color=COLORS["over"],
        s=10,
        alpha=0.7,
        linewidth=0.5,
        label=create_label("Over", t_total[2]),
    )

    # Add threshold lines
    ax.axvline(x=-log2fc_threshold, color="gray", linestyle=":", lw=1, alpha=0.85)
    ax.axvline(x=+log2fc_threshold, color="gray", linestyle=":", lw=1, alpha=0.85)

    ax.set_title("Test", fontsize=12, pad=15)
    ax.set_ylabel("Predicted Mean Gene Expression", fontsize=12)
    ax.set_xlabel("log₂ Fold Change", fontsize=12)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(x_min, x_max)

    # Custom x-axis ticks: show actual data min/max, -0.5, +0.5
    x_ticks = [data_min, -0.5, +0.5, data_max]
    x_labels = [f"{data_min:.2f}", "-0.5", "+0.5", f"{data_max:.2f}"]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    ax.tick_params(labelsize=11)
    ax.tick_params(axis="x", rotation=90)
    ax.grid(alpha=0.3, linestyle="-", linewidth=0.5)

    # Legend positioned at upper right
    ax.legend(
        fontsize=12,
        loc="upper right",
        frameon=True,
    )

    plt.tight_layout()


    filename = f"log2fold_{output_prefix}_mean_test_only_{x_range_mode}"
    plt.savefig(f"ZNF326/{filename}.pdf", dpi=1200, bbox_inches="tight")
    plt.savefig(f"ZNF326/{filename}.png", dpi=1200, bbox_inches="tight")

    print(f"\nPlot saved as: {filename}.pdf and {filename}.png")
 

    plt.show()
    return fig

improved_mean_plot_test_only(
    test_csv="ZNF326/perturbation_model_evaluation_test_ZNF326_filtered.csv",
    output_prefix="ZNF326",
    log2fc_threshold=0.5,
    x_range_mode="auto",
)
