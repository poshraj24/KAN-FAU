import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _flatten_points_from_file(csv_path):
    """
    Read one per-gene summary CSV (with JSON arrays) and return
    flat arrays of per-sample log2FC (x) and predicted Gene Expression (y).
    Expects columns: 'predicted_values', 'log2_fold_changes' (JSON arrays).
    """
    df = pd.read_csv(csv_path)

    xs, ys = [], []
    for _, row in df.iterrows():
        # Parse arrays
        pv = row["predicted_values"]
        l2 = row["log2_fold_changes"]

        if isinstance(pv, str):
            pv = json.loads(pv)
        if isinstance(l2, str):
            l2 = json.loads(l2)

        # Ensure same length
        m = min(len(pv), len(l2))
        if m == 0:
            continue

        # Convert to numeric & drop pairs with missing
        pv_arr = np.array(
            [np.nan if v is None else float(v) for v in pv[:m]], dtype=float
        )
        l2_arr = np.array(
            [np.nan if v is None else float(v) for v in l2[:m]], dtype=float
        )

        valid = ~(np.isnan(pv_arr) | np.isnan(l2_arr))
        if valid.any():
            xs.append(l2_arr[valid])
            ys.append(pv_arr[valid])

    if not xs:
        return np.array([]), np.array([])

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return x, y


def improved_flow_plot_test_only(
    test_csv,
    output_prefix,
    log2fc_threshold=0.5,
    x_range_mode="auto",
    x_manual_range=None,
):
    """
    Test-only version that accommodates all points and shows accurate counts.

    Parameters:
    - x_range_mode: "auto" (fit all data), "manual" (use x_manual_range), "threshold" (±3*threshold)
    - x_manual_range: tuple (x_min, x_max) if x_range_mode="manual"
    """
    # ---- Load & flatten per-sample points ----
    xt, yt = _flatten_points_from_file(test_csv)

    if xt.size == 0 or yt.size == 0:
        raise RuntimeError(f"No valid points parsed from {test_csv}")

    # ---- Determine x-axis range ----
    all_x = xt

    if x_range_mode == "auto":
        # Use 5th and 95th percentiles to avoid extreme outliers
        x_min, x_max = np.percentile(all_x, [5, 95])
        # Add some padding
        x_range = x_max - x_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
    elif x_range_mode == "manual" and x_manual_range:
        x_min, x_max = x_manual_range
    elif x_range_mode == "threshold":
        # Show ±3 times the threshold (or at least ±1.5)
        x_range = max(3 * log2fc_threshold, 1.5)
        x_min, x_max = -x_range, x_range
    else:
        # Fallback to full range
        x_min, x_max = np.min(all_x), np.max(all_x)

    print(f"Data range: [{np.min(all_x):.2f}, {np.max(all_x):.2f}]")
    print(f"Plot range: [{x_min:.2f}, {x_max:.2f}]")

    # ---- Categorize points by threshold ----
    def categorize_with_counts(x, y, thr, x_min_plot, x_max_plot):
        under = x <= -thr
        acc = (x > -thr) & (x < thr)
        over = x >= thr

        # Count visible points
        visible = (x >= x_min_plot) & (x <= x_max_plot)

        under_vis = under & visible
        acc_vis = acc & visible
        over_vis = over & visible

        total_counts = (int(under.sum()), int(acc.sum()), int(over.sum()))
        visible_counts = (int(under_vis.sum()), int(acc_vis.sum()), int(over_vis.sum()))

        return under, acc, over, total_counts, visible_counts

    t_under, t_acc, t_over, t_total, t_visible = categorize_with_counts(
        xt, yt, log2fc_threshold, x_min, x_max
    )

    COLORS = {"under": "#1976D2", "accurate": "#388E3C", "over": "#D32F2F"}

    # Y-limits
    y_min = np.nanmin(yt)
    y_max = np.nanmax(yt)
    pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    y_lo, y_hi = y_min - pad, y_max + pad

    # ---- Create single figure for test only ----
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), dpi=150)

    # ---- Helper function to create labels ----
    def create_label(category, total_count, visible_count):
        # Always show total count only
        return f"{category} (n={total_count})"

    # Calculate total points for reference
    t_total_all = len(xt)

    # ---------- Test Plot ----------
    ax.scatter(
        xt[t_under],
        yt[t_under],
        color=COLORS["under"],
        s=10,
        alpha=0.7,
        linewidth=0.5,
        label=create_label("Under", t_total[0], t_visible[0]),
    )
    ax.scatter(
        xt[t_acc],
        yt[t_acc],
        color=COLORS["accurate"],
        s=10,
        alpha=0.7,
        linewidth=0.5,
        label=create_label("Accurate", t_total[1], t_visible[1]),
    )
    ax.scatter(
        xt[t_over],
        yt[t_over],
        color=COLORS["over"],
        s=10,
        alpha=0.7,
        linewidth=0.5,
        label=create_label("Over", t_total[2], t_visible[2]),
    )

    # Add threshold lines
    ax.axvline(x=-log2fc_threshold, color="gray", linestyle=":", lw=1, alpha=0.85)
    ax.axvline(x=+log2fc_threshold, color="gray", linestyle=":", lw=1, alpha=0.85)

    ax.set_title("Test", fontsize=12, pad=15)
    ax.set_ylabel("Predicted Gene Expression", fontsize=12)
    ax.set_xlabel("log₂ Fold Change", fontsize=12)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(x_min, x_max)

    # Custom x-axis ticks: show only min, -0.5, +0.5, max
    x_ticks = [x_min, -0.5, +0.5, x_max]
    x_labels = [f"{x_min:.1f}", "-0.5", "+0.5", f"{x_max:.1f}"]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    ax.tick_params(labelsize=12)
    ax.tick_params(axis="x", rotation=90)
    ax.grid(alpha=0.3, linestyle="-", linewidth=0.5)

    # Legend positioned at upper right
    ax.legend(
        fontsize=12,
        loc="upper center",
        frameon=True,
    )

    plt.tight_layout()

    filename = f"log2fold_{output_prefix}_overall_test_only_{x_range_mode}"
    plt.savefig(f"ZNF326/{filename}.pdf", dpi=1200, bbox_inches="tight")
    plt.savefig(f"ZNF326/{filename}.png", dpi=1200, bbox_inches="tight")

    print(f"\nPlot saved as: {filename}.pdf and {filename}.png")

    plt.show()
    return fig


print("=== AUTO RANGE (recommended) ===")
improved_flow_plot_test_only(
    test_csv="ZNF326/perturbation_model_evaluation_test_ZNF326_filtered_overall.csv",
    output_prefix="ZNF326",
    log2fc_threshold=0.5,
    x_range_mode="auto",
)
