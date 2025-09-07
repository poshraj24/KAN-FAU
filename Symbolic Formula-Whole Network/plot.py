#!/usr/bin/env python3
"""
Pipeline:

A) Read per-gene TSVs from actual_vs_predicted/ and build a per-gene table of:
      gene_name, mean_actual, mean_predicted, n_samples
   -> Plot a single scatter with one point per gene:
      mean(actual) vs mean(predicted), with y=x and a least-squares line.
      Saves PNG (1200 dpi) and PDF.

   -> NEW: All-points plot (every sample across all genes)
      actual vs predicted with y=x, ±RMSE band, OLS line, and metrics.

B) Read core_results.tsv and
   1) Plot side-by-side histograms of Pearson and Spearman correlations.
   2) Plot box plots (NOT bars) for counts of significant genes:
        - using adjusted p-values (required)
        - using raw p-values (optional, if columns exist)
      Each box shows the count and annotates the number above the box.
   3) Plot box plots of correlation values grouped by adjusted significance
      (Pearson and Spearman separately).

All figures are saved into ./plots_out
Font size for all text is 12; PNGs at 1200 dpi by default.
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, DrawingArea
from matplotlib.lines import Line2D
from scipy import stats
import sys


ATT_FIGSIZE = (4, 4)
ATT_POINT_KW = dict(
    color="steelblue", s=10, alpha=0.7, edgecolors="white", linewidth=0.5
)
ATT_YX_LINE_KW = dict(color="black", linestyle="--", linewidth=2, alpha=0.85)
ATT_FIT_LINE_KW = dict(color="orange", linewidth=2, alpha=0.9)
ATT_LEGEND_KW = dict(fontsize=12, loc="lower right", frameon=False)

# ======================= CONFIG =======================
ROOT = Path("KAN_Project/core_evaluation_results")

# Folder with Gene_X.tsv files
DIR_AVP = ROOT / "actual_vs_predicted"

# Core results file
CORE_RESULTS = ROOT / "core_results.tsv"

# Output folder for plots
OUT_DIR = Path("KAN_Project/plots_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Appearance / thresholds
FONT_SIZE = 12
DPI = 1200
ADJ_P_THRESH = 0.05
RAW_P_THRESH = 0.05


# Numerical stability thresholds
MIN_VARIANCE_THRESHOLD = 1e-6  # Minimum variance for numerical stability


def format_pvalue(p: float) -> str:
    """
    Format p-value for display. Handles very small p-values robustly.
    """
    if np.isnan(p) or p < 0:
        return "p = N/A"

    # Handle extremely small p-values
    if p == 0.0 or p < sys.float_info.min:
        return f"p < {sys.float_info.min:.0e}"
    elif p < 1e-15:
        return "p < 1e-15"
    elif p < 1e-10:
        return f"p = {p:.0e}"
    elif p < 0.001:
        return f"p = {p:.2e}"
    else:
        return f"p = {p:.3f}"


def _add_fit_title(ax, slope, intercept, color="orange", fontsize=12):
    """Title-like badge centered above the axes: yellow line + 'Fit: y=...'."""
    da = DrawingArea(28, 10, 0, 0)
    da.add_artist(Line2D([0, 28], [5, 5], lw=2, color=color))
    txt = TextArea(
        rf"Fit: $y={slope:.3f}x{intercept:+.3f}$", textprops=dict(fontsize=fontsize)
    )
    box = HPacker(children=[da, txt], align="center", pad=0, sep=6)
    ab = AnnotationBbox(
        box,
        (0.5, 1.02),
        xycoords=ax.transAxes,
        box_alignment=(0.5, 0.0),
        frameon=False,
        pad=0.0,
        annotation_clip=False,
        zorder=6,
    )
    ax.add_artist(ab)


# --------- Finite-value helper (drop NaN/Inf everywhere) ----------
def _valid_xy(x_arr, y_arr):
    """Return x,y filtered to finite values only (ignore NaN/Inf)."""
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _filter_for_numerical_stability(means_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter genes to ensure numerical stability in regression:
    1. Remove genes with identical actual/predicted means
    2. Ensure sufficient variance across all remaining genes
    """
    # Start with all finite data
    df_clean = means_df.dropna().copy()

    # Remove genes where actual and predicted means are identical
    df_clean = df_clean[df_clean["mean_actual"] != df_clean["mean_predicted"]]

    if len(df_clean) < 3:
        print(f"Only {len(df_clean)} genes remaining after filtering identical means")
        return df_clean

    # Check variance across genes
    var_actual = np.var(df_clean["mean_actual"])
    var_predicted = np.var(df_clean["mean_predicted"])

    print(f"Variance in mean_actual across genes: {var_actual:.6f}")
    print(f"Variance in mean_predicted across genes: {var_predicted:.6f}")

    # If variance is too low, try removing extreme outliers
    if var_actual < MIN_VARIANCE_THRESHOLD or var_predicted < MIN_VARIANCE_THRESHOLD:
        print("Low variance detected, attempting outlier removal...")

        # Remove extreme outliers using IQR method
        for col in ["mean_actual", "mean_predicted"]:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            before_count = len(df_clean)
            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            ]
            after_count = len(df_clean)

            if before_count != after_count:
                print(f"Removed {before_count - after_count} outliers based on {col}")

        # Recalculate variance after outlier removal
        if len(df_clean) >= 3:
            var_actual = np.var(df_clean["mean_actual"])
            var_predicted = np.var(df_clean["mean_predicted"])
            print(f"Post-filtering variance in mean_actual: {var_actual:.6f}")
            print(f"Post-filtering variance in mean_predicted: {var_predicted:.6f}")

    return df_clean


# -------------------- Helpers: reading data --------------------
def read_gene_means(avp_dir: Path) -> pd.DataFrame:
    """
    Read all TSVs in `avp_dir` with header:
        sample_index  actual_value  predicted_value
    Return a DataFrame with per-gene means:
        gene_name, mean_actual, mean_predicted, n_samples
    """
    rows = []
    tsv_files = sorted(glob.glob(str(avp_dir / "*.tsv")))
    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in {avp_dir}")

    for fp in tsv_files:
        gene_name = Path(fp).stem
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception:
            df = pd.read_csv(fp)  # fallback

        # Normalize column names
        lower_map = {c.lower(): c for c in df.columns}
        a_col = lower_map.get("actual_value")
        p_col = lower_map.get("predicted_value")
        if a_col is None or p_col is None:
            raise ValueError(
                f"Expected columns 'actual_value' and 'predicted_value' in {fp}. "
                f"Found: {df.columns.tolist()}"
            )

        a = pd.to_numeric(df[a_col], errors="coerce")
        p = pd.to_numeric(df[p_col], errors="coerce")

        # means are computed on valid (finite) values only
        a = a[np.isfinite(a)]
        p = p[np.isfinite(p)]

        n = min(len(a), len(p))
        if n == 0:
            continue

        rows.append(
            {
                "gene_name": gene_name,
                "mean_actual": float(a.iloc[:n].mean()),
                "mean_predicted": float(p.iloc[:n].mean()),
                "n_samples": int(n),
            }
        )

    if not rows:
        raise RuntimeError("No valid gene rows parsed from TSVs.")
    return pd.DataFrame(rows)


def read_all_points(avp_dir: Path) -> pd.DataFrame:
    """
    Stack ALL (actual, predicted) rows across all gene files.
    Returns a DataFrame with columns: actual_value, predicted_value
    """
    tsv_files = sorted(glob.glob(str(avp_dir / "*.tsv")))
    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in {avp_dir}")
    chunks = []
    for fp in tsv_files:
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception:
            df = pd.read_csv(fp)
        lower_map = {c.lower(): c for c in df.columns}
        a_col = lower_map.get("actual_value")
        p_col = lower_map.get("predicted_value")
        if a_col is None or p_col is None:
            continue
        a = pd.to_numeric(df[a_col], errors="coerce")
        p = pd.to_numeric(df[p_col], errors="coerce")
        # keep only finite rows
        m = np.isfinite(a) & np.isfinite(p)
        if m.any():
            chunks.append(
                pd.DataFrame(
                    {
                        "actual_value": a[m].astype(float),
                        "predicted_value": p[m].astype(float),
                    }
                )
            )
    if not chunks:
        raise RuntimeError("No valid (actual, predicted) pairs found.")
    return pd.concat(chunks, ignore_index=True)


# -------------------- NEW: All-points plot --------------------
def all_points_deviation_plot(points_df: pd.DataFrame, out_dir: Path):
    """
    All points scatter with y=x and OLS fit.
    """
    x = points_df["actual_value"].to_numpy(float)
    y = points_df["predicted_value"].to_numpy(float)

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        print("Not enough finite points for all-points plot. Skipping.")
        return

    # Outliers removed for regression only
    def _rm_outliers(x, y, k=3):
        q1x, q3x = np.percentile(x, [25, 75])
        q1y, q3y = np.percentile(y, [25, 75])
        iqr_x, iqr_y = q3x - q1x, q3y - q1y
        lx, ux = q1x - k * iqr_x, q3x + k * iqr_x
        ly, uy = q1y - k * iqr_y, q3y + k * iqr_y
        keep = (x >= lx) & (x <= ux) & (y >= ly) & (y <= uy)
        return x[keep], y[keep]

    x_clean, y_clean = _rm_outliers(x, y)
    if x_clean.size < 2:
        x_clean, y_clean = x, y

    # Calculate correlation with proper p-value formatting
    r, p_scipy = stats.pearsonr(x_clean, y_clean)
    p_text = format_pvalue(p_scipy)

    slope, intercept, _, _, _ = stats.linregress(x_clean, y_clean)

    # Styles
    point_kw = globals().get(
        "ATT_POINT_KW",
        dict(color="steelblue", s=20, alpha=0.7, edgecolors="white", linewidth=0.5),
    )
    yx_kw = globals().get(
        "ATT_YX_LINE_KW", dict(color="black", linestyle="--", linewidth=2, alpha=0.85)
    )
    fit_kw = globals().get(
        "ATT_FIT_LINE_KW", dict(color="orange", linewidth=2, alpha=0.9)
    )

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    # Scatter (ALL points)
    ax.scatter(x, y, **point_kw, rasterized=True)

    # Fixed limits / ticks
    X_MIN, X_MAX = 0, 10
    Y_MIN, Y_MAX = -4, 16
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(np.arange(X_MIN, X_MAX + 1, 2))
    ax.set_yticks(np.arange(Y_MIN, Y_MAX + 1, 2))

    # y = x (no label)
    grid = np.linspace(min(X_MIN, Y_MIN), max(X_MAX, Y_MAX), 200)
    ax.plot(grid, grid, **yx_kw)

    # OLS fit
    with np.errstate(over="ignore", invalid="ignore"):
        y_fit = np.clip(slope * grid + intercept, -1e10, 1e10)
        mfit = np.isfinite(y_fit)
        if np.any(mfit):
            fit_label = f"Fit: y = {slope:.3f}x{intercept:+.3f}"
            ax.plot(grid[mfit], y_fit[mfit], label=fit_label, **fit_kw)

    # Labels & title (r, p only) - use formatted p-value
    ax.set_xlabel("Actual Gene Expression Value", fontsize=FONT_SIZE)
    ax.set_ylabel("Predicted Gene Expression Value", fontsize=FONT_SIZE)
    ax.set_title(f"r = {r:.3f}    {p_text}", fontsize=FONT_SIZE, pad=10)

    # Bottom-right box the fit entry
    leg = ax.legend(loc="lower right", frameon=True, fontsize=FONT_SIZE - 1)
    if leg is not None:
        leg.set_title(None)

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "all_points_deviation_from_perfect_fit.png"
    pdf = out_dir / "all_points_deviation_from_perfect_fit.pdf"
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Saved all-points plot: {png} | {pdf}")


# -------------------- Mean scatter --------------------
def scatter_means(
    means_df: pd.DataFrame,
    out_dir: Path,
    dpi: int = DPI,
    x_lim_left: float = 0.0,
    x_lim_right: float = 10.0,
    y_lim_down: float = -2.0,
    y_lim_up: float = 12.0,
    tick_step: float = 2.0,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    means_df_filtered = _filter_for_numerical_stability(means_df)
    if len(means_df_filtered) < 3:
        print("Insufficient genes remaining for regression analysis")
        return

    x = pd.to_numeric(means_df_filtered["mean_actual"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(means_df_filtered["mean_predicted"], errors="coerce").to_numpy(
        float
    )
    x, y = _valid_xy(x, y)

    point_kw = globals().get(
        "ATT_POINT_KW",
        dict(color="steelblue", s=20, alpha=0.7, edgecolors="white", linewidth=0.5),
    )
    yx_kw = globals().get(
        "ATT_YX_LINE_KW", dict(color="black", linestyle="--", linewidth=2, alpha=0.85)
    )
    fit_kw = globals().get(
        "ATT_FIT_LINE_KW", dict(color="orange", linewidth=2, alpha=0.9)
    )

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    ax.scatter(x, y, **point_kw)

    ax.set_xlim(x_lim_left, x_lim_right)
    ax.set_ylim(y_lim_down, y_lim_up)
    ax.set_aspect("equal", adjustable="box")
    xs = np.linspace(x_lim_left, x_lim_right, 200)

    ax.plot(xs, xs, **yx_kw)  # y = x

    if x.size >= 2:
        slope, intercept, r, p_scipy, _ = stats.linregress(x, y)
        ax.plot(xs, slope * xs + intercept, **fit_kw)

        # Format p-value properly
        p_text = format_pvalue(p_scipy)

        # r, p in TOP-LEFT
        ax.text(
            0.02,
            0.98,
            f"r = {r:.3f}\n{p_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE - 2,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
        )

        # Title area: yellow line + Fit: y=...
        _add_fit_title(
            ax,
            slope,
            intercept,
            color=fit_kw.get("color", "orange"),
            fontsize=FONT_SIZE - 2,
        )
    else:
        ax.text(
            0.02,
            0.98,
            "r = n/a\np = n/a",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE - 2,
        )

    ax.set_xlabel("Actual Mean Gene Expression Value", fontsize=FONT_SIZE - 2)
    ax.set_ylabel("Predicted Mean Gene Expression Value", fontsize=FONT_SIZE - 2)
    ax.set_xticks(np.arange(x_lim_left, x_lim_right + 1e-9, tick_step))
    ax.set_yticks(np.arange(y_lim_down, y_lim_up + 1e-9, tick_step))
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 2)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # leave space for the title badge above the axes
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])

    png = out_dir / "all_genes_mean_actual_vs_predicted.png"
    pdf = out_dir / "all_genes_mean_actual_vs_predicted.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)


def read_core_results(path: Path) -> pd.DataFrame:
    """
    Read core_results.tsv with columns:
      gene_name n_samples pearson_correlation pearson_adjusted_p_value pearson_significant
      spearman_correlation spearman_adjusted_p_value spearman_significant error

    """
    df = pd.read_csv(path, sep="\t")

    # Coerce numerics
    for c in [
        "pearson_correlation",
        "spearman_correlation",
        "pearson_adjusted_p_value",
        "spearman_adjusted_p_value",
        "pearson_p_value",
        "spearman_p_value",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Coerce booleans (if present)
    for c in ["pearson_significant", "spearman_significant"]:
        if c in df.columns and df[c].dtype != bool:
            df[c] = df[c].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])

    return df


def correlation_histograms(core_df: pd.DataFrame, out_dir: Path):
    """
    Side-by-side histograms for Pearson and Spearman correlation coefficients.
    Uses only finite values (NaN/Inf ignored).
    """

    def _finite(series):
        arr = pd.to_numeric(series, errors="coerce").to_numpy(float)
        return arr[np.isfinite(arr)]

    pear = (
        _finite(core_df["pearson_correlation"])
        if "pearson_correlation" in core_df.columns
        else np.array([])
    )
    spear = (
        _finite(core_df["spearman_correlation"])
        if "spearman_correlation" in core_df.columns
        else np.array([])
    )

    fig = plt.figure(figsize=(8, 4), dpi=DPI)
    ax1 = fig.add_subplot(121)
    if pear.size > 0:
        ax1.hist(pear, bins=30)
    ax1.set_xlabel("r (Pearson)", fontsize=FONT_SIZE)
    ax1.set_ylabel("Count", fontsize=FONT_SIZE)
    ax1.tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    ax2 = fig.add_subplot(122)
    if spear.size > 0:
        ax2.hist(spear, bins=30)
    ax2.set_xlabel("r (Spearman)", fontsize=FONT_SIZE)
    ax2.set_ylabel("Count", fontsize=FONT_SIZE)
    ax2.tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    plt.tight_layout()
    png = out_dir / "correlation_histograms.png"
    pdf = out_dir / "correlation_histograms.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Saved histograms: {png}\nSaved histograms: {pdf}")


def boxplot_significance_counts(core_df: pd.DataFrame, out_dir: Path):
    """
    Box plots for counts of significant genes, for:
    adjusted p-values and raw p-values

    """
    from matplotlib.ticker import MaxNLocator

    # ---- Adjusted p-values ----
    have_adj = all(
        c in core_df.columns
        for c in ["pearson_adjusted_p_value", "spearman_adjusted_p_value"]
    )
    if have_adj:
        pear_adj_sig = int((core_df["pearson_adjusted_p_value"] < ADJ_P_THRESH).sum())
        spear_adj_sig = int((core_df["spearman_adjusted_p_value"] < ADJ_P_THRESH).sum())
        counts = [pear_adj_sig, spear_adj_sig]
        labels = ["Pearson (adj)", "Spearman (adj)"]

        fig, ax = plt.subplots(figsize=(6.5, 6), dpi=DPI)
        ax.boxplot([[c] for c in counts], labels=labels, widths=0.5)
        for i, val in enumerate(counts, start=1):
            ax.text(
                i,
                val,
                f"{val}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE,
                fontweight="bold",
            )

        ax.set_ylabel("Number of Significant Genes", fontsize=FONT_SIZE)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        plt.tight_layout()
    else:
        print(
            "Adjusted p-value columns not found; skipping adjusted significance box plot."
        )

    # ---- Raw p-values----
    have_raw = all(
        c in core_df.columns for c in ["pearson_p_value", "spearman_p_value"]
    )
    if have_raw:
        pear_raw_sig = int((core_df["pearson_p_value"] < RAW_P_THRESH).sum())
        spear_raw_sig = int((core_df["spearman_p_value"] < RAW_P_THRESH).sum())
        counts = [pear_raw_sig, spear_raw_sig]
        labels = ["Pearson (raw)", "Spearman (raw)"]

        fig, ax = plt.subplots(figsize=(6.5, 6), dpi=DPI)
        ax.boxplot([[c] for c in counts], labels=labels, widths=0.5)
        for i, val in enumerate(counts, start=1):
            ax.text(
                i,
                val,
                f"{val}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE,
                fontweight="bold",
            )

        ax.set_ylabel("Number of Significant Genes", fontsize=FONT_SIZE)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        plt.tight_layout()
        png = out_dir / "significant_counts_raw_boxplot.png"
        pdf = out_dir / "significant_counts_raw_boxplot.pdf"
        fig.savefig(png, dpi=DPI)
        fig.savefig(pdf)
        plt.close(fig)
        print(f"Saved raw significance box plot: {png}\nSaved: {pdf}")
    else:
        print("Raw p-value columns not present; skipping raw significance box plot.")


def boxplots_cor_by_significance(core_df: pd.DataFrame, out_dir: Path):
    """
    Box plots of correlation values grouped by *adjusted* significance (p < ADJ_P_THRESH),
    for Pearson and Spearman separately

    """
    # ---------------- Pearson ----------------
    if all(
        c in core_df.columns
        for c in ["pearson_correlation", "pearson_adjusted_p_value"]
    ):
        dfp = (
            core_df[["pearson_correlation", "pearson_adjusted_p_value"]].dropna().copy()
        )
        dfp["Significant (adj)"] = dfp["pearson_adjusted_p_value"] < ADJ_P_THRESH

        groups = [
            dfp.loc[~dfp["Significant (adj)"], "pearson_correlation"],
            dfp.loc[dfp["Significant (adj)"], "pearson_correlation"],
        ]
        counts = [len(groups[0]), len(groups[1])]

        fig, ax = plt.subplots(figsize=(4, 4), dpi=DPI)
        ax.boxplot(groups, labels=["Not Sig", "Sig"])
        ax.set_ylabel("Pearson Correlation", fontsize=FONT_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)

        ymin, ymax = ax.get_ylim()
        y_range = max(1e-9, ymax - ymin)
        offset = 0.05 * y_range

        for i, (vals, n) in enumerate(zip(groups, counts), start=1):
            if len(vals) == 0:
                continue
            y_max = float(np.max(vals))
            ax.text(
                i,
                y_max + offset,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE,
                fontweight="bold",
            )

        ax.set_ylim(ymin, ymax + 2 * offset)

        plt.tight_layout()
        png = out_dir / "pearson_boxplot_by_adj_significance_counts.png"
        pdf = out_dir / "pearson_boxplot_by_adj_significance_counts.pdf"
        fig.savefig(png, dpi=DPI)
        fig.savefig(pdf)
        plt.close(fig)
        print(f"Pearson box plot with counts: {png} | {pdf}")
    else:
        print("Missing Pearson columns; skipping Pearson box plot.")

    # ---------------- Spearman ----------------
    if all(
        c in core_df.columns
        for c in ["spearman_correlation", "spearman_adjusted_p_value"]
    ):
        dfs = (
            core_df[["spearman_correlation", "spearman_adjusted_p_value"]]
            .dropna()
            .copy()
        )
        dfs["Significant (adj)"] = dfs["spearman_adjusted_p_value"] < ADJ_P_THRESH

        groups = [
            dfs.loc[~dfs["Significant (adj)"], "spearman_correlation"],
            dfs.loc[dfs["Significant (adj)"], "spearman_correlation"],
        ]
        counts = [len(groups[0]), len(groups[1])]

        fig, ax = plt.subplots(figsize=(4, 4), dpi=DPI)
        ax.boxplot(groups, labels=["Not Sig", "Sig"])
        ax.set_ylabel("Spearman Correlation", fontsize=FONT_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)

        ymin, ymax = ax.get_ylim()
        y_range = max(1e-9, ymax - ymin)
        offset = 0.05 * y_range

        for i, (vals, n) in enumerate(zip(groups, counts), start=1):
            if len(vals) == 0:
                continue
            y_max = float(np.max(vals))
            ax.text(
                i,
                y_max + offset,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE,
                fontweight="bold",
            )

        ax.set_ylim(ymin, ymax + 2 * offset)

        plt.tight_layout()
        png = out_dir / "spearman_boxplot_by_adj_significance_counts.png"
        pdf = out_dir / "spearman_boxplot_by_adj_significance_counts.pdf"
        fig.savefig(png, dpi=DPI)
        fig.savefig(pdf)
        plt.close(fig)
        print(f"Spearman box plot with counts: {png} | {pdf}")
    else:
        print("Missing Spearman columns; skipping Spearman box plot.")


# -------------------- Main --------------------
def main():
    # NEW: All-points
    all_points_df = read_all_points(DIR_AVP)
    all_points_deviation_plot(all_points_df, OUT_DIR)

    # Part A: Actual vs predicted (per-gene means)
    means_df = read_gene_means(DIR_AVP)
    scatter_means(means_df, OUT_DIR)

    # Part B: Core results-driven plots
    if CORE_RESULTS.exists():
        core_df = read_core_results(CORE_RESULTS)
        correlation_histograms(core_df, OUT_DIR)
        boxplot_significance_counts(core_df, OUT_DIR)
        boxplots_cor_by_significance(core_df, OUT_DIR)
    else:
        print(f"{CORE_RESULTS} not found; skipping correlation/significance plots.")


if __name__ == "__main__":
    main()
