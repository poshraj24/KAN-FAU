#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List, Tuple, Dict
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------- Configuration ----------
ROOT_DIR = Path(
    "KAN_Implementation/kan_models"
)  # folder containing per-gene subfolders
OUT_DIR = Path("KAN_Implementation/plots")
DPI = 1200  # figure export DPI
FONT_SIZE = 12  # global font size
POINT_SIZE = 20  # scatter point size
ALPHA = 0.6  # point transparency
# ----------------------------------

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
    }
)

REQUIRED_COLS = [
    "sample_index",
    "actual_value",
    "predicted_value",
    "absolute_error",
    "squared_error",
]


def format_pvalue(p: float) -> str:
    """
    Format p-value for display. If p is too small to represent properly,
    show as p < minimum representable value.
    """
    # Get the minimum positive float that can be represented
    min_float = sys.float_info.min

    # If p-value is effectively zero or smaller than minimum representable
    if p == 0.0 or p < min_float:
        return f"p < {min_float:.0e}"
    # If p-value is very small but representable
    elif p < 1e-10:
        return f"p = {p:.0e}"
    # If p-value is small but not extremely small
    elif p < 0.001:
        return f"p = {p:.2e}"
    # For larger p-values, use regular formatting
    else:
        return f"p = {p:.3f}"


def load_gene_tables(gene_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate test_comparison.csv and validation_comparison.csv for a gene.
    Returns an empty DataFrame if neither exists or columns are missing.
    """
    dfs = []
    for fname in ["test_comparison.csv", "validation_comparison.csv"]:
        fpath = gene_dir / fname
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                missing = [c for c in REQUIRED_COLS if c not in df.columns]
                if missing:
                    print(f"{fpath} missing columns {missing}; skipping this file.")
                    continue
                dfs.append(df[["actual_value", "predicted_value"]].copy())
            except Exception as e:
                print(f"Failed to read {fpath}: {e}")
    if not dfs:
        return pd.DataFrame(columns=["actual_value", "predicted_value"])
    return pd.concat(dfs, axis=0, ignore_index=True)


def walk_genes(root: Path) -> Dict[str, pd.DataFrame]:
    """
    Walk ROOT_DIR and build {gene_name: concatenated_df}.
    A 'gene folder' is any directory directly under ROOT_DIR that contains at least one required csv.
    """
    genes: Dict[str, pd.DataFrame] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir():
            df = load_gene_tables(child)
            if not df.empty:
                genes[child.name] = df
            else:

                pass
    return genes


def compute_per_gene_means(genes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each gene, compute mean(actual) and mean(predicted).
    """
    rows = []
    for g, df in genes.items():
        mean_act = df["actual_value"].mean()
        mean_pred = df["predicted_value"].mean()
        rows.append(
            {
                "gene": g,
                "mean_actual": mean_act,
                "mean_predicted": mean_pred,
                "n_samples": len(df),
            }
        )
    return pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)


def _axes_limits(
    x: np.ndarray, y: np.ndarray, pad: float = 0.05
) -> Tuple[float, float]:
    """
    Symmetric limits around data range expanded by 'pad' fraction.
    """
    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])
    span = hi - lo if hi > lo else (abs(hi) + 1.0)
    lo_p = lo - pad * span
    hi_p = hi + pad * span
    return lo_p, hi_p


def scatter_with_fit(
    x: np.ndarray,
    y: np.ndarray,
    # title: str,
    xlabel: str,
    ylabel: str,
    out_prefix: Path,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    xstep: float = None,
    ystep: float = None,
):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # if len(x) < 2:
    #     print(f"[WARN] Not enough points to plot '{title}'. Skipping.")
    #     return

    r, p = stats.pearsonr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(x, y, s=POINT_SIZE, alpha=ALPHA, edgecolors="none")

    grid = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 200)
    ax.plot(grid, grid, "--", color="black", linewidth=2)
    ax.plot(
        grid,
        slope * grid + intercept,
        color="orange",
        linewidth=2,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Add fit equation to title with orange dash
    sign = "+" if intercept >= 0 else "-"
    intercept_abs = abs(intercept)
    # Add fit equation to title with orange dash
    ax.text(
        0.02,
        1.02,
        "—",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="orange",
        fontsize=FONT_SIZE,
        fontweight="bold",
    )
    ax.text(
        0.08,
        1.02,
        f" Fit: y = {slope:.3f}x {sign} {intercept_abs:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="black",
        fontsize=FONT_SIZE,
    )

    if xlim:
        ax.set_xlim(xlim)
        if xstep:
            ax.set_xticks(np.arange(xlim[0], xlim[1] + xstep, xstep))
    if ylim:
        ax.set_ylim(ylim)
        if ystep:
            ax.set_yticks(np.arange(ylim[0], ylim[1] + ystep, ystep))

    # Use the improved p-value formatting function
    pvalue_text = format_pvalue(p)
    ax.text(
        0.02,
        0.98,
        f"r = {r:.3f}\n{pvalue_text}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_prefix}.png", dpi=DPI, bbox_inches="tight")
    # fig.savefig(OUT_DIR / f"{out_prefix}.pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {OUT_DIR / f'{out_prefix}.png'}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Read all gene data
    genes = walk_genes(ROOT_DIR)
    if not genes:
        print(f"No usable gene data found under '{ROOT_DIR}'.")
        return
    print(f"Loaded {len(genes)} genes with valid data.")

    # 2) Per-gene means scatter (one point per gene)
    means_df = compute_per_gene_means(genes)

    means_df.to_csv(OUT_DIR / "per_gene_means.tsv", sep="\t", index=False)

    scatter_with_fit(
        x=means_df["mean_actual"].to_numpy(),
        y=means_df["mean_predicted"].to_numpy(),
        # title="Per-Gene Mean Expression: Actual vs Predicted",
        xlabel="Actual Mean Gene Expression ",
        ylabel="Predicted Mean Gene Expression",
        out_prefix=Path("per_gene_mean_actual_vs_predicted"),
        xlim=(0, 4),
        ylim=(-2, 4),
        xstep=1,
        ystep=1,
    )

    # 3) Pooled all-samples scatter (all genes, all samples)
    pooled_actual = np.concatenate(
        [df["actual_value"].to_numpy() for df in genes.values()]
    )
    pooled_pred = np.concatenate(
        [df["predicted_value"].to_numpy() for df in genes.values()]
    )
    scatter_with_fit(
        x=pooled_actual,
        y=pooled_pred,
        # title="All Samples Pooled: Actual vs Predicted",
        xlabel="Actual Gene Expression",
        ylabel="Predicted Gene Expression",
        out_prefix=Path("all_samples_actual_vs_predicted"),
        xlim=(0, 10),
        ylim=(-2, 16),
        xstep=2,
        ystep=2,
    )


if __name__ == "__main__":
    main()
