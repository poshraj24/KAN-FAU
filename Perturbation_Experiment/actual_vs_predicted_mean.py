#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

GENE_COL_CANDIDATES = ("gene", "gene_name", "symbol")


def _format_fit_label(slope: float, intercept: float) -> str:
    return (
        f"Fit: (y={slope:.3f}x+{intercept:.3f})"
        if intercept >= 0
        else f"Fit: (y={slope:.3f}x{intercept:.3f})"
    )


def _p_string_from_r(r: float, n: int) -> str:
    """
    Two-sided p-value for Pearson r with n samples

    """
    df = max(1, n - 2)
    r = float(np.clip(r, -0.9999999999999999, 0.9999999999999999))
    denom = max(1e-300, 1.0 - r * r)
    t = r * np.sqrt(df / denom)
    logp = np.log(2.0) + stats.t.logsf(abs(t), df)  #
    log10p = logp / np.log(10.0)
    if log10p >= -308:
        p = 10.0**log10p
        return f"P = {p:.2e}"
    else:
        return f"P ≈ 10^{log10p:.1f}"


def _extract_means(df: pd.DataFrame):
    """
    Returns (x_means, y_means, n_genes, source_mode)
      source_mode in {"explicit_means", "grouped_by_gene", "assumed_means"}
    """
    if {"mean_actual", "mean_predicted"}.issubset(df.columns):
        x = pd.to_numeric(df["mean_actual"], errors="coerce")
        y = pd.to_numeric(df["mean_predicted"], errors="coerce")
        mask = ~(x.isna() | y.isna())
        x, y = x[mask].to_numpy(), y[mask].to_numpy()
        return x, y, x.size, "explicit_means"

    if {"actual_value", "predicted_value"}.issubset(df.columns):
        gene_col = next((c for c in GENE_COL_CANDIDATES if c in df.columns), None)
        if gene_col is not None:
            dfc = df.copy()
            dfc["actual_value"] = pd.to_numeric(dfc["actual_value"], errors="coerce")
            dfc["predicted_value"] = pd.to_numeric(
                dfc["predicted_value"], errors="coerce"
            )
            dfc = dfc.dropna(subset=["actual_value", "predicted_value", gene_col])
            g = dfc.groupby(gene_col, as_index=False)[
                ["actual_value", "predicted_value"]
            ].mean()
            x = g["actual_value"].to_numpy()
            y = g["predicted_value"].to_numpy()
            return x, y, x.size, "grouped_by_gene"

        x = pd.to_numeric(df["actual_value"], errors="coerce")
        y = pd.to_numeric(df["predicted_value"], errors="coerce")
        mask = ~(x.isna() | y.isna())
        x, y = x[mask].to_numpy(), y[mask].to_numpy()
        return x, y, x.size, "assumed_means"

    raise ValueError(
        "CSV must contain either "
        "['mean_actual','mean_predicted'] or "
        "['actual_value','predicted_value']"
    )


def plot_means_per_gene(csv_path, out_png, out_pdf=None, title=None):
    df = pd.read_csv(csv_path)
    x, y, n_genes, mode = _extract_means(df)
    if n_genes < 3:
        raise ValueError("Not enough genes after cleaning.")

    # Regression (for line) and Pearson r
    slope, intercept, _, _, _ = stats.linregress(x, y)
    r, _ = stats.pearsonr(x, y)
    p_text = _p_string_from_r(r, n_genes)

    # Axis range
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad
    xx = np.linspace(lo, hi, 200)

    # Figure
    plt.figure(figsize=(4, 4), dpi=150)
    plt.scatter(
        x, y, color="steelblue", s=20, alpha=0.7, edgecolors="white", linewidth=0.5
    )
    plt.plot(xx, xx, "k--", linewidth=2, alpha=0.85)
    plt.plot(
        xx,
        intercept + slope * xx,
        color="orange",
        linewidth=2,
        alpha=0.9,
        label=_format_fit_label(slope, intercept),
    )
    plt.legend(fontsize=12, loc="lower right", bbox_to_anchor=(1, -0.03), frameon=False)

    stats_text = f"r = {r:.3f}\n{p_text}\n(n={n_genes})"
    plt.text(
        0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12, va="top"
    )

    subtitle = {
        "explicit_means": "Input = explicit per-gene means",
        "grouped_by_gene": "Input = all-points grouped by gene",
        "assumed_means": "Input = one row per gene",
    }[mode]
    if title:
        plt.title(title, fontsize=12, pad=12)
    plt.suptitle(subtitle, y=1.02, fontsize=9)

    plt.xlabel("Actual Mean Gene Expression Value", fontsize=12)
    plt.ylabel("Predicted Mean Gene Expression Value", fontsize=12)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=1200, bbox_inches="tight")
    if out_pdf:
        plt.savefig(out_pdf, dpi=1200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_means_per_gene(
        csv_path="BHLHE40/BHLHE40_regulated_test_model_performance.csv",
        out_png="BHLHE40/BHLHE40_2.png",
        out_pdf="BHLHE40/BHLHE40_2.pdf",
        title="Per-Gene Means",
    )
