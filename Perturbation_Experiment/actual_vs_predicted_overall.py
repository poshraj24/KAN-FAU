#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


def _format_fit_label(slope: float, intercept: float) -> str:
    return (
        f"Fit: (y={slope:.3f}x+{intercept:.3f})"
        if intercept >= 0
        else f"Fit: (y={slope:.3f}x{intercept:.3f})"
    )


def _p_string_from_r(r: float, n: int) -> str:
    df = max(1, n - 2)
    r = float(np.clip(r, -0.9999999999999999, 0.9999999999999999))
    denom = max(1e-300, 1.0 - r * r)
    t = r * np.sqrt(df / denom)
    logp = np.log(2.0) + stats.t.logsf(abs(t), df)
    log10p = logp / np.log(10.0)
    if log10p >= -308:
        p = 10.0**log10p
        return f"P = {p:.2e}"
    else:
        return f"P ≈ 10^{log10p:.1f}"


def plot_overall_points(csv_path, out_png, out_pdf=None, title=None):
    df = pd.read_csv(csv_path)
    if not {"actual_value", "predicted_value"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain 'actual_value' and 'predicted_value' columns."
        )

    x = pd.to_numeric(df["actual_value"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["predicted_value"], errors="coerce").to_numpy()
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = x.size
    if n < 3:
        raise ValueError("Not enough valid points after cleaning.")

    slope, intercept, _, _, _ = stats.linregress(x, y)
    r, _ = stats.pearsonr(x, y)
    p_text = _p_string_from_r(r, n)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad
    xx = np.linspace(lo, hi, 200)

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

    stats_text = f"r = {r:.3f}\n{p_text}\n(n={n})"
    plt.text(
        0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12, va="top"
    )

    if title:
        plt.title(title, fontsize=12, pad=12)
    plt.xlabel("Actual Gene Expression Value", fontsize=12)
    plt.ylabel("Predicted Gene Expression Value", fontsize=12)
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
    plot_overall_points(
        csv_path="BHLHE40/BHLHE40_validation_all_points.csv",
        out_png="BHLHE40/BHLHE40_1.png",
        out_pdf="BHLHE40/BHLHE40_1.pdf",
        title="All Points",
    )
