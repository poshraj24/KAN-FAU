#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys


def _p_string_from_r(r: float, n: int) -> str:
    """
    Calculate p-value from correlation coefficient and format it properly.
    """
    print(f"  FALLBACK P-VALUE CALCULATION:")
    print(f"  r = {r:.6f}, n = {n}")

    if n < 3:
        print(f"  Insufficient sample size (n < 3)")
        return "p = N/A"

    # Handle perfect or near-perfect correlations
    if abs(r) >= 0.99999:
        print(f"  Near-perfect correlation detected (|r| >= 0.99999)")
        return "p < 1e-15"

    try:
        # Use scipy's built-in p-value calculation when possible
        df = max(1, n - 2)
        denom = max(1e-15, 1.0 - r * r)
        t = abs(r) * np.sqrt(df / denom)

        print(f"  df = {df}, denom = {denom:.10f}, t = {t:.6f}")

        # Calculate p-value more precisely for very large t-values
        if t > 100:
            # For extremely large t-statistics, estimate using log scale
            log_p = stats.t.logsf(t, df) + np.log(2)  # two-tailed
            if log_p < -50:  # log(1e-22) ≈ -50
                result = "p < 1e-22"
            else:
                p_value = np.exp(log_p)
                result = f"p = {p_value:.0e}"
            print(f"  Extremely large t-statistic, using log calculation")
        elif t > 50:
            result = "p < 1e-15"
            print(f"  Very large t-statistic (t > 50)")
        else:
            p_value = 2 * stats.t.sf(t, df)
            print(f"  Calculated p-value = {p_value}")

            # Format p-value appropriately
            if p_value == 0.0 or p_value < 1e-15:
                result = "p < 1e-15"
            elif p_value < 1e-10:
                result = f"p = {p_value:.0e}"
            elif p_value < 0.001:
                result = f"p = {p_value:.2e}"
            else:
                result = f"p = {p_value:.3f}"

        print(f"  Formatted result: {result}")
        return result

    except (OverflowError, ZeroDivisionError, ValueError) as e:
        print(f"  Exception in calculation: {e}")
        return "p < 1e-15"


def create_correlation_plot(csv_file, output_file="correlation_plot.png", title=None):
    """
    Scatter of actual vs predicted with:
      - Perfect fit line (y=x)
      - Regression line
      - r and two-sided p-value (displayed robustly)
    Expected columns: 'actual_value', 'predicted_value'
    """
    df = pd.read_csv(csv_file)
    if not {"actual_value", "predicted_value"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain 'actual_value' and 'predicted_value' columns."
        )

    actual = pd.to_numeric(df["actual_value"], errors="coerce").to_numpy()
    predicted = pd.to_numeric(df["predicted_value"], errors="coerce").to_numpy()

    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    n = actual.size
    if n < 3:
        raise ValueError("Not enough valid points after cleaning.")

    # Pearson r and p
    r_to_show, p_scipy = stats.pearsonr(actual, predicted)

    # Debug output
    print(f"\n=== MEAN VALUES PLOT DEBUG ===")
    print(f"Dataset: Mean values per gene")
    print(f"Sample size (n): {n}")
    print(
        f"Actual values - min: {actual.min():.3f}, max: {actual.max():.3f}, mean: {actual.mean():.3f}, std: {actual.std():.3f}"
    )
    print(
        f"Predicted values - min: {predicted.min():.3f}, max: {predicted.max():.3f}, mean: {predicted.mean():.3f}, std: {predicted.std():.3f}"
    )
    print(f"Correlation coefficient (r): {r_to_show:.6f}")
    print(f"Scipy p-value: {p_scipy}")
    print(f"Is scipy p-value NaN?: {np.isnan(p_scipy)}")
    print(f"Is scipy p-value exactly 0?: {p_scipy == 0.0}")
    print(f"Is scipy p-value > 1?: {p_scipy > 1.0}")

    # Use scipy's p-value if it's reasonable, otherwise use our robust calculation
    if np.isnan(p_scipy) or p_scipy == 0.0 or p_scipy > 1.0:
        print(f"Using fallback p-value calculation")
        p_text = _p_string_from_r(r_to_show, n)
    else:
        print(f"Using scipy p-value")
        # Format scipy's p-value using the same logic
        if p_scipy < 1e-15:
            p_text = "p < 1e-15"
        elif p_scipy < 1e-10:
            p_text = f"p = {p_scipy:.0e}"
        elif p_scipy < 0.001:
            p_text = f"p = {p_scipy:.2e}"
        else:
            p_text = f"p = {p_scipy:.3f}"

    print(f"Final p-value text: {p_text}")
    print(f"==============================\n")

    # Regression line
    slope, intercept, _, _, _ = stats.linregress(actual, predicted)

    vmin = min(np.min(actual), np.min(predicted))
    vmax = max(np.max(actual), np.max(predicted))
    pad = (vmax - vmin) * 0.05 if vmax > vmin else 1.0
    lo, hi = vmin - pad, vmax + pad
    x_line = np.linspace(lo, hi, 200)

    plt.figure(figsize=(4, 4), dpi=150)
    plt.scatter(
        actual,
        predicted,
        s=10,
        alpha=0.7,
        color="steelblue",
        edgecolors="white",
        linewidth=0.5,
    )
    plt.plot(x_line, x_line, "k--", linewidth=2, alpha=0.85)

    label_str = (
        f"Fit: (y={slope:.3f}x+{intercept:.3f})"
        if intercept >= 0
        else f"Fit: (y={slope:.3f}x{intercept:.3f})"
    )
    plt.plot(
        x_line,
        slope * x_line + intercept,
        color="orange",
        linewidth=2,
        alpha=0.9,
        label=label_str,
    )

    plt.xlabel("Actual Mean Gene Expression Value", fontsize=12)
    plt.ylabel("Predicted Mean Gene Expression Value", fontsize=12)
    if title is None:
        title = os.path.splitext(os.path.basename(csv_file))[0]
    plt.title(title, fontsize=12, pad=12)

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    plt.tick_params(axis="both", labelsize=12)

    txt = f"r = {r_to_show:.3f}\n{p_text}"
    plt.text(
        0.05, 0.95, txt, transform=plt.gca().transAxes, ha="left", va="top", fontsize=12
    )

    plt.legend(fontsize=12, loc="lower right", bbox_to_anchor=(1, -0.03), frameon=False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=1200, bbox_inches="tight")
    plt.savefig(output_file.replace(".png", ".pdf"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    create_correlation_plot(
        csv_file="ZNF326/perturbation_model_evaluation_test_ZNF326_filtered.csv",
        output_file="ZNF326/ZNF326_4.png",
        title="Predicted vs Actual",
    )
