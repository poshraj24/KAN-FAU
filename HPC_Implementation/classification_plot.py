import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator
import numpy as np

# ---------- Data ----------
thresholds = ["Unfiltered", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]

GRN = {
    "TP": [1431, 293, 152, 97, 58, 42, 13],
    "FP": [5378, 1157, 636, 362, 222, 141, 36],
    "FN": [0, 1139, 1280, 1335, 1374, 1390, 1419],
}

KAN = {
    "TP": [1431, 274, 132, 78, 38, 24, 9],
    "FP": [5378, 1176, 656, 381, 242, 159, 40],
    "FN": [0, 1158, 1300, 1354, 1394, 1408, 1423],
}

metrics = ["TP", "FP", "FN"]

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


# ---------- Piecewise scaling ----------
def forward_compress_above(y, cutoff=500, factor=2.0):
    """Expand below cutoff, compress above."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    mask = y <= cutoff
    out[mask] = y[mask]
    out[~mask] = cutoff + (y[~mask] - cutoff) / factor
    return out


def inverse_compress_above(yt, cutoff=500, factor=9.0):
    out = np.empty_like(yt)
    mask = yt <= cutoff
    out[mask] = yt[mask]
    out[~mask] = cutoff + (yt[~mask] - cutoff) * factor
    return out


def forward_compress_below(y, cutoff=1000, factor=2.0):
    """Compress below cutoff, expand above."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    mask = y <= cutoff
    out[mask] = y[mask] / factor
    out[~mask] = (y[~mask] - cutoff) + cutoff / factor
    return out


def inverse_compress_below(yt, cutoff=1000, factor=9.0):
    """Inverse for compress_below."""
    out = np.empty_like(yt)
    split = cutoff / factor
    mask = yt <= split
    out[mask] = yt[mask] * factor
    out[~mask] = (yt[~mask] - split) + cutoff
    return out


def set_function_scale(ax, forward, inverse):
    ax.set_yscale("function", functions=(forward, inverse))


# ---------- Plotter ----------
def make_scaled_plot(metric, fname, factor=5.0):
    fig, ax = plt.subplots(figsize=(4, 4))

    if metric == "TP":
        cutoff = 500
        set_function_scale(
            ax,
            lambda y: forward_compress_above(y, cutoff, factor),
            lambda yt: inverse_compress_above(yt, cutoff, factor),
        )

    elif metric == "FP":
        cutoff = 1000
        set_function_scale(
            ax,
            lambda y: forward_compress_above(y, cutoff, factor),
            lambda yt: inverse_compress_above(yt, cutoff, factor),
        )

    elif metric == "FN":
        cutoff = 1100
        set_function_scale(
            ax,
            lambda y: forward_compress_below(y, cutoff, factor),
            lambda yt: inverse_compress_below(yt, cutoff, factor),
        )

        ax.set_ylim(0, 1432)
        ticks = [0, 250, 500, 750, 1000, 1200, 1432]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="plain", axis="y")

        fig.subplots_adjust(top=0.92)

    # Plot lines
    ax.plot(
        thresholds,
        GRN[metric],
        marker="o",
        markersize=5,
        linewidth=2,
        linestyle="-",
        color="tab:red",
        label="GRNBOOST2",
    )
    ax.plot(
        thresholds,
        KAN[metric],
        marker="s",
        markersize=5,
        linewidth=2,
        linestyle=":",
        color="tab:blue",
        label="KAN",
    )

    ax.set_title(metric)
    ax.set_xlabel("z-score cutoff")
    ax.set_ylabel("Number of edges")
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True)

    # guide line at cutoff
    # ax.axhline(cutoff, linestyle=":", linewidth=1, color="gray")

    plt.tight_layout()
    plt.savefig(fname, dpi=1200)
    plt.show()


for m in metrics:
    make_scaled_plot(m, f"grn_{m.lower()}_plot.png")
