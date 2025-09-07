import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class ResearchPaperPlots:
    def __init__(self, save_path="research_plots"):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.dpi": 1200,
                "font.size": 12,
                "font.family": "serif",
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "figure.figsize": (10, 8),
            }
        )

    def franke_2d(self, x, y):
        term1 = 0.75 * np.exp(-((9 * x - 2) ** 2 + (9 * y - 2) ** 2) / 4)
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
        term3 = 0.5 * np.exp(-((9 * x - 7) ** 2 + (9 * y - 3) ** 2) / 4)
        term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
        return term1 + term2 + term3 + term4

    def hartmann_3d(self, x, y, z):
        A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        P = (
            np.array(
                [
                    [3689, 1170, 2673],
                    [4699, 4387, 7470],
                    [1091, 8732, 5547],
                    [381, 5743, 8828],
                ]
            )
            * 1e-4
        )
        c = np.array([1.0, 1.2, 3.0, 3.2])

        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        result = np.zeros_like(x, dtype=float)

        for i in range(4):
            term1 = A[i, 0] * (x - P[i, 0]) ** 2
            term2 = A[i, 1] * (y - P[i, 1]) ** 2
            term3 = A[i, 2] * (z - P[i, 2]) ** 2
            inner_sum = term1 + term2 + term3
            result -= c[i] * np.exp(-inner_sum)
        return result

    def ackley_function(self, X):
        a, b, c = 20, 0.2, 2 * np.pi
        d = X.shape[0]
        sum1 = np.sum(X**2, axis=0)
        sum2 = np.sum(np.cos(c * X), axis=0)
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        return term1 + term2 + a + np.exp(1)

    def michalewicz_function(self, X):
        m = 10
        d = X.shape[0]
        result = np.zeros(X.shape[1:])
        for i in range(d):
            result -= np.sin(X[i]) * (np.sin((i + 1) * X[i] ** 2 / np.pi)) ** (2 * m)
        return result

    def levy_function(self, X):
        d = X.shape[0]
        w = 1 + (X - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum(
            (w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2), axis=0
        )
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        return term1 + term2 + term3

    def plot_franke_2d(self):
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(111, projection="3d")

        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.franke_2d(X, Y)

        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap="viridis",
            alpha=0.9,
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        ax.set_xlabel("x₁", fontsize=14, labelpad=10)
        ax.set_ylabel("x₂", fontsize=14, labelpad=10)
        ax.set_zlabel("f(x₁, x₂)", fontsize=14, labelpad=3)
        ax.set_title("Franke 2D Function", fontsize=16, pad=20)
        ax.view_init(elev=30, azim=45)

        plt.tight_layout()
        plt.savefig(
            f"{self.save_path}/franke_2d.png",
            bbox_inches="tight",
            dpi=1200,
            pad_inches=0.5,
        )
        plt.savefig(
            f"{self.save_path}/franke_2d.pdf",
            bbox_inches="tight",
            dpi=1200,
            pad_inches=0.5,
        )
        plt.close()

    def plot_hartmann_3d(self):
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(111, projection="3d")

        x = np.linspace(0, 1, 80)
        y = np.linspace(0, 1, 80)
        X, Y = np.meshgrid(x, y)
        Z_slice = 0.5 * np.ones_like(X)
        Z = self.hartmann_3d(X, Y, Z_slice)

        surf = ax.plot_surface(
            X, Y, Z, cmap="plasma", alpha=0.9, linewidth=0, antialiased=True, shade=True
        )

        ax.set_xlabel("x₁", fontsize=14, labelpad=10)
        ax.set_ylabel("x₂", fontsize=14, labelpad=10)
        ax.set_zlabel("f(x₁, x₂, 0.5)", fontsize=14, labelpad=3)
        ax.set_title("Hartmann 3D Function (x₃ = 0.5)", fontsize=16, pad=20)
        ax.view_init(elev=30, azim=45)

        plt.tight_layout()
        plt.savefig(
            f"{self.save_path}/hartmann_3d.png",
            bbox_inches="tight",
            dpi=1200,
            pad_inches=0.5,
        )
        plt.savefig(
            f"{self.save_path}/hartmann_3d.pdf",
            bbox_inches="tight",
            dpi=1200,
            pad_inches=0.5,
        )
        plt.close()

    def plot_ackley_5d(self):
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(111, projection="3d")

        x1 = np.linspace(-5, 5, 100)
        x2 = np.linspace(-5, 5, 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.array([X1, X2, np.zeros_like(X1), np.zeros_like(X1), np.zeros_like(X1)])
        Z = self.ackley_function(X)

        surf = ax.plot_surface(
            X1,
            X2,
            Z,
            cmap="viridis",
            alpha=0.9,
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        ax.set_xlabel("x₁", fontsize=14, labelpad=10)
        ax.set_ylabel("x₂", fontsize=14, labelpad=10)
        ax.set_zlabel("f(x₁, x₂, 0, 0, 0)", fontsize=14, labelpad=3)
        ax.set_title("Ackley 5D Function (x₃=x₄=x₅=0)", fontsize=16, pad=20)
        ax.view_init(elev=30, azim=45)

        plt.tight_layout()
        plt.savefig(
            f"{self.save_path}/ackley_5d.png",
            bbox_inches="tight",
            dpi=1200,
            pad_inches=0.5,
        )
        plt.savefig(
            f"{self.save_path}/ackley_5d.pdf",
            bbox_inches="tight",
            dpi=1200,
            pad_inches=0.5,
        )
        plt.close()

    def plot_michalewicz_7d(self):
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(111, projection="3d")

        x1 = np.linspace(0, np.pi, 100)
        x2 = np.linspace(0, np.pi, 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.array([X1, X2] + [np.pi / 2 * np.ones_like(X1) for _ in range(5)])
        Z = self.michalewicz_function(X)

        surf = ax.plot_surface(
            X1,
            X2,
            Z,
            cmap="plasma",
            alpha=0.9,
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        ax.set_xlabel("x₁", fontsize=14, labelpad=10)
        ax.set_ylabel("x₂", fontsize=14, labelpad=10)
        ax.set_zlabel("f(x₁, x₂, π/2, ...)", fontsize=14, labelpad=3)
        ax.set_title("Michalewicz 7D Function (x₃-x₇ = π/2)", fontsize=16, pad=20)
        ax.view_init(elev=30, azim=45)

        plt.tight_layout()
        plt.savefig(
            f"{self.save_path}/michalewicz_7d.png",
            bbox_inches="tight",
            dpi=1200,
            pad_inches=0.5,
        )
        plt.savefig(
            f"{self.save_path}/michalewicz_7d.pdf",
            bbox_inches="tight",
            dpi=1200,
            pad_inches=0.5,
        )
        plt.close()

    def plot_levy_10d(self):
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(111, projection="3d")

        x1 = np.linspace(-5, 5, 100)
        x2 = np.linspace(-5, 5, 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.array([X1, X2] + [np.ones_like(X1) for _ in range(8)])
        Z = self.levy_function(X)

        surf = ax.plot_surface(
            X1,
            X2,
            Z,
            cmap="viridis",
            alpha=0.9,
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        ax.set_xlabel("x₁", fontsize=14, labelpad=10)
        ax.set_ylabel("x₂", fontsize=14, labelpad=10)
        ax.set_zlabel("f(x₁, x₂, 1, ...)", fontsize=14, labelpad=3)
        ax.set_title("Levy 10D Function (x₃-x₁₀ = 1)", fontsize=16, pad=20)
        ax.view_init(elev=30, azim=45)

        plt.tight_layout()
        plt.savefig(
            f"{self.save_path}/levy_10d.png",
            bbox_inches="tight",
            dpi=1200,
            pad_inches=0.5,
        )
        plt.savefig(
            f"{self.save_path}/levy_10d.pdf",
            bbox_inches="tight",
            dpi=1200,
            pad_inches=0.5,
        )
        plt.close()

    def generate_all_plots(self):
        """Generate all benchmark function plots"""
        self.plot_franke_2d()
        self.plot_hartmann_3d()
        self.plot_ackley_5d()
        self.plot_michalewicz_7d()
        self.plot_levy_10d()

        print(f"All plots saved in '{self.save_path}/' folder")


# Generate research paper quality plots
plotter = ResearchPaperPlots()
plotter.generate_all_plots()
