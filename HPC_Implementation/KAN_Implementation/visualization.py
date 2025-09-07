import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class GeneTrainingAnalyzer:
    def __init__(self, base_path="KAN_Implementation/kan_models"):
        self.base_path = Path(base_path)
        self.gene_data = []
        self.epoch_data = []

    def extract_single_log(self, log_path):
        """Extract metrics from a single training_log.txt file"""
        try:
            with open(log_path, "r") as file:
                content = file.read()

            # Extract gene name from path
            gene_name = log_path.parent.name

            # Extract summary metrics
            metrics = {"gene_name": gene_name}

            # Configuration metrics
            metrics["total_parameters"] = self._extract_value(
                content, r"Total Parameters:\s*(\d+)"
            )
            metrics["batch_size"] = self._extract_value(content, r"Batch Size:\s*(\d+)")
            metrics["learning_rate"] = self._extract_value(
                content, r"Learning Rate:\s*([\d.e-]+)"
            )

            # Performance metrics
            metrics["best_val_loss"] = self._extract_value(
                content, r"Best Validation Loss:\s*([\d.-]+)"
            )
            metrics["best_val_r2"] = self._extract_value(
                content, r"Best Validation R2:\s*([\d.-]+)"
            )
            metrics["best_val_rmse"] = self._extract_value(
                content, r"Best Validation RMSE:\s*([\d.-]+)"
            )
            metrics["best_val_mae"] = self._extract_value(
                content, r"Best Validation MAE:\s*([\d.-]+)"
            )
            metrics["final_epochs"] = self._extract_value(
                content, r"Final Epochs Completed:\s*(\d+)"
            )

            # Data information
            metrics["total_samples"] = self._extract_value(
                content, r"Total Samples:\s*(\d+)"
            )
            metrics["training_samples"] = self._extract_value(
                content, r"Training Samples:\s*(\d+)"
            )

            # Resource usage
            metrics["training_duration"] = self._extract_value(
                content, r"Total Training Duration:\s*([\d.]+)"
            )
            metrics["peak_gpu_memory"] = self._extract_value(
                content, r"Peak GPU Memory:\s*([\d.]+)"
            )

            # Extract epoch-wise data
            epoch_pattern = r"Epoch (\d+): Train Loss = ([\d.]+), Val Loss = ([\d.]+), Test Loss = ([\d.]+), Val R2 = ([\d.-]+), Val RMSE = ([\d.]+), Val MAE = ([\d.]+)"
            epoch_matches = re.findall(epoch_pattern, content)

            for match in epoch_matches:
                epoch_data = {
                    "gene_name": gene_name,
                    "epoch": int(match[0]),
                    "train_loss": float(match[1]),
                    "val_loss": float(match[2]),
                    "test_loss": float(match[3]),
                    "val_r2": float(match[4]),
                    "val_rmse": float(match[5]),
                    "val_mae": float(match[6]),
                }
                self.epoch_data.append(epoch_data)

            return metrics

        except Exception as e:
            print(f"Error processing {log_path}: {e}")
            return None

    def _extract_value(self, content, pattern):
        """Extract numeric value using regex pattern"""
        match = re.search(pattern, content)
        if match:
            try:
                return float(match.group(1))
            except:
                return match.group(1)
        return None

    def process_all_logs(self):
        """Process all training_log.txt files in the directory structure"""

        # Find all training_log.txt files
        log_files = list(self.base_path.rglob("training_log.txt"))
        print(f"Found {len(log_files)} training log files")

        for log_file in log_files:
            metrics = self.extract_single_log(log_file)
            if metrics:
                self.gene_data.append(metrics)

        # Convert to DataFrames
        self.df_summary = pd.DataFrame(self.gene_data)
        self.df_epochs = pd.DataFrame(self.epoch_data)

        print(f"Successfully processed {len(self.gene_data)} genes")
        print(f"Extracted {len(self.epoch_data)} epoch records")

        return self.df_summary, self.df_epochs

    def create_consolidated_diagrams(self, save_path="gene_analysis_plots"):
        """Create comprehensive consolidated diagrams"""

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. MAIN PERFORMANCE OVERVIEW
        self._create_performance_overview(save_path)

        # 2. TRAINING CONVERGENCE ANALYSIS
        self._create_convergence_analysis(save_path)

        # 3. CORRELATION MATRIX
        self._create_correlation_matrix(save_path)

        # 4. RESOURCE EFFICIENCY ANALYSIS
        self._create_resource_analysis(save_path)

        # 5. COMPREHENSIVE SUMMARY TABLE
        self._create_summary_table(save_path)

        print(f"All diagrams saved to {save_path}/")

    def _create_performance_overview(self, save_path):
        """Create the main performance overview - BEST for thesis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Gene Training Performance Overview (N=1000 Genes)",
            fontsize=16,
            fontweight="bold",
        )

        # R² Distribution
        axes[0, 0].hist(
            self.df_summary["best_val_r2"].dropna(),
            bins=50,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        axes[0, 0].axvline(
            self.df_summary["best_val_r2"].median(),
            color="red",
            linestyle="--",
            label=f'Median: {self.df_summary["best_val_r2"].median():.3f}',
        )
        axes[0, 0].set_xlabel("Validation R²")
        axes[0, 0].set_ylabel("Number of Genes")
        axes[0, 0].set_title("A) Model Performance Distribution (R²)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # RMSE Distribution
        axes[0, 1].hist(
            self.df_summary["best_val_rmse"].dropna(),
            bins=50,
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
        )
        axes[0, 1].axvline(
            self.df_summary["best_val_rmse"].median(),
            color="red",
            linestyle="--",
            label=f'Median: {self.df_summary["best_val_rmse"].median():.3f}',
        )
        axes[0, 1].set_xlabel("Validation RMSE")
        axes[0, 1].set_ylabel("Number of Genes")
        axes[0, 1].set_title("B) Prediction Error Distribution (RMSE)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Training Epochs
        axes[1, 0].hist(
            self.df_summary["final_epochs"].dropna(),
            bins=30,
            alpha=0.7,
            color="orange",
            edgecolor="black",
        )
        axes[1, 0].axvline(
            self.df_summary["final_epochs"].median(),
            color="red",
            linestyle="--",
            label=f'Median: {self.df_summary["final_epochs"].median():.0f}',
        )
        axes[1, 0].set_xlabel("Training Epochs")
        axes[1, 0].set_ylabel("Number of Genes")
        axes[1, 0].set_title("C) Training Convergence (Epochs)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Performance vs Complexity
        valid_data = self.df_summary.dropna(subset=["total_parameters", "best_val_r2"])
        scatter = axes[1, 1].scatter(
            valid_data["total_parameters"],
            valid_data["best_val_r2"],
            alpha=0.6,
            c=valid_data["final_epochs"],
            cmap="viridis",
            s=20,
        )
        axes[1, 1].set_xlabel("Model Parameters")
        axes[1, 1].set_ylabel("Validation R²")
        axes[1, 1].set_title("D) Model Complexity vs Performance")
        axes[1, 1].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label("Training Epochs")

        plt.tight_layout()
        plt.savefig(
            f"{save_path}/01_performance_overview.png", dpi=600, bbox_inches="tight"
        )
        plt.savefig(f"{save_path}/01_performance_overview.pdf", bbox_inches="tight")
        plt.show()

    def _create_convergence_analysis(self, save_path):
        """Analyze training convergence patterns"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Training Convergence Analysis", fontsize=16, fontweight="bold")

        # Sample representative genes for convergence curves
        sample_genes = self.df_epochs["gene_name"].unique()[:20]  # First 20 genes
        sample_data = self.df_epochs[self.df_epochs["gene_name"].isin(sample_genes)]

        # Training Loss Curves
        for gene in sample_genes:
            gene_data = sample_data[sample_data["gene_name"] == gene]
            axes[0].plot(
                gene_data["epoch"], gene_data["train_loss"], alpha=0.3, linewidth=0.8
            )

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("A) Training Loss Convergence\n(Sample of 20 Genes)")
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)

        # Average convergence pattern
        avg_convergence = (
            self.df_epochs.groupby("epoch")
            .agg({"train_loss": "mean", "val_loss": "mean", "val_r2": "mean"})
            .reset_index()
        )

        axes[1].plot(
            avg_convergence["epoch"],
            avg_convergence["train_loss"],
            "b-",
            label="Training Loss",
            linewidth=2,
        )
        axes[1].plot(
            avg_convergence["epoch"],
            avg_convergence["val_loss"],
            "r-",
            label="Validation Loss",
            linewidth=2,
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("B) Average Loss Curves\n(All Genes)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # R² improvement over epochs
        axes[2].plot(
            avg_convergence["epoch"], avg_convergence["val_r2"], "g-", linewidth=2
        )
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Validation R²")
        axes[2].set_title("C) Average R² Improvement\n(All Genes)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{save_path}/02_convergence_analysis.png", dpi=600, bbox_inches="tight"
        )
        plt.savefig(f"{save_path}/02_convergence_analysis.pdf", bbox_inches="tight")
        plt.show()

    def _create_correlation_matrix(self, save_path):
        """Create correlation analysis"""
        metrics_cols = [
            "best_val_r2",
            "best_val_rmse",
            "best_val_mae",
            "final_epochs",
            "total_parameters",
            "training_duration",
            "total_samples",
        ]

        corr_data = self.df_summary[metrics_cols].dropna()
        correlation_matrix = corr_data.corr()

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        heatmap = sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title(
            "Correlation Matrix of Training Metrics", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            f"{save_path}/03_correlation_matrix.png", dpi=600, bbox_inches="tight"
        )
        plt.savefig(f"{save_path}/03_correlation_matrix.pdf", bbox_inches="tight")
        plt.show()

    def _create_resource_analysis(self, save_path):
        """Analyze computational resource usage"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Computational Resource Analysis", fontsize=16, fontweight="bold")

        # Training Duration Distribution
        axes[0].hist(
            self.df_summary["training_duration"].dropna(),
            bins=40,
            alpha=0.7,
            color="purple",
            edgecolor="black",
        )
        axes[0].axvline(
            self.df_summary["training_duration"].median(),
            color="red",
            linestyle="--",
            label=f'Median: {self.df_summary["training_duration"].median():.1f}s',
        )
        axes[0].set_xlabel("Training Duration (seconds)")
        axes[0].set_ylabel("Number of Genes")
        axes[0].set_title("A) Training Duration Distribution")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Memory Usage vs Performance
        valid_data = self.df_summary.dropna(subset=["peak_gpu_memory", "best_val_r2"])
        scatter = axes[1].scatter(
            valid_data["peak_gpu_memory"],
            valid_data["best_val_r2"],
            alpha=0.6,
            c=valid_data["training_duration"],
            cmap="plasma",
            s=30,
        )
        axes[1].set_xlabel("Peak GPU Memory (GB)")
        axes[1].set_ylabel("Validation R²")
        axes[1].set_title("B) Memory Usage vs Performance")
        axes[1].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label("Training Duration (s)")

        plt.tight_layout()
        plt.savefig(
            f"{save_path}/04_resource_analysis.png", dpi=600, bbox_inches="tight"
        )
        plt.savefig(f"{save_path}/04_resource_analysis.pdf", bbox_inches="tight")
        plt.show()

    def _create_summary_table(self, save_path):
        """Create comprehensive summary statistics"""
        metrics_cols = [
            "best_val_r2",
            "best_val_rmse",
            "best_val_mae",
            "final_epochs",
            "total_parameters",
            "training_duration",
            "peak_gpu_memory",
            "total_samples",
        ]

        summary_stats = self.df_summary[metrics_cols].describe()

        # Save as CSV
        summary_stats.to_csv(f"{save_path}/05_summary_statistics.csv")

        # Create a formatted table plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("tight")
        ax.axis("off")

        table_data = summary_stats.round(3).T
        table = ax.table(
            cellText=table_data.values,
            rowLabels=table_data.index,
            colLabels=table_data.columns,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)

        plt.title(
            "Summary Statistics for All Genes (N=2000)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.savefig(f"{save_path}/05_summary_table.png", dpi=600, bbox_inches="tight")
        plt.savefig(f"{save_path}/05_summary_table.pdf", bbox_inches="tight")
        plt.show()


def main():

    analyzer = GeneTrainingAnalyzer(base_path="KAN_Implementation/kan_models")

    df_summary, df_epochs = analyzer.process_all_logs()

    analyzer.create_consolidated_diagrams(save_path="gene_analysis_results")

    df_summary.to_csv("gene_analysis_results/gene_summary_data.csv", index=False)
    df_epochs.to_csv("gene_analysis_results/epoch_data.csv", index=False)

    return analyzer, df_summary, df_epochs


if __name__ == "__main__":
    analyzer, df_summary, df_epochs = main()
