import torch
import numpy as np
import json
import os
import re
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from data_loader import HPCSharedGeneDataManager
import math
from scipy import stats
import scipy.special
from statsmodels.stats.multitest import multipletests
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


class DualOutputLogger:
    """
    Logger class that writes output to both console and a log file simultaneously.
    """

    def __init__(self, log_file_path: Path):
        """
        Initialize the dual logger.

        Args:
            log_file_path: Path to the log file
        """
        self.log_file_path = log_file_path
        self.terminal = sys.stdout
        self.log_file = None

        # Create log file and write header
        self._open_log_file()
        self._write_log_header()

    def _open_log_file(self):
        """Open the log file for writing."""
        try:
            self.log_file = open(self.log_file_path, "w", encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not open log file {self.log_file_path}: {e}")
            self.log_file = None

    def _write_log_header(self):
        """Write header information to the log file."""
        if self.log_file:
            header = f"""
{'='*80}
SYMBOLIC FORMULA EVALUATION LOG
{'='*80}
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log File: {self.log_file_path}
{'='*80}

"""
            self.log_file.write(header)
            self.log_file.flush()

    def write(self, message):
        """Write message to both console and log file."""
        # Write to console
        self.terminal.write(message)
        self.terminal.flush()

        # Write to log file
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

    def close(self):
        """Close the log file and restore original stdout."""
        if self.log_file:
            # Write footer
            footer = f"""
{'='*80}
End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log completed successfully.
{'='*80}
"""
            self.log_file.write(footer)
            self.log_file.close()
            self.log_file = None


class SymbolicFormulaEvaluator:
    """
    Evaluates symbolic formulas for gene expression prediction.
    Uses raw data without normalization and reports Pearson & Spearman correlations only.
    """

    def __init__(self, data_manager: HPCSharedGeneDataManager):
        """
        Initialize the evaluator with a data manager.

        Args:
            data_manager: HPCSharedGeneDataManager instance with loaded data
        """
        self.data_manager = data_manager
        self.results = {}

    def read_symbolic_formula(self, formula_file: Path) -> str:
        """
        Read symbolic formula from file.

        Args:
            formula_file: Path to symbolic_formula.txt file

        Returns:
            String containing the symbolic formula
        """
        try:
            with open(formula_file, "r") as f:
                formula = f.read().strip()
            return formula
        except Exception as e:
            print(f"Error reading formula file {formula_file}: {e}")
            return ""

    def read_related_genes(self, genes_file: Path) -> Dict[str, float]:
        """
        Read related genes values from JSON file.

        Args:
            genes_file: Path to related_genes.json file

        Returns:
            Dictionary mapping gene names to their values
        """
        try:
            with open(genes_file, "r") as f:
                genes_data = json.load(f)
            return genes_data
        except Exception as e:
            print(f"Error reading genes file {genes_file}: {e}")
            return {}

    def convert_formula_to_executable(self, formula: str) -> str:
        """
        Convert symbolic formula to executable Python code with comprehensive
        mathematical function support.

        Args:
            formula: Symbolic formula string

        Returns:
            Executable Python formula string
        """
        import re

        # Start with the original formula
        executable_formula = formula

        # Comprehensive mathematical function mappings
        function_mappings = {
            # Basic arithmetic functions
            "abs": "np.abs",
            "Abs": "np.abs",
            "ABS": "np.abs",
            # Exponential and logarithmic functions
            "exp": "np.exp",
            "Exp": "np.exp",
            "EXP": "np.exp",
            "log": "np.log",
            "Log": "np.log",
            "LOG": "np.log",
            "log1p": "np.log1p",
            "Log1p": "np.log1p",
            "LOG1P": "np.log1p",
            "log2": "np.log2",
            "Log2": "np.log2",
            "LOG2": "np.log2",
            "log10": "np.log10",
            "Log10": "np.log10",
            "LOG10": "np.log10",
            # Power functions
            "sqrt": "np.sqrt",
            "Sqrt": "np.sqrt",
            "SQRT": "np.sqrt",
            "power": "np.power",
            "Power": "np.power",
            "POWER": "np.power",
            "pow": "np.power",
            "Pow": "np.power",
            "POW": "np.power",
            # Trigonometric functions
            "sin": "np.sin",
            "Sin": "np.sin",
            "SIN": "np.sin",
            "cos": "np.cos",
            "Cos": "np.cos",
            "COS": "np.cos",
            "tan": "np.tan",
            "Tan": "np.tan",
            "TAN": "np.tan",
            # Hyperbolic functions
            "sinh": "np.sinh",
            "Sinh": "np.sinh",
            "SINH": "np.sinh",
            "cosh": "np.cosh",
            "Cosh": "np.cosh",
            "COSH": "np.cosh",
            "tanh": "np.tanh",
            "Tanh": "np.tanh",
            "TANH": "np.tanh",
            # Special functions
            "erf": "scipy.special.erf",
            "Erf": "scipy.special.erf",
            "ERF": "scipy.special.erf",
            "gamma": "scipy.special.gamma",
            "Gamma": "scipy.special.gamma",
            "GAMMA": "scipy.special.gamma",
            # Rounding functions
            "floor": "np.floor",
            "Floor": "np.floor",
            "FLOOR": "np.floor",
            "ceil": "np.ceil",
            "Ceil": "np.ceil",
            "CEIL": "np.ceil",
            "round": "np.round",
            "Round": "np.round",
            "ROUND": "np.round",
            "sign": "np.sign",
            "Sign": "np.sign",
            "SIGN": "np.sign",
            # Min/Max functions
            "min": "np.minimum",
            "Min": "np.minimum",
            "MIN": "np.minimum",
            "max": "np.maximum",
            "Max": "np.maximum",
            "MAX": "np.maximum",
            # Constants
            "pi": "np.pi",
            "Pi": "np.pi",
            "PI": "np.pi",
            "e": "np.e",
            "E": "np.e",
            "euler": "np.e",
            "Euler": "np.e",
            "inf": "np.inf",
            "Inf": "np.inf",
            "INF": "np.inf",
            "nan": "np.nan",
            "NaN": "np.nan",
            "NAN": "np.nan",
        }

        # Apply replacements using regex to match whole function names
        for func_name, replacement in function_mappings.items():
            pattern = r"\b" + re.escape(func_name) + r"\("
            replacement_str = replacement + "("
            executable_formula = re.sub(pattern, replacement_str, executable_formula)

        # Handle mathematical operators
        operator_replacements = {
            "^": "**",
            "mod": "%",
            "MOD": "%",
            "Mod": "%",
        }

        for old_op, new_op in operator_replacements.items():
            if old_op not in ["**"]:
                executable_formula = executable_formula.replace(old_op, new_op)

        # Clean up any double replacements
        executable_formula = re.sub(r"np\.np\.", "np.", executable_formula)
        executable_formula = re.sub(r"scipy\.scipy\.", "scipy.", executable_formula)
        executable_formula = re.sub(r"\*\*+", "**", executable_formula)

        return executable_formula

    def evaluate_formula_vectorized(
        self, formula: str, gene_values_dict: Dict[str, List[float]]
    ) -> List[float]:
        """
        Evaluate the symbolic formula with given gene values for all samples.

        Args:
            formula: Executable formula string
            gene_values_dict: Dictionary of gene names to lists of values

        Returns:
            List of computed formula results for all samples
        """
        try:
            import scipy.special

            # Create evaluation environment
            eval_env = {
                "np": np,
                "math": math,
                "scipy": scipy,
                "__builtins__": {},
                # Constants
                "pi": np.pi,
                "Pi": np.pi,
                "PI": np.pi,
                "e": np.e,
                "E": np.e,
                "inf": np.inf,
                "Inf": np.inf,
                "INF": np.inf,
                "nan": np.nan,
                "NaN": np.nan,
                "NAN": np.nan,
            }

            # Convert lists to numpy arrays for vectorized operations
            for gene_name, values in gene_values_dict.items():
                eval_env[gene_name] = np.array(values)

            # Get the number of samples
            sample_size = len(next(iter(gene_values_dict.values())))

            # Evaluate the formula for all samples
            result = eval(formula, eval_env)

            # Handle different result types
            if np.isscalar(result):
                return [float(result)] * sample_size
            elif hasattr(result, "__len__"):
                # Convert to list of floats, handling any NaN or inf values
                result_list = []
                for x in result:
                    if np.isnan(x) or np.isinf(x):
                        result_list.append(0.0)  # Replace NaN/inf with 0
                    else:
                        result_list.append(float(x))
                return result_list
            else:
                return [float(result)] * sample_size

        except Exception as e:
            print(f"Error evaluating formula: {e}")
            print(f"Formula: {formula}")

            # If all else fails, return zeros
            sample_size = len(next(iter(gene_values_dict.values())))
            print(f"Falling back to zero predictions for {sample_size} samples")
            return [0.0] * sample_size

    def process_gene_folder(self, gene_folder: Path) -> Dict:
        """
        Process a single gene folder containing symbolic formula and related genes.
        Returns only actual values, predicted values, and correlations.

        Args:
            gene_folder: Path to gene folder

        Returns:
            Dictionary containing results for this gene
        """
        gene_name = gene_folder.name
        print(f"Processing gene: {gene_name}")

        # File paths
        formula_file = gene_folder / "symbolic_formula.txt"
        genes_file = gene_folder / "related_genes.json"

        # Check if files exist
        if not formula_file.exists():
            print(f"Formula file not found: {formula_file}")
            return {"gene_name": gene_name, "error": "Formula file not found"}

        if not genes_file.exists():
            print(f"Related genes file not found: {genes_file}")
            return {"gene_name": gene_name, "error": "Related genes file not found"}

        # Read symbolic formula
        formula = self.read_symbolic_formula(formula_file)
        if not formula:
            return {"gene_name": gene_name, "error": "Failed to read formula"}

        # Read related genes values (validation set data - RAW values)
        gene_values_dict = self.read_related_genes(genes_file)
        if not gene_values_dict:
            return {"gene_name": gene_name, "error": "Failed to read gene values"}

        # Extract raw actual values for the target gene
        if gene_name not in gene_values_dict:
            print(f"Target gene {gene_name} not found in related_genes.json")
            return {
                "gene_name": gene_name,
                "error": "Target gene not in validation data",
            }

        # Extract actual values for the target gene
        actual_values = gene_values_dict[gene_name]

        # Remove target gene from input variables for formula evaluation
        input_gene_values = {
            k: v for k, v in gene_values_dict.items() if k != gene_name
        }

        # Convert formula to executable form
        executable_formula = self.convert_formula_to_executable(formula)

        # Debug information
        print(f"  Formula length: {len(formula)} characters")
        print(f"  Input genes: {len(input_gene_values)}")
        print(f"  Validation samples: {len(actual_values)}")

        # Compute predicted values for all validation samples
        predicted_values = self.evaluate_formula_vectorized(
            executable_formula, input_gene_values
        )

        # Ensure same length
        min_length = min(len(actual_values), len(predicted_values))
        actual_values = actual_values[:min_length]
        predicted_values = predicted_values[:min_length]

        # Calculate correlation statistics
        actual_array = np.array(actual_values)
        predicted_array = np.array(predicted_values)

        # Initialize correlation values
        pearson_corr, pearson_p = 0.0, 1.0
        spearman_corr, spearman_p = 0.0, 1.0

        # Pearson correlation
        if (
            len(actual_values) > 2
            and np.std(actual_array) > 1e-10
            and np.std(predicted_array) > 1e-10
        ):
            pearson_corr, pearson_p = stats.pearsonr(actual_array, predicted_array)
        else:
            print(
                f"WARNING: Cannot compute Pearson correlation - insufficient variance"
            )

        # Spearman correlation
        if len(actual_values) > 2:
            try:
                spearman_corr, spearman_p = stats.spearmanr(
                    actual_array, predicted_array
                )
                # Handle NaN results from spearmanr
                if np.isnan(spearman_corr) or np.isnan(spearman_p):
                    spearman_corr, spearman_p = 0.0, 1.0
            except Exception as e:
                print(f"WARNING: Error computing Spearman correlation: {e}")
                spearman_corr, spearman_p = 0.0, 1.0
        else:
            print(
                f"WARNING: Cannot compute Spearman correlation - insufficient samples"
            )

        results = {
            "gene_name": gene_name,
            "n_samples": len(actual_values),
            "actual_values": actual_values,
            "predicted_values": predicted_values,
            # Correlation statistics
            "pearson_correlation": float(pearson_corr),
            "pearson_p_value": float(pearson_p),
            "spearman_correlation": float(spearman_corr),
            "spearman_p_value": float(spearman_p),
        }

        print(f"Pearson r: {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"Spearman S.P: {spearman_corr:.4f} (p={spearman_p:.4f})")

        return results

    def process_all_genes(self, base_folder: Path) -> Dict[str, Dict]:
        """
        Process all gene folders in the base directory.

        Args:
            base_folder: Base directory containing gene folders

        Returns:
            Dictionary mapping gene names to their results
        """
        all_results = {}

        # Find all gene folders (subdirectories)
        gene_folders = [f for f in base_folder.iterdir() if f.is_dir()]

        print(f"Found {len(gene_folders)} gene folders to process")

        for gene_folder in gene_folders:
            try:
                results = self.process_gene_folder(gene_folder)
                all_results[gene_folder.name] = results
            except Exception as e:
                print(f"Error processing gene folder {gene_folder.name}: {e}")
                all_results[gene_folder.name] = {
                    "gene_name": gene_folder.name,
                    "error": str(e),
                }

        return all_results

    def save_results_to_tsv(self, results: Dict[str, Dict], output_file: Path):
        """
        Save results to a TSV file with only core metrics.
        Includes significance flags based on Benjamini-Hochberg adjusted p-values at 95% confidence level.

        Args:
            results: Results dictionary from process_all_genes
            output_file: Path to output TSV file
        """
        try:
            # Collect valid results (no errors)
            valid_results = [r for r in results.values() if "error" not in r]

            if not valid_results:
                print("No valid results to save")
                return

            # Extract p-values for multiple testing correction
            pearson_p_values = [r["pearson_p_value"] for r in valid_results]
            spearman_p_values = [r["spearman_p_value"] for r in valid_results]

            # Apply Benjamini-Hochberg correction for both correlation types with alpha=0.05 (95% confidence)
            if len(pearson_p_values) > 1:
                pearson_rejected, pearson_adjusted_p_values, _, _ = multipletests(
                    pearson_p_values, alpha=0.05, method="fdr_bh"
                )

                spearman_rejected, spearman_adjusted_p_values, _, _ = multipletests(
                    spearman_p_values, alpha=0.05, method="fdr_bh"
                )
            else:
                pearson_adjusted_p_values = pearson_p_values
                spearman_adjusted_p_values = spearman_p_values
                # For single test case, just compare to alpha directly
                pearson_rejected = [p <= 0.05 for p in pearson_p_values]
                spearman_rejected = [p <= 0.05 for p in spearman_p_values]

            # Prepare TSV data
            tsv_data = []

            for i, (gene_name, result) in enumerate(results.items()):
                if "error" in result:
                    # Handle error cases
                    row = {
                        "gene_name": gene_name,
                        "n_samples": "NA",
                        "pearson_correlation": "NA",
                        "pearson_adjusted_p_value": "NA",
                        "pearson_significant": "NA",
                        "spearman_correlation": "NA",
                        "spearman_adjusted_p_value": "NA",
                        "spearman_significant": "NA",
                        "error": result["error"],
                    }
                else:
                    # Find index in valid results for adjusted p-values
                    valid_idx = next(
                        j
                        for j, vr in enumerate(valid_results)
                        if vr["gene_name"] == gene_name
                    )

                    # Determine significance based on adjusted p-value (alpha=0.05)
                    pearson_significant = "YES" if pearson_rejected[valid_idx] else "NO"
                    spearman_significant = (
                        "YES" if spearman_rejected[valid_idx] else "NO"
                    )

                    row = {
                        "gene_name": result["gene_name"],
                        "n_samples": result["n_samples"],
                        "pearson_correlation": f"{result['pearson_correlation']:.6f}",
                        "pearson_adjusted_p_value": f"{pearson_adjusted_p_values[valid_idx]:.2e}",
                        "pearson_significant": pearson_significant,
                        "spearman_correlation": f"{result['spearman_correlation']:.6f}",
                        "spearman_adjusted_p_value": f"{spearman_adjusted_p_values[valid_idx]:.2e}",
                        "spearman_significant": spearman_significant,
                        "error": "None",
                    }

                tsv_data.append(row)

            # Sort by Pearson correlation coefficient (descending, treating NA as -inf)
            def sort_key(row):
                if row["pearson_correlation"] == "NA":
                    return -float("inf")
                return float(row["pearson_correlation"])

            tsv_data.sort(key=sort_key, reverse=True)

            # Write TSV file
            fieldnames = [
                "gene_name",
                "n_samples",
                "pearson_correlation",
                "pearson_adjusted_p_value",
                "pearson_significant",
                "spearman_correlation",
                "spearman_adjusted_p_value",
                "spearman_significant",
                "error",
            ]

            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(tsv_data)

            # Calculate and print statistics about significant genes
            if valid_results:
                pearson_sig_count = sum(
                    1 for row in tsv_data if row.get("pearson_significant") == "YES"
                )
                spearman_sig_count = sum(
                    1 for row in tsv_data if row.get("spearman_significant") == "YES"
                )
                valid_count = len(valid_results)

                print(f"Results saved to {output_file}")
                print(f"Total genes processed: {len(results)}")
                print(f"Valid results: {valid_count}")
                print(
                    f"Significant genes (Pearson, 95% confidence): {pearson_sig_count} ({pearson_sig_count/valid_count*100:.1f}%)"
                )
                print(
                    f"Significant genes (Spearman, 95% confidence): {spearman_sig_count} ({spearman_sig_count/valid_count*100:.1f}%)"
                )
            else:
                print(f"Results saved to {output_file}")
                print(f"Total genes processed: {len(results)}")
                print(f"Valid results: 0")

        except Exception as e:
            print(f"Error saving results to TSV: {e}")
            import traceback

            traceback.print_exc()

    def save_detailed_results(self, results: Dict[str, Dict], output_dir: Path):
        """
        Save detailed results including actual vs predicted values for each gene.

        Args:
            results: Results dictionary from process_all_genes
            output_dir: Directory to save detailed results
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            for gene_name, result in results.items():
                if "error" not in result and "actual_values" in result:
                    # Save detailed comparison
                    detailed_file = output_dir / f"{gene_name}.tsv"

                    with open(detailed_file, "w", newline="") as f:
                        writer = csv.writer(f, delimiter="\t")
                        writer.writerow(
                            [
                                "sample_index",
                                "actual_value",
                                "predicted_value",
                            ]
                        )

                        for i, (actual, predicted) in enumerate(
                            zip(result["actual_values"], result["predicted_values"])
                        ):
                            writer.writerow(
                                [
                                    i,
                                    f"{actual:.6f}",
                                    f"{predicted:.6f}",
                                ]
                            )

            print(f"Detailed results saved to {output_dir}")

        except Exception as e:
            print(f"Error saving detailed results: {e}")


def main(
    base_folder: str,
    expression_file: str = None,
    network_file: str = None,
    output_dir: str = "evaluation_results",
):
    """
    Main function to run the symbolic formula evaluation on all gene folders.
    Reports only core metrics: actual values, predicted values, correlations, and adjusted p-values.

    Args:
        base_folder: Path to directory containing gene folders
        expression_file: Path to expression data file (optional if data_manager already loaded)
        network_file: Path to network data file (optional if data_manager already loaded)
        output_dir: Directory to save results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set up logging to capture all output
    log_file_path = output_path / "evaluation_log.txt"
    logger = DualOutputLogger(log_file_path)

    # Redirect stdout to our logger
    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("SYMBOLIC FORMULA EVALUATOR - CORE METRICS ONLY")
        print("=" * 60)
        print(
            "Output: Actual Values, Predicted Values, Correlations, Adjusted P-values"
        )
        print("=" * 60)

        # Initialize data manager
        print("Initializing data manager...")
        data_manager = HPCSharedGeneDataManager(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load data if files provided
        if expression_file and network_file:
            print("Loading data...")
            data_manager.load_data(Path(expression_file), Path(network_file))
        else:
            print(
                "Warning: No data files provided. Make sure data_manager is already loaded."
            )

        # Initialize evaluator
        evaluator = SymbolicFormulaEvaluator(data_manager)

        # Process all gene folders
        print(f"Processing gene folders in: {base_folder}")
        base_path = Path(base_folder)

        if not base_path.exists():
            print(f"Error: Base folder {base_folder} does not exist")
            return

        # Process all genes
        all_results = evaluator.process_all_genes(base_path)

        # Save results to TSV
        tsv_file = output_path / "core_results.tsv"
        evaluator.save_results_to_tsv(all_results, tsv_file)

        # Save detailed results (actual vs predicted values)
        detailed_dir = output_path / "actual_vs_predicted"
        evaluator.save_detailed_results(all_results, detailed_dir)

        # Save complete results as JSON
        json_file = output_path / "complete_results.json"
        with open(json_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\nEvaluation complete! Results saved to {output_dir}/")
        print(f"Complete log saved to: {log_file_path}")
        print("\nKey files generated:")
        print(f"  - {tsv_file.name}: Core results (correlations & adjusted p-values)")
        print(f"  - {detailed_dir.name}/: Per-gene actual vs predicted values")
        print(f"  - {json_file.name}: Complete results with all values")

        # Clean up
        data_manager.cleanup()

    except Exception as e:
        print(f"Fatal error in main execution: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Restore original stdout and close logger
        sys.stdout = original_stdout
        logger.close()


if __name__ == "__main__":

    base_folder = "KAN_Implementation/kan_models"  # Update this path
    expression_file = "KAN_Implementation/Data/simulated_gene_expression_1139.h5ad"  # Update this path
    network_file = "KAN_Implementation/Data/grnboost2_1139.tsv"  # Update this path

    main(
        base_folder=base_folder,
        expression_file=expression_file,
        network_file=network_file,
        output_dir="core_evaluation_results",
    )
