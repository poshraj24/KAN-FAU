import torch
import numpy as np
import json
import os
import csv
import sys
from pathlib import Path
from typing import Dict, List
import math
from scipy import stats
import scipy.special
from statsmodels.stats.multitest import multipletests
import warnings
from datetime import datetime
from data_loader import HPCSharedGeneDataManager

warnings.filterwarnings("ignore")


class DualOutputLogger:
    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        self.terminal = sys.stdout
        self.log_file = None
        self._open_log_file()
        self._write_log_header()

    def _open_log_file(self):
        try:
            self.log_file = open(self.log_file_path, "w", encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not open log file {self.log_file_path}: {e}")
            self.log_file = None

    def _write_log_header(self):
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
        self.terminal.write(message)
        self.terminal.flush()
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

    def close(self):
        if self.log_file:
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
    def __init__(self, data_manager: HPCSharedGeneDataManager):
        self.data_manager = data_manager

    def read_symbolic_formula(self, formula_file: Path) -> str:
        try:
            with open(formula_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading formula file {formula_file}: {e}")
            return ""

    def read_related_genes(self, genes_file: Path) -> Dict[str, List[float]]:
        try:
            with open(genes_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading genes file {genes_file}: {e}")
            return {}

    def convert_formula_to_executable(self, formula: str) -> str:
        import re

        f = formula
        fn = {
            "abs": "np.abs",
            "Abs": "np.abs",
            "ABS": "np.abs",
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
            "sqrt": "np.sqrt",
            "Sqrt": "np.sqrt",
            "SQRT": "np.sqrt",
            "power": "np.power",
            "Power": "np.power",
            "POWER": "np.power",
            "pow": "np.power",
            "Pow": "np.power",
            "POW": "np.power",
            "sin": "np.sin",
            "Sin": "np.sin",
            "SIN": "np.sin",
            "cos": "np.cos",
            "Cos": "np.cos",
            "COS": "np.cos",
            "tan": "np.tan",
            "Tan": "np.tan",
            "TAN": "np.tan",
            "sinh": "np.sinh",
            "Sinh": "np.sinh",
            "SINH": "np.sinh",
            "cosh": "np.cosh",
            "Cosh": "np.cosh",
            "COSH": "np.cosh",
            "tanh": "np.tanh",
            "Tanh": "np.tanh",
            "TANH": "np.tanh",
            "erf": "scipy.special.erf",
            "Erf": "scipy.special.erf",
            "ERF": "scipy.special.erf",
            "gamma": "scipy.special.gamma",
            "Gamma": "scipy.special.gamma",
            "GAMMA": "scipy.special.gamma",
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
            "min": "np.minimum",
            "Min": "np.minimum",
            "MIN": "np.minimum",
            "max": "np.maximum",
            "Max": "np.maximum",
            "MAX": "np.maximum",
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
        for k, v in fn.items():
            f = re.sub(r"\b" + k + r"\(", v + "(", f)
        f = (
            f.replace("^", "**")
            .replace("mod", "%")
            .replace("MOD", "%")
            .replace("Mod", "%")
        )
        f = re.sub(r"np\.np\.", "np.", f)
        f = re.sub(r"scipy\.scipy\.", "scipy.", f)
        return f

    def evaluate_formula_vectorized(
        self, formula: str, gene_values_dict: Dict[str, List[float]]
    ) -> List[float]:
        try:
            import scipy.special

            env = {
                "np": np,
                "math": math,
                "scipy": scipy,
                "__builtins__": {},
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
            for g, v in gene_values_dict.items():
                env[g] = np.array(v)
            n = len(next(iter(gene_values_dict.values())))
            res = eval(formula, env)
            if np.isscalar(res):
                return [float(res)] * n
            out = []
            for x in np.asarray(res).ravel():
                out.append(0.0 if (np.isnan(x) or np.isinf(x)) else float(x))
            return out
        except Exception as e:
            print(f"Error evaluating formula: {e}")
            print(f"Formula: {formula}")
            n = len(next(iter(gene_values_dict.values())))
            print(f"Falling back to zero predictions for {n} samples")
            return [0.0] * n

    def process_gene(
        self, base_data_dir: Path, base_formula_dir: Path, gene_folder_name: str
    ) -> Dict:
        data_dir = base_data_dir / gene_folder_name
        form_dir = base_formula_dir / gene_folder_name
        gene_name = gene_folder_name

        print(f"Processing gene: {gene_name}")

        formula_file = (
            form_dir / "symbolic_formula_reduced.txt"
        )  # from gene_folder2/cutoff_x.x
        genes_file = data_dir / "related_genes.json"  # from gene_folder1

        if not formula_file.exists():
            print(f"Formula file not found: {formula_file}")
            return {"gene_name": gene_name, "error": "Formula file not found"}

        if not genes_file.exists():
            print(f"Related genes file not found: {genes_file}")
            return {"gene_name": gene_name, "error": "Related genes file not found"}

        formula = self.read_symbolic_formula(formula_file)
        if not formula:
            return {"gene_name": gene_name, "error": "Failed to read formula"}

        gene_values = self.read_related_genes(genes_file)
        if not gene_values:
            return {"gene_name": gene_name, "error": "Failed to read gene values"}

        if gene_name not in gene_values:
            print(f"Target gene {gene_name} not found in related_genes.json")
            return {
                "gene_name": gene_name,
                "error": "Target gene not in validation data",
            }

        actual = gene_values[gene_name]
        inputs = {k: v for k, v in gene_values.items() if k != gene_name}

        exec_formula = self.convert_formula_to_executable(formula)
        print(f"Formula length: {len(formula)} characters")
        print(f"Input genes: {len(inputs)}")
        print(f"Validation samples: {len(actual)}")

        pred = self.evaluate_formula_vectorized(exec_formula, inputs)

        m = min(len(actual), len(pred))
        actual = actual[:m]
        pred = pred[:m]
        a = np.array(actual)
        p = np.array(pred)

        pr, pp = 0.0, 1.0
        sr, sp = 0.0, 1.0
        if len(actual) > 2 and np.std(a) > 1e-10 and np.std(p) > 1e-10:
            pr, pp = stats.pearsonr(a, p)
        else:
            print("WARNING: Cannot compute Pearson correlation - insufficient variance")
        if len(actual) > 2:
            try:
                sr, sp = stats.spearmanr(a, p)
                if np.isnan(sr) or np.isnan(sp):
                    sr, sp = 0.0, 1.0
            except Exception as e:
                print(f"  WARNING: Error computing Spearman correlation: {e}")
                sr, sp = 0.0, 1.0
        else:
            print("WARNING: Cannot compute Spearman correlation - insufficient samples")

        print(f"Pearson r: {pr:.4f} (p={pp:.4f})")
        print(f"Spearman S.P: {sr:.4f} (p={sp:.4f})")

        return {
            "gene_name": gene_name,
            "n_samples": len(actual),
            "actual_values": actual,
            "predicted_values": pred,
            "pearson_correlation": float(pr),
            "pearson_p_value": float(pp),
            "spearman_correlation": float(sr),
            "spearman_p_value": float(sp),
        }

    def process_all_genes_for_cutoff(
        self, base_data_dir: Path, cutoff_formula_dir: Path
    ) -> Dict[str, Dict]:
        if not cutoff_formula_dir.exists():
            raise FileNotFoundError(
                f"Cutoff folder does not exist: {cutoff_formula_dir}"
            )
        if not base_data_dir.exists():
            raise FileNotFoundError(f"Data base folder does not exist: {base_data_dir}")

        gene_folders = [p for p in cutoff_formula_dir.iterdir() if p.is_dir()]
        print(
            f"Found {len(gene_folders)} gene folders with formulas in: {cutoff_formula_dir}"
        )

        results = {}
        for gf in gene_folders:
            gene_name = gf.name
            try:
                results[gene_name] = self.process_gene(
                    base_data_dir, cutoff_formula_dir, gene_name
                )
            except Exception as e:
                print(f"Error processing {gene_name}: {e}")
                results[gene_name] = {"gene_name": gene_name, "error": str(e)}
        return results

    def save_results_to_tsv(self, results: Dict[str, Dict], output_file: Path):
        try:
            valid = [r for r in results.values() if "error" not in r]
            if not valid:
                print("No valid results to save")
                return

            pearson_p = [r["pearson_p_value"] for r in valid]
            spearman_p = [r["spearman_p_value"] for r in valid]

            if len(pearson_p) > 1:
                pr_rej, pr_adj, _, _ = multipletests(
                    pearson_p, alpha=0.05, method="fdr_bh"
                )
                sp_rej, sp_adj, _, _ = multipletests(
                    spearman_p, alpha=0.05, method="fdr_bh"
                )
            else:
                pr_adj, sp_adj = pearson_p, spearman_p
                pr_rej = [p <= 0.05 for p in pearson_p]
                sp_rej = [p <= 0.05 for p in spearman_p]

            name2idx = {r["gene_name"]: i for i, r in enumerate(valid)}
            rows = []
            for g, r in results.items():
                if "error" in r:
                    rows.append(
                        {
                            "gene_name": g,
                            "n_samples": "NA",
                            "pearson_correlation": "NA",
                            "pearson_adjusted_p_value": "NA",
                            "pearson_significant": "NA",
                            "spearman_correlation": "NA",
                            "spearman_adjusted_p_value": "NA",
                            "spearman_significant": "NA",
                            "error": r["error"],
                        }
                    )
                else:
                    i = name2idx[g]
                    rows.append(
                        {
                            "gene_name": r["gene_name"],
                            "n_samples": r["n_samples"],
                            "pearson_correlation": f"{r['pearson_correlation']:.6f}",
                            "pearson_adjusted_p_value": f"{pr_adj[i]:.2e}",
                            "pearson_significant": "YES" if pr_rej[i] else "NO",
                            "spearman_correlation": f"{r['spearman_correlation']:.6f}",
                            "spearman_adjusted_p_value": f"{sp_adj[i]:.2e}",
                            "spearman_significant": "YES" if sp_rej[i] else "NO",
                            "error": "None",
                        }
                    )

            def skey(row):
                return (
                    -float("inf")
                    if row["pearson_correlation"] == "NA"
                    else float(row["pearson_correlation"])
                )

            rows.sort(key=skey, reverse=True)

            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "gene_name",
                        "n_samples",
                        "pearson_correlation",
                        "pearson_adjusted_p_value",
                        "pearson_significant",
                        "spearman_correlation",
                        "spearman_adjusted_p_value",
                        "spearman_significant",
                        "error",
                    ],
                    delimiter="\t",
                )
                w.writeheader()
                w.writerows(rows)

            pr_sig = sum(1 for r in rows if r.get("pearson_significant") == "YES")
            sp_sig = sum(1 for r in rows if r.get("spearman_significant") == "YES")
            print(f"Results saved to {output_file}")
            print(
                f"Valid results: {len(valid)} | Pearson sig: {pr_sig} | Spearman sig: {sp_sig}"
            )
        except Exception as e:
            print(f"Error saving results to TSV: {e}")
            import traceback

            traceback.print_exc()

    def save_detailed_results(self, results: Dict[str, Dict], output_dir: Path):
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            for g, r in results.items():
                if "error" not in r and "actual_values" in r:
                    f = output_dir / f"{g}.tsv"
                    with open(f, "w", newline="", encoding="utf-8") as h:
                        w = csv.writer(h, delimiter="\t")
                        w.writerow(["sample_index", "actual_value", "predicted_value"])
                        for i, (a, p) in enumerate(
                            zip(r["actual_values"], r["predicted_values"])
                        ):
                            w.writerow([i, f"{a:.6f}", f"{p:.6f}"])
            print(f"Detailed results saved to {output_dir}")
        except Exception as e:
            print(f"Error saving detailed results: {e}")


def run_all_cutoffs(
    base_data_dir: Path,
    formula_root_dir: Path,
    output_root_dir: Path,
    data_manager: HPCSharedGeneDataManager,
):
    """
    Discover all cutoff_* folders inside formula_root_dir and process each one.
    """
    evaluator = SymbolicFormulaEvaluator(data_manager)

    cutoff_dirs = sorted(
        [
            d
            for d in formula_root_dir.iterdir()
            if d.is_dir() and d.name.startswith("cutoff_")
        ]
    )
    print(f"Discovered {len(cutoff_dirs)} cutoff folders under: {formula_root_dir}")

    for cdir in cutoff_dirs:
        cutoff_name = cdir.name  # e.g., 'cutoff_0.5'
        print("\n" + "=" * 60)
        print(f"Processing cutoff: {cutoff_name}")
        print("=" * 60)

        # Process this cutoff
        results = evaluator.process_all_genes_for_cutoff(base_data_dir, cdir)

        # Per-cutoff outputs
        out_dir = output_root_dir / cutoff_name
        out_dir.mkdir(parents=True, exist_ok=True)

        tsv_file = out_dir / "core_results.tsv"
        evaluator.save_results_to_tsv(results, tsv_file)

        detailed_dir = out_dir / "actual_vs_predicted"
        evaluator.save_detailed_results(results, detailed_dir)

        json_file = out_dir / "complete_results.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Completed cutoff {cutoff_name}. Results in: {out_dir}")


def main(
    gene_folder1: str,
    gene_folder2: str,
    expression_file: str = None,
    network_file: str = None,
    output_dir: str = "core_evaluation_results",
):
    """
    - Read formulas from: gene_folder2/cutoff_*/<GeneName>/symbolic_formula_reduced.txt
    - Read data from:     gene_folder1/<GeneName>/related_genes.json
    - Write outputs under: output_dir/cutoff_*/
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    log_file_path = output_path / "evaluation_log.txt"
    logger = DualOutputLogger(log_file_path)

    original_stdout = sys.stdout
    sys.stdout = logger
    try:
        print("SYMBOLIC FORMULA EVALUATOR - CORE METRICS ONLY")
        print("=" * 60)
        print(
            "Output: Actual Values, Predicted Values, Correlations, Adjusted P-values"
        )
        print("=" * 60)

        print("Initializing data manager...")
        dm = HPCSharedGeneDataManager(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        if expression_file and network_file:
            print("Loading data...")
            dm.load_data(Path(expression_file), Path(network_file))
        else:
            print(
                "Warning: No expression/network files provided. Proceeding with provided related_genes.json files."
            )

        base_data_dir = Path(gene_folder1)  # has related_genes.json per gene
        formula_root_dir = Path(gene_folder2)  # has cutoff_* subfolders
        output_root_dir = Path(output_dir)

        if not formula_root_dir.exists():
            print(f"Error: Formula root folder does not exist: {formula_root_dir}")
            return
        if not base_data_dir.exists():
            print(f"Error: Data base folder does not exist: {base_data_dir}")
            return

        print(f"Reading formulas from: {formula_root_dir}/cutoff_*")
        print(f"Reading data from:     {base_data_dir}")

        run_all_cutoffs(base_data_dir, formula_root_dir, output_root_dir, dm)

        print(f"\nEvaluation complete! See per-cutoff results under: {output_root_dir}")
        print(f"Complete log saved to: {log_file_path}")

        dm.cleanup()
    except Exception as e:
        print(f"Fatal error in main execution: {e}")
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        logger.close()


if __name__ == "__main__":

    gene_folder1 = "KAN_Implementation/kan_models"  # DATA source
    gene_folder2 = (
        "KAN_Implementation/kan_models_reduced"  # FORMULA root with cutoff_* subfolders
    )
    expression_file = "KAN_Implementation/Data/simulated_gene_expression_1139.h5ad"
    network_file = "KAN_Implementation/Data/grnboost2_1139.tsv"

    main(
        gene_folder1=gene_folder1,
        gene_folder2=gene_folder2,
        expression_file=expression_file,
        network_file=network_file,
        output_dir="core_evaluation_results",  # will contain cutoff_0.5/, cutoff_1.0/, ...
    )
