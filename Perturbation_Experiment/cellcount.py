import pandas as pd
import numpy as np
import os
import glob
from scipy.io import mmread
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import anndata as ad

    ANNDATA_AVAILABLE = True
except ImportError:
    print("Warning: anndata not installed. Install with: pip install anndata")
    ANNDATA_AVAILABLE = False


class GeneExpressionAnalyzer:
    """
    A class to analyze gene expression files and count cells and samples
    """

    def __init__(self):
        self.file_info = []
        self.total_cells = 0
        self.total_samples = 0

    def analyze_csv_tsv(
        self, file_path, sep=",", header=0, index_col=0, sample_column=None
    ):
        """
        Analyze CSV/TSV files
        Assumes: rows = genes, columns = cells (standard format)
        sample_column: column name that contains sample information
        """
        try:
            df = pd.read_csv(file_path, sep=sep, header=header, index_col=index_col)

            # Number of cells (columns after gene names)
            n_cells = df.shape[1]
            n_genes = df.shape[0]

            # Try to determine number of samples
            n_samples = self._determine_samples_from_columns(df.columns, file_path)

            file_info = {
                "file": os.path.basename(file_path),
                "format": "CSV/TSV",
                "cells": n_cells,
                "genes": n_genes,
                "samples": n_samples,
                "non_zero_entries": np.count_nonzero(df.values),
                "mean_expression_per_cell": df.mean(axis=0).mean(),
                "mean_genes_per_cell": (df > 0).sum(axis=0).mean(),
            }

            return file_info, df

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None, None

    def _determine_samples_from_columns(self, columns, file_path):
        """
        Try to determine number of samples from column names
        """
        # Convert columns to string for analysis
        col_strings = [str(col) for col in columns]

        # Method 1: Look for common sample patterns
        sample_patterns = []

        # Pattern: Sample_1, Sample_2, etc.
        sample_ids = set()
        for col in col_strings:
            # Look for patterns like: Sample1, Sample_1, sample1, S1, etc.
            import re

            matches = re.findall(r"[Ss]ample[_\-]?(\d+)|[Ss](\d+)|Sample(\d+)", col)
            if matches:
                for match in matches:
                    sample_id = next(x for x in match if x)
                    sample_ids.add(sample_id)

        if sample_ids:
            n_samples = len(sample_ids)
            print(f"  Found {n_samples} unique samples based on naming pattern")
            return n_samples

        # Method 2: Look for underscores or common delimiters (e.g., SampleA_Cell1, SampleA_Cell2)
        prefixes = set()
        for col in col_strings:
            if "_" in col:
                prefix = col.split("_")[0]
                prefixes.add(prefix)
            elif "-" in col:
                prefix = col.split("-")[0]
                prefixes.add(prefix)

        if len(prefixes) > 1 and len(prefixes) < len(col_strings):
            print(f"  Found {len(prefixes)} unique samples based on prefix pattern")
            return len(prefixes)

        # Method 3: Default assumption - treat each file as one sample
        print(f"  Could not determine sample structure, assuming 1 sample per file")
        return 1

    def analyze_h5_file(self, file_path):
        """
        Analyze H5 files (common in single-cell RNA-seq)
        """
        try:
            with h5py.File(file_path, "r") as f:
                # Common H5 structures for scRNA-seq
                if "matrix" in f.keys():
                    # 10X Genomics format
                    matrix = f["matrix"]
                    if "data" in matrix.keys():
                        data = matrix["data"][:]
                        indices = matrix["indices"][:]
                        indptr = matrix["indptr"][:]

                        # Reconstruct sparse matrix dimensions
                        n_cells = len(indptr) - 1
                        n_genes = (
                            matrix["shape"][0]
                            if "shape" in matrix.keys()
                            else max(indices) + 1
                        )
                    else:
                        # Dense matrix
                        data = matrix[:]
                        n_genes, n_cells = data.shape
                else:
                    # Try to find expression matrix
                    keys = list(f.keys())
                    print(f"Available keys in {file_path}: {keys}")
                    return None

                n_samples = 1

                file_info = {
                    "file": os.path.basename(file_path),
                    "format": "H5",
                    "cells": n_cells,
                    "genes": n_genes,
                    "samples": n_samples,
                    "non_zero_entries": (
                        len(data) if hasattr(data, "__len__") else "Unknown"
                    ),
                }

                return file_info

        except Exception as e:
            print(f"Error reading H5 file {file_path}: {e}")
            return None

    def analyze_mtx_file(self, mtx_path, features_path=None, barcodes_path=None):
        """
        Analyze Matrix Market (.mtx) files with optional features and barcodes
        """
        try:
            # Read sparse matrix
            matrix = mmread(mtx_path)

            # Matrix is typically genes x cells
            n_genes, n_cells = matrix.shape

            # For MTX files, typically assume 1 sample per file
            n_samples = 1

            file_info = {
                "file": os.path.basename(mtx_path),
                "format": "MTX",
                "cells": n_cells,
                "genes": n_genes,
                "samples": n_samples,
                "non_zero_entries": matrix.nnz,
                "sparsity": 1 - (matrix.nnz / (n_genes * n_cells)),
            }

            # Read additional files if provided
            if features_path and os.path.exists(features_path):
                features = pd.read_csv(features_path, sep="\t", header=None)
                file_info["features_file"] = os.path.basename(features_path)

            if barcodes_path and os.path.exists(barcodes_path):
                barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)
                file_info["barcodes_file"] = os.path.basename(barcodes_path)

            return file_info

        except Exception as e:
            print(f"Error reading MTX file {mtx_path}: {e}")
            return None

    def analyze_excel(self, file_path, sheet_name=0):
        """
        Analyze Excel files
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

            n_cells = df.shape[1]
            n_genes = df.shape[0]

            # Determine number of samples from column names
            n_samples = self._determine_samples_from_columns(df.columns, file_path)

            file_info = {
                "file": os.path.basename(file_path),
                "format": "Excel",
                "sheet": sheet_name,
                "cells": n_cells,
                "genes": n_genes,
                "samples": n_samples,
                "non_zero_entries": np.count_nonzero(df.values),
                "mean_expression_per_cell": df.mean(axis=0).mean(),
            }

            return file_info, df

        except Exception as e:
            print(f"Error reading Excel file {file_path}: {e}")
            return None, None

    def analyze_h5ad_file(self, file_path):
        """
        Analyze H5AD (AnnData) files - common in single-cell RNA-seq
        """
        if not ANNDATA_AVAILABLE:
            print(f"Cannot analyze {file_path}: anndata package not installed")
            print("Install with: pip install anndata")
            return None

        try:
            # Read AnnData object
            adata = ad.read_h5ad(file_path)

            # Basic information
            n_cells = adata.n_obs  # number of observations (cells)
            n_genes = adata.n_vars  # number of variables (genes)

            # Determine number of samples
            n_samples = self._determine_samples_from_anndata(adata)

            # Calculate additional metrics
            X = adata.X
            if hasattr(X, "nnz"):  # sparse matrix
                non_zero_entries = X.nnz
                sparsity = 1 - (non_zero_entries / (n_cells * n_genes))
            else:  # dense matrix
                non_zero_entries = np.count_nonzero(X)
                sparsity = 1 - (non_zero_entries / X.size)

            # Try to get some expression statistics
            try:
                if hasattr(X, "toarray"):
                    X_dense = (
                        X.toarray() if X.nnz < 1e6 else None
                    )  # Avoid memory issues
                else:
                    X_dense = X

                if X_dense is not None:
                    mean_expr_per_cell = np.mean(X_dense, axis=1).mean()
                    mean_genes_per_cell = np.mean(X_dense > 0, axis=1).mean() * n_genes
                else:
                    mean_expr_per_cell = "Too large to compute"
                    mean_genes_per_cell = "Too large to compute"
            except:
                mean_expr_per_cell = "Unable to compute"
                mean_genes_per_cell = "Unable to compute"

            file_info = {
                "file": os.path.basename(file_path),
                "format": "H5AD (AnnData)",
                "cells": n_cells,
                "genes": n_genes,
                "samples": n_samples,
                "non_zero_entries": non_zero_entries,
                "sparsity": f"{sparsity:.3f}",
                "mean_expression_per_cell": mean_expr_per_cell,
                "mean_genes_per_cell": mean_genes_per_cell,
                "obs_keys": (
                    list(adata.obs.columns) if len(adata.obs.columns) > 0 else "None"
                ),
                "var_keys": (
                    list(adata.var.columns) if len(adata.var.columns) > 0 else "None"
                ),
            }

            # Additional AnnData-specific info
            if adata.obs.shape[1] > 0:
                file_info["metadata_columns"] = len(adata.obs.columns)

            if hasattr(adata, "obsm") and len(adata.obsm.keys()) > 0:
                file_info["embeddings"] = list(adata.obsm.keys())

            print(f"Successfully analyzed {os.path.basename(file_path)}")
            print(f"  Cells: {n_cells:,}")
            print(f"  Genes: {n_genes:,}")
            print(f"  Samples: {n_samples}")
            print(f"  Sparsity: {sparsity:.1%}")
            if adata.obs.shape[1] > 0:
                print(f"  Metadata columns: {list(adata.obs.columns)}")

            return file_info, adata

        except Exception as e:
            print(f"Error reading H5AD file {file_path}: {e}")
            return None, None

    def _determine_samples_from_anndata(self, adata):
        """
        Determine number of samples from AnnData object
        """
        # Common sample identification columns
        sample_columns = [
            "sample",
            "sample_id",
            "donor",
            "donor_id",
            "patient",
            "patient_id",
            "batch",
            "experiment",
            "condition",
        ]

        for col in sample_columns:
            if col in adata.obs.columns:
                n_samples = len(adata.obs[col].unique())
                print(f"  Found {n_samples} unique samples in '{col}' column")
                return n_samples

        # Try case-insensitive search
        obs_cols_lower = [col.lower() for col in adata.obs.columns]
        for sample_col in sample_columns:
            if sample_col in obs_cols_lower:
                actual_col = adata.obs.columns[obs_cols_lower.index(sample_col)]
                n_samples = len(adata.obs[actual_col].unique())
                print(f"  Found {n_samples} unique samples in '{actual_col}' column")
                return n_samples

        # Look for columns containing sample-related keywords
        for col in adata.obs.columns:
            col_lower = col.lower()
            if any(
                keyword in col_lower
                for keyword in ["sample", "donor", "patient", "batch"]
            ):
                n_samples = len(adata.obs[col].unique())
                print(
                    f"  Found {n_samples} unique samples in '{col}' column (keyword match)"
                )
                return n_samples

        # Default: assume all cells from one sample
        print(f"  Could not identify sample column, assuming 1 sample")
        return 1

    def analyze_directory(self, directory_path, file_patterns=None):
        """
        Analyze all gene expression files in a directory
        """
        if file_patterns is None:
            file_patterns = [
                "*.csv",
                "*.tsv",
                "*.txt",
                "*.h5",
                "*.h5ad",
                "*.mtx",
                "*.xlsx",
                "*.xls",
            ]

        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(os.path.join(directory_path, pattern)))

        print(f"Found {len(all_files)} files to analyze...")

        for file_path in all_files:
            self.analyze_single_file(file_path)

    def analyze_single_file(self, file_path):
        """
        Analyze a single file based on its extension
        """
        file_ext = Path(file_path).suffix.lower()

        if file_ext in [".csv", ".txt"]:
            # Try comma first, then tab
            info, data = self.analyze_csv_tsv(file_path, sep=",")
            if info is None:
                info, data = self.analyze_csv_tsv(file_path, sep="\t")

        elif file_ext == ".tsv":
            info, data = self.analyze_csv_tsv(file_path, sep="\t")

        elif file_ext == ".h5":
            info = self.analyze_h5_file(file_path)

        elif file_ext == ".h5ad":
            info, data = self.analyze_h5ad_file(file_path)

        elif file_ext == ".mtx":
            # Look for accompanying files
            base_dir = os.path.dirname(file_path)
            features_file = os.path.join(base_dir, "features.tsv") or os.path.join(
                base_dir, "genes.tsv"
            )
            barcodes_file = os.path.join(base_dir, "barcodes.tsv")
            info = self.analyze_mtx_file(file_path, features_file, barcodes_file)

        elif file_ext in [".xlsx", ".xls"]:
            info, data = self.analyze_excel(file_path)

        else:
            print(f"Unsupported file format: {file_ext}")
            return

        if info:
            self.file_info.append(info)
            self.total_cells += info["cells"]
            self.total_samples += info["samples"]

    def get_summary(self):
        """
        Get summary statistics across all analyzed files
        """
        if not self.file_info:
            print("No files analyzed yet!")
            return

        df_summary = pd.DataFrame(self.file_info)

        print("=" * 60)
        print("GENE EXPRESSION FILE ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total files analyzed: {len(self.file_info)}")
        print(f"Total cells across all files: {self.total_cells:,}")
        print(f"Total samples across all files: {self.total_samples:,}")
        print(f"Total genes across all files: {df_summary['genes'].sum():,}")
        print(f"Average cells per file: {df_summary['cells'].mean():.1f}")
        print(f"Average samples per file: {df_summary['samples'].mean():.1f}")
        print(f"Average genes per file: {df_summary['genes'].mean():.1f}")
        print("\nPer-file breakdown:")
        print(
            df_summary[["file", "format", "cells", "genes", "samples"]].to_string(
                index=False
            )
        )

        return df_summary


# Usage Examples
def main():
    """
    Main function with usage examples
    """
    analyzer = GeneExpressionAnalyzer()

    #  Analyze a single file
    analyzer.analyze_single_file("Data\exp_new.h5ad")


if __name__ == "__main__":
    main()
