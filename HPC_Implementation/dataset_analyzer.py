#######For NEURS-2021 GSEAPY dataset
# import scanpy as sc
# import anndata as ad
# from pathlib import Path
# import pooch
# import matplotlib.pyplot as plt

#
# sc.settings.set_figure_params(dpi=200, facecolor="white", fontsize=16)
# expression_file = "KAN_Implementation/Data/exp_new.h5ad"
# adata = sc.read_h5ad(expression_file)
# print(f"Original data shape: {adata.shape}")
# # Normalizing to median total counts
# # sc.pp.normalize_total(adata)
# # Logarithmize the data
# # sc.pp.log1p(adata)
# # mitochondrial genes, "MT-" for human, "Mt-" for mouse
# adata.var["mt"] = adata.var_names.str.startswith("MT-")
# # ribosomal genes
# adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# # hemoglobin genes
# adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

# sc.pp.calculate_qc_metrics(
#     adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
# )

# sc.pl.violin(
#     adata,
#     ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
#     jitter=0.4,
#     multi_panel=True,
# )
# # sc.pp.highly_variable_genes(adata, n_top_genes=1000, batch_key="sample")
# # sc.pl.highly_variable_genes(adata)


# # ##Statistics
# import scanpy as sc
# import pandas as pd
# from scipy.stats import skew

# # ---------------------------------------------------
# # Load data
# # ---------------------------------------------------
# sc.settings.set_figure_params(dpi=200, facecolor="white", fontsize=16)
# expression_file = "KAN_Implementation/Data/exp_new.h5ad"
# adata = sc.read_h5ad(expression_file)
# sc.pp.normalize_total(adata)
# sc.pp.log1p(adata)

# print(f"Original data shape: {adata.shape}")

# # ---------------------------------------------------
# # Annotate gene categories
# # ---------------------------------------------------
# # Mitochondrial genes (MT- prefix for human)
# adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
# # Ribosomal genes
# adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# # Hemoglobin genes
# adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]", regex=True)

# # ---------------------------------------------------
# # Compute QC metrics
# # ---------------------------------------------------
# sc.pp.calculate_qc_metrics(
#     adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
# )

# # ---------------------------------------------------
# # Plot QC metrics
# # ---------------------------------------------------
# # sc.pl.violin(
# #     adata,
# #     ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
# #     jitter=0.4,
# #     multi_panel=True,
# # )

# # ---------------------------------------------------
# # Compute and display statistics
# # ---------------------------------------------------
# metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]

# print("\n=== Dataset dimensions ===")
# print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")

# print("\n=== Summary statistics ===")
# for metric in metrics:
#     if metric in adata.obs.columns:
#         vals = adata.obs[metric].to_numpy()
#         print(f"\n{metric}")
#         print(f"  Mean      : {vals.mean():.2f}")
#         print(f"  Median    : {pd.Series(vals).median():.2f}")
#         print(f"  Std Dev   : {vals.std():.2f}")
#         print(f"  Min       : {vals.min():.2f}")
#         print(f"  Max       : {vals.max():.2f}")
#         print(f"  Skewness  : {skew(vals):.2f}")
#     else:
#         print(f"\n{metric} not found in adata.obs")


# ###--------for scmultisim
# # Load file
# import scanpy as sc
# from scipy.stats import skew
# import anndata as ad
# from pathlib import Path
# import numpy as np

# adata = sc.read_h5ad("KAN_Implementation/Data/ctrl_only.h5ad")

# # Get basic dimensions
# n_cells, n_genes = adata.shape
# print(f"Cells: {n_cells}, Genes: {n_genes}")
# X = adata.X
# if not isinstance(X, np.ndarray):
#     X = X.toarray()

# adata.obs["total_counts"] = X.sum(axis=1)
# adata.obs["n_genes_by_counts"] = (X > 0).sum(axis=1)

# metrics = ["n_genes_by_counts", "total_counts"]

# for metric in metrics:
#     vals = adata.obs[metric]
#     print(f"\n{metric}")
#     print(f" Mean: {vals.mean():.2f}")
#     print(f" Median: {vals.median():.2f}")
#     print(f" Std: {vals.std():.2f}")
#     print(f" Min: {vals.min():.2f}")
#     print(f" Max: {vals.max():.2f}")
#     print(f" Skewness: {skew(vals):.2f}")

########For perturbed files from Gears Adamson dataset

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew

# Set figure parameters
sc.settings.set_figure_params(dpi=200, facecolor="white", fontsize=16)

# Load dataset
adata = sc.read_h5ad("KAN_Implementation/Data/ctrl_only_CREB1_zero.h5ad")
print(f"Original data shape: {adata.shape}")

# Annotate gene categories for QC
adata.var["mt"] = adata.var_names.str.startswith("MT-")  # mitochondrial genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))  # ribosomal genes
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")  # hemoglobin genes

# Compute QC metrics (adds total_counts, n_genes_by_counts, pct_counts_mt)
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)

# Create violin plots
sc.pl.violin(
    adata,
    keys=["n_genes_by_counts", "total_counts"],
    jitter=0.4,
    multi_panel=True,
    rotation=15,
)
X = adata.X
if not isinstance(X, np.ndarray):
    X = X.toarray()

adata.obs["total_counts"] = X.sum(axis=1)
adata.obs["n_genes_by_counts"] = (X > 0).sum(axis=1)

metrics = ["n_genes_by_counts", "total_counts"]

for metric in metrics:
    vals = adata.obs[metric]
    print(f"\n{metric}")
    print(f" Mean: {vals.mean():.2f}")
    print(f" Median: {vals.median():.2f}")
    print(f" Std: {vals.std():.2f}")
    print(f" Min: {vals.min():.2f}")
    print(f" Max: {vals.max():.2f}")
    print(f" Skewness: {skew(vals):.2f}")
