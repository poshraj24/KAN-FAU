#!/usr/bin/env python3
"""
GRNBoost2 over ALL genes (no HVG prefilter).

Loads an AnnData .h5ad expression file

Uses ALL genes in the expression matrix as candidates (no HVG filtering)
Builds TF list from a reference GRN's regulator column (intersected with available genes)
Runs arboreto.grnboost2 with a local Dask cluster
Saves a TSV: regulator, target, importance

Requires:
  pip install scanpy arboreto dask[distributed] threadpoolctl pandas numpy

Usage:
  python GRNBOOST2_noHVG.py \
      --expr KAN_Implementation/Data/gene_expression_1139.h5ad \
      --ref-grn KAN_Implementation/Data/grn_1139_truth.tsv \
      --out KAN_Implementation/Data/grnboost2_allgenes.tsv \
      --workers 8 \
      --normalize false
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# 3rd-party libs
import scanpy as sc
from threadpoolctl import threadpool_limits
from distributed import Client, LocalCluster
from arboreto.algo import grnboost2


def read_expression_h5ad(expr_path: Path, normalize: bool = False) -> pd.DataFrame:
    """
    Read .h5ad and return a DataFrame with genes x samples (dense).

    """
    print(f"Reading expression: {expr_path}")
    adata = sc.read_h5ad(str(expr_path))
    print(f"Raw AnnData shape (cells x genes): {adata.shape}")

    if normalize:
        print("ormalize_total -> log1p")
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    # Convert to dense
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    # genes x samples (transpose)
    expr_df = pd.DataFrame(
        X.T, index=adata.var_names.astype(str), columns=adata.obs_names.astype(str)
    )
    print(
        f"Expression table (genes x samples): {expr_df.shape[0]} x {expr_df.shape[1]}"
    )
    return expr_df


def read_reference_grn_tfs(ref_grn_path: Path, gene_index: pd.Index) -> list:
    """
    Build the TF (regulator) list from the first column of a reference GRN,
    intersected with the genes present in the expression matrix.
    """
    print(f"Reading reference GRN: {ref_grn_path}")
    ref = pd.read_csv(ref_grn_path, sep=None, engine="python", header=None)
    if ref.shape[1] < 2:
        raise ValueError(
            "Reference GRN must have at least two columns (regulator, target)."
        )

    regulators = ref.iloc[:, 0].astype(str).str.strip().unique().tolist()
    print(f"[TF] Regulators in ref GRN (unique): {len(regulators)}")

    # Intersection with expression genes
    tf_set = list(sorted(set(regulators).intersection(set(map(str, gene_index)))))
    print(f"[TF] Regulators present in expression: {len(tf_set)}")
    if len(tf_set) == 0:
        # fallback: use all genes as TFs (not ideal, but prevents a dead run)
        print("[TF] WARNING: No overlap. Falling back to ALL genes as TFs.")
        tf_set = list(map(str, gene_index))
    return tf_set


def run_grnboost2_allgenes(
    expr_df: pd.DataFrame, tf_names: list, n_workers: int = None
) -> pd.DataFrame:
    """
    Run arboreto.grnboost2 on ALL genes.
    expr_df must be genes x samples. Arboreto expects samples x genes -> we transpose.
    """
    # Arboreto uses joblib; cap native threads to avoid oversubscription
    n_workers = n_workers or os.cpu_count() or 4
    threadpool_limits(n_workers)

    print(f"Spinning up LocalCluster with {n_workers} workers...")
    cluster = LocalCluster(
        n_workers=n_workers, threads_per_worker=1, processes=True, silence_logs=False
    )
    client = Client(cluster)
    print(f"Dask client: {client}")

    try:
        # samples x genes for arboreto
        samples_x_genes = expr_df.T
        print(
            f"Matrix for GRNBoost2: {samples_x_genes.shape[0]} samples x {samples_x_genes.shape[1]} genes"
        )
        print(f"#TFs provided: {len(tf_names)}")

        # Sanity: TFs must be subset of provided gene set
        tf_names = [t for t in tf_names if t in expr_df.index]
        if len(tf_names) == 0:
            raise ValueError(
                "No TFs overlap with expression genes (after intersection)."
            )

        # Run
        network = grnboost2(
            expression_data=samples_x_genes, tf_names=tf_names, client_or_address=client
        )

        # Standardize column names
        network = network.rename(
            columns={"TF": "regulator", "target": "target", "importance": "importance"}
        )
        network = network[["regulator", "target", "importance"]].sort_values(
            "importance", ascending=False
        )
        print(f"GRN edges produced: {len(network)}")
        return network

    finally:
        try:
            client.close()
            cluster.close()
        except Exception:
            pass


def main():
    # ---FILE PATHS ---
    expr_path = Path("KAN_Implementation/Data/simulated_gene_expression_100.h5ad")
    ref_grn_path = Path("KAN_Implementation/Data/grn_100_truth.tsv")
    out_path = Path("KAN_Implementation/Data/grnboost2_100.tsv")
    workers = 8
    normalize = False  # True or False
    # ---------------------------------

    # 1) Expression (ALL GENES)
    expr_df = read_expression_h5ad(expr_path, normalize=normalize)

    # 2) TF list from reference GRN regulators (∩ expression genes)
    tf_names = read_reference_grn_tfs(ref_grn_path, expr_df.index)

    # 3) Run GRNBoost2
    network = run_grnboost2_allgenes(expr_df, tf_names, n_workers=workers)

    # 4) Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    network.to_csv(out_path, sep="\t", index=False)
    print(f"[IO] Saved network: {out_path}  ({len(network)} edges)")
    print("=" * 70)
    print("DONE.")


if __name__ == "__main__":
    main()
