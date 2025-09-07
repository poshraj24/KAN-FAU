import os
import pandas as pd
import numpy as np
from scipy import stats


def extract_BHLHE40_regulated_genes(grn_file="grnboost2_perturb.csv"):
    """Return list of targets regulated by BHLHE40 from a TSV GRN (TF, Target, Confidence)."""
    grn_df = pd.read_csv(
        grn_file, sep="\t", header=None, names=["TF", "Target", "Confidence"]
    )
    targets = grn_df.loc[grn_df["TF"] == "BHLHE40", "Target"].tolist()
    return sorted(set(targets))


def get_available_genes(perturbed_folder, file_type):
    """Genes that have the required comparison CSV"""
    genes = []
    for item in os.listdir(perturbed_folder):
        gdir = os.path.join(perturbed_folder, item)
        fpath = os.path.join(gdir, file_type)
        if os.path.isdir(gdir) and os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath, nrows=1)
                if {"actual_value", "predicted_value", "sample_index"}.issubset(
                    df.columns
                ):
                    genes.append(item)
            except Exception:
                pass
    return sorted(genes)


def load_gene_samples(perturbed_folder, gene, file_type):
    """Load per-sample rows for one gene and add gene_name."""
    fpath = os.path.join(perturbed_folder, gene, file_type)
    df = pd.read_csv(fpath)
    # keep only necessary columns; coerce to numeric
    df = df[["sample_index", "actual_value", "predicted_value"]].copy()
    df["actual_value"] = pd.to_numeric(df["actual_value"], errors="coerce")
    df["predicted_value"] = pd.to_numeric(df["predicted_value"], errors="coerce")
    df["sample_index"] = pd.to_numeric(
        df["sample_index"], errors="coerce", downcast="integer"
    )
    df = df.dropna(subset=["actual_value", "predicted_value"])
    df.insert(0, "gene_name", gene)
    return df


def build_all_points_csv(genes, perturbed_folder, file_type, dataset_label, out_csv):
    """Concatenate all per-sample rows across genes and save a flat CSV."""
    frames = []
    for i, g in enumerate(genes, 1):
        try:
            frames.append(load_gene_samples(perturbed_folder, g, file_type))
        except Exception as e:
            print(f"  Skipping {g} ({e})")
        if i % 50 == 0:
            print(f"  Processed {i}/{len(genes)} genes...")

    if not frames:
        print(f"No data found for {dataset_label}.")
        return None

    all_df = pd.concat(frames, ignore_index=True)
    all_df["dataset"] = dataset_label

    # Overall Pearson r,p over ALL points (across all genes & samples)
    r, p = stats.pearsonr(
        all_df["actual_value"].values, all_df["predicted_value"].values
    )
    print(f"[{dataset_label}] Overall Pearson r = {r:.4f}, p = {p:.3e}")
    print(
        f"[{dataset_label}] Rows = {len(all_df):,} across {all_df['gene_name'].nunique()} genes"
    )

    all_df.to_csv(out_csv, index=False)
    print(f"[{dataset_label}] Saved: {out_csv}")
    return all_df


def main():
    # --- Config---
    grn_file = "grnboost2_perturb.csv"
    perturbed_folder = "KAN_BHLHE40_Perturbed"

    # 1) Get regulated genes
    if not os.path.exists(grn_file):
        raise FileNotFoundError(f"GRN file not found: {grn_file}")
    genes_BHLHE40 = extract_BHLHE40_regulated_genes(grn_file)
    if not genes_BHLHE40:
        raise RuntimeError("No BHLHE40-regulated targets found in GRN.")

    # 2) Build per-dataset all-points CSVs (validation & test)
    if not os.path.exists(perturbed_folder):
        raise FileNotFoundError(f"Perturbed folder not found: {perturbed_folder}")

    for file_type, label in [
        ("validation_comparison.csv", "validation"),
        ("test_comparison.csv", "test"),
    ]:
        print("\n" + "=" * 70)
        print(f"Building ALL-POINTS table for: {label.upper()}")
        avail = get_available_genes(perturbed_folder, file_type)
        target_genes = sorted(set(genes_BHLHE40).intersection(avail))
        print(f"  Target genes found: {len(target_genes)} / {len(genes_BHLHE40)}")

        out_csv = f"BHLHE40_{label}_all_points.csv"
        build_all_points_csv(
            genes=target_genes,
            perturbed_folder=perturbed_folder,
            file_type=file_type,
            dataset_label=label,
            out_csv=out_csv,
        )

    print("Files created successfully")


if __name__ == "__main__":
    main()
