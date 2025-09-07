##########For Mean Gene Expression
import os
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings("ignore")


def extract_BHLHE40_regulated_genes(grn_file="grnboost2_perturb.csv"):
    """
    Extract list of genes that are regulated by BHLHE40 from the GRN file.

    """
    try:
        print(f"Loading BHLHE40 regulated genes from GRN file: {grn_file}")

        # Read the tab-separated GRN file
        grn_df = pd.read_csv(
            grn_file, sep="\t", header=None, names=["TF", "Target", "Confidence"]
        )

        print(f" Loaded GRN data: {len(grn_df)} TF-target relationships")

        # Filter for BHLHE40 as transcription factor
        BHLHE40_targets = grn_df[grn_df["TF"] == "BHLHE40"]

        if len(BHLHE40_targets) == 0:
            print(f" No targets found for BHLHE40 in GRN file")
            return None

        print(f" Found {len(BHLHE40_targets)} genes regulated by BHLHE40")

        # Extract target gene names
        regulated_genes = BHLHE40_targets["Target"].tolist()

        print(f" BHLHE40-regulated genes: {len(regulated_genes)}")
        print(f"First 10 BHLHE40-regulated genes: {regulated_genes[:10]}")

        return regulated_genes

    except Exception as e:
        print(f"Error loading BHLHE40 regulated genes from GRN file: {e}")
        return None


def load_test_sample_data(gene_folder, file_type):
    """
    Load test/validation sample data from CSV file to determine which samples to use.

    Args:
        gene_folder (str): Path to the gene folder
        file_type (str): Either "test_comparison.csv" or "validation_comparison.csv"

    Returns:
        pandas.DataFrame or None: DataFrame with sample_index and predicted_value
    """
    csv_file = os.path.join(gene_folder, file_type)

    if not os.path.exists(csv_file):
        return None

    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Check required columns
        if not all(col in df.columns for col in ["sample_index", "predicted_value"]):
            print(f" Missing required columns in {csv_file}")
            return None

        df = df.sort_values("sample_index").reset_index(drop=True)

        return df[["sample_index", "predicted_value"]]

    except Exception as e:
        print(f" Error processing {csv_file}: {str(e)}")
        return None


def consolidate_test_samples(base_folder, file_type, target_genes=None):
    """
    Consolidate test/validation sample information from specified target genes only.

    Args:
        base_folder (str): Path to the base folder (perturbed predictions)
        file_type (str): Either "test_comparison.csv" or "validation_comparison.csv"
        target_genes (list): List of target genes to process (BHLHE40-regulated genes)

    Returns:
        dict: Dictionary with gene names as keys and sample DataFrames as values
    """
    gene_test_data = {}

    if not os.path.exists(base_folder):
        print(f"Warning: Folder '{base_folder}' not found!")
        return gene_test_data

    # Use target genes if provided
    if target_genes is not None:
        gene_folders = target_genes
        print(
            f"Reading test sample indices from {len(gene_folders)} BHLHE40-regulated genes for {file_type}"
        )
    else:
        gene_folders = [
            f
            for f in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, f))
        ]
        print(
            f"Reading test sample indices from {len(gene_folders)} genes for {file_type}"
        )

    successful_genes = 0
    for gene_name in gene_folders:
        gene_path = os.path.join(base_folder, gene_name)

        # Check if gene folder exists
        if not os.path.exists(gene_path):
            print(f" Gene folder not found: {gene_name}")
            continue

        gene_folders = [
            f
            for f in os.listdir(gene_path)
            if os.path.isdir(os.path.join(gene_path, f))
        ]
        print(
            f"Reading test sample indices from {len(gene_folders)} genes for {file_type}"
        )

        successful_genes = 0
        for gene_name in gene_folders:
            gene_path = os.path.join(gene_path, gene_name)

            # Check if gene folder exists
            if not os.path.exists(gene_path):
                print(f" Gene folder not found: {gene_name}")
                continue

            test_df = load_test_sample_data(gene_path, file_type)

        if test_df is not None and len(test_df) > 0:
            gene_test_data[gene_name] = test_df
            successful_genes += 1
            print(
                f" Loaded test samples for gene: {gene_name} ({len(test_df)} samples)"
            )
        else:
            print(f" Failed to load test samples for gene: {gene_name}")

    print(
        f" Successfully loaded test sample info for {successful_genes}/{len(gene_folders)} genes"
    )
    return gene_test_data


def load_ground_truth_data(h5ad_file):
    """
    Load ground truth perturbed data from h5ad file.
    This contains real experimental gene expression when BHLHE40 was knocked out.

    Args:
        h5ad_file (str): Path to the h5ad file containing actual BHLHE40 knockout data

    Returns:
        dict: Dictionary with gene names as keys and expression arrays as values
    """
    try:
        print(f"Loading ground truth BHLHE40 knockout data from: {h5ad_file}")
        adata = sc.read_h5ad(h5ad_file)
        print(f"Ground truth data shape: {adata.shape}")
        print(f"Available genes: {adata.n_vars}")
        print(f"Available samples (cells): {adata.n_obs}")

        ground_truth_data = {}
        gene_names = adata.var_names.tolist()

        for i, gene_name in enumerate(gene_names):

            gene_expr = adata.X[:, i]

            # Handle sparse matrices
            if hasattr(gene_expr, "toarray"):
                gene_expr = gene_expr.toarray().flatten()
            else:
                gene_expr = gene_expr.flatten()

            ground_truth_data[gene_name] = gene_expr

        print(f" Loaded ground truth knockout data for {len(ground_truth_data)} genes")
        return ground_truth_data

    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        return None


def calculate_correlation_and_significance(ground_truth_values, model_predictions):
    """
    Calculate Pearson correlation between ground truth knockout data and model predictions.

    Returns:
        tuple: (pearson_corr, pearson_p)
    """
    try:
        # Remove NaN values
        valid_mask = ~(np.isnan(ground_truth_values) | np.isnan(model_predictions))
        gt_clean = ground_truth_values[valid_mask]
        pred_clean = model_predictions[valid_mask]

        if len(gt_clean) < 3:
            return 0.0, 1.0

        # Pearson correlation
        if np.std(gt_clean) > 1e-10 and np.std(pred_clean) > 1e-10:
            pearson_corr, pearson_p = stats.pearsonr(gt_clean, pred_clean)
        else:
            pearson_corr, pearson_p = 0.0, 1.0

        # Handle NaN results
        if np.isnan(pearson_corr) or np.isnan(pearson_p):
            pearson_corr, pearson_p = 0.0, 1.0

        return float(pearson_corr), float(pearson_p)

    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return 0.0, 1.0


def perform_perturbation_analysis(
    ground_truth_h5ad,
    perturbed_folder,
    regulator_name,
    BHLHE40_regulated_genes,
    file_type="test_comparison.csv",
):
    """
    Perform perturbation analysis with correct sample alignment for BHLHE40-regulated genes only.

    Args:
        ground_truth_h5ad (str): Path to h5ad file with real BHLHE40 knockout data
        perturbed_folder (str): Path to folder with model predictions
        regulator_name (str): Name of the perturbed regulator (e.g., "BHLHE40")
        BHLHE40_regulated_genes (list): List of BHLHE40-regulated genes to analyze
        file_type (str): Either "test_comparison.csv" or "validation_comparison.csv"

    Returns:
        pandas.DataFrame: Results dataframe with one row per gene
    """

    print(f"\n{'='*80}")
    print(f"PERTURBATION ANALYSIS - {file_type.upper().replace('.CSV', '')} DATA")
    print(f"Regulator: {regulator_name}")
    print(
        f"Filtering for BHLHE40-regulated genes only: {len(BHLHE40_regulated_genes)} genes"
    )  # NEW
    print(f"{'='*80}")
    print(f"Strategy: Use test/validation sample indices to align data")

    # Step 1: Load test/validation sample information for BHLHE40-regulated genes only
    print(
        f"\n Loading test/validation sample indices from CSV files (BHLHE40-regulated genes only)"
    )
    gene_test_data = consolidate_test_samples(
        perturbed_folder, file_type, BHLHE40_regulated_genes
    )

    if not gene_test_data:
        print("Error: No valid test sample data found for BHLHE40-regulated genes")
        return None

    # Step 2: Load ground truth knockout data
    print(f"\nLoading ground truth {regulator_name} knockout data")
    ground_truth_data = load_ground_truth_data(ground_truth_h5ad)

    if ground_truth_data is None:
        print("Error: Failed to load ground truth knockout data")
        return None

    # Step 3: Find intersection between BHLHE40-regulated genes, test data, and ground truth
    BHLHE40_set = set(BHLHE40_regulated_genes)
    test_genes = set(gene_test_data.keys())
    ground_truth_genes = set(ground_truth_data.keys())

    # Triple intersection: BHLHE40-regulated AND has test data AND has ground truth
    common_genes = list(BHLHE40_set & test_genes & ground_truth_genes)
    common_genes.sort()

    print(f"BHLHE40-regulated genes: {len(BHLHE40_set)}")
    print(f"Test sample genes: {len(test_genes)}")
    print(f"Ground truth genes: {len(ground_truth_genes)}")
    print(f"BHLHE40-regulated genes with test data: {len(BHLHE40_set & test_genes)}")
    print(
        f"BHLHE40-regulated genes with ground truth: {len(BHLHE40_set & ground_truth_genes)}"
    )
    print(f"Final common genes (BHLHE40 + test + ground truth): {len(common_genes)}")

    if len(common_genes) == 0:
        print("ERROR: No common BHLHE40-regulated genes found across all datasets!")
        return None

    # Step 4: Process each gene with sample alignment
    print(f"\nProcessing {len(common_genes)} common BHLHE40-regulated genes")

    all_results = []
    gene_level_stats = []

    for i, gene_name in enumerate(common_genes):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"Processing gene {i + 1}/{len(common_genes)}: {gene_name}")

        try:
            # Get test sample information for this gene
            test_df = gene_test_data[gene_name]
            sample_indices = test_df["sample_index"].values
            model_predictions = test_df["predicted_value"].values

            gt_full_expression = ground_truth_data[gene_name]

            max_gt_index = len(gt_full_expression) - 1
            valid_mask = sample_indices <= max_gt_index

            if not np.all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                print(
                    f"Warning: {gene_name} has {invalid_count} invalid sample indices"
                )

                sample_indices = sample_indices[valid_mask]
                model_predictions = model_predictions[valid_mask]

            if len(sample_indices) < 3:
                print(
                    f"Warning: Too few valid samples for {gene_name} (only {len(sample_indices)} samples)"
                )
                continue

            # Extract ground truth values for the specific test/validation sample indices
            ground_truth_values = gt_full_expression[sample_indices]

            # Calculate correlation between ground truth and model predictions
            # for the same test/validation samples
            pearson_corr, pearson_p = calculate_correlation_and_significance(
                ground_truth_values, model_predictions
            )

            # Store for multiple testing correction
            gene_level_stats.append(
                {
                    "gene_name": gene_name,
                    "pearson_p": pearson_p,
                    "pearson_corr": pearson_corr,
                }
            )

            gt_mean = float(np.mean(ground_truth_values))
            pred_mean = float(np.mean(model_predictions))

            # Calculate log2 fold change (predicted vs ground truth)
            try:
                pseudocount = 1e-6
                pred_mean_adj = pred_mean + pseudocount
                gt_mean_adj = gt_mean + pseudocount
                log2_fold_change = np.log2(pred_mean_adj / gt_mean_adj)
            except Exception as e:
                print(f"Error calculating log2 fold change for {gene_name}: {e}")
                log2_fold_change = 0.0

            # Store result
            result = {
                "gene_name": gene_name,
                "n_samples": len(sample_indices),
                "actual_value": round(
                    float(gt_mean), 4
                ),  # Ground truth mean (test samples)
                "predicted_value": round(
                    float(pred_mean), 4
                ),  # Model prediction mean (test samples)
                "correlation": round(float(pearson_corr), 4),
                "p_value": round(float(pearson_p), 4),
                "log2_fold_change": round(float(log2_fold_change), 4),
            }
            all_results.append(result)

        except Exception as e:
            print(f"Error processing {gene_name}: {e}")
            continue

    if not all_results:
        print("No valid results generated")
        return None

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    gene_stats_df = pd.DataFrame(gene_level_stats)

    print(f"\nApplying multiple testing correction...")

    # Apply Benjamini-Hochberg correction
    if len(gene_stats_df) > 0:
        _, pearson_p_adj, _, _ = multipletests(
            gene_stats_df["pearson_p"], alpha=0.05, method="fdr_bh"
        )

        # Add adjusted p-values to results
        pearson_adj_map = dict(zip(gene_stats_df["gene_name"], pearson_p_adj))
        results_df["p_value_adjusted"] = (
            results_df["gene_name"].map(pearson_adj_map).round(4)
        )

        # Determine significance
        results_df["significant"] = (results_df["p_value_adjusted"] < 0.05).map(
            {True: "Yes", False: "No"}
        )

    return results_df


def main():
    """
    Main function to run both test and validation analyses for BHLHE40-regulated genes only
    """

    # Configuration
    ground_truth_h5ad = (
        "Data/BHLHE40_perturbed_only.h5ad"  # Real experimental BHLHE40 knockout data
    )
    perturbed_folder = (
        "KAN_BHLHE40_Perturbed"  # Model predictions with test/validation splits
    )
    regulator_name = "BHLHE40"

    # Check if files/folders exist
    if not os.path.exists("grnboost2_perturb.csv"):
        print(f"ERROR: GRN file 'grnboost2_perturb.csv' not found!")
        return

    if not os.path.exists(ground_truth_h5ad):
        print(f"ERROR: Ground truth file '{ground_truth_h5ad}' not found!")
        return

    if not os.path.exists(perturbed_folder):
        print(f"ERROR: Model predictions folder '{perturbed_folder}' not found!")
        return

    # Step 1: Extract BHLHE40-regulated genes from GRN file
    print(f"\n{'='*60}")
    print("EXTRACTING BHLHE40-REGULATED GENES")
    print("=" * 60)

    BHLHE40_regulated_genes = extract_BHLHE40_regulated_genes("grnboost2_perturb.csv")

    if BHLHE40_regulated_genes is None or len(BHLHE40_regulated_genes) == 0:
        print("ERROR: Could not extract BHLHE40-regulated genes from GRN file!")
        return

    # Process test data
    print(f"\n{'='*60}")
    print("PROCESSING TEST DATA - BHLHE40-REGULATED GENES ONLY")
    print("=" * 60)

    test_results = perform_perturbation_analysis(
        ground_truth_h5ad=ground_truth_h5ad,
        perturbed_folder=perturbed_folder,
        regulator_name=regulator_name,
        BHLHE40_regulated_genes=BHLHE40_regulated_genes,
        file_type="test_comparison.csv",
    )

    if test_results is not None:
        test_output_file = (
            f"perturbation_model_evaluation_test_{regulator_name}_filtered.csv"
        )
        test_results.to_csv(test_output_file, index=False)
        print(f"\n Test results saved to: {test_output_file}")
        print(f" Test data shape: {test_results.shape}")

    # Process validation data
    print(f"\n{'='*60}")
    print("PROCESSING VALIDATION DATA - BHLHE40-REGULATED GENES ONLY")
    print("=" * 60)

    validation_results = perform_perturbation_analysis(
        ground_truth_h5ad=ground_truth_h5ad,
        perturbed_folder=perturbed_folder,
        regulator_name=regulator_name,
        BHLHE40_regulated_genes=BHLHE40_regulated_genes,
        file_type="validation_comparison.csv",
    )

    if validation_results is not None:
        validation_output_file = f"perturbation_model_evaluation_validation_{regulator_name}_filtered.csv"  # NEW FILENAME
        validation_results.to_csv(validation_output_file, index=False)
        print(f"\n Validation results saved to: {validation_output_file}")
        print(f" Validation data shape: {validation_results.shape}")


if __name__ == "__main__":
    main()

##For Overall Gene Expression
