import os
import warnings
import json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# -------------------- Loaders --------------------
def extract_BHLHE40_regulated_genes(grn_file="grnboost2_perturb.csv"):
    """Return list of targets regulated by BHLHE40 from a TSV GRN (TF, Target, Confidence)."""
    if not os.path.exists(grn_file):
        raise FileNotFoundError(f"GRN file not found: {grn_file}")
    grn_df = pd.read_csv(
        grn_file, sep="\t", header=None, names=["TF", "Target", "Confidence"]
    )
    targets = grn_df.loc[grn_df["TF"] == "BHLHE40", "Target"].astype(str).tolist()
    targets = sorted(set(targets))
    print(f"[GRN] BHLHE40 targets: {len(targets)} genes")
    return targets


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

        df = pd.read_csv(csv_file)

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

        # Load test sample data
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


def load_all_points_from_ground_truth(
    ground_truth_h5ad, perturbed_folder, file_type, expected_label
):
    """
    Load all-points data by combining ground truth h5ad data with model predictions.

    Args:
        ground_truth_h5ad (str): Path to h5ad file with real knockout data
        perturbed_folder (str): Path to folder with model predictions
        file_type (str): Either "test_comparison.csv" or "validation_comparison.csv"
        expected_label (str): Dataset label (e.g., "test" or "validation")

    Returns:
        pd.DataFrame: DataFrame with columns: gene_name, sample_index, actual_value, predicted_value, dataset
    """
    print(
        f"Loading all-points data for {expected_label} from ground truth and predictions..."
    )

    # Load ground truth data
    ground_truth_data = load_ground_truth_data(ground_truth_h5ad)
    if ground_truth_data is None:
        raise RuntimeError("Failed to load ground truth data")

    # Get all gene folders
    gene_folders = [
        f
        for f in os.listdir(perturbed_folder)
        if os.path.isdir(os.path.join(perturbed_folder, f))
    ]

    gene_test_data = consolidate_test_samples(perturbed_folder, file_type, gene_folders)

    if not gene_test_data:
        raise RuntimeError(f"No valid test sample data found for {file_type}")

    # Find intersection between test data and ground truth
    test_genes = set(gene_test_data.keys())
    ground_truth_genes = set(ground_truth_data.keys())
    common_genes = list(test_genes & ground_truth_genes)

    print(f"Test sample genes: {len(test_genes)}")
    print(f"Ground truth genes: {len(ground_truth_genes)}")
    print(f"Common genes: {len(common_genes)}")

    if not common_genes:
        raise RuntimeError("No common genes found between test data and ground truth")

    all_records = []

    for gene_name in common_genes:
        try:

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

            if len(sample_indices) < 1:
                print(f"Warning: No valid samples for {gene_name}")
                continue

            # Extract ground truth values for the specific test/validation sample indices
            ground_truth_values = gt_full_expression[sample_indices]

            # Create records for each sample
            for i in range(len(sample_indices)):
                record = {
                    "gene_name": gene_name,
                    "sample_index": int(sample_indices[i]),
                    "actual_value": float(ground_truth_values[i]),
                    "predicted_value": float(model_predictions[i]),
                    "dataset": expected_label,
                }
                all_records.append(record)

        except Exception as e:
            print(f"Error processing {gene_name}: {e}")
            continue

    if not all_records:
        raise RuntimeError("No valid records generated")

    df = pd.DataFrame(all_records)

    # Remove any NaN values
    df = df.dropna(subset=["actual_value", "predicted_value"])

    print(
        f"Generated all-points dataframe: {len(df):,} rows, {df['gene_name'].nunique()} genes"
    )
    return df


# -------------------- Math helpers --------------------
def pearson_safe(x, y):
    """Compute Pearson r & p safely (handle <3 points or zero variance)."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.isclose(x.std(ddof=1), 0) or np.isclose(y.std(ddof=1), 0):
        return 0.0, 1.0
    r, p = stats.pearsonr(x, y)
    if np.isnan(r) or np.isnan(p):
        return 0.0, 1.0
    return float(r), float(p)


def pointwise_log2fc(pred_vals, act_vals, pseudocount=1e-6):
    """
    Compute per-sample log2FC = log2((pred + pc) / (actual + pc)).
    Returns an array with NaN where the ratio is invalid (<=0).
    """
    pred = np.asarray(pred_vals, dtype=float) + pseudocount
    act = np.asarray(act_vals, dtype=float) + pseudocount
    # Invalid where non-positive after adjustment
    valid = (pred > 0) & (act > 0)
    l2fc = np.full_like(pred, fill_value=np.nan, dtype=float)
    l2fc[valid] = np.log2(pred[valid] / act[valid])
    return l2fc


def _round_list(xs, ndigits=6):
    return [
        (
            None
            if (v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))))
            else round(float(v), ndigits)
        )
        for v in xs
    ]


# -------------------- Core summarizer --------------------
def summarize_all_points(
    df_all: pd.DataFrame, regulated_genes: set, regulator_name: str
) -> pd.DataFrame:
    """
    From an all-points dataframe, compute per-gene summary over ALL samples.
    Output columns:
      gene_name, n_samples,
      sample_indices (JSON), actual_values (JSON), predicted_values (JSON),
      log2_fold_changes (JSON)  <-- per-sample,
      correlation, p_value, p_value_adjusted, significant
    """
    present = set(df_all["gene_name"].unique())
    target_genes = sorted(regulated_genes & present)
    if not target_genes:
        raise RuntimeError(
            f"No {regulator_name}-regulated genes present in the all-points file."
        )

    print(
        f"[FILTER] Using {len(target_genes)} {regulator_name}-regulated genes present in this dataset."
    )

    records = []

    for i, g in enumerate(target_genes, 1):
        sub = df_all.loc[
            df_all["gene_name"] == g,
            ["sample_index", "actual_value", "predicted_value"],
        ].dropna()
        sub = sub.sort_values("sample_index")  # stable ordering
        n = len(sub)
        if n < 3:
            continue

        sample_idx = sub["sample_index"].astype(int).tolist()
        actual_vals = sub["actual_value"].to_numpy()
        predicted_vals = sub["predicted_value"].to_numpy()

        # Correlation across all samples for this gene
        r, p = pearson_safe(actual_vals, predicted_vals)

        # Point-wise log2 fold changes
        l2fc = pointwise_log2fc(predicted_vals, actual_vals, pseudocount=1e-6)

        rec = {
            "gene_name": g,
            "n_samples": int(n),
            "sample_indices": json.dumps(sample_idx, separators=(",", ":")),
            "actual_values": json.dumps(
                _round_list(actual_vals), separators=(",", ":")
            ),
            "predicted_values": json.dumps(
                _round_list(predicted_vals), separators=(",", ":")
            ),
            "log2_fold_changes": json.dumps(_round_list(l2fc), separators=(",", ":")),
            "correlation": round(float(r), 4),
            "p_value": round(float(p), 4),
        }
        records.append(rec)

        if i % 200 == 0:
            print(f"processed {i}/{len(target_genes)} genes...")

    if not records:
        raise RuntimeError("No genes had at least 3 samples to compute statistics.")

    res = pd.DataFrame.from_records(records)

    # BH adjust p-values across genes
    _, p_adj, _, _ = multipletests(res["p_value"].values, alpha=0.05, method="fdr_bh")
    res["p_value_adjusted"] = np.round(p_adj, 4)
    res["significant"] = (res["p_value_adjusted"] < 0.05).map(
        {True: "Yes", False: "No"}
    )

    # Sort by |correlation| desc
    res = res.iloc[
        res["correlation"].abs().sort_values(ascending=False).index
    ].reset_index(drop=True)

    # Console summary
    print(f"[SUMMARY] {regulator_name}: {len(res)} genes summarized")
    print(f"  Significant (BH q<0.05): {(res['significant']=='Yes').sum()} genes")
    print(
        f"  Mean r: {res['correlation'].mean():.4f} | Median r: {res['correlation'].median():.4f}"
    )
    return res


# -------------------- Runner --------------------
def run_from_ground_truth(
    ground_truth_h5ad="Data/BHLHE40_perturbed_only.h5ad",
    perturbed_folder="KAN_BHLHE40_Perturbed",
    grn_file="grnboost2_perturb.csv",
    regulator_name="BHLHE40",
):
    """
    Run perturbation analysis using ground truth h5ad data and model predictions.

    Args:
        ground_truth_h5ad (str): Path to h5ad file with real knockout data
        perturbed_folder (str): Path to folder with model predictions
        grn_file (str): Path to GRN file
        regulator_name (str): Name of the regulator (e.g., "BHLHE40")
    """
    print("=" * 80)
    print(f"PERTURBATION MODEL EVALUATION - {regulator_name}-REGULATED GENES")
    print("=" * 80)
    print(f"Ground truth file: {ground_truth_h5ad}")
    print(f"Model predictions folder: {perturbed_folder}")
    print(f"Regulator: {regulator_name}")
    # Check if files/folders exist
    if not os.path.exists(grn_file):
        raise FileNotFoundError(f"GRN file not found: {grn_file}")
    if not os.path.exists(ground_truth_h5ad):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_h5ad}")
    if not os.path.exists(perturbed_folder):
        raise FileNotFoundError(
            f"Model predictions folder not found: {perturbed_folder}"
        )

    # Extract regulated genes
    regulated = set(extract_BHLHE40_regulated_genes(grn_file))
    if not regulated:
        raise RuntimeError(f"No {regulator_name}-regulated genes found in GRN file")

    print("\n" + "=" * 60 + "\nVALIDATION (all points)\n" + "=" * 60)
    df_val = load_all_points_from_ground_truth(
        ground_truth_h5ad, perturbed_folder, "validation_comparison.csv", "validation"
    )
    val_res = summarize_all_points(df_val, regulated, regulator_name)
    val_out = f"perturbation_model_evaluation_validation_{regulator_name}_filtered.csv"
    val_res.to_csv(val_out, index=False)
    print(f"[WRITE] {val_out}  (rows: {len(val_res)})")

    print("\n" + "=" * 60 + "\nTEST (all points)\n" + "=" * 60)
    df_test = load_all_points_from_ground_truth(
        ground_truth_h5ad, perturbed_folder, "test_comparison.csv", "test"
    )
    test_res = summarize_all_points(df_test, regulated, regulator_name)
    test_out = f"perturbation_model_evaluation_test_{regulator_name}_filtered.csv"
    test_res.to_csv(test_out, index=False)
    print(f"[WRITE] {test_out}  (rows: {len(test_res)})")


if __name__ == "__main__":
    run_from_ground_truth(
        ground_truth_h5ad="Data/BHLHE40_perturbed_only.h5ad",
        perturbed_folder="KAN_BHLHE40_Perturbed",
        grn_file="grnboost2_perturb.csv",
        regulator_name="BHLHE40",
    )
