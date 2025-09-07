import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# Load ground truth GRN
try:
    truth_df = pd.read_csv("KAN_Implementation/Data/grn_1139_truth.tsv", sep="\t")
    print(f"Ground truth file loaded successfully. Shape: {truth_df.shape}")
    print(f"Ground truth columns: {list(truth_df.columns)}")
    print("First few rows of ground truth:")
    print(truth_df.head())

    if truth_df.shape[1] < 2:
        raise ValueError(
            f"Ground truth file needs at least 2 columns, found {truth_df.shape[1]}"
        )

    truth_edges = set(zip(truth_df.iloc[:, 0], truth_df.iloc[:, 1]))

except FileNotFoundError:
    print("Error: Ground truth file not found")

    exit()
except Exception as e:
    print(f"Error loading ground truth file: {e}")
    print("Trying different separators...")

    # Try different separators
    for sep in [",", " ", "|"]:
        try:
            truth_df = pd.read_csv(
                "KAN_Implementation/Data/grn_1139_truth.tsv", sep=sep
            )
            if truth_df.shape[1] >= 2:
                print(
                    f"Successfully loaded with separator '{sep}'. Shape: {truth_df.shape}"
                )
                truth_edges = set(zip(truth_df.iloc[:, 0], truth_df.iloc[:, 1]))
                break
        except:
            continue
    else:
        print("Could not load ground truth file with any separator")
        exit()

# Load predicted GRN (TSV format)
try:
    pred_df = pd.read_csv(
        "KAN_Implementation/Data/kan_1139_filtered_zscore_3.0_equalized.csv",
        sep=",",
        header=None,
    )
    print(f"\nPredicted file loaded successfully. Shape: {pred_df.shape}")
    print(f"Predicted columns: {list(pred_df.columns)}")
    print("First few rows of predicted:")
    print(pred_df.head())

except FileNotFoundError:
    print("Error: Predicted file  not found")
    print("Please check the file path and name")
    exit()
except Exception as e:
    print(f"Error loading predicted file: {e}")
    exit()

print(f"Ground truth columns: {list(truth_df.columns)}")
print(f"Predicted columns: {list(pred_df.columns)}")

# Use proper column names if available, otherwise use positional indexing
if "regulator" in truth_df.columns and "target" in truth_df.columns:
    truth_edges = set(zip(truth_df["regulator"], truth_df["target"]))
    print("Using named columns for ground truth")
elif truth_df.shape[1] >= 2:
    truth_edges = set(zip(truth_df.iloc[:, 0], truth_df.iloc[:, 1]))
    print("Using positional columns for ground truth")
else:
    print(f"Error: Ground truth file has insufficient columns: {truth_df.shape[1]}")
    exit()

# Use the actual column names from predicted file
if "regulator" in pred_df.columns and "target" in pred_df.columns:
    pred_edges = set(zip(pred_df["regulator"], pred_df["target"]))
    print("Using named columns for predictions")
else:
    print("Warning: 'regulator' and 'target' columns not found in predictions")
    if pred_df.shape[1] >= 2:
        pred_edges = set(zip(pred_df.iloc[:, 0], pred_df.iloc[:, 1]))
        print("Using first two columns for predictions")
    else:
        print(f"Error: Predicted file has insufficient columns: {pred_df.shape[1]}")
        exit()

print(f"Ground truth edges: {len(truth_edges)}")
print(f"Predicted edges: {len(pred_edges)}")

# Compute evaluation metrics
tp = truth_edges & pred_edges
fp = pred_edges - truth_edges
fn = truth_edges - pred_edges

print(f"True Positives: {len(tp)}")
print(f"False Positives: {len(fp)}")
print(f"False Negatives: {len(fn)}")

precision = len(tp) / (len(tp) + len(fp)) if len(pred_edges) > 0 else 0
recall = len(tp) / (len(tp) + len(fn)) if len(truth_edges) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
jaccard = (
    len(tp) / len(truth_edges | pred_edges) if len(truth_edges | pred_edges) > 0 else 0
)

print(f"\nEVALUATION RESULTS:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Jaccard Index: {jaccard:.4f}")

print("=== DIAGNOSTIC INFO ===")
print(f"Truth file shape: {truth_df.shape}")
print(f"Pred file shape: {pred_df.shape}")
print(f"First few truth edges: {list(truth_edges)[:3]}")
print(f"First few pred edges: {list(pred_edges)[:3]}")

# Check individual gene overlaps
if "regulator" in truth_df.columns and "target" in truth_df.columns:
    truth_genes = set(truth_df["regulator"]) | set(truth_df["target"])
else:
    truth_genes = set(truth_df.iloc[:, 0]) | set(truth_df.iloc[:, 1])

if "regulator" in pred_df.columns and "target" in pred_df.columns:
    pred_genes = set(pred_df["regulator"]) | set(pred_df["target"])
else:
    pred_genes = set(pred_df.iloc[:, 0]) | set(pred_df.iloc[:, 1])

gene_overlap = truth_genes & pred_genes

print(f"Unique genes in ground truth: {len(truth_genes)}")
print(f"Unique genes in predictions: {len(pred_genes)}")
print(f"Gene name overlap: {len(gene_overlap)}")
print(f"Sample truth genes: {list(truth_genes)[:5]}")
print(f"Sample pred genes: {list(pred_genes)[:5]}")

# Check if edges might be reversed
reversed_pred_edges = set((b, a) for a, b in pred_edges)
tp_reversed = truth_edges & reversed_pred_edges
print(f"True positives with reversed edges: {len(tp_reversed)}")

# Additional debugging: check for case sensitivity or whitespace issues
if len(gene_overlap) == 0:
    print("\n=== CHECKING FOR FORMATTING ISSUES ===")
    # Check case sensitivity
    if "regulator" in truth_df.columns and "target" in truth_df.columns:
        truth_genes_upper = set(
            str(g).upper().strip() for g in truth_df["regulator"]
        ) | set(str(g).upper().strip() for g in truth_df["target"])
    else:
        truth_genes_upper = set(
            str(g).upper().strip() for g in truth_df.iloc[:, 0]
        ) | set(str(g).upper().strip() for g in truth_df.iloc[:, 1])

    if "regulator" in pred_df.columns and "target" in pred_df.columns:
        pred_genes_upper = set(
            str(g).upper().strip() for g in pred_df["regulator"]
        ) | set(str(g).upper().strip() for g in pred_df["target"])
    else:
        pred_genes_upper = set(
            str(g).upper().strip() for g in pred_df.iloc[:, 0]
        ) | set(str(g).upper().strip() for g in pred_df.iloc[:, 1])

    case_overlap = truth_genes_upper & pred_genes_upper
    print(f"Gene overlap after case normalization: {len(case_overlap)}")
