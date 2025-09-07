import scanpy as sc
import numpy as np
import os


input_file = "Data/ctrl_only.h5ad"  # control-only data
output_file = "Data/ctrl_only_CREB1_zero.h5ad"  # Output file with knocked out

# Step 2: Load the control-only dataset
print(f"Loading dataset from: {input_file}")
adata = sc.read_h5ad(input_file)

print(f"Loaded AnnData: shape = {adata.shape}")
print(f"Total genes: {adata.n_vars}")
print(f"Total cells: {adata.n_obs}")

# Step 3: Check if CREB1 exists in the dataset
target_gene = "CREB1"
print(f"\nLooking for gene: {target_gene}")

if target_gene in adata.var_names:
    print(f" Found {target_gene} in the dataset")

    # Get the index of CREB1
    CREB1_idx = adata.var_names.get_loc(target_gene)
    print(f"Gene index: {CREB1_idx}")

    # Show original expression statistics for CREB1
    original_expression = adata.X[:, CREB1_idx]
    if hasattr(original_expression, "toarray"):
        original_expression = original_expression.toarray().flatten()
    else:
        original_expression = original_expression.flatten()

    print(f"\nOriginal {target_gene} expression statistics:")
    print(f"Mean: {np.mean(original_expression):.4f}")
    print(f"Std: {np.std(original_expression):.4f}")
    print(f"Min: {np.min(original_expression):.4f}")
    print(f"Max: {np.max(original_expression):.4f}")
    print(
        f"Non-zero cells: {np.count_nonzero(original_expression)}/{len(original_expression)}"
    )

    # Step 4: Create a copy and perform knockout
    knockout_adata = adata.copy()

    # Set CREB1 expression to 0 for all cells
    print(f"\nPerforming in silico knockout of {target_gene}...")

    if hasattr(knockout_adata.X, "toarray"):  # Sparse matrix
        knockout_adata.X = knockout_adata.X.toarray()  # Convert to dense

    knockout_adata.X[:, CREB1_idx] = 0  # Set all CREB1 values to 0

    # Verify the knockout
    knockout_expression = knockout_adata.X[:, CREB1_idx]
    print(f" Knockout successful - all {target_gene} values set to 0")
    print(f"Verification: max value = {np.max(knockout_expression)}")
    print(f"Verification: sum = {np.sum(knockout_expression)}")

    # Step 5: Add metadata to track the modification
    knockout_adata.uns[f"{target_gene}_knockout"] = {
        "knockout_gene": target_gene,
        "knockout_gene_index": int(CREB1_idx),
        "original_mean_expression": float(np.mean(original_expression)),
        "original_nonzero_cells": int(np.count_nonzero(original_expression)),
        "modification_date": str(np.datetime64("today")),
    }

    # Step 6: Save the knockout dataset
    sc.write(output_file, knockout_adata)
    print(f"\n Saved knockout dataset to: {output_file}")

    # Step 7: Summary
    print(f"\n{'='*50}")
    print("KNOCKOUT SUMMARY:")
    print(f"{'='*50}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Dataset shape: {knockout_adata.shape}")
    print(f"Knockout gene: {target_gene}")
    print(f"Gene index: {CREB1_idx}")
    print(f"Original mean expression: {np.mean(original_expression):.4f}")
    print(f"New expression (all zeros): {np.max(knockout_expression):.4f}")
    print(f"File saved as: {os.path.abspath(output_file)}")

    # Step 8: Verification check
    print(f"\n{'='*50}")
    print("VERIFICATION:")
    print(f"{'='*50}")

    verification_adata = sc.read_h5ad(output_file)
    verification_expression = verification_adata.X[:, CREB1_idx]
    if hasattr(verification_expression, "toarray"):
        verification_expression = verification_expression.toarray().flatten()
    else:
        verification_expression = verification_expression.flatten()

    print(f"Loaded knockout file successfully")
    print(f"{target_gene} max value in saved file: {np.max(verification_expression)}")
    print(f"{target_gene} sum in saved file: {np.sum(verification_expression)}")
    print(
        f"Knockout metadata saved: {'knockout' in str(verification_adata.uns.keys())}"
    )

    if np.all(verification_expression == 0):
        print(f" SUCCESS: {target_gene} knockout verified in saved file!")
    else:
        print(f" ERROR: {target_gene} knockout not properly saved!")

else:
    print(f" ERROR: Gene '{target_gene}' not found in the dataset!")
    print("Available genes (first 20):")
    for i, gene in enumerate(adata.var_names[:20]):
        print(f"  {i+1:2d}. {gene}")

    # Search for similar gene names
    similar_genes = [gene for gene in adata.var_names if "CREB1" in gene.upper()]
    if similar_genes:
        print(f"\nSimilar genes found ({len(similar_genes)} genes with 'CREB1'):")
        for gene in similar_genes[:10]:  # Show first 10
            print(f"  - {gene}")
        if len(similar_genes) > 10:
            print(f"  ... and {len(similar_genes) - 10} more")
    else:
        print("\nNo similar genes found with 'CREB1' in the name")

print(f"\nScript completed!")
