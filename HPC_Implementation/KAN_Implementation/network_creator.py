import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from pathlib import Path


def build_gene_regulatory_network(
    root_dir,
    output_file="KAN_Implementation\gene_regulatory_network.csv",
    filter_method="zscore",
    zscore_threshold=2.0,
    importance_threshold=0.000,
):
    """
    Process feature importance CSV files across multiple gene folders to build a regulatory network.

    Parameters:
    - root_dir: Root directory containing gene folders
    - output_file: Output file path for the network
    - filter_method: "zscore" for z-score filtering, "importance" for importance threshold filtering
    - zscore_threshold: Z-score cutoff threshold (default: 2.0)
    - importance_threshold: Minimum importance score threshold (default: 0.000)


    """

    network_relationships = []

    gene_folders = [
        f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))
    ]
    print(f"Found {len(gene_folders)} gene folders to process")

    if filter_method == "zscore":
        print(f"Using z-score filtering with threshold: {zscore_threshold}")
    else:
        print(
            f"Using importance score filtering with threshold: {importance_threshold}"
        )

    # Process each gene folder
    for regulated_gene in tqdm(gene_folders, desc="Processing gene folders"):
        csv_path = os.path.join(root_dir, regulated_gene, "feature_importance.csv")

        if not os.path.exists(csv_path):
            continue

        try:

            df = pd.read_csv(csv_path)

            # Extract column names
            gene_col = [col for col in df.columns if "gene" in col.lower()][0]
            importance_col = [
                col
                for col in df.columns
                if "importance" in col.lower() or "score" in col.lower()
            ][0]

            if filter_method == "zscore":
                # Calculate z-scores for importance values
                df["z_score"] = (df[importance_col] - df[importance_col].mean()) / df[
                    importance_col
                ].std()
                # Filter by z-score
                filtered_df = df[df["z_score"] > zscore_threshold]
            else:
                # Filter by importance score threshold
                filtered_df = df[df[importance_col] > importance_threshold]

            # Add filtered relationships to our network list
            for _, row in filtered_df.iterrows():
                network_relationships.append(
                    {
                        "regulator_gene": row[gene_col],
                        "regulated_gene": regulated_gene,
                        "importance_score": row[importance_col],
                    }
                )
        except Exception as e:
            print(f"Error processing {regulated_gene}: {str(e)}")

    # Convert relationships to DataFrame and save
    if network_relationships:
        network_df = pd.DataFrame(network_relationships)
        network_df.to_csv(output_file, index=False)
        print(f"Network file created with {len(network_df)} relationships")
        return network_df
    else:
        print("No relationships found meeting the criteria")
        return pd.DataFrame(
            columns=["regulator_gene", "regulated_gene", "importance_score"]
        )


def get_user_choice():
    """
    Get user's choice for filtering method and parameters.
    """
    print("\n" + "=" * 60)
    print("GENE REGULATORY NETWORK BUILDER")
    print("=" * 60)
    print("\nChoose filtering method:")
    print("1. Create whole network (importance score > 0.000)")
    print("2. Create filtered network (z-score cutoff)")

    while True:
        try:
            choice = int(input("\nEnter your choice (1 or 2): "))
            if choice in [1, 2]:
                break
            else:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number (1 or 2).")

    if choice == 1:
        # Option 1: Importance threshold filtering
        filter_method = "importance"
        importance_threshold = 0.000  # Default threshold
        zscore_threshold = None

        # Ask if user wants to modify the importance threshold
        modify = input(
            f"\nUse default importance threshold ({importance_threshold})? (y/n): "
        ).lower()
        if modify == "n":
            while True:
                try:
                    importance_threshold = float(input("Enter importance threshold: "))
                    break
                except ValueError:
                    print("Please enter a valid number.")

        print(
            f"\nSelected: Importance threshold filtering (threshold: {importance_threshold})"
        )
        return filter_method, zscore_threshold, importance_threshold

    else:
        # Option 2: Z-score filtering
        filter_method = "zscore"
        importance_threshold = None

        # Get z-score threshold from user
        while True:
            try:
                zscore_threshold = float(
                    input("\nEnter z-score cutoff threshold (e.g., 2.0): ")
                )
                break
            except ValueError:
                print("Please enter a valid number.")

        print(f"\nSelected: Z-score filtering (threshold: {zscore_threshold})")
        return filter_method, zscore_threshold, importance_threshold


if __name__ == "__main__":
    # Get user preferences
    filter_method, zscore_threshold, importance_threshold = get_user_choice()

    # Root directory containing all gene folders
    ROOT_DIR = "KAN_Implementation\kan_models/"

    # Generate output filename based on filtering method
    if filter_method == "zscore":
        output_file = f"kan_1139_original_filtered_{zscore_threshold}.csv"
    else:
        output_file = f"kan_1139_original_filtered_{importance_threshold}.csv"

    print(f"\nOutput file: {output_file}")
    print("\nStarting network construction...")

    # Run the network builder
    if filter_method == "zscore":
        network_df = build_gene_regulatory_network(
            ROOT_DIR,
            output_file,
            filter_method="zscore",
            zscore_threshold=zscore_threshold,
        )
    else:
        network_df = build_gene_regulatory_network(
            ROOT_DIR,
            output_file,
            filter_method="importance",
            importance_threshold=importance_threshold,
        )

    # Display summary statistics
    if not network_df.empty:
        print("\n" + "=" * 60)
        print("NETWORK SUMMARY")
        print("=" * 60)
        print(f"Total relationships: {len(network_df)}")
        print(f"Unique regulator genes: {network_df['regulator_gene'].nunique()}")
        print(f"Unique regulated genes: {network_df['regulated_gene'].nunique()}")
        print(f"Average importance score: {network_df['importance_score'].mean():.6f}")
        print(
            f"Importance score range: {network_df['importance_score'].min():.6f} - {network_df['importance_score'].max():.6f}"
        )

        # Show top relationships
        print(f"\nTop 10 relationships by importance score:")
        print(
            network_df.nlargest(10, "importance_score")[
                ["regulator_gene", "regulated_gene", "importance_score"]
            ].to_string(index=False)
        )
    else:
        print("\nNo relationships found meeting the specified criteria.")
