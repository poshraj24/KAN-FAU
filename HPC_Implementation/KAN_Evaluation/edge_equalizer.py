import pandas as pd
import numpy as np
from pathlib import Path


def equalize_network_sizes(file1, file2, target_edges=None, output_suffix="_equalized"):
    """
    Equalize the number of edges between two network files by selecting top edges.

    Parameters:
    - file1, file2: Paths to network CSV files
    - target_edges: Target number of edges (if None, uses minimum of both)
    - output_suffix: Suffix to add to output filenames
    """

    def smart_read_csv(filepath):
        """CSV reader that handles different separators and formats"""
        print(f"Attempting to read {filepath}...")

        # Try comma separator first
        try:
            df = pd.read_csv(filepath)
            # Check if we actually got proper columns (more than 1 column)
            if df.shape[1] > 1:
                print(f"Loaded with comma separator: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                return df
            else:
                # Only one column - likely wrong separator
                print(f"Only 1 column with comma separator, trying tab...")
                raise ValueError("Wrong separator")
        except:
            pass

        # Try tab separator
        try:
            df = pd.read_csv(filepath, sep="\t")
            print(f"Loaded with tab separator: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return df
        except:
            pass

        # Try semicolon separator
        try:
            df = pd.read_csv(filepath, sep=";")
            if df.shape[1] > 1:
                print(f"Loaded with semicolon separator: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                return df
        except:
            pass

        # Try space separator
        try:
            df = pd.read_csv(filepath, sep=" ")
            if df.shape[1] > 1:
                print(f"Loaded with space separator: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                return df
        except:
            pass

        # Final attempt: auto-detection
        try:
            df = pd.read_csv(filepath, sep=None, engine="python")
            print(f"Loaded with auto-detection: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Failed to read file: {e}")
            raise

    # Load both networks
    print("Loading networks...")
    print(f"Loading {file1}...")
    net1 = smart_read_csv(file1)
    print(f"Loading {file2}...")
    net2 = smart_read_csv(file2)

    print(f"\nAfter loading:")
    print(f"Network 1 ({file1}): {len(net1)} edges")
    print(f"Network 2 ({file2}): {len(net2)} edges")

    print(f"Network 1 ({file1}): {len(net1)} edges")
    print(f"Network 2 ({file2}): {len(net2)} edges")

    # Determine target edge count
    if target_edges is None:
        target_edges = min(len(net1), len(net2))
        print(f"Using minimum size as target: {target_edges} edges")
    else:
        print(f"Using specified target: {target_edges} edges")

    # Function to get importance column name
    def get_importance_col(df):
        importance_cols = [
            col
            for col in df.columns
            if "importance" in col.lower() or "score" in col.lower()
        ]
        if importance_cols:
            return importance_cols[0]
        else:

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            return numeric_cols[-1] if len(numeric_cols) > 0 else df.columns[-1]

    # Get importance columns
    imp_col1 = get_importance_col(net1)
    imp_col2 = get_importance_col(net2)

    print(f"Using importance columns: '{imp_col1}' and '{imp_col2}'")

    # Select top edges for each network
    net1_top = net1.nlargest(target_edges, imp_col1)
    net2_top = net2.nlargest(target_edges, imp_col2)

    # Generate output filenames
    out_file1 = file1.replace(".csv", f"{output_suffix}.csv")
    out_file2 = file2.replace(".csv", f"{output_suffix}.csv")

    # Save equalized networks
    net1_top.to_csv(out_file1, index=False)
    net2_top.to_csv(out_file2, index=False)

    print(f"\nEqualized networks saved:")
    print(f"  {out_file1}: {len(net1_top)} edges")
    print(f"  {out_file2}: {len(net2_top)} edges")

    # Display comparison statistics
    print(f"\nComparison statistics:")
    print(
        f"Network 1 importance range: {net1_top[imp_col1].min():.6f} - {net1_top[imp_col1].max():.6f}"
    )
    print(
        f"Network 2 importance range: {net2_top[imp_col2].min():.6f} - {net2_top[imp_col2].max():.6f}"
    )
    print(f"Network 1 mean importance: {net1_top[imp_col1].mean():.6f}")
    print(f"Network 2 mean importance: {net2_top[imp_col2].mean():.6f}")

    return net1_top, net2_top


if __name__ == "__main__":

    print("=" * 60)
    print("NETWORK EDGE COUNT EQUALIZER")
    print("=" * 60)

    # Get file paths
    file1 = r"KAN_Implementation/Data/grnboost2_1139_filtered_zscore_3.0.csv"
    file2 = r"KAN_Implementation/Data/kan_1139_filtered_zscore_3.0.csv"

    print(f"File 1: {file1}")
    print(f"File 2: {file2}")

    import os

    if not os.path.exists(file1):
        print(f" Error: File not found: {file1}")
        exit(1)
    if not os.path.exists(file2):
        print(f" Error: File not found: {file2}")
        exit(1)

    # Ask for target edge count
    print(f"\nChoose target edge count:")
    print("1. Use minimum of both networks (auto-equalize)")
    print("2. Specify custom edge count")

    choice = input("Enter choice (1 or 2): ").strip()

    target_edges = None
    if choice == "2":
        while True:
            try:
                target_edges = int(input("Enter target number of edges: "))
                if target_edges > 0:
                    break
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid integer.")

    # Equalize networks
    try:
        net1_eq, net2_eq = equalize_network_sizes(file1, file2, target_edges)
        print("\n Network equalization completed successfully!")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback

        traceback.print_exc()
