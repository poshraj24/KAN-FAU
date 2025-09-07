import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def load_gene_expression_data(filepath):
    """
    Load gene expression data from CSV file
    """
    df = pd.read_csv(filepath, index_col=0)
    print(f"Data shape: {df.shape[0]} genes x {df.shape[1]} cells")
    return df


def calculate_gene_correlations(df, method="pearson", min_correlation=0.0):
    """
    Calculate pairwise correlations between all genes

    Parameters:
    - df: DataFrame with genes as rows and cells as columns
    - method: correlation method ('pearson', 'spearman', 'kendall')
    - min_correlation: minimum correlation threshold to consider

    Returns:
    - correlation_matrix: full correlation matrix
    - correlation_pairs: list of (gene1, gene2, correlation) tuples
    """
    print("Calculating gene-gene correlations...")

    # Calculate correlation matrix
    correlation_matrix = df.T.corr(method=method)

    # Extract upper triangle (avoid duplicates and self-correlations)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    correlation_pairs = []

    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            gene1 = correlation_matrix.index[i]
            gene2 = correlation_matrix.index[j]
            corr_value = correlation_matrix.iloc[i, j]

            # Only include correlations above threshold and not NaN
            if not np.isnan(corr_value) and abs(corr_value) >= min_correlation:
                correlation_pairs.append((gene1, gene2, corr_value))

    print(
        f"Found {len(correlation_pairs)} gene pairs with |correlation| >= {min_correlation}"
    )
    return correlation_matrix, correlation_pairs


def get_top_correlated_genes(correlation_pairs, percentile=80):
    """
    Get genes that are in the top percentile of correlations

    Parameters:
    - correlation_pairs: list of (gene1, gene2, correlation) tuples
    - percentile: percentile threshold (e.g., 80 for top 80%)

    Returns:
    - top_genes: set of genes in top correlations
    - threshold: correlation threshold used
    """
    # Get absolute correlation values
    abs_correlations = [abs(pair[2]) for pair in correlation_pairs]

    # Calculate threshold for top percentile
    threshold = np.percentile(abs_correlations, percentile)

    print(f"Correlation threshold for top {percentile}%: {threshold:.4f}")

    # Get genes that participate in top correlations
    top_genes = set()
    top_pairs = []

    for gene1, gene2, corr in correlation_pairs:
        if abs(corr) >= threshold:
            top_genes.add(gene1)
            top_genes.add(gene2)
            top_pairs.append((gene1, gene2, corr))

    print(f"Number of highly correlated gene pairs: {len(top_pairs)}")
    print(f"Number of genes in top {percentile}% correlations: {len(top_genes)}")

    return top_genes, threshold, top_pairs


def filter_expression_matrix(df, selected_genes):
    """
    Filter the expression matrix to keep only selected genes
    """
    filtered_df = df.loc[list(selected_genes)]
    print(
        f"Filtered matrix shape: {filtered_df.shape[0]} genes x {filtered_df.shape[1]} cells"
    )
    return filtered_df


def plot_correlation_distribution(correlation_pairs, threshold=None):
    """
    Plot distribution of correlations
    """
    correlations = [pair[2] for pair in correlation_pairs]

    plt.figure(figsize=(10, 6))
    plt.hist(correlations, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Frequency")
    plt.title("Distribution of Gene-Gene Correlations")

    if threshold:
        plt.axvline(
            threshold,
            color="red",
            linestyle="--",
            label=f"Top 80% threshold: {threshold:.4f}",
        )
        plt.axvline(-threshold, color="red", linestyle="--")
        plt.legend()

    plt.grid(True, alpha=0.3)
    plt.show()


def save_results(top_genes, top_pairs, filtered_df, output_prefix="highly_correlated"):
    """
    Save results to files
    """
    # Save list of highly correlated genes
    genes_df = pd.DataFrame(list(top_genes), columns=["Gene_ID"])
    genes_df.to_csv(f"{output_prefix}_genes.csv", index=False)
    print(
        f"Saved {len(top_genes)} highly correlated genes to '{output_prefix}_genes.csv'"
    )

    # Save top correlation pairs
    pairs_df = pd.DataFrame(top_pairs, columns=["Gene1", "Gene2", "Correlation"])
    pairs_df = pairs_df.sort_values("Correlation", key=abs, ascending=False)
    pairs_df.to_csv(f"{output_prefix}_pairs.csv", index=False)
    print(f"Saved {len(top_pairs)} gene pairs to '{output_prefix}_pairs.csv'")

    # Save filtered expression matrix
    filtered_df.to_csv(f"{output_prefix}_expression_matrix.csv")
    print(
        f"Saved filtered expression matrix to '{output_prefix}_expression_matrix.csv'"
    )

    return genes_df, pairs_df


def main():
    """
    Main analysis pipeline
    """
    # Load data
    filepath = "KAN_Implementation/Data/simulated_gene_expression.csv"
    df = load_gene_expression_data(filepath)

    # Calculate correlations
    correlation_matrix, correlation_pairs = calculate_gene_correlations(df)

    # Get top 80% highly correlated genes
    top_genes, threshold, top_pairs = get_top_correlated_genes(
        correlation_pairs, percentile=80
    )

    # Filter expression matrix
    filtered_df = filter_expression_matrix(df, top_genes)

    # Plot correlation distribution
    plot_correlation_distribution(correlation_pairs, threshold)

    # Save results
    genes_df, pairs_df = save_results(top_genes, top_pairs, filtered_df)

    # Print summary statistics
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Original dataset: {df.shape[0]} genes x {df.shape[1]} cells")
    print(f"Total gene pairs analyzed: {len(correlation_pairs)}")
    print(f"Correlation threshold (80th percentile): {threshold:.4f}")
    print(f"Highly correlated genes selected: {len(top_genes)}")
    print(f"Percentage of genes retained: {len(top_genes)/df.shape[0]*100:.1f}%")
    print(f"Highly correlated pairs: {len(top_pairs)}")

    # Show top 10 correlations
    print(f"\nTop 10 highest correlations:")
    print("-" * 50)
    top_10 = sorted(top_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]
    for i, (gene1, gene2, corr) in enumerate(top_10, 1):
        print(f"{i:2d}. {gene1} - {gene2}: {corr:.4f}")

    return filtered_df, top_genes, top_pairs


if __name__ == "__main__":

    filtered_df, top_genes, top_pairs = main()
