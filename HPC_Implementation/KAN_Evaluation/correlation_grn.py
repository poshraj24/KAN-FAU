import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import networkx as nx


def identify_transcription_factors_from_network(
    correlation_pairs_file,
    tf_identification_method="degree_centrality",
    top_tf_percentage=10,
):
    """
    Identify transcription factors from correlation network topology


    Parameters:
    - correlation_pairs_file: Path to highly_correlated_pairs.csv
    - tf_identification_method: Method to identify TFs
                               'degree_centrality' - High degree nodes
                               'betweenness_centrality' - Nodes with high betweenness
                               'hub_genes' - Genes with many connections
    - top_tf_percentage: Top percentage of genes to consider as TFs

    Returns:
    - tf_set: Set of identified TF gene names
    - network_stats: Network analysis statistics
    """

    print("IDENTIFYING TRANSCRIPTION FACTORS FROM NETWORK TOPOLOGY")
    print("=" * 60)

    # Load correlation pairs
    df = pd.read_csv(correlation_pairs_file)
    print(f"Loaded {len(df):,} correlation pairs")

    # Create undirected graph for TF identification
    G = nx.Graph()

    # Add edges to graph
    for _, row in df.iterrows():
        gene1, gene2, corr = row["Gene1"], row["Gene2"], row["Correlation"]
        G.add_edge(gene1, gene2, weight=abs(corr))

    print(
        f"Created network with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges"
    )

    # Calculate network properties for TF identification
    if tf_identification_method == "degree_centrality":
        # Genes with highest degree (most connections) are likely TFs
        centrality_scores = nx.degree_centrality(G)

    elif tf_identification_method == "betweenness_centrality":
        # Genes that are bridges between other genes (regulatory hubs)
        centrality_scores = nx.betweenness_centrality(G)

    elif tf_identification_method == "hub_genes":
        # Simple degree-based approach
        degrees = dict(G.degree())
        centrality_scores = {
            gene: degree / max(degrees.values()) for gene, degree in degrees.items()
        }

    else:
        raise ValueError(
            f"Unknown TF identification method: {tf_identification_method}"
        )

    # Select top genes as TFs
    sorted_genes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
    num_tfs = max(1, int(len(sorted_genes) * top_tf_percentage / 100))

    tf_set = set([gene for gene, score in sorted_genes[:num_tfs]])

    print(f"\nTF Identification Results:")
    print(f"- Method used: {tf_identification_method}")
    print(f"- Top {top_tf_percentage}% of genes selected as TFs")
    print(f"- Number of TFs identified: {len(tf_set):,}")

    # Show top TFs
    print(f"\nTop 10 identified TFs:")
    for i, (gene, score) in enumerate(sorted_genes[:10], 1):
        print(f"{i:2d}. {gene}: {score:.4f}")

    network_stats = {
        "total_genes": G.number_of_nodes(),
        "total_correlations": G.number_of_edges(),
        "tfs_identified": len(tf_set),
        "tf_percentage": (len(tf_set) / G.number_of_nodes()) * 100,
        "method_used": tf_identification_method,
    }

    return tf_set, network_stats


def create_directed_grn(
    correlation_pairs_file,
    tf_set,
    min_correlation=0.0,
    save_results=True,
    output_prefix="rule_grn",
):
    """
    Create directed GRN using  EXACT rule

     Rule:
    - If g1 is TF and g2 is not TF → add edge g1 → g2
    - If g1 and g2 are both TFs → add edges g1 → g2 AND g2 → g1
    - If neither is TF → skip

    Parameters:
    - correlation_pairs_file: Path to highly_correlated_pairs.csv
    - tf_set: Set of identified TF gene names
    - min_correlation: Additional correlation threshold
    - save_results: Whether to save results
    - output_prefix: Prefix for output files

    Returns:
    - directed_edges: List of (source, target, weight) tuples
    - stats: Creation statistics
    """

    print(f"\nCREATING DIRECTED GRN USING  RULE")
    print("=" * 50)

    # Load correlation pairs
    df = pd.read_csv(correlation_pairs_file)

    directed_edges = []
    stats = {
        "total_correlation_pairs": len(df),
        "processed_pairs": 0,
        "tf_to_gene_edges": 0,  # TF → Gene (single direction)
        "tf_to_tf_pairs": 0,  # TF ↔ TF (bidirectional)
        "non_tf_pairs_skipped": 0,  # Neither is TF
        "below_threshold": 0,  # Below correlation threshold
        "total_edges_created": 0,
    }

    print(f"Applying  rule to {len(df):,} correlation pairs...")
    print(f"Using {len(tf_set):,} identified TFs")
    print(f"Minimum correlation threshold: {min_correlation}")

    # Apply  exact rule
    for _, row in df.iterrows():
        gene1, gene2, corr = row["Gene1"], row["Gene2"], row["Correlation"]
        stats["processed_pairs"] += 1

        # Skip if below threshold
        if abs(corr) < min_correlation:
            stats["below_threshold"] += 1
            continue

        # Check TF status
        is_gene1_tf = gene1 in tf_set
        is_gene2_tf = gene2 in tf_set

        if is_gene1_tf and not is_gene2_tf:
            # Rule 1: g1 is TF, g2 is not TF → g1 → g2
            directed_edges.append((gene1, gene2, abs(corr)))
            stats["tf_to_gene_edges"] += 1
            stats["total_edges_created"] += 1

        elif not is_gene1_tf and is_gene2_tf:
            # Rule 1 (reverse): g1 is not TF, g2 is TF → g2 → g1
            directed_edges.append((gene2, gene1, abs(corr)))
            stats["tf_to_gene_edges"] += 1
            stats["total_edges_created"] += 1

        elif is_gene1_tf and is_gene2_tf:
            # Rule 2: Both are TFs → bidirectional g1 ↔ g2
            directed_edges.append((gene1, gene2, abs(corr)))
            directed_edges.append((gene2, gene1, abs(corr)))
            stats["tf_to_tf_pairs"] += 1
            stats["total_edges_created"] += 2

        else:
            # Neither is TF → skip
            stats["non_tf_pairs_skipped"] += 1

    # Print results
    print(f"\n Rule Application Results:")
    print(f"Total correlation pairs: {stats['total_correlation_pairs']:,}")
    print(f"Processed pairs: {stats['processed_pairs']:,}")
    print(f"TF → Gene edges: {stats['tf_to_gene_edges']:,}")
    print(
        f"TF - TF pairs: {stats['tf_to_tf_pairs']:,} (= {stats['tf_to_tf_pairs']*2:,} edges)"
    )
    print(f"Non-TF pairs skipped: {stats['non_tf_pairs_skipped']:,}")
    print(f"Below threshold skipped: {stats['below_threshold']:,}")
    print(f"Total directed edges created: {stats['total_edges_created']:,}")

    if save_results:
        save__rule_results(directed_edges, tf_set, stats, output_prefix)

    return directed_edges, stats


def save__rule_results(directed_edges, tf_set, stats, output_prefix):
    """
    Save all results from  rule application
    """

    # 1. Directed edges
    edges_df = pd.DataFrame(directed_edges, columns=["Source", "Target", "Weight"])
    edges_file = f"{output_prefix}_directed_edges.csv"
    edges_df.to_csv(edges_file, index=False)
    print(f" Saved {len(directed_edges):,} directed edges to '{edges_file}'")

    # 2. Identified TFs
    tf_df = pd.DataFrame(list(tf_set), columns=["TF"])
    tf_file = f"{output_prefix}_identified_tfs.csv"
    tf_df.to_csv(tf_file, index=False)
    print(f" Saved {len(tf_set)} identified TFs to '{tf_file}'")

    # 3. Statistics
    stats_df = pd.DataFrame([stats]).T
    stats_df.columns = ["Count"]
    stats_file = f"{output_prefix}_stats.csv"
    stats_df.to_csv(stats_file)
    print(f" Saved statistics to '{stats_file}'")

    # 4. Edge classification
    edge_types = []
    for source, target, weight in directed_edges:
        if source in tf_set and target in tf_set:
            edge_type = "TF_to_TF"
        elif source in tf_set:
            edge_type = "TF_to_Gene"
        else:
            edge_type = "Gene_to_TF"

        edge_types.append(
            {
                "Source": source,
                "Target": target,
                "Weight": weight,
                "EdgeType": edge_type,
            }
        )

    edge_types_df = pd.DataFrame(edge_types)
    edge_types_file = f"{output_prefix}_edge_types.csv"
    edge_types_df.to_csv(edge_types_file, index=False)
    print(f" Saved edge classification to '{edge_types_file}'")


def analyze_final_grn(directed_edges, tf_set):
    """
    Analyze the final directed GRN structure
    """

    # Basic network properties
    sources = [edge[0] for edge in directed_edges]
    targets = [edge[1] for edge in directed_edges]
    all_nodes = set(sources + targets)

    # Node classification
    tfs_in_network = all_nodes.intersection(tf_set)
    genes_in_network = all_nodes - tf_set

    # Degree calculations
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)

    for source, target, weight in directed_edges:
        out_degree[source] += 1
        in_degree[target] += 1

    # TF analysis
    tf_out_degrees = [out_degree[tf] for tf in tfs_in_network]
    tf_in_degrees = [in_degree[tf] for tf in tfs_in_network]

    # Gene analysis
    gene_in_degrees = [in_degree[gene] for gene in genes_in_network]

    analysis = {
        "total_nodes": len(all_nodes),
        "total_edges": len(directed_edges),
        "tfs_in_network": len(tfs_in_network),
        "genes_in_network": len(genes_in_network),
        "avg_tf_out_degree": np.mean(tf_out_degrees) if tf_out_degrees else 0,
        "max_tf_out_degree": max(tf_out_degrees) if tf_out_degrees else 0,
        "avg_tf_in_degree": np.mean(tf_in_degrees) if tf_in_degrees else 0,
        "avg_gene_in_degree": np.mean(gene_in_degrees) if gene_in_degrees else 0,
        "max_gene_in_degree": max(gene_in_degrees) if gene_in_degrees else 0,
    }

    print(f"\nFinal GRN Analysis:")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.2f}")
        else:
            print(f"- {key}: {value:,}")

    return analysis


def main_rule_grn(
    correlation_pairs_file="highly_correlated_pairs.csv",
    tf_identification_method="degree_centrality",
    top_tf_percentage=10,
    min_correlation=0.0,
):
    """
    Complete workflow: Identify TFs and create directed GRN using  rule

    Parameters:
    - correlation_pairs_file: Your highly_correlated_pairs.csv
    - tf_identification_method: How to identify TFs from network topology
    - top_tf_percentage: Percentage of top genes to consider as TFs
    - min_correlation: Additional correlation threshold

    Returns:
    - directed_edges: Final directed GRN
    - tf_set: Identified TFs
    - stats: Creation statistics
    - analysis: Network analysis
    """

    # Step 1: Identify TFs from network topology
    tf_set, network_stats = identify_transcription_factors_from_network(
        correlation_pairs_file=correlation_pairs_file,
        tf_identification_method=tf_identification_method,
        top_tf_percentage=top_tf_percentage,
    )

    # Step 2: Apply  rule to create directed GRN
    directed_edges, stats = create_directed_grn(
        correlation_pairs_file=correlation_pairs_file,
        tf_set=tf_set,
        min_correlation=min_correlation,
        save_results=True,
        output_prefix="rule_grn",
    )

    # Step 3: Analyze final GRN
    analysis = analyze_final_grn(directed_edges, tf_set)

    return directed_edges, tf_set, stats, analysis


if __name__ == "__main__":
    # Complete workflow: Identify TFs and create directed GRN
    directed_edges, tf_set, stats, analysis = main_rule_grn(
        correlation_pairs_file="highly_correlated_pairs.csv",
        tf_identification_method="degree_centrality",  # High connectivity genes = TFs
        top_tf_percentage=100,  # Top 80% of genes as TFs
        min_correlation=0.0,
    )
