# ##########HANDLES ALREADY NORMALIZED DATA############################################
# import torch
# import numpy as np
# import scanpy as sc
# import pandas as pd
# from scipy.sparse import csc_matrix
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
# from collections import defaultdict
# import os
# import gc
# import time


# class HPCSharedGeneDataManager:
#     """
#     Manages gene expression data as a shared GPU object.
#     Keeps data in GPU memory for shared access by multiple models
#     Filters to top 1000 highly variable genes
#     """

#     def __init__(self, device="cuda", scratch_dir=None, n_top_genes=1000):
#         """
#         Initialize the shared data manager

#         Args:
#             device: The device to store data on
#             scratch_dir: HPC scratch directory for temporary files
#             n_top_genes: Number of top highly variable genes to keep (default: 1000)
#         """
#         # Use SCRATCH directory if provided, otherwise use TMPDIR if available
#         self.scratch_dir = scratch_dir
#         if self.scratch_dir is None and "TMPDIR" in os.environ:
#             self.scratch_dir = os.path.join(
#                 os.environ["TMPDIR"], f"shared_data_{time.time()}"
#             )
#             os.makedirs(self.scratch_dir, exist_ok=True)
#             print(f"Using node-local storage for temporary files: {self.scratch_dir}")

#         # Set device
#         self.device = torch.device(
#             device if torch.cuda.is_available() and device == "cuda" else "cpu"
#         )
#         self.is_initialized = False
#         self.n_top_genes = n_top_genes

#         # Shared data structures
#         self.expr_tensor = None  # The full expression matrix as a torch tensor
#         self.gene_names = None  # Array of gene names (filtered to HVGs)
#         self.sample_names = None  # Array of sample names
#         self.hvg_genes = None  # Set of highly variable gene names

#         # Mapping dictionaries
#         self.gene_to_idx = {}  # Maps gene names to indices (for filtered genes)
#         self.gene_network = defaultdict(list)  # Maps target genes to related genes

#         # Gene data dictionary
#         self.gene_data_views = {}  # Stores views for each gene

#         # Memory usage tracking
#         self.memory_usage = {
#             "total_allocated": 0,
#             "expr_matrix_size": 0,
#             "num_genes": 0,
#             "num_samples": 0,
#         }

#         print(f"HPCSharedGeneDataManager initialized on {self.device}")
#         print(f"Will filter to top {self.n_top_genes} highly variable genes")

#     def _identify_highly_variable_genes(self, adata) -> List[str]:
#         """
#         Identify top highly variable genes based on variance

#         Args:
#             adata: AnnData object with expression data

#         Returns:
#             List of gene names for top highly variable genes
#         """
#         print(f"Identifying top {self.n_top_genes} highly variable genes...")

#         # Calculate variance for each gene
#         if hasattr(adata.X, "toarray"):
#             expr_array = adata.X.toarray()
#         else:
#             expr_array = adata.X

#         # Calculate variance across samples for each gene
#         gene_variances = np.var(expr_array, axis=0)

#         # Get indices of top variable genes
#         top_indices = np.argsort(gene_variances)[-self.n_top_genes :]

#         # Get gene names
#         hvg_genes = [adata.var_names[i] for i in top_indices]

#         print(f"Selected {len(hvg_genes)} highly variable genes")
#         print(
#             f"Variance range: {gene_variances[top_indices].min():.4f} - {gene_variances[top_indices].max():.4f}"
#         )

#         return hvg_genes

#     def load_data(self, expression_file: Path, network_file: Path) -> None:
#         """
#         Load data into shared GPU memory, filtering to top highly variable genes
#         Args:
#             expression_file: Path to h5ad expression data file
#             network_file: Path to network TSV file with regulator and regulated columns
#         """
#         print(f"Loading expression data into shared memory on {self.device}...")
#         start_time = time.time()

#         # Load expression data
#         adata = sc.read_h5ad(expression_file)
#         print(f"Original data shape: {adata.shape}")

#         # Identify highly variable genes
#         hvg_genes = self._identify_highly_variable_genes(adata)
#         self.hvg_genes = set(hvg_genes)

#         # Filter AnnData to only include HVGs
#         hvg_mask = [gene in self.hvg_genes for gene in adata.var_names]
#         adata_filtered = adata[:, hvg_mask].copy()
#         print(f"Filtered data shape: {adata_filtered.shape}")

#         # Convert sparse matrix to dense if needed
#         if hasattr(adata_filtered.X, "tocsc"):
#             # Using CSC format for efficient column slicing later
#             expr_matrix = adata_filtered.X.tocsc().toarray()
#             print(f"Converted sparse expression matrix to dense: {expr_matrix.shape}")
#         else:
#             expr_matrix = adata_filtered.X
#             print(f"Using dense expression matrix: {expr_matrix.shape}")

#         # Store gene and sample names (filtered)
#         self.gene_names = np.array(adata_filtered.var_names.tolist())
#         self.sample_names = np.array(adata_filtered.obs_names.tolist())
#         self.memory_usage["num_genes"] = len(self.gene_names)
#         self.memory_usage["num_samples"] = len(self.sample_names)

#         # Create mapping dictionary for filtered genes
#         self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_names)}

#         # Transfer entire matrix to GPU as a single tensor
#         self.expr_tensor = torch.tensor(
#             expr_matrix, dtype=torch.float32, device=self.device
#         )

#         print(f"Expression data loaded in {time.time() - start_time:.2f} seconds")
#         print(
#             f"Matrix shape: {self.expr_tensor.shape}, Features: {len(self.gene_names)}"
#         )

#         print("Loading network data...")
#         network_start = time.time()

#         # Try different separators to handle tab-separated files
#         separators = ["\t", ",", " ", ";", "|"]
#         network_df = None

#         for sep in separators:
#             try:

#                 test_df = pd.read_csv(network_file, sep=sep, header=None, nrows=5)

#                 print(f"Testing separator '{sep}': shape {test_df.shape}")

#                 # Check if we have at least 2 columns
#                 if test_df.shape[1] >= 2:

#                     network_df = pd.read_csv(network_file, sep=sep, header=None)
#                     print(f" Successfully read network file with separator '{sep}'")
#                     print(f"Network shape: {network_df.shape}")


#                     print("Sample network data:")
#                     print(network_df.head(3))
#                     break

#             except Exception as e:
#                 print(f"Failed with separator '{sep}': {e}")
#                 continue

#         if network_df is None or network_df.shape[1] < 2:
#             print(" Could not read network file with any separator!")
#             print("Attempting to parse manually...")


#             try:
#                 raw_df = pd.read_csv(network_file, header=None)
#                 if raw_df.shape[1] == 1:
#                     print("Detected single column, attempting to split on tabs...")

#                     # Split the single column on tabs
#                     split_data = raw_df.iloc[:, 0].str.split("\t", expand=True)

#                     if split_data.shape[1] >= 2:
#                         network_df = split_data.iloc[:, [0, 1]]  # Take first 2 columns
#                         network_df.columns = [0, 1]
#                         print(f" Successfully parsed: {network_df.shape}")
#                         print("Sample data after parsing:")
#                         print(network_df.head(3))
#                     else:
#                         raise ValueError("Could not split data into 2 columns")

#             except Exception as e:
#                 print(f"Manual parsing failed: {e}")
#                 print("Creating minimal sample network for testing...")

#                 # Create minimal sample network
#                 gene_sample = list(self.gene_names)[: min(50, len(self.gene_names))]
#                 n_connections = min(100, len(gene_sample) * 2)

#                 np.random.seed(42)
#                 regulators = np.random.choice(gene_sample, n_connections)
#                 regulated = np.random.choice(gene_sample, n_connections)

#                 network_df = pd.DataFrame({0: regulators, 1: regulated})

#                 # Remove self-loops
#                 network_df = network_df[network_df[0] != network_df[1]]
#                 print(f"Created sample network with {len(network_df)} connections")

#         total_connections = len(network_df)
#         print(f"Processing {total_connections} network connections...")

#         # Build network from genes that are in our filtered gene set
#         filtered_network = defaultdict(list)
#         valid_connections = 0
#         parsing_errors = 0

#         for idx, row in network_df.iterrows():
#             try:

#                 regulator = str(row.iloc[0]).strip()
#                 regulated = str(row.iloc[1]).strip()

#                 # Skip empty or NaN values
#                 if (
#                     pd.isna(regulator)
#                     or pd.isna(regulated)
#                     or regulator == ""
#                     or regulated == ""
#                 ):
#                     continue

#                 # Only keep connections where both genes are in our filtered gene set
#                 if regulator in self.gene_to_idx and regulated in self.gene_to_idx:
#                     filtered_network[regulated].append(regulator)
#                     valid_connections += 1

#             except Exception as e:
#                 parsing_errors += 1
#                 if parsing_errors <= 5:
#                     print(f"Warning: Error processing row {idx}: {e}")
#                     print(f"Row content: {row.values}")
#                 continue

#         if parsing_errors > 5:
#             print(f"... and {parsing_errors - 5} more parsing errors")


#         self.gene_network = filtered_network


#         unique_source_genes = set()
#         for target, sources in self.gene_network.items():
#             unique_source_genes.update(sources)

#         print(f"Network data loaded in {time.time() - network_start:.2f} seconds")
#         print(f"Valid connections: {valid_connections}/{total_connections}")
#         print(f"Target genes in network: {len(self.gene_network)}")
#         print(f"Unique source genes in network: {len(unique_source_genes)}")

#         if valid_connections == 0:
#             print("WARNING: No valid network connections found!")
#             print(
#                 "This might indicate that gene names in the network file don't match gene names in the expression file"
#             )
#             print("First 5 genes in expression data:", list(self.gene_names[:5]))
#             if len(network_df) > 0:
#                 print("First 5 regulators in network:", list(network_df.iloc[:5, 0]))
#                 print("First 5 regulated in network:", list(network_df.iloc[:5, 1]))
#         self.is_initialized = True
#         if self.device.type == "cuda":
#             cuda_total_memory = torch.cuda.get_device_properties(
#                 self.device
#             ).total_memory
#             print(f"Total GPU memory: {cuda_total_memory / 1e9:.2f} GB")
#             print(
#                 f"Current GPU memory usage: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB"
#             )

#     def get_data_for_gene(
#         self, target_gene: str
#     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[str]]:
#         """
#         Get data for a specific target gene, creating GPU tensors on demand.

#         Args:
#             target_gene: Name of the target gene

#         Returns:
#             Tuple of (input tensor, target tensor, related genes list)
#         """
#         if not self.is_initialized:
#             raise RuntimeError("Data manager not initialized. Call load_data() first.")

#         if target_gene in self.gene_data_views:
#             return self.gene_data_views[target_gene]

#         # Check if target gene is in our HVG set
#         if target_gene not in self.hvg_genes:
#             print(f"Warning: {target_gene} is not in the highly variable genes set")
#             return None, None, []

#         # Get related genes for this target
#         related_genes = self.gene_network.get(target_gene, [])
#         if not related_genes:
#             print(f"Warning: No related genes found for {target_gene}")
#             return None, None, []

#         # Get indices for this target and its related genes
#         target_idx = self.gene_to_idx[target_gene]
#         related_indices = [self.gene_to_idx[gene] for gene in related_genes]

#         try:
#             # Create a view of the shared tensor
#             X = self.expr_tensor[
#                 :, related_indices
#             ]  # All samples, related genes features
#             y = self.expr_tensor[
#                 :, target_idx
#             ].squeeze()  # All samples, target gene expression

#             # Store in cache to avoid recreating views
#             self.gene_data_views[target_gene] = (X, y, related_genes)

#             return X, y, related_genes, target_gene

#         except Exception as e:
#             print(f"Error creating data for gene {target_gene}: {e}")
#             return None, None, []

#     def prefetch_gene_batch(self, gene_batch: List[str]) -> None:
#         """
#         Prefetch data for a batch of genes to GPU memory.

#         Args:
#             gene_batch: List of genes to prefetch
#         """
#         print(f"Prefetching data for {len(gene_batch)} genes...")
#         for gene in gene_batch:
#             if gene not in self.gene_data_views and gene in self.gene_network:
#                 # Just call get_data_for_gene which handles the loading
#                 self.get_data_for_gene(gene)

#         if self.device.type == "cuda":
#             print(
#                 f"Current GPU memory usage after prefetch: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB"
#             )

#     def get_model_config(self, target_gene: str) -> Dict:
#         """Get model configuration for a gene."""
#         X, y, _ = self.get_data_for_gene(target_gene)
#         if X is None:
#             return {}

#         # Return configuration based on input dimensions
#         return {
#             "width": [X.shape[1], 3, 1],  # Input size -> hidden -> output
#             "grid": 4,
#             "k": 3,
#             "seed": 63,
#         }

#     def evict_gene_data(self, genes_to_evict: List[str]) -> None:
#         """
#         Remove specific genes from GPU memory to free space.

#         Args:
#             genes_to_evict: List of genes to remove from cache
#         """
#         for gene in genes_to_evict:
#             if gene in self.gene_data_views:
#                 del self.gene_data_views[gene]

#         # Force garbage collection
#         gc.collect()
#         if self.device.type == "cuda":
#             torch.cuda.empty_cache()
#             print(
#                 f"Current GPU memory usage after eviction: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB"
#             )

#     def get_all_target_genes(self) -> List[str]:
#         """Returns list of all valid target genes in the network (filtered to HVGs)."""
#         return list(self.gene_network.keys())

#     def get_hvg_genes(self) -> List[str]:
#         """Returns list of all highly variable genes."""
#         return list(self.hvg_genes) if self.hvg_genes else []

#     def cleanup(self):
#         """Release GPU memory and clean up temporary files."""
#         print("Cleaning up shared data manager resources...")
#         self.expr_tensor = None
#         self.gene_data_views = {}
#         self.hvg_genes = None

#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()


#         if self.scratch_dir and os.path.exists(self.scratch_dir):
#             try:
#                 import shutil

#                 shutil.rmtree(self.scratch_dir)
#                 print(f"Removed temporary directory: {self.scratch_dir}")
#             except Exception as e:
#                 print(f"Warning: Failed to remove temporary directory: {e}")

#         self.is_initialized = False
#         print("Shared data manager cleanup complete")


###Code that handles unnormalized data
import torch
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.sparse import csc_matrix
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import os
import gc
import time


class HPCSharedGeneDataManager:
    """
    Manages gene expression data as a shared GPU object.
    Keeps data in GPU memory for shared access by multiple models
    Filters to top 1000 highly variable genes
    """

    def __init__(self, device="cuda", scratch_dir=None, n_top_genes=1000):
        """
        Initialize the shared data manager

        Args:
            device: The device to store data on
            scratch_dir: HPC scratch directory for temporary files
            n_top_genes: Number of top highly variable genes to keep (default: 1000)
        """
        # Use SCRATCH directory if provided, otherwise use TMPDIR if available
        self.scratch_dir = scratch_dir
        if self.scratch_dir is None and "TMPDIR" in os.environ:
            self.scratch_dir = os.path.join(
                os.environ["TMPDIR"], f"shared_data_{time.time()}"
            )
            os.makedirs(self.scratch_dir, exist_ok=True)
            print(f"Using node-local storage for temporary files: {self.scratch_dir}")

        # Set device
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.is_initialized = False
        self.n_top_genes = n_top_genes

        # Shared data structures
        self.expr_tensor = None  # The full expression matrix as a torch tensor
        self.gene_names = None  # Array of gene names (filtered to HVGs)
        self.sample_names = None  # Array of sample names
        self.hvg_genes = None  # Set of highly variable gene names

        # Mapping dictionaries
        self.gene_to_idx = {}  # Maps gene names to indices (for filtered genes)
        self.gene_network = defaultdict(list)  # Maps target genes to related genes

        # Gene data dictionary
        self.gene_data_views = {}  # Stores views for each gene

        # Memory usage tracking
        self.memory_usage = {
            "total_allocated": 0,
            "expr_matrix_size": 0,
            "num_genes": 0,
            "num_samples": 0,
        }

        print(f"HPCSharedGeneDataManager initialized on {self.device}")
        print(f"Will filter to top {self.n_top_genes} highly variable genes")

    def _identify_highly_variable_genes(self, adata) -> List[str]:
        """
        Identify top highly variable genes based on variance

        Args:
            adata: AnnData object with expression data

        Returns:
            List of gene names for top highly variable genes
        """
        print(f"Identifying top {self.n_top_genes} highly variable genes...")

        # Calculate variance for each gene
        if hasattr(adata.X, "toarray"):
            expr_array = adata.X.toarray()
        else:
            expr_array = adata.X

        # Calculate variance across samples for each gene
        gene_variances = np.var(expr_array, axis=0)

        # Get indices of top variable genes
        top_indices = np.argsort(gene_variances)[-self.n_top_genes :]

        # Get gene names
        hvg_genes = [adata.var_names[i] for i in top_indices]

        print(f"Selected {len(hvg_genes)} highly variable genes")
        print(
            f"Variance range: {gene_variances[top_indices].min():.4f} - {gene_variances[top_indices].max():.4f}"
        )

        return hvg_genes

    def load_data(self, expression_file: Path, network_file: Path) -> None:
        """
        Targets = top-N HVGs from h5ad.
        Regulators (features) = regulators of those targets from GRN (col 0) that exist in h5ad.
        """

        import time, numpy as np, pandas as pd, torch
        from collections import defaultdict

        print(f"Loading expression data into shared memory on {self.device}.")
        t0 = time.time()

        # ── 1) Read + normalize expression ─────────────────────────────────────────
        adata = sc.read_h5ad(expression_file)
        print(f"Original data shape (cells x genes): {adata.shape}")

        print("Starting data normalization pipeline...")
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        print("Normalization done.")

        expr_gene_list = list(map(str, adata.var_names))
        expr_gene_set = set(expr_gene_list)
        print(f"Genes available in expression (h5ad): {len(expr_gene_set)}")

        # ── 2) HVG selection → these are the ONLY TARGETS ──────────────────────────
        hvg_targets_list = list(map(str, self._identify_highly_variable_genes(adata)))

        self.hvg_genes = set(hvg_targets_list)
        print(f"HVG targets selected from h5ad: {len(self.hvg_genes)}")

        # ── 3) Read GRN ───────
        print("Loading GRN...")
        try:

            grn = pd.read_csv(
                network_file, sep=None, engine="python", header=None, usecols=[0, 1]
            )
        except Exception:
            # fallback: try tab, comma
            for sep in ["\t", ",", ";", "|", r"\s+"]:
                try:
                    engine = "python" if sep == r"\s+" else "c"
                    grn = pd.read_csv(
                        network_file,
                        sep=sep,
                        engine=engine,
                        header=None,
                        usecols=[0, 1],
                    )
                    break
                except Exception:
                    grn = None
            if grn is None:
                raise RuntimeError(f"Could not read GRN file: {network_file}")

        grn.columns = ["regulator", "target"]
        grn["regulator"] = grn["regulator"].astype(str).str.strip()
        grn["target"] = grn["target"].astype(str).str.strip()

        total_edges = len(grn)
        raw_unique_targets = grn["target"].nunique()
        raw_unique_regs = grn["regulator"].nunique()
        print(
            f"GRN loaded: edges={total_edges}, unique_targets={raw_unique_targets}, unique_regulators={raw_unique_regs}"
        )

        # ── 4) Filter edges: keep those where target ∈ HVG targets and regulator ∈ expression ─
        # (We do NOT require regulator to be HVG; only to exist in h5ad so we have its expression.)
        mask = (
            grn["target"].isin(self.hvg_genes)
            & grn["regulator"].isin(expr_gene_set)
            & (grn["regulator"] != grn["target"])
        )
        grn_f = grn.loc[mask, ["regulator", "target"]]

        kept_edges = len(grn_f)
        kept_targets = grn_f["target"].nunique()
        kept_regs = grn_f["regulator"].nunique()
        missing_hvg_targets_in_grn = len(self.hvg_genes - set(grn["target"]))
        print(
            f"After filtering: edges={kept_edges}, targets_kept={kept_targets}/{len(self.hvg_genes)}, regulators_kept={kept_regs}"
        )
        if missing_hvg_targets_in_grn > 0:
            print(
                f"HVG targets not present as targets in GRN (will have no incoming edges): {missing_hvg_targets_in_grn}"
            )

        # ── 5) Build feature set = HVG targets ∪ kept regulators ───────────────────
        kept_regulators = set(grn_f["regulator"].unique())
        feature_genes = sorted(self.hvg_genes | kept_regulators)
        print(
            f"Total feature genes (targets ∪ regulators_in_h5ad): {len(feature_genes)}"
        )

        # ── 6) Subset expression to features and build tensors/mapping ─────────────
        feature_set = set(feature_genes)
        mask_cols = [g in feature_set for g in adata.var_names]
        ad_f = adata[:, mask_cols].copy()

        # Dense conversion if needed
        if hasattr(ad_f.X, "tocsc"):
            expr_matrix = ad_f.X.tocsc().toarray()
            print(f"Converted sparse expression to dense: {expr_matrix.shape}")
        else:
            expr_matrix = ad_f.X
            print(f"Using dense expression matrix: {expr_matrix.shape}")

        self.gene_names = np.array(ad_f.var_names.tolist())
        self.sample_names = np.array(ad_f.obs_names.tolist())
        self.gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}

        self.expr_tensor = torch.tensor(
            expr_matrix, dtype=torch.float32, device=self.device
        )
        print(
            f"Matrix shape: {self.expr_tensor.shape}, Features: {len(self.gene_names)}"
        )

        # ── 7) Build gene_network: for each HVG target, list its regulators kept ───

        self.gene_network = {}
        if not grn_f.empty:
            grouped = grn_f.groupby("target")["regulator"].apply(
                lambda s: list(dict.fromkeys(s.tolist()))
            )
            # Only include targets that got ≥1 regulator after filtering
            self.gene_network = {
                t: regs for t, regs in grouped.items() if len(regs) > 0
            }

        # Diagnostics
        targets_with_regs = len(self.gene_network)
        hvg_no_regs = len(self.hvg_genes) - targets_with_regs
        print(f"Targets with ≥1 regulator: {targets_with_regs} / {len(self.hvg_genes)}")
        print(f"HVG targets with NO regulators (in GRN ∩ h5ad): {hvg_no_regs}")

        # ── 8) Finalize ─────────────────────────────────────────────────────────────
        self.is_initialized = True
        print(f"Expression data + network loaded in {time.time() - t0:.2f} s")

        if self.device.type == "cuda":
            cuda_total_memory = torch.cuda.get_device_properties(
                self.device
            ).total_memory
            print(f"Total GPU memory: {cuda_total_memory / 1e9:.2f} GB")
            print(
                f"Current GPU memory usage: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB"
            )

    def get_data_for_gene(
        self, target_gene: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[str]]:
        """
        Get data for a specific target gene, creating GPU tensors on demand.

        Args:
            target_gene: Name of the target gene

        Returns:
            Tuple of (input tensor, target tensor, related genes list)
        """
        if not self.is_initialized:
            raise RuntimeError("Data manager not initialized. Call load_data() first.")

        if target_gene in self.gene_data_views:
            return self.gene_data_views[target_gene]

        # Check if target gene is in our HVG set
        if target_gene not in self.hvg_genes:
            print(f"Warning: {target_gene} is not in the highly variable genes set")
            return None, None, []

        # Get related genes for this target
        related_genes = self.gene_network.get(target_gene, [])
        if not related_genes:
            print(f"Warning: No related genes found for {target_gene}")
            return None, None, []

        # Get indices for this target and its related genes
        target_idx = self.gene_to_idx[target_gene]
        related_indices = [self.gene_to_idx[gene] for gene in related_genes]

        try:
            # Create a view of the shared tensor
            X = self.expr_tensor[
                :, related_indices
            ]  # All samples, related genes features
            y = self.expr_tensor[
                :, target_idx
            ].squeeze()  # All samples, target gene expression

            # Store in cache to avoid recreating views
            self.gene_data_views[target_gene] = (X, y, related_genes)

            return X, y, related_genes, target_gene

        except Exception as e:
            print(f"Error creating data for gene {target_gene}: {e}")
            return None, None, []

    def prefetch_gene_batch(self, gene_batch: List[str]) -> None:
        """
        Prefetch data for a batch of genes to GPU memory.

        Args:
            gene_batch: List of genes to prefetch
        """
        print(f"Prefetching data for {len(gene_batch)} genes...")
        for gene in gene_batch:
            if gene not in self.gene_data_views and gene in self.gene_network:

                self.get_data_for_gene(gene)

        if self.device.type == "cuda":
            print(
                f"Current GPU memory usage after prefetch: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB"
            )

    def get_model_config(self, target_gene: str) -> Dict:
        """Get model configuration for a gene."""
        X, y, _ = self.get_data_for_gene(target_gene)
        if X is None:
            return {}

        # Return configuration based on input dimensions
        return {
            "width": [X.shape[1], 2, 1],  # Input size -> hidden -> output
            "grid": 4,
            "k": 3,
            "seed": 63,
        }

    def evict_gene_data(self, genes_to_evict: List[str]) -> None:
        """
        Remove specific genes from GPU memory to free space.

        Args:
            genes_to_evict: List of genes to remove from cache
        """
        for gene in genes_to_evict:
            if gene in self.gene_data_views:
                del self.gene_data_views[gene]

        # Force garbage collection
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print(
                f"Current GPU memory usage after eviction: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB"
            )

    def get_all_target_genes(self) -> List[str]:
        """Returns list of all valid target genes in the network (filtered to HVGs)."""
        return list(self.gene_network.keys())

    def get_hvg_genes(self) -> List[str]:
        """Returns list of all highly variable genes."""
        return list(self.hvg_genes) if self.hvg_genes else []

    def cleanup(self):
        """Release GPU memory and clean up temporary files."""
        print("Cleaning up shared data manager resources...")
        self.expr_tensor = None
        self.gene_data_views = {}
        self.hvg_genes = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clean up scratch directory
        if self.scratch_dir and os.path.exists(self.scratch_dir):
            try:
                import shutil

                shutil.rmtree(self.scratch_dir)
                print(f"Removed temporary directory: {self.scratch_dir}")
            except Exception as e:
                print(f"Warning: Failed to remove temporary directory: {e}")

        self.is_initialized = False
        print("Shared data manager cleanup complete")
