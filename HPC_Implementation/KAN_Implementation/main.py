from pathlib import Path
import warnings
import os
import json
import argparse
import sys
import time
from tqdm import tqdm
from data_loader import HPCSharedGeneDataManager


from train_with_copies import (
    train_kan_models_parallel_hpc_with_copies,
)

# Suppress warnings
warnings.filterwarnings("ignore", message="meta NOT subset.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="Importing read_umi_tools from `anndata` is deprecated",
    category=FutureWarning,
)


def find_existing_checkpoint(output_dir, search_paths=None):
    """
    Robustly find existing checkpoint files in multiple possible locations
    """
    if search_paths is None:
        search_paths = []

    checkpoint_locations = [
        os.path.join(output_dir, "training_checkpoint.json"),
        # Alternative locations
        "training_checkpoint.json",
        os.path.join("kan_models", "training_checkpoint.json"),
        os.path.join("..", "kan_models", "training_checkpoint.json"),
        os.path.join("KAN_Implementation", "kan_models", "training_checkpoint.json"),
        *search_paths,
    ]

    if "WORK" in os.environ:
        work_dir = os.environ["WORK"]
        checkpoint_locations.extend(
            [
                os.path.join(work_dir, "kan_results", "training_checkpoint.json"),
                os.path.join(work_dir, "kan_models", "training_checkpoint.json"),
            ]
        )

    if "SLURM_SUBMIT_DIR" in os.environ:
        submit_dir = os.environ["SLURM_SUBMIT_DIR"]
        checkpoint_locations.extend(
            [
                os.path.join(submit_dir, "kan_models", "training_checkpoint.json"),
                os.path.join(submit_dir, "training_checkpoint.json"),
            ]
        )

    print(
        f"DEBUG: Searching for checkpoint in {len(checkpoint_locations)} locations..."
    )

    for i, location in enumerate(checkpoint_locations):
        abs_location = os.path.abspath(location)
        print(f"DEBUG: [{i+1}] Checking: {abs_location}")

        if os.path.exists(location):
            try:

                with open(location, "r") as f:
                    checkpoint_data = json.load(f)
                    if "completed_genes" in checkpoint_data:
                        print(f"DEBUG:  Found valid checkpoint at: {abs_location}")
                        print(
                            f"DEBUG: Checkpoint contains {len(checkpoint_data['completed_genes'])} completed genes"
                        )
                        return location, checkpoint_data
                    else:
                        print(f"DEBUG:  Invalid checkpoint format at: {abs_location}")
            except (json.JSONDecodeError, Exception) as e:
                print(f"DEBUG:  Error reading checkpoint at {abs_location}: {e}")
        else:
            print(f"DEBUG:  Not found: {abs_location}")

    print("DEBUG: No valid checkpoint found in any location")
    return None, None


def copy_checkpoint_to_output_dir(checkpoint_path, output_dir):
    """
    Copy found checkpoint to the output directory if it's not already there
    """
    target_path = os.path.join(output_dir, "training_checkpoint.json")

    if os.path.abspath(checkpoint_path) != os.path.abspath(target_path):
        try:
            print(f"DEBUG: Copying checkpoint from {checkpoint_path} to {target_path}")
            os.makedirs(output_dir, exist_ok=True)
            import shutil

            shutil.copy2(checkpoint_path, target_path)
            print("DEBUG: Checkpoint copied successfully")
            return target_path
        except Exception as e:
            print(f"DEBUG: Warning - could not copy checkpoint: {e}")
            return checkpoint_path

    return checkpoint_path


def load_checkpoint_with_fallback(output_dir):
    """
    Robust checkpoint loading with multiple fallback locations
    """

    checkpoint_path, checkpoint_data = find_existing_checkpoint(output_dir)

    if checkpoint_path is None:
        print("DEBUG: No checkpoint found, starting fresh")
        return []

    final_checkpoint_path = copy_checkpoint_to_output_dir(checkpoint_path, output_dir)

    completed_genes = checkpoint_data.get("completed_genes", [])
    print(
        f"DEBUG: Successfully loaded checkpoint with {len(completed_genes)} completed genes"
    )

    return completed_genes


def debug_gene_computation(data_manager):
    """Debug the gene list computation"""

    print("\n" + " " * 60)
    print("GENE COMPUTATION DEBUG ANALYSIS")
    print(" " * 60)

    print(f"Parameter Check:")
    print(f"n_top_genes setting: {data_manager.n_top_genes}")
    print(f"Expected: 1000")

    if data_manager.n_top_genes != 1000:
        print(f"{data_manager.n_top_genes}")
        print(f"data_manager.n_top_genes genes")

    # Step 2: Check HVG selection results
    hvg_genes = data_manager.get_hvg_genes()
    print(f"\n HVG Selection Results:")
    print(f"Total HVG genes selected: {len(hvg_genes)}")
    print(f"Should equal n_top_genes: {len(hvg_genes) == data_manager.n_top_genes}")

    # Step 3: Check network filtering results
    target_genes = data_manager.get_all_target_genes()
    print(f"\n Network Filtering Results:")
    print(f"Target genes from network: {len(target_genes)}")
    print(
        f"Percentage of HVGs that become targets: {len(target_genes)/len(hvg_genes)*100:.1f}%"
    )

    # Step 4: Verify the mathematical constraint
    print(f"\n Mathematical Check:")
    print(f"HVG genes: {len(hvg_genes)}")
    print(f"Target genes: {len(target_genes)}")
    print(f"Target genes ≤ HVG genes: {len(target_genes) <= len(hvg_genes)}")

    if len(target_genes) > len(hvg_genes):
        print(f"ERROR: This should be impossible!")

    # Step 5: Check for genes that appear as targets but aren't in HVG set
    hvg_set = set(hvg_genes)
    target_set = set(target_genes)
    targets_not_in_hvg = target_set - hvg_set

    print(f"Target genes not in HVG set: {len(targets_not_in_hvg)}")
    if targets_not_in_hvg:
        print(f"ERROR: Found {len(targets_not_in_hvg)} target genes not in HVG set!")
        print(f"Examples: {list(targets_not_in_hvg)[:5]}")
    else:
        print(f"All target genes are properly in HVG set")

    # Step 6: Show the breakdown
    print(f"\nFinal Breakdown:")
    print(
        f"Expression data : {data_manager.n_top_genes} HVG genes : {len(target_genes)} target genes"
    )
    print(f"training will process {len(target_genes)} genes total")

    # Step 7: Check if this was caused by parameter change
    if data_manager.n_top_genes != 1000:
        print(f"Resolution:")
        print(f"n_top_genes parameter is {data_manager.n_top_genes}")

        print(f"Current setting will give ~{len(target_genes)} genes")

    # Save analysis
    analysis = {
        "n_top_genes_parameter": data_manager.n_top_genes,
        "hvg_genes_selected": len(hvg_genes),
        "target_genes_final": len(target_genes),
        "hvg_to_target_ratio": len(target_genes) / len(hvg_genes) if hvg_genes else 0,
        "targets_not_in_hvg": len(targets_not_in_hvg),
        "parameter_matches_expectation": data_manager.n_top_genes == 1000,
        "mathematical_constraint_satisfied": len(target_genes) <= len(hvg_genes),
    }

    with open("gene_computation_debug.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"Analysis saved to: gene_computation_debug.json")
    print("*" * 60)

    return analysis


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train KAN models for gene regulatory networks using HPC-optimized shared data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="kan_models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for training"
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=10,
        help="Maximum number of models to train simultaneously",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs for training"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--expression-file",
        type=str,
        default="KAN_Implementation/Data/simulated_gene_expression_1139.h5ad",
        help="Path to expression data file",
    )
    parser.add_argument(
        "--network-file",
        type=str,
        default="KAN_Implementation/Data/kan_1139_original_filtered_zscore_2.5.csv",
        help="Path to network file",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=-1,
        help="Process specific chunk of genes (use with --total-chunks)",
    )
    parser.add_argument(
        "--total-chunks",
        type=int,
        default=1,
        help="Total number of chunks to divide gene list into",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoint, start fresh",
    )

    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=None,
        help="Scratch directory for temporary files",
    )
    parser.add_argument(
        "--home-dir",
        type=str,
        default=None,
        help="Home directory where essential files will be saved",
    )
    parser.add_argument(
        "--data-workers",
        type=int,
        default=0,
        help="Number of data loader workers (0 = auto-detect)",
    )
    # Add new argument for choosing the training method
    parser.add_argument(
        "--use-data-copies",
        action="store_true",
        help="Use data copies instead of views for potentially better performance",
    )
    parser.add_argument(
        "--generate-symbolic",
        action="store_true",
        default=True,
        help="Generate symbolic formulas",
    )
    parser.add_argument(
        "--checkpoint-search-paths",
        type=str,
        nargs="*",
        default=[],
        help="Additional paths to search for existing checkpoints",
    )

    args = parser.parse_args()

    # Get chunk ID and total chunks from environment variables if not specified
    if args.chunk_id == -1 and "SLURM_ARRAY_TASK_ID" in os.environ:
        args.chunk_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if args.total_chunks == 1 and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        args.total_chunks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    # Set up scratch and home directories based on HPC environment
    if args.scratch_dir is None and "TMPDIR" in os.environ:
        args.scratch_dir = os.environ["TMPDIR"]
        print(f"Using TMPDIR as scratch directory: {args.scratch_dir}")

    # Print run configuration
    print("=" * 50)
    print("KAN Training Configuration (HPC-Optimized):")
    print(f"Output directory: {args.output_dir}")
    print(f"Expression file: {args.expression_file}")
    print(f"Network file: {args.network_file}")
    print(
        f"Training parameters: batch_size={args.batch_size}, max_models={args.max_models}, epochs={args.epochs}"
    )
    print(
        f"Training method: {'Data Copies' if args.use_data_copies else 'Shared Views'}"
    )
    print(f"Resume from checkpoint: {'No' if args.no_resume else 'Yes'}")

    if args.chunk_id >= 0:
        print(f"Processing chunk {args.chunk_id + 1} of {args.total_chunks}")

    print("\nHPC settings:")
    print(f"Scratch directory: {args.scratch_dir}")
    print(f"Data workers: {args.data_workers if args.data_workers > 0 else 'Auto'}")
    print(f"Generate symbolic formulas: {args.generate_symbolic}")
    print("=" * 50)

    # Set global optimization settings for data loading
    if args.data_workers > 0:
        os.environ["DATA_LOADER_WORKERS"] = str(args.data_workers)

    # Determine output directory
    if args.chunk_id >= 0:
        chunk_output_dir = os.path.join(args.output_dir, f"chunk_{args.chunk_id}")
    else:
        chunk_output_dir = args.output_dir

    # Create output directory
    os.makedirs(chunk_output_dir, exist_ok=True)

    # Initialize the shared data manager
    data_manager = HPCSharedGeneDataManager(device="cuda", scratch_dir=args.scratch_dir)

    # Load data once into shared memory
    data_manager.load_data(Path(args.expression_file), Path(args.network_file))

    debug_analysis = debug_gene_computation(data_manager)

    # Get all available genes
    all_genes = data_manager.get_all_target_genes()
    print(f"Found {len(all_genes)} total target genes")

    # If processing chunks, select specific genes
    if args.chunk_id >= 0:
        # Calculate chunk size and get genes for this chunk
        chunk_size = (len(all_genes) + args.total_chunks - 1) // args.total_chunks
        start_idx = args.chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_genes))
        genes_list = all_genes[start_idx:end_idx]

        print(
            f"Processing chunk {args.chunk_id + 1}/{args.total_chunks}: genes {start_idx + 1}-{end_idx} of {len(all_genes)}"
        )
    else:
        genes_list = all_genes

    # Load checkpoint and filter genes if resuming
    if not args.no_resume:
        print("\n" + "=" * 50)
        print("CHECKPOINT SYSTEM")
        print("=" * 50)

        completed_genes = load_checkpoint_with_fallback(chunk_output_dir)

        if completed_genes:
            # Filter out already processed genes
            original_count = len(genes_list)
            genes_list = [g for g in genes_list if g not in completed_genes]
            filtered_count = len(genes_list)

            print(
                f"Resuming from checkpoint: {len(completed_genes)} genes already processed"
            )
            print(f"Original gene list: {original_count} genes")
            print(f"Filtered gene list: {filtered_count} genes")
            print(f"Genes to process: {filtered_count}")

            if filtered_count == 0:
                print("All genes have been processed!")
                data_manager.cleanup()
                print("Training complete!")
                return
        else:
            print("No checkpoint found or checkpoint loading failed")
            print("Starting fresh training")

        print("=" * 50)
    else:
        print("Checkpoint resuming disabled by --no-resume flag")

    print(f"Processing {len(genes_list)} genes in this run")

    # Train models using the selected approach
    start_time = time.time()

    # Always use data copies implementation
    print("Using data copies implementation")
    train_kan_models_parallel_hpc_with_copies(
        gene_list=genes_list,
        data_manager=data_manager,
        output_dir=chunk_output_dir,
        home_dir=None,
        batch_size=args.batch_size,
        max_models=args.max_models,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.learning_rate,
        resume_from_checkpoint=not args.no_resume,
        generate_symbolic=args.generate_symbolic,
    )

    end_time = time.time()

    # Print total runtime
    print(f"Total runtime: {(end_time - start_time) / 3600:.2f} hours")

    # If processing chunks, create a flag file for this chunk
    if args.chunk_id >= 0:
        with open(
            os.path.join(chunk_output_dir, f"chunk_{args.chunk_id}_complete.flag"), "w"
        ) as f:
            f.write(
                f"Chunk {args.chunk_id} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Processed {len(genes_list)} genes\n")
            f.write(f"Runtime: {(end_time - start_time) / 3600:.2f} hours\n")
            f.write("Training method: Data Copies\n")

    # Cleanup shared data manager
    data_manager.cleanup()

    print("Training complete!")


if __name__ == "__main__":
    main()
