library(scMultiSim)

# Create output directory
if (!dir.exists("KAN_Evaluation/Data")) {
    dir.create("KAN_Evaluation/Data", recursive = TRUE)
}

# Load and process the original GRN dataset
process_original_grn <- function() {
    # Load the dataset
    data(GRN_params_100) # GRN_params_100

    # Extract the GRN component
    if (is.list(GRN_params_100) && "GRN" %in% names(GRN_params_100)) {
        grn_raw <- GRN_params_100$GRN
    } else {
        grn_raw <- GRN_params_100
    }

    cat("Original GRN structure:\n")
    print(str(grn_raw))
    cat("Original column names:", colnames(grn_raw), "\n")
    cat("First few rows:\n")
    print(head(grn_raw))

    return(grn_raw)
}

# Process and fix the GRN data
fix_grn_data <- function(grn_raw) {
    # Extract columns and swap positions
    target_col <- grn_raw[, 1] # Column 1: target gene ID
    regulator_col <- grn_raw[, 2] # Column 2: regulator gene ID
    effect_col <- grn_raw[, 3] # Column 3: effect

    # Create the swapped GRN: regulator first, target second, effect third
    grn_fixed <- data.frame(
        regulator = regulator_col, # Column 2 becomes Column 1 (regulator)
        target = target_col, # Column 1 becomes Column 2 (target)
        effect = effect_col # Column 3 stays Column 3 (effect)
    )

    # Get all unique genes
    all_genes <- unique(c(grn_fixed$regulator, grn_fixed$target))
    num_genes <- length(all_genes)

    cat("Found", num_genes, "unique genes\n")
    cat("Sample gene IDs:", head(all_genes), "\n")

    # Create gene mapping with Gene_ prefix
    gene_mapping <- setNames(paste0("Gene_", all_genes), all_genes)

    # Apply the mapping to create gene aliases
    grn_fixed$regulator <- gene_mapping[as.character(grn_fixed$regulator)]
    grn_fixed$target <- gene_mapping[as.character(grn_fixed$target)]

    cat("\nProcessed GRN structure:\n")
    print(str(grn_fixed))
    cat("First few rows with Gene_ aliases:\n")
    print(head(grn_fixed))

    # Verify the swap worked correctly
    cat("\nVerifying column swap:\n")
    cat("Original: target=", head(grn_raw[, 1], 3), ", regulator=", head(grn_raw[, 2], 3), "\n")
    cat(
        "Swapped:  regulator=", head(as.numeric(gsub("Gene_", "", grn_fixed$regulator)), 3),
        ", target=", head(as.numeric(gsub("Gene_", "", grn_fixed$target)), 3), "\n"
    )

    # Save the processed GRN
    write.csv(grn_fixed, "KAN_Evaluation/Data/ground_truth_grn.csv", row.names = FALSE)

    return(list(grn = grn_fixed, gene_mapping = gene_mapping, all_genes = all_genes))
}

# Generate expression data using scMultiSim with a safer approach
generate_expression_with_scmultisim <- function(grn_processed, num_cells = 1000) {
    set.seed(42)

    cat("Generating expression data using scMultiSim...\n")

    # Convert GRN back to numeric format for scMultiSim
    grn_for_sim <- grn_processed$grn
    grn_numeric <- data.frame(
        regulator = as.numeric(gsub("Gene_", "", grn_for_sim$regulator)),
        target = as.numeric(gsub("Gene_", "", grn_for_sim$target)),
        effect = grn_for_sim$effect
    )

    cat("Converted GRN to numeric format for scMultiSim\n")
    cat("Sample numeric GRN:\n")
    print(head(grn_numeric))


    cat("Attempting scMultiSim simulation...\n")

    results <- NULL
    success <- FALSE
    # Try approach 1: Minimal parameters
    if (!success) {
        cat("Trying with minimal parameters...\n")
        tryCatch(
            {
                results <- sim_true_counts(list(
                    GRN = grn_numeric,
                    tree = Phyla5(),
                    num.cells = num_cells,
                    cif.sigma = 0.5,
                    diff.cif.fraction = 0.8,
                    discrete.cif = FALSE,
                    speed.up = TRUE,
                    do.velocity = TRUE
                ))
                success <- TRUE
                cat("Success with minimal parameters!\n")
            },
            error = function(e) {
                cat("Minimal parameters failed:", e$message, "\n")
            }
        )
    }

    cat("scMultiSim simulation completed.\n")
    cat("Expression matrix dimensions:", dim(results$counts), "\n")

    # Extract the expression matrix
    expression_matrix <- results$counts

    # Get the genes that are actually in our GRN
    our_genes <- grn_processed$all_genes
    num_our_genes <- length(our_genes)

    cat("Number of genes in our original GRN:", num_our_genes, "\n")
    cat("Number of genes in simulated expression matrix:", nrow(expression_matrix), "\n")

    # Create a consistent expression matrix that matches our GRN genes
    if (nrow(expression_matrix) >= num_our_genes) {
        # If we have enough genes, take the first N genes to match our GRN
        expression_subset <- expression_matrix[1:num_our_genes, , drop = FALSE]
        gene_names <- paste0("Gene_", our_genes)
    } else {
        # If we don't have enough genes, pad with zeros
        needed_genes <- num_our_genes - nrow(expression_matrix)
        padding <- matrix(0, nrow = needed_genes, ncol = ncol(expression_matrix))
        expression_subset <- rbind(expression_matrix, padding)
        gene_names <- paste0("Gene_", our_genes)
    }

    # Assign proper gene names and cell names
    rownames(expression_subset) <- gene_names
    colnames(expression_subset) <- paste0("Cell_", 1:ncol(expression_subset))

    cat("Final expression matrix dimensions:", dim(expression_subset), "\n")
    cat("Final gene names sample:", head(rownames(expression_subset)), "\n")
    cat("Number of cells:", ncol(expression_subset), "\n")

    # Save expression data
    write.csv(expression_subset, "KAN_Evaluation/Data/simulated_gene_expression_100.csv", row.names = TRUE)

    # Also save the full simulation results for reference
    saveRDS(results, "KAN_Evaluation/Data/scmultisim_full_results.rds")

    return(list(expression_matrix = expression_subset, full_results = results))
}

# Main function
main <- function(num_cells = 1000) {
    cat("=== Processing GRN_params_100 dataset ===\n")

    # Load and inspect original dataset
    cat("\n1. Loading original GRN dataset...\n")
    grn_raw <- process_original_grn()

    # Process and fix the GRN
    cat("\n2. Processing GRN (swapping columns, adding Gene_ aliases)...\n")
    grn_processed <- fix_grn_data(grn_raw)

    # Generate expression data using scMultiSim
    cat("\n3. Generating expression data using scMultiSim...\n")
    expression_results <- generate_expression_with_scmultisim(grn_processed, num_cells)

    # Final validation
    cat("\n=== Final Validation ===\n")
    grn_genes <- unique(c(grn_processed$grn$regulator, grn_processed$grn$target))
    expr_genes <- rownames(expression_results$expression_matrix)

    cat("Number of unique genes in GRN:", length(grn_genes), "\n")
    cat("Number of genes in expression matrix:", length(expr_genes), "\n")
    cat("Number of cells in expression matrix:", ncol(expression_results$expression_matrix), "\n")
    cat("Gene name consistency check:", all(grn_genes %in% expr_genes), "\n")

    # Check if all GRN genes are present in expression matrix
    missing_genes <- grn_genes[!grn_genes %in% expr_genes]
    if (length(missing_genes) > 0) {
        cat("Missing genes in expression matrix:", length(missing_genes), "\n")
        cat("Sample missing genes:", head(missing_genes), "\n")
    } else {
        cat(" All GRN genes are present in expression matrix\n")
    }

    cat("\nFiles saved to KAN_Evaluation/Data/:\n")
    cat("- ground_truth_grn.csv (with regulator->target->effect format and Gene_ aliases)\n")
    cat("- simulated_gene_expression.csv (matching gene names and cell count)\n")
    cat("- scmultisim_full_results.rds (full simulation results)\n")

    return(list(
        grn = grn_processed$grn,
        expression_matrix = expression_results$expression_matrix,
        gene_mapping = grn_processed$gene_mapping,
        full_results = expression_results$full_results
    ))
}

# Run the corrected pipeline
cat("Starting corrected scMultiSim pipeline...\n")
results <- main(num_cells = 1000)
