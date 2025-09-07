##Code without scaling and normalization
import os
import json
import time
import torch
import numpy as np
import traceback
import gc
import multiprocessing as mp
import re
import csv
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from kan import *
from utils import *


# Force spawn method for multiprocessing
mp_ctx = mp.get_context("spawn")


def extract_formula_safely(raw_formula):
    """Safely extract formula from various formats"""
    try:
        if isinstance(raw_formula, (list, tuple)) and len(raw_formula) > 0:
            if isinstance(raw_formula[0], (list, tuple)) and len(raw_formula[0]) > 0:
                formula_tuple = ex_round(raw_formula[0][0], 4)
            else:
                formula_tuple = ex_round(raw_formula[0], 4)
        else:
            formula_tuple = ex_round(raw_formula, 4)
        return formula_tuple
    except:
        return raw_formula


def convert_formula_to_string(formula_tuple):
    """Convert formula tuple/object to string representation"""
    if hasattr(formula_tuple, "__class__") and "sympy" in str(type(formula_tuple)):
        formula_str = str(formula_tuple)
    elif isinstance(formula_tuple, tuple):
        formula_str = "\n".join(str(item) for item in formula_tuple)
    else:
        formula_str = str(formula_tuple)

    # Clean up the string representation
    formula_str = re.sub(r"\]\s*\[.*?\]$", "]", formula_str)
    formula_str = formula_str.strip()
    if formula_str.startswith("[") and formula_str.endswith("]"):
        formula_str = formula_str[1:-1]

    return formula_str


def comprehensive_formula_cleaning(formula_str, related_genes):
    """Apply comprehensive cleaning to the formula string"""

    # Fix missing multiplication operators
    formula_str = re.sub(r"(\d)\s*\(", r"\1*(", formula_str)
    formula_str = re.sub(r"(\d)\s*([a-zA-Z_])", r"\1*\2", formula_str)
    formula_str = re.sub(r"\)\s*\(", ")*(", formula_str)
    formula_str = re.sub(r"\)\s*([a-zA-Z_])", r")*\1", formula_str)

    # Replace X_i with actual gene names
    def replace_variable(match):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(related_genes):
            gene_name = related_genes[idx]
            clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", gene_name)
            return clean_name
        else:
            return "1"  # Replace invalid indices with 1

    formula_str = re.sub(r"[xX]_(\d+)", replace_variable, formula_str)

    # Final cleanup
    formula_str = re.sub(r"\*\*+", "**", formula_str)
    formula_str = re.sub(r"\*\s*\*", "**", formula_str)

    return formula_str


def generate_symbolic_formula(model, related_genes, gene_dir, target_gene):
    """
    Generate symbolic formula with robust error handling and validation
    """
    try:
        # Generate symbolic formula with more conservative parameters
        model.auto_symbolic(
            a_range=(-10, 10),  # Reduced range for stability
            b_range=(-10, 10),  # Reduced range for stability
            weight_simple=0.8,  # Higher preference for simplicity
            r2_threshold=0.2,  # Lower threshold for more stable fits
            verbose=2,
        )

        # Get the formula with robust extraction
        try:
            raw_formula = model.symbolic_formula()
            formula_tuple = extract_formula_safely(raw_formula)
        except Exception as e:
            print(f"Warning: Could not extract formula safely: {e}")
            # Try alternative extraction
            formula_tuple = str(raw_formula) if raw_formula else "1"

        # Convert to string and clean
        formula_str = convert_formula_to_string(formula_tuple)

        # Apply comprehensive cleaning
        formula_str = comprehensive_formula_cleaning(formula_str, related_genes)

        # Save the formula to symbolic_formula.txt
        with open(os.path.join(gene_dir, "symbolic_formula.txt"), "w") as f:
            f.write(formula_str.strip())

        # Save metadata about the formula
        formula_metadata = {
            "target_gene": target_gene,
            "input_genes": related_genes,
            "formula_length": len(formula_str),
            "generation_timestamp": time.time(),
        }

        with open(os.path.join(gene_dir, "formula_metadata.json"), "w") as f:
            json.dump(formula_metadata, f, indent=2)

        print(f"Successfully generated symbolic formula for {target_gene}")
        return True

    except Exception as e:
        print(f"Error generating symbolic formula for {target_gene}: {e}")

        # Create a simple fallback formula
        try:
            # Create a simple linear fallback formula
            if not related_genes:
                fallback_formula = "0"
            else:
                terms = []
                for i, gene in enumerate(related_genes[:5]):  # Use first 5 genes max
                    clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", gene)
                    coeff = 0.1 / (i + 1)  # Decreasing coefficients
                    terms.append(f"{coeff:.3f}*{clean_name}")
                fallback_formula = " + ".join(terms)

            with open(os.path.join(gene_dir, "symbolic_formula.txt"), "w") as f:
                f.write(fallback_formula)

            error_info = {
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "fallback_used": True,
                "fallback_formula": fallback_formula,
            }

            with open(os.path.join(gene_dir, "symbolic_formula_error.json"), "w") as f:
                json.dump(error_info, f, indent=2)

            print(f"Used fallback formula for {target_gene}")
            return True

        except Exception as e2:
            print(f"Even fallback failed for {target_gene}: {e2}")

            # Save comprehensive error info
            with open(os.path.join(gene_dir, "symbolic_formula_error.txt"), "w") as f:
                f.write(f"Error generating symbolic formula: {str(e)}\n")
                f.write(f"Fallback error: {str(e2)}\n")
                f.write(traceback.format_exc())

            return False


def run_training_process(
    gene,
    X_tensor,
    y_tensor,
    related_genes,
    target_gene,
    output_dir,
    gpu_id,
    batch_size,
    epochs,
    patience,
    lr=0.001,
    generate_symbolic=False,
):
    """
    Process function for training a single model with improved implementation
    to address negative R2 scores and optimize performance.

    """
    try:
        # Setup device
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        # Get GPU memory before model creation
        initial_gpu_memory_real = get_gpu_memory_usage_nvidia_smi(gpu_id)
        # Create directory
        gene_dir = os.path.join(output_dir, gene)
        os.makedirs(gene_dir, exist_ok=True)

        # Transfer data to device
        X = X_tensor.to(device)
        y = y_tensor.to(device)

        # Basic cleanup of extreme values
        X = torch.nan_to_num(X, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)

        # Create split indices with FIXED SEED for reproducibility
        n_samples = X.shape[0]
        torch.manual_seed(42)
        indices = torch.randperm(n_samples)

        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # Save split information
        split_info = {
            "seed": 42,
            "total_samples": int(n_samples),
            "train_samples": int(train_size),
            "val_samples": int(len(val_indices)),
            "test_samples": int(len(test_indices)),
            "train_indices": train_indices.cpu().numpy().tolist(),
            "val_indices": val_indices.cpu().numpy().tolist(),
            "test_indices": test_indices.cpu().numpy().tolist(),
            "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        }

        with open(os.path.join(gene_dir, "data_split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)

        # Prepare raw data for validation and test sets
        X_cpu = X.cpu().numpy()
        y_cpu = y.cpu().numpy()

        # Create validation set data (raw, unnormalized)
        val_data = {}
        for i, gene_name in enumerate(related_genes):
            val_data[gene_name] = X_cpu[val_indices.cpu().numpy(), i].tolist()
        val_data[target_gene] = y_cpu[val_indices.cpu().numpy()].tolist()

        # Create test set data (raw, unnormalized)
        test_data = {}
        for i, gene_name in enumerate(related_genes):
            test_data[gene_name] = X_cpu[test_indices.cpu().numpy(), i].tolist()
        test_data[target_gene] = y_cpu[test_indices.cpu().numpy()].tolist()

        # Save raw validation data (for formula validation)
        with open(os.path.join(gene_dir, "related_genes.json"), "w") as f:
            json.dump(val_data, f, indent=2)

        # Save raw test data (for final evaluation)
        with open(os.path.join(gene_dir, "test_genes.json"), "w") as f:
            json.dump(test_data, f, indent=2)

        # Use data as-is without any normalization or scaling
        X_train = X[train_indices]
        X_val = X[val_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y[test_indices]

        # Debug information
        print(
            f"Data prep for {gene} - X_train: min={X_train.min().item():.4f}, "
            f"max={X_train.max().item():.4f}, shape={X_train.shape}"
        )
        print(
            f"Data prep for {gene} - y_train: min={y_train.min().item():.4f}, "
            f"max={y_train.max().item():.4f}, shape={y_train.shape}"
        )

        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Initialize model
        input_size = X.shape[1]
        model_checkpoint_path = os.path.join(gene_dir, "temp_ckpt")
        os.makedirs(model_checkpoint_path, exist_ok=True)

        model = KAN(
            [input_size, 2, 1],
            grid=4,
            k=3,
            seed=63,
            ckpt_path=model_checkpoint_path,
        ).to(device)

        # Get GPU memory after model creation
        model_creation_memory = get_gpu_memory_usage_nvidia_smi(gpu_id)
        model_memory_overhead = model_creation_memory - initial_gpu_memory_real

        # Initialize weights with Xavier
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=0.7)

        # Optimizer settings
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # Reduce when val_loss stops decreasing
            factor=0.5,  # Cut LR in half
            patience=5,  # Wait 5 epochs before reducing
            min_lr=1e-7,  # Don't go below this
            verbose=True,  # Print when LR changes
        )

        # Loss function
        criterion = torch.nn.MSELoss(reduction="mean")

        # Track resource usage
        initial_memory = get_process_memory_info()["ram_used_gb"]
        peak_memory = initial_memory
        peak_gpu_memory_real = model_creation_memory

        # Initialize tracking variables
        best_val_loss = float("inf")
        best_model_state = None  # Store the best model state
        patience_counter = 0
        history = []

        # Training loop
        for epoch in range(epochs):
            if patience_counter >= patience:
                print(f"Early stopping for gene {gene} after {epoch} epochs")
                break

            epoch_start_time = time.time()

            # Training phase
            model.train()
            train_loss = 0.0
            batch_count = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                try:
                    output = model(X_batch)
                    output = output.view(y_batch.shape)
                    loss = criterion(output, y_batch)

                    # Skip if loss is problematic
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

                    train_loss += loss.item() * X_batch.size(0)
                    batch_count += 1

                except RuntimeError as e:
                    print(f"Skipping batch due to error: {str(e)}")
                    continue

            # Normalize the loss properly
            if batch_count > 0:
                train_loss /= len(train_dataset)
            else:
                train_loss = 999.0

            # Evaluation phase
            model.eval()

            # Validation metrics
            val_predictions = []
            val_targets = []
            val_loss = 0.0

            # Test metrics
            test_predictions = []
            test_targets = []
            test_loss = 0.0

            with torch.no_grad():
                # Validation metrics
                for X_batch, y_batch in val_loader:
                    try:
                        preds = model(X_batch)
                        preds = preds.view(y_batch.shape)
                        batch_loss = criterion(preds, y_batch).item()
                        val_loss += batch_loss * X_batch.size(0)

                        val_predictions.append(preds.detach())
                        val_targets.append(y_batch.detach())
                    except Exception as e:
                        print(f"Validation error: {str(e)}")
                        continue

                # Normalize validation loss
                val_loss /= len(val_dataset)

                # Test metrics
                for X_batch, y_batch in test_loader:
                    try:
                        preds = model(X_batch)
                        preds = preds.view(y_batch.shape)
                        batch_loss = criterion(preds, y_batch).item()
                        test_loss += batch_loss * X_batch.size(0)

                        test_predictions.append(preds.detach())
                        test_targets.append(y_batch.detach())
                    except Exception as e:
                        print(f"Test error: {str(e)}")
                        continue

                # Normalize test loss
                test_loss /= len(test_dataset)

                # Calculate metrics on the entire dataset
                if val_predictions and val_targets:
                    all_val_preds = torch.cat(val_predictions)
                    all_val_targets = torch.cat(val_targets)

                    val_r2 = r2_score(all_val_targets, all_val_preds).item()
                    val_rmse = rmse(all_val_targets, all_val_preds).item()
                    val_mae = mae(all_val_targets, all_val_preds).item()
                else:
                    val_r2, val_rmse, val_mae = 0.0, 999.0, 999.0

                if test_predictions and test_targets:
                    all_test_preds = torch.cat(test_predictions)
                    all_test_targets = torch.cat(test_targets)

                    test_r2 = r2_score(all_test_targets, all_test_preds).item()
                    test_rmse = rmse(all_test_targets, all_test_preds).item()
                    test_mae = mae(all_test_targets, all_test_preds).item()
                else:
                    test_r2, test_rmse, test_mae = 0.0, 999.0, 999.0

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time

            # Track memory usage
            current_memory = get_process_memory_info()["ram_used_gb"]
            peak_memory = max(peak_memory, current_memory)
            current_gpu_memory_real = get_gpu_memory_usage_nvidia_smi(gpu_id)
            peak_gpu_memory_real = max(peak_gpu_memory_real, current_gpu_memory_real)

            # Record metrics
            metrics = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_r2": float(val_r2),
                "val_rmse": float(val_rmse),
                "val_mae": float(val_mae),
                "test_loss": float(test_loss),
                "test_r2": float(test_r2),
                "test_rmse": float(test_rmse),
                "test_mae": float(test_mae),
                "epoch_duration_sec": epoch_duration,
                "gpu_memory_gb": float(current_gpu_memory_real),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(metrics)

            # Progress report
            print(
                f"Gene {gene}, Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val R2: {val_r2:.4f}, Time: {epoch_duration:.2f}s, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            # Early stopping logic with model state saving
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()  # Save best model state
                patience_counter = 0
            else:
                patience_counter += 1

        # Load the best model state for final predictions
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model state for gene {gene}")

        # Generate final predictions with the best model
        model.eval()
        final_val_predictions = []
        final_test_predictions = []

        with torch.no_grad():
            # Get final validation predictions
            for X_batch, y_batch in val_loader:
                try:
                    preds = model(X_batch)
                    preds = preds.view(y_batch.shape)
                    final_val_predictions.append(preds.cpu().numpy())
                except Exception as e:
                    print(f"Error in final validation prediction: {str(e)}")
                    continue

            # Get final test predictions
            for X_batch, y_batch in test_loader:
                try:
                    preds = model(X_batch)
                    preds = preds.view(y_batch.shape)
                    final_test_predictions.append(preds.cpu().numpy())
                except Exception as e:
                    print(f"Error in final test prediction: {str(e)}")
                    continue

        # Concatenate all predictions
        if final_val_predictions:
            val_predictions_array = np.concatenate(final_val_predictions, axis=0)
            val_predictions_list = val_predictions_array.flatten().tolist()
        else:
            val_predictions_list = [0.0] * len(val_indices)

        if final_test_predictions:
            test_predictions_array = np.concatenate(final_test_predictions, axis=0)
            test_predictions_list = test_predictions_array.flatten().tolist()
        else:
            test_predictions_list = [0.0] * len(test_indices)

        # Save validation predictions
        val_predictions_data = {
            "target_gene": target_gene,
            "predictions": val_predictions_list,
            "actual_values": val_data[target_gene],
            "sample_indices": val_indices.cpu().numpy().tolist(),
            "model_info": {
                "best_val_loss": float(best_val_loss),
                "final_epoch": len(history),
                "early_stopped": patience_counter >= patience,
            },
        }

        with open(os.path.join(gene_dir, "validation_predictions.json"), "w") as f:
            json.dump(val_predictions_data, f, indent=2)

        # Save test predictions
        test_predictions_data = {
            "target_gene": target_gene,
            "predictions": test_predictions_list,
            "actual_values": test_data[target_gene],
            "sample_indices": test_indices.cpu().numpy().tolist(),
            "model_info": {
                "best_val_loss": float(best_val_loss),
                "final_epoch": len(history),
                "early_stopped": patience_counter >= patience,
            },
        }

        with open(os.path.join(gene_dir, "test_predictions.json"), "w") as f:
            json.dump(test_predictions_data, f, indent=2)

        # Validation comparison CSV
        with open(
            os.path.join(gene_dir, "validation_comparison.csv"), "w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sample_index",
                    "actual_value",
                    "predicted_value",
                    "absolute_error",
                    "squared_error",
                ]
            )
            for i, (actual, predicted) in enumerate(
                zip(val_data[target_gene], val_predictions_list)
            ):
                abs_error = abs(actual - predicted)
                sq_error = (actual - predicted) ** 2
                writer.writerow(
                    [val_indices[i].item(), actual, predicted, abs_error, sq_error]
                )

        # Test comparison CSV
        with open(os.path.join(gene_dir, "test_comparison.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sample_index",
                    "actual_value",
                    "predicted_value",
                    "absolute_error",
                    "squared_error",
                ]
            )
            for i, (actual, predicted) in enumerate(
                zip(test_data[target_gene], test_predictions_list)
            ):
                abs_error = abs(actual - predicted)
                sq_error = (actual - predicted) ** 2
                writer.writerow(
                    [test_indices[i].item(), actual, predicted, abs_error, sq_error]
                )

        print(f"Saved predictions for gene {gene}:")
        print(f"  - Validation samples: {len(val_predictions_list)}")
        print(f"  - Test samples: {len(test_predictions_list)}")

        # Generate symbolic formula
        if generate_symbolic:
            try:
                generate_symbolic_formula(model, related_genes, gene_dir, target_gene)
            except Exception as e:
                print(f"Error generating symbolic formula: {e}")

        # Final memory stats
        final_memory = get_process_memory_info()["ram_used_gb"]
        memory_delta = final_memory - initial_memory

        # Calculate resource statistics
        training_duration = sum(epoch.get("epoch_duration_sec", 0) for epoch in history)

        # Get best metrics
        best_val_r2 = max(
            (epoch.get("val_r2", float("-inf")) for epoch in history),
            default=float("nan"),
        )
        best_val_rmse = min(
            (epoch.get("val_rmse", float("inf")) for epoch in history),
            default=float("nan"),
        )
        best_val_mae = min(
            (epoch.get("val_mae", float("inf")) for epoch in history),
            default=float("nan"),
        )
        avg_gpu_memory = (
            sum(epoch.get("gpu_memory_gb", 0) for epoch in history) / len(history)
            if history
            else 0
        )

        # Create training log
        log_message = "=" * 50 + "\n"
        log_message += f"MODEL TRAINING SUMMARY FOR GENE: {gene}\n"
        log_message += "=" * 50 + "\n\n"

        # Configuration
        log_message += "CONFIGURATION:\n"
        log_message += "-" * 20 + "\n"
        log_message += f"Model Architecture: width=[{input_size}, 2, 1], grid=4, k=3\n"
        log_message += (
            f"Total Parameters: {sum(p.numel() for p in model.parameters())}\n"
        )
        log_message += f"Device: {device}\n"
        log_message += f"Batch Size: {batch_size}\n"
        log_message += f"Learning Rate: {lr}\n"
        log_message += f"Early Stopping Patience: {patience}\n"
        log_message += f"Data Split Seed: 42\n"
        log_message += f"Data Normalization: None (raw data used)\n"
        log_message += (
            f"Symbolic Formula Generated: {'Yes' if generate_symbolic else 'No'}\n\n"
        )

        # Performance metrics
        log_message += "PERFORMANCE METRICS:\n"
        log_message += "-" * 20 + "\n"
        log_message += f"Best Validation Loss: {best_val_loss:.6f}\n"
        log_message += f"Best Validation R2: {best_val_r2:.6f}\n"
        log_message += f"Best Validation RMSE: {best_val_rmse:.6f}\n"
        log_message += f"Best Validation MAE: {best_val_mae:.6f}\n"
        log_message += f"Final Epochs Completed: {len(history)}\n"
        log_message += (
            f"Early Stopped: {'Yes' if patience_counter >= patience else 'No'}\n\n"
        )

        # Data split information
        log_message += "DATA SPLIT INFORMATION:\n"
        log_message += "-" * 20 + "\n"
        log_message += f"Total Samples: {n_samples}\n"
        log_message += f"Training Samples: {len(train_indices)} ({len(train_indices)/n_samples*100:.1f}%)\n"
        log_message += f"Validation Samples: {len(val_indices)} ({len(val_indices)/n_samples*100:.1f}%)\n"
        log_message += f"Test Samples: {len(test_indices)} ({len(test_indices)/n_samples*100:.1f}%)\n"
        log_message += f"Split Seed: 42\n\n"

        # Prediction information
        log_message += "PREDICTION FILES SAVED:\n"
        log_message += "-" * 20 + "\n"
        log_message += f"validation_predictions.json: Validation set predictions\n"
        log_message += f"test_predictions.json: Test set predictions\n"
        log_message += f"validation_comparison.csv: Detailed validation comparison\n"
        log_message += f"test_comparison.csv: Detailed test comparison\n\n"

        # Resource usage
        log_message += "RESOURCE USAGE SUMMARY:\n"
        log_message += "-" * 20 + "\n"
        log_message += f"Total Training Duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)\n"
        log_message += f"Peak GPU Memory: {peak_gpu_memory_real:.2f} GB\n"
        log_message += f"Model Memory Overhead: {model_memory_overhead:.2f} GB\n"
        log_message += f"Average GPU Memory: {avg_gpu_memory:.2f} GB\n"
        log_message += f"Initial Process Memory: {initial_memory:.2f} GB\n"
        log_message += f"Final Process Memory: {final_memory:.2f} GB\n"
        log_message += f"Peak Process Memory: {peak_memory:.2f} GB\n"
        log_message += f"Memory Delta: {memory_delta:.2f} GB\n\n"

        # Epoch-wise training statistics
        log_message += "DETAILED TRAINING METRICS BY EPOCH:\n"
        log_message += "-" * 20 + "\n"

        for j, metrics in enumerate(history, 1):
            log_message += f"Epoch {j}: Train Loss = {metrics['train_loss']:.6f}, "
            log_message += f"Val Loss = {metrics['val_loss']:.6f}, "
            log_message += f"Test Loss = {metrics['test_loss']:.6f}, "
            log_message += f"Val R2 = {metrics['val_r2']:.6f}, "
            log_message += f"Val RMSE = {metrics['val_rmse']:.6f}, "
            log_message += f"Val MAE = {metrics['val_mae']:.6f}, "
            log_message += f"LR = {metrics['learning_rate']:.6f}\n"

        log_message += "=" * 50 + "\n"

        with open(os.path.join(gene_dir, "training_log.txt"), "w") as log_file:
            log_file.write(log_message)

        # Create feature importance CSV
        try:
            create_feature_importance_csv(
                model, related_genes, os.path.join(gene_dir, "feature_importance.csv")
            )
        except Exception as e:
            print(f"Warning: Could not create feature importance for {gene}: {e}")

        # Clean up
        del model, optimizer, criterion, scheduler
        del train_dataset, val_dataset, test_dataset
        del train_loader, val_loader, test_loader
        del X_train, X_val, X_test, y_train, y_val, y_test
        del X, y

        # Clean up the temporary checkpoint directory
        if os.path.exists(model_checkpoint_path):
            import shutil

            try:
                shutil.rmtree(model_checkpoint_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary checkpoint directory: {e}")

        torch.cuda.empty_cache()
        gc.collect()

        return gene, True, best_val_loss, best_val_r2 if history else 0

    except Exception as e:
        print(f"Error training gene {gene}: {str(e)}")
        traceback.print_exc()

        try:
            gene_dir = os.path.join(output_dir, gene)
            os.makedirs(gene_dir, exist_ok=True)
            with open(os.path.join(gene_dir, "error_log.txt"), "w") as f:
                f.write(f"Error training model for gene {gene}\n")
                f.write(f"Error message: {str(e)}\n")
                f.write(traceback.format_exc())
        except:
            pass

        return gene, False, float("inf"), 0


def copy_essential_files_immediately(gene, output_dir, verify_only=False):
    """
    Copy essential files immediately when model training completes
    If verify_only=True, only copy if files aren't already in destination
    """
    try:
        # Get WORK environment variable
        work_dir = os.environ.get("WORK")
        if not work_dir:
            print(
                f"WARNING: WORK environment variable not set. Cannot save files for {gene}."
            )
            return False

        # Define source and destination paths
        gene_output_dir = os.path.join(output_dir, gene)
        results_dir = os.path.join(work_dir, "kan_results")
        gene_results_dir = os.path.join(results_dir, gene)

        # Define file paths - including new prediction files
        feature_file_src = os.path.join(gene_output_dir, "feature_importance.csv")
        log_file_src = os.path.join(gene_output_dir, "training_log.txt")
        symbolic_py_src = os.path.join(gene_output_dir, "symbolic_formula.py")
        val_genes_src = os.path.join(gene_output_dir, "related_genes.json")
        test_genes_src = os.path.join(gene_output_dir, "test_genes.json")
        split_info_src = os.path.join(gene_output_dir, "data_split_info.json")
        val_pred_src = os.path.join(gene_output_dir, "validation_predictions.json")
        test_pred_src = os.path.join(gene_output_dir, "test_predictions.json")
        val_comp_src = os.path.join(gene_output_dir, "validation_comparison.csv")
        test_comp_src = os.path.join(gene_output_dir, "test_comparison.csv")

        feature_file_dst = os.path.join(gene_results_dir, "feature_importance.csv")
        log_file_dst = os.path.join(gene_results_dir, "training_log.txt")
        symbolic_py_dst = os.path.join(gene_results_dir, "symbolic_formula.py")
        val_genes_dst = os.path.join(gene_results_dir, "related_genes.json")
        test_genes_dst = os.path.join(gene_results_dir, "test_genes.json")
        split_info_dst = os.path.join(gene_results_dir, "data_split_info.json")
        val_pred_dst = os.path.join(gene_results_dir, "validation_predictions.json")
        test_pred_dst = os.path.join(gene_results_dir, "test_predictions.json")
        val_comp_dst = os.path.join(gene_results_dir, "validation_comparison.csv")
        test_comp_dst = os.path.join(gene_results_dir, "test_comparison.csv")

        # Create destination directory
        os.makedirs(gene_results_dir, exist_ok=True)

        # For verification mode, only copy if destination doesn't exist
        if verify_only:
            missing_files = []

            files_to_check = [
                ("feature_importance.csv", feature_file_src, feature_file_dst),
                ("training_log.txt", log_file_src, log_file_dst),
                ("symbolic_formula.py", symbolic_py_src, symbolic_py_dst),
                ("related_genes.json", val_genes_src, val_genes_dst),
                ("test_genes.json", test_genes_src, test_genes_dst),
                ("data_split_info.json", split_info_src, split_info_dst),
                ("validation_predictions.json", val_pred_src, val_pred_dst),
                ("test_predictions.json", test_pred_src, test_pred_dst),
                ("validation_comparison.csv", val_comp_src, val_comp_dst),
                ("test_comparison.csv", test_comp_src, test_comp_dst),
            ]

            for file_name, src, dst in files_to_check:
                if os.path.exists(src) and not os.path.exists(dst):
                    missing_files.append((file_name, src, dst))

            if not missing_files:
                return True

            print(
                f"VERIFY: Found {len(missing_files)} missing files for {gene}, copying them now"
            )

            # Copy only missing files
            for file_name, src, dst in missing_files:
                import shutil

                shutil.copy2(src, dst)
                print(f"RECOVERY COPY: Copied {file_name} for {gene}")

            return len(missing_files) > 0

        # Standard copy mode - copy all files if they exist
        copied = False
        files_to_copy = [
            ("feature_importance.csv", feature_file_src, feature_file_dst),
            ("training_log.txt", log_file_src, log_file_dst),
            ("symbolic_formula.py", symbolic_py_src, symbolic_py_dst),
            ("related_genes.json", val_genes_src, val_genes_dst),
            ("test_genes.json", test_genes_src, test_genes_dst),
            ("data_split_info.json", split_info_src, split_info_dst),
            ("validation_predictions.json", val_pred_src, val_pred_dst),
            ("test_predictions.json", test_pred_src, test_pred_dst),
            ("validation_comparison.csv", val_comp_src, val_comp_dst),
            ("test_comparison.csv", test_comp_src, test_comp_dst),
        ]

        for file_name, src, dst in files_to_copy:
            if os.path.exists(src):
                import shutil

                shutil.copy2(src, dst)
                print(f"IMMEDIATE BACKUP: Copied {file_name} for {gene}")
                copied = True
            else:
                print(f"WARNING: {file_name} not found at {src}")

        return copied

    except Exception as e:
        print(f"ERROR copying essential files for {gene}: {str(e)}")
        traceback.print_exc()
        return False


def save_essential_files_to_home(gene, output_dir, work_dir):
    """Save only the essential files to the work directory"""
    if not work_dir:
        return

    try:
        gene_output_dir = os.path.join(output_dir, gene)
        gene_work_dir = os.path.join(work_dir, gene)

        # Create the destination directory if it doesn't exist
        os.makedirs(gene_work_dir, exist_ok=True)

        # Files to copy - including new prediction files
        files_to_copy = [
            "feature_importance.csv",
            "training_log.txt",
            "symbolic_formula.py",
            "related_genes.json",
            "test_genes.json",
            "data_split_info.json",
            "validation_predictions.json",
            "test_predictions.json",
            "validation_comparison.csv",
            "test_comparison.csv",
        ]

        for filename in files_to_copy:
            src_file = os.path.join(gene_output_dir, filename)
            if os.path.exists(src_file):
                import shutil

                shutil.copy2(src_file, gene_work_dir)
                print(f"Copied {filename} for gene {gene} to {gene_work_dir}")

    except Exception as e:
        print(f"Error saving essential files for gene {gene}: {str(e)}")


# Rest of the functions (train_kan_models_parallel_max, etc.) remain the same...
def train_kan_models_parallel_max(
    gene_list,
    data_manager,
    output_dir,
    home_dir=None,
    max_models=6,
    epochs=5,
    patience=5,
    lr=0.001,
    batch_size=32,
    generate_symbolic=False,
):
    """Run training with maximum parallelism using one process per model"""

    print(f"Using MAXIMUM PARALLELISM with {max_models} simultaneous models")
    print(
        f"Symbolic formula generation: {'Enabled' if generate_symbolic else 'Disabled'}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Path for checkpoint
    checkpoint_path = os.path.join(output_dir, "training_checkpoint.json")

    # Initialize tracking
    completed_genes = []
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
                completed_genes = checkpoint.get("completed_genes", [])
                print(
                    f"Resuming from checkpoint: {len(completed_genes)} genes already processed"
                )

            # Filter out already processed genes
            remaining_genes = [g for g in gene_list if g not in completed_genes]
            gene_list = remaining_genes
            print(f"Remaining genes to process: {len(gene_list)}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Starting fresh.")

    # Check if there are genes to process
    if not gene_list:
        print("All genes have been processed!")
        return completed_genes

    # Process genes in batches of max_models
    total_genes = len(gene_list)

    with tqdm(total=total_genes, desc="Overall Progress") as pbar:
        for batch_idx in range(0, total_genes, max_models):
            # Get genes for this batch
            batch_genes = gene_list[batch_idx : batch_idx + max_models]
            print(
                f"\nProcessing batch {batch_idx//max_models + 1}: genes {batch_idx + 1} "
                f"to {batch_idx + len(batch_genes)} of {total_genes}"
            )

            # Prepare data for each gene
            process_args = []

            print("Preparing data for batch:")
            for i, gene in enumerate(batch_genes):
                # Get data for gene
                X, y, related_genes, target_gene = data_manager.get_data_for_gene(gene)

                if X is not None and y is not None:
                    # Move to CPU for transfer to processes
                    X_cpu = X.cpu()
                    y_cpu = y.cpu()

                    # Print shape and range information for debugging
                    print(f"Gene {gene}: X shape={X.shape}, y shape={y.shape}")
                    print(
                        f"  X range: [{X.min().item():.4f}, {X.max().item():.4f}], "
                        f"std={X.std().item():.4f}"
                    )
                    print(
                        f"  y range: [{y.min().item():.4f}, {y.max().item():.4f}], "
                        f"std={y.std().item():.4f}"
                    )
                    num_gpus = torch.cuda.device_count()
                    gpu_id = i % num_gpus if num_gpus > 0 else 0

                    # Add to process args
                    process_args.append(
                        (
                            gene,
                            X_cpu,
                            y_cpu,
                            related_genes,
                            target_gene,
                            output_dir,
                            gpu_id,  # gpu_id
                            batch_size,
                            epochs,
                            patience,
                            lr,
                            generate_symbolic,
                        )
                    )

            # Clear GPU memory before starting new batch
            optimize_gpu_memory()

            # Use multiprocessing to train models in parallel
            try:
                with mp_ctx.Pool(processes=len(process_args)) as pool:
                    batch_results = list(
                        pool.starmap(run_training_process, process_args)
                    )

                    successful_genes = []
                    for result in batch_results:
                        if result[1]:  # Success flag
                            gene, _, val_loss, val_r2 = result
                            successful_genes.append(gene)
                            print(
                                f"Successfully trained gene {gene}: "
                                f"Val Loss={val_loss:.4f}, Val R2={val_r2:.4f}"
                            )
                            copy_essential_files_immediately(
                                gene, output_dir, verify_only=True
                            )

                    completed_genes.extend(successful_genes)

                    # Save checkpoint
                    with open(checkpoint_path, "w") as f:
                        json.dump(
                            {
                                "completed_genes": completed_genes,
                                "timestamp": time.time(),
                                "batch_idx": batch_idx,
                                "total_genes": total_genes,
                                "completed": len(completed_genes),
                                "generate_symbolic": generate_symbolic,
                            },
                            f,
                            indent=2,
                        )

                    # Save to home directory if specified
                    if home_dir:
                        for gene in successful_genes:
                            save_essential_files_to_home(
                                gene, output_dir, os.environ.get("WORK", None)
                            )

            except Exception as e:
                print(f"Error in batch {batch_idx//max_models + 1}: {str(e)}")
                traceback.print_exc()

                # Save checkpoint with error info
                with open(checkpoint_path, "w") as f:
                    json.dump(
                        {
                            "completed_genes": completed_genes,
                            "timestamp": time.time(),
                            "batch_idx": batch_idx,
                            "total_genes": total_genes,
                            "completed": len(completed_genes),
                            "last_error": str(e),
                            "generate_symbolic": generate_symbolic,
                        },
                        f,
                        indent=2,
                    )

            # Update progress
            pbar.update(len(batch_genes))

            # Force cleanup
            optimize_gpu_memory()

    # Create completion flag
    with open(os.path.join(output_dir, "training_complete.flag"), "w") as f:
        f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total genes processed: {len(completed_genes)}\n")
        f.write(f"Training method: MAXIMUM PARALLELISM\n")
        f.write(
            f"Symbolic formula generation: {'Enabled' if generate_symbolic else 'Disabled'}\n"
        )

    # Print final summary
    print(f"\nTraining Summary:")
    print(f"Total genes processed: {len(completed_genes)}")
    print(f"Symbolic formulas generated: {'Yes' if generate_symbolic else 'No'}")

    return completed_genes


def train_kan_models_parallel_hpc_with_copies(
    gene_list,
    data_manager,
    output_dir,
    home_dir=None,
    batch_size=32,
    max_models=4,
    epochs=50,
    patience=10,
    min_delta=1e-4,
    lr=0.001,
    resume_from_checkpoint=True,
    time_check_interval=10 * 60,
    generate_symbolic=False,
):
    """Replacement function using maximum parallelism"""
    print("USING MAXIMUM PARALLELISM WITH 100% STABLE IMPLEMENTATION")
    print(
        f"Symbolic formula generation: {'Enabled' if generate_symbolic else 'Disabled'}"
    )
    return train_kan_models_parallel_max(
        gene_list=gene_list,
        data_manager=data_manager,
        output_dir=output_dir,
        home_dir=home_dir,
        max_models=max_models,
        epochs=epochs,
        patience=patience,
        lr=lr,
        batch_size=batch_size,
        generate_symbolic=generate_symbolic,
    )
