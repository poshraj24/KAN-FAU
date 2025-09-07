import os
import re
from pathlib import Path


def main():
    """
    Main function to run the analysis
    """
    base_directory = "kan_models"
    base_path = Path(base_directory)

    print(f"Looking in directory: {base_path.absolute()}")

    # Check if directory exists
    if not base_path.exists():
        print(f"ERROR: Directory {base_directory} does not exist!")
        return

    all_training_losses = []
    all_validation_losses = []
    all_validation_r2 = []
    all_parameters = []
    all_training_times = []
    all_gpu_memory = []

    processed_count = 0

    # Process each Gene folder
    for gene_folder in base_path.iterdir():
        if gene_folder.is_dir():
            log_file = gene_folder / "training_log.txt"

            if log_file.exists():
                processed_count += 1
                print(f"Processing: {gene_folder.name}")

                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Show debug for first file
                    if processed_count == 1:
                        print("\n=== DEBUG FIRST FILE ===")
                        print(f"File: {log_file}")
                        print(f"Content length: {len(content)} chars")

                        # Check for training losses
                        train_matches = re.findall(
                            r"Train Loss = ([0-9]+\.?[0-9]*)", content
                        )
                        print(f"Training loss matches found: {len(train_matches)}")
                        if train_matches:
                            print(f"First 3 training losses: {train_matches[:3]}")
                            best_train = min(float(x) for x in train_matches)
                            print(f"Best training loss: {best_train}")

                        # Check for validation loss
                        val_match = re.search(
                            r"Best Validation Loss: ([0-9]+\.?[0-9]*)", content
                        )
                        if val_match:
                            print(f"Best validation loss: {val_match.group(1)}")
                        else:
                            print("No 'Best Validation Loss' found")

                        print("=== END DEBUG ===\n")

                    # Extract training loss (best from all epochs)
                    train_loss_matches = re.findall(
                        r"Train Loss = ([0-9]+\.?[0-9]*)", content
                    )
                    if train_loss_matches:
                        train_losses = [float(loss) for loss in train_loss_matches]
                        best_training_loss = min(train_losses)
                        all_training_losses.append(best_training_loss)

                    # Extract validation loss (from summary)
                    val_loss_match = re.search(
                        r"Best Validation Loss: ([0-9]+\.?[0-9]*)", content
                    )
                    if val_loss_match:
                        all_validation_losses.append(float(val_loss_match.group(1)))

                    # Extract validation R2
                    val_r2_match = re.search(
                        r"Best Validation R2: ([-]?[0-9]+\.?[0-9]*)", content
                    )
                    if val_r2_match:
                        all_validation_r2.append(float(val_r2_match.group(1)))

                    # Extract total parameters
                    params_match = re.search(r"Total Parameters: ([0-9,]+)", content)
                    if params_match:
                        all_parameters.append(
                            int(params_match.group(1).replace(",", ""))
                        )

                    # Extract training time
                    time_match = re.search(
                        r"Total Training Duration: ([0-9]+\.?[0-9]*) seconds", content
                    )
                    if time_match:
                        all_training_times.append(float(time_match.group(1)))

                    # Extract peak GPU memory
                    gpu_match = re.search(
                        r"Peak GPU Memory: ([0-9]+\.?[0-9]*) GB", content
                    )
                    if gpu_match:
                        all_gpu_memory.append(float(gpu_match.group(1)))

                except Exception as e:
                    print(f"Error processing {gene_folder.name}: {e}")

    print(f"\nProcessed {processed_count} files")

    # Calculate and display averages
    print(f"\nData collected:")
    print(f"Training losses: {len(all_training_losses)} values")
    print(f"Validation losses: {len(all_validation_losses)} values")
    print(f"Validation R2: {len(all_validation_r2)} values")

    if len(all_training_losses) > 0:
        print(f"Sample training losses: {all_training_losses[:3]}")
    if len(all_validation_losses) > 0:
        print(f"Sample validation losses: {all_validation_losses[:3]}")

    print("\nAVERAGES:")
    print("-" * 20)

    if all_training_losses:
        avg_train = sum(all_training_losses) / len(all_training_losses)
        print(f"Training Loss: {avg_train:.6f}")
    else:
        print("Training Loss: No data")

    if all_validation_losses:
        avg_val = sum(all_validation_losses) / len(all_validation_losses)
        print(f"Validation Loss: {avg_val:.6f}")
    else:
        print("Validation Loss: No data")

    if all_validation_r2:
        avg_r2 = sum(all_validation_r2) / len(all_validation_r2)
        print(f"Validation R2: {avg_r2:.6f}")
    else:
        print("Validation R2: No data")

    if all_parameters:
        avg_params = sum(all_parameters) / len(all_parameters)
        print(f"Total Parameters: {avg_params:.6f}")
    else:
        print("Total Parameters: No data")

    if all_training_times:
        avg_time = sum(all_training_times) / len(all_training_times)
        print(f"Training Time: {avg_time:.6f}")
    else:
        print("Training Time: No data")

    if all_gpu_memory:
        avg_gpu = sum(all_gpu_memory) / len(all_gpu_memory)
        print(f"Peak Gpu Memory: {avg_gpu:.6f}")
    else:
        print("Peak Gpu Memory: No data")


if __name__ == "__main__":
    main()
