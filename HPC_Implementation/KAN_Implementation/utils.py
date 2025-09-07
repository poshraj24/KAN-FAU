import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import os
import psutil
import time
import threading
from collections import deque
import shutil


# resource monitoring class
class ResourceMonitor:
    def __init__(self, interval=1.0, history_size=60):
        """
        Monitor system resources (CPU, RAM, GPU)

        Args:
            interval: Sampling interval in seconds
            history_size: Number of samples to keep in history
        """
        self.interval = interval
        self.history_size = history_size

        # Initialize history containers
        self.cpu_percent_history = deque(maxlen=history_size)
        self.ram_percent_history = deque(maxlen=history_size)
        self.ram_used_history = deque(maxlen=history_size)

        # For GPU tracking
        self.gpu_memory_used_history = {}
        self.gpu_memory_total = {}

        # Monitoring state
        self.running = False
        self.monitor_thread = None

        # Get number of GPUs
        self.num_gpus = torch.cuda.device_count()
        for i in range(self.num_gpus):
            self.gpu_memory_used_history[i] = deque(maxlen=history_size)
            _, total = torch.cuda.mem_get_info(i)
            self.gpu_memory_total[i] = total / 1e9  # Convert to GB

    def start(self):
        """Start the monitoring thread"""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.interval * 2)

    def _monitor_resources(self):
        """Resource monitoring loop"""
        while self.running:
            # CPU usage
            self.cpu_percent_history.append(psutil.cpu_percent())

            # RAM usage
            memory = psutil.virtual_memory()
            self.ram_percent_history.append(memory.percent)
            self.ram_used_history.append(memory.used / (1024**3))  # Convert to GB

            # GPU memory for each GPU
            for i in range(self.num_gpus):
                try:
                    # Get current GPU memory usage
                    torch.cuda.synchronize(i)
                    used_memory = torch.cuda.memory_reserved(i) / 1e9  # Convert to GB
                    self.gpu_memory_used_history[i].append(used_memory)
                except Exception as e:
                    print(f"Error getting GPU {i} memory: {e}")

            # Wait for next sampling
            time.sleep(self.interval)

    def get_summary(self):
        """Get resource usage summary"""
        if not self.cpu_percent_history:
            return {
                "cpu": {"current": 0, "average": 0, "max": 0},
                "ram": {"current_percent": 0, "current_gb": 0, "max_gb": 0},
                "gpu": {},
            }

        # Create summary dictionary
        summary = {
            "cpu": {
                "current": self.cpu_percent_history[-1],
                "average": sum(self.cpu_percent_history)
                / len(self.cpu_percent_history),
                "max": max(self.cpu_percent_history),
            },
            "ram": {
                "current_percent": self.ram_percent_history[-1],
                "current_gb": self.ram_used_history[-1],
                "max_gb": max(self.ram_used_history),
            },
            "gpu": {},
        }

        # Add GPU info
        for i in range(self.num_gpus):
            if self.gpu_memory_used_history[i]:
                summary["gpu"][i] = {
                    "current_gb": self.gpu_memory_used_history[i][-1],
                    "max_gb": max(self.gpu_memory_used_history[i]),
                    "total_gb": self.gpu_memory_total[i],
                }

        return summary


class HPCSharedGeneDataset(Dataset):
    """Dataset class that works with shared GPU data"""

    def __init__(self, X, y, indices=None):
        """
        Initialize dataset with GPU-resident tensors

        Args:
            X: Feature tensor (already on GPU)
            y: Target tensor (already on GPU)
            indices: Optional indices for train/val/test split
        """
        self.X = X  # Already on GPU
        self.y = y  # Already on GPU
        self.indices = indices

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.X)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        return self.X[idx], self.y[idx]


def prepare_hpc_data_loaders(X, y, batch_size=32, seed=42, num_workers=None):
    """
    Create optimized DataLoaders from shared GPU tensors

    Args:
        X: Feature tensor (already on GPU)
        y: Target tensor (already on GPU)
        batch_size: Batch size for training
        seed: Random seed for reproducibility
        num_workers: Number of worker processes, if None will auto-detect

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed
    torch.manual_seed(int(time.time()))
    np.random.seed(int(time.time()))

    # Generate indices for train/val/test split (80/10/10)
    indices = torch.randperm(len(X))
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create datasets using the shared tensors
    train_dataset = HPCSharedGeneDataset(X, y, train_indices)
    val_dataset = HPCSharedGeneDataset(X, y, val_indices)
    test_dataset = HPCSharedGeneDataset(X, y, test_indices)

    # Determine optimal number of workers
    if num_workers is None:
        # Use environment variable if set
        if "DATA_LOADER_WORKERS" in os.environ:
            num_workers = int(os.environ["DATA_LOADER_WORKERS"])
        else:
            # Otherwise use half of available cores (common HPC practice)
            num_workers = max(1, multiprocessing.cpu_count() // 2)

    # Create optimized DataLoaders

    # persistent_workers=True for HPC**
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        # persistent_workers=True if num_workers > 0 else False, **uncomment this when implementing on HPC
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        # persistent_workers=True if num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # change to num_workers for HPC in all loaders
        pin_memory=False,
        # persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, test_loader


# Metrics calculation functions
def r2_score(y_true, y_pred):
    """Calculate R^2 (coefficient of determination) score."""
    # Ensure tensors have compatible shapes
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # Handle edge case where ss_tot is very small
    if ss_tot < 1e-8:
        return torch.tensor(0.0).to(y_true.device)

    return 1 - ss_res / ss_tot


def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    # Ensure tensors have compatible shapes
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    # Ensure tensors have compatible shapes
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    return torch.mean(torch.abs(y_true - y_pred))


def log_symbolic_formula(formula_tuple, log_file, gene_names):
    """
    Log symbolic formula to training log file with actual gene names

    Args:
        formula_tuple: The symbolic formula from model.symbolic_formula()
        log_file: Path to the log file
        gene_names: List of gene names corresponding to input features
    """
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("Symbolic Formula with Gene Names:\n")
        f.write("-" * 20 + "\n")

        # Convert tuple elements to string
        formula_str = "\n".join(str(item) for item in formula_tuple)

        # Replace X_i patterns with actual gene names

        import re

        replaced_formula = formula_str
        pattern = re.compile(r"[xX]_(\d+)")

        def replace_with_gene(match):
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(gene_names):
                return f"{gene_names[idx]}"
            else:
                return match.group(0)

        replaced_formula = pattern.sub(replace_with_gene, formula_str)

        f.write(replaced_formula + "\n")
        f.write("=" * 50 + "\n\n")

        f.write("Variable Mapping:\n")
        for i, gene in enumerate(gene_names):
            if i < len(gene_names):
                f.write(f"X_{i+1} = {gene}\n")
        f.write("=" * 50 + "\n\n")


def get_process_memory_info():
    """Get current memory usage for the current process in GB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "ram_used_gb": mem_info.rss / (1024**3),  # Convert to GB
        "virtual_memory_gb": mem_info.vms / (1024**3),  # Virtual memory in GB
    }


def get_system_memory_info():
    """Get system-wide memory usage information in GB"""
    mem = psutil.virtual_memory()
    return {
        "total_ram_gb": mem.total / (1024**3),
        "available_ram_gb": mem.available / (1024**3),
        "used_ram_gb": mem.used / (1024**3),
        "ram_percent": mem.percent,
    }


def get_cpu_info():
    """Get CPU usage information"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "cpu_count": psutil.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
    }


def get_gpu_info(device):
    """Get GPU memory information for the current device"""
    if not torch.cuda.is_available():
        return {
            "gpu_memory_allocated_gb": 0,
            "gpu_memory_reserved_gb": 0,
            "gpu_memory_total_gb": 0,
        }

    torch.cuda.synchronize(device)

    # Get memory information
    allocated = torch.cuda.memory_allocated(device) / 1e9  # Convert to GB
    reserved = torch.cuda.memory_reserved(device) / 1e9  # Convert to GB

    # Get total GPU memory if available
    total = 0
    try:
        free, total_bytes = torch.cuda.mem_get_info(device)
        total = total_bytes / 1e9  # Convert to GB
    except:

        total = -1

    return {
        "gpu_memory_allocated_gb": allocated,
        "gpu_memory_reserved_gb": reserved,
        "gpu_memory_total_gb": total,
    }


def track_resources(resource_dict, device):
    """
    Update resource tracking dictionary with current resource usage

    Args:
        resource_dict: Dictionary to update
        device: GPU device to monitor

    Returns:
        Updated resource dictionary
    """
    # Get current process memory
    proc_memory = get_process_memory_info()

    # Update maximum values
    resource_dict["ram_used_gb_max"] = max(
        resource_dict["ram_used_gb_max"], proc_memory["ram_used_gb"]
    )
    resource_dict["virtual_memory_gb_max"] = max(
        resource_dict["virtual_memory_gb_max"], proc_memory["virtual_memory_gb"]
    )

    # Store current values for averaging
    resource_dict["ram_used_gb_avg"].append(proc_memory["ram_used_gb"])

    # Update GPU tracking
    gpu_info = get_gpu_info(device)
    resource_dict["gpu_memory_gb_max"] = max(
        resource_dict["gpu_memory_gb_max"], gpu_info["gpu_memory_allocated_gb"]
    )
    resource_dict["gpu_memory_gb_avg"].append(gpu_info["gpu_memory_allocated_gb"])

    # Store current timestamp
    resource_dict["timestamps"].append(time.time())

    return resource_dict


def save_essential_files_to_home(gene, tmp_dir, home_dir):
    """
    Save only the essential files to $HOME directory:
    - training_log.txt
    - symbolic_formula.txt
    - feature_importance.csv
    - model .pt file

    Args:
        gene: Name of the gene
        tmp_dir: Temporary directory with all files
        home_dir: $HOME directory to save essential files
    """
    gene_tmp_dir = os.path.join(tmp_dir, gene)
    gene_home_dir = os.path.join(home_dir, gene)

    # Create gene directory in home if it doesn't exist
    os.makedirs(gene_home_dir, exist_ok=True)

    # List of essential files to copy
    essential_files = [
        "training_log.txt",
        "symbolic_formula.txt",
        "feature_importance.csv",
    ]

    # Copy model file
    if os.path.exists(os.path.join(gene_tmp_dir, "best_model.pt")):
        essential_files.append("best_model.pt")
    elif os.path.exists(os.path.join(gene_tmp_dir, "final_model.pt")):
        essential_files.append("final_model.pt")

    # Copy each file if it exists
    for filename in essential_files:
        src_file = os.path.join(gene_tmp_dir, filename)
        if os.path.exists(src_file):
            dst_file = os.path.join(gene_home_dir, filename)
            try:
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"Warning: Failed to copy {src_file} to {dst_file}: {str(e)}")

    print(f" Saved essential files for gene {gene} to {gene_home_dir}")


def create_feature_importance_csv(model, gene_names, output_path):
    """
    Create a CSV file with feature importance scores

    Args:
        model: Trained KAN model
        gene_names: List of gene names
        output_path: Path to save the CSV file
    """
    try:
        # Set model to evaluation mode
        model.eval()

        # Get feature importance scores
        feature_scores = model.feature_score.cpu().detach().numpy()

        # Map scores to gene names
        gene_importance = {}
        for i, gene in enumerate(gene_names):
            if i < len(feature_scores):
                gene_importance[gene] = float(feature_scores[i])

        # # Normalize scores
        # total = sum(abs(v) for v in gene_importance.values())
        # if total > 0:
        #     gene_importance = {k: abs(v) / total for k, v in gene_importance.items()}

        # Save as CSV
        with open(output_path, "w") as f:
            f.write("Gene,Importance\n")
            for gene, score in gene_importance.items():
                f.write(f"{gene},{score}\n")

        print(f"Feature importance saved to {output_path}")
        return gene_importance
    except Exception as e:
        print(f"Error creating feature importance CSV: {str(e)}")
        return {}


def optimize_gpu_memory():
    """Optimize GPU memory by clearing unused memory"""
    # Clear PyTorch's CUDA cache
    torch.cuda.empty_cache()
    # Force garbage collection
    import gc

    gc.collect()


def get_available_gpu_memory():
    """Get available memory for each GPU in GB"""
    available_memory = {}
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        available_memory[i] = free / 1e9  # Convert to GB
    return available_memory


def distribute_models_to_gpus(gene_batch, num_gpus):
    """
    Distribute models across available GPUs based on available memory

    Args:
        gene_batch: List of genes to process
        num_gpus: Number of available GPUs

    Returns:
        List of (gene, gpu_id) pairs
    """
    if num_gpus == 0:
        return [(gene, 0) for gene in gene_batch]  # Use CPU (device 0)

    # Get available memory per GPU
    available_memory = get_available_gpu_memory()

    # Sort GPUs by available memory
    sorted_gpus = sorted(
        available_memory.keys(), key=lambda x: available_memory[x], reverse=True
    )

    # Distribute genes across GPUs
    assignments = []
    for i, gene in enumerate(gene_batch):
        # Assign to GPU with most available memory
        gpu_id = sorted_gpus[i % len(sorted_gpus)]
        assignments.append((gene, gpu_id))

    # Print distribution
    gpu_counts = {}
    for _, gpu_id in assignments:
        gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1

    print("GPU assignment distribution:")
    for gpu_id, count in gpu_counts.items():
        print(f"GPU {gpu_id}: {count} models")

    return assignments


def get_gpu_memory_usage_nvidia_smi(gpu_id=0):
    """Get actual GPU memory usage using nvidia-smi"""
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
        )

        return float(result.stdout.strip()) / 1024  # Convert MB to GB
    except:
        return 0.0


def get_total_gpu_memory(gpu_id=0):
    """Get total GPU memory using nvidia-smi"""
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.total",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
        )

        return float(result.stdout.strip()) / 1024  # Convert MB to GB
    except:
        return 0.0


def measure_model_memory_overhead(input_size, batch_size=32, device="cuda:0"):
    """Measure the actual memory required for a single KAN model"""
    import subprocess
    from kan import KAN

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Get initial memory usage
    initial_memory = get_gpu_memory_usage_nvidia_smi(int(device.split(":")[-1]))

    # Create model and move to GPU
    model = KAN([input_size, 2, 1], grid=5, k=4, seed=42).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Create dummy data and do forward-backward pass
    dummy_data = torch.randn(batch_size, input_size).to(device)
    dummy_target = torch.randn(batch_size, 1).to(device)

    output = model(dummy_data)
    loss = torch.nn.MSELoss()(output.squeeze(), dummy_target.squeeze())
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()

    # Get final memory usage
    final_memory = get_gpu_memory_usage_nvidia_smi(int(device.split(":")[-1]))

    # Cleanup
    del model, optimizer, dummy_data, dummy_target, output, loss
    torch.cuda.empty_cache()

    memory_overhead = final_memory - initial_memory

    return memory_overhead
