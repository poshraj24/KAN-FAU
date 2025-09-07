
#!/bin/bash -l
#SBATCH --gres=gpu:a100:1 -p a100
#SBATCH --time=01:00:00
#SBATCH --job-name=kan_train
#SBATCH --output=kan_train_%j.out
#SBATCH --error=kan_train_%j.err
#SBATCH --export=NONE

# Exit on error
set -e

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

# Set up directories based on HPC file system
SUBMIT_DIR=$SLURM_SUBMIT_DIR
WORK_DIR=$WORK
TMP_DIR=$TMPDIR
RESULTS_DIR=${WORK}/kan_results1

# Create directories
echo "Creating directories..."
mkdir -p $TMP_DIR/Data
mkdir -p $TMP_DIR/kan_models
mkdir -p $RESULTS_DIR

# Use the pytorch module which has Python+PyTorch pre-installed
echo "Setting up environment..."
module purge
module load python/pytorch2.6py3.12
module load cuda

# Set up proxy for internet access
echo "Setting up proxy for package installation..."
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# Install required packages with proxy
echo "Installing required packages..."
pip install --user pykan numpy scikit-learn pandas h5py matplotlib setuptools tqdm pandas seaborn pyyaml scanpy
echo "Installed packages:"
pip list | grep -E 'torch|numpy|kolmogorov|scikit|pandas|h5py|matplotlib'

# Ensure user-installed packages are in path
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
unset PYTHONPATH_ORIG

# Check for required data files
echo "Checking for required data files..."
if [ ! -f $SUBMIT_DIR/Data/expression_data1.h5ad ] || [ ! -f $SUBMIT_DIR/Data/net_grn.tsv ]; then
    echo "ERROR: Required data files not found!"
    exit 1
fi

# Copy data and code
echo "Copying data files and code to temporary directory..."
cp $SUBMIT_DIR/Data/simulated_gene_expression.csv $TMP_DIR/Data/
cp $SUBMIT_DIR/Data/synthetic_ground_truth.csv $TMP_DIR/Data/
cp $SUBMIT_DIR/*.py $TMP_DIR/

# Set up backup mechanism as a safety net - runs in background
echo "Setting up safety net backup mechanism (checking every 5 minutes)..."
(
    while true; do
        echo "$(date): Safety check - looking for any missed files..."
        if [ -d "$TMP_DIR/kan_models" ]; then
            gene_dirs=$(find $TMP_DIR/kan_models -maxdepth 1 -type d -not -path "$TMP_DIR/kan_models" 2>/dev/null)
            
            if [ -n "$gene_dirs" ]; then
                echo "$gene_dirs" | while read dir; do
                    gene=$(basename "$dir")
                    
                    # Source and destination paths
                    feature_src="$dir/feature_importance.csv"
                    log_src="$dir/training_log.txt"
                    feature_dst="$RESULTS_DIR/$gene/feature_importance.csv"
                    log_dst="$RESULTS_DIR/$gene/training_log.txt"
                    
                    # Only copy files that exist in source but not in destination
                    if [ -f "$feature_src" ] && [ ! -f "$feature_dst" ]; then
                        echo "$(date): SAFETY NET - Copying feature_importance.csv for gene $gene"
                        mkdir -p "$RESULTS_DIR/$gene"
                        cp "$feature_src" "$feature_dst"
                    fi
                    
                    if [ -f "$log_src" ] && [ ! -f "$log_dst" ]; then
                        echo "$(date): SAFETY NET - Copying training_log.txt for gene $gene"
                        mkdir -p "$RESULTS_DIR/$gene"
                        cp "$log_src" "$log_dst"
                    fi
                done
            fi
        fi
        sleep 300  # Check every 5 minutes
    done
) > "$RESULTS_DIR/safety_backup.log" 2>&1 &
BACKUP_PID=$!

# Verify Python environment
echo "Verifying Python environment..."
python -c "import sys; print('Python paths:'); [print(p) for p in sys.path]"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import kan; print('KAN is available')" || { echo "ERROR: KAN package still not available after installation. Check if there are permissions issues."; exit 1; }

# Add a trap to ensure we copy essential files even if the job fails
cleanup() {
    echo "$(date): Job is ending, performing final safety check for missed files..."
    
    if [ -d "$TMP_DIR/kan_models" ]; then
        gene_dirs=$(find $TMP_DIR/kan_models -maxdepth 1 -type d -not -path "$TMP_DIR/kan_models" 2>/dev/null)
        
        if [ -n "$gene_dirs" ]; then
            echo "$gene_dirs" | while read dir; do
                gene=$(basename "$dir")
                
                # Source and destination paths
                feature_src="$dir/feature_importance.csv"
                log_src="$dir/training_log.txt"
                feature_dst="$RESULTS_DIR/$gene/feature_importance.csv"
                log_dst="$RESULTS_DIR/$gene/training_log.txt"
                
                # Only copy files that exist in source but not in destination
                if [ -f "$feature_src" ] && [ ! -f "$feature_dst" ]; then
                    echo "$(date): FINAL SAFETY NET - Copying feature_importance.csv for gene $gene"
                    mkdir -p "$RESULTS_DIR/$gene"
                    cp "$feature_src" "$feature_dst"
                fi
                
                if [ -f "$log_src" ] && [ ! -f "$log_dst" ]; then
                    echo "$(date): FINAL SAFETY NET - Copying training_log.txt for gene $gene"
                    mkdir -p "$RESULTS_DIR/$gene"
                    cp "$log_src" "$log_dst"
                fi
            done
        fi
    fi
    
    # Clean up any temp_ckpt directories that may be left
    echo "Cleaning up temporary checkpoint directories..."
    find $TMP_DIR -name "temp_ckpt" -type d -exec rm -rf {} \; 2>/dev/null || true
    
    if kill -0 $BACKUP_PID 2>/dev/null; then
        echo "Terminating backup process PID $BACKUP_PID"
        kill $BACKUP_PID
    fi
    echo "Backup process complete. Job finished at $(date)"
}
trap cleanup EXIT

# Configure CUDA for optimal performance
export CUDA_AUTO_BOOST=1
export CUDA_CACHE_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8

# Get system info
NUM_CORES=$(nproc)
MEMORY=$(free -g | awk '/^Mem:/{print $2}')
echo "System has $NUM_CORES CPU cores and ${MEMORY}GB RAM"
echo "GPU information:"
nvidia-smi

# Calculate optimal settings for data loading
DATA_WORKERS=$(( $NUM_CORES / 2 ))
echo "Using $DATA_WORKERS workers for data loading"
BATCH_SIZE=512
MAX_MODELS=30

echo "Using memory-optimized settings: Batch size=$BATCH_SIZE, Max parallel models=$MAX_MODELS"

# Run the code with optimizations
echo "Starting main training script with memory optimizations..."
cd $TMP_DIR  # Ensure we're in the right directory
python main.py --output-dir kan_models \
               --expression-file Data/expression_data1.h5ad \
               --network-file Data/net_grn.tsv \
               --batch-size $BATCH_SIZE \
               --epochs 50 \
               --max-models $MAX_MODELS \
               --data-workers $DATA_WORKERS \
               --patience 5 \
               --learning-rate 0.01 \
               --use-data-copies