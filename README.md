# KAN-FAU

Kolmogorov–Arnold Network (KAN) framework for GRN inference, perturbation experiments, and HPC-based implementation.

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
``` 
## Folder Structure

```text
KAN-FAU/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── HPC_Implementation
│   ├── architecture_analyzer.py
│   ├── benchmarking_functions.py
│   ├── classification_plot.py
│   ├── dataset_analyzer.py
│   ├── z-score_grn.py
│   │
│   ├── KAN_Implementation
│   │   ├── actual_vs_predicted_plot.py
│   │   ├── data_loader.py
│   │   ├── formula_value_original.py
│   │   ├── formula_value_reduced.py
│   │   ├── GRNBOOST2.py
│   │   ├── main.py
│   │   ├── network_creator.py
│   │   ├── reduced_formula.py
│   │   ├── train_with_copies.py
│   │   ├── utils.py
│   │   ├── Variable_matcher.py
│   │   ├── visualization.py
│   │   ├── run_kan.sh
│   │   ├── versions.yml
│   │   ├── safety_backup.log
│   │   ├── Data/
│   │   └── plots/
│   │
│   └── KAN_Evaluation
│       ├── correlation.py
│       ├── correlation_grn.py
│       ├── edge_equalizer.py
│       ├── evaluate_grn.py
│       ├── run_kan.sh
│       ├── scMultiSim.r
│       └── Data/
│
├── Perturbation_Experiment
│   ├── actual_vs_predicted_file_creator.py
│   ├── actual_vs_predicted_mean.py
│   ├── actual_vs_predicted_overall.py
│   ├── cellcount.py
│   ├── knockout_creator.py
│   ├── perturbation_analysis_for_mean.py
│   ├── perturbation_analysis_for_overall.py
│   ├── perturbed_unperturbed_filter.py
│   ├── plot_final_KO_mean.py
│   ├── plot_final_KO_overall.py
│   ├── plot_log2foldchange_mean.py
│   ├── plot_log2foldchange_overall.py
│   ├── grnboost2_perturb.csv
│   ├── Data/
│   ├── BHLHE40/
│   ├── CREB1/
│   ├── DDIT3/
│   └── ZNF326/
│
└── Symbolic Formula-Whole Network
    ├── plot.py
    ├── Data/
    └── SC1139_Symbolic/
```
## Usage

1. KAN Training \n
   Keep all the training data inside HPC_Implementation/KAN_Implementation/Data folder. 

a. Run KAN training and GRN inference- For HPC
```bash
sbatch HPC_Implementation/KAN_Implementation/run_kan.sh 
```
b. Run KAN training and GRN inference- For Workstation
```bash
python HPC_Implementation/KAN_Implementation/main.py
```

2. GRN Creation\n
   Keep the trained folder inside HPC_Implementation/KAN_Implementation
   ```bash
    python HPC_Implementation/KAN_Implementation/network_creator.py
   ```
3. Symbolic Formula Value Generator
  Ensure that the trained folder is inside HPC_Implementation/KAN_Implementation
   ```bash
    python HPC_Implementation/KAN_Implementation/formula_value_original.py
   ```
4. Z-Score Based GRN Maker
   ```bash
    python HPC_Implementation/z-score_grn.py
   ```
5. Perturbation Experiment\n
   a. First create the manually knocked out datasets. Ensure that the actual dataset is inside Data folder ctrl_only_Genename*_zero.h5ad
   ```bash
    python Perturbation_Experiment/knockout_creator.py
   ```
   b. Now place the actual perturbed data for each TF as a separate file inside Data folder. eg- CREB1_perturbed_only.h5ad and create a csv file for further analysis
   ```bash
    python Perturbation_Experiment/actual_vs_perdicted_file_creator.py
   ```
   c. To generate the correlation plot, run the following code. Replace *_mean.py with _*overall.py when needed. 
   ```bash
    python Perturbation_Experiment/plot_final_KO_mean.py
   ```
   d. Log2FC Analysis: To generate log2fc run the following code. Replace *_mean.py with _*overall.py when needed. 
   ```bash
    python Perturbation_Experiment/plot_log2foldchange_mean.py
   ```
   
   
