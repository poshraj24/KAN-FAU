# KAN-FAU

Kolmogorov–Arnold Network (KAN) framework for GRN inference, perturbation experiments, and HPC-based implementation.

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
``` 
KAN-FAU/
├── 📄 .gitignore
├── 📄 LICENSE
├── 📄 README.md
├── 📄 requirements.txt
│
├── 📂 HPC_Implementation
│   ├── 📄 architecture_analyzer.py
│   ├── 📄 benchmarking_functions.py
│   ├── 📄 classification_plot.py
│   ├── 📄 dataset_analyzer.py
│   ├── 📄 z-score_grn.py
│   │
│   ├── 📂 KAN_Implementation
│   │   ├── 📄 actual_vs_predicted_plot.py
│   │   ├── 📄 data_loader.py
│   │   ├── 📄 formula_value_original.py
│   │   ├── 📄 formula_value_reduced.py
│   │   ├── 📄 GRNBOOST2.py
│   │   ├── 📄 main.py
│   │   ├── 📄 network_creator.py
│   │   ├── 📄 reduced_formula.py
│   │   ├── 📄 train_with_copies.py
│   │   ├── 📄 utils.py
│   │   ├── 📄 Variable_matcher.py
│   │   ├── 📄 visualization.py
│   │   ├── 📄 run_kan.sh
│   │   ├── 📄 versions.yml
│   │   ├── 📄 safety_backup.log
│   │   ├── 📂 Data
│   │   └── 📂 plots
│   │
│   └── 📂 KAN_Evaluation
│       ├── 📄 correlation.py
│       ├── 📄 correlation_grn.py
│       ├── 📄 edge_equalizer.py
│       ├── 📄 evaluate_grn.py
│       ├── 📄 run_kan.sh
│       ├── 📄 scMultiSim.r
│       └── 📂 Data
│
├── 📂 Perturbation_Experiment
│   ├── 📄 actual_vs_predicted_file_creator.py
│   ├── 📄 actual_vs_predicted_mean.py
│   ├── 📄 actual_vs_predicted_overall.py
│   ├── 📄 cellcount.py
│   ├── 📄 knockout_creator.py
│   ├── 📄 perturbation_analysis_for_mean.py
│   ├── 📄 perturbation_analysis_for_overall.py
│   ├── 📄 perturbed_unperturbed_filter.py
│   ├── 📄 plot_final_KO_mean.py
│   ├── 📄 plot_final_KO_overall.py
│   ├── 📄 plot_log2foldchange_mean.py
│   ├── 📄 plot_log2foldchange_overall.py
│   ├── 📄 grnboost2_perturb.csv
│   ├── 📂 Data
│   ├── 📂 BHLHE40
│   ├── 📂 CREB1
│   ├── 📂 DDIT3
│   └── 📂 ZNF326
│
└── 📂 Symbolic Formula-Whole Network
    ├── 📄 plot.py
    ├── 📂 Data
    └── 📂 SC1139_Symbolic
## Usage
1. HPC Implementation

Run KAN training and GRN inference
