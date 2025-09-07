# KAN-FAU

Kolmogorov–Arnold Network (KAN) framework for GRN inference, perturbation experiments, and HPC-based implementation.
````markdown


---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
````

---

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

---

## Usage
To define KAN Architecture, edit the file `HPC_Implementation/KAN_Implementation/train_with_copies.py` and edit in line 298-308
```python
model = KAN(
            [input_size, 2, 1],
            grid=4,
            k=3,
            seed=63,
            ckpt_path=model_checkpoint_path,
        ).to(device)
```


### 1. KAN Training

Keep all the training data inside `HPC_Implementation/KAN_Implementation/Data` folder.

* **Run KAN training and GRN inference on HPC**

  ```bash
  sbatch HPC_Implementation/KAN_Implementation/run_kan.sh
  ```

* **Run KAN training and GRN inference on workstation**

  ```bash
  python HPC_Implementation/KAN_Implementation/main.py
  ```

---

### 2. GRN Creation

Keep the trained folder inside `HPC_Implementation/KAN_Implementation`.

```bash
python HPC_Implementation/KAN_Implementation/network_creator.py
```

---

### 3. Symbolic Formula Value Generator

Ensure that the trained folder is inside `HPC_Implementation/KAN_Implementation`.

```bash
python HPC_Implementation/KAN_Implementation/formula_value_original.py
```

---

### 4. Z-Score Based GRN Maker

```bash
python HPC_Implementation/z-score_grn.py
```

---

### 5. Perturbation Experiment

* **a. Create manually knocked-out datasets**
  Ensure that the actual dataset is inside the `Data/` folder as
  `ctrl_only_Genename*_zero.h5ad`.

  ```bash
  python Perturbation_Experiment/knockout_creator.py
  ```

* **b. Place actual perturbed data for each TF inside `Data/`**
  Example: `CREB1_perturbed_only.h5ad`
  Then create a CSV file for further analysis:

  ```bash
  python Perturbation_Experiment/actual_vs_predicted_file_creator.py
  ```

* **c. Generate correlation plot**
  Replace `*_mean.py` with `*_overall.py` as needed.

  ```bash
  python Perturbation_Experiment/plot_final_KO_mean.py
  ```

* **d. Log2FC Analysis**
  Replace `*_mean.py` with `*_overall.py` as needed.

  ```bash
  python Perturbation_Experiment/plot_log2foldchange_mean.py
  ```


## GPU Acceleration

The code automatically detects and uses GPU acceleration if available.
CUDA optimizations are enabled by default.

---

## Citation

If you use this code in your research, please cite:

```
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Soljačić, Marin and Hou, Thomas Y. and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756v4},
  year={2024}
}
```

