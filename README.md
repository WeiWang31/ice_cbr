# IceCBR: Climate-Shift-Aware Case Adaptation for Non-Stationary Arctic Sea-Ice Forecasting

This repository contains the official code for the paper:

**IceCBR: Climate-Shift-Aware Case Adaptation for Non-Stationary Arctic Sea-Ice Forecasting**  
Accepted as an **Oral presentation** at **ICCBR 2026**.

## Repository Layout

- `icecbr/`: main IceCBR package.
- `scripts/`: command-line scripts.
- `outputs/`: generated table artifacts.
- `data_preprocess/`: processed input data required for running new experiments.
- `result/`: experiment result files used for reproducing the main paper tables (downloaded from Releases).

## Reproduce the Main Paper Table

The result files required to reproduce the main paper table are provided in the repository **Releases**.

### Step 1: Download and extract result files

Download `result.zip` from the latest Release.

Extract the archive and place the **entire `result/` folder** under the **project root directory**.

After extraction, the project structure should look like:

```text
project_root/
├── icecbr/
├── scripts/
├── outputs/
├── result/
│   ├── target_month_export_2016_2023_a0_icemamba_inputs_v1/
│   ├── target_month_export_2016_2023_a0_mrlinear_icemamba_inputs_v1/
│   ├── target_month_export_2016_2023_revise_icemamba_inputs_v1/
│   ├── target_month_export_2016_2023_revise_retain_final_icemamba_inputs_v1/
│   └── ...
└── ...
```

### Step 2: Build the main results table

Run:

```bash
python scripts/build_main_results_table.py
```

This generates:

- `outputs/main_results_table.csv`
- `outputs/main_results_table.tex`

## Run New Predictions

To run IceCBR on your own prepared data:

### Step 1: Prepare data

Place the required processed inputs under:

```text
data_preprocess/
```

### Step 2: Run prediction export

```bash
python scripts/run_main_export.py --output-tag my_run --models a0_mrlinear
```

### Step 3: Evaluate predictions

```bash
python scripts/evaluate_main_export.py --output-tag my_run
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
