# IceCBR Paper Code

This repository contains the IceCBR code and the result files needed to reproduce the main CBR results table in the paper.

## Layout

- `icecbr/`: main package.
- `scripts/`: command-line scripts.
- `results/`: bundled result files for the main paper table.
- `outputs/`: generated table artifacts.

## Reproduce the Main Table

Generate the main CBR results table from the bundled result files:

```bash
python scripts/build_main_results_table.py
```

This writes:

- `outputs/main_results_table.csv`
- `outputs/main_results_table.tex`

## Run New Predictions

To run IceCBR on your own prepared data, place the required processed inputs under `data_preprocess/`, then run:

```bash
python scripts/run_main_export.py --output-tag my_run --models a0_mrlinear
```

You can then evaluate the exported predictions with:

```bash
python scripts/evaluate_main_export.py --output-tag my_run
```
