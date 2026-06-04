# Zirconia Conductivity PIML Code Release

This repository contains the code used for the zirconia conductivity database, Arrhenius-constrained conductivity prediction, inverse-design diagnostics, uncertainty analysis and manuscript figure generation.

The repository is organized as a code release. Processed data tables, model checkpoints and raw in-house impedance spectra are not included by default. They should be placed under `data/` or `results/` after obtaining them from the corresponding author.

## Directory Structure

```text
.
├── scripts/
│   ├── data_cleaning/      # database loading and text-field normalization scripts
│   ├── modeling/           # PIML/DNN training and evaluation entry points
│   ├── diagnostics/        # bootstrap, sparse-temperature and ablation diagnostics
│   ├── inverse_design/     # candidate search and checkpoint sensitivity scripts
│   ├── dft/                # DFT input-setting extraction utilities
│   └── figures/            # manuscript figure-redrawing scripts
├── data/                   # place input tables here
├── results/                # generated outputs
├── config/                 # optional local configuration files
├── requirements.txt
└── RUN_ORDER.md
```

## Environment

Create a Python environment and install the core dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch separately using the CPU or CUDA command appropriate for your machine. The scripts were developed and tested primarily in a Windows PowerShell workflow.

## Data Expected by the Scripts

The scripts expect processed conductivity records, model-ready feature tables, split files and selected result tables. These files are not bundled in this release because some in-house experimental records are shared only on reasonable request.

Recommended local layout:

```text
data/
├── processed/
├── model_ready/
└── dft/

results/
├── model_training/
├── review_validation_full/
├── piml_advantage_search/
└── figures/
```

## Main Reproduction Steps

See `RUN_ORDER.md` for the recommended order:

1. Prepare or request processed data tables.
2. Run model training / grouped evaluation.
3. Run temperature-extrapolation and uncertainty diagnostics.
4. Run inverse-design and candidate sensitivity checks.
5. Extract DFT settings and regenerate manuscript figures.

## Notes

- The code is intended to reproduce the analyses reported in the manuscript, not to provide a polished software package.
- Several scripts use fixed paths in the original project. If running in a new repository, adjust path constants or run from the repository root.
- Raw in-house impedance spectra and additional experimental metadata are available from the corresponding author upon reasonable request, subject to institutional constraints.
