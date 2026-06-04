# Suggested Run Order

This file gives a practical order for reproducing the manuscript analyses once the required data tables are available.

## 1. Data Cleaning and Database Construction

Scripts:

- `scripts/data_cleaning/load_raw_data_to_mysql.py`
- `scripts/data_cleaning/translate_column_material_source_and_purity.py`
- `scripts/data_cleaning/translate_column_synthesis_method_and_processing_route.py`

Purpose:

- Load raw or semi-structured records into the relational database workflow.
- Normalize textual source, purity, synthesis-method and processing-route fields.
- Numerical conductivity, composition and temperature fields should be extracted and checked by rule-based scripts rather than modified by text normalization.

## 2. Model Training and Grouped Evaluation

Scripts:

- `scripts/modeling/run_physics_enhanced_campaign.py`
- `scripts/modeling/run_piml_optimization.py`
- `scripts/modeling/run_review_validation_experiments.py`
- `scripts/modeling/evaluate_optimized_candidates.py`

Purpose:

- Train and evaluate Arrhenius-constrained PIML and DNN models.
- Run grouped confirmation, row-wise benchmark and temperature-extrapolation evaluations.
- Export predictions and summary metrics.

## 3. Uncertainty and Sparse-Temperature Diagnostics

Scripts:

- `scripts/diagnostics/review_bootstrap_temperature_extrapolation.py`
- `scripts/diagnostics/piml_sparse_temperature_extrapolation.py`
- `scripts/diagnostics/review_dnn_inverse_temperature_baseline.py`
- `scripts/diagnostics/review_ga_context_sensitivity.py`
- `scripts/diagnostics/prepare_llm_text_audit_sample.py`

Purpose:

- Reproduce paired bootstrap analyses.
- Evaluate sparse low-temperature matched high-temperature diagnostics.
- Check text/context sensitivity and diagnostic DNN variants.

## 4. Inverse Design and Candidate Sensitivity

Scripts:

- `scripts/inverse_design/common_predictor.py`
- `scripts/inverse_design/autonomous_inverse_design.py`
- `scripts/inverse_design/checkpoint_backtracking.py`
- `scripts/inverse_design/optimizer_comparison.py`

Purpose:

- Evaluate candidate compositions under fixed processing and measurement assumptions.
- Compare search strategies.
- Check whether the selected Sc--Mg region is stable across model realizations.

## 5. DFT and CHGNet Supporting Utilities

Scripts:

- `scripts/dft/extract_dft_input_settings.py`

Purpose:

- Extract Quantum ESPRESSO input settings and summarize DFT provenance for the supplementary material.
- CHGNet MD outputs are expected as input result tables; the MD workflow itself may require separate environment setup.

## 6. Manuscript Figures

Scripts:

- `scripts/figures/make_database_overview.py`
- `scripts/figures/redraw_jmateriomics_figures.py`
- `scripts/figures/redraw_ea_diagnostics_jmateriomics.py`
- `scripts/figures/redraw_ab_review_figures.py`

Purpose:

- Regenerate manuscript and supplementary figures from processed result tables.

## Practical Advice

Start with a smoke test using a small subset of the processed data before running full model training. Keep new outputs under `results/` and do not overwrite archived manuscript results until the reproduced metrics are checked.
