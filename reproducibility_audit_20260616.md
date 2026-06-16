# Reproducibility Audit, 2026-06-16

## Scope

This audit checked whether the current manuscript analyses can be reproduced from the project files on this workstation without reading the full raw cohort CSV tables.

## Available Inputs

- Deduplicated CSV cohort files are present for CHARLS, ELSA, HRS, KLoSA, LASI, MHAS, and SHARE under `F:\目前养老官方数据库\NC启动\01_data_deduped\csv`.
- Gap-filled harmonised analysis file is present: `03_analysis_pipeline\outputs_gap_filled\harmonized_core_gap_filled.parquet`.
- Additional harmonised Stata inputs used for gap filling are present under `F:\目前养老官方数据库\数据库新20260615\harmonized数据`, including SHARE, MHAS, KLoSA, and LASI files.
- Manuscript result CSVs are present under `03_analysis_pipeline\outputs_nc_longitudinal` and `03_analysis_pipeline\outputs_nc_upgrade`.

## Available Code

The following analysis scripts are present:

- `03_analysis_pipeline\scripts\harmonize_and_analyze_aging.py`
- `03_analysis_pipeline\scripts\fill_share_mhas_klosa_lasi_gaps.py`
- `03_analysis_pipeline\scripts\run_nc_longitudinal_analysis.py`
- `03_analysis_pipeline\scripts\run_nc_upgrade_analyses.py`

The following manuscript figure scripts are present:

- `05_mutilrewrite_strict\make_selected_bubble_style_figures.py`
- `05_mutilrewrite_strict\make_essential_fig5_domain.py`
- `05_mutilrewrite_strict\make_fig1_fig2_large_font_review.py`

## Environment Check

The required Python packages checked on this workstation are available:

- `numpy`
- `pandas`
- `statsmodels`
- `matplotlib`
- `plotly`
- `kaleido`
- `pyarrow`

## Reproducibility Judgment

The current project is reproducible on this workstation in practical terms: the core inputs, intermediate harmonised parquet files, analysis scripts, result CSVs, figure scripts, LaTeX source, bibliography, and figure PDFs are present.

The project is not yet packaged as a fully portable public reproduction bundle. Two issues remain:

1. Some scripts contain workstation-specific default paths on the F drive. They can be overridden partly by command-line arguments, but a clean `config.yaml` or documented run script would make reproduction safer.
2. The underlying individual-level CHARLS/HRS-family cohort data are governed by separate data-use agreements. They should not be publicly redistributed unless the relevant data providers explicitly allow it.

## Recommended Submission Wording

Data availability:

The analysis used harmonised secondary ageing-cohort data prepared for this project from CHARLS, ELSA, HRS, KLoSA, MHAS, SHARE, and LASI resources. Access to the underlying individual-level cohort data is governed by each study's data-use agreement and registration process. Derived analytic summaries used for the manuscript tables and figures can be provided by the corresponding author upon reasonable request, subject to the applicable cohort data-use conditions.

Code availability:

Analysis scripts and figure-generation scripts can be provided by the corresponding author upon reasonable request as part of the project archive.

## Recommended Next Packaging Step

Before submission or reviewer response, create a small `reproducibility_package` folder containing:

- run-order README;
- scripts used for harmonisation, longitudinal analysis, upgrade analysis, and figures;
- environment file listing Python package versions;
- all result CSV files used by manuscript tables and figures;
- figure-generation scripts and final PDF figures;
- a note that raw individual-level cohort data must be obtained from the original cohort data providers.
