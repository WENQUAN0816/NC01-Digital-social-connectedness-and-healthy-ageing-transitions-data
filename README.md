# NC01 public reproducibility and aggregate data repository

This repository contains the public reproducibility materials for:

**Digital-social connectedness and healthy ageing transitions in a multiple cohort study**

It is the public companion repository for the Nature Communications submission. The main manuscript/project repository is:

https://github.com/WENQUAN0816/NC01-Digital-social-connectedness-and-healthy-ageing-transitions-in-a-multiple-cohort-study

## Included

- Analysis scripts used to harmonise variables and run longitudinal, joint-exposure, exposure-history, transition, heterogeneity, robustness, IPW, and E-value analyses.
- Aggregate CSV outputs used to construct manuscript tables and figures.
- Manuscript LaTeX source and bibliography.
- Main figure PDFs/SVGs and supplementary information PDF.
- Audit notes documenting data provenance and reproducibility boundaries.

## Not included

This public package intentionally excludes raw individual-level secondary cohort data, harmonised person-level or person-wave files, and participant identifiers. Excluded examples include raw CSV/DTA/XLSX data, harmonized_core.parquet, and harmonized_core_gap_filled.parquet.

## Data access

Underlying individual-level data are controlled by the original cohort providers and must be obtained from CHARLS, ELSA, HRS/RAND HRS, KLoSA, MHAS, SHARE, and LASI through their official registration and data-use agreement procedures. After obtaining approved source data, users can adapt the scripts in scripts/ to rebuild the harmonised analysis files and aggregate outputs.

## Reproducibility level

The package supports verification of reported aggregate estimates, figures, and tables from the included aggregate outputs. Full person-level reruns require independent access to the original cohort data under the relevant data-use agreements.

## Repository structure

- `scripts/`: analysis and harmonisation scripts.
- `aggregate_outputs/`: aggregate model estimates, cohort summaries, transition summaries, sensitivity analyses, and figure/table source data.
- `figures_pdf_svg/`: manuscript figure files in PDF/SVG format.
- `manuscript_source/`: manuscript LaTeX source and bibliography for transparency.
- `supplementary_information.pdf`: supplementary information corresponding to the submission.
- `data_trace_audit_report.md` and `reproducibility_audit_20260616.md`: data boundary and reproducibility audit notes.

## Citation

If using this repository, please cite the article once available and cite this repository URL as the public code and aggregate data record:

https://github.com/WENQUAN0816/NC01-Digital-social-connectedness-and-healthy-ageing-transitions-data
