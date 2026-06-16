# Data Trace Audit Report

- Manuscript: `F:\目前养老官方数据库\NC启动\05_mutilrewrite_strict\manuscript.tex`
- Audit date: 2026-06-16
- Rule: manuscript numbers must come from user-run result CSV files only; no simulated, illustrative, or invented values are allowed.
- Checked key manuscript values: 98
- Numeric mismatches found: 0
- Unsupported baseline-characteristic values removed: yes

## Corrections Made

- Removed the unsupported baseline-characteristic Table 1 values that were not present in the result CSV outputs.
- Restored Table 1 to cohort coverage and exposure-availability counts copied from `sample_flow.csv` and `lagged_availability.csv`.
- Confirmed that the old values `60.7`, `52.5`, `42.6`, and related baseline age/sex/healthy-ageing percentages no longer appear in `manuscript.tex` or the final PDF text.

## Traced Result Blocks

- Flow and coverage counts: `sample_flow.csv` and `lagged_availability.csv`.
- Main lagged internet/social estimates: `model_lagged_internet_healthy_ageing.csv`, `model_lagged_social_healthy_ageing.csv`, and `random_effects_meta_analysis.csv`.
- Joint connectedness profile estimates: `model_joint_exposure_by_cohort.csv` and `meta_joint_exposure.csv`.
- Multiplicative and additive interaction results: `meta_joint_multiplicative_interaction.csv` and `meta_joint_additive_interaction_reri.csv`.
- Exposure-history results: `meta_exposure_history.csv`.
- Healthy-ageing transition results: `meta_transition_joint_exposure.csv`.
- Domain-specific component results: `meta_domain_outcomes.csv`.
- Robustness results: `meta_ipw_attrition.csv`, `meta_alternative_definitions.csv`, and `e_values.csv`.
- Subgroup and heterogeneity displays: `subgroup_pooled_models.csv` and leave-one-cohort-out CSV outputs.

## Current Status

- All checked manuscript results trace to the existing result CSV files under:
  - `F:\目前养老官方数据库\NC启动\03_analysis_pipeline\outputs_nc_longitudinal`
  - `F:\目前养老官方数据库\NC启动\03_analysis_pipeline\outputs_nc_upgrade`
- No raw full-data tables were read for this audit.
- Remaining scientific interpretation should keep observational language: associated with, linked to, or consistent with; no causal effect is claimed.
