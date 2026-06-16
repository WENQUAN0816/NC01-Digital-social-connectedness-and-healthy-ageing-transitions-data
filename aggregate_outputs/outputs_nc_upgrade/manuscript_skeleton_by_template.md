# Manuscript Skeleton Based on Nature Portfolio Template

## Template Article

Primary template:

Yuan et al. **Associations between internet use, social participation, and cognitive function among middle-aged and older adults**. *npj Mental Health Research*. 2025.  
https://www.nature.com/articles/s44184-025-00162-6

Why this is the closest template:

- Same Nature Portfolio style.
- Same core exposures: internet use and social participation.
- Same HRS-family cohorts: CHARLS, MHAS, HRS, ELSA, SHARE.
- Same writing sequence: exposure prevalence -> main associations -> subgroup analyses -> joint exposure -> transition patterns -> sensitivity/limitations.

Your manuscript should imitate the structure, not the wording.

## Working Title

Digital-social connectedness and multidimensional healthy ageing transitions across international ageing cohorts

## Abstract Draft

### Background

Digitalisation and population ageing are reshaping the ways in which older adults remain connected to information, services and social networks. Although internet use and social participation have each been linked to later-life health, their joint and dynamic associations with multidimensional healthy ageing remain unclear across countries.

### Methods

We harmonised longitudinal data from international ageing cohorts, including CHARLS, ELSA, HRS, KLoSA, MHAS and SHARE. Digital connectedness was defined by internet use, social connectedness by social participation, and healthy ageing was measured using a multidimensional index covering self-rated health, functional limitations, multimorbidity, depressive symptoms and cognition. We used lagged models in which connectedness at wave t predicted healthy ageing at wave t+1, adjusting for baseline healthy ageing, age, age squared, sex, education, marital status, rurality, wave and follow-up interval. Cohort-specific estimates were combined using random-effects meta-analysis. We further examined joint connectedness profiles, exposure histories, healthy ageing transitions, heterogeneity, attrition weighting and alternative healthy ageing definitions.

### Findings

Internet use and social participation were each associated with higher odds of subsequent healthy ageing. The random-effects meta-analysis showed an OR of 1.51 for internet use and 1.21 for social participation. Compared with older adults with neither form of connectedness, those with both digital and social connectedness had the most favourable profile for subsequent healthy ageing (OR 1.65, 95% CI 1.36-2.00). This pattern was observed for both gaining healthy ageing (OR 1.68, 95% CI 1.37-2.06) and maintaining healthy ageing (OR 1.65, 95% CI 1.30-2.10). Persistent internet use showed stronger associations than newly adopted use, whereas discontinued use was not clearly associated with subsequent healthy ageing. Results were robust to inverse-probability weighting for attrition and alternative definitions of healthy ageing.

### Interpretation

Digital and social connectedness may represent complementary and dynamic resources associated with favourable healthy ageing transitions in later life. These findings extend prior work from isolated digital behaviours and single health domains to multidimensional healthy ageing trajectories across harmonised international ageing cohorts.

## Results Structure

### 1. Participant Characteristics and Connectedness Patterns

Purpose: imitate the template article's opening descriptive section and Fig. 1.

Report:

- Cohorts included in longitudinal analyses.
- Person-waves and participants by cohort.
- Internet and social participation availability.
- Joint connectedness distribution.
- Exposure history distribution: never, adopted, stopped, persistent.

Use:

- `sample_flow.csv`
- `lagged_availability.csv`
- `joint_exposure_counts.csv`
- `model_exposure_history.csv`

Suggested text:

> The harmonised analytic dataset included repeated observations from multiple ageing cohorts, with substantial variation in the prevalence of digital and social connectedness across cohorts. Internet-use measures were available for CHARLS, ELSA, HRS and SHARE, whereas social participation measures were available for CHARLS, ELSA, HRS, KLoSA, MHAS and SHARE. LASI was not included in the longitudinal main analyses because only one wave was available.

### 2. Internet Use, Social Participation and Subsequent Healthy Ageing

Purpose: imitate the template article's main association table.

Your results:

- Internet use: random-effects OR 1.51, 95% CI 1.43-1.60.
- Social participation: random-effects OR 1.21, 95% CI 1.05-1.40.

Suggested text:

> In cohort-specific lagged models adjusted for baseline healthy ageing and sociodemographic covariates, internet use was consistently associated with higher odds of healthy ageing at the next wave. The random-effects pooled OR was 1.51 (95% CI 1.43-1.60). Social participation also showed a positive association with subsequent healthy ageing, with a random-effects pooled OR of 1.21 (95% CI 1.05-1.40), although between-cohort heterogeneity was substantial.

Use:

- `model_lagged_internet_healthy_ageing.csv`
- `model_lagged_social_healthy_ageing.csv`
- `random_effects_meta_analysis.csv`

### 3. Joint Digital and Social Connectedness

Purpose: imitate the template article's joint association figure, but make it your central result.

Your results:

- Internet only vs neither: OR 1.49, 95% CI 1.44-1.55.
- Social only vs neither: OR 1.15, 95% CI 1.01-1.31.
- Both vs neither: OR 1.65, 95% CI 1.36-2.00.

Suggested text:

> When digital and social connectedness were modelled jointly, older adults with both forms of connectedness had the highest odds of subsequent healthy ageing compared with those with neither form. Internet-only connectedness also showed a strong association, whereas social-only connectedness showed a weaker but positive association. Formal tests did not support clear multiplicative or additive interaction, indicating that the joint-exposure group should be interpreted as the most favourable connectedness profile rather than evidence of statistical synergy.

Use:

- `meta_joint_exposure.csv`
- `model_joint_exposure_by_cohort.csv`
- `meta_joint_multiplicative_interaction.csv`
- `meta_joint_additive_interaction_reri.csv`
- `figure_joint_exposure_forest.png`

### 4. Connectedness Histories Over Time

Purpose: imitate the template article's internet-use transition section.

Your results:

- Adopted internet use: OR 1.24, 95% CI 1.15-1.35.
- Persistent internet use: OR 1.51, 95% CI 1.32-1.74.
- Stopped internet use: OR 1.09, 95% CI 0.91-1.30.
- Adopted social participation: OR 1.19, 95% CI 1.03-1.38.
- Persistent social participation: OR 1.32, 95% CI 1.04-1.69.

Suggested text:

> Exposure-history analyses suggested that sustained connectedness was more strongly associated with healthy ageing than transient connectedness. Persistent internet use showed the strongest association with subsequent healthy ageing, followed by newly adopted internet use, whereas discontinued internet use was not clearly associated with subsequent healthy ageing. A similar but more heterogeneous pattern was observed for social participation histories.

Use:

- `meta_exposure_history.csv`
- `model_exposure_history.csv`
- `leave_one_cohort_out_exposure_history.csv`

### 5. Healthy Ageing Transitions

Purpose: this is your main upgrade beyond the template article.

Your results:

- Gain healthy ageing, both vs neither: OR 1.68, 95% CI 1.37-2.06.
- Maintain healthy ageing, both vs neither: OR 1.65, 95% CI 1.30-2.10.
- Internet only also robust for both transitions.

Suggested text:

> Connectedness was associated not only with remaining healthy among those already healthy at baseline, but also with gaining healthy ageing among those initially unhealthy. Compared with older adults with neither digital nor social connectedness, those with both forms had higher odds of gaining healthy ageing and maintaining healthy ageing at the next wave.

Use:

- `meta_transition_joint_exposure.csv`
- `model_transition_joint_exposure.csv`

### 6. Stratified Analyses and Heterogeneity

Purpose: imitate the template article's stratified analyses, but interpret high heterogeneity as a cross-national issue.

Report:

- Age group, sex, education, rurality, baseline health.
- Leave-one-cohort-out.
- High I2, especially for social participation.

Suggested text:

> Stratified analyses showed generally positive associations across demographic and baseline-health subgroups, but the magnitude varied. Between-cohort heterogeneity was substantial for several social participation estimates, underscoring that connectedness-health associations may depend on national context, measurement differences, digital infrastructure and social participation norms.

Use:

- `subgroup_pooled_models.csv`
- `figure_subgroup_heatmap.png`
- `leave_one_cohort_out_joint.csv`
- `leave_one_cohort_out_domain.csv`

### 7. Domain-Specific Patterns and Potential Pathways

Purpose: borrow the Nature Communications intrinsic-capacity paper's domain contribution logic.

Your results:

- Internet use -> cognition_not_low: OR 2.50.
- Internet use -> no ADL limitation: OR 1.53.
- Internet use -> not depressive: OR 1.39.
- Social participation -> cognition_not_low: OR 1.35.
- Social participation -> no ADL limitation: OR 1.32.
- Social participation -> not depressive: OR 1.19.

Suggested text:

> Domain-specific analyses suggested that cognitive, functional and psychological domains contributed strongly to the observed pattern. These analyses should be interpreted as mechanistic clues rather than formal mediation, because these domains are components of the healthy ageing construct.

Use:

- `meta_domain_outcomes.csv`
- `model_domain_outcomes.csv`

### 8. Robustness Analyses

Purpose: imitate Nature Portfolio robustness sections.

Your results:

- IPW internet use: OR 1.51, 95% CI 1.43-1.59.
- IPW social participation: OR 1.21, 95% CI 1.05-1.40.
- Strict healthy ageing definition remains positive.
- Without cognition component remains positive.

Suggested text:

> Findings were robust to inverse-probability weighting for attrition and to alternative definitions of healthy ageing, including a stricter all-component definition and an outcome excluding the cognition component.

Use:

- `meta_ipw_attrition.csv`
- `ipw_attrition_diagnostics.csv`
- `meta_alternative_definitions.csv`
- `figure_robustness_grid.png`

## Figure Plan

| Figure | Content | Template logic |
|---|---|---|
| Fig. 1 | Study design, cohort timeline, connectedness prevalence, exposure history distribution | Template Fig. 1 |
| Fig. 2 | Joint digital-social connectedness forest plot | Template Fig. 3 |
| Fig. 3 | Exposure histories: adopted, stopped, persistent | Template Fig. 4-5 |
| Fig. 4 | Healthy ageing transitions: gain and maintain | Your novelty upgrade |
| Fig. 5 | Subgroup heatmap and robustness grid | Template stratified/sensitivity logic |
| Fig. 6 | Domain-specific health patterns | Nature Communications domain-contribution logic |

## Discussion Skeleton

### Paragraph 1: Principal Findings

In this harmonised multi-cohort study, digital and social connectedness were associated with favourable healthy ageing transitions. Older adults with both digital and social connectedness had the highest odds of subsequent healthy ageing, and this pattern was evident for both gaining and maintaining healthy ageing.

### Paragraph 2: Comparison With Prior Work

This study extends previous Nature Portfolio work on internet use, social participation and cognition by shifting the outcome from single-domain cognitive function to multidimensional healthy ageing. It also extends digital-divide research by examining cross-national harmonised cohorts and dynamic connectedness histories rather than a single-country static exposure.

### Paragraph 3: Dynamic Connectedness

The exposure-history results suggest that sustained connectedness may be more informative than a single report of internet use or social participation. Persistent internet users showed stronger associations with healthy ageing than newly adopted users, whereas discontinued use was not clearly associated with healthy ageing.

### Paragraph 4: Potential Pathways

The domain-specific results point to cognitive, functional and psychological health as possible pathways. However, because these domains are also components of healthy ageing, these findings should be interpreted as descriptive domain patterns and mechanistic clues rather than formal causal mediation.

### Paragraph 5: Heterogeneity

Substantial heterogeneity across cohorts, especially for social participation, may reflect differences in national digital infrastructure, social participation norms, welfare systems, survey measurement and baseline health composition. This heterogeneity is expected in cross-national ageing research and should be interpreted as part of the substantive finding.

### Paragraph 6: Strengths and Limitations

Strengths include harmonised international ageing cohorts, lagged longitudinal design, cohort-specific modelling with random-effects meta-analysis, dynamic exposure histories, healthy ageing transition outcomes, attrition weighting and multiple outcome definitions. Limitations include observational design, residual confounding, self-reported exposure measurement, harmonisation differences, attrition and lack of formal competing-risk modelling for death.

## Safe Novelty Claim

> This study extends prior work by shifting the focus from isolated digital behaviours or single health domains to dynamic digital-social connectedness patterns and multidimensional healthy ageing transitions across harmonised international ageing cohorts.

## Source Articles To Cite

1. Yuan et al. *npj Mental Health Research* 2025. Internet use, social participation and cognitive function.  
   https://www.nature.com/articles/s44184-025-00162-6

2. Li et al. *Nature Communications* 2026. Intrinsic capacity and stroke risk in a multiple cohort study.  
   https://www.nature.com/articles/s41467-026-70524-x

3. Mak et al. *Nature Medicine* 2023. Hobby engagement and mental wellbeing in 16 countries.  
   https://www.nature.com/articles/s41591-023-02506-1

4. Li et al. *npj Digital Medicine* 2025. Digital divides and healthy aging in CHARLS.  
   https://www.nature.com/articles/s41746-025-02076-1
