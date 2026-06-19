# NC01: Digital-Social Connectedness and Healthy Aging

This repository contains the complete reproducibility package for the manuscript:

**"Digital-social connectedness predicts healthy aging via cognitive and mental health pathways: an explainable machine learning analysis"**

Submitted to: npj Digital Medicine

---

## 📁 Repository Structure

```
NC01-Digital-social-connectedness-and-healthy-ageing-transitions-data/
├── manuscript/                          # Manuscript files
│   ├── manuscript_npj_final.pdf        # Final manuscript (13 pages)
│   ├── manuscript_npj_final.tex        # LaTeX source
│   └── figures/                         # All figures (PDF + PNG)
│       ├── figure1_mediation.pdf       # Mediation pathway diagram
│       ├── figure2_roc_plotly.pdf      # ROC curves (6 models)
│       └── figure3_shap_plotly.pdf     # SHAP feature importance
│
├── scripts/                             # Analysis scripts
│   ├── 00_complete_analysis_fixed.py   # Main analysis pipeline
│   ├── 01_mediation_analysis.py        # Mediation pathway analysis
│   ├── 02_ml_shap_analysis.py          # XGBoost + SHAP analysis
│   ├── 03_moderator_analysis.py        # Age/sex subgroup analysis
│   ├── 07_sensitivity_and_subgroup_analysis.py  # Robustness tests
│   ├── 11_generate_figure2_plotly.py   # Figure 2 generation (Plotly)
│   └── 11_generate_figure3_plotly.py   # Figure 3 generation (Plotly)
│
├── ml_analysis_output/                  # Analysis results
│   ├── model_performance.csv           # Model comparison metrics
│   ├── shap_importance.csv             # SHAP feature rankings
│   ├── mediation_results.csv           # Pathway coefficients
│   └── subgroup_analysis.csv           # Stratified results
│
└── NC01_DATA_QUALITY_AND_ML_READINESS_REPORT.md  # Data quality report

```

---

## 📊 Key Findings

### Machine Learning Performance
- **Best Model**: XGBoost (AUC 0.7215, 95% CI 0.717-0.726)
- **Improvement**: 7.4% over logistic regression (p<0.001)
- **Robustness**: All gradient boosting methods within 0.06% (0.7213-0.7217)

### Feature Importance (SHAP)
1. Heart disease (45.2%)
2. Hypertension (41.7%)
3. Diabetes (22.2%)
4. **Digital connectedness (10.0%)** ⭐
5. Age (8.3%)

### Mediation Pathways
- **Cognitive pathway**: 36.2% mediated (95% CI: 33.1-39.5%)
- **Mental health pathway**: 59.6% mediated (95% CI: 55.8-63.7%)
- **Total mediation**: 95.7% (95% CI: 93.5-97.8%)

### Age Heterogeneity
- **Adults ≥70 years**: AUC 0.7345 (strongest effects)
- **Adults <70 years**: AUC 0.6978
- **Difference**: 5.3% (p<0.001) — compensatory effect in oldest-old

---

## 🔬 Methods

### Data Source
- **Cohort**: China Health and Retirement Longitudinal Study (CHARLS)
- **Waves**: 2011, 2013, 2015, 2018, 2020
- **Sample**: 88,456 observations from 25,873 individuals
- **Age**: ≥45 years

### Exposure
**Digital-social connectedness** (binary):
- Internet use (past month) OR
- Social participation (≥1 organized activity/month)

### Outcome
**Healthy aging** (≥4 of 5 domains):
1. Good self-rated health
2. No ADL difficulties
3. Low depression (CESD-10 <10)
4. Good cognition (z-score >-1.5)
5. <2 chronic conditions

### Analysis Pipeline
1. **Mediation analysis**: Cognitive + mental health pathways
2. **Machine learning**: 6 algorithms with 5-fold CV
3. **SHAP analysis**: Feature importance + explainability
4. **Sensitivity tests**: 3 outcome definitions, 6 algorithms
5. **Subgroup analysis**: Age, sex, rural residence

---

## 📈 Figures

### Figure 1: Mediation Pathways
- Cognitive pathway: a₁=0.184, b₁=0.095
- Mental health pathway: a₂=-0.106, b₂=-0.268
- Direct effect: c'=0.019

### Figure 2: ROC Curves
- 6 models compared
- XGBoost: AUC 0.7215 (best)
- Logistic Regression: AUC 0.7141 (baseline)

### Figure 3: SHAP Feature Importance
- Beeswarm plot
- 8 features ranked
- Digital connectedness: 4th rank, 10.0% contribution

---

## 🖥️ Requirements

### Python Dependencies
```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm catboost \
            shap plotly matplotlib seaborn
```

### LaTeX Dependencies
```bash
# TeX Live 2025 or later
pdflatex, bibtex, tikz, float, booktabs, hyperref
```

---

## 🚀 Reproducibility

### Generate All Figures
```bash
cd scripts/
python 11_generate_figure2_plotly.py  # ROC curves
python 11_generate_figure3_plotly.py  # SHAP plot
```

### Run Complete Analysis
```bash
python 00_complete_analysis_fixed.py  # Main pipeline
python 01_mediation_analysis.py       # Mediation
python 02_ml_shap_analysis.py         # ML + SHAP
```

### Compile Manuscript
```bash
cd manuscript/
pdflatex manuscript_npj_final.tex
```

---

## 📊 Data Availability

**CHARLS data** are publicly available at: http://charls.pku.edu.cn/en

Following registration and data use agreement, researchers can download:
- Harmonized datasets
- Codebooks
- Survey instruments

**Note**: This repository contains only aggregate results and analysis scripts. Individual-level data must be obtained directly from CHARLS.

---

## 📄 Citation

If using this code or reproducing analyses, please cite:

```bibtex
@article{nc01_2026,
  title={Digital-social connectedness predicts healthy aging via cognitive and mental health pathways: an explainable machine learning analysis},
  author={[Authors]},
  journal={npj Digital Medicine},
  year={2026},
  note={Submitted}
}
```

---

## 📧 Contact

For questions about the analysis or code:
- **Repository**: https://github.com/WENQUAN0816/NC01-Digital-social-connectedness-and-healthy-ageing-transitions-data
- **Corresponding Author**: [Email]

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- CHARLS team at Peking University
- All study participants
- Funding agencies

---

**Last Updated**: 2026-06-19
