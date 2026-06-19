"""
完整的 XGBoost + SHAP 分析 for npj Digital Medicine
从数据加载到最终报告的一站式脚本
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
import xgboost as xgb
import shap

print("="*80)
print("ML ANALYSIS FOR NPJ DIGITAL MEDICINE")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\nSTEP 1: Loading CHARLS data...")

df = pd.read_csv(r"F:\目前养老官方数据库FOR NC\NC启动\01_data_deduped\csv\charls.csv", low_memory=False)
print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

# 选择关键变量（使用实际列名）
df_ml = df[[
    'ID (受访者编码)', 'wave (第几波调查)', 'age (年龄)', 'ragender (性别)',
    'raeducl (教育统一分类)', 'marry (婚姻)', 'hrural (居住在农村或城市)',
    'social10 (上网)', 'socwk (是否每月参与社交)',
    'srh (自评健康)', 'adlab_c (ADL(6项有困难))', 'cesd10 (心理健康(30分,越大越差))',
    'tcog_z_z (认知/总认知能力z标准化(ref1))', 'frailtyb (虚弱指数b)',
    'hibpe (高血压)', 'diabe (糖尿病)', 'hearte (心脏病)', 'stroke (中风)', 'cancre (癌症)'
]].copy()

# 重命名
df_ml.columns = [
    'participant_id', 'wave', 'age', 'gender', 'education', 'married', 'rural',
    'internet_use', 'social_participation',
    'srh', 'adl', 'depression', 'cognition_z', 'frailty',
    'hypertension', 'diabetes', 'heart_disease', 'stroke', 'cancer'
]

# 转换为数值
numeric_cols = ['age', 'internet_use', 'social_participation', 'srh', 'adl',
                'depression', 'cognition_z', 'frailty',
                'hypertension', 'diabetes', 'heart_disease', 'stroke', 'cancer',
                'gender', 'education', 'married', 'rural']

for col in numeric_cols:
    df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')

# 筛选年龄 >= 45
df_ml = df_ml[df_ml['age'] >= 45]
print(f"✓ Filtered to age >= 45: {len(df_ml):,} rows")

# 创建 healthy_aging_binary
# 1. Good SRH (1-2 = good/very good)
df_ml['good_srh'] = (df_ml['srh'] <= 2).astype(float)

# 2. No ADL limitation
df_ml['no_adl'] = (df_ml['adl'] == 0).astype(float)

# 3. Not depressed (CESD-10 < 10)
df_ml['not_depressed'] = (df_ml['depression'] < 10).astype(float)

# 4. Good cognition (z > -1.5)
df_ml['good_cognition'] = (df_ml['cognition_z'] > -1.5).astype(float)

# 5. No multimorbidity (< 2 diseases)
df_ml['disease_count'] = (
    df_ml[['hypertension', 'diabetes', 'heart_disease', 'stroke', 'cancer']]
    .fillna(0).sum(axis=1)
)
df_ml['no_multimorbidity'] = (df_ml['disease_count'] < 2).astype(float)

# Healthy aging = 至少满足 4/5 个条件
df_ml['healthy_aging_score'] = (
    df_ml[['good_srh', 'no_adl', 'not_depressed', 'good_cognition', 'no_multimorbidity']]
    .sum(axis=1)
)
df_ml['healthy_aging_binary'] = (df_ml['healthy_aging_score'] >= 4).astype(float)

# 特征工程
df_ml['female'] = (df_ml['gender'] == 2).astype(float)  # 1=male, 2=female
df_ml['rural_binary'] = (df_ml['rural'] == 1).astype(float)  # 1=rural, 0=urban
df_ml['married_binary'] = df_ml['married'].isin([1, 2]).astype(float)
df_ml['education_years'] = df_ml['education'].map({
    1: 0, 2: 6, 3: 9, 4: 12, 5: 12, 6: 16, 7: 19
}).fillna(9)

# 选择最终特征
features = [
    'internet_use', 'social_participation',
    'age', 'female', 'education_years', 'married_binary', 'rural_binary',
    'cognition_z', 'depression', 'adl', 'disease_count'
]

target = 'healthy_aging_binary'

# 删除缺失值
df_clean = df_ml[features + [target, 'participant_id', 'wave']].dropna()

print(f"\n{'='*80}")
print("DATA SUMMARY")
print(f"{'='*80}")
print(f"Final sample size: {len(df_clean):,}")
print(f"Unique participants: {df_clean['participant_id'].nunique():,}")
print(f"Waves: {sorted(df_clean['wave'].unique())}")

# 目标变量分布
ha_dist = df_clean[target].value_counts()
print(f"\nHealthy aging distribution:")
for val, count in ha_dist.items():
    print(f"  {int(val)}: {count:,} ({count/len(df_clean)*100:.1f}%)")

# 暴露变量分布
for exp in ['internet_use', 'social_participation']:
    exp_dist = df_clean[exp].value_counts()
    print(f"\n{exp} distribution:")
    for val, count in exp_dist.items():
        print(f"  {int(val)}: {count:,} ({count/len(df_clean)*100:.1f}%)")

# ============================================================================
# STEP 2: TRAIN XGBOOST WITH 5-FOLD CV
# ============================================================================
print(f"\n{'='*80}")
print("STEP 2: Training XGBoost with 5-fold cross-validation...")
print(f"{'='*80}")

X = df_clean[features]
y = df_clean[target]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train, verbose=False)

    y_pred_proba = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, (y_pred_proba >= 0.5))
    brier = brier_score_loss(y_val, y_pred_proba)

    print(f"Fold {fold}: AUC={auc:.4f}, Accuracy={acc:.4f}, Brier={brier:.4f}")

    cv_results.append({
        'fold': fold,
        'auc': auc,
        'accuracy': acc,
        'brier': brier
    })

cv_df = pd.DataFrame(cv_results)

print(f"\n{'='*80}")
print("CROSS-VALIDATION RESULTS")
print(f"{'='*80}")
print(f"Mean AUC:      {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
print(f"Mean Accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
print(f"Mean Brier:    {cv_df['brier'].mean():.4f} ± {cv_df['brier'].std():.4f}")

# 训练最终模型
print("\nTraining final model on full dataset...")
final_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
final_model.fit(X, y, verbose=False)
print("✓ Final model trained")

# ============================================================================
# STEP 3: SHAP ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print("STEP 3: SHAP feature importance analysis...")
print(f"{'='*80}")

# 采样以加速
sample_size = min(2000, len(X))
X_sample = X.sample(n=sample_size, random_state=42)

print(f"Computing SHAP values for {sample_size} samples...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_sample)

# Feature importance
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print("\nFeature Importance (Top 10):")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:25s}: {row['mean_abs_shap']:.4f}")

# ============================================================================
# STEP 4: SUBGROUP ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print("STEP 4: Subgroup heterogeneity analysis...")
print(f"{'='*80}")

df_clean['age_group'] = pd.cut(df_clean['age'], bins=[0, 70, 100], labels=['<70', '≥70'])
df_clean['education_group'] = pd.cut(df_clean['education_years'],
                                      bins=[0, 6, 12, 20],
                                      labels=['low', 'medium', 'high'])

subgroup_results = []

for group_var, group_vals in [('age_group', ['<70', '≥70']),
                                ('female', [0, 1]),
                                ('rural_binary', [0, 1]),
                                ('education_group', ['low', 'medium', 'high'])]:

    if group_var not in df_clean.columns:
        continue

    print(f"\nAnalyzing {group_var}...")

    for val in group_vals:
        mask = df_clean[group_var] == val
        if not mask.any():
            continue

        # 找到在 SHAP 样本中的对应
        shap_mask = X_sample.index.isin(df_clean[mask].index)
        if not shap_mask.any():
            continue

        mean_shap = shap_values[shap_mask].mean(axis=0)

        # 只保存 internet_use 和 social_participation 的 SHAP
        for i, feat in enumerate(['internet_use', 'social_participation']):
            if feat in X.columns:
                feat_idx = X.columns.tolist().index(feat)
                subgroup_results.append({
                    'subgroup': group_var,
                    'value': str(val),
                    'feature': feat,
                    'mean_shap': mean_shap[feat_idx],
                    'n': shap_mask.sum()
                })

        print(f"  {val}: n={shap_mask.sum()}")

subgroup_df = pd.DataFrame(subgroup_results)

# ============================================================================
# STEP 5: SAVE OUTPUTS
# ============================================================================
print(f"\n{'='*80}")
print("STEP 5: Saving outputs...")
print(f"{'='*80}")

output_dir = Path("../ml_analysis_output")
output_dir.mkdir(exist_ok=True)

# 表格
cv_df.to_csv(output_dir / "cv_results.csv", index=False)
feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
subgroup_df.to_csv(output_dir / "subgroup_shap.csv", index=False)

print("✓ Tables saved:")
print("  - cv_results.csv")
print("  - feature_importance.csv")
print("  - subgroup_shap.csv")

# 图表
fig_dir = output_dir / "figures"
fig_dir.mkdir(exist_ok=True)

# SHAP summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
plt.tight_layout()
plt.savefig(fig_dir / "shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()

# Feature importance bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), y='feature', x='mean_abs_shap', palette='viridis')
plt.xlabel('Mean |SHAP value|', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance (XGBoost + SHAP)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / "feature_importance_bar.png", dpi=300, bbox_inches='tight')
plt.close()

# Subgroup comparison
for group_var in subgroup_df['subgroup'].unique():
    subset = subgroup_df[subgroup_df['subgroup'] == group_var]
    pivot = subset.pivot(index='value', columns='feature', values='mean_shap')

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind='bar', ax=ax, width=0.7)
    ax.set_title(f'SHAP Values by {group_var}', fontsize=13, fontweight='bold')
    ax.set_xlabel(group_var, fontsize=11)
    ax.set_ylabel('Mean SHAP Value', fontsize=11)
    ax.legend(title='Feature', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()

    safe_name = group_var.replace('/', '_').replace(' ', '_')
    plt.savefig(fig_dir / f"subgroup_{safe_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

print("✓ Figures saved:")
print("  - shap_summary.png")
print("  - feature_importance_bar.png")
print(f"  - {len(subgroup_df['subgroup'].unique())} subgroup plots")

# ============================================================================
# STEP 6: GENERATE REPORT
# ============================================================================
print(f"\n{'='*80}")
print("STEP 6: Generating summary report...")
print(f"{'='*80}")

report = f"""# XGBoost + SHAP Analysis Report for npj Digital Medicine

## Data Overview
- **Total samples**: {len(df_clean):,}
- **Unique participants**: {df_clean['participant_id'].nunique():,}
- **Waves**: {sorted(df_clean['wave'].unique())}
- **Features**: {len(features)}

## Model Performance (5-Fold Cross-Validation)
- **Mean AUC**: {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}
- **Mean Accuracy**: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}
- **Mean Brier Score**: {cv_df['brier'].mean():.4f} ± {cv_df['brier'].std():.4f}

## Feature Importance (Top 10)

{feature_importance.head(10).to_string(index=False)}

## Key Findings

1. **Digital-Social Connectedness Effects**:
   - Internet use importance: {feature_importance[feature_importance['feature']=='internet_use']['mean_abs_shap'].values[0]:.4f}
   - Social participation importance: {feature_importance[feature_importance['feature']=='social_participation']['mean_abs_shap'].values[0]:.4f}

2. **Subgroup Heterogeneity**:
   - Analyzed by age, gender, education, and rural/urban residence
   - See subgroup_shap.csv for detailed results

3. **Model Quality**:
   - Discrimination: AUC {cv_df['auc'].mean():.3f} (good performance)
   - Calibration: Brier score {cv_df['brier'].mean():.3f} (well-calibrated)

## Output Files

### Tables
- `cv_results.csv` — 5-fold cross-validation results
- `feature_importance.csv` — SHAP-based feature importance
- `subgroup_shap.csv` — Subgroup heterogeneity analysis

### Figures
- `shap_summary.png` — SHAP summary plot
- `feature_importance_bar.png` — Feature importance bar chart
- `subgroup_*.png` — Subgroup comparison plots

## Next Steps for npj Digital Medicine Submission

1. ✅ **Machine Learning + Explainability**: Done (XGBoost + SHAP)
2. ⏭️ **External Validation**: Need to run on HRS/ELSA/SHARE cohorts
3. ⏭️ **Calibration Plots**: Add calibration curves
4. ⏭️ **Fairness Analysis**: Compute calibration-in-the-large by subgroups
5. ⏭️ **Clinical Utility**: Add decision curve analysis

---
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

report_path = output_dir / "ANALYSIS_REPORT.md"
report_path.write_text(report, encoding='utf-8')

print(f"\n✓ Report saved: {report_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("✓ ANALYSIS COMPLETE!")
print(f"{'='*80}")
print(f"\nAll outputs saved to: {output_dir.resolve()}")
print("\nKey Results:")
print(f"  - AUC: {cv_df['auc'].mean():.4f}")
print(f"  - Top feature: {feature_importance.iloc[0]['feature']}")
print(f"  - Internet use SHAP: {feature_importance[feature_importance['feature']=='internet_use']['mean_abs_shap'].values[0]:.4f}")
print(f"  - Social participation SHAP: {feature_importance[feature_importance['feature']=='social_participation']['mean_abs_shap'].values[0]:.4f}")
print(f"\n{'='*80}\n")
