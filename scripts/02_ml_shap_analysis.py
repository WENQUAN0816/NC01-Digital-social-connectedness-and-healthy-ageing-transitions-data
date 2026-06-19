"""
机器学习 + SHAP 分析：XGBoost预测模型 + 可解释性分析
for npj Digital Medicine submission
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, brier_score_loss,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import shap

print("="*80)
print("XGBoost + SHAP Analysis for npj Digital Medicine")
print("="*80)

# ============================================================================
# STEP 1: 加载并准备数据
# ============================================================================
print("\nSTEP 1: 加载数据...")

charls_path = r"F:\目前养老官方数据库FOR NC\NC启动\01_data_deduped\csv\charls.csv"
df = pd.read_csv(charls_path, low_memory=False)

print(f"✓ 加载 {len(df):,} 行")

# 处理数据（复用之前的逻辑）
df_ml = df[[
    'ID (受访者编码)', 'wave (第几波调查)', 'age (年龄)', 'ragender (性别)',
    'raeducl (教育统一分类)', 'marry (婚姻)', 'hrural (居住在农村或城市)',
    'social10 (上网)', 'socwk (是否每月参与社交)',
    'srh (自评健康)', 'adlab_c (ADL(6项有困难))', 'cesd10 (心理健康(30分,越大越差))',
    'tcog_z_z (认知/总认知能力z标准化(ref1))', 'frailtyb (虚弱指数b)',
    'hibpe (高血压)', 'diabe (糖尿病)', 'hearte (心脏病)', 'stroke (中风)', 'cancre (癌症)'
]].copy()

df_ml.columns = [
    'participant_id', 'wave', 'age', 'gender', 'education', 'married', 'rural',
    'internet_use', 'social_participation',
    'srh', 'adl', 'depression', 'cognition_z', 'frailty',
    'hypertension', 'diabetes', 'heart_disease', 'stroke_disease', 'cancer'
]

# 转换
df_ml['internet_use'] = df_ml['internet_use'].map({'是': 1, '否': 0})
df_ml['social_participation'] = df_ml['social_participation'].map({'是': 1, '否': 0})
df_ml['srh'] = df_ml['srh'].map({'很好': 1, '较好': 2, '一般': 3, '较差': 4, '很差': 5})

for col in ['hypertension', 'diabetes', 'heart_disease', 'stroke_disease', 'cancer']:
    df_ml[col] = df_ml[col].map({'是': 1, '否': 0})

numeric_cols = ['age', 'gender', 'education', 'married', 'rural', 'adl',
                'depression', 'cognition_z', 'frailty']
for col in numeric_cols:
    df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')

df_ml = df_ml[df_ml['age'] >= 45]

# 创建结果变量
df_ml['good_srh'] = (df_ml['srh'] <= 2).astype(float)
df_ml['no_adl'] = (df_ml['adl'] == 0).astype(float)
df_ml['not_depressed'] = (df_ml['depression'] < 10).astype(float)
df_ml['good_cognition'] = (df_ml['cognition_z'] > -1.5).astype(float)
df_ml['disease_count'] = df_ml[['hypertension', 'diabetes', 'heart_disease', 'stroke_disease', 'cancer']].sum(axis=1)
df_ml['no_multimorbidity'] = (df_ml['disease_count'] < 2).astype(float)

df_ml['healthy_aging_score'] = df_ml[['good_srh', 'no_adl', 'not_depressed', 'good_cognition', 'no_multimorbidity']].sum(axis=1)
df_ml['healthy_aging_binary'] = (df_ml['healthy_aging_score'] >= 4).astype(float)

# 联合暴露变量
df_ml['digital_connected'] = ((df_ml['internet_use'] == 1) | (df_ml['social_participation'] == 1)).astype(float)

# 特征工程
df_ml['female'] = (df_ml['gender'] == 2).astype(float)
df_ml['rural_binary'] = (df_ml['rural'] == 1).astype(float)
df_ml['married_binary'] = df_ml['married'].isin([1, 2]).astype(float)
df_ml['education_years'] = df_ml['education'].map({1: 0, 2: 6, 3: 9, 4: 12, 5: 12, 6: 16, 7: 19}).fillna(9)

# 特征列
features = [
    'digital_connected',
    'age', 'female', 'education_years', 'married_binary', 'rural_binary',
    'cognition_z', 'depression', 'adl', 'disease_count'
]

target = 'healthy_aging_binary'

df_clean = df_ml[features + [target]].dropna()

print(f"✓ 清洗后样本: {len(df_clean):,}")
print(f"  特征数: {len(features)}")
print(f"  目标分布: 0={df_clean[target].value_counts()[0]}, 1={df_clean[target].value_counts()[1]}")

X = df_clean[features]
y = df_clean[target]

# ============================================================================
# STEP 2: 5-Fold Cross-Validation
# ============================================================================
print("\n" + "="*80)
print("STEP 2: 5-Fold Cross-Validation")
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
all_y_true = []
all_y_pred = []

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
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(X_train, y_train, verbose=False)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    brier = brier_score_loss(y_val, y_pred_proba)

    print(f"Fold {fold}: AUC={auc:.4f}, Acc={acc:.4f}, Prec={precision:.4f}, Recall={recall:.4f}, Brier={brier:.4f}")

    cv_results.append({
        'fold': fold,
        'auc': auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'brier': brier
    })

    all_y_true.extend(y_val.values)
    all_y_pred.extend(y_pred_proba)

cv_df = pd.DataFrame(cv_results)

print(f"\n{'='*80}")
print("Cross-Validation Summary")
print(f"{'='*80}")
print(f"Mean AUC:       {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
print(f"Mean Accuracy:  {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
print(f"Mean Precision: {cv_df['precision'].mean():.4f} ± {cv_df['precision'].std():.4f}")
print(f"Mean Recall:    {cv_df['recall'].mean():.4f} ± {cv_df['recall'].std():.4f}")
print(f"Mean Brier:     {cv_df['brier'].mean():.4f} ± {cv_df['brier'].std():.4f}")

# ============================================================================
# STEP 3: 训练最终模型
# ============================================================================
print("\n" + "="*80)
print("STEP 3: 训练最终模型...")
print("="*80)

final_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

final_model.fit(X, y, verbose=False)
print("✓ 最终模型训练完成")

# ============================================================================
# STEP 4: SHAP 分析
# ============================================================================
print("\n" + "="*80)
print("STEP 4: SHAP 可解释性分析...")
print("="*80)

sample_size = min(2000, len(X))
X_sample = X.sample(n=sample_size, random_state=42)

print(f"计算 {sample_size} 样本的SHAP值...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_sample)

# Feature importance
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print("\nFeature Importance (SHAP):")
for i, row in feature_importance.iterrows():
    print(f"  {i+1:2d}. {row['feature']:25s}: {row['mean_abs_shap']:.4f}")

# ============================================================================
# STEP 5: 亚组分析
# ============================================================================
print("\n" + "="*80)
print("STEP 5: 亚组异质性分析...")
print("="*80)

# 添加亚组变量
df_clean['age_group'] = pd.cut(df_clean['age'], bins=[0, 70, 100], labels=['<70', '≥70'])
df_clean['education_group'] = pd.cut(df_clean['education_years'],
                                      bins=[0, 6, 12, 20],
                                      labels=['low', 'medium', 'high'])

subgroup_results = []

subgroups = [
    ('age_group', ['<70', '≥70']),
    ('female', [0, 1]),
    ('rural_binary', [0, 1]),
    ('education_group', ['low', 'medium', 'high'])
]

for group_var, group_vals in subgroups:
    if group_var not in df_clean.columns:
        continue

    print(f"\n分析 {group_var}:")

    for val in group_vals:
        mask = df_clean[group_var] == val
        if not mask.any():
            continue

        shap_mask = X_sample.index.isin(df_clean[mask].index)
        if not shap_mask.any():
            continue

        mean_shap = shap_values[shap_mask].mean(axis=0)

        # 只保存 digital_connected 的 SHAP
        digital_idx = list(X.columns).index('digital_connected')

        subgroup_results.append({
            'subgroup': group_var,
            'value': str(val),
            'digital_connected_shap': mean_shap[digital_idx],
            'n': shap_mask.sum()
        })

        print(f"  {val}: digital_connected SHAP={mean_shap[digital_idx]:.4f}, n={shap_mask.sum()}")

subgroup_df = pd.DataFrame(subgroup_results)

# ============================================================================
# STEP 6: 保存结果
# ============================================================================
print("\n" + "="*80)
print("STEP 6: 保存结果...")
print("="*80)

output_dir = Path("../ml_analysis_output")
output_dir.mkdir(exist_ok=True)

# 保存表格
cv_df.to_csv(output_dir / "ml_cv_results.csv", index=False)
feature_importance.to_csv(output_dir / "ml_feature_importance.csv", index=False)
subgroup_df.to_csv(output_dir / "ml_subgroup_shap.csv", index=False)

print("✓ 表格已保存")

# ============================================================================
# STEP 7: 可视化
# ============================================================================
print("\nSTEP 7: 生成图表...")

fig_dir = output_dir / "figures"
fig_dir.mkdir(exist_ok=True)

# 1. ROC Curve
fpr, tpr, _ = roc_curve(all_y_true, all_y_pred)
auc_overall = roc_auc_score(all_y_true, all_y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC={auc_overall:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve: Healthy Aging Prediction', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "ml_roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ ml_roc_curve.png")

# 2. Calibration Plot
prob_true, prob_pred = calibration_curve(all_y_true, all_y_pred, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='XGBoost')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Observed Proportion', fontsize=12)
plt.title('Calibration Plot', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "ml_calibration.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ ml_calibration.png")

# 3. SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False, max_display=10)
plt.tight_layout()
plt.savefig(fig_dir / "ml_shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ ml_shap_summary.png")

# 4. Feature Importance Bar
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), y='feature', x='mean_abs_shap',
            palette='viridis')
plt.xlabel('Mean |SHAP Value|', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance (SHAP)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / "ml_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ ml_feature_importance.png")

# 5. Subgroup Comparison
for group_var in subgroup_df['subgroup'].unique():
    subset = subgroup_df[subgroup_df['subgroup'] == group_var]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(subset['value'], subset['digital_connected_shap'],
                   color=['#3498db', '#e74c3c', '#2ecc71'][:len(subset)])
    ax.set_xlabel(group_var, fontsize=12)
    ax.set_ylabel('Digital Connectedness SHAP Value', fontsize=12)
    ax.set_title(f'Effect Heterogeneity by {group_var}', fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars, subset['digital_connected_shap']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

    plt.tight_layout()
    safe_name = group_var.replace('/', '_').replace(' ', '_')
    plt.savefig(fig_dir / f"ml_subgroup_{safe_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ml_subgroup_{safe_name}.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✓✓✓ XGBoost + SHAP 分析完成 ✓✓✓")
print("="*80)
print(f"\n关键结果:")
print(f"  1. 模型性能: AUC={cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
print(f"  2. Top 3 特征:")
for i, row in feature_importance.head(3).iterrows():
    print(f"     {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")
print(f"  3. Digital connectedness 排名: #{feature_importance[feature_importance['feature']=='digital_connected'].index[0]+1}")
print(f"  4. 亚组数: {len(subgroup_df)}")
print(f"\n✓ 所有结果已保存到: {output_dir.resolve()}")
print("="*80 + "\n")
