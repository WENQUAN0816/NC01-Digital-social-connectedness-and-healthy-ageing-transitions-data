"""
测试最新算法：CatBoost, HistGradientBoosting, Neural Network, Ensemble
对比AUC性能
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

# 尝试导入CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost未安装，将跳过该算法")

print("="*80)
print("测试最新机器学习算法")
print("="*80)

# ============================================================================
# 加载数据
# ============================================================================
print("\n加载数据...")
charls_path = r"F:\目前养老官方数据库FOR NC\NC启动\01_data_deduped\csv\charls.csv"
df = pd.read_csv(charls_path, low_memory=False)

# 数据处理（复用之前的代码）
df_ml = df[[
    'wave (第几波调查)', 'age (年龄)', 'ragender (性别)',
    'raeducl (教育统一分类)', 'marry (婚姻)', 'hrural (居住在农村或城市)',
    'social10 (上网)', 'socwk (是否每月参与社交)',
    'srh (自评健康)', 'adlab_c (ADL(6项有困难))', 'cesd10 (心理健康(30分,越大越差))',
    'tcog_z_z (认知/总认知能力z标准化(ref1))',
    'hibpe (高血压)', 'diabe (糖尿病)', 'hearte (心脏病)', 'stroke (中风)', 'cancre (癌症)'
]].copy()

df_ml.columns = [
    'wave', 'age', 'gender', 'education', 'married', 'rural',
    'internet_use', 'social_participation',
    'srh', 'adl', 'depression', 'cognition_z',
    'hypertension', 'diabetes', 'heart_disease', 'stroke_disease', 'cancer'
]

# 转换
df_ml['internet_use'] = df_ml['internet_use'].map({'是': 1, '否': 0})
df_ml['social_participation'] = df_ml['social_participation'].map({'是': 1, '否': 0})
df_ml['srh'] = df_ml['srh'].map({'很好': 1, '较好': 2, '一般': 3, '较差': 4, '很差': 5})

for col in ['hypertension', 'diabetes', 'heart_disease', 'stroke_disease', 'cancer']:
    df_ml[col] = df_ml[col].map({'是': 1, '否': 0})

numeric_cols = ['age', 'gender', 'education', 'married', 'rural', 'adl', 'depression', 'cognition_z']
for col in numeric_cols:
    df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')

df_ml = df_ml[df_ml['age'] >= 45]

# 创建结果
df_ml['good_srh'] = (df_ml['srh'] <= 2).astype(float)
df_ml['no_adl'] = (df_ml['adl'] == 0).astype(float)
df_ml['not_depressed'] = (df_ml['depression'] < 10).astype(float)
df_ml['good_cognition'] = (df_ml['cognition_z'] > -1.5).astype(float)
df_ml['disease_count'] = df_ml[['hypertension', 'diabetes', 'heart_disease', 'stroke_disease', 'cancer']].sum(axis=1)
df_ml['no_multimorbidity'] = (df_ml['disease_count'] < 2).astype(float)

df_ml['healthy_aging_score'] = df_ml[['good_srh', 'no_adl', 'not_depressed', 'good_cognition', 'no_multimorbidity']].sum(axis=1)
df_ml['healthy_aging_cutoff4'] = (df_ml['healthy_aging_score'] >= 4).astype(float)

df_ml['digital_connected'] = ((df_ml['internet_use'] == 1) | (df_ml['social_participation'] == 1)).astype(float)

# 特征
df_ml['female'] = (df_ml['gender'] == 2).astype(float)
df_ml['rural_binary'] = (df_ml['rural'] == 1).astype(float)
df_ml['education_years'] = df_ml['education'].map({1: 0, 2: 6, 3: 9, 4: 12, 5: 12, 6: 16, 7: 19}).fillna(9)

features_ml = ['digital_connected', 'age', 'female', 'education_years', 'rural_binary',
               'hypertension', 'diabetes', 'heart_disease']

df_clean = df_ml[features_ml + ['healthy_aging_cutoff4']].dropna()
X = df_clean[features_ml]
y = df_clean['healthy_aging_cutoff4']

print(f"✓ 样本: {len(X):,}, 特征: {len(features_ml)}")

# 标准化（用于Neural Network）
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# ============================================================================
# 定义所有模型
# ============================================================================
print("\n" + "="*80)
print("定义模型...")
print("="*80)

models = {}

# 1. 基准模型（已有）
models['XGBoost'] = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    random_state=42, use_label_encoder=False, eval_metric='logloss'
)

models['LightGBM'] = lgb.LGBMClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    random_state=42, verbose=-1
)

# 2. 新算法1: CatBoost (2017)
if CATBOOST_AVAILABLE:
    models['CatBoost'] = cb.CatBoostClassifier(
        iterations=100, depth=4, learning_rate=0.05,
        random_state=42, verbose=False
    )
    print("✓ CatBoost 已加入")

# 3. 新算法2: HistGradientBoosting (sklearn 2019+)
models['HistGradientBoosting'] = HistGradientBoostingClassifier(
    max_iter=100, max_depth=4, learning_rate=0.05,
    random_state=42
)

# 4. 新算法3: Neural Network (MLP)
models['Neural Network'] = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    max_iter=200,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2
)

# 5. 新算法4: Ensemble Stacking
print("\n构建Ensemble Stacking模型...")
base_estimators = [
    ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                              random_state=42, use_label_encoder=False, eval_metric='logloss')),
    ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                               random_state=42, verbose=-1)),
    ('hist', HistGradientBoostingClassifier(max_iter=100, max_depth=4, learning_rate=0.05,
                                             random_state=42))
]

if CATBOOST_AVAILABLE:
    base_estimators.append(
        ('cat', cb.CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05,
                                       random_state=42, verbose=False))
    )

models['Ensemble Stacking'] = StackingClassifier(
    estimators=base_estimators,
    final_estimator=xgb.XGBClassifier(n_estimators=50, max_depth=3,
                                      random_state=42, use_label_encoder=False, eval_metric='logloss'),
    cv=3
)

print(f"✓ 共{len(models)}个模型准备就绪")

# ============================================================================
# 5-Fold CV 测试
# ============================================================================
print("\n" + "="*80)
print("5-Fold Cross-Validation")
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for model_name, model in models.items():
    print(f"\n测试 {model_name}...")
    aucs = []

    # 判断是否需要标准化
    X_train_input = X_scaled if model_name == 'Neural Network' else X

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_input, y), 1):
        X_train = X_train_input.iloc[train_idx]
        X_val = X_train_input.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            aucs.append(auc)

            if fold == 1:
                print(f"  Fold {fold}: AUC={auc:.4f}")
        except Exception as e:
            print(f"  ✗ Fold {fold} 失败: {str(e)[:50]}")
            continue

    if aucs:
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

        results.append({
            'model': model_name,
            'auc_mean': mean_auc,
            'auc_std': std_auc,
            'auc_min': np.min(aucs),
            'auc_max': np.max(aucs)
        })

# ============================================================================
# 结果对比
# ============================================================================
print("\n" + "="*80)
print("最终结果对比")
print("="*80)

df_results = pd.DataFrame(results).sort_values('auc_mean', ascending=False)

print("\n排名:")
for i, row in df_results.iterrows():
    rank_emoji = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else ""))
    print(f"  {rank_emoji} {row['model']:25s}: AUC={row['auc_mean']:.4f} ± {row['auc_std']:.4f} (range: {row['auc_min']:.4f}-{row['auc_max']:.4f})")

# 找出最佳模型
best_model = df_results.iloc[0]
improvement = (best_model['auc_mean'] - 0.7215) / 0.7215 * 100

print(f"\n🏆 最佳模型: {best_model['model']}")
print(f"   AUC = {best_model['auc_mean']:.4f}")
print(f"   相比原XGBoost (0.7215): {improvement:+.2f}%")

# ============================================================================
# 保存结果
# ============================================================================
print("\n" + "="*80)
print("保存结果...")
print("="*80)

output_dir = Path("../ml_analysis_output")
output_dir.mkdir(exist_ok=True)

df_results.to_csv(output_dir / "new_algorithms_comparison.csv", index=False)
print("✓ new_algorithms_comparison.csv")

# 可视化
fig, ax = plt.subplots(figsize=(12, 6))

# 条形图
bars = ax.barh(df_results['model'], df_results['auc_mean'],
               xerr=df_results['auc_std'], capsize=5,
               color=['gold' if i==0 else 'silver' if i==1 else 'chocolate' if i==2 else 'steelblue'
                      for i in range(len(df_results))])

# 添加数值标签
for i, (idx, row) in enumerate(df_results.iterrows()):
    ax.text(row['auc_mean'] + 0.002, i, f"{row['auc_mean']:.4f}",
            va='center', fontsize=10, fontweight='bold')

# 参考线：原XGBoost
ax.axvline(0.7215, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Original XGBoost (0.7215)')

ax.set_xlabel('AUC (5-Fold CV)', fontsize=12, fontweight='bold')
ax.set_ylabel('Model', fontsize=12, fontweight='bold')
ax.set_title('Algorithm Performance Comparison (2024-2025 Latest)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='x')
ax.set_xlim(0.70, 0.74)

plt.tight_layout()
fig_dir = output_dir / "figures_publication"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / 'Algorithm_Comparison_2024.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Algorithm_Comparison_2024.png")

# ============================================================================
# 最终总结
# ============================================================================
print("\n" + "="*80)
print("✓✓✓ 新算法测试完成")
print("="*80)
print(f"\n关键发现:")
print(f"  1. 测试了 {len(models)} 个最新算法")
print(f"  2. 最佳模型: {best_model['model']} (AUC={best_model['auc_mean']:.4f})")
print(f"  3. 性能提升: {improvement:+.2f}% vs 原XGBoost")
if improvement > 1:
    print(f"  4. ✅ 建议更新论文使用 {best_model['model']}")
else:
    print(f"  4. ⚠️ 提升有限，原XGBoost已经很好")
print(f"\n✓ 结果保存在: {output_dir.resolve()}")
print("="*80 + "\n")
