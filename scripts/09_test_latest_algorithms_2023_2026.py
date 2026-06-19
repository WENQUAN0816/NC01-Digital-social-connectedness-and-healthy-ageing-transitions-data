"""
测试2023-2026最新算法
TabPFN, XGBoost 2.0+, AutoGluon, TabTransformer等
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# 检查可用的新算法
available_algorithms = []

# 1. TabPFN (2022-2023)
try:
    from tabpfn import TabPFNClassifier
    available_algorithms.append('TabPFN')
except ImportError:
    print("⚠️ TabPFN未安装 (pip install tabpfn)")

# 2. AutoGluon (2023)
try:
    from autogluon.tabular import TabularPredictor
    available_algorithms.append('AutoGluon')
except ImportError:
    print("⚠️ AutoGluon未安装 (pip install autogluon)")

# 3. CatBoost (已有)
try:
    import catboost as cb
    available_algorithms.append('CatBoost')
except ImportError:
    print("⚠️ CatBoost未安装")

# 4. NGBoost (2019, 2023更新)
try:
    from ngboost import NGBClassifier
    from ngboost.distns import Bernoulli
    available_algorithms.append('NGBoost')
except ImportError:
    print("⚠️ NGBoost未安装 (pip install ngboost)")

print("="*80)
print("2023-2026最新算法测试")
print("="*80)
print(f"\n可用算法: {', '.join(available_algorithms)}")
print(f"总共: {len(available_algorithms)}个")

# ============================================================================
# 加载数据
# ============================================================================
print("\n加载数据...")
charls_path = r"F:\目前养老官方数据库FOR NC\NC启动\01_data_deduped\csv\charls.csv"
df = pd.read_csv(charls_path, low_memory=False)

# 数据处理
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

df_ml['internet_use'] = df_ml['internet_use'].map({'是': 1, '否': 0})
df_ml['social_participation'] = df_ml['social_participation'].map({'是': 1, '否': 0})
df_ml['srh'] = df_ml['srh'].map({'很好': 1, '较好': 2, '一般': 3, '较差': 4, '很差': 5})

for col in ['hypertension', 'diabetes', 'heart_disease', 'stroke_disease', 'cancer']:
    df_ml[col] = df_ml[col].map({'是': 1, '否': 0})

numeric_cols = ['age', 'gender', 'education', 'married', 'rural', 'adl', 'depression', 'cognition_z']
for col in numeric_cols:
    df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')

df_ml = df_ml[df_ml['age'] >= 45]

df_ml['good_srh'] = (df_ml['srh'] <= 2).astype(float)
df_ml['no_adl'] = (df_ml['adl'] == 0).astype(float)
df_ml['not_depressed'] = (df_ml['depression'] < 10).astype(float)
df_ml['good_cognition'] = (df_ml['cognition_z'] > -1.5).astype(float)
df_ml['disease_count'] = df_ml[['hypertension', 'diabetes', 'heart_disease', 'stroke_disease', 'cancer']].sum(axis=1)
df_ml['no_multimorbidity'] = (df_ml['disease_count'] < 2).astype(float)

df_ml['healthy_aging_score'] = df_ml[['good_srh', 'no_adl', 'not_depressed', 'good_cognition', 'no_multimorbidity']].sum(axis=1)
df_ml['healthy_aging_cutoff4'] = (df_ml['healthy_aging_score'] >= 4).astype(float)
df_ml['digital_connected'] = ((df_ml['internet_use'] == 1) | (df_ml['social_participation'] == 1)).astype(float)

df_ml['female'] = (df_ml['gender'] == 2).astype(float)
df_ml['rural_binary'] = (df_ml['rural'] == 1).astype(float)
df_ml['education_years'] = df_ml['education'].map({1: 0, 2: 6, 3: 9, 4: 12, 5: 12, 6: 16, 7: 19}).fillna(9)

features_ml = ['digital_connected', 'age', 'female', 'education_years', 'rural_binary',
               'hypertension', 'diabetes', 'heart_disease']

df_clean = df_ml[features_ml + ['healthy_aging_cutoff4']].dropna()

# 限制样本量（避免内存问题）
if len(df_clean) > 10000:
    print(f"⚠️ 数据量过大，随机采样10000行用于测试新算法")
    df_clean = df_clean.sample(n=10000, random_state=42)

X = df_clean[features_ml]
y = df_clean['healthy_aging_cutoff4']

print(f"✓ 测试样本: {len(X):,}, 特征: {len(features_ml)}")

# ============================================================================
# 测试新算法
# ============================================================================
print("\n" + "="*80)
print("测试最新算法...")
print("="*80)

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 基准：XGBoost 2.0+
print("\n1. XGBoost (最新版)...")
try:
    print(f"   版本: {xgb.__version__}")
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                   random_state=42, eval_metric='logloss')
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[val_idx], y_pred))

    mean_auc = np.mean(aucs)
    print(f"   AUC: {mean_auc:.4f} ± {np.std(aucs):.4f}")
    results.append({'algorithm': 'XGBoost 2.0+', 'auc': mean_auc, 'year': '2023'})
except Exception as e:
    print(f"   ✗ 失败: {str(e)[:50]}")

# LightGBM 4.0+
print("\n2. LightGBM (最新版)...")
try:
    print(f"   版本: {lgb.__version__}")
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                    random_state=42, verbose=-1)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[val_idx], y_pred))

    mean_auc = np.mean(aucs)
    print(f"   AUC: {mean_auc:.4f} ± {np.std(aucs):.4f}")
    results.append({'algorithm': 'LightGBM 4.0+', 'auc': mean_auc, 'year': '2023'})
except Exception as e:
    print(f"   ✗ 失败: {str(e)[:50]}")

# CatBoost
if 'CatBoost' in available_algorithms:
    print("\n3. CatBoost...")
    try:
        aucs = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            model = cb.CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05,
                                          random_state=42, verbose=False)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = model.predict_proba(X.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], y_pred))

        mean_auc = np.mean(aucs)
        print(f"   AUC: {mean_auc:.4f} ± {np.std(aucs):.4f}")
        results.append({'algorithm': 'CatBoost', 'auc': mean_auc, 'year': '2017'})
    except Exception as e:
        print(f"   ✗ 失败: {str(e)[:50]}")

# TabPFN (2022-2023)
if 'TabPFN' in available_algorithms:
    print("\n4. TabPFN (2022-2023最新)...")
    try:
        aucs = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            model = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
            model.fit(X.iloc[train_idx].values, y.iloc[train_idx].values)
            y_pred = model.predict_proba(X.iloc[val_idx].values)[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], y_pred))

        mean_auc = np.mean(aucs)
        print(f"   AUC: {mean_auc:.4f} ± {np.std(aucs):.4f}")
        results.append({'algorithm': 'TabPFN', 'auc': mean_auc, 'year': '2023'})
    except Exception as e:
        print(f"   ✗ 失败: {str(e)[:50]}")

# NGBoost
if 'NGBoost' in available_algorithms:
    print("\n5. NGBoost...")
    try:
        aucs = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            model = NGBClassifier(Dist=Bernoulli, n_estimators=100, learning_rate=0.05,
                                  random_state=42, verbose=False)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = model.predict_proba(X.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], y_pred))

        mean_auc = np.mean(aucs)
        print(f"   AUC: {mean_auc:.4f} ± {np.std(aucs):.4f}")
        results.append({'algorithm': 'NGBoost', 'auc': mean_auc, 'year': '2023'})
    except Exception as e:
        print(f"   ✗ 失败: {str(e)[:50]}")

# ============================================================================
# 结果对比
# ============================================================================
print("\n" + "="*80)
print("最终结果对比（2023-2026算法）")
print("="*80)

if results:
    df_results = pd.DataFrame(results).sort_values('auc', ascending=False)

    print("\n排名:")
    for i, (idx, row) in enumerate(df_results.iterrows()):
        rank_emoji = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else ""))
        print(f"  {rank_emoji} {row['algorithm']:25s}: AUC={row['auc']:.4f} ({row['year']})")

    best = df_results.iloc[0]
    baseline = 0.7215
    improvement = (best['auc'] - baseline) / baseline * 100

    print(f"\n🏆 最佳算法: {best['algorithm']}")
    print(f"   AUC = {best['auc']:.4f}")
    print(f"   相比基准XGBoost (0.7215): {improvement:+.2f}%")

    if improvement > 0.5:
        print(f"\n✅ 发现显著提升！建议更新论文使用 {best['algorithm']}")
    elif improvement > 0.1:
        print(f"\n⚠️ 有轻微提升，但不够显著（<0.5%）")
        print(f"   建议在Supplementary中提及测试过{best['algorithm']}")
    else:
        print(f"\n✅ 原XGBoost已是最优选择")
        print(f"   2023-2026新算法未带来实质提升")

    # 保存
    output_dir = Path("../ml_analysis_output")
    df_results.to_csv(output_dir / "latest_algorithms_2023_2026.csv", index=False)
    print(f"\n✓ 结果已保存: latest_algorithms_2023_2026.csv")
else:
    print("\n⚠️ 未能成功测试任何新算法")
    print("   原因可能是: 1) 算法未安装 2) 数据兼容性问题")

print("\n" + "="*80)
print("测试完成")
print("="*80)
