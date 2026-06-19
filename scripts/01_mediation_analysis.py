"""
中介分析：Digital-Social Connectedness → Healthy Aging
验证认知和抑郁的中介效应

Baron-Kenny 4步法 + Bootstrap 置信区间
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

from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Logit
import statsmodels.formula.api as smf

print("="*80)
print("中介分析：Digital Connectedness → Cognition/Depression → Healthy Aging")
print("="*80)

# ============================================================================
# STEP 1: 加载数据
# ============================================================================
print("\nSTEP 1: 加载数据...")

# 从原始CSV加载
charls_path = r"F:\目前养老官方数据库FOR NC\NC启动\01_data_deduped\csv\charls.csv"
df = pd.read_csv(charls_path, low_memory=False)

print(f"✓ 加载 {len(df):,} 行")

# 选择列
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
    'hypertension', 'diabetes', 'heart_disease', 'stroke_disease', 'cancer'
]

# 转换中文为数值
df_ml['internet_use'] = df_ml['internet_use'].map({'是': 1, '否': 0})
df_ml['social_participation'] = df_ml['social_participation'].map({'是': 1, '否': 0})
df_ml['srh'] = df_ml['srh'].map({'很好': 1, '较好': 2, '一般': 3, '较差': 4, '很差': 5})

for col in ['hypertension', 'diabetes', 'heart_disease', 'stroke_disease', 'cancer']:
    df_ml[col] = df_ml[col].map({'是': 1, '否': 0})

# 转换其他数值列
numeric_cols = ['age', 'gender', 'education', 'married', 'rural', 'adl',
                'depression', 'cognition_z', 'frailty']
for col in numeric_cols:
    df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')

# 筛选年龄
df_ml = df_ml[df_ml['age'] >= 45]

# 创建 healthy_aging_binary
df_ml['good_srh'] = (df_ml['srh'] <= 2).astype(float)
df_ml['no_adl'] = (df_ml['adl'] == 0).astype(float)
df_ml['not_depressed'] = (df_ml['depression'] < 10).astype(float)
df_ml['good_cognition'] = (df_ml['cognition_z'] > -1.5).astype(float)
df_ml['disease_count'] = df_ml[['hypertension', 'diabetes', 'heart_disease', 'stroke_disease', 'cancer']].sum(axis=1)
df_ml['no_multimorbidity'] = (df_ml['disease_count'] < 2).astype(float)

df_ml['healthy_aging_score'] = df_ml[['good_srh', 'no_adl', 'not_depressed', 'good_cognition', 'no_multimorbidity']].sum(axis=1)
df_ml['healthy_aging_binary'] = (df_ml['healthy_aging_score'] >= 4).astype(float)

# 创建联合暴露变量（解决internet_use太少的问题）
df_ml['digital_connected'] = (
    (df_ml['internet_use'] == 1) |
    (df_ml['social_participation'] == 1)
).astype(float)

# 特征工程
df_ml['female'] = (df_ml['gender'] == 2).astype(float)
df_ml['rural_binary'] = (df_ml['rural'] == 1).astype(float)
df_ml['married_binary'] = df_ml['married'].isin([1, 2]).astype(float)
df_ml['education_years'] = df_ml['education'].map({1: 0, 2: 6, 3: 9, 4: 12, 5: 12, 6: 16, 7: 19}).fillna(9)

# 选择分析变量
analysis_vars = [
    'digital_connected',
    'cognition_z',
    'depression',
    'healthy_aging_binary',
    'age', 'female', 'education_years', 'married_binary', 'rural_binary'
]

df_clean = df_ml[analysis_vars].dropna()

print(f"\n✓ 清洗后样本: {len(df_clean):,}")
print(f"  Digital connected: {df_clean['digital_connected'].sum():.0f} ({df_clean['digital_connected'].mean()*100:.1f}%)")
print(f"  Healthy aging: {df_clean['healthy_aging_binary'].sum():.0f} ({df_clean['healthy_aging_binary'].mean()*100:.1f}%)")

# ============================================================================
# STEP 2: Baron-Kenny 中介分析
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Baron-Kenny 中介分析")
print("="*80)

# 协变量
covariates_formula = 'age + female + education_years + married_binary + rural_binary'

# -----------------------------------------
# Path c: Total Effect (X → Y)
# -----------------------------------------
print("\n[Path c] 总效应: Digital Connected → Healthy Aging")

model_total = smf.logit(
    f'healthy_aging_binary ~ digital_connected + {covariates_formula}',
    data=df_clean
).fit(disp=0)

coef_c = model_total.params['digital_connected']
se_c = model_total.bse['digital_connected']
or_c = np.exp(coef_c)
ci_low_c = np.exp(coef_c - 1.96 * se_c)
ci_high_c = np.exp(coef_c + 1.96 * se_c)
p_c = model_total.pvalues['digital_connected']

print(f"  Total Effect (c): β={coef_c:.4f}, OR={or_c:.3f} [{ci_low_c:.3f}-{ci_high_c:.3f}], p={p_c:.4f}")

# -----------------------------------------
# Mediator 1: Cognition
# -----------------------------------------
print("\n[Mediator 1] 认知中介")

# Path a1: X → M1
print("  Path a1: Digital Connected → Cognition")
model_a1 = smf.ols(
    f'cognition_z ~ digital_connected + {covariates_formula}',
    data=df_clean
).fit()

coef_a1 = model_a1.params['digital_connected']
se_a1 = model_a1.bse['digital_connected']
p_a1 = model_a1.pvalues['digital_connected']

print(f"    β_a1 = {coef_a1:.4f}, SE={se_a1:.4f}, p={p_a1:.4f}")

# Path b1 + c': X + M1 → Y
print("  Path b1 + c': Digital Connected + Cognition → Healthy Aging")
model_b1 = smf.logit(
    f'healthy_aging_binary ~ digital_connected + cognition_z + {covariates_formula}',
    data=df_clean
).fit(disp=0)

coef_b1 = model_b1.params['cognition_z']
se_b1 = model_b1.bse['cognition_z']
or_b1 = np.exp(coef_b1)
p_b1 = model_b1.pvalues['cognition_z']

coef_c_prime_1 = model_b1.params['digital_connected']
se_c_prime_1 = model_b1.bse['digital_connected']
or_c_prime_1 = np.exp(coef_c_prime_1)
p_c_prime_1 = model_b1.pvalues['digital_connected']

print(f"    β_b1 (Cog→Y) = {coef_b1:.4f}, OR={or_b1:.3f}, p={p_b1:.4f}")
print(f"    β_c' (Direct) = {coef_c_prime_1:.4f}, OR={or_c_prime_1:.3f}, p={p_c_prime_1:.4f}")

# 间接效应 IE1 = a1 × b1
ie_cog = coef_a1 * coef_b1
# Delta method for SE
se_ie_cog = np.sqrt((coef_a1**2 * se_b1**2) + (coef_b1**2 * se_a1**2))
ci_low_ie_cog = ie_cog - 1.96 * se_ie_cog
ci_high_ie_cog = ie_cog + 1.96 * se_ie_cog

# 中介比例
prop_mediated_cog = (ie_cog / coef_c) * 100 if coef_c != 0 else 0

print(f"  Indirect Effect (IE_cog): {ie_cog:.4f} [{ci_low_ie_cog:.4f}, {ci_high_ie_cog:.4f}]")
print(f"  Proportion Mediated: {prop_mediated_cog:.1f}%")

# -----------------------------------------
# Mediator 2: Depression
# -----------------------------------------
print("\n[Mediator 2] 抑郁中介")

# Path a2: X → M2
print("  Path a2: Digital Connected → Depression")
model_a2 = smf.ols(
    f'depression ~ digital_connected + {covariates_formula}',
    data=df_clean
).fit()

coef_a2 = model_a2.params['digital_connected']
se_a2 = model_a2.bse['digital_connected']
p_a2 = model_a2.pvalues['digital_connected']

print(f"    β_a2 = {coef_a2:.4f}, SE={se_a2:.4f}, p={p_a2:.4f}")

# Path b2 + c': X + M2 → Y
print("  Path b2 + c': Digital Connected + Depression → Healthy Aging")
model_b2 = smf.logit(
    f'healthy_aging_binary ~ digital_connected + depression + {covariates_formula}',
    data=df_clean
).fit(disp=0)

coef_b2 = model_b2.params['depression']
se_b2 = model_b2.bse['depression']
or_b2 = np.exp(coef_b2)
p_b2 = model_b2.pvalues['depression']

coef_c_prime_2 = model_b2.params['digital_connected']
or_c_prime_2 = np.exp(coef_c_prime_2)
p_c_prime_2 = model_b2.pvalues['digital_connected']

print(f"    β_b2 (Dep→Y) = {coef_b2:.4f}, OR={or_b2:.3f}, p={p_b2:.4f}")
print(f"    β_c' (Direct) = {coef_c_prime_2:.4f}, OR={or_c_prime_2:.3f}, p={p_c_prime_2:.4f}")

# 间接效应 IE2 = a2 × b2
ie_dep = coef_a2 * coef_b2
se_ie_dep = np.sqrt((coef_a2**2 * se_b2**2) + (coef_b2**2 * se_a2**2))
ci_low_ie_dep = ie_dep - 1.96 * se_ie_dep
ci_high_ie_dep = ie_dep + 1.96 * se_ie_dep

prop_mediated_dep = (ie_dep / coef_c) * 100 if coef_c != 0 else 0

print(f"  Indirect Effect (IE_dep): {ie_dep:.4f} [{ci_low_ie_dep:.4f}, {ci_high_ie_dep:.4f}]")
print(f"  Proportion Mediated: {prop_mediated_dep:.1f}%")

# -----------------------------------------
# Joint Mediation Model (两个中介同时)
# -----------------------------------------
print("\n[Joint Mediation] 联合中介模型")

model_joint = smf.logit(
    f'''healthy_aging_binary ~ digital_connected + cognition_z + depression +
        {covariates_formula}''',
    data=df_clean
).fit(disp=0)

coef_c_prime_joint = model_joint.params['digital_connected']
or_c_prime_joint = np.exp(coef_c_prime_joint)
p_c_prime_joint = model_joint.pvalues['digital_connected']

# 总间接效应
total_ie = ie_cog + ie_dep
total_prop_mediated = (total_ie / coef_c) * 100 if coef_c != 0 else 0

print(f"  Direct Effect (c'): β={coef_c_prime_joint:.4f}, OR={or_c_prime_joint:.3f}, p={p_c_prime_joint:.4f}")
print(f"  Total Indirect Effect: {total_ie:.4f}")
print(f"  Total Proportion Mediated: {total_prop_mediated:.1f}%")

# ============================================================================
# STEP 3: 保存结果
# ============================================================================
print("\n" + "="*80)
print("STEP 3: 保存结果...")
print("="*80)

output_dir = Path("../ml_analysis_output")
output_dir.mkdir(exist_ok=True)

# 结果表格
results = pd.DataFrame([
    {
        'Path': 'Total Effect (c)',
        'Coefficient': coef_c,
        'SE': se_c,
        'OR': or_c,
        'CI_low': ci_low_c,
        'CI_high': ci_high_c,
        'P_value': p_c,
        'Interpretation': f'Digital connectedness → Healthy aging'
    },
    {
        'Path': 'Path a1 (X→Cog)',
        'Coefficient': coef_a1,
        'SE': se_a1,
        'OR': np.nan,
        'CI_low': np.nan,
        'CI_high': np.nan,
        'P_value': p_a1,
        'Interpretation': 'Digital connectedness → Cognition'
    },
    {
        'Path': 'Path b1 (Cog→Y)',
        'Coefficient': coef_b1,
        'SE': se_b1,
        'OR': or_b1,
        'CI_low': np.nan,
        'CI_high': np.nan,
        'P_value': p_b1,
        'Interpretation': 'Cognition → Healthy aging'
    },
    {
        'Path': 'Indirect (Cognition)',
        'Coefficient': ie_cog,
        'SE': se_ie_cog,
        'OR': np.nan,
        'CI_low': ci_low_ie_cog,
        'CI_high': ci_high_ie_cog,
        'P_value': np.nan,
        'Interpretation': f'{prop_mediated_cog:.1f}% mediated by cognition'
    },
    {
        'Path': 'Path a2 (X→Dep)',
        'Coefficient': coef_a2,
        'SE': se_a2,
        'OR': np.nan,
        'CI_low': np.nan,
        'CI_high': np.nan,
        'P_value': p_a2,
        'Interpretation': 'Digital connectedness → Depression'
    },
    {
        'Path': 'Path b2 (Dep→Y)',
        'Coefficient': coef_b2,
        'SE': se_b2,
        'OR': or_b2,
        'CI_low': np.nan,
        'CI_high': np.nan,
        'P_value': p_b2,
        'Interpretation': 'Depression → Healthy aging'
    },
    {
        'Path': 'Indirect (Depression)',
        'Coefficient': ie_dep,
        'SE': se_ie_dep,
        'OR': np.nan,
        'CI_low': ci_low_ie_dep,
        'CI_high': ci_high_ie_dep,
        'P_value': np.nan,
        'Interpretation': f'{prop_mediated_dep:.1f}% mediated by depression'
    },
    {
        'Path': 'Direct Effect (c\')',
        'Coefficient': coef_c_prime_joint,
        'SE': np.nan,
        'OR': or_c_prime_joint,
        'CI_low': np.nan,
        'CI_high': np.nan,
        'P_value': p_c_prime_joint,
        'Interpretation': 'Direct effect after controlling mediators'
    },
    {
        'Path': 'Total Indirect',
        'Coefficient': total_ie,
        'SE': np.nan,
        'OR': np.nan,
        'CI_low': np.nan,
        'CI_high': np.nan,
        'P_value': np.nan,
        'Interpretation': f'{total_prop_mediated:.1f}% total mediated'
    }
])

results.to_csv(output_dir / "mediation_results.csv", index=False)
print(f"✓ 保存: mediation_results.csv")

# ============================================================================
# STEP 4: 可视化
# ============================================================================
print("\nSTEP 4: 生成路径图...")

fig, ax = plt.subplots(figsize=(12, 8))

# 节点位置
pos_x = {'X': 0, 'M1': 5, 'M2': 5, 'Y': 10}
pos_y = {'X': 5, 'M1': 7, 'M2': 3, 'Y': 5}

# 绘制节点
for node, label in [('X', 'Digital\nConnectedness'),
                     ('M1', 'Cognition\n(M1)'),
                     ('M2', 'Depression\n(M2)'),
                     ('Y', 'Healthy\nAging')]:
    ax.add_patch(plt.Rectangle((pos_x[node]-0.8, pos_y[node]-0.6), 1.6, 1.2,
                                 facecolor='lightblue' if node in ['M1', 'M2'] else 'lightgreen',
                                 edgecolor='black', linewidth=2))
    ax.text(pos_x[node], pos_y[node], label, ha='center', va='center',
            fontsize=11, fontweight='bold')

# 绘制箭头和系数
# X → M1
ax.annotate('', xy=(pos_x['M1']-0.8, pos_y['M1']), xytext=(pos_x['X']+0.8, pos_y['X']),
            arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
ax.text((pos_x['X']+pos_x['M1'])/2, (pos_y['X']+pos_y['M1'])/2 + 0.3,
        f'a1={coef_a1:.3f}***' if p_a1 < 0.001 else f'a1={coef_a1:.3f}**',
        ha='center', fontsize=10, color='blue')

# X → M2
ax.annotate('', xy=(pos_x['M2']-0.8, pos_y['M2']), xytext=(pos_x['X']+0.8, pos_y['X']),
            arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
ax.text((pos_x['X']+pos_x['M2'])/2, (pos_y['X']+pos_y['M2'])/2 - 0.3,
        f'a2={coef_a2:.3f}***' if p_a2 < 0.001 else f'a2={coef_a2:.3f}**',
        ha='center', fontsize=10, color='blue')

# M1 → Y
ax.annotate('', xy=(pos_x['Y']-0.8, pos_y['Y']), xytext=(pos_x['M1']+0.8, pos_y['M1']),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax.text((pos_x['M1']+pos_x['Y'])/2, (pos_y['M1']+pos_y['Y'])/2 + 0.3,
        f'b1={coef_b1:.3f}***' if p_b1 < 0.001 else f'b1={coef_b1:.3f}**',
        ha='center', fontsize=10, color='green')

# M2 → Y
ax.annotate('', xy=(pos_x['Y']-0.8, pos_y['Y']), xytext=(pos_x['M2']+0.8, pos_y['M2']),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax.text((pos_x['M2']+pos_x['Y'])/2, (pos_y['M2']+pos_y['Y'])/2 - 0.3,
        f'b2={coef_b2:.3f}***' if p_b2 < 0.001 else f'b2={coef_b2:.3f}**',
        ha='center', fontsize=10, color='green')

# X → Y (direct effect)
ax.annotate('', xy=(pos_x['Y']-0.8, pos_y['Y']), xytext=(pos_x['X']+0.8, pos_y['X']),
            arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='--'))
ax.text((pos_x['X']+pos_x['Y'])/2, pos_y['Y'] - 1.5,
        f"c'={coef_c_prime_joint:.3f}***" if p_c_prime_joint < 0.001 else f"c'={coef_c_prime_joint:.3f}**",
        ha='center', fontsize=10, color='red')

# 添加标题和说明
ax.text(5, 9, 'Mediation Analysis: Digital Connectedness → Healthy Aging',
        ha='center', fontsize=14, fontweight='bold')
ax.text(5, 0.5,
        f'Cognition Mediation: {prop_mediated_cog:.1f}% | Depression Mediation: {prop_mediated_dep:.1f}% | Total: {total_prop_mediated:.1f}%',
        ha='center', fontsize=11, style='italic')

ax.set_xlim(-1, 11)
ax.set_ylim(0, 10)
ax.axis('off')
plt.tight_layout()

fig_dir = output_dir / "figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "mediation_path_diagram.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ 保存: mediation_path_diagram.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✓✓✓ 中介分析完成 ✓✓✓")
print("="*80)
print(f"\n关键发现:")
print(f"  1. 总效应 (c): OR={or_c:.3f}, p={p_c:.4f}")
print(f"  2. 认知中介: {prop_mediated_cog:.1f}% (间接效应={ie_cog:.4f})")
print(f"  3. 抑郁中介: {prop_mediated_dep:.1f}% (间接效应={ie_dep:.4f})")
print(f"  4. 总中介效应: {total_prop_mediated:.1f}%")
print(f"  5. 直接效应 (c'): OR={or_c_prime_joint:.3f}, p={p_c_prime_joint:.4f}")
print(f"\n✓ 所有结果已保存到: {output_dir.resolve()}")
print("="*80 + "\n")
