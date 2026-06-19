"""
清洗和整合多国Harmonized数据
统一变量名，创建分析数据集
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("多国Harmonized数据清洗和整合")
print("="*80)

base_path = Path("F:/目前养老官方数据库FOR NC/数据库新20260615/harmonized数据")

# ============================================================================
# STEP 1: 读取各国数据（先读前5000行测试）
# ============================================================================
print("\nSTEP 1: 读取各国数据...")

def load_country_data(country, file_path, test_mode=True):
    """读取单个国家数据"""
    try:
        if not Path(file_path).exists():
            print(f"  ✗ {country}: 文件不存在")
            return None

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, low_memory=False)
        else:
            df = pd.read_stata(file_path)

        if test_mode:
            df = df.head(5000)  # 测试模式只读5000行

        df['cohort'] = country
        print(f"  ✓ {country:10s}: {len(df):,} 行, {len(df.columns)} 列")
        return df
    except Exception as e:
        print(f"  ✗ {country:10s}: {str(e)[:60]}")
        return None

# 定义数据路径
data_files = {
    'CHARLS': base_path / "CHARLS_Harmonized数据/H_CHARLS_D_Data_extracted_20250906_165246.csv",
    'HRS': base_path / "HRS_Harmonized数据/H_HRS_d_stata/H_HRS_d.dta",
    'ELSA': base_path / "ELSA_Harmonized数据/Harmonized_ELSA_G3/h_elsa_g3.dta",
    'KLoSA': base_path / "KLoSA_Harmonized数据/H_KLoSA_e3.dta",
}

country_dfs = {}
for country, file_path in data_files.items():
    df = load_country_data(country, str(file_path), test_mode=True)
    if df is not None:
        country_dfs[country] = df

print(f"\n✓ 成功读取 {len(country_dfs)} 个国家")

# ============================================================================
# STEP 2: 检查关键变量的列名
# ============================================================================
print("\n" + "="*80)
print("STEP 2: 检查各国的关键变量列名")
print("="*80)

# 关键变量的可能列名
key_vars_pattern = {
    'id': ['hhidpn', 'id', 'ID', 'hhid', 'houseid'],
    'wave': ['wave', 'inw', 'raiwwave', 'Wave'],
    'age': ['age', 'r.*age', 'raage', 'rage'],
    'sex': ['sex', 'gender', 'ragender', 'r.*gender'],
    'education': ['educ', 'raeduc', 'education', 'r.*educ'],
    'internet': ['internet', 'r.*internet', 'r.*computer', 'r.*email'],
    'social': ['social', 'r.*social', 'r.*activity'],
    'srh': ['srh', 'r.*health', 'shlt', 'srhlt'],
    'cognition': ['cog', 'r.*cog', 'cognition', 'memory'],
    'depression': ['cesd', 'depression', 'r.*depress'],
    'adl': ['adl', 'r.*adl'],
}

for country, df in country_dfs.items():
    print(f"\n{country} 列名样例（前20个）:")
    print(df.columns[:20].tolist())

# ============================================================================
# STEP 3: 提取和标准化变量（示例：CHARLS）
# ============================================================================
print("\n" + "="*80)
print("STEP 3: 标准化CHARLS变量（作为示例）")
print("="*80)

if 'CHARLS' in country_dfs:
    df_charls = country_dfs['CHARLS']

    # 手动映射CHARLS变量（基于你之前用的列名）
    charls_mapping = {
        'ID (受访者编码)': 'participant_id',
        'wave (第几波调查)': 'wave',
        'age (年龄)': 'age',
        'ragender (性别)': 'sex',
        'raeducl (教育统一分类)': 'education',
        'social10 (上网)': 'internet_use',
        'socwk (是否每月参与社交)': 'social_participation',
        'srh (自评健康)': 'srh',
        'adlab_c (ADL(6项有困难))': 'adl',
        'cesd10 (心理健康(30分,越大越差))': 'depression',
        'tcog_z_z (认知/总认知能力z标准化(ref1))': 'cognition_z',
        'hibpe (高血压)': 'hypertension',
        'diabe (糖尿病)': 'diabetes',
        'hearte (心脏病)': 'heart_disease',
    }

    # 选择存在的列
    available_cols = [col for col in charls_mapping.keys() if col in df_charls.columns]
    df_charls_clean = df_charls[available_cols].copy()
    df_charls_clean = df_charls_clean.rename(columns=charls_mapping)

    # 转换类型
    if 'internet_use' in df_charls_clean.columns:
        df_charls_clean['internet_use'] = df_charls_clean['internet_use'].map({'是': 1, '否': 0})
    if 'social_participation' in df_charls_clean.columns:
        df_charls_clean['social_participation'] = df_charls_clean['social_participation'].map({'是': 1, '否': 0})
    if 'srh' in df_charls_clean.columns:
        df_charls_clean['srh'] = df_charls_clean['srh'].map({'很好': 1, '较好': 2, '一般': 3, '较差': 4, '很差': 5})

    for col in ['hypertension', 'diabetes', 'heart_disease']:
        if col in df_charls_clean.columns:
            df_charls_clean[col] = df_charls_clean[col].map({'是': 1, '否': 0})

    df_charls_clean['country'] = 'China'
    df_charls_clean['cohort'] = 'CHARLS'

    print(f"\n✓ CHARLS清洗完成: {len(df_charls_clean)} 行, {len(df_charls_clean.columns)} 列")
    print(f"标准化的列: {df_charls_clean.columns.tolist()}")

# ============================================================================
# STEP 4: 保存清洗后的数据
# ============================================================================
print("\n" + "="*80)
print("STEP 4: 保存清洗后的数据...")
print("="*80)

output_dir = Path("../ml_analysis_output/harmonized_clean")
output_dir.mkdir(exist_ok=True, parents=True)

if 'CHARLS' in country_dfs and 'df_charls_clean' in locals():
    output_file = output_dir / "charls_clean.csv"
    df_charls_clean.to_csv(output_file, index=False)
    print(f"✓ CHARLS: {output_file}")

# ============================================================================
# STEP 5: 提供其他国家的变量映射模板
# ============================================================================
print("\n" + "="*80)
print("STEP 5: 其他国家需要的变量映射模板")
print("="*80)

print("""
# HRS变量映射（需要根据实际列名调整）
hrs_mapping = {
    'hhidpn': 'participant_id',
    'raiwwave': 'wave',
    'raage': 'age',
    'ragender': 'sex',
    'raeduc': 'education',
    'r*internet': 'internet_use',  # 需要确认实际列名
    'r*social': 'social_participation',
    'r*shlt': 'srh',
    # ... 其他变量
}

# ELSA变量映射
elsa_mapping = {
    # 待补充
}

# KLoSA变量映射
klosa_mapping = {
    # 待补充
}
""")

# ============================================================================
# 最终总结
# ============================================================================
print("\n" + "="*80)
print("✓✓✓ 数据清洗流程完成")
print("="*80)
print(f"\n当前状态:")
print(f"  - CHARLS: 已清洗完成 ✅")
print(f"  - HRS: 需要变量映射 ⚠️")
print(f"  - ELSA: 需要变量映射 ⚠️")
print(f"  - KLoSA: 需要变量映射 ⚠️")
print(f"\n下一步:")
print(f"  1. 检查HRS/ELSA/KLoSA的实际列名")
print(f"  2. 创建对应的变量映射字典")
print(f"  3. 运行完整的清洗流程")
print(f"  4. 合并所有国家数据")
print("="*80 + "\n")
