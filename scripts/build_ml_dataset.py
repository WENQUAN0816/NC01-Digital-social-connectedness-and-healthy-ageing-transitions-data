"""
从原始 CHARLS CSV 构建 ML 分析数据集
"""

import pandas as pd
import numpy as np
from pathlib import Path

def build_ml_dataset():
    """构建机器学习分析数据集"""

    print("Loading CHARLS data...")
    charls_path = r"F:\目前养老官方数据库FOR NC\NC启动\01_data_deduped\csv\charls.csv"

    df = pd.read_csv(charls_path, low_memory=False)
    print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # 变量映射
    var_map = {
        # 基础人口学
        'ID': 'participant_id',
        'wave': 'wave',
        'age (年龄)': 'age',
        'ragender (性别)': 'female',
        'raeducl (教育统一分类)': 'education_group',
        'marry (婚姻)': 'married',
        'hrural (居住在农村或城市)': 'rural',

        # 暴露变量
        'social10 (上网)': 'internet_use',
        'socwk (每周至少参加一次社交活动)': 'social_participation',

        # 健康结果
        'srh (自评健康)': 'srh',
        'adlab_c (ADL(6项有困难))': 'adl_limitation',
        'iadl (IADL(5项有困难))': 'iadl_limitation',
        'cesd10 (心理健康(30分': 'depression_score',
        'tcog_z_z (认知能力z分)': 'cognition_z',
        'frailtyb (虚弱指数b)': 'frailty_binary',

        # 疾病
        'hibpe (高血压)': 'hypertension',
        'diabe (糖尿病)': 'diabetes',
        'hearte (心脏病)': 'heart_disease',
        'stroke (中风)': 'stroke',
        'cancre (癌症)': 'cancer',
    }

    # 重命名存在的列
    rename_dict = {}
    for old_name, new_name in var_map.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name

    df = df.rename(columns=rename_dict)

    print(f"\n✓ Mapped {len(rename_dict)} variables")

    # 创建 healthy_aging_binary
    # 定义为：good SRH + no ADL + no depression + good cognition

    # 转换数值类型
    numeric_cols = ['age', 'internet_use', 'social_participation', 'srh',
                    'adl_limitation', 'iadl_limitation', 'depression_score',
                    'cognition_z', 'frailty_binary',
                    'hypertension', 'diabetes', 'heart_disease', 'stroke', 'cancer',
                    'married', 'rural', 'female']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1. Good self-rated health (1-2 = good/very good, 3-5 = fair/poor)
    if 'srh' in df.columns:
        df['good_srh'] = (df['srh'] <= 2).astype(float)

    # 2. No ADL limitation
    if 'adl_limitation' in df.columns:
        df['no_adl'] = (df['adl_limitation'] == 0).astype(float)

    # 3. Not depressed (CESD-10 < 10 为正常)
    if 'depression_score' in df.columns:
        df['not_depressed'] = (df['depression_score'] < 10).astype(float)

    # 4. Good cognition (z-score > -1.5)
    if 'cognition_z' in df.columns:
        df['good_cognition'] = (df['cognition_z'] > -1.5).astype(float)

    # 5. No multimorbidity (< 2 chronic diseases)
    disease_cols = [c for c in ['hypertension', 'diabetes', 'heart_disease', 'stroke', 'cancer'] if c in df.columns]
    if disease_cols:
        df['disease_count'] = df[disease_cols].sum(axis=1)
        df['no_multimorbidity'] = (df['disease_count'] < 2).astype(float)

    # 计算 healthy aging score (0-5)
    component_cols = ['good_srh', 'no_adl', 'not_depressed', 'good_cognition', 'no_multimorbidity']
    available_components = [c for c in component_cols if c in df.columns]

    if available_components:
        df['healthy_aging_score'] = df[available_components].sum(axis=1)
        df['healthy_aging_binary'] = (df['healthy_aging_score'] >= 4).astype(float)
        print(f"\n✓ Created healthy_aging from {len(available_components)} components")

    # 处理分类变量
    if 'female' in df.columns:
        df['female'] = df['female'].map({1: 0, 2: 1})  # 1=male, 2=female

    if 'education_group' in df.columns:
        df['education_numeric'] = df['education_group'].map({
            1: 0,  # illiterate
            2: 1,  # elementary
            3: 2,  # middle
            4: 3,  # high school
            5: 4,  # vocational
            6: 5,  # college+
        })

    if 'rural' in df.columns:
        df['rural_binary'] = (df['rural'] == 1).astype(float)  # 1=rural, 0=urban

    if 'married' in df.columns:
        df['married_binary'] = df['married'].isin([1, 2]).astype(float)  # 1=married, 2=partnered

    # 选择分析列
    analysis_cols = [
        'participant_id', 'wave', 'age', 'female',
        'education_numeric', 'married_binary', 'rural_binary',
        'internet_use', 'social_participation',
        'cognition_z', 'depression_score', 'adl_limitation',
        'disease_count', 'frailty_binary',
        'healthy_aging_score', 'healthy_aging_binary'
    ]

    available_cols = [c for c in analysis_cols if c in df.columns]
    df_clean = df[available_cols].copy()

    # 只保留年龄 >= 45 的样本
    if 'age' in df_clean.columns:
        df_clean = df_clean[df_clean['age'] >= 45]
        print(f"✓ Filtered to age >= 45: {len(df_clean):,} rows")

    # 统计
    print(f"\n{'='*80}")
    print("DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Total observations: {len(df_clean):,}")

    if 'participant_id' in df_clean.columns:
        print(f"Unique participants: {df_clean['participant_id'].nunique():,}")

    if 'wave' in df_clean.columns:
        print(f"Waves: {sorted(df_clean['wave'].unique())}")

    # 缺失值
    print(f"\nMissing values:")
    for col in df_clean.columns:
        missing_pct = df_clean[col].isna().mean() * 100
        if missing_pct > 0:
            print(f"  {col:30s}: {missing_pct:5.1f}%")

    # 分布统计
    if 'healthy_aging_binary' in df_clean.columns:
        ha_dist = df_clean['healthy_aging_binary'].value_counts()
        print(f"\nHealthy aging distribution:")
        for val, count in ha_dist.items():
            pct = count / len(df_clean) * 100
            print(f"  {val}: {count:,} ({pct:.1f}%)")

    if 'internet_use' in df_clean.columns:
        internet_dist = df_clean['internet_use'].value_counts()
        print(f"\nInternet use distribution:")
        for val, count in internet_dist.items():
            pct = count / len(df_clean) * 100
            print(f"  {val}: {count:,} ({pct:.1f}%)")

    # 保存
    output_dir = Path("../ml_analysis_output")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "charls_ml_dataset.csv"
    df_clean.to_csv(output_file, index=False)

    print(f"\n✓ Saved to: {output_file}")
    print(f"  Columns: {len(df_clean.columns)}")
    print(f"  Rows: {len(df_clean):,}")

    return df_clean

if __name__ == "__main__":
    df = build_ml_dataset()
