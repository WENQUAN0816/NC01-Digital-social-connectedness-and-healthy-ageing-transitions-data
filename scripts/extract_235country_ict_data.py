"""
从235国数字经济指标数据中提取NC01论文所需的国家和指标
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("提取235国ICT数据 for NC01 Moderator Analysis")
print("="*80)

# 读取数据
file_path = r"F:\目前养老官方数据库\数据库新20260615\235个国家数字经济发展指标（2000-2022年）\235个国家数字经济发展指标（2000-2022年）.xls"

print("\n正在读取数据...")
df = pd.read_excel(file_path)
print(f"✓ 读取完成: {len(df):,} 行 × {len(df.columns)} 列")

# 定义国家映射（数据库中的名称 → 你的队列）
country_mapping = {
    # 直接匹配
    'China': 'China',
    'United States': 'United States',
    'United Kingdom': 'United Kingdom',
    'Mexico': 'Mexico',
    'Germany': 'Germany',
    'France': 'France',
    'Spain': 'Spain',
    'Italy': 'Italy',
    'Netherlands': 'Netherlands',
    'Belgium': 'Belgium',
    'Austria': 'Austria',
    'Denmark': 'Denmark',
    'Sweden': 'Sweden',
    'Switzerland': 'Switzerland',
    'Poland': 'Poland',
    'Greece': 'Greece',
    'Portugal': 'Portugal',

    # 可能需要匹配的变体
    'Korea, Rep.': 'South Korea',
    'Korea': 'South Korea',
    'Czech Republic': 'Czech Republic',
    'Israel': 'Israel',
}

# 查找实际的国家名
print("\n匹配国家名称:")
matched_countries = []
for search_name, standard_name in country_mapping.items():
    matches = df[df['CountryName'].str.contains(search_name, case=False, na=False, regex=False)]['CountryName'].unique()
    if len(matches) > 0:
        actual_name = matches[0]
        matched_countries.append((actual_name, standard_name))
        print(f"  ✓ {search_name:20s} → {actual_name}")
    else:
        print(f"  ✗ {search_name:20s} → 未找到")

# 筛选国家和年份
target_countries_raw = [c[0] for c in matched_countries]

df_filtered = df[
    (df['CountryName'].isin(target_countries_raw)) &
    (df['年份'] >= 2011) &
    (df['年份'] <= 2020)
].copy()

print(f"\n筛选后数据:")
print(f"  国家数: {df_filtered['CountryName'].nunique()}")
print(f"  年份: 2011-2020")
print(f"  总行数: {len(df_filtered)}")

# 添加标准化国家名
country_dict = dict(matched_countries)
df_filtered['country_standard'] = df_filtered['CountryName'].map(country_dict)

# 重命名列为英文
column_mapping = {
    'CountryName': 'country_raw',
    '年份': 'year',
    'CountryCode': 'iso3',
    '移动网络使用人数': 'mobile_network_users',
    '固定电话普及率': 'fixed_telephone_rate',
    '固定宽带普及率': 'fixed_broadband_rate',
    '安全互联网服务器（每百万人）': 'secure_internet_servers',
    'ICT服务出口': 'ict_service_exports',
    '信息通讯服务出口': 'ict_communication_exports',
    '营商便利度评分': 'business_ease_score',
    '创业便利度评分': 'startup_ease_score',
    '科技期刊文章': 'scientific_journal_articles',
    '研发支出占GDP比重': 'rd_gdp_ratio',
    '高等教育劳动力占比': 'higher_education_labor',
    '高等教育入学率': 'higher_education_enrollment',
    'ICT前沿技术准备度': 'ict_tech_readiness',
    '最新技术可获得度': 'latest_tech_availability',
    '政府决策透明度': 'govt_transparency',
    '电信服务收入美元': 'telecom_revenue_usd',
    '移动网络收入美元': 'mobile_revenue_usd',
    '风险资本可获得性': 'venture_capital_availability',
    'ICT相关论文发表量': 'ict_papers',
    '教育支出总额占比': 'education_expenditure',
    '政府效率': 'govt_efficiency',
    '物流绩效指数': 'logistics_index',
    '移动蜂窝订阅量（每100人）': 'mobile_cellular_per100'
}

df_filtered = df_filtered.rename(columns=column_mapping)

# 选择最重要的8个moderator指标
key_indicators = [
    'country_standard',
    'year',
    'iso3',
    'fixed_broadband_rate',           # 固定宽带普及率
    'mobile_cellular_per100',          # 移动蜂窝订阅量
    'higher_education_labor',          # 高等教育劳动力占比
    'rd_gdp_ratio',                    # 研发支出占GDP
    'govt_efficiency',                 # 政府效率
    'logistics_index',                 # 物流绩效指数
    'business_ease_score',             # 营商便利度
    'ict_papers',                      # ICT论文发表量
]

df_final = df_filtered[key_indicators].copy()

# 缺失值统计
print(f"\n缺失值统计 (2011-2020):")
for col in key_indicators[3:]:  # 跳过country, year, iso3
    missing = df_final[col].isna().sum()
    total = len(df_final)
    pct = missing / total * 100
    print(f"  {col:30s}: {missing:4d}/{total:4d} ({pct:5.1f}%)")

# 按国家汇总（2020年数据）
print(f"\n按国家查看 (2020年数据):")
df_2020 = df_final[df_final['year'] == 2020].sort_values('fixed_broadband_rate', ascending=False)
if len(df_2020) > 0:
    print(df_2020[['country_standard', 'fixed_broadband_rate', 'mobile_cellular_per100',
                   'govt_efficiency']].head(15).to_string(index=False))

# 保存
output_dir = Path("../ml_analysis_output")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "ict_indicators_235countries_extracted.csv"
df_final.to_csv(output_file, index=False)

print(f"\n{'='*80}")
print("✓ 数据提取完成!")
print(f"{'='*80}")
print(f"\n保存位置: {output_file}")
print(f"数据规模: {len(df_final)} 行 × {len(df_final.columns)} 列")
print(f"国家数: {df_final['country_standard'].nunique()}")
print(f"年份范围: {df_final['year'].min():.0f}-{df_final['year'].max():.0f}")

# 数据质量评估
print(f"\n数据质量评估:")
completeness = (1 - df_final[key_indicators[3:]].isna().sum().sum() /
                (len(df_final) * len(key_indicators[3:]))) * 100
print(f"  整体完整度: {completeness:.1f}%")

if completeness > 70:
    print(f"  评级: ⭐⭐⭐⭐⭐ 优秀")
elif completeness > 50:
    print(f"  评级: ⭐⭐⭐⭐ 良好")
else:
    print(f"  评级: ⭐⭐⭐ 中等")

print(f"\n{'='*80}\n")
