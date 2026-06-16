# 七国 HRS-family 数据第一版探索分析

本报告由 `scripts/harmonize_and_analyze_aging.py` 自动生成，目标是先验证 CHARLS/HRS/ELSA/SHARE/KLoSA/LASI/MHAS 简化数据能否按 Nature Communications 常见的多队列套路跑通。

## 样本概览

| Cohort | Person-waves | People | Mean age | Female % | Internet % | Social % | Healthy ageing % |
|---|---:|---:|---:|---:|---:|---:|---:|
| CHARLS | 96628 | 25873 | 60.68 | 52.54 | 8.01 | 47.44 | 42.61 |
| ELSA | 90069 | 19802 | 66.16 | 55.70 | 72.35 | 16.95 | 68.14 |
| HRS | 208674 | 36529 | 67.24 | 58.75 | 47.86 | 52.20 | 55.59 |
| KLoSA | 44273 | 9238 | 67.74 | 57.31 | NA | 59.60 | 42.39 |
| LASI | 73408 | 73408 | 57.92 | 57.58 | 6.58 | 2.82 | 70.93 |
| MHAS | 76506 | 26839 | 64.75 | 57.82 | NA | NA | 71.62 |
| SHARE | 327266 | 136843 | 66.64 | 56.41 | 51.70 | 27.08 | 40.35 |

## 初步模型

结局为探索性 `healthy_aging_binary`：自评健康、ADL、多病共存、抑郁、认知五类指标中，至少 4 类可用且健康比例达到 80%。模型为 person-wave 层面的 logistic GLM，控制年龄、性别、教育、婚姻、城乡、工作状态；pooled 模型额外控制 cohort 固定效应。标准误优先按个体聚类。

### Internet use -> healthy ageing

| Model | N | OR | 95% CI | P | Status |
|---|---:|---:|---|---:|---|
| pooled_with_cohort_fixed_effects | 520993 | 1.57 | 1.53-1.60 | 0.0000 | ok |
| CHARLS_specific | 51975 | 1.40 | 1.30-1.51 | 0.0000 | ok |
| ELSA_specific | 31212 | 1.84 | 1.69-2.00 | 0.0000 | ok |
| HRS_specific | 175649 | 1.35 | 1.30-1.39 | 0.0000 | ok |
| LASI_specific | 71608 | 1.26 | 1.16-1.38 | 0.0000 | ok |
| SHARE_specific | 190549 | 1.76 | 1.71-1.81 | 0.0000 | ok |

### Social participation -> healthy ageing

| Model | N | OR | 95% CI | P | Status |
|---|---:|---:|---|---:|---|
| pooled_with_cohort_fixed_effects | 529432 | 1.51 | 1.48-1.53 | 0.0000 | ok |
| CHARLS_specific | 85430 | 1.23 | 1.19-1.27 | 0.0000 | ok |
| ELSA_specific | 38874 | 1.21 | 1.12-1.32 | 0.0000 | ok |
| HRS_specific | 45753 | 1.38 | 1.32-1.44 | 0.0000 | ok |
| KLoSA_specific | 42533 | 1.76 | 1.67-1.86 | 0.0000 | ok |
| LASI_specific | 71610 | 1.18 | 1.06-1.33 | 0.0034 | ok |
| SHARE_specific | 245232 | 1.71 | 1.67-1.75 | 0.0000 | ok |

## 重要解释

- 这是第一版可行性分析，不是最终论文模型。
- 当前模型使用 person-wave 数据，已聚类个体标准误，但仍应升级为严格纵向设计，例如滞后暴露、固定效应、轨迹模型或状态转移模型。
- KLoSA 和 MHAS 的互联网变量缺失或不可比，互联网模型不会覆盖全部七国。
- 各国 CESD 量表和认知量表不同，当前做了实用型 harmonization；正式投稿前需要在方法中逐一说明阈值和敏感性分析。
- 下一步最值得做的是：健康老龄化轨迹、frailty/intrinsic capacity 转移、多队列异质性森林图。

## 输出文件

- `harmonized_variable_mapping.csv`：跨国变量映射。
- `cohort_overview.csv`：队列层面描述统计。
- `wave_counts.csv`：各队列各波次样本量。
- `variable_availability.csv`：关键变量可用率。
- `model_internet_healthy_aging.csv`：互联网使用模型。
- `model_social_healthy_aging.csv`：社会参与模型。
- `harmonized_core.parquet`：统一后的 person-wave 核心数据。