# Gap-filled 数据说明

生成日期：2026-06-15

本目录是在旧 `harmonized_core.parquet` 基础上做保守补值，原始输出未覆盖。

## 补值口径

- SHARE 抑郁：使用 Gateway Harmonized SHARE G 的 `r*eurod`，`depressive_symptoms = EURO-D >= 4`。
- MHAS 社会参与：使用 Harmonized MHAS C.2 的 `r3socwk`-`r5socwk`，只补第 3-5 轮。
- KLoSA/LASI/MHAS frailty：按可得组成项构造 0-100 代理 frailty index，至少 3 个组成项可用才给分。
- Frailty 组成项包括低 BMI、疲乏/耗竭、低握力、步速/行动受限、低体力活动。KLoSA 没有标准步速变量，行动受限使用 `bedb_k` 作为保守代理。
- `frail_25pct` 按 `frailty_index >= 25` 重新计算。
- SHARE 抑郁补齐后，`healthy_aging_*` 派生变量已重新计算。

## 补值计数

| Step | Variable | Filled n | Valid before | Valid after |
|---|---|---:|---:|---:|
| SHARE EURO-D | depression_score | 327266 | 538746 | 866012 |
| SHARE EURO-D | depressive_symptoms | 327266 | 497815 | 825081 |
| MHAS weekly social activity | social_participation | 44053 | 545416 | 589469 |
| KLoSA/LASI/MHAS frailty components | frailty_index | 187760 | 572089 | 759849 |
| Recompute derived healthy ageing | healthy_aging_binary | 65640 | 828979 | 839340 |

## 目标变量覆盖率

| Cohort | Variable | Before % | After % |
|---|---|---:|---:|
| CHARLS | depression_score | 90.82 | 90.82 |
| CHARLS | depressive_symptoms | 90.82 | 90.82 |
| CHARLS | social_participation | 95.02 | 95.02 |
| CHARLS | frailty_index | 79.53 | 79.53 |
| CHARLS | healthy_aging_binary | 90.48 | 90.48 |
| ELSA | depression_score | 95.04 | 95.04 |
| ELSA | depressive_symptoms | 49.60 | 49.60 |
| ELSA | social_participation | 43.18 | 43.18 |
| ELSA | frailty_index | 29.81 | 29.81 |
| ELSA | healthy_aging_binary | 49.91 | 49.91 |
| HRS | depression_score | 93.15 | 93.15 |
| HRS | depressive_symptoms | 93.15 | 93.15 |
| HRS | social_participation | 22.41 | 22.41 |
| HRS | frailty_index | 100.00 | 100.00 |
| HRS | healthy_aging_binary | 93.12 | 93.12 |
| KLoSA | depression_score | 65.17 | 65.17 |
| KLoSA | depressive_symptoms | 65.17 | 65.17 |
| KLoSA | social_participation | 97.92 | 97.92 |
| KLoSA | frailty_index | 0.00 | 99.98 |
| KLoSA | healthy_aging_binary | 98.15 | 98.15 |
| LASI | depression_score | 97.42 | 97.42 |
| LASI | depressive_symptoms | 97.42 | 97.42 |
| LASI | social_participation | 98.54 | 98.54 |
| LASI | frailty_index | 0.00 | 99.20 |
| LASI | healthy_aging_binary | 98.60 | 98.60 |
| MHAS | depression_score | 92.34 | 92.34 |
| MHAS | depressive_symptoms | 92.34 | 92.34 |
| MHAS | social_participation | 0.00 | 57.58 |
| MHAS | frailty_index | 0.00 | 92.38 |
| MHAS | healthy_aging_binary | 92.38 | 92.38 |
| SHARE | depression_score | 0.00 | 100.00 |
| SHARE | depressive_symptoms | 0.00 | 100.00 |
| SHARE | social_participation | 77.08 | 77.08 |
| SHARE | frailty_index | 79.36 | 79.36 |
| SHARE | healthy_aging_binary | 96.49 | 99.65 |

## 输出文件

- `harmonized_core_gap_filled.parquet`：补齐后的 person-wave 核心表。
- `variable_availability_gap_filled.csv`：补齐后的变量覆盖率。
- `gap_fill_summary.csv`：各补值步骤的新增有效样本数。
- `frailty_component_metadata.csv`：KLoSA/LASI/MHAS frailty 组成项和组件数。
- `cohort_overview_gap_filled.csv`、`wave_counts_gap_filled.csv`：补齐后描述统计。

## 重要限制

这里的 KLoSA/LASI/MHAS frailty 是论文启动阶段的可比代理指标，不等同于各数据库已经发布的官方 frailty 成品变量。正式投稿前建议把 frailty 作为敏感性分析，或统一从所有国家重新构建同一套 deficit index。