# NC01 纵向中介分析第一版

设计：`wave t` 的 connectedness 暴露 -> `wave t+1` 的中介 -> `wave t+2` 的健康老龄化结局。

为避免循环定义，检验某一中介时，结局删除对应健康域：认知中介使用 `healthy_excl_cognition`，抑郁中介使用 `healthy_excl_depression`，功能中介使用 `healthy_excl_function`。

模型为探索性 product-of-coefficients：中介模型用线性模型，结局模型用 logistic GLM，标准误按个体聚类。协变量包括年龄、年龄平方、性别、教育、婚姻、城乡、两段随访间隔、基线波次和队列固定效应。

## 队列可用性

| Cohort | Person-waves | People | Social valid | Internet valid | Cognition valid | Depression valid | Function valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| CHARLS | 102344 | 25586 | 72469 | 0 | 64306 | 70220 | 75254 |
| ELSA | 178218 | 19802 | 67525 | 0 | 85712 | 85606 | 89840 |
| KLoSA | 89392 | 11174 | 63214 | 0 | 0 | 28852 | 63215 |
| MHAS | 134195 | 26839 | 44053 | 0 | 70541 | 70644 | 70626 |
| SHARE | 1270112 | 158764 | 381955 | 0 | 444211 | 396713 | 457545 |

## 三波样本

| Cohort | Triad rows | People | Waves |
|---|---:|---:|---:|
| CHARLS | 27848 | 15860 | 2 |
| ELSA | 51509 | 12681 | 7 |
| KLoSA | 40838 | 8883 | 6 |
| MHAS | 11170 | 11170 | 1 |
| SHARE | 154423 | 64360 | 6 |

## 中介结果

| Analysis | N | People | Cohorts | Total OR | Direct OR | Indirect OR | Indirect log-OR 95% CI | Mediated % | Status |
|---|---:|---:|---|---:|---:|---:|---|---:|---|
| social_to_cognition | 195538 | 91707 | CHARLS,ELSA,MHAS,SHARE | 1.51 | 1.42 | 1.06 | 0.0588 to 0.0668 | 15.22 | ok |
| social_to_not_depressive | 196149 | 99305 | CHARLS,ELSA,KLoSA,MHAS,SHARE | 1.58 | 1.51 | 1.06 | 0.0562 to 0.0658 | 13.31 | ok |
| social_to_no_adl_limitation | 223156 | 100732 | CHARLS,ELSA,KLoSA,MHAS,SHARE | 1.57 | 1.52 | 1.06 | 0.0528 to 0.0629 | 12.75 | ok |

## 队列特异性结果

| Analysis | Cohort | N | Total OR | Direct OR | Indirect OR | Mediated % |
|---|---|---:|---:|---:|---:|---:|
| social_to_cognition | CHARLS | 22894 | 1.10 | 1.08 | 1.02 | 24.27 |
| social_to_cognition | ELSA | 39646 | 1.33 | 1.29 | 1.04 | 12.16 |
| social_to_cognition | MHAS | 9602 | 1.05 | 1.03 | 1.02 | 44.35 |
| social_to_cognition | SHARE | 123396 | 1.76 | 1.63 | 1.09 | 14.83 |
| social_to_not_depressive | CHARLS | 22179 | 1.18 | 1.13 | 1.05 | 29.01 |
| social_to_not_depressive | ELSA | 39612 | 1.37 | 1.32 | 1.05 | 14.22 |
| social_to_not_depressive | KLoSA | 19272 | 1.43 | 1.41 | 1.02 | 4.41 |
| social_to_not_depressive | MHAS | 9606 | 1.05 | 1.05 | 1.01 | 14.39 |
| social_to_not_depressive | SHARE | 105480 | 1.94 | 1.84 | 1.08 | 12.11 |
| social_to_no_adl_limitation | CHARLS | 22616 | 1.23 | 1.20 | 1.04 | 18.48 |
| social_to_no_adl_limitation | ELSA | 39814 | 1.42 | 1.36 | 1.06 | 17.50 |
| social_to_no_adl_limitation | KLoSA | 25957 | 1.36 | 1.35 | 1.03 | 9.37 |
| social_to_no_adl_limitation | MHAS | 9590 | 1.08 | 1.08 | 1.01 | 14.61 |
| social_to_no_adl_limitation | SHARE | 125179 | 1.86 | 1.80 | 1.07 | 10.70 |

## 解释边界

- 这是机制增强的第一版结果，适合判断是否值得加入论文，不应直接改写为因果机制证明。
- 本机 `E:\nc数据` 的 Stata 宽表中，CHARLS/ELSA/SHARE 的 internet 变量未能用当前文件可靠定位；因此本版 mediation 输出只报告社会参与路径，internet mediation 不作为多队列主结果。
- 社会参与路径覆盖现有可定位队列；若后续找到原始 `01_data_deduped/csv` 或生成 `harmonized_core_gap_filled.parquet`，可以按同一脚本补齐 internet/joint mediation。