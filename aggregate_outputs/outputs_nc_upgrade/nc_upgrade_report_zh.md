# NC 升级分析结果包

目标：把第一版主效应升级为 connectedness resources 与 healthy ageing trajectories 的多层证据链。

## 1. 联合暴露作为中心发现

| Contrast vs neither | K | OR | 95% CI | I2 % |
|---|---:|---:|---|---:|
| both | 4 | 1.65 | 1.36-2.00 | 94.77 |
| internet_only | 4 | 1.49 | 1.44-1.55 | 0.00 |
| social_only | 4 | 1.15 | 1.01-1.31 | 91.74 |

## 2. 联合暴露交互

| Metric | K | Estimate | 95% CI | I2 % |
|---|---:|---:|---|---:|
| multiplicative interaction OR | 4 | 1.00 | 0.94-1.06 | 0.00 |
| additive interaction RERI | 4 | 0.07 | -0.07-0.20 | 57.61 |

## 3. 数字/社会参与动态暴露

| Exposure | History vs never | K | OR | 95% CI | I2 % |
|---|---|---:|---:|---|---:|
| internet_use | adopted | 4 | 1.24 | 1.15-1.35 | 38.12 |
| internet_use | persistent | 4 | 1.51 | 1.32-1.74 | 91.17 |
| internet_use | stopped | 4 | 1.09 | 0.91-1.30 | 77.72 |
| social_participation | adopted | 5 | 1.19 | 1.03-1.38 | 90.47 |
| social_participation | persistent | 5 | 1.32 | 1.04-1.69 | 97.99 |
| social_participation | stopped | 5 | 1.18 | 1.01-1.37 | 91.54 |

## 4. 健康老龄化状态转移

| Transition | Contrast | K | OR | 95% CI | I2 % |
|---|---|---:|---:|---|---:|
| gain_healthy | both | 4 | 1.68 | 1.37-2.06 | 86.31 |
| gain_healthy | internet_only | 4 | 1.44 | 1.33-1.55 | 20.48 |
| gain_healthy | social_only | 4 | 1.13 | 0.98-1.32 | 83.38 |
| maintain_healthy | both | 4 | 1.65 | 1.30-2.10 | 93.92 |
| maintain_healthy | internet_only | 4 | 1.54 | 1.47-1.62 | 0.00 |
| maintain_healthy | social_only | 4 | 1.17 | 1.02-1.35 | 86.56 |

## 5. 健康域驱动结果

下面列出每个健康域的随机效应 meta。若某健康域结果强，后续可作为机制/解释重点。

| Exposure | Domain | K | OR | 95% CI | I2 % |
|---|---|---:|---:|---|---:|
| internet_use | cognition_not_low | 4 | 2.50 | 2.14-2.92 | 91.71 |
| social_participation | cognition_not_low | 6 | 1.35 | 1.16-1.56 | 95.96 |
| internet_use | good_self_rated_health | 4 | 1.45 | 1.29-1.61 | 94.36 |
| social_participation | good_self_rated_health | 6 | 1.16 | 0.98-1.39 | 98.45 |
| internet_use | no_adl_limitation | 4 | 1.53 | 1.44-1.64 | 71.47 |
| social_participation | no_adl_limitation | 6 | 1.32 | 1.15-1.52 | 96.07 |
| internet_use | no_multimorbidity | 4 | 1.25 | 1.14-1.38 | 83.20 |
| social_participation | no_multimorbidity | 6 | 1.07 | 0.94-1.21 | 94.23 |
| internet_use | not_depressive | 4 | 1.39 | 1.34-1.44 | 38.17 |
| social_participation | not_depressive | 6 | 1.19 | 1.07-1.32 | 95.87 |

## 6. 稳健性分析

| Analysis | Exposure | K | OR | 95% CI | I2 % |
|---|---|---:|---:|---|---:|
| lenient_60pct_threshold | internet_use | 4 | 1.58 | 1.44-1.73 | 88.08 |
| lenient_60pct_threshold | social_participation | 6 | 1.28 | 1.07-1.53 | 98.29 |
| strict_all_available_components | internet_use | 4 | 1.34 | 1.30-1.38 | 0.00 |
| strict_all_available_components | social_participation | 6 | 1.15 | 1.04-1.27 | 92.18 |
| without_cognition_component | internet_use | 4 | 1.49 | 1.41-1.57 | 74.59 |
| without_cognition_component | social_participation | 6 | 1.20 | 1.04-1.39 | 97.94 |
| ipw_attrition | internet_use | 4 | 1.51 | 1.43-1.59 | 73.67 |
| ipw_attrition | social_participation | 6 | 1.21 | 1.05-1.40 | 97.86 |

## 7. 异质性信号

- 已运行 pooled subgroup models：26 条模型记录。
- 重点看 `subgroup_pooled_models.csv` 和 `figure_subgroup_heatmap.png`。
- 如果某些分层 OR 明显不同，下一轮应改成分队列 subgroup meta，而不是只用 pooled 模型。

## 8. E-value

| Analysis | Term/Exposure | OR | CI low | E-value point | E-value CI |
|---|---|---:|---:|---:|---:|
| joint_exposure | both | 1.65 | 1.36 | 2.68 | 2.05 |
| joint_exposure | internet_only | 1.49 | 1.44 | 2.36 | 2.24 |
| joint_exposure | social_only | 1.15 | 1.01 | 1.56 | 1.09 |
| transition | both | 1.68 | 1.37 | 2.74 | 2.07 |
| transition | internet_only | 1.44 | 1.33 | 2.23 | 2.00 |
| transition | social_only | 1.13 | 0.98 | 1.52 | 1.00 |
| transition | both | 1.65 | 1.30 | 2.69 | 1.93 |
| transition | internet_only | 1.54 | 1.47 | 2.46 | 2.31 |
| transition | social_only | 1.17 | 1.02 | 1.62 | 1.17 |
| exposure_history | adopted | 1.24 | 1.15 | 1.79 | 1.56 |
| exposure_history | persistent | 1.51 | 1.32 | 2.39 | 1.96 |
| exposure_history | stopped | 1.09 | 0.91 | 1.39 | 1.00 |
| exposure_history | adopted | 1.19 | 1.03 | 1.68 | 1.21 |
| exposure_history | persistent | 1.32 | 1.04 | 1.98 | 1.23 |
| exposure_history | stopped | 1.18 | 1.01 | 1.63 | 1.10 |
| alternative_definition | internet_use | 1.58 | 1.44 | 2.53 | 2.24 |
| alternative_definition | social_participation | 1.28 | 1.07 | 1.87 | 1.33 |
| alternative_definition | internet_use | 1.34 | 1.30 | 2.02 | 1.93 |
| alternative_definition | social_participation | 1.15 | 1.04 | 1.56 | 1.25 |
| alternative_definition | internet_use | 1.49 | 1.41 | 2.34 | 2.17 |
| alternative_definition | social_participation | 1.20 | 1.04 | 1.69 | 1.25 |
| ipw_attrition | internet_use | 1.51 | 1.43 | 2.38 | 2.21 |
| ipw_attrition | social_participation | 1.21 | 1.05 | 1.72 | 1.29 |

## 9. 现在距离 NC 还差什么

- 必须把 joint exposure 提到主结果，而不是补充模型。
- 必须用状态转移结果讲 trajectories：maintain healthy 和 gain healthy。
- 动态暴露结果用于回答 adoption/persistence，比静态 internet use 更有论文新意。
- RERI 若不显著，不能硬写协同；可以写 both 组风险优势最大，但 additive interaction 证据有限。
- 高 I2 不是小问题，需要在正文中解释为跨国家/队列差异，并用 subgroup/leave-one-out 支撑。
- IPW 已作为第一层失访校正；若后续能接死亡/退出状态，应补 competing risk 或死亡复合结局。
- domain outcome 是机制线索，不应直接写成中介因果；正式投稿前可进一步做 t -> mediator -> t+2 的探索性 mediation。