# NC 纵向主分析第一版结果

分析目标：用上一波数字包容/社会参与预测下一波健康老龄化，模拟 Nature Communications / Nature Aging 多队列 harmonized data 论文套路。

## 样本流向

| Cohort | Person-waves | People | Waves | Lagged rows | Lagged people | Interval 1-6y rows |
|---|---:|---:|---:|---:|---:|---:|
| CHARLS | 96628 | 25873 | 5 | 70755 | 23474 | 70121 |
| ELSA | 90069 | 19802 | 9 | 70267 | 15388 | 69922 |
| HRS | 208674 | 36529 | 11 | 171918 | 32960 | 171268 |
| KLoSA | 44273 | 9238 | 6 | 35035 | 8566 | 35001 |
| LASI | 73408 | 73408 | 1 | 0 | 0 | 0 |
| MHAS | 76506 | 26839 | 5 | 49657 | 19797 | 39318 |
| SHARE | 327266 | 136843 | 7 | 190423 | 83352 | 182452 |

## 主模型

模型：`next healthy ageing ~ exposure + baseline healthy ageing + age + age^2 + sex + education + marital status + rural category + wave + years to next wave`；pooled 辅助模型额外加入 cohort fixed effects。标准误按个体聚类。主证据采用分队列模型并做 DerSimonian-Laird random-effects meta-analysis。`work` 不放主模型，放入敏感性分析。

### Internet use

| Model | N | People | OR | 95% CI | P | Status |
|---|---:|---:|---:|---|---:|---|
| pooled_with_cohort_wave_fixed_effects | 301492 | 117486 | 1.58 | 1.54-1.62 | 0.0000 | ok |
| CHARLS_specific | 45545 | 20646 | 1.37 | 1.25-1.49 | 0.0000 | ok |
| ELSA_specific | 20852 | 8778 | 1.49 | 1.37-1.63 | 0.0000 | ok |
| HRS_specific | 141513 | 29637 | 1.53 | 1.48-1.58 | 0.0000 | ok |
| SHARE_specific | 93582 | 58425 | 1.61 | 1.55-1.67 | 0.0000 | ok |

### Social participation

| Model | N | People | OR | 95% CI | P | Status |
|---|---:|---:|---:|---|---:|---|
| pooled_with_cohort_wave_fixed_effects | 316577 | 140941 | 1.28 | 1.26-1.31 | 0.0000 | ok |
| CHARLS_specific | 60334 | 21038 | 1.10 | 1.06-1.14 | 0.0000 | ok |
| ELSA_specific | 27996 | 9840 | 1.02 | 0.93-1.11 | 0.6774 | ok |
| HRS_specific | 36950 | 19136 | 1.09 | 1.03-1.16 | 0.0029 | ok |
| KLoSA_specific | 33197 | 8374 | 1.49 | 1.41-1.59 | 0.0000 | ok |
| MHAS_specific | 23097 | 13499 | 1.14 | 1.07-1.22 | 0.0001 | ok |
| SHARE_specific | 135003 | 69054 | 1.50 | 1.46-1.55 | 0.0000 | ok |

## Random-effects meta-analysis

| Exposure | K | OR | 95% CI | I2 % | Status |
|---|---:|---:|---|---:|---|
| internet_use | 4 | 1.51 | 1.43-1.60 | 77.90 | ok |
| social_participation | 6 | 1.21 | 1.05-1.40 | 98.03 | ok |

## Joint exposure

| Contrast vs neither | N | OR | 95% CI | P |
|---|---:|---:|---|---:|
| internet_only | 194463 | 1.51 | 1.45-1.57 | 0.0000 |
| social_only | 194463 | 1.16 | 1.12-1.20 | 0.0000 |
| both | 194463 | 1.85 | 1.78-1.92 | 0.0000 |

## 解释边界

- 这是第一版正式纵向模型，不是最终投稿版。
- Internet 主分析只应解释为有标准互联网/邮件/数字使用变量的队列结果；KLoSA 和 MHAS 不强行纳入 internet 主模型。
- LASI 只有一波，不进入纵向主模型，可作为横断面补充或未来新版数据补充。
- 下一步应补权重、队列专属变量定义附表、失访/IPW、负对照和更严格的状态转移模型。