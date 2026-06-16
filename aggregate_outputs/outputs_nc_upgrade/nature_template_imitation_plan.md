# Nature 系列仿写对象与文章结构映射

## 结论

最适合你的主仿写对象不是单纯 Nature Communications 的疾病风险文章，而是：

**主仿写对象：**

Yuan et al. *Associations between internet use, social participation, and cognitive function among middle-aged and older adults*. **npj Mental Health Research**. 2025.  
https://www.nature.com/articles/s44184-025-00162-6

原因：它和你的项目在暴露、队列、统计套路上最接近：

- 暴露同样是 internet use 和 social participation。
- 数据同样使用 CHARLS、MHAS、HRS、ELSA、SHARE。
- 有 internet transition。
- 有 joint exposure。
- 有 subgroup analysis。
- 有 random-effects meta-analysis。

**NC 级结构参照：**

Li et al. *Intrinsic capacity and stroke risk in a multiple cohort study*. **Nature Communications**. 2026.  
https://www.nature.com/articles/s41467-026-70524-x

原因：它更像 Nature Communications 的多队列健康老龄化论文结构：

- 用多队列数据。
- 用多维综合指标。
- 强调 WHO / healthy ageing 理论框架。
- 有 sample flow、baseline table、domain contribution、sensitivity analysis。

**高影响跨国队列写法参照：**

Mak et al. *Hobby engagement and mental wellbeing among people aged 65 years and older in 16 countries*. **Nature Medicine**. 2023.  
https://www.nature.com/articles/s41591-023-02506-1

原因：它的跨国老龄化队列叙事非常适合你借鉴：

- 先讲跨国差异。
- 再讲 longitudinal association。
- 再讲方向性、异质性和政策意义。

**数字健康领域背景参照：**

Li et al. *Ten-year trends of the digital divides and its effect on healthy aging among older adults in China from 2011 to 2020*. **npj Digital Medicine**. 2025.  
https://www.nature.com/articles/s41746-025-02076-1

原因：它适合引用在 introduction 中说明 digital divide / digital inclusion 与 healthy aging 的背景。

## 你的文章应该怎么定位

不要写成：

> Internet use and social participation are associated with healthy ageing.

这太普通。

应该写成：

> Digital and social connectedness as joint and dynamic resources for multidimensional healthy ageing transitions across harmonized international ageing cohorts.

中文理解：

> 跨国 harmonized ageing cohorts 中，数字连接和社会连接作为联合且动态变化的资源，与多维健康老龄化状态转移相关。

## 和主仿写对象的逐项映射

| npj Mental Health Research 那篇 | 你的文章对应写法 |
|---|---|
| Internet use + social participation | Digital connectedness + social connectedness |
| Cognitive function | Multidimensional healthy ageing |
| Memory / orientation / executive function domains | Self-rated health / ADL / multimorbidity / depression / cognition domains |
| Internet use transition: Yes->Yes, No->Yes, Yes->No, No->No | Exposure history: persistent, adopted, stopped, never |
| Joint association of internet and social participation | Joint connectedness: neither, internet only, social only, both |
| Stratified analyses | Age, sex, education, rurality, baseline health subgroup analyses |
| Sensitivity analyses | IPW attrition, alternative healthy ageing definitions, leave-one-cohort-out |
| Random-effects meta-analysis | Cohort-specific estimates + random-effects meta-analysis |

## 文章标题建议

### 最推荐

Digital-social connectedness and multidimensional healthy ageing transitions across international ageing cohorts

### 更稳健

Digital and social connectedness trajectories and healthy ageing transitions in harmonized ageing cohorts

### 更接近 Nature Communications

Digital-social connectedness as a dynamic resource for healthy ageing in a multiple cohort study

## Abstract 仿写结构

按 Nature Portfolio 常见结构写 6 句：

1. 背景问题：digitalization and population ageing are reshaping later-life connectedness。
2. 文献缺口：prior studies examined digital use or social participation separately and often focused on single health domains。
3. 数据和设计：we used harmonized longitudinal ageing cohorts and lagged exposure-outcome models。
4. 主发现：both digital and social connectedness showed the most favourable healthy ageing transitions。
5. 扩展发现：persistent internet use was more strongly associated than adoption, while discontinued use was not clearly associated。
6. 意义：connectedness may represent a modifiable resource for healthy ageing, but causal interpretation requires caution。

## Results 结构仿写

### 1. Cohort characteristics and connectedness patterns

对应主仿写文章的 “Characteristics of the participants” 和 Fig. 1。

你这里应该写：

- 样本来自 CHARLS、ELSA、HRS、KLoSA、MHAS、SHARE，LASI 作为横断面补充或不进入纵向主模型。
- 各队列 wave、person-waves、有效样本。
- internet use 和 social participation 覆盖率。
- joint connectedness 四分类比例。
- dynamic exposure history 比例。

建议图表：

- Fig. 1a: study design timeline。
- Fig. 1b: cohort/wave/sample flow。
- Fig. 1c: connectedness prevalence by cohort。
- Fig. 1d: exposure history distribution。

已有对应文件：

- `sample_flow.csv`
- `lagged_availability.csv`
- `joint_exposure_counts.csv`

### 2. Main associations of digital and social connectedness with healthy ageing

对应主仿写文章的 Table 1。

你这里应该写：

- internet use 与下一波 healthy ageing。
- social participation 与下一波 healthy ageing。
- 队列内模型 + random-effects meta。

建议表：

- Table 1: baseline/sample characteristics。
- Table 2: cohort-specific and pooled random-effects estimates。

已有核心结果：

- internet use random-effects OR = 1.51, 95% CI 1.43-1.60。
- social participation random-effects OR = 1.21, 95% CI 1.05-1.40。

### 3. Joint connectedness profiles

对应主仿写文章的 Fig. 3。

你这里要把它升级成正文核心图，而不是补充图。

结果：

- both vs neither: OR = 1.65, 95% CI 1.36-2.00。
- internet only vs neither: OR = 1.49, 95% CI 1.44-1.55。
- social only vs neither: OR = 1.15, 95% CI 1.01-1.31。

建议图：

- Fig. 2: cohort-specific forest plots for internet only / social only / both。

已有图：

- `figure_joint_exposure_forest.png`

注意写法：

不要写 synergy。因为：

- multiplicative interaction OR = 1.00。
- RERI = 0.07, 95% CI -0.07 to 0.20。

应该写：

> The joint connectedness group showed the most favourable profile, although formal evidence for multiplicative or additive interaction was limited.

### 4. Dynamic exposure histories

对应主仿写文章的 Fig. 4 和 Fig. 5。

你这里是最能突出新颖性的部分。

结果：

- adopted internet use: OR = 1.24, 95% CI 1.15-1.35。
- persistent internet use: OR = 1.51, 95% CI 1.32-1.74。
- stopped internet use: OR = 1.09, 95% CI 0.91-1.30。
- adopted social participation: OR = 1.19, 95% CI 1.03-1.38。
- persistent social participation: OR = 1.32, 95% CI 1.04-1.69。

建议图：

- Fig. 3: dynamic connectedness trajectories and subsequent healthy ageing。

推荐写法：

> Persistent digital engagement, rather than discontinued use, was most consistently associated with favourable healthy ageing transitions.

### 5. Healthy ageing transitions

这是你的文章超越主仿写对象的关键。

主仿写文章是 cognitive outcome；你应该升级为 trajectory/transition。

结果：

- gain healthy, both vs neither: OR = 1.68, 95% CI 1.37-2.06。
- maintain healthy, both vs neither: OR = 1.65, 95% CI 1.30-2.10。

建议图：

- Fig. 4a: transition diagram or heatmap。
- Fig. 4b: forest plot for gain healthy。
- Fig. 4c: forest plot for maintain healthy。

推荐写法：

> Connectedness was associated not only with maintaining healthy ageing among those already healthy, but also with gaining healthy ageing among those initially unhealthy.

### 6. Heterogeneity and robustness

对应主仿写文章的 stratified analyses 和 sensitivity analyses。

你的结果：

- subgroup pooled models 已经生成。
- leave-one-cohort-out 已经生成。
- IPW attrition 后结果基本不变。
- alternative healthy ageing definitions 方向一致。

建议图：

- Fig. 5a: subgroup heatmap。
- Fig. 5b: robustness grid。
- Extended Data: leave-one-cohort-out。

已有图：

- `figure_subgroup_heatmap.png`
- `figure_robustness_grid.png`

### 7. Domain-specific analyses / mechanism clues

对应 Nature Communications intrinsic capacity 那篇的 domain contribution 逻辑。

你这里不做强机制结论，只做健康域分解：

- cognition_not_low: internet OR = 2.50; social OR = 1.35。
- no ADL limitation: internet OR = 1.53; social OR = 1.32。
- not depressive: internet OR = 1.39; social OR = 1.19。

建议图：

- Fig. 6 或 Extended Data Fig.: domain-specific associations。

推荐写法：

> Domain-specific analyses suggested that cognitive, psychological, and functional health domains may be particularly relevant to the observed associations.

## 你文章的最终图表安排

| Figure/Table | 内容 | 仿写来源 |
|---|---|---|
| Fig. 1 | Study design, cohort timeline, sample flow, connectedness prevalence | npj Mental Health Research Fig. 1 + NC flowchart |
| Table 1 | Cohort/sample characteristics | Nature Communications Table 1 |
| Fig. 2 | Joint connectedness forest plot | npj Mental Health Research Fig. 3 |
| Fig. 3 | Dynamic exposure history: adopted/stopped/persistent | npj Mental Health Research Fig. 4-5 |
| Fig. 4 | Healthy ageing transitions: gain and maintain | 你的创新升级 |
| Fig. 5 | Subgroup heterogeneity heatmap + robustness grid | npj stratified + Nature Medicine heterogeneity |
| Fig. 6 / Extended Data | Domain-specific health pathways | Nature Communications domain table |
| Supplementary Table 1 | cohort waves and inclusion criteria | all templates |
| Supplementary Table 2 | variable harmonization dictionary | all templates |
| Supplementary Table 3 | variable availability matrix | all templates |
| Extended Data | IPW, leave-one-out, alternative definitions, RERI | Nature Portfolio robustness style |

## Introduction 仿写逻辑

### Paragraph 1: ageing + digitalization

全球老龄化和数字化正在同时改变老年人的健康资源结构。数字服务、线上沟通和信息获取可能帮助老年人维持功能和社会连接，但数字鸿沟也可能扩大健康不平等。

### Paragraph 2: connectedness theory

数字连接和社会参与可以被理解为两类 connectedness resources。前者提供信息、认知刺激和服务可及性；后者提供面对面互动、情感支持和行为激活。

### Paragraph 3: prior evidence and gap

已有 Nature Portfolio 文章分别研究了 internet use/social participation 与 cognition，多维 intrinsic capacity 与疾病风险，以及 digital divide 与 healthy aging。但仍缺少跨国 harmonized cohorts 中，digital-social connectedness 的联合、动态模式与 multidimensional healthy ageing transitions 的系统性证据。

### Paragraph 4: current study

本研究使用多个 HRS-family harmonized ageing cohorts，考察 digital and social connectedness 与后续 healthy ageing 的关系，并进一步分析 joint profiles、exposure histories、health transitions、heterogeneity、robustness 和 domain-specific patterns。

## Discussion 仿写逻辑

### 第一段：主发现

在多个 harmonized ageing cohorts 中，digital and social connectedness 与更有利的 healthy ageing transitions 相关。同时具备两类 connectedness 的人群表现出最高的后续 healthy ageing 概率。

### 第二段：和既往文献对比

你的结果扩展了既往 internet/social participation 与 cognition 的研究，因为你的结局是 multidimensional healthy ageing，而且不仅分析静态暴露，还分析 adoption/persistence/stopping 和 gain/maintain transitions。

### 第三段：动态暴露解释

持续互联网使用的关联强于新采用，而停止使用不显著。这提示 sustained digital engagement 可能比短期接触更能代表稳定的数字资源、技能和社会-信息嵌入。

### 第四段：机制线索

domain-specific results 指向 cognition、psychological health、functional status，但不能写成 confirmed mediation。

### 第五段：异质性

高 I2 和 subgroup 差异说明跨国环境、数字基础设施、社会参与文化和样本测量差异可能影响关联强度。

### 第六段：优势和限制

优势：

- harmonized multi-cohort design。
- lagged longitudinal design。
- joint and dynamic exposure patterns。
- healthy ageing transition outcomes。
- IPW and alternative definitions。

限制：

- observational data。
- residual confounding。
- measurement heterogeneity。
- attrition/death competing risk。
- mechanism evidence exploratory。

## 可直接使用的新颖性句子

> Previous studies have examined digital technology use, social participation, cognition, frailty, or healthy ageing in isolation, often within a single country or focusing on a single health domain. This study extends prior work by integrating harmonized longitudinal data from multiple ageing cohorts to examine digital and social connectedness as joint and dynamic resources for multidimensional healthy ageing transitions.

> Unlike prior studies that focused on static internet use or single-domain outcomes, we examined joint connectedness profiles, changes in connectedness over time, and transitions in healthy ageing status across international ageing cohorts.

> Our findings shift the focus from whether older adults use the internet to whether they remain connected through digital and social channels, and how these connectedness trajectories relate to maintaining or gaining healthy ageing.

## 需要引用的 Nature 系列文章

1. Yuan T, Liu K, Liang L, et al. Associations between internet use, social participation, and cognitive function among middle-aged and older adults. **npj Mental Health Research**. 2025.  
   https://www.nature.com/articles/s44184-025-00162-6

2. Li Y, Chen Y, Chen Y, et al. Intrinsic capacity and stroke risk in a multiple cohort study. **Nature Communications**. 2026.  
   https://www.nature.com/articles/s41467-026-70524-x

3. Mak H, et al. Hobby engagement and mental wellbeing among people aged 65 years and older in 16 countries. **Nature Medicine**. 2023.  
   https://www.nature.com/articles/s41591-023-02506-1

4. Li S, Ouyang Y, Hu M. Ten-year trends of the digital divides and its effect on healthy aging among older adults in China from 2011 to 2020. **npj Digital Medicine**. 2025.  
   https://www.nature.com/articles/s41746-025-02076-1

5. Yang T, Deng J, et al. Bidirectional relationship between digital inclusion and healthy ageing among Chinese older adults: a four-wave cross-lagged study. **Humanities and Social Sciences Communications**. 2026.  
   https://www.nature.com/articles/s41599-025-06486-0

## 下一步执行建议

1. 按这个结构重排现有 `nc_upgrade_report_zh.md`。
2. 把 `figure_joint_exposure_forest.png` 设为核心 Fig. 2。
3. 新增 exposure history 图，作为 Fig. 3。
4. 新增 transition 图，作为 Fig. 4。
5. 把 subgroup heatmap 和 robustness grid 放 Fig. 5 或 Extended Data。
6. 把 domain-specific results 做成 Fig. 6 或 Extended Data。
7. 开始写英文 Introduction 和 Results，严格照主仿写对象的结果推进顺序，但所有数据和表述用自己的。
