# 归档：v1 终态口径（commit f3baac9）

> 这是 2026-10-19 v1 README 的原样归档，仅用于版本对照。
> v1 的 `OOT 批准坏账率 14.39%` 来自只保留终态贷款的删失样本，已在 v2 中废弃；请以 `README.md` / `outputs/report_v2.html` 为准。
> v1 报告归档：`outputs/report_v1_terminal.html`；v1 监控方案归档：`outputs/monitoring_plan_v1_terminal.md`。

---
# LendingClub 2012–2018 信用评分卡 · OOT / PSI / Vintage / 监控

基于 LendingClub 公开贷款数据 `accepted_2007_to_2018Q4.csv.gz`，只取 **2012–2018** 年放款样本，按 `issue_d` 做时间切分，完成 **WOE/IV 评分卡 + OOT 验证 + PSI + Vintage + 阈值扫描 + 上线后监控方案 + LightGBM 对照** 的完整项目。

> 数据文件：`D:\DATAsupport\accepted_2007_to_2018Q4.csv.gz`  
> 项目目录：`D:\DATAsupport\lendingclub_scorecard_2018`  
> 上一版 2012–2016 项目保留在 `..\lendingclub_scorecard`，未被覆盖。

---

## 1. 项目概览

| 项目 | 说明 |
|---|---|
| 目标 | 预测贷款是否 **Charged Off / Default**（1=坏，0=Fully Paid） |
| 样本 | 2012–2018 共 2,218,133 条；清洗后 2,216,466 条 |
| 时间切分 | 训练 **2013–2015**；验证 **2016**；OOT **2017–2018** |
| 训练 / 验证 / OOT | 733,451 / 293,072 / 225,313 |
| 坏客户率 | 18.81% / 23.28% / **21.30%** |
| 评分卡 | WOE + Logistic Regression，12 个变量，600 基准 / PDO 20 / 好坏比 4:1 |
| OOT AUC / KS | **0.6646 / 0.2403**（评分卡）；LightGBM **0.7077 / 0.3006** |
| Score PSI | 训练→验证 **0.0115**；训练→OOT **0.0272** |
| 阈值 | 验证集选定 PD 阈值 **0.16**；OOT 批准率 57.57%，批准坏账率 **14.39%** |
| 监控方案 | `outputs/monitoring_plan.md`，含 PSI 阈值、预警流程、案例 A/B、重训触发条件 |

### 关键结论

- 2017–2018 OOT 的坏账率为 21.30%，低于 2016 验证集（23.28%），但评分卡 OOT AUC 从 0.6839 降到 0.6646，说明时间外排序能力仍有衰减。
- **PSI 稳定不一定代表校准稳定**：OOT 平均预测 PD 仅 16.91%，实际坏账率 21.30%；校准截距约 0.036，斜率约 0.819，说明绝对概率被低估且高风险段校准偏松。
- 阈值 0.16 在 2016 验证集批准坏账率 14.59%，在 2017–2018 OOT 为 14.39%，本次跨期表现较好，但仍需按月监控。
- 2017–2018 OOT 中 Current 占比约 73.4%，其中 2018 年约 **86.3%**，属于严重早成熟 cohort；绝对坏账率会随账龄上升，OOT 更适合看排序能力与 PSI。
- LightGBM OOT AUC 比评分卡高 **0.0431**，但用 6.8× 变量、169.5× 单笔计算量、147× 文件大小换来，需显式权衡可解释性代价。

## 2. 数据与目标

### 2.1 目标定义

```text
bad_flag = 1  if loan_status in {Charged Off, Default}
bad_flag = 0  if loan_status == Fully Paid
```

`Current`、`Late`、`In Grace Period` 等未定状态不进入训练与 OOT 评估。

### 2.2 时间切分

| 数据集 | 时间窗口 | 样本数 | 坏客户率 |
|---|---|---:|---:|
| 训练 | 2013-01-01 ~ 2015-12-31 | 733,451 | 18.81% |
| 验证 | 2016-01-01 ~ 2016-12-31 | 293,072 | 23.28% |
| OOT | 2017-01-01 ~ 2018-12-31 | 225,313 | 21.30% |
| Vintage | 2012–2018 全部 | 2,216,466 | — |

2012 年不进入建模（部分征信字段缺失），只用于 Vintage 客群结构展示。

### 2.3 特征口径

- 只使用申请时点可获得的申请表、征信局、交易线变量。
- **不使用** LendingClub 自身的 `grade`、`sub_grade`、`int_rate`，避免复用 LC 的评级/定价结果。
- 新数据包含 `fico_range_low/high`（完整），但为保持与上一版特征口径一致，**主模型不使用 FICO**；FICO 是申请时点可得的合法变量，后续可单独纳入做增量评估。
- 2016 才补齐的 `il_util`、`all_util`、`open_rv_12m`、`inq_last_12m` 等在训练期整列缺失，已剔除，避免“未来字段”泄漏。

### 2.4 泄露变量排查清单

报告 2.3 节列出完整清单，核心原则：

- 目标/贷后字段：`loan_status`、`total_pymnt`、`recoveries`、`out_prncp`、`last_pymnt_*`、`hardship_*`、`settlement_*` 等一律剔除。
- 平台自身判断：`grade`、`sub_grade`、`int_rate`、`installment` 不作为主模型特征。
- 时间字段：`issue_d` 只用于切分；`earliest_cr_line` 只派生放款时点的信用历史长度。
- 未来字段：2016 才补齐的字段整列剔除，避免时间泄漏。
- 标识/文本：`id`、`member_id`、`url`、`desc`、`title`、`emp_title`、`zip_code` 剔除。
- 所有缺失填补、截断、分箱、WOE 均只在训练集拟合，避免验证/OOT 信息泄漏。

## 3. 方法

```text
原始 gz CSV（分块读取，2012–2018）
  ├─ 过滤 2012–2018，保留终态贷款
  ├─ 目标：Charged Off/Default=1，Fully Paid=0
  ├─ 时间切分 2013-15 / 2016 / 2017-18
  ├─ 预处理：缺失标记、999 哨兵、分位截断、one-hot
  ├─ 分箱：等频 + 小箱合并 + 方向单调合并
  ├─ WOE：ln(坏客户占比 / 好客户占比)
  ├─ IV 排序 + 相关性去重 + p 值/系数方向逐步删减
  ├─ Logistic Regression（statsmodels）
  ├─ OOT：AUC/KS/Gini/PR-AUC/Brier/校准/提升
  ├─ PSI：Score PSI、特征 PSI、季度 Score PSI
  ├─ Vintage：按放款季度看坏账率与 Current/Late 构成
  ├─ 阈值扫描：2016 选阈值，2017–2018 OOT 评估
  ├─ LightGBM：同特征、2016 早停、2017–2018 OOT
  └─ 监控方案：PSI 阈值、预警流程、案例演练、重训触发条件
```

### 评分卡公式

```text
B = PDO / ln(2) = 28.8539
A = 基准分 − B × ln(基准好坏比) = 560.0
Score = A − B × logit(PD)
基础分 = A − B × β0
某箱得分 = −B × βi × WOEij
```

分数越高，违约概率越低。评分等级按 PD 划分：A < 8%，B 8%–12%，C 12%–18%，D 18%–25%，E ≥ 25%。

## 4. 主要结果

### 4.1 评分卡

| 指标 | 训练 2013-15 | 验证 2016 | OOT 2017-18 |
|---|---:|---:|---:|
| AUC | 0.7019 | 0.6839 | **0.6646** |
| KS | 0.2929 | 0.2637 | **0.2403** |
| Gini | 0.4038 | 0.3678 | 0.3292 |
| PR-AUC | 0.3558 | 0.3936 | 0.3372 |
| Brier | 0.1398 | 0.1681 | 0.1604 |
| LogLoss | 0.4428 | 0.5148 | 0.4977 |
| 平均预测 PD | 18.81% | 17.63% | 16.91% |
| 实际坏账率 | 18.81% | 23.28% | 21.30% |

### 4.2 LightGBM 对照

| 指标 | 训练 2013-15 | 验证 2016 | OOT 2017-18 |
|---|---:|---:|---:|
| AUC | 0.7732 | 0.7182 | **0.7077** |
| KS | 0.4018 | 0.3136 | **0.3006** |
| Gini | 0.5464 | 0.4364 | 0.4154 |
| PR-AUC | 0.4628 | 0.4337 | 0.3831 |
| Brier | 0.1284 | 0.1615 | 0.1539 |
| LogLoss | 0.4080 | 0.4951 | 0.4766 |

LightGBM 最佳迭代 2,034 轮（2016 早停）。OOT AUC 比评分卡高 0.0431。

### 4.3 PSI 与 Vintage

- Score PSI：训练→验证 **0.0115**，训练→OOT **0.0272**，均 < 0.10。
- 特征 PSI：`percent_bc_gt_75` 约 **0.178**、`bc_open_to_buy` 约 **0.125**，处于“需关注”区间；其余入模特征 < 0.10。
- 2017 年 Current 占比约 59.0%，2018 年约 **86.3%**；2018 年终态样本仅 5.6 万，绝对坏账率明显受成熟度影响。
- OOT 平均预测 PD 仅 16.91%，实际坏账率 21.30%，说明 **分布稳定 ≠ 校准稳定**，监控必须覆盖分数段实际坏账率。

### 4.4 阈值扫描

验证集选定 PD 阈值 **0.16**：

| 指标 | 验证 2016 | OOT 2017-18 |
|---|---:|---:|
| 批准率 | 53.97% | 57.57% |
| 批准坏账率 | 14.59% | **14.39%** |
| 拒绝人群坏账率 | 33.49% | 30.28% |
| 坏客户捕获率 | 66.19% | 61.11% |
| 批准坏账率下降 | 8.70pp | 6.91pp |

OOT 批准坏账率仍控制在 15% 以内，但批准率与捕获率随客群变化；固定阈值仍需按月监控。

### 4.5 校准分析

| 指标 | WOE 评分卡 | LightGBM |
|---|---:|---:|
| 平均预测 PD | 16.91% | 17.87% |
| 实际坏账率 | 21.30% | 21.30% |
| 校准截距（理想 0） | 0.0360 | 0.0525 |
| 校准斜率（理想 1） | 0.8191 | 0.8603 |
| ECE | 4.39% | 3.53% |
| MCE | 6.70% | 5.48% |
| Brier | 0.1604 | 0.1539 |

报告 4.2 的图 4-3 给出带 95% Wilson 置信区间的校准曲线与分箱校准误差。校准斜率 <1、截距为正，说明绝对概率被低估；若 PD 要直接用于定价/拨备，需先做截距校准或 Platt/isotonic 校准。

### 4.6 Swap-set 分析

在相同 OOT 批准人数 **129,720（批准率 57.57%）** 下：

| 指标 | 结果 |
|---|---:|
| 两模型批准集合重合率 | 85.78% |
| Jaccard 相似度 | 75.10% |
| Swap-in（仅 LightGBM 批准） | 18,448 人，坏客户率 **16.93%** |
| Swap-out（仅评分卡批准） | 18,448 人，坏客户率 **29.52%** |
| 净减少坏客户 | **2,322** |
| 净增加好客户 | 2,322 |

LightGBM 的增量主要来自“同批准规模下重新排序”。

### 4.7 LightGBM 可解释性代价量化

| 维度 | WOE 评分卡 | LightGBM | 代价/差异 |
|---|---|---|---|
| 模型变量数 | 12 | 82 | 6.8× |
| 单笔评分计算 | 12 次查表+求和 | 2,034 棵树遍历 | 169.5× |
| 模型结构 | 1 个线性方程、60 个分箱 | 2,034 棵树、63,054 个叶子、最大深度 21 | 高 |
| 单笔解释 | 12 项整数分相加，严格可加 | 需 SHAP/树路径，非严格可加 | 需额外解释器 |
| 全局单调性 | WOE 方向受控 | 无全局单调约束 | 合规/沟通成本 |
| 模型文件 | 20.4 KB | 2.93 MB | 147× |
| OOT AUC 增量 | 基准 | +0.0431 | 每增加 1 特征 +0.00062；每 100 棵树 +0.00212 |
| 特征集中度 | 12 个变量均有业务含义 | Top 10 / Top 20 gain 占 59.4% / 75.7%；80% gain 需 24 个特征 | 解释面更宽 |

结论：LightGBM 用 6.8× 变量、169.5× 单笔计算量、147× 文件大小，换来 OOT AUC +0.0431。若监管要求逐笔可复核原因码，评分卡仍是更优选择。

## 5. 上线后监控方案

完整方案见 **`outputs/monitoring_plan.md`**，报告第 8 节同步展示。

### 5.1 PSI 阈值

| PSI | 等级 | 动作 |
|---|---|---|
| < 0.10 | 绿色 / 稳定 | 正常监控 |
| 0.10–0.25 | 黄色 / 关注 | 2 个工作日内归因；评估收紧阈值 |
| > 0.25 | 红色 / 预警 | 1 个工作日内升级；暂停自动审批/人工复核；准备重训或回滚 |

### 5.2 预警流程

1. **T+0 自动检测**：数据质量、Score PSI、批准率、缺失率；红色自动告警。
2. **T+0~T+1 分级响应**：黄色 2 个工作日内归因；红色 1 个工作日内临时风险会。
3. **T+1~T+3 归因**：按渠道、客群、分数段、特征 PSI 拆解；核对数据口径与进件政策。
4. **T+3~T+10 处置**：维持观察 / 截距校准 / 收紧阈值 / 重新分箱 / 重训 / 回滚。
5. **月度复盘**：模型委员会 review AUC/KS、PSI、Vintage、批准率、坏账率与策略收益。

### 5.3 假设性报警案例

- **案例 A：某月 Score PSI 突升到 0.32**：先查数据完整性与渠道结构 → 拆分数分布 → 拆 Top 10 特征 PSI → 对齐业务事件 → 在成熟 vintage 上重算 AUC/KS、校准与批准坏账率；再决定修复数据、收紧 cutoff、校准或重训。
- **案例 B：PSI 仅 0.08，但批准坏账率升到 22%**：先做分数段校准检查 → 查客群/账龄结构 → 查策略定价 → 对照 Vintage → 在验证集重算阈值；不要因 PSI 绿色忽略校准报警。

### 5.4 重训触发条件

- Score PSI > 0.25 连续 2 个月，或单月 > 0.40。
- OOT AUC 较基线下降 > 0.03，或 KS 下降 > 0.05。
- 同账龄 Vintage 坏账率连续 2 个 cohort 相对恶化 > 25%。
- 批准率/拒绝率相对基线偏移 > 10pp，且无法用策略调整解释。
- 关键特征口径、数据源、进件政策重大变化；模型有效期到期。

### 5.5 项目局限与后续计划

- **拒绝推断**：只使用已批准且已终态贷款，未使用 RejectStats；存在样本选择偏差。
- **校准**：OOT PD 被低估，未做上线前概率校准。
- **数据年份**：覆盖 2012–2018，但 2018 年大量贷款未成熟（Current 86.3%）；Vintage 只是“已终态 cohort”口径，不是真正 months-on-book 曲线。
- **其他**：新数据含 FICO 但主模型未使用；无外部征信/另类数据；阈值未做 LGD/EAD 与收益优化；未做公平性、压力测试、独立验证与生产部署。
- **后续优先级**：P0 拒绝推断 + 校准 + 扩展 2019+ 与逐月表现；P1 风险调整收益阈值 + champion-challenger/Swap-set 监控；P2 公平性/压力测试/模型治理 + 实时评分服务。

## 6. 快速开始

环境：Python 3.10+；本机已安装依赖（复用 `D:\DATAsupport\credit_scorecard\.venv`，已含 LightGBM 4.7.0）。

```powershell
cd D:\DATAsupport\lendingclub_scorecard_2018

# 本机可直接运行
D:\DATAsupport\credit_scorecard\.venv\Scripts\python.exe scripts\run_pipeline.py

# 强制重建数据缓存
D:\DATAsupport\credit_scorecard\.venv\Scripts\python.exe scripts\run_pipeline.py --refresh-data
```

打开：

- `outputs/report.html`：完整中文报告（20 张图、27 张表）
- `outputs/monitoring_plan.md`：上线后监控方案
- `outputs/scorecard.csv` / `scorecard.json`：评分卡
- `outputs/predictions/oot_scores.csv`：OOT 逐笔分数、PD、等级（文件约 173MB，未提交到 Git；运行 pipeline 自动生成）
- `outputs/tables/calibration_metrics_oot.csv`：校准指标
- `outputs/tables/swap_set_summary.csv` / `swap_set_groups.csv`：Swap-set
- `outputs/figures/20_calibration_curve.png`：校准曲线

### 对新数据批量打分

输入 CSV 需包含原始申请字段（`loan_amnt`、`annual_inc`、`dti`、`issue_d`、`earliest_cr_line`、各征信字段与类别字段）。

```powershell
D:\DATAsupport\credit_scorecard\.venv\Scripts\python.exe scripts\score_csv.py `
  --input new_applicants.csv --output scored.csv
```

输出：`Id, Probability, score, grade`。

## 7. 目录结构

```text
lendingclub_scorecard_2018/
├── src/lendingclub_scorecard/
│   ├── config.py          # 数据口径、时间切分、评分卡参数
│   ├── data.py            # 分块读取、清洗、时间切分、预处理
│   ├── binning.py         # 等频/二值分箱、单调合并、WOE/IV
│   ├── model.py           # Logistic + LightGBM + 变量选择
│   ├── evaluation.py      # AUC/KS/校准/提升/PSI/阈值扫描
│   ├── scorecard.py       # 标准评分卡
│   ├── plots.py           # 20 张图
│   ├── monitoring.py      # 监控方案与基线
│   ├── report.py          # 中文 HTML 报告
│   └── pipeline.py        # 全流程编排
├── scripts/run_pipeline.py
├── scripts/score_csv.py
├── tests/
├── outputs/
├── requirements.txt
└── pyproject.toml
```

## 8. 数据与限制

- 数据来源：LendingClub 公开 LoanStats `accepted_2007_to_2018Q4.csv.gz`。
- 主模型不使用 LC 的 `grade` / `sub_grade` / `int_rate`；新数据含 FICO，但为与上一版特征口径一致不使用。
- 只使用终态贷款，存在样本选择偏差；2018 为严重早成熟 OOT（Current 86.3%）。
- 本项目仅用于学习、建模复现与监控方案演示，不构成真实授信建议。上线前需补充拒绝推断、概率校准、压力测试、公平性评估、独立验证与合规审批。

