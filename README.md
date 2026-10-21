# LendingClub 2012–2018 信用评分卡 · v3（P0：阈值加权 / 终身口径 / 打分链路）

基于 LendingClub 公开贷款数据 `accepted_2007_to_2018Q4.csv.gz`，只取 **2012–2018** 年放款样本，完成 **WOE/IV 评分卡 + MOB12 固定窗口 OOT + 概率校准 + PSI + Vintage + 阈值敏感性 + LightGBM 对照 + 上线后监控**。

> **v3 修订说明（2026-10-21）：** 修复 v2 `threshold_sensitivity` 对终身加权标签 `1/e` 的 `int` 截断：12%/15%/18% 目标对应的阈值由错误的 0.45/0.90/0.90 修正为 **0.21/0.32/0.57**，批准率 **63.25%/84.27%/97.65%**；“15%/18% 目标非绑定”的结论不再成立。终身坏账率统一用 **E[y/e] = 18.97%**；0.27 阈值下拒绝人群坏账率由错误的 55.88% 修正为 **36.31%**。新增 `outputs/decision_policy.json`，打分脚本现在输出校准后 MOB12 PD、终身 PD、按终身 PD 划分的 A–E 等级和 `approve` 标记。v2 文档归档为 `README_v2_mob12.md`、`EVALUATION_RESPONSE_v2.md`、`outputs/report_v2.html`、`outputs/monitoring_plan_v2_mob12.md`。
>
> **v2 修订说明：** 旧版只保留终态贷款，把“终态批准坏账率 14.39%”当成达标结论和监控基线，存在删失偏差。v2 改为 **MOB12 固定表现窗口**（所有 cohort 截到 12 个月），再按期限分段把 vintage PD 换算成终身 PD 做审批决策。旧基线已废弃，旧报告归档为 `outputs/report_v1_terminal.html`。
>
> 数据文件：默认从 `LC_DATA_DIR`（或项目父目录）读取 `accepted_2007_to_2018Q4.csv.gz`；本地示例路径为 `D:\DATAsupport\accepted_2007_to_2018Q4.csv.gz`。
>
> 逐条处理记录：[EVALUATION_RESPONSE.md](EVALUATION_RESPONSE.md)
> 版本记录：[CHANGELOG.md](CHANGELOG.md)；v1 归档：[README_v1_terminal.md](README_v1_terminal.md)、`outputs/report_v1_terminal.html`、`outputs/monitoring_plan_v1_terminal.md`；v2 归档：[README_v2_mob12.md](README_v2_mob12.md)、`outputs/report_v2.html`、`outputs/monitoring_plan_v2_mob12.md`。

---

## 1. 项目概览

| 项目 | 说明 |
|---|---|
| 目标 | MOB12 内是否 **Charged Off / Default**；再换算为终身 PD 做审批 |
| 样本 | 2012–2018 共 2,218,133 条；清洗后 2,216,466 条 |
| 时间切分 | 训练 **2013–2015**；验证 **2016**；OOT **2017–2018Q1** |
| 训练 / 验证 / OOT | 791,536 / 434,350 / 550,697（均 MOB12 可评估） |
| MOB12 坏账率 | 5.24% / 6.68% / **5.71%** |
| 终身坏账率（OOT 加权） | **18.97%** |
| 评分卡 | WOE + Logistic Regression，14 个变量，600 基准 / PDO 20 / 好坏比 4:1 |
| OOT AUC / KS（MOB12） | **0.6598 / 0.2314** |
| OOT AUC（终身） | **0.6571** |
| 阈值 | 终身 PD **0.27**；OOT 批准率 **76.81%**，批准人群终身坏账率 **13.74%**，拒绝人群坏账率 **36.31%** |
| 监控方案 | `outputs/monitoring_plan.md`（v3，MOB12/终身双口径） |

### 关键结论

- **删失偏差已修：** OOT 上界收到 `2018-03-31`（快照 2019-03-31 减 12 个月）；2018Q2+ 观察不足 12 个月，不进入 OOT。
- **绝对坏账率不再混用：** MOB12 坏账率 5.71% 是建模标签尺度；终身加权坏账率 18.97% 才是业务口径。旧版 14.39% 终态基线作废。
- **概率校准：** 原始 PD 的 OOT ECE 1.15%，Platt 校准后 0.41%；校准器只在 2016 验证集 fit。
- **阈值：** 0.27 在 OOT 上批准率 76.81%、批准人群终身坏账率 13.74%、捕获率 44.38%，低于 15% 目标。注意这是**验证集 2016 选出的固定阈值**；OOT 事后重选阈值会更高，见 §3.5 敏感性表。
- **阈值敏感性已修：** 12%/15%/18% 目标对应 0.21/0.32/0.57，批准率 63.25%/84.27%/97.65%；15% 约束是绑定的，18% 也不是完全非绑定。
- **FICO 的边际：** 在 82 变量 LightGBM 里只加 FICO，OOT AUC +0.0036；真正强的是 LC `grade`（+0.0253），但它是平台自身评级，不能入主模型。**尚未做 14 变量 WOE 评分卡 + FICO 的对照**，因此不能把 +0.0036 直接外推到主模型。
- **LightGBM 增量拆解：** 同 14 变量 AUC 0.6654，比评分卡 +0.0056；换到 82 变量才 +0.0286——主要增量来自变量数量，不是“树模型天然更强”。
- **可解释性代价：** 单笔约 10,053 次节点比较 vs 评分卡 14 次查表（约 718×），文件约 77×，换来 OOT AUC +0.0342。

## 2. 数据与标签

### 2.1 MOB12 定义

```text
bad_flag_v12 = 1  if loan_status in {Charged Off, Default} and 核销发生在 MOB <= 12
bad_flag_v12 = 0  if obs_mob >= 12 and 未在 MOB12 前核销
bad_flag_v12 = NaN if obs_mob < 12（剔除）
```

数据快照 `2019-03-31`；OOT 只到 `2018-03-31`，避免 2018Q2+ 的未熟样本。

### 2.2 时间切分

| 数据集 | 时间窗口 | 样本数 | MOB12 坏账率 |
|---|---|---:|---:|
| 训练 | 2013-01-01 ~ 2015-12-31 | 791,536 | 5.24% |
| 验证 | 2016-01-01 ~ 2016-12-31 | 434,350 | 6.68% |
| OOT | 2017-01-01 ~ 2018-03-31 | 550,697 | 5.71% |
| Vintage | 2012–2018 | 2,216,466 | — |

2012 年仅用于 Vintage 客群结构展示；主模型不使用 FICO/grade；`last_pymnt_d` 只用于计算 MOB，不进入特征。

### 2.3 终身口径换算

```text
p_lifetime = p_vintage / e(term)
```

- 36 期：`e ≈ 0.339`，终身坏账率 12.74% vs MOB12 4.32%
- 60 期：`e ≈ 0.250`，终身坏账率 25.79% vs MOB12 6.45%
- 总体：`e ≈ 0.304`，成熟 cohort 终身 15.91% vs MOB12 4.84%

不能用总体 e 统一换算长期限贷款，否则会系统性低估。

## 3. 主要结果

### 3.1 评分卡（MOB12 / 终身）

| 口径 | 数据集 | AUC | KS | Gini | PR-AUC | Brier | 平均 PD | 坏账率（终身行为 E[y/e] 估计） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MOB12 | 训练 | 0.6732 | 0.2522 | 0.3464 | 0.1001 | 0.0487 | 5.24% | 5.24% |
| MOB12 | 验证 | 0.6697 | 0.2478 | 0.3394 | 0.1242 | 0.0611 | 4.98% | 6.68% |
| MOB12 | OOT | **0.6598** | **0.2314** | 0.3197 | 0.1022 | 0.0530 | 4.56% | 5.71% |
| 终身 | 验证 | 0.6656 | 0.2395 | 0.3312 | 0.1222 | 0.0956 | 21.91% | — |
| 终身 | OOT | **0.6571** | 0.2263 | 0.3141 | 0.1008 | 0.0852 | 20.30% | **18.97%** |

> 终身 OOT 的 18.97% 不是观察到的终身结果，而是用成熟 cohort 的 e 对 MOB12 标签做的 E[y/e] 加权估计；它与阈值表/捕获率使用同一估计量。

### 3.2 LightGBM 对照

| 模型 | 变量数 | OOT AUC | OOT KS | 相对评分卡 |
|---|---:|---:|---:|---:|
| WOE 评分卡 | 14 | 0.6598 | 0.2314 | 基准 |
| LightGBM-同变量（14 变量） | 14 | 0.6654 | 0.2381 | +0.0056 |
| LightGBM-full | 82 | 0.6940 | 0.2820 | +0.0342 |

最佳迭代 1,239 轮（2016 早停）。AUC 增量拆解：模型形式 +0.0056，变量数量 +0.0286。

### 3.3 概率校准

| 校准器 | 平均预测 PD | 实际坏账率 | 校准截距 | 校准斜率 | ECE | MCE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始 PD | 4.56% | 5.71% | 0.0560 | 0.9356 | 1.15% | 1.76% | 0.0530 |
| Platt | 6.12% | 5.71% | -0.2273 | 0.9412 | **0.41%** | 1.74% | 0.0529 |
| Isotonic | 6.12% | 5.71% | -0.2655 | 0.9261 | 0.42% | 1.61% | 0.0529 |

校准器只在 2016 验证集 fit；OOT 只 transform。

### 3.4 按终身 PD 重划等级

旧版按整数分硬切，90% 的 OOT 落在 A，A 段实际坏账 8.70% 越过 8% 定义。v2 在报告层改为按**终身 PD**划等级，v3 进一步把 `score_csv.py` 的生产等级也切到终身 PD：

| 等级 | 样本数 | 占比 | 实际坏账率 | 平均预测 PD |
|---|---:|---:|---:|---:|
| A（<8%） | 72,825 | 13.22% | 5.49% | 6.00% |
| B（8–12%） | 95,387 | 17.32% | 9.56% | 10.02% |
| C（12–18%） | 129,529 | 23.52% | 14.45% | 14.86% |
| D（18–25%） | 103,840 | 18.86% | 20.26% | 21.23% |
| E（≥25%） | 149,116 | 27.08% | 34.61% | 37.95% |

### 3.5 阈值与敏感性

验证集选定终身 PD 阈值 **0.27**：

| 指标 | 验证 2016 | OOT 2017–2018Q1 |
|---|---:|---:|
| 批准率 | — | 76.81% |
| 批准人群终身坏账率 | 14.88% | **13.74%** |
| 坏客户捕获率 | — | 44.38% |
| 拒绝人群终身坏账率 | — | 36.31% |

目标坏账率敏感性：

| 目标 | 阈值（终身 PD） | 批准率 | 批准坏账率 | 捕获率 | 拒绝坏账率 | 说明 |
|---:|---:|---:|---:|---:|---:|---|
| 12% | 0.21 | 63.25% | 11.82% | 60.60% | 31.29% | 约束有效，显著收紧 |
| 15% | 0.32 | 84.27% | 14.90% | 33.82% | 40.80% | 约束有效 |
| 18% | 0.57 | 97.65% | 17.98% | 7.47% | 60.24% | 约束有效但很松 |

这是 **OOT 上的事后重选阈值**，不是生产阈值：生产阈值 0.27 是验证集 2016 选出的，固定后再在 OOT 上复核。15%/18% 并不是“非绑定”；v2 的 0.90 结论来自加权标签 `int` 截断，v3 已修复。阈值仍应结合收益/额度/定价重新设定。

### 3.6 FICO / grade 消融

| 特征集 | 变量数 | OOT AUC | 相对 base |
|---|---:|---:|---:|
| base（无 FICO/grade） | 82 | 0.6940 | — |
| +FICO | 83 | 0.6976 | +0.0036 |
| +grade | 83 | 0.7193 | +0.0253 |
| +FICO+grade | 84 | 0.7204 | +0.0263 |

FICO 单变量 IV 0.100；grade IV 0.444。评价里“FICO 应带来 0.72–0.78”的假设在本数据 + MOB12 标签下不成立；真正强的是平台评级，但它不能作为自建评分卡特征。需要说明：+0.0036 是 **82 变量 LightGBM** 的边际，主模型是 14 变量 WOE 评分卡；14 变量 + FICO 的对照尚未做，不能把该数字直接外推到主模型。

### 3.7 VIF 与可解释性代价

- 剔除前初始集最大 VIF **2.94**（`total_bc_limit`，后续被剔除）；最终集最大 **2.38**。
- `selection_log` 记录剔除时 VIF、最大相关、单变量方向；`total_bc_limit` 与 `bc_open_to_buy` 相关 0.737，系数 -0.042，属于共线性符号翻转，剔除有据。
- 单笔节点比较约 **10,053** 次 vs 评分卡 **14** 次查表（约 **718×**）；文件 23.7 KB vs 1.78 MB（约 **77×**）；OOT AUC +0.0342。
- 旧版“2034/12=169.5×”口径已废弃，改为树遍历节点比较。

## 4. 上线后监控

完整方案见 **`outputs/monitoring_plan.md`**（v3）。核心变化：

- 基线分三层：MOB12（建模标签）、校准后 vintage、终身（业务决策）。
- 旧 `14.39%` 基线作废；v3 OOT 批准人群终身坏账率 **13.74%**，目标 ≤15%；OOT 终身坏账率（E[y/e] 加权估计）为 **18.97%**。
- PSI 阈值仍为 0.10/0.25；OOT 评分 PSI 0.0503，特征 PSI 最高 `bc_open_to_buy` 约 0.101。
- 2018Q2+ 不进入 OOT；监控按季度滚动补充最近成熟 cohort。
- 案例 A（PSI 0.32）与案例 B（PSI 0.08 但批准坏账 18%）保留，均按终身口径排查。

## 5. 快速开始

```powershell
cd D:\DATAsupport\lendingclub_scorecard_2018

# 运行完整 pipeline（会重建数据缓存，约 5–30 分钟，取决于是否复用 ablation 缓存）
D:\DATAsupport\credit_scorecard\.venv\Scripts\python.exe scripts\run_pipeline.py

# 只重算 v3 阈值/校准表（读取 OOT 打分，不重新训练）
D:\DATAsupport\credit_scorecard\.venv\Scripts\python.exe scripts\recompute_v3_tables.py

# 生成 v3 报告（读取 outputs，不重新训练）
D:\DATAsupport\credit_scorecard\.venv\Scripts\python.exe scripts\build_v3_report.py

# 对新申请人打分（输出校准 PD、终身 PD、等级和 approve）
D:\DATAsupport\credit_scorecard\.venv\Scripts\python.exe scripts\score_csv.py --input new_applicants.csv --output scored.csv
```

打开：

- `outputs/report_v3.html`：v3 主报告（阈值加权修复/终身口径/打分链路）
- `outputs/report.html`：同一份 v3 主报告
- `outputs/report_v2.html`：v2 报告归档
- `outputs/report_v1_terminal.html`：v1 终态口径归档
- `outputs/monitoring_plan.md`：v3 监控方案
- `outputs/decision_policy.json`：Platt 校准参数 + e(term) + 0.27 阈值 + 终身 PD 等级带
- `outputs/scorecard.csv` / `scorecard.json`：评分卡（`grade_cutoffs` 是 MOB12 旧口径，生产等级以 `decision_policy.json` 为准）
- `outputs/tables/calibrator_comparison_oot.csv`、`threshold_sensitivity.csv`、`threshold_stability.csv`、`sensitivity_ablation.csv`、`vif_initial.csv`、`grade_calibration_lifetime_oot.csv`
- `outputs/figures/20_calibration_curve.png`

`outputs/predictions/oot_scores.csv` 约 173MB，未提交到 Git；运行 pipeline 自动生成。

## 6. 目录结构

```text
lendingclub_scorecard_2018/
├── src/lendingclub_scorecard/
│   ├── config.py          # 数据口径、MOB12、时间切分、评分卡参数
│   ├── data.py            # 分块读取、清洗、MOB 标签、时间切分、预处理
│   ├── calibration.py     # Platt/isotonic、暴露比例、终身换算
│   ├── binning.py         # 等频/二值分箱、单调合并、WOE/IV
│   ├── model.py           # Logistic + LightGBM + 消融
│   ├── evaluation.py      # AUC/KS/校准/提升/PSI/阈值扫描
│   ├── scorecard.py       # 标准评分卡
│   ├── policy.py          # 校准 + 终身 PD + 等级 + 审批策略
│   ├── plots.py           # 20 张图
│   ├── monitoring.py      # 监控方案与基线
│   ├── report.py          # 旧版 HTML 报告生成器（v1）
│   └── pipeline.py        # 全流程编排
├── scripts/run_pipeline.py
├── scripts/build_v2_report.py      # v2 报告归档
├── scripts/build_v3_report.py      # v3 报告
├── scripts/recompute_v3_tables.py  # 不重训重算 v3 表
├── scripts/score_csv.py
├── tests/
├── outputs/
└── README.md
```

## 7. 评价回应与剩余局限

- **① 删失偏差：** 已改为 MOB12 固定窗口 + 终身换算；旧 14.39% 基线废弃。
- **② 阈值敏感性：** v3 修复了加权标签 `int` 截断；12%/15%/18% 对应 0.21/0.32/0.57，批准率 63.25%/84.27%/97.65%。15% 约束绑定，18% 也非完全非绑定。
- **③ FICO：** 已做 LightGBM 消融；FICO 仅 +0.0036 AUC。尚未做 14 变量 WOE 评分卡 + FICO 的对照，不能直接外推。
- **④ term：** 已披露期限结构（36/60 期终身 12.7%/25.8%）；生产必须锁定 term 时点。主流程只按 term 估算 e，尚未使用 grade×term 的细分 e。
- **⑤ VIF：** 已双口径（剔除前 2.94 / 最终 2.38）并解释负系数。
- **⑥ LightGBM 公平性：** 已加同 14 变量对照（+0.0056）。
- **⑦ 代码瑕疵：** v3 修复阈值敏感性 `int` 截断；统一 14 变量标签；补齐 weighted-label 单测；PSI 用 score 口径已说明。
- **⑧ 校准：** Platt/isotonic 已完成；终身 PD 在 grade 分层仍有 0.76–0.92 的系统性偏高，需要 term×score 分层 e 或终身 PD 重校准。
- **⑨ 打分链路：** v3 新增 `decision_policy.json`，`score_csv.py` 输出 `prob_lifetime`/`grade`/`approve`；不再把未校准 MOB12 PD 当作审批概率。
- **仍未做：** RejectStats 拒绝推断；LGD/EAD、资金成本、获客成本、定价联动的收益优化；公平性/反歧视；独立验证与生产部署。

## 8. 版本归档

| 版本 | 主要变化 | 归档 |
|---|---|---|
| v3.0-p0（2026-10-21） | 修阈值加权 bug；统一 E[y/e]=18.97%；新增决策策略与打分链路 | 当前 v3 分支/README、`outputs/report_v3.html` |
| v2.0-mob12（2026-10-20） | MOB12 固定窗口、终身换算、校准、FICO/grade 消融 | `README_v2_mob12.md`、`outputs/report_v2.html`、`EVALUATION_RESPONSE_v2.md` |
| v1.0-terminal（2026-10-19） | 初版终态口径（已废弃 14.39% 基线） | `README_v1_terminal.md`、`outputs/report_v1_terminal.html` |

> 本项目仅用于学习、建模复现与监控方案演示，不构成真实授信建议。
