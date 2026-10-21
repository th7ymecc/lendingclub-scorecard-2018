# 版本记录

本项目保留完整 Git 历史；旧版报告/监控方案单独归档，v1、v2 与 v3 均可追溯。

## v3.0-p0 — 阈值加权修复 / 终身口径统一 / 打分链路（2026-10-21）

**核心修正**

- 修复 `threshold_sensitivity` 对终身加权标签 `0/1/e` 的 `int` 截断：12%/15%/18% 目标对应阈值由 0.45/0.90/0.90 修正为 **0.21/0.32/0.57**，批准率 **63.25%/84.27%/97.65%**。
- 收回“15%/18% 目标非绑定”的结论：15% 约束绑定，18% 也并非完全非绑定。
- 终身坏账率统一为 **E[y/e] = 18.97%**（替代 mean(y)/mean(e)=18.18% 的比值估计）；0.27 阈值下拒绝人群坏账率由 55.88% 修正为 **36.31%**。
- 新增 `outputs/decision_policy.json`：Platt 参数 + e(term) + 0.27 阈值 + 终身 PD 等级带；`score_csv.py` 输出 `prob_mob12_raw`、`prob_mob12_calibrated`、`prob_lifetime`、`grade`、`approve`。
- 生产等级改为按终身 PD 8/12/18/25% 划分；`scorecard.json` 的旧分数切点标注为 legacy。
- 统一 14 变量标签；修正 config docstring 的“logit 平移”表述。
- 新增 `tests/test_calibration.py`、`tests/test_policy.py`；新增 `scripts/recompute_v3_tables.py`（不重训重算）。
- 新增 v3 报告 `outputs/report_v3.html`；`outputs/report.html` 指向 v3。

**归档**

- `README_v2_mob12.md`、`EVALUATION_RESPONSE_v2.md`、`outputs/report_v2.html`、`outputs/monitoring_plan_v2_mob12.md`。
- `outputs/run_metadata_v2.json`、`outputs/tables/threshold_sensitivity_v2.csv`、`outputs/tables/threshold_stability_v2.csv`。
- Git tag：`v3.0-p0`。

## v2.0 — MOB12 / 终身 PD / 校准 / 敏感性（2026-10-20）

**核心修正**

- 标签从“只保留终态贷款”改为 **MOB12 固定窗口**：所有 cohort 截到 12 个月，2018Q2+ 观察不足 12 个月不进入 OOT。
- 增加**期限分段暴露比例 e**，把 vintage PD 换算成终身 PD：`p_lifetime = p_vintage / e(term)`。
- 旧版 `OOT 批准坏账率 14.39%` 基线**作废**；v2 为终身口径 **13.74%**（目标 ≤15%）。
- 验证集拟合 **Platt / isotonic** 校准器，OOT 只 transform；原始 ECE 1.15% → Platt 0.41%。
- 新增阈值目标 **12%/15%/18%** 敏感性表。
- 新增 **FICO / grade 消融**：FICO +0.0036 AUC，grade +0.0253（grade 不可入主模型）。
- 新增 **LightGBM-12（同 14 变量）** 对照：模型形式 +0.0056，变量数量 +0.0286。
- VIF 改为**剔除前 + 最终**双口径；selection_log 记录剔除时 VIF/相关/单变量方向。
- 可解释性代价从“树数/变量数”改为**节点比较次数**：约 718×，文件约 77×。
- 等级改为按**终身 PD** 划分，解决旧版 90% 落在 A、A 段实际 8.70% 越过 8% 定义的问题。

**归档**

- `README_v1_terminal.md`：v1 README 原样归档。
- `outputs/report_v1_terminal.html`：v1 终态口径报告。
- `outputs/monitoring_plan_v1_terminal.md`：v1 监控方案。
- v1 commit：`f3baac9`；v2 commit：`232bc84`。
- Git tag：`v1.0-terminal`、`v2.0-mob12`。

## v1.0 — 初版（2026-10-19）

- 2012–2018 数据，训练 2013–2015 / 验证 2016 / OOT 2017–2018。
- 终态标签（Fully Paid vs Charged Off/Default）。
- WOE/IV 评分卡、PSI、Vintage、阈值扫描、LightGBM 对照、监控方案。
- **已知问题：** OOT 只保留终态贷款，2018 年进入 OOT 的是最快结清样本；14.39% 批准坏账率被低估，不能作为监控基线。v2 已修正。
