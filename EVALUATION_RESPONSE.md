# 评审意见处理记录（v3 / P0）

> 对应第二轮评审。v2 的 `EVALUATION_RESPONSE.md` 已归档为 `EVALUATION_RESPONSE_v2.md`；v2 报告保留在 `outputs/report_v2.html`。

## 结论先说

第二轮评审指出的核心问题成立：v2 的 `threshold_sensitivity` 把终身加权标签 `0/1/e` 强转成 `int`，36 期权重 2.95 被截成 2、60 期权重 4.0 保留为 4，导致“12%/15%/18% 目标都几乎不约束”的错误结论；同时报告头图用 18.18%、阈值表用 18.97%，两套终身口径混用；报告层的终身 PD/等级/0.27 阈值也没有进入 `score_csv.py`。

v3 只做 P0 修复，不重训模型：修正代码、重算阈值/校准表、统一终身口径、补齐打分链路、归档 v2，并把更细的 e 分层与终身 PD 校准留到 P1。

## 逐条处理

| 评审点 | 状态 | 处理 | v3 结果 |
|---|---|---|---|
| ① 阈值敏感性 `int` 截断 | 已修 | `calibration.threshold_sensitivity` 改为 `dtype=float`；新增 weighted-label 回归测试 | 12%/15%/18% → 0.21/0.32/0.57，批准率 63.25%/84.27%/97.65% |
| ② “15%/18% 目标非绑定” | 已收回 | 用 `threshold_scan_oot.csv` 复算并重建报告 | 15% 约束绑定（84.27% 批准）；18% 也非完全非绑定（97.65%） |
| ③ 18.18% vs 18.97% | 已统一 | pipeline 统一用 `E[y/e]`；`calibration_by_scale`、`pd_scale_comparison`、metadata、README、监控方案同步 | OOT 终身坏账率 **18.97%**；终身平均 PD 20.30%，保守偏差约 1.33pp |
| ④ 0.27 阈值下拒绝坏账率 55.88% | 已修 | 该值来自 bug 表的 0.90 行；按 0.27 重算 | **36.31%** |
| ⑤ 报告与打分脚本脱节 | 已修 | 新增 `outputs/decision_policy.json`；`score_csv.py` 加载校准器、e(term)、0.27 阈值、终身 PD 等级带 | 输出 `prob_mob12_raw`、`prob_mob12_calibrated`、`prob_lifetime`、`grade`、`approve` |
| ⑥ 等级仍按分数切、90% A | 已修 | 生产等级改为按 `prob_lifetime` 的 8/12/18/25% 切；`scorecard.json` 保留旧 `grade_cutoffs` 并标注 legacy | CLI 输出终身 PD 等级；报告 A 段占比 13.22%，不再是 90% |
| ⑦ 12/14 变量标签不一致 | 已修 | pipeline 表标签与 CSV 统一为 14 变量；LightGBM 同变量模型改名为 `LightGBM-同变量（14 变量）` | 报告表与变量数一致 |
| ⑧ config docstring 写 logit 平移 | 已修 | 改为概率空间 `p_lifetime = p_vintage / e(term)` | 与 calibration.py/report 一致 |
| ⑨ 加权标签测试缺失 | 已补 | 新增 `tests/test_calibration.py`（敏感性/稳定性）与 `tests/test_policy.py`（校准/换算/等级/审批） | 全部单测通过 |
| ⑩ 固定路径/可复现性 | 部分 | README 改为说明 `LC_DATA_DIR`；新增 `scripts/recompute_v3_tables.py` 可不重训重算 | 仍需数据文件才能跑完整 pipeline |

## 关键数字（v3）

- OOT MOB12 坏账率 5.71%；OOT 终身坏账率（E[y/e]）**18.97%**；平均终身 PD 20.30%。
- 评分卡 OOT AUC 0.6598 / KS 0.2314；LightGBM-full 0.6940。
- 生产阈值 0.27（验证集 2016 选出）：OOT 批准率 76.81%、批准坏账 13.74%、拒绝坏账 **36.31%**、捕获 44.38%。
- 敏感性（OOT 事后重选）：12% → 0.21 / 63.25% / 11.82%；15% → 0.32 / 84.27% / 14.90%；18% → 0.57 / 97.65% / 17.98%。
- Platt ECE 1.15% → 0.41%；终身 PD 20.30% vs 18.97%，高估约 1.33pp。

## 仍未做（P1/P2）

- `e` 只按 term 分段；`emergence_grade_within_term.csv` 显示 grade×term 仍有明显差异，终身 PD 在 grade 分层 ratio 0.76–0.92，需要更细的 e 或终身 PD 重校准。
- FICO 只做了 82 变量 LightGBM 的消融，未做 14 变量 WOE 评分卡 + FICO 的对照。
- 拒绝推断（RejectStats）、LGD/EAD/资金成本/定价联动、公平性/反歧视、压力测试、独立验证仍未做。
- `last_pymnt_d` 仍是核销时点的近似；MOB12 标签与真正逐月 MOB 曲线有差距。
- 生产部署前仍需模型风险评级、审批留痕与独立验证。
