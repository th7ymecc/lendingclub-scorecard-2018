"""上线后监控方案：指标、PSI 阈值、预警流程与重训触发条件。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def psi_threshold_table(cfg) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PSI 区间": f"< {cfg.psi_green:.2f}",
                "等级": "绿色 / 稳定",
                "含义": "评分或特征分布与训练集基本一致",
                "动作": "正常监控，无需干预",
            },
            {
                "PSI 区间": f"{cfg.psi_green:.2f} ~ {cfg.psi_amber:.2f}",
                "等级": "黄色 / 关注",
                "含义": "出现中等漂移，需定位是客群变化还是数据问题",
                "动作": "2 个工作日内完成归因；加强监控；评估是否收紧阈值",
            },
            {
                "PSI 区间": f"> {cfg.psi_amber:.2f}",
                "等级": "红色 / 预警",
                "含义": "显著漂移，模型排序/校准可能失效",
                "动作": "1 个工作日内升级；暂停自动审批或人工复核；准备重训/回滚",
            },
        ]
    )


def monitoring_metrics_table(cfg) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("评分 PSI", "月度", f"<{cfg.psi_green:.2f} / {cfg.psi_amber:.2f}", "连续 2 月 >0.25 触发重训"),
            ("特征 PSI（Top 10）", "月度", f"单特征 >{cfg.psi_amber:.2f}", "定位漂移来源与客群/渠道"),
            ("OOT AUC", "季度（成熟 vintage）", f"较基线下降 >{cfg.auc_drop_amber:.2f} 关注，>{cfg.auc_drop_red:.2f} 预警", "区分能力下降"),
            ("OOT KS", "季度（成熟 vintage）", f"较基线下降 >{cfg.ks_drop_amber:.2f} 关注，>{cfg.ks_drop_red:.2f} 预警", "排序能力下降"),
            ("批准率", "日/周", "较基线偏移 ±5pp", "策略或客群变化"),
            ("批准人群坏账率", "月度", f"较基线相对上升 >{cfg.bad_rate_rel_amber:.0%} 关注，>{cfg.bad_rate_rel_red:.0%} 预警", "资产质量恶化"),
            ("校准/分数段实际坏账率", "月度", "实际坏账率 − 平均预测 PD >5pp，或校准斜率显著偏离", "绝对风险与定价"),
            ("Vintage 坏账率", "季度", "同账龄对比，连续 2 个 cohort 恶化", "客群/宏观/策略变化"),
            ("评分均值/标准差", "月度", "均值偏移 >20 分或标准差变化 >20%", "分数分布异常"),
            ("关键字段缺失率", "日", "较训练集上升 >2 倍或 >5pp", "数据链路/进件渠道异常"),
            ("拒绝率/人工复核率", "日", "较策略基线偏移 ±5pp", "审批策略或模型异常"),
            ("模型/数据延迟", "日", "超过 SLA（如 T+1 12:00）", "数据/系统故障"),
        ],
        columns=["监控指标", "频率", "预警阈值", "业务含义"],
    )


def alert_flow() -> list[tuple[str, str, str]]:
    return [
        (
            "T+0 自动检测",
            "监控平台每日跑数据质量、评分 PSI、批准率、缺失率；任一红色指标自动告警到风控模型群。",
            "监控工程师",
        ),
        (
            "T+0 ~ T+1 分级响应",
            "黄色：模型分析师 2 个工作日内归因；红色：模型负责人 1 个工作日内组织临时风险会，必要时收紧 cutoff / 人工复核。",
            "模型负责人",
        ),
        (
            "T+1 ~ T+3 归因",
            "按渠道、客群、分数段、特征 PSI 拆解；核对数据口径、进件政策、宏观/季节性因素；输出归因报告。",
            "模型 + 策略",
        ),
        (
            "T+3 ~ T+10 处置",
            "选择：维持观察 / 调整截距校准 / 收紧阈值 / 重新分箱 / 重训 / 回滚；所有变更需审批与留痕。",
            "风险委员会",
        ),
        (
            "月度复盘",
            "模型委员会月度 review：AUC/KS、PSI、Vintage、批准率、坏账率、策略收益；更新监控基线。",
            "模型委员会",
        ),
    ]


def hypothetical_cases() -> list[dict]:
    """用于演练的假设性报警案例。"""
    return [
        {
            "case": "案例 A：某月 Score PSI 突升到 0.32",
            "signal": "月度 Score PSI 从 0.03 升到 0.32（红色）；批准率同期从 82% 降到 74%。",
            "steps": [
                "先查数据：进件量、字段缺失率、T+1 数据延迟、渠道占比是否突变；确认不是数据链路问题。",
                "拆分数分布：按渠道/产品/客群看 score 均值、P25/P50/P75 与拒绝率，判断漂移来自哪一段。",
                "拆特征 PSI：取 Top 10 特征 PSI，检查是否由单一特征（如 term、loan_income_ratio、bc_open_to_buy）贡献 >50%。",
                "对齐业务事件：是否新渠道上线、客群准入政策变化、额度/定价调整、宏观冲击或季节性。",
                "验证影响：在最近成熟 vintage 上重算 AUC/KS、分数段实际坏账率、批准坏账率与校准斜率。",
            ],
            "decision": "若为数据问题：修复链路并暂停自动审批；若为真实客群漂移但排序尚可：收紧 cutoff/人工复核并评估截距校准；若 AUC/KS 明显下降：启动重训或回滚。",
        },
        {
            "case": "案例 B：PSI 仅 0.08，但批准坏账率升到 22%",
            "signal": "Score PSI 仍为绿色（0.08），但批准人群坏账率 22%，高于 15% 目标；平均预测 PD 仅 16%。",
            "steps": [
                "先做校准检查：按分数段对比平均预测 PD 与实际坏账率，拟合校准斜率/截距，确认是否整体低估。",
                "检查客群/账龄结构：2018 类早成熟 cohort 占比是否上升；坏账率是否被“更早核销”的样本结构放大。",
                "查策略与定价：审批阈值、额度、利率、渠道组合是否变化；是否存在风险更高的新客群。",
                "对照 Vintage：同账龄 cohort 的坏账率是否连续恶化；是宏观/客群问题还是模型失效。",
                "在验证集上重算目标阈值：若只是校准偏移，优先截距校准/收紧阈值；若排序下降，再考虑重训。",
            ],
            "decision": "不要因为 PSI 绿色就忽略校准报警。若确认是绝对风险上移，先收紧阈值/人工复核并做截距校准；连续 2 个月恶化则启动重训。",
        },
    ]


def baseline_table(ctx: dict) -> pd.DataFrame:
    cfg = ctx["cfg"]
    oot = ctx["metrics"]["logit"]["oot"]
    valid = ctx["metrics"]["logit"]["valid"]
    lgbm_oot = ctx["metrics"]["lgbm"]["oot"]
    conf = ctx["threshold_confusion"]
    rows = [
        ("训练期", "2013-01 ~ 2015-12", f"{ctx['train_shape'][0]:,}"),
        ("验证期", "2016-01 ~ 2016-12", f"{ctx['valid_shape'][0]:,}"),
        ("OOT 期", "2017-01 ~ 2018-12", f"{ctx['oot_shape'][0]:,}"),
        ("验证集 AUC", "—", f"{valid['auc']:.4f}"),
        ("OOT AUC（评分卡）", "—", f"{oot['auc']:.4f}"),
        ("OOT KS（评分卡）", "—", f"{oot['ks']:.4f}"),
        ("OOT 平均预测 PD", "—", f"{oot['mean_pd']:.2%}"),
        ("OOT 实际坏账率", "—", f"{oot['bad_rate']:.2%}"),
        ("OOT AUC（LightGBM）", "—", f"{lgbm_oot['auc']:.4f}"),
        ("评分 PSI（训练 vs 验证）", "阈值 <0.10", f"{ctx['psi_valid']:.4f}"),
        ("评分 PSI（训练 vs OOT）", "阈值 <0.10", f"{ctx['psi_oot']:.4f}"),
        ("选定阈值", "验证集选择", f"{conf['threshold']:.2f}"),
        ("OOT 批准率", "—", f"{conf['approval_rate']:.2%}"),
        ("OOT 批准人群坏账率", f"目标 ≤{cfg.threshold_target_bad_rate:.0%}", f"{conf['approved_bad_rate']:.2%}"),
        ("OOT 坏客户捕获率", "—", f"{conf['bad_capture']:.2%}"),
    ]
    return pd.DataFrame(rows, columns=["项目", "口径", "基线值"])


def build_monitoring_plan(ctx: dict, path: str | Path) -> None:
    cfg = ctx["cfg"]
    psi_table = psi_threshold_table(cfg)
    metrics = monitoring_metrics_table(cfg)
    baseline = baseline_table(ctx)

    def md_table(frame: pd.DataFrame) -> str:
        headers = "| " + " | ".join(frame.columns) + " |"
        sep = "|" + "|".join(["---"] * len(frame.columns)) + "|"
        rows = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([headers, sep, *rows])

    flow = "\n".join(
        f"{i}. **{stage}**（{owner}）：{action}"
        for i, (stage, action, owner) in enumerate(alert_flow(), start=1)
    )
    case_md = "\n\n".join(
        "### " + case["case"] + "\n\n"
        + f"**触发信号：** {case['signal']}\n\n"
        + "**排查顺序：**\n\n"
        + "\n".join(f"{i}. {step}" for i, step in enumerate(case["steps"], start=1))
        + f"\n\n**处置建议：** {case['decision']}"
        for case in hypothetical_cases()
    )
    text = f"""# 上线后监控方案

> 适用对象：LendingClub 个人信贷申请评分卡（WOE + Logistic Regression）。  
> 监控目标：在客群、渠道、宏观环境或数据链路变化时，尽早发现模型排序能力、校准能力与资产质量的变化，并给出可执行的处置动作。

## 1. 监控原则

- **分层监控**：数据质量 → 分数分布 → 区分能力 → 资产质量，逐层递进。
- **先归因、后动作**：PSI 报警不等于模型失效，必须先区分数据问题、客群变化、策略变化与真实风险恶化。
- **阈值可审计**：所有预警、阈值调整、模型变更均需记录时间、原因、审批人与影响评估。
- **成熟度优先**：坏账率与 AUC/KS 只在贷款成熟（如放款后 12/24 个月）的 vintage 上评估，避免近期 cohort 的删失偏差。

## 2. PSI 监控阈值

{md_table(psi_table)}

> **重要：** PSI 稳定不代表校准稳定。本项目 2017–2018 OOT 的平均预测 PD 为 {ctx['metrics']['logit']['oot']['mean_pd']:.2%}，实际坏账率为 {ctx['metrics']['logit']['oot']['bad_rate']:.2%}，说明分数分布稳定但绝对风险被低估。因此监控矩阵必须同时包含“分数段实际坏账率/校准”，不能只看 PSI。

补充规则：

- 评分 PSI 连续 2 个月处于黄色区间，升级为红色管理。
- 单特征 PSI > {cfg.psi_amber:.2f} 且对评分贡献较大（IV/系数排名前 10）时，优先排查该特征的口径与进件渠道。
- 特征 PSI 需同时看“统计漂移”和“业务合理性”：例如收入分布整体上移可能是客群升级，也可能是数据口径变化。

## 3. 监控指标矩阵

{md_table(metrics)}

## 4. 预警流程

{flow}

### 升级与降级

- 黄色预警连续 2 个周期未收敛 → 升级红色。
- 红色预警完成归因且指标恢复 → 降级为黄色观察 1 个月，再回到绿色。
- 任何阈值调整、模型重训、回滚都必须经过模型委员会审批并保留变更记录。

## 5. 假设性报警案例

{case_md}

## 6. 处置动作库

| 场景 | 首选动作 | 备选动作 |
|---|---|---|
| 数据口径/缺失异常 | 修复数据链路，暂停自动审批 | 临时规则兜底 |
| 客群结构变化但排序能力尚可 | 调整截距/分数校准 | 分客群阈值 |
| PSI 高且 AUC/KS 明显下降 | 重新分箱/重训 | 回滚上一版本 |
| 资产质量恶化但模型区分度正常 | 收紧 cutoff、提高人工复核比例 | 调整授信额度/定价 |
| 宏观/政策冲击 | 临时风险策略、压力测试 | 暂停新增、限额管理 |

## 7. 重训触发条件

满足任一条件即启动重训评估：

1. 评分 PSI > {cfg.psi_amber:.2f} 连续 2 个月，或单月 > 0.40。
2. OOT AUC 较基线下降 > {cfg.auc_drop_red:.2f}，或 KS 下降 > {cfg.ks_drop_red:.2f}。
3. 同账龄 Vintage 坏账率连续 2 个 cohort 相对恶化 > {cfg.bad_rate_rel_red:.0%}。
4. 批准率或拒绝率相对基线偏移 > 10pp，且无法用策略调整解释。
5. 关键特征口径、数据源、进件政策发生重大变化。
6. 监管/合规要求或模型有效期到期（建议至少每年独立验证一次）。

## 8. 角色与留痕

| 角色 | 职责 |
|---|---|
| 模型负责人 | 模型性能、归因、重训/回滚决策 |
| 风险策略 | 阈值、批准率、人工复核策略 |
| 数据工程 | 数据链路、字段口径、延迟与缺失监控 |
| 监控平台 | 自动跑批、告警、看板与历史留痕 |
| 模型委员会 | 审批重大变更、月度复盘 |

## 9. 本项目监控基线

{md_table(baseline)}

> 注意：2017–2018 OOT 对应 cohort 中约 {ctx['current_share_oot']:.1%} 为 Current，其中 2018 年 Current 占比约 {ctx['current_share_2018']:.1%}，属于“早成熟” cohort；其绝对坏账率会随账龄继续上升。因此 OOT 更适合看排序能力（AUC/KS）与 PSI 漂移，绝对坏账率应结合同账龄 vintage 曲线解读。
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
