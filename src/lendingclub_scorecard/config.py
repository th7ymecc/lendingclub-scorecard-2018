"""LendingClub 信用风险评分卡项目配置。

目标定义（MOB 对齐 vintage 口径）
---------------------------------
数据快照日为 2019-03-31，各放款 cohort 的表现成熟度差异极大：2013 年放款已观察
70+ 个月，2018Q4 放款只观察约 5 个月。旧版"只保留终态样本"的做法会让不同 cohort
处在不同的表现窗口上，坏账率不可横向比较（实测 2016Q2 旧口径 25.35%，而 MOB12
vintage 口径仅 6.81%，相差 3.7 倍）。

因此本版统一采用固定表现窗口的 vintage 口径，窗口长度 PERFORMANCE_MOB 个月：
    bad_flag_v12 = 1  若贷款状态为 Charged Off / Default 且核销发生在 MOB <= 12
    bad_flag_v12 = 0  若 obs_mob >= 12（该 cohort 已可观察到 MOB12）且未在 MOB12 前核销
    bad_flag_v12 = NaN 若 obs_mob < 12（成熟度不足，无法评估，剔除出建模样本）

该口径下所有 cohort 都截到同一个 MOB12 时点，坏账率可跨期公平比较。

注意 vintage 口径的绝对值低于终身坏账率：MOB12 时坏样本仅暴露约 30.4%，实测
MOB12 vintage 坏账率 4.84%，而成熟 cohort 终身坏账率 15.91%。业务决策需换算到
终身口径，换算通过 logit 空间平移 log(1 / 暴露占比) 实现（详见 data.py）。

时间切分（按 issue_d）
-----------------------
    Train : 2013-01-01 ~ 2015-12-31
    Valid : 2016-01-01 ~ 2016-12-31
    OOT   : 2017-01-01 ~ 2018-03-31
OOT 上界收窄到 2018-03-31：快照日 2019-03-31 减去 12 个月表现窗口，2018Q2 及以后
放款的贷款观察期不足 12 个月，无法在 MOB12 口径下评估，必须排除而非硬算。

2012 年仅用于 Vintage 客群结构展示，不进入建模（部分征信字段缺失）。
主模型不使用 FICO，但本版提供 FICO 对照实验以量化该取舍的代价（见 pipeline.py）。

特征口径
--------
主模型不使用 LendingClub 自身的 grade / sub_grade / int_rate，避免复用 LC 的
评级与定价结果；只使用申请时点可获得的申请表/征信局/交易线变量。

last_pymnt_d 仅用于计算 MOB（表现成熟度），不进入特征矩阵，因此不构成标签泄漏：
它决定的是"这笔贷款的标签观察到第几个月"，与"用什么信息预测"是两件事。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(os.environ.get("LC_DATA_DIR", str(PROJECT_ROOT.parent)))

ID_COLUMN = "loan_id"
TARGET = "bad_flag"
RAW_TARGET = "loan_status"

# ---------------------------------------------------------------- MOB 对齐口径
# 数据快照日：由 last_pymnt_d 最大值（2019-03）与 Current 贷款分布实测推断。
# 所有"已观察多少个月"的计算都以这一天为基准，不得凭空假设。
SNAPSHOT_DATE = "2019-03-31"

# 统一表现窗口（months on book）。选 12 个月的理由：
#   1) 快照日 2019-03-31 往前推 12 个月 = 2018-03-31，可保留最大范围的 OOT；
#   2) 12 个月已覆盖约 30% 的终身坏账暴露，信号量足够建模；
#   3) 更长的窗口（如 24/36 个月）会把 OOT 上界推到 2017 年，样本骤减。
PERFORMANCE_MOB = 12

# vintage 口径标签（MOB12 前是否核销），与旧版终身口径 bad_flag 并存以便对比
TARGET_VINTAGE = "bad_flag_v12"

# 用于计算 MOB 的日期字段：last_pymnt_d 给出终态发生的时点。
# 这些字段只参与成熟度计算，绝不进入特征矩阵。
MOB_DATE_COLUMNS = ("last_pymnt_d",)

# 成熟 cohort（2013-2014 放款、观察 >= 60 个月）实测的坏账暴露曲线，
# 用于把 MOB12 vintage 坏账率换算到终身口径。来源见 outputs/tables/emergence_curve.csv。
BAD_EMERGENCE_BY_MOB = {
    6: 0.0971,
    9: 0.1886,
    12: 0.3040,
    15: 0.3865,
    18: 0.4895,
    21: 0.5902,
    24: 0.6789,
    30: 0.8178,
    36: 0.9067,
    48: 0.9724,
    60: 0.9945,
}

TERMINAL_STATUSES = ("Fully Paid", "Charged Off", "Default")
BAD_STATUSES = ("Charged Off", "Default")
EXCLUDED_STATUSES = (
    "Current",
    "Late (31-120 days)",
    "In Grace Period",
    "Late (16-30 days)",
    "Issued",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
)

NINF = float("-inf")
PINF = float("inf")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    name_cn: str
    description: str
    group: str
    kind: str = "numeric"
    direction: int = 0  # 0 = 自动判断；+1/-1 强制方向
    edges: tuple[float, ...] | None = None
    monotonic: bool = True


# 只保留 2013-2015 训练期有稳定取值的字段：
# il_util / all_util / open_rv_12m / open_rv_24m / open_il_12m / open_il_24m /
# open_acc_6m / total_cu_tl / inq_fi / inq_last_12m / open_act_il /
# mths_since_rcnt_il / total_bal_il / max_bal_bc 在训练期整列缺失，已剔除。
_SPECS = [
    # ---------------- 申请与收入 ----------------
    ("loan_amnt", "贷款金额", "借款人申请的贷款本金（美元）", "申请与收入"),
    ("annual_inc", "年收入", "借款人自报年收入（美元）", "申请与收入"),
    ("dti", "负债收入比", "除本次贷款外的月负债 / 月收入（%）", "申请与收入"),
    ("emp_length", "工作年限", "雇佣年限：0=<1年，10=10年以上，-1=缺失", "申请与收入"),
    # ---------------- 信用历史 ----------------
    ("delinq_2yrs", "近两年逾期次数", "过去 24 个月 30+ 天逾期次数", "信用历史"),
    ("inq_last_6mths", "近半年查询次数", "过去 6 个月信用查询次数（不含车贷/房贷）", "信用历史"),
    ("mths_since_last_delinq", "距上次逾期月数", "距离最近一次逾期的月数；缺失=无逾期", "信用历史"),
    ("mths_since_last_record", "距上次公共记录月数", "距离最近公共记录的月数；缺失=无记录", "信用历史"),
    ("open_acc", "开放账户数", "信用档案中当前开放的信用账户数", "信用历史"),
    ("pub_rec", "公共记录数", "负面公共记录数量", "信用历史"),
    ("total_acc", "总账户数", "信用档案中的总账户数", "信用历史"),
    ("mort_acc", "房贷账户数", "房贷账户数量", "信用历史"),
    ("pub_rec_bankruptcies", "破产记录数", "公共破产记录数量", "信用历史"),
    ("tax_liens", "税收留置权数", "税收留置权数量", "信用历史"),
    ("num_accts_ever_120_pd", "曾120+逾期账户数", "历史上曾 120 天以上逾期的账户数", "信用历史"),
    ("mo_sin_old_rev_tl_op", "最早循环账户月数", "最早循环账户开立至今月数", "信用历史"),
    ("mo_sin_rcnt_rev_tl_op", "最近循环账户月数", "最近循环账户开立至今月数", "信用历史"),
    ("mo_sin_rcnt_tl", "最近账户月数", "最近一个新账户开立至今月数", "信用历史"),
    ("mo_sin_old_il_acct", "最早分期账户月数", "最早分期贷款账户开立至今月数", "信用历史"),
    ("pct_tl_nvr_dlq", "从未逾期账户占比", "从未逾期账户占全部交易线的比例（%）", "信用历史"),
    # ---------------- 负债、额度与使用率 ----------------
    ("revol_bal", "循环信用余额", "循环信用账户总余额", "负债与额度"),
    ("revol_util", "循环额度使用率", "循环信用余额 / 循环授信额度（%）", "负债与额度"),
    ("tot_hi_cred_lim", "总授信额度", "全部账户的高信用额度合计", "负债与额度"),
    ("total_bal_ex_mort", "除房贷外总余额", "除房贷外全部账户的余额合计", "负债与额度"),
    ("total_bc_limit", "银行卡总额度", "银行卡账户总额度", "负债与额度"),
    ("total_il_high_credit_limit", "分期贷款总额度", "分期贷款账户总额度", "负债与额度"),
    ("total_rev_hi_lim", "循环信用总额度", "循环信用账户总额度", "负债与额度"),
    ("bc_util", "银行卡使用率", "银行卡余额 / 银行卡总额度（%）", "负债与额度"),
    ("bc_open_to_buy", "银行卡可用额度", "银行卡总可用额度", "负债与额度"),
    ("percent_bc_gt_75", "银行卡超75%额度占比", "额度使用率超过 75% 的银行卡占比（%）", "负债与额度"),
    ("avg_cur_bal", "平均账户余额", "全部账户平均当前余额", "负债与额度"),
    # ---------------- 账户与查询行为 ----------------
    ("acc_open_past_24mths", "近24个月新开账户数", "过去 24 个月新开的交易账户数", "账户行为"),
    ("num_actv_rev_tl", "活跃循环账户数", "当前活跃的循环信用账户数", "账户行为"),
    ("num_rev_accts", "循环账户数", "循环信用账户总数", "账户行为"),
    ("num_sats", "满意账户数", "状态为满意的账户数", "账户行为"),
    ("num_tl_90g_dpd_24m", "近24个月90+逾期账户数", "过去 24 个月 90 天以上逾期的账户数", "账户行为"),
    ("num_tl_30dpd", "当前30天逾期账户数", "当前 30 天逾期的账户数（近 2 个月更新）", "账户行为"),
    ("num_tl_op_past_12m", "近12个月新开账户数", "过去 12 个月新开的账户数", "账户行为"),
    ("mths_since_recent_inq", "距最近查询月数", "距离最近一次信用查询的月数；缺失=无查询", "账户行为"),
    ("mths_since_recent_bc", "距最近银行卡开户月数", "距离最近一张银行卡开户的月数", "账户行为"),
    ("num_il_tl", "分期贷款账户数", "分期贷款账户数量", "账户行为"),
    ("num_bc_tl", "银行卡账户数", "银行卡账户数量", "账户行为"),
    ("num_bc_sats", "满意银行卡账户数", "状态为满意的银行卡账户数", "账户行为"),
    ("num_op_rev_tl", "开放循环账户数", "开放的循环信用账户数", "账户行为"),
    ("num_actv_bc_tl", "活跃银行卡账户数", "当前活跃的银行卡账户数", "账户行为"),
    ("num_rev_tl_bal_gt_0", "有余额循环账户数", "余额大于 0 的循环信用账户数", "账户行为"),
    # ---------------- 逾期与负面记录 ----------------
    ("mths_since_recent_revol_delinq", "距最近循环逾期月数", "距离最近一次循环信用逾期的月数；缺失=无", "逾期记录"),
    ("mths_since_recent_bc_dlq", "距最近银行卡逾期月数", "距离最近一次银行卡逾期的月数；缺失=无", "逾期记录"),
    ("mths_since_last_major_derog", "距最近重大负面月数", "距离最近一次 90+ 天负面记录的月数；缺失=无", "逾期记录"),
    ("chargeoff_within_12_mths", "近12个月核销数", "过去 12 个月的核销账户数", "逾期记录"),
    ("collections_12_mths_ex_med", "近12个月催收数", "过去 12 个月除医疗外的催收账户数", "逾期记录"),
    ("acc_now_delinq", "当前逾期账户数", "当前处于逾期状态的账户数", "逾期记录"),
    ("delinq_amnt", "当前逾期金额", "当前逾期账户的欠款金额", "逾期记录"),
    # ---------------- 衍生变量 ----------------
    ("credit_hist_months", "信用历史长度(月)", "issue_d − earliest_cr_line，单位月", "衍生变量"),
    ("loan_income_ratio", "贷款收入比", "loan_amnt / annual_inc", "衍生变量"),
    ("revol_bal_income_ratio", "循环余额收入比", "revol_bal / annual_inc", "衍生变量"),
    ("total_bal_income_ratio", "总余额收入比", "total_bal_ex_mort / annual_inc", "衍生变量"),
    ("open_acc_ratio", "开放账户占比", "open_acc / total_acc", "衍生变量"),
]

FEATURE_SPECS: tuple[FeatureSpec, ...] = tuple(
    FeatureSpec(*row) for row in _SPECS
)
FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURE_SPECS)
FEATURE_BY_NAME: dict[str, FeatureSpec] = {f.name: f for f in FEATURE_SPECS}
CN_NAMES: dict[str, str] = {f.name: f.name_cn for f in FEATURE_SPECS}
DESCRIPTIONS: dict[str, str] = {f.name: f.description for f in FEATURE_SPECS}
FEATURE_GROUPS: dict[str, str] = {f.name: f.group for f in FEATURE_SPECS}

# 这些字段的缺失本身有业务含义（无逾期/无记录），用 999 作为哨兵值
SENTINEL_FEATURES: tuple[str, ...] = (
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    "mths_since_recent_revol_delinq",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "term",
    "home_ownership",
    "verification_status",
    "purpose",
    "initial_list_status",
    "application_type",
)

ENGINEERED_FEATURES: tuple[str, ...] = (
    "credit_hist_months",
    "loan_income_ratio",
    "revol_bal_income_ratio",
    "total_bal_income_ratio",
    "open_acc_ratio",
)

# 原始 CSV 中需要用到的列（另加 issue_d / loan_status / earliest_cr_line）
# 敏感性对照专用列：读入但**绝不进入主模型特征矩阵**。
# 主模型不用 FICO 是为了与上一版口径一致，但这个取舍必须有代价量化——否则读者
# 无法判断 OOT AUC 0.66 究竟是方法问题还是特征集天花板问题。
# grade 同理：旧版注释写"仅用于敏感性对照"却从未真的做对照，属于未兑现的承诺。
SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "fico_range_low",
    "fico_range_high",
)

RAW_COLUMNS: tuple[str, ...] = (
    "issue_d",
    "loan_status",
    "earliest_cr_line",
    *[f for f in FEATURE_NAMES if f not in ENGINEERED_FEATURES],
    *CATEGORICAL_FEATURES,
    "grade",  # 敏感性对照与分段校准用，不进入主模型
    *SENSITIVITY_COLUMNS,  # 敏感性对照用，不进入主模型
    # last_pymnt_d 仅用于计算 MOB（表现成熟度），由 Preprocessor 保证不入特征矩阵
    *MOB_DATE_COLUMNS,
)

# 去掉重复列（emp_length 同时在 FEATURE_SPECS 和 CATEGORICAL_FEATURES 中）
RAW_COLUMNS = tuple(dict.fromkeys(RAW_COLUMNS))

# 泄露变量排查清单（报告直接展示）
LEAKAGE_CHECKLIST: tuple[tuple[str, str, str, str], ...] = (
    ("loan_status", "目标标签", "直接等于/决定目标变量", "仅用于构造 bad_flag，绝不入模"),
    ("out_prncp / out_prncp_inv", "贷后现金流", "放款后剩余本金，只有未来才知道", "剔除"),
    ("total_pymnt / total_pymnt_inv", "贷后现金流", "累计还款额，包含未来还款", "剔除"),
    ("total_rec_prncp / total_rec_int / total_rec_late_fee", "贷后现金流", "已回收本金/利息/滞纳金", "剔除"),
    ("recoveries / collection_recovery_fee", "贷后现金流", "核销后回收与催收费用", "剔除"),
    (
        "last_pymnt_d / last_pymnt_amnt / next_pymnt_d",
        "贷后行为",
        "最近/下一次还款，放款后产生",
        "全部不入模。last_pymnt_d 额外用于计算 MOB（表现成熟度），只决定标签观察到第几个月，不作为特征",
    ),
    ("last_credit_pull_d / last_fico_range_*", "贷后行为", "贷后拉取记录/最新 FICO", "剔除"),
    ("fico_range_low / fico_range_high", "申请时可得（非泄漏）", "申请时点的 FICO，不是泄露变量；本版为保持与上一版特征口径一致未入模", "可选，未使用"),
    ("hardship_* / deferral_term / payment_plan_start_date", "贷后救助", "困难计划在放款后发生", "剔除"),
    ("settlement_* / debt_settlement_*", "贷后处置", "债务和解/减免在违约后发生", "剔除"),
    ("grade / sub_grade", "平台评级", "LendingClub 自身风险评级，复用会掩盖自建模型能力", "主模型剔除，仅可作敏感性对照"),
    ("int_rate", "平台定价", "由 LC 评级与期限决定，属于平台风险判断结果", "剔除"),
    ("installment", "平台定价派生", "由贷款金额、期限、利率计算，间接含定价信息", "剔除"),
    ("issue_d", "时间字段", "用于时间切分；若作为特征会把未来时间信息带入", "只做切分，不入模"),
    ("earliest_cr_line", "时间字段", "原始日期不能直接入模；应转为放款时点的信用历史长度", "仅派生 credit_hist_months"),
    ("id / member_id / url", "标识符", "与违约无业务关系，可能造成记忆/索引泄漏", "剔除"),
    ("desc / title / emp_title / zip_code", "文本/高基数字段", "高基数、自由文本，易过拟合且难以部署", "剔除"),
    ("funded_amnt / funded_amnt_inv", "放款结果", "审批/撮合后的实际放款额，晚于申请", "剔除，使用 loan_amnt"),
    ("annual_inc_joint / dti_joint / sec_app_*", "联名/副申请人", "仅联名申请有值，缺失机制特殊且覆盖不足", "剔除"),
    ("il_util / all_util / open_rv_12m / inq_last_12m 等", "未来才有的字段", "2016 才补齐，训练期整列缺失，构成时间泄漏", "剔除"),
    ("policy_code / disbursement_method", "行政字段", "产品/放款方式，非风险变量且可能随时间变化", "剔除"),
)

GRADE_ORDER: tuple[str, ...] = ("A", "B", "C", "D", "E")
GRADE_LABELS: dict[str, str] = {
    "A": "A（低风险）",
    "B": "B（较低风险）",
    "C": "C（中等风险）",
    "D": "D（较高风险）",
    "E": "E（高风险）",
}

GRADE_PD_CUTOFFS: tuple[float, ...] = (0.08, 0.12, 0.18, 0.25)


@dataclass
class PipelineConfig:
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    raw_file: str = "accepted_2007_to_2018Q4.csv.gz"

    sample_start: str = "2012-01-01"
    sample_end: str = "2018-12-31"
    train_start: str = "2013-01-01"
    train_end: str = "2015-12-31"
    valid_start: str = "2016-01-01"
    valid_end: str = "2016-12-31"
    oot_start: str = "2017-01-01"
    # OOT 上界 = 快照日 − PERFORMANCE_MOB。2018Q2 及以后放款观察期不足 12 个月，
    # 在 MOB 对齐口径下无法评估，必须排除；旧版写 2018-12-31 会引入未成熟样本。
    oot_end: str = "2018-03-31"

    # MOB 对齐口径参数
    snapshot_date: str = SNAPSHOT_DATE
    performance_mob: int = PERFORMANCE_MOB

    random_state: int = 42
    n_jobs: int = 2

    max_bins: int = 6
    min_bin_pct: float = 0.02
    iv_threshold: float = 0.02
    max_features: int = 15
    corr_threshold: float = 0.75
    p_value_threshold: float = 0.05

    base_score: int = 600
    pdo: int = 20
    base_odds: float = 4.0  # good:bad = 4:1，对应 PD=20%
    grade_pd_cutoffs: tuple[float, ...] = GRADE_PD_CUTOFFS

    cap_lower_q: float = 0.001
    cap_upper_q: float = 0.999
    missing_flag_threshold: float = 0.005

    bootstrap_rounds: int = 200
    # 坏账率目标一律按**终身口径**解释（"这批贷款最终核销比例"）。
    # 若在 MOB12 vintage 口径下解释，15% 的约束会形同虚设——该口径整体坏账率只有
    # 6~7%，几乎所有阈值都自动达标。
    threshold_target_bad_rate: float = 0.15
    threshold_min_approval: float = 0.50
    # 阈值敏感性：多档目标坏账率下的阈值/批准率对照，用于暴露选择规则的脆弱性
    threshold_sensitivity_targets: tuple[float, ...] = (0.12, 0.15, 0.18)
    # 阈值稳定性：对基准目标施加 ±该步长的扰动，观察阈值与批准率漂移幅度
    threshold_sensitivity_step: float = 0.02

    lgbm_n_estimators: int = 2500
    lgbm_learning_rate: float = 0.03
    lgbm_num_leaves: int = 31
    lgbm_min_child_samples: int = 50
    lgbm_subsample: float = 0.80
    lgbm_colsample_bytree: float = 0.80
    lgbm_reg_alpha: float = 0.1
    lgbm_reg_lambda: float = 1.0
    lgbm_early_stopping: int = 100

    # 监控阈值
    psi_green: float = 0.10
    psi_amber: float = 0.25
    # 单期 PSI 的最小样本量。低于此值时 PSI 主要由抽样噪声驱动，不可用于预警。
    psi_min_period_n: int = 20_000
    auc_drop_amber: float = 0.02
    auc_drop_red: float = 0.03
    ks_drop_amber: float = 0.03
    ks_drop_red: float = 0.05
    bad_rate_rel_amber: float = 0.15
    bad_rate_rel_red: float = 0.25

    make_report: bool = True

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2.0)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    @property
    def grade_cutoffs(self) -> dict[str, int]:
        cuts = {}
        for grade, pd_cut in zip(GRADE_ORDER[:-1], self.grade_pd_cutoffs):
            logit = math.log(pd_cut / (1.0 - pd_cut))
            cuts[grade] = int(round(self.offset - self.factor * logit))
        return cuts

    def ensure_dirs(self) -> None:
        for sub in ("figures", "tables", "models", "predictions", "data"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
