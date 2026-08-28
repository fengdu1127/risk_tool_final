# riskflow

信贷风控建模与决策工具包：从原始申请数据到一份可以直接上线的决策策略。

覆盖完整闭环 —— **切分 → 分箱 → 变量筛选 → 建模 → 概率校准 → 规则挖掘 → 阈值设定 → 打包 → 打分 → 漂移监控**。

```bash
pip install -e ".[train,report,dev]"
riskflow demo                    # 生成数据、训练、出报告、上线，一条命令
```

---

## 1. 核心设计：训练用 ML 库，上线只用 numpy

整个打分产物是**一个 JSON 文件**。没有 pickle。

```
bundle.json
├── schema        列的类型契约
├── woe           每个变量的分箱切点与 WOE 值
├── space         模型吃的矩阵怎么拼出来
├── predictor     模型本身（线性系数，或展平成数组的树集成）
├── calibrator    等渗校准曲线（插值表）
├── policy        拒绝规则 + 全局阈值 + 分客群阈值
└── drift         训练集的分箱占比基线
```

打分时 `LogisticRegression` 变成一次点积，XGBoost 变成一次向量化的树遍历。
两者都是纯 numpy 实现。这带来三个后果：

- **产物不再被库版本绑架。** 原来"requirements 必须锁死，否则 pickle 加载失败"的问题直接消失；
  一年前的 bundle 今天仍然能读。
- **打分服务不需要装 scikit-learn 或 XGBoost。** 依赖只有 numpy + pandas。
  测试里真的把这些库 import 掉，验证打分照样跑通。
- **产物可读、可 diff、可评审。** 风控策略是要给人看的，模型风险审查能直接读 JSON。

风险显然是"导出错了怎么办"。所以每个导出器都有一条**数值一致性测试**，
断言导出后的预测与原库 `predict_proba` 的最大偏差 ≤ 1e-6：

| 导出 | 最大偏差 |
|---|---|
| `LogisticRegression`（含标准化折叠进系数） | 2e-16 |
| `XGBClassifier`（含单调约束、样本加权、缺失方向） | 2e-7 |
| `HistGradientBoostingClassifier` | 2e-16 |
| `IsotonicRegression`（含越界裁剪） | 0 |

XGBoost 那 2e-7 来自它内部用 float32 累加。为了做到这个精度，导出读的是
booster 的**原生 JSON**而不是文本 dump（dump 会四舍五入切点，把落在分裂边界上的样本
悄悄分到另一边），并且遍历时把特征值 cast 成 float32 以复现它的比较语义。
`split_op` 和 `dtype` 都写进 bundle，不靠约定。

---

## 2. 哪个样本能做什么

三个样本，职责严格分离：

| | train | test | holdout |
|---|---|---|---|
| 分箱 / WOE | ✅ | | |
| 变量筛选 | ✅ | 只做稳定性检验 | |
| 超参搜索 | ✅（内部再切一刀） | | |
| 模型选择 | | ✅ | |
| 概率校准 | ✅（out-of-fold） | | |
| 规则挖掘 | ✅ | 过滤 | 否决 |
| 阈值选择 | 阈值取自其分位 | ✅ | 否决 |
| 最终验证 | | | ✅ |

两条规矩：

**调参和校准都不碰 test。** 超参搜索在 train 内部再切一刀评估；等渗校准拟合在
train 的 **out-of-fold 预测**上。原来常见的做法是"在 test 上选模型、又在 test 上校准"，
两次消费同一份样本，乐观偏差会叠加。

**holdout 只能否决，不能选择。** 规则和阈值都在 test 上选定，holdout 只用来
把"只在一个样本上成立"的东西打掉。这样最终验证仍然诚实，同时又不会把一条
没复现过的规则送上线。这是一个有意识的取舍，写在这里而不是藏在代码里。

---

## 3. 一次 demo 跑出来是什么样

12000 条合成申请，其中埋了：单调风险变量、两个纯噪声列、非随机缺失、
一个只有两个条件同时成立才出现的高风险口袋，以及随时间的轻微人群漂移。

```
features kept    11 of 15          ← 两个噪声列被 IV 关卡拦掉
model            gbdt (xgboost)    ← 按 test KS 选出
stable rules     1 of 13 mined
overfit verdict  acceptable

model dataset  rows  bad_rate    auc      ks
 gbdt   train  8191    0.1144  0.804   0.450
 gbdt    test  1756    0.1145  0.791   0.477
 gbdt holdout  2053    0.1330  0.747   0.378

在封存的 holdout 上：拒绝 11.2% 的申请，抓住 32.2% 的坏账，通过件坏账率 7.41%
```

**规则挖掘准确复原了埋进去的口袋**，13 条候选里只有它活下来：

```
description                                        lift_train  lift_test  lift_holdout  stable  verdict
max_delinquency > 1.5 AND utilization > 0.7951           7.61       6.72          6.77    True   stable
max_delinq > 1.5 AND debt_ratio <= 0.38 AND util > 0.77  5.77       5.82          5.01   False   lift decays 0.81 from test to holdout
debt_ratio > 0.36 AND inquiries_6m > 2.5 AND dr > 0.55   4.08       3.92          2.95   False   lift decays 0.97 from test to holdout
utilization >= 0.821197                                  2.73       4.16          2.37   False   lift decays 1.79 from test to holdout
income <= 2964.38                                        2.62       1.46          4.51   False   lift 1.46 on test (need 2.0)
loan_amount <= 1000                                      2.50       0.00          3.76   False   only 2 hit(s) on test (need 5)
```

每条被淘汰的规则都带一句人话理由。校准的效果同样明显 —— `scale_pos_weight`
让原始概率整体偏高，校准后逐档误差从 0.15–0.46 降到 0.003–0.03。

---

## 4. 用法

```bash
# 训练
riskflow train --data applications.csv --label is_bad \
               --time-col apply_time --id-col application_id --config my.json

# 上线 / 打分
riskflow promote run_20260828_141154
riskflow score --data new_batch.csv --output scored.csv

# 迭代与排查
riskflow runs                                   # 列出所有 run，标出生产版本
riskflow compare run_A run_B --sample batch.csv # 指标 diff + 决策翻转率
riskflow explain --data batch.csv --row 42      # 单个申请人为什么是这个结论
```

作为库使用：

```python
from riskflow import train, ScoringBundle, Settings

result = train(data=df, label="is_bad", settings=Settings().merged(
    {"split": {"time_col": "apply_time"},
     "cutoffs": {"segment_features": ["channel", "city_tier"]}}
))

bundle = ScoringBundle.load(result.run.bundle_path)
decisions = bundle.score(new_applications)
```

`explain` 的输出：

```
Decision: APPROVE  (score below the review threshold)
Model score 0.1867 | expected bad rate 3.97% | credit score 564
Thresholds in force: decline at 0.7191, review at 0.5522

              feature     value                bin  bin_bad_rate       woe
               income  39560.76     (23986.9, inf]      0.0678   -0.5737
      max_delinquency         2         (1.5, 2.5]      0.1579    0.3726
           debt_ratio    0.2495 (0.18275, 0.31195]      0.0843   -0.3389
```

---

## 5. 每一层在做什么

```
riskflow/
├── settings.py          冻结的 dataclass 配置树，函数式覆盖，逐 run 快照
├── data/
│   ├── schema.py        列的类型契约，训练时确定、打分时复核
│   ├── splitting.py     随机分层 / 时间外切分
│   └── synth.py         合成数据（结构是刻意埋的，用于 demo 和测试）
├── features/
│   ├── binning.py       监督分箱 + 单调性合并 + WOE（全项目唯一实现）
│   ├── woe.py           每变量一个分箱，整体可 JSON 化
│   ├── diagnostics.py   IV / 单调性 / PSI / 相关性 / VIF → 筛选漏斗
│   └── space.py         模型输入矩阵的配方（训练与上线共用同一份）
├── models/
│   ├── predictors.py    纯 numpy 推理：LinearScorer / TreeEnsemble / IsotonicCurve
│   ├── export.py        sklearn·XGBoost → 上述结构（有一致性测试兜底）
│   ├── training.py      随机搜索 / 模型选择 / OOF 校准 / 单调约束
│   ├── scorecard.py     分数刻度与评分卡，与模型对账
│   └── metrics.py       KS / AUC / Gini / Lift / 增益表（numpy 实现）
├── policy/
│   ├── predicates.py    结构化规则条件（不是字符串）
│   ├── mining.py        单变量 + 多棵浅树的规则挖掘
│   ├── validation.py    三样本回测与稳定性关卡
│   ├── thresholds.py    全局与分客群阈值搜索
│   └── decision.py      DecisionPolicy：申请人最终怎么处理
├── monitoring/drift.py  按模型自身分箱算的 PSI 基线
├── bundle.py            打分产物；bundle.score() 是全项目唯一的打分路径
├── registry.py          run 目录与 PRODUCTION 指针
├── train.py             编排
└── reporting/           单文件自包含 HTML 报告
```

---

## 6. 几个具体的设计选择

**WOE 方向统一为"越高越危险"**（`ln(坏账占比 / 好账占比)`）。
于是模型系数符号、GBDT 单调约束方向、评分卡扣分方向全部自洽，
不需要在某一层插一个负号来纠正。空箱用 0.5 连续性修正，不会出现无穷大的 WOE。

**分箱会做单调性合并，不只是检测。** 相邻箱按最小违反程度依次合并，
直到坏账率单调。原来的做法是检测到不单调就把变量丢掉 —— 但很多变量只是
某一段有噪声，合并两个箱就能救回来。

**规则条件是结构化对象，不是字符串。** `Predicate(feature, op, value)` 序列化成 JSON 对象。
把 `"debt_ratio > 0.55"` 存成字符串再用 `split()` 解析，遇到列名带空格、带比较符号
或带中文时就会静默出错。测试里专门有一个叫 `odd name [x]` 的列来守这一点。

**缺失值永远不满足比较。** "utilization 超过 0.9 就拒绝"不应该命中一个
utilization 未知的人。要针对缺失，得显式写 `is_null` 谓词。

**候选阈值按目标覆盖率布点，不是均匀分位。** 一条规则的覆盖率上限是 5%，
那均匀 5% 分位网格的最细粒度恰好等于上限，窄规则根本生不出来。
这里按 `min_hits/n → max_coverage` 的几何网格布点，分辨率落在真正有用的尾部。

**树规则挖掘用多棵树。** 单棵树只会沿着"当前看起来最好"的分裂往下走，
把它的次优竞争者所参与的交叉全埋掉了。这里拟合若干棵随机特征子集的浅树，
把候选池扩开，让后面的关卡去筛。

**训练结束后会用保存的 bundle 重新打分一遍**，逐样本比对内存中的模型，
偏差超过 1e-6 直接让这次 run 失败。报告里的数字来自将来上线的那条代码路径，
而不是训练时的内存对象。

---

## 7. 配置

全部是 frozen dataclass，函数式覆盖，未知键会报错并列出合法键：

```bash
riskflow train --data d.csv --label is_bad --config my.json
```

```json
{
  "split":    {"time_col": "apply_time", "oot_months": 3},
  "binning":  {"max_bins": 6, "enforce_monotonic": true},
  "screening":{"min_iv": 0.03, "max_psi": 0.08},
  "model":    {"algorithms": ["logistic"], "search_iterations": 60},
  "rules":    {"max_coverage": 0.03, "min_lift": 3.0},
  "cutoffs":  {"segment_features": ["channel", "city_tier"], "reject_rate_grid": [0.03, 0.05]}
}
```

生效的完整配置会快照进 `<run>/settings.json`。

---

## 8. 测试

```bash
pytest                  # 117 个，约 5 秒
pytest -m "not slow"    # 跳过端到端
```

值得单独一提的几条：

- `test_predictors.py` —— 导出一致性，本项目所有安全性的地基
- `test_end_to_end.py::test_scoring_needs_no_training_libraries` —— 把 sklearn / xgboost / scipy
  的 import 全部拦掉，验证 bundle 照样打分
- `test_training.py::test_a_monotone_gbdt_never_scores_a_worse_applicant_better` ——
  只抬高逾期次数，风险估计不允许下降
- `test_policy.py::test_a_rule_that_only_works_on_train_is_rejected` —— 稳定性关卡真的会拦人
- `test_screening_and_drift.py::test_binned_psi_uses_the_model_s_own_bins` ——
  一个在自身分位上完全稳定的批次，在模型分箱上是明显漂移的

---

## 9. 已知边界

- 只支持二分类 0/1 标签。
- 树集成导出不支持 XGBoost 的原生类别型分裂（类别变量走 WOE 编码），
  遇到会明确报错而不是给出错误结果。
- 阈值搜索的目标函数是"在 lift 仍达标的前提下尽量放宽拒绝面"，
  这是一个可配置的代理指标，不是利润最优化 —— 真要做后者需要接入
  收益和损失金额，目前不在范围内。
- `bundle.json` 里嵌了树集成的完整结构，几百棵树时会到几 MB 量级。
