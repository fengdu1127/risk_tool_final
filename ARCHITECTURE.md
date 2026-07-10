# 风控自动化分析平台 — 项目框架

> 训练分析 → 版本对比 → 上线 → 打分决策 + 漂移监控 的完整闭环。

## 1. 目录结构

```
risk_tool_final/
├── pipeline.py                 # 训练入口：EDA → 双模型 → 策略挖掘 → 报告
├── score.py                    # 打分入口：模型 + 校准 + 策略决策 + 漂移检查
├── promote.py                  # 上线：把某次 run 写入 reports/PRODUCTION 指针
├── compare_runs.py             # 迭代：两次 run 的指标/策略 diff
├── requirements.txt            # 锁定版本（与模型产物兼容性绑定）
├── pytest.ini
├── config/
│   └── config.py               # 5 个配置字典 + JSON 覆盖 + 快照
├── data/
│   └── generate_sample.py      # 固定种子生成 data/sample.csv（不入库）
├── modules/
│   ├── eda/auto_eda.py         # 模块一：数据质量 / WOE·IV / 单调性 / 相关性 / VIF / PSI
│   ├── model/auto_model.py     # 模块二：WOE 编码 / LR+XGB / 调优 / 校准 / 评分刻度
│   ├── strategy/auto_strategy.py  # 模块三：变量筛选 / 规则挖掘 / 三集回测 / policy.json
│   └── reporting/dashboard.py  # 静态 HTML 可视化报告
├── utils/
│   ├── binning.py              # 统一分箱：决策树切点 + WOE 统计（EDA/模型共用）
│   └── helpers.py              # 划分 / 指标 / prob_to_score / 持久化 / 校验
└── tests/                      # 100 个测试（含 pipeline→score 端到端）
```

## 2. 训练链路（pipeline.py）

```
CSV ─ validate_inputs
  │
  ├─ [1] 数据划分
  │      有 time_col：最近 N 月 → valid（近期验证），历史随机分层 → train/test
  │      无 time_col：随机分层 70/15/15
  │
  ├─ [2] AutoEDA.run(train, time_col)
  │      质量 → IV/WOE（缺失独立分箱）→ 单调性（阈值走配置，排除缺失箱）
  │      → 相关性/VIF → PSI（有时间列按时间前后二分）→ 综合建议
  │      └→ iv_table ────────────────────────────┐
  │                                              │
  ├─ [3] AutoModel.run                           │
  │      WOEEncoder 一次拟合（train），LR 全特征 WOE，XGB 数值原值+类别 WOE
  │      ├─ LR：Pipeline(Scaler+LR)，Optuna CV 调优（无泄漏）
  │      ├─ XGB：单调性约束（Spearman 自动定向），调参只用 train 内部 8/2 切分
  │      ├─ 按 test KS 选最优 → isotonic 校准（test 拟合）
  │      ├─ 产物：model_*.pkl/.json、woe_encoder、calibrator、model_meta.json
  │      │        （特征列表、约束、漂移基线、依赖版本）、scorecard、calibration_report
  │      └→ scored train/test/valid（model_score + calibrated_prob + credit_score）
  │                                              │
  ├─ [4] AutoStrategy.run  ←─────────────────────┘
  │      变量筛选（相关/单调/PSI）→ 单维规则 + 决策树规则 → 重叠去重
  │      → 三集回测 → 稳定规则 → 全局/分客群阈值网格搜索 → 三档推荐
  │      └→ policy.json（score.py 直接消费的机器可读策略）
  │
  └─ [5] run_summary.json（含配置快照）→ dashboard.html → run.log
```

## 3. 部署链路（promote.py + score.py）

```
python promote.py reports/run_xxx     # 校验产物齐全 → 写 reports/PRODUCTION
python score.py --data new.csv        # 缺省读 PRODUCTION 指针

score.py 内部：
  漂移检查（新批次 vs 训练基线：逐特征 PSI + 缺失率漂移，超限告警 + *_drift.csv）
  → WOE 特征工程（训练时的切点/缺失箱） → 模型打分
  → calibrated_prob（预期坏账率） + credit_score（base 600 / PDO 20）
  → 策略决策：稳定拒绝规则 OR 分数阈值（分客群阈值优先）
  → 输出 decision（reject/review/approve）+ 命中规则 + 使用阈值
```

## 4. 模块职责

| 文件 | 核心类/函数 | 职责 |
|---|---|---|
| `utils/binning.py` | `tree_bin_thresholds` / `woe_stats` | 全项目唯一分箱实现，保证 EDA/模型/策略切点一致 |
| `modules/eda/auto_eda.py` | `AutoEDA`, `calc_woe_iv`, `check_monotonicity` | 特征体检与筛选建议；缺失单独成箱（bin=-1） |
| `modules/model/auto_model.py` | `WOEEncoder`, `AutoModel`, `build_scorecard` | 双模型训练/调优/校准/持久化/漂移基线 |
| `modules/strategy/auto_strategy.py` | `VariableSelector`, `SingleRuleMiner`, `TreeRuleMiner`, `StrategyBacktest`, `AutoStrategy` | 规则挖掘、三集稳定性验证、策略推荐与序列化 |
| `modules/reporting/dashboard.py` | `generate_dashboard` | 汇总 CSV/PNG 为单页 HTML 报告 |
| `utils/helpers.py` | `split_dataset*`, `calc_ks/auc/psi/lift`, `prob_to_score`, `validate_inputs` | 公共工具 |

## 5. 配置系统（config/config.py）

| 字典 | 管什么 |
|---|---|
| `SPLIT_CONFIG` | 划分比例、time_col、valid_months、随机种子 |
| `EDA_CONFIG` | 缺失/IV/PSI/相关性/VIF 阈值、单调性阈值、分箱数 |
| `MODEL_CONFIG` | 算法、CV、optuna 次数、缺失分箱最小样本、XGB 单调约束开关、评分卡刻度 |
| `STRATEGY_CONFIG` | 规则覆盖率/Lift/稳定性约束、分客群网格与判别过滤 |
| `REPORT_CONFIG` | 输出目录、图表参数、scored CSV 瘦身开关 |

- 运行时覆盖：`pipeline.py --config my.json`（`apply_config_overrides`，兼容 BOM）
- 可复现：每次 run 的生效配置整体快照进 `run_summary.json["config"]`

## 6. 方法论保障

- **无泄漏调参**：LR 用训练集 CV，XGB 只用 train 内部切分；test 仅用于模型选择与校准，valid 全程封存到最终验证
- **训练/上线一致**：WOE 切点、缺失箱、训练中位数全部持久化在编码器里；打分不依赖当前批次统计量
- **可解释性**：XGB 单调性约束方向自动推导并记录；LR 评分卡与 `prob_to_score` 同一刻度（base 600 / PDO 20）
- **概率可信**：isotonic 校准修正 `scale_pos_weight` 带来的概率偏移，calibration_report 留痕
- **上线监控**：训练分布基线存 `model_meta.json`，每次打分自动 PSI + 缺失率漂移检查
- **规则防过拟合**：覆盖率/Lift/命中数三集全过 + test-valid Lift 回撤约束才算稳定规则

## 7. 质量与运维

- 测试：`python -m pytest tests/`（100 个，含端到端；`slow` 标记训练类用例）
- 日志：控制台 + `<run>/run.log` 全量镜像
- 版本管理：`compare_runs.py` diff 两次 run；`promote.py` 维护生产指针
- 产物兼容：requirements 锁定版本，与 `model_meta.json` 记录的训练环境一致
