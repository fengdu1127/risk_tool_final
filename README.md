# 风控自动化分析平台

## 目录结构

```
risk_tool/
├── pipeline.py              # 主入口（端到端训练 + 分析）
├── score.py                 # 打分入口（用训练好的模型+策略给新数据做决策）
├── requirements.txt
├── config/
│   └── config.py            # 全局配置（阈值、参数等）
├── data/
│   ├── generate_sample.py   # 生成测试数据
│   └── sample.csv           # 测试数据
├── modules/
│   ├── eda/
│   │   └── auto_eda.py      # 模块一：自动化数据分析
│   ├── model/
│   │   └── auto_model.py    # 模块二：模型开发（LR/XGBoost）
│   └── strategy/
│       └── auto_strategy.py # 模块三：策略开发引擎
├── utils/
│   └── helpers.py           # 公共工具函数
└── reports/                 # 输出目录（自动生成）
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 生成示例数据（首次运行）

```bash
python data/generate_sample.py   # 固定随机种子，生成 data/sample.csv
```

### 运行完整流程

```bash
# 使用 LR + XGBoost 双模型（含超参调优）
python pipeline.py --data data/sample.csv --label is_overdue --algo both

# 仅使用 XGBoost，跳过调优
python pipeline.py --data data/sample.csv --label is_overdue --algo xgboost --no-tune

# 仅使用 LR
python pipeline.py --data data/sample.csv --label is_overdue --algo lr

# 指定输出目录
python pipeline.py --data data/sample.csv --label is_overdue --algo both --output reports/my_run

# 用时间列做近期验证集切分（推荐，检测真实时序漂移）
python pipeline.py --data data/sample.csv --label is_overdue --time-col apply_time

# 用外部 JSON 覆盖配置（无需改代码）
python pipeline.py --data data/sample.csv --label is_overdue --config my_config.json
```

### 给新数据打分（部署）

```bash
python score.py --run-dir reports/my_run --data new_applications.csv --output scored.csv
```

输出每行的 `model_score`、`decision`（reject / review / approve）、命中的规则和使用的阈值。
打分使用训练时保存的 WOE 编码器（缺失值独立分箱）、最优模型（XGB 优先加载原生
`model_xgb.json`）和 `strategy/policy.json` 中的策略（全局阈值 + 分客群阈值 + 稳定拒绝规则）。

打分时自动做**输入漂移检查**：新批次每个特征相对训练集的 PSI 和缺失率变化，
PSI ≥ 0.25 或缺失率漂移 > 10pp 会在日志中告警，明细写入 `<output>_drift.csv`。

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 输入数据路径（CSV） | 必填 |
| `--label` | 标签列名（0/1） | 必填 |
| `--algo` | 算法：`lr` / `xgboost` / `both` | `both` |
| `--no-tune` | 跳过超参调优 | False |
| `--features` | 指定特征列（空格分隔） | 全部列 |
| `--output` | 输出目录 | 自动生成 |
| `--time-col` | 申请时间列（启用近期验证集切分） | 无 |
| `--valid-months` | 最近 N 个月作为验证集 | 3 |
| `--config` | JSON 配置覆盖文件 | 无 |

## 数据集划分

| 集合 | 比例 | 用途 |
|------|------|------|
| 训练集 | 70% | 模型训练 + 规则挖掘 |
| 测试集 | 15% | 模型评估 + 规则初步验证 |
| 泛化验证集（Holdout） | 15% | **封存**，规则确定后一次性最终验证 |

## 策略约束（config/config.py 可调整）

| 约束 | 默认值 | 说明 |
|------|--------|------|
| 高相关性剔除阈值 | 0.7 | Spearman 相关系数 |
| VIF 剔除阈值 | 10 | 方差膨胀因子 |
| WOE 单调性阈值 | 0.6 | Spearman 绝对值 |
| 缺失独立分箱最小样本 | 50 | 低于则缺失取中性 WOE 0 |
| XGB 单调性约束 | 开启 | 方向由特征-标签 Spearman 自动推导 |
| PSI 稳定性上限 | 0.1 | 超过则剔除 |
| 单规则覆盖率上限 | 5% | 超过不输出 |
| 单规则 Lift 下限 | 2.0 | 低于不输出 |
| 决策树最大深度 | 3 | 控制变量数 ≤ 3 |

## 输出文件

每次运行生成 `reports/run_YYYYMMDD_HHMMSS/` 目录，包含：

```
run_xxx/
├── run_summary.json           # 运行汇总（模型指标、规则数、生效配置快照）
├── dashboard.html             # 可视化报告
├── split_train/test/valid.csv
├── eda/
│   ├── iv_ranking.png         # IV 排名图
│   ├── corr_heatmap.png       # 相关性热图
│   └── woe_trends.png         # WOE 趋势图
├── model/
│   ├── model_lr.pkl           # LR 模型
│   ├── model_xgb.pkl / .json  # XGBoost 模型（pickle + 原生 JSON）
│   ├── model_meta.json        # 打分元数据（特征列表、最优算法、依赖版本）
│   ├── woe_encoder.pkl        # WOE 编码器（含训练集中位数）
│   ├── scorecard_lr.csv       # 标准评分卡
│   ├── model_comparison.csv   # 模型对比指标
│   ├── xgb_shap.png           # SHAP 特征解释图
│   ├── *_roc_ks_*.png         # ROC/KS 曲线
│   ├── *_lift_*.png           # Lift 曲线
│   └── scored_*.csv           # 模型评分（默认仅 label/score/bin，可配置输出全量）
└── strategy/
    ├── policy.json            # 机器可读策略（score.py 直接消费）
    ├── variable_selection.csv # 变量筛选报告
    ├── rules_all.csv          # 全部规则清单
    ├── backtest_results.csv   # 三集回测结果
    ├── decision_tree.png      # 决策树可视化
    ├── rule_comparison.png    # 规则跨集对比图
    └── coverage_lift_scatter.png  # 覆盖率-Lift 散点图
```

## 调整配置

推荐用 `--config` 传 JSON 覆盖文件，无需修改代码：

```json
{
  "STRATEGY_CONFIG": {"monotone_spearman_min": 0.4, "rule_max_coverage": 0.03},
  "MODEL_CONFIG": {"optuna_trials": 20},
  "SPLIT_CONFIG": {"train_ratio": 0.60, "holdout_ratio": 0.25}
}
```

也可以直接编辑 `config/config.py`。每次运行生效的完整配置会快照到 `run_summary.json` 的 `config` 字段，保证可复现。
