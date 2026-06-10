# 风控自动化分析平台 — 代码框架

## 1. 项目目录结构

```
risk_tool_final/
├── pipeline.py                 # 主入口，编排全流程
├── requirements.txt            # 依赖
├── ARCHITECTURE.md             # 本文件
├── config/
│   ├── __init__.py
│   └── config.py               # 全局配置（4个字典控制所有阈值）
├── data/
│   ├── generate_sample.py      # 生成模拟数据
│   └── sample.csv
├── modules/
│   ├── __init__.py
│   ├── eda/
│   │   ├── __init__.py
│   │   └── auto_eda.py         # 模块一：探索性数据分析
│   ├── model/
│   │   ├── __init__.py
│   │   └── auto_model.py       # 模块二：模型训练（LR + XGBoost）
│   └── strategy/
│       ├── __init__.py
│       └── auto_strategy.py    # 模块三：策略规则挖掘
├── utils/
│   ├── __init__.py
│   └── helpers.py              # 公共工具（划分/指标/持久化/校验）
└── tests/
    ├── __init__.py
    ├── test_eda.py
    ├── test_helpers.py
    ├── test_model.py
    └── test_strategy.py
```

## 2. 数据流

```
CSV 数据
  │
  ├─ pipeline.run_pipeline()
  │
  ├─ [0] 数据加载 & 校验 ── validate_inputs()
  ├─ [1] 数据集划分 ── split_dataset() → train(70%) / test(15%) / holdout(15%)
  │
  ├─ [2] 模块一: AutoEDA.run()
  │        ├─ 数据质量检测
  │        ├─ WOE/IV 计算 (calc_woe_iv)
  │        ├─ WOE 单调性检验 (check_monotonicity)
  │        ├─ 相关性矩阵 & 高相关检出
  │        ├─ VIF 多重共线性
  │        ├─ PSI 稳定性
  │        └─ 综合评分汇总 (_feature_summary)
  │        │
  │        └──→ iv_table ──────────────────────────┐
  │                                                 │
  ├─ [3] 模块二: AutoModel.run()                    │
  │        ├─ WOEEncoder (LR用)                     │
  │        ├─ _train_lr() → LogisticRegression      │
  │        │   ├─ Optuna 超参调优                    │
  │        │   ├─ ROC/KS/Lift 图                     │
  │        │   └─ 评分卡 (build_scorecard)           │
  │        ├─ _train_xgboost() → XGBClassifier       │
  │        │   ├─ Optuna 超参调优                    │
  │        │   ├─ ROC/KS/Lift 图                     │
  │        │   ├─ SHAP 特征重要性                    │
  │        │   └─ XGBoost 特征重要性                 │
  │        ├─ 模型对比 → 选最优 (按 test KS)         │
  │        └─ _score_all() → 写回 model_score 列     │
  │        │                                          │
  │        └──→ scored_datasets ──────────────────┐  │
  │                                                │  │
  ├─ [4] 模块三: AutoStrategy.run()                │  │
  │        ├─ VariableSelector.fit()  ←───────────┘  │
  │        │   ├─ 高相关剔除 (数值型)                 │
  │        │   ├─ WOE单调性筛选 (数值型)              │
  │        │   ├─ PSI稳定性筛选 (数值型)              │
  │        │   └─ 类别型: IV基础检查                  │
  │        ├─ SingleRuleMiner.mine()                  │
  │        │   ├─ 数值型: >= / <= 枚举切点            │
  │        │   └─ 类别型: == 枚举类别值               │
  │        ├─ TreeRuleMiner.mine()                    │
  │        │   └─ DecisionTreeClassifier 叶节点规则    │
  │        └─ StrategyBacktest.run() × 3              │
  │           └─ 训练集/测试集/保留集 回测验证        │
  │
  └─ [5] 汇总 → run_summary.json
```

## 3. 模块一: auto_eda.py — 探索性数据分析

| 函数/类 | 作用 |
|---|---|
| `calc_woe_iv()` | 单变量 WOE/IV 计算，支持 quantile/uniform/tree 三种分箱，自动识别类别型特征 |
| `classify_iv()` | IV 值分级：无预测力/弱/中/强/极强 |
| `check_monotonicity()` | WOE 单调性检验，返回 Spearman 相关系数 |
| **`AutoEDA`** | 主类，`.run()` 一键执行全量 EDA |
| `._data_quality()` | 缺失率、数据类型、唯一值统计 |
| `._iv_analysis()` | 遍历所有特征计算 IV，返回排名表 |
| `._monotonicity()` | 对所有 WOE 表做单调性检验 |
| `._correlation()` | Spearman 相关矩阵，检出高相关对 (默认 >0.7) |
| `._vif()` | 方差膨胀因子（仅数值型） |
| `._psi_internal()` | 随机二分计算 PSI 稳定性 |
| `._feature_summary()` | 综合多维度给出 OK/WARN 建议 |
| 可视化 | `_plot_iv_bar` / `_plot_corr_heatmap` / `_plot_woe_trends` |

## 4. 模块二: auto_model.py — 模型开发

| 函数/类 | 作用 |
|---|---|
| **`WOEEncoder`** | 特征编码器，数值型用决策树分箱→WOE，类别型直接计算每类 WOE |
| `.fit()` / `.transform()` | 训练/应用 WOE 映射，自动区分数值/类别 |
| `build_scorecard()` | LR 系数 + WOE → 标准评分卡 (pdo=20, base=600, odds=1/15) |
| `plot_roc_ks()` | 双图：ROC 曲线 + KS 曲线 |
| `plot_lift_curve()` | 双图：分箱 Bad Rate + Lift 曲线 |
| `plot_shap_summary()` | SHAP 摘要图 + CSV 输出（兼容 XGBoost 3.x） |
| **`AutoModel`** | 主类，`.run()` 一键训练和评估 |
| `._train_lr()` | WOE编码 → StandardScaler → LogisticRegression → ROC/KS/Lift 评估 |
| `._train_xgboost()` | 仅数值列 → XGBClassifier (自动 scale_pos_weight) → 评估 + SHAP |
| `._tune_lr()` / `._tune_xgb()` | Optuna 贝叶斯超参搜索 (CV AUC) |
| `._score_all()` | 最优模型评分写回各数据集 (model_score + model_score_bin) |

## 5. 模块三: auto_strategy.py — 策略引擎

| 类 | 作用 |
|---|---|
| **`VariableSelector`** | 三步预筛选：高相关 → 单调性 → PSI，数值型全流程，类别型仅 IV 检查 |
| `._drop_high_corr()` | Spearman 相关 > 阈值 → 保留 IV 高者 |
| `._filter_monotone()` | WOE Spearman < 0.6 → 剔除 |
| `._filter_stable()` | PSI > 0.1 → 剔除 |
| **`SingleRuleMiner`** | 单维规则挖掘：覆盖率 ≤ 5%，Lift ≥ 2 |
| `.mine()` | 数值型枚举百分位切点 (>= / <=)，类别型枚举值 (==) |
| **`TreeRuleMiner`** | 决策树组合规则：max_depth=3, max_features=3 |
| `.mine()` | 仅数值型 → DecisionTreeClassifier → 提取叶节点条件路径 |
| `._extract_leaf_rules()` | 递归遍历树，筛选满足覆盖率/Lift 约束的叶节点 |
| `.plot_tree()` | 决策树可视化 |
| **`StrategyBacktest`** | 规则回测验证 |
| `.run()` | 逐规则计算覆盖率/逾期率/Lift，三数据集验证 |
| `.combined_effect()` | 所有规则 OR 合并的整体拒绝率 |
| **`AutoStrategy`** | 主类，`.run()` 一键执行全流程 |
| `._merge_rules()` | 合并单维 + 树规则为统一格式 |
| 可视化 | `_plot_rule_comparison` / `_plot_coverage_lift` |

## 6. 配置系统: config.py

4 个字典集中控制所有超参数：

| 字典 | 关键参数 |
|---|---|
| `SPLIT_CONFIG` | train/test/holdout 比例 70/15/15，随机种子 42 |
| `EDA_CONFIG` | missing_threshold=0.5, iv_min=0.02, corr_threshold=0.7, psi_stable=0.1 |
| `MODEL_CONFIG` | cv=5, optuna_trials=50, LR/XGB 默认参数, scorecard (pdo=20) |
| `STRATEGY_CONFIG` | corr=0.7, monotone_spearman_min=0.6, psi_max=0.1, coverage≤5%, lift≥2, tree_depth=3 |

## 7. 公共工具: helpers.py

| 函数 | 作用 |
|---|---|
| `get_logger()` | 统一日志格式 |
| `split_dataset()` | 按时间或随机分层 (stratify) 划分 |
| `calc_ks()` | KS = max(TPR - FPR) |
| `calc_auc()` | ROC AUC |
| `calc_psi()` | 群体稳定性指标 |
| `calc_lift()` | bad_rate / overall_bad_rate |
| `model_report()` | 一键输出 AUC/KS/Gini |
| `save_pickle()` / `load_pickle()` / `save_json()` | 持久化 |
| `validate_inputs()` | 输入校验：样本量/标签分布/特征存在性/零方差 |

## 8. 关键设计特点

- **数值/类别双轨**: WOE 编码和规则挖掘均自动识别特征类型，数值型走分箱/切点枚举，类别型走直接映射/值枚举
- **兼容性处理**: SHAP 使用 `shap.Explainer(model.predict, background)` 规避 XGBoost 3.x 的 `base_score` 格式问题；Windows GBK 控制台使用 ASCII 替代 emoji
- **XGBoost 仅数值输入**: 自动过滤非数值列，`scale_pos_weight` 从标签分布自动计算
- **封存验证**: holdout 集在策略阶段才首次使用，防止过拟合
- **输出目录结构**: `reports/run_{timestamp}/` → `eda/` `model/` `strategy/` 三级子目录，含 CSV + PNG 图表
