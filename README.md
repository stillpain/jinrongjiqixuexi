# 金融机器学习：A股月度收益方向预测

本项目基于公司特征和宏观指标，使用 Random Forest 与 XGBoost 对 A 股个股下一月收益率方向进行样本外预测。

## 环境依赖

建议使用 Python 3.10 及以上版本。运行前安装依赖：

```powershell
pip install pandas scikit-learn xgboost
```

如果需要使用项目中生成的 CSV 结果继续分析，也可以安装：

```powershell
pip install numpy
```

说明：`numpy` 通常会随 `pandas`、`scikit-learn` 或 `xgboost` 自动安装。

## 数据文件

CSV 数据文件不上传到 GitHub。运行脚本前，请将以下数据放到项目根目录：

- `CHN_sample_data.csv`：A 股个股月度面板数据。
- `CHN_Marco_predictors.csv`：月度宏观预测因子，包含 `Vol`、`GDPgrowth`、`CPIgrowth`。

`CHN_Macro_sample.csv` 是宏观原始变量表，当前第一版训练脚本不直接使用。

## `clean_data.py`

功能：清洗原始数据，构造下一月涨跌标签，合并滞后一月宏观变量，并生成训练集、验证集和模拟集。

示例：

```powershell
python clean_data.py --stock CHN_sample_data.csv --macro CHN_Marco_predictors.csv --out data/cleaned_panel.csv
```

主要处理：

- 将 `Dates` 统一为 `YYYYMM` 整数。
- 按显式日历下一月构造 `y_next`，避免个股停牌、退市或新上市导致的断档错配。
- 使用滞后一月宏观变量，降低信息泄漏风险。
- 默认剔除 `y_next == 0` 的样本。
- 默认加入 1 个月 embargo：
  - 训练集：到 `201511`
  - 验证集：`201601-201711`
  - 模拟集：`201801-201911`，实际会因 `y_next` 保留到 `201910`
- 默认使用按月横截面中位数处理特征缺失。

可选参数示例：

```powershell
python clean_data.py --missing-strategy drop
python clean_data.py --keep-zero-return
```

## `train_model.py`

功能：读取清洗后的面板数据，训练 Random Forest 和 XGBoost，输出分类指标、样本外预测、特征重要性和简单组合回测结果。

示例：

```powershell
python train_model.py --data data/cleaned_panel.csv --out-dir outputs
```

主要输出：

- `outputs/metrics.csv`：验证集与模拟集分类指标。
- `outputs/sim_predictions.csv`：模拟期每只股票的预测概率和预测标签。
- `outputs/feature_importance_rf.csv`：Random Forest 特征重要性。
- `outputs/feature_importance_xgb.csv`：XGBoost 特征重要性。
- `outputs/portfolio_returns.csv`：Top/Bottom 10% 月度组合收益。
- `outputs/portfolio_summary.csv`：组合累计收益、年化收益、Sharpe 和最大回撤。

模型设计：

- 用训练集训练模型。
- 用验证集选择候选模型和分类阈值。
- XGBoost 使用训练集内部 holdout 做 early stopping，避免重复使用验证集。
- 用模拟集作为最终样本外评估。

## 注意事项

- 当前脚本默认预测绝对涨跌方向，即 `label = 1[y_next > 0]`。
- 财务类公司特征是否为 point-in-time 可得，需要结合数据来源说明进一步核对。
- 分类指标不等同于投资价值，建议重点结合组合回测结果解释模型表现。
