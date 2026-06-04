# 金融机器学习：A股月度收益方向预测

本项目基于公司特征与宏观指标，使用 Random Forest 和 XGBoost 预测 A 股个股下一月收益率方向。

## 数据清洗

```powershell
python clean_data.py --stock CHN_sample_data.csv --macro CHN_Marco_predictors.csv --out data/cleaned_panel.csv
```

清洗脚本会：

- 将 `Dates` 统一为 `YYYYMM` 整数；
- 按显式日历下一月构造 `y_next`，避免股票断档时标签错配；
- 使用滞后一月宏观变量；
- 默认剔除 `y_next == 0`；
- 默认加入 1 个月 embargo，避免切分边界泄漏；
- 默认按月横截面中位数处理特征缺失。

## 模型训练

```powershell
python train_model.py --data data/cleaned_panel.csv --out-dir outputs
```

训练脚本会：

- 训练 Random Forest 和 XGBoost；
- 使用验证集选择模型和分类阈值；
- 对 XGBoost 使用训练集内部 holdout 做早停；
- 输出分类指标、模拟期预测、特征重要性和 Top/Bottom 10% 组合回测结果。

## 说明

`CHN_sample_data.csv` 与 `data/cleaned_panel.csv` 文件较大，未纳入 Git。运行脚本前请将原始个股面板数据放在项目根目录。
