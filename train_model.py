import argparse
import sys
from math import sqrt
from pathlib import Path

import pandas as pd


EXCLUDED_FEATURE_COLUMNS = {"Dates", "stkcd", "y", "y_next", "label", "split"}
BENCHMARK_FEATURES = ["size", "BM", "mom12m"]
METRIC_COLUMNS = [
    "model",
    "split",
    "accuracy",
    "balanced_accuracy",
    "auc",
    "precision",
    "recall",
    "f1",
    "threshold",
    "rows",
    "threshold_source",
]


def import_ml_dependencies():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing scikit-learn. Install dependencies with: "
            "pip install pandas scikit-learn xgboost scipy"
        ) from exc

    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing xgboost. Install dependencies with: pip install xgboost"
        ) from exc

    return {
        "RandomForestClassifier": RandomForestClassifier,
        "XGBClassifier": XGBClassifier,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Random Forest and XGBoost classifiers on the cleaned A-share "
            "panel and write diagnostics for cross-sectional stock selection."
        )
    )
    parser.add_argument("--data", default="data/cleaned_panel.csv")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--risk-free-monthly", type=float, default=0.0)
    parser.add_argument("--transaction-cost", type=float, default=0.0)
    parser.add_argument("--portfolio-quantile", type=float, default=0.1)
    parser.add_argument("--n-buckets", type=int, default=5)
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use yearly expanding-window folds instead of the static train/valid/sim split.",
    )
    parser.add_argument("--walk-forward-start-year", type=int, default=2016)
    parser.add_argument("--walk-forward-end-year", type=int, default=2019)
    return parser.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Dates", "stkcd", "y_next", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    labels = set(df["label"].unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected binary labels 0/1; got {sorted(labels)}")

    df["year"] = (df["Dates"].astype(int) // 100).astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = EXCLUDED_FEATURE_COLUMNS | {"year"}
    feature_columns = [col for col in df.columns if col not in excluded]
    non_numeric = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"Feature columns must be numeric. Non-numeric columns: {non_numeric}")
    if not feature_columns:
        raise ValueError("No feature columns remain after excluding id/target columns.")
    return feature_columns


def rf_candidates(RandomForestClassifier, random_state: int, n_jobs: int):
    return [
        (
            "rf_depth8_leaf50",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=50,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        ),
        (
            "rf_depth12_leaf30",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=30,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        ),
        (
            "rf_depth16_leaf20",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=16,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        ),
    ]


def xgb_candidates(XGBClassifier, random_state: int, n_jobs: int, scale_pos_weight: float):
    common = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": 100,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    return [
        (
            "xgb_depth3_lr005",
            XGBClassifier(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                **common,
            ),
        ),
        (
            "xgb_depth4_lr003",
            XGBClassifier(
                n_estimators=700,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=3.0,
                min_child_weight=15,
                **common,
            ),
        ),
        (
            "xgb_depth5_lr002",
            XGBClassifier(
                n_estimators=1000,
                max_depth=5,
                learning_rate=0.02,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                reg_alpha=0.5,
                min_child_weight=20,
                **common,
            ),
        ),
    ]


def predict_prob(model, X: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index)


def optimize_threshold(y_true: pd.Series, prob: pd.Series, deps: dict) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for i in range(5, 96):
        threshold = i / 100
        pred = (prob >= threshold).astype(int)
        score = deps["balanced_accuracy_score"](y_true, pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def compute_metrics(
    name: str,
    split: str,
    y_true: pd.Series,
    prob: pd.Series,
    deps: dict,
    threshold: float,
    threshold_source: str,
) -> dict:
    pred = (prob >= threshold).astype(int)
    try:
        auc = deps["roc_auc_score"](y_true, prob)
    except ValueError:
        auc = float("nan")

    return {
        "model": name,
        "split": split,
        "accuracy": deps["accuracy_score"](y_true, pred),
        "balanced_accuracy": deps["balanced_accuracy_score"](y_true, pred),
        "auc": auc,
        "precision": deps["precision_score"](y_true, pred, zero_division=0),
        "recall": deps["recall_score"](y_true, pred, zero_division=0),
        "f1": deps["f1_score"](y_true, pred, zero_division=0),
        "threshold": threshold,
        "rows": len(y_true),
        "threshold_source": threshold_source,
    }


def select_best_model(
    candidates: list[tuple[str, object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    deps: dict,
) -> tuple[str, object, float, list[dict]]:
    validation_rows = []
    best_name = ""
    best_model = None
    best_auc = -1.0
    best_threshold = 0.5

    for name, model in candidates:
        print(f"Training candidate: {name}")
        model.fit(X_train, y_train)
        valid_prob = predict_prob(model, X_valid)
        threshold = optimize_threshold(y_valid, valid_prob, deps)
        row = compute_metrics(
            name, "valid", y_valid, valid_prob, deps, threshold, "optimized_on_valid"
        )
        validation_rows.append(row)
        if pd.notna(row["auc"]) and row["auc"] > best_auc:
            best_auc = row["auc"]
            best_name = name
            best_model = model
            best_threshold = threshold

    if best_model is None:
        raise ValueError("Could not select a model; all validation AUC values are missing.")
    return best_name, best_model, best_threshold, validation_rows


def select_xgb_model(
    candidates: list[tuple[str, object]],
    X_train_inner: pd.DataFrame,
    y_train_inner: pd.Series,
    X_early_stop: pd.DataFrame,
    y_early_stop: pd.Series,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    deps: dict,
    scale_factor: float,
) -> tuple[str, object, float, list[dict]]:
    validation_rows = []
    best_name = ""
    best_model = None
    best_auc = -1.0
    best_threshold = 0.5

    for name, model in candidates:
        print(f"Training candidate with internal early stopping: {name}")
        model.fit(
            X_train_inner,
            y_train_inner,
            eval_set=[(X_early_stop, y_early_stop)],
            verbose=False,
        )
        best_iteration = getattr(model, "best_iteration", None)
        raw_best = model.n_estimators if best_iteration is None else best_iteration + 1
        best_n_estimators = min(max(int(raw_best * scale_factor), 1), int(model.n_estimators * 1.5))

        params = model.get_params()
        params.pop("early_stopping_rounds", None)
        params["n_estimators"] = best_n_estimators
        final_model = type(model)(**params)
        final_model.fit(X_train_full, y_train_full)

        valid_prob = predict_prob(final_model, X_valid)
        threshold = optimize_threshold(y_valid, valid_prob, deps)
        row = compute_metrics(
            name, "valid", y_valid, valid_prob, deps, threshold, "optimized_on_valid"
        )
        row["model"] = f"{name}_best{best_n_estimators}"
        validation_rows.append(row)
        if pd.notna(row["auc"]) and row["auc"] > best_auc:
            best_auc = row["auc"]
            best_name = row["model"]
            best_model = final_model
            best_threshold = threshold

    if best_model is None:
        raise ValueError("Could not select an XGBoost model; all validation AUC values are missing.")
    return best_name, best_model, best_threshold, validation_rows


def feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def monthly_long_short(
    frame: pd.DataFrame,
    score_col: str,
    name: str,
    quantile: float,
    seed: int | None,
    fold_year: int | str,
) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("Dates", sort=True):
        if len(group) < 2:
            continue
        bucket_size = max(1, int(len(group) * quantile))
        ranked = group.sort_values(score_col, ascending=False)
        long_names = set(ranked.head(bucket_size)["stkcd"])
        short_names = set(ranked.tail(bucket_size)["stkcd"])
        rows.append(
            {
                "model": name,
                "Dates": date,
                "long_return": ranked.head(bucket_size)["y_next"].mean(),
                "short_return": ranked.tail(bucket_size)["y_next"].mean(),
                "long_short_return": ranked.head(bucket_size)["y_next"].mean()
                - ranked.tail(bucket_size)["y_next"].mean(),
                "bottom_top_return": ranked.tail(bucket_size)["y_next"].mean()
                - ranked.head(bucket_size)["y_next"].mean(),
                "n_long": bucket_size,
                "n_short": bucket_size,
                "long_names": "|".join(map(str, sorted(long_names))),
                "short_names": "|".join(map(str, sorted(short_names))),
                "seed": seed,
                "fold_year": fold_year,
            }
        )
    return pd.DataFrame(rows)


def add_turnover(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    if portfolio_df.empty:
        return portfolio_df
    result = portfolio_df.sort_values(["model", "seed", "fold_year", "Dates"]).copy()
    turnovers = []
    for _, group in result.groupby(["model", "seed"], sort=False):
        prev_long: set[str] | None = None
        prev_short: set[str] | None = None
        for idx, row in group.iterrows():
            long_set = set(str(row["long_names"]).split("|")) if row["long_names"] else set()
            short_set = set(str(row["short_names"]).split("|")) if row["short_names"] else set()
            if prev_long is None or prev_short is None:
                turnover = 1.0
            else:
                long_turnover = 1 - len(long_set & prev_long) / max(len(long_set), 1)
                short_turnover = 1 - len(short_set & prev_short) / max(len(short_set), 1)
                turnover = (long_turnover + short_turnover) / 2
            turnovers.append((idx, turnover))
            prev_long = long_set
            prev_short = short_set
    result["turnover"] = pd.Series(dict(turnovers))
    return result.drop(columns=["long_names", "short_names"])


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return drawdown.min()


def portfolio_summary(
    portfolio_df: pd.DataFrame,
    risk_free_monthly: float,
    transaction_cost: float,
) -> pd.DataFrame:
    cost_map = {
        "long_return": 2 * transaction_cost,
        "short_return": 2 * transaction_cost,
        "long_short_return": 4 * transaction_cost,
        "bottom_top_return": 4 * transaction_cost,
    }
    rows = []
    group_cols = [col for col in ["model", "seed"] if col in portfolio_df.columns]
    for group_key, group in portfolio_df.groupby(group_cols, sort=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row_base = dict(zip(group_cols, key_values))
        for column in ["long_return", "short_return", "long_short_return", "bottom_top_return"]:
            returns = group[column]
            excess = returns - risk_free_monthly
            returns_ac = returns - cost_map[column]
            excess_ac = returns_ac - risk_free_monthly
            std = excess.std()
            std_ac = excess_ac.std()
            rows.append(
                {
                    **row_base,
                    "portfolio": column,
                    "months": len(returns),
                    "mean_monthly_return": returns.mean(),
                    "annualized_return": (1 + returns.mean()) ** 12 - 1,
                    "annualized_sharpe": float("nan") if std == 0 else excess.mean() / std * sqrt(12),
                    "max_drawdown": max_drawdown(returns),
                    "cumulative_return": (1 + returns).prod() - 1,
                    "mean_monthly_return_ac": returns_ac.mean(),
                    "annualized_return_ac": (1 + returns_ac.mean()) ** 12 - 1,
                    "annualized_sharpe_ac": (
                        float("nan") if std_ac == 0 else excess_ac.mean() / std_ac * sqrt(12)
                    ),
                    "max_drawdown_ac": max_drawdown(returns_ac),
                    "cumulative_return_ac": (1 + returns_ac).prod() - 1,
                    "mean_turnover": group.get("turnover", pd.Series(dtype=float)).mean(),
                }
            )
    return pd.DataFrame(rows)


def bucket_returns(
    frame: pd.DataFrame,
    score_col: str,
    name: str,
    n_buckets: int,
    seed: int | None,
    fold_year: int | str,
) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("Dates", sort=True):
        if group[score_col].nunique() < 2:
            continue
        buckets = pd.qcut(group[score_col].rank(method="first"), n_buckets, labels=False) + 1
        temp = group.assign(bucket=buckets)
        for bucket, bucket_group in temp.groupby("bucket", sort=True):
            rows.append(
                {
                    "model": name,
                    "Dates": date,
                    "bucket": int(bucket),
                    "mean_return": bucket_group["y_next"].mean(),
                    "count": len(bucket_group),
                    "seed": seed,
                    "fold_year": fold_year,
                }
            )
    return pd.DataFrame(rows)


def bucket_summary(bucket_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bucket_df.empty:
        return pd.DataFrame()
    for keys, group in bucket_df.groupby(["model", "seed"], sort=False):
        model, seed = keys
        by_bucket = group.groupby("bucket")["mean_return"].mean().sort_index()
        spread = by_bucket.iloc[-1] - by_bucket.iloc[0] if len(by_bucket) >= 2 else float("nan")
        monotonic = by_bucket.index.to_series().corr(by_bucket, method="spearman")
        row = {"model": model, "seed": seed, "q_high_minus_q_low": spread, "spearman_monotonicity": monotonic}
        for bucket, value in by_bucket.items():
            row[f"bucket_{bucket}_mean_return"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def monthly_ic(
    frame: pd.DataFrame,
    score_col: str,
    name: str,
    seed: int | None,
    fold_year: int | str,
) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("Dates", sort=True):
        if group[score_col].nunique() < 2 or group["y_next"].nunique() < 2:
            ic = float("nan")
        else:
            ic = group[score_col].corr(group["y_next"], method="spearman")
        rows.append({"model": name, "Dates": date, "rank_ic": ic, "seed": seed, "fold_year": fold_year})
    return pd.DataFrame(rows)


def newey_west_t(values: pd.Series, max_lag: int | None = None) -> float:
    x = values.dropna().astype(float)
    n = len(x)
    if n < 2:
        return float("nan")
    centered = x - x.mean()
    if max_lag is None:
        max_lag = int(n ** 0.25)
    gamma0 = (centered @ centered) / n
    var = gamma0
    for lag in range(1, max_lag + 1):
        cov = (centered.iloc[lag:].to_numpy() @ centered.iloc[:-lag].to_numpy()) / n
        var += 2 * (1 - lag / (max_lag + 1)) * cov
    se = sqrt(max(var, 0) / n)
    return float("nan") if se == 0 else x.mean() / se


def ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if ic_df.empty:
        return pd.DataFrame()
    for keys, group in ic_df.groupby(["model", "seed"], sort=False):
        model, seed = keys
        ic = group["rank_ic"].dropna()
        rows.append(
            {
                "model": model,
                "seed": seed,
                "months": len(ic),
                "mean_ic": ic.mean(),
                "ic_std": ic.std(),
                "icir": float("nan") if ic.std() == 0 else ic.mean() / ic.std() * sqrt(12),
                "newey_west_t": newey_west_t(ic),
            }
        )
    return pd.DataFrame(rows)


def equity_and_drawdown(portfolio_df: pd.DataFrame, transaction_cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = []
    drawdowns = []
    if portfolio_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    for keys, group in portfolio_df.groupby(["model", "seed"], sort=False):
        model, seed = keys
        ordered = group.sort_values("Dates")
        returns = ordered["long_short_return"]
        returns_ac = returns - 4 * transaction_cost
        wealth = (1 + returns).cumprod()
        wealth_ac = (1 + returns_ac).cumprod()
        dd = wealth / wealth.cummax() - 1
        dd_ac = wealth_ac / wealth_ac.cummax() - 1
        for date, value, value_ac in zip(ordered["Dates"], wealth, wealth_ac):
            curves.append({"model": model, "seed": seed, "Dates": date, "equity": value, "equity_ac": value_ac})
        for date, value, value_ac in zip(ordered["Dates"], dd, dd_ac):
            drawdowns.append({"model": model, "seed": seed, "Dates": date, "drawdown": value, "drawdown_ac": value_ac})
    return pd.DataFrame(curves), pd.DataFrame(drawdowns)


def benchmark_scores(test: pd.DataFrame, seed: int) -> list[tuple[str, pd.Series]]:
    rng = pd.Series(index=test.index, data=0.0)
    for date, idx in test.groupby("Dates").groups.items():
        random_values = pd.Series(range(len(idx)), index=idx).sample(frac=1, random_state=seed + int(date))
        rng.loc[idx] = random_values.rank(pct=True)

    scores = [("benchmark_random", rng)]
    for feature in BENCHMARK_FEATURES:
        if feature in test.columns:
            scores.append((f"benchmark_{feature}", test[feature]))
    return scores


def benchmark_market(test: pd.DataFrame, seed: int | None, fold_year: int | str) -> pd.DataFrame:
    rows = []
    for date, group in test.groupby("Dates", sort=True):
        long_names = "|".join(map(str, sorted(group["stkcd"])))
        rows.append(
            {
                "model": "benchmark_equal_weight",
                "Dates": date,
                "long_return": group["y_next"].mean(),
                "short_return": 0.0,
                "long_short_return": group["y_next"].mean(),
                "bottom_top_return": -group["y_next"].mean(),
                "n_long": len(group),
                "n_short": 0,
                "long_names": long_names,
                "short_names": "",
                "seed": seed,
                "fold_year": fold_year,
            }
        )
    return pd.DataFrame(rows)


def add_model_diagnostics(
    test: pd.DataFrame,
    score: pd.Series,
    model_name: str,
    seed: int,
    fold_year: int | str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = test.copy()
    frame["score"] = score
    portfolios = monthly_long_short(frame, "score", model_name, args.portfolio_quantile, seed, fold_year)
    buckets = bucket_returns(frame, "score", model_name, args.n_buckets, seed, fold_year)
    ics = monthly_ic(frame, "score", model_name, seed, fold_year)
    return portfolios, buckets, ics


def run_single_seed_fold(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    seed: int,
    deps: dict,
    fold_year: int | str,
) -> dict[str, pd.DataFrame]:
    X_train = train[feature_cols]
    y_train = train["label"]
    X_valid = valid[feature_cols]
    y_valid = valid["label"]
    X_test = test[feature_cols]
    y_test = test["label"]

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / positive_count

    rf_name, rf_model, rf_threshold, rf_valid_rows = select_best_model(
        rf_candidates(deps["RandomForestClassifier"], seed, args.n_jobs),
        X_train,
        y_train,
        X_valid,
        y_valid,
        deps,
    )

    train_dates = sorted(train["Dates"].unique())
    n_inner_dates = max(1, int(len(train_dates) * 0.8))
    early_stop_mask = train["Dates"] > train_dates[n_inner_dates - 1]
    if early_stop_mask.all() or not early_stop_mask.any():
        raise ValueError("Cannot create internal XGBoost early-stopping split from train dates.")

    xgb_name, xgb_model, xgb_threshold, xgb_valid_rows = select_xgb_model(
        xgb_candidates(deps["XGBClassifier"], seed, args.n_jobs, scale_pos_weight),
        train.loc[~early_stop_mask, feature_cols],
        train.loc[~early_stop_mask, "label"],
        train.loc[early_stop_mask, feature_cols],
        train.loc[early_stop_mask, "label"],
        X_train,
        y_train,
        X_valid,
        y_valid,
        deps,
        scale_factor=len(train_dates) / n_inner_dates,
    )

    rf_valid_prob = predict_prob(rf_model, X_valid)
    xgb_valid_prob = predict_prob(xgb_model, X_valid)
    rf_test_prob = predict_prob(rf_model, X_test)
    xgb_test_prob = predict_prob(xgb_model, X_test)

    metrics = rf_valid_rows + xgb_valid_rows
    metrics.extend(
        [
            compute_metrics(
                f"{rf_name}_selected",
                "valid",
                y_valid,
                rf_valid_prob,
                deps,
                rf_threshold,
                "optimized_on_valid",
            ),
            compute_metrics(
                f"{rf_name}_selected",
                "sim",
                y_test,
                rf_test_prob,
                deps,
                rf_threshold,
                "held_out",
            ),
            compute_metrics(
                f"{xgb_name}_selected",
                "valid",
                y_valid,
                xgb_valid_prob,
                deps,
                xgb_threshold,
                "optimized_on_valid",
            ),
            compute_metrics(
                f"{xgb_name}_selected",
                "sim",
                y_test,
                xgb_test_prob,
                deps,
                xgb_threshold,
                "held_out",
            ),
        ]
    )
    metrics_df = pd.DataFrame(metrics, columns=METRIC_COLUMNS)
    metrics_df["seed"] = seed
    metrics_df["fold_year"] = fold_year

    predictions = test[["Dates", "stkcd", "y_next", "label"]].copy()
    predictions["rf_prob"] = rf_test_prob
    predictions["xgb_prob"] = xgb_test_prob
    predictions["rf_pred"] = (rf_test_prob >= rf_threshold).astype(int)
    predictions["xgb_pred"] = (xgb_test_prob >= xgb_threshold).astype(int)
    predictions["seed"] = seed
    predictions["fold_year"] = fold_year

    all_portfolios = []
    all_buckets = []
    all_ics = []
    for model_name, score in [
        (f"{rf_name}_selected", rf_test_prob),
        (f"{xgb_name}_selected", xgb_test_prob),
    ]:
        p, b, i = add_model_diagnostics(test, score, model_name, seed, fold_year, args)
        all_portfolios.append(p)
        all_buckets.append(b)
        all_ics.append(i)

    benchmark_portfolios = [benchmark_market(test, seed, fold_year)]
    benchmark_buckets = []
    benchmark_ics = []
    for name, score in benchmark_scores(test, seed):
        p, b, i = add_model_diagnostics(test, score, name, seed, fold_year, args)
        benchmark_portfolios.append(p)
        benchmark_buckets.append(b)
        benchmark_ics.append(i)

    print(f"[seed={seed} fold={fold_year}] RF: {rf_name}, XGBoost: {xgb_name}")
    return {
        "metrics": metrics_df,
        "predictions": predictions,
        "portfolios": pd.concat(all_portfolios, ignore_index=True),
        "benchmarks": pd.concat(benchmark_portfolios, ignore_index=True),
        "buckets": pd.concat(all_buckets + benchmark_buckets, ignore_index=True),
        "ics": pd.concat(all_ics + benchmark_ics, ignore_index=True),
        "rf_importance": feature_importance(rf_model, feature_cols),
        "xgb_importance": feature_importance(xgb_model, feature_cols),
    }


def make_folds(df: pd.DataFrame, walk_forward: bool, start_year: int, end_year: int):
    if not walk_forward:
        return [
            (
                "static",
                df[df["split"] == "train"],
                df[df["split"] == "valid"],
                df[df["split"] == "sim"],
            )
        ]
    folds = []
    for test_year in range(start_year, end_year + 1):
        train = df[df["year"] <= test_year - 2]
        valid = df[df["year"] == test_year - 1]
        test = df[df["year"] == test_year]
        if train.empty or valid.empty or test.empty:
            print(f"Skipping fold {test_year}: train/valid/test is empty.")
            continue
        folds.append((test_year, train, valid, test))
    if not folds:
        raise ValueError("No valid walk-forward folds were created.")
    return folds


def avg_importance(imps: list[pd.DataFrame]) -> pd.DataFrame:
    if len(imps) == 1:
        return imps[0]
    return (
        pd.concat(imps)
        .groupby("feature", sort=False)["importance"]
        .mean()
        .reset_index()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> int:
    args = parse_args()
    deps = import_ml_dependencies()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_panel(Path(args.data))
    feature_cols = get_feature_columns(df)
    folds = make_folds(
        df,
        args.walk_forward,
        args.walk_forward_start_year,
        args.walk_forward_end_year,
    )

    results = []
    for fold_year, train, valid, test in folds:
        for seed in args.random_seeds:
            results.append(run_single_seed_fold(train, valid, test, feature_cols, args, seed, deps, fold_year))

    metrics_df = pd.concat([r["metrics"] for r in results], ignore_index=True)
    predictions_df = pd.concat([r["predictions"] for r in results], ignore_index=True)
    portfolios_df = add_turnover(pd.concat([r["portfolios"] for r in results], ignore_index=True))
    benchmarks_df = add_turnover(pd.concat([r["benchmarks"] for r in results], ignore_index=True))
    buckets_df = pd.concat([r["buckets"] for r in results], ignore_index=True)
    ics_df = pd.concat([r["ics"] for r in results], ignore_index=True)

    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    predictions_df.to_csv(out_dir / "sim_predictions.csv", index=False)
    portfolios_df.to_csv(out_dir / "portfolio_returns.csv", index=False)
    portfolio_summary(portfolios_df, args.risk_free_monthly, args.transaction_cost).to_csv(
        out_dir / "portfolio_summary.csv", index=False
    )
    benchmarks_df.to_csv(out_dir / "benchmark_returns.csv", index=False)
    portfolio_summary(benchmarks_df, args.risk_free_monthly, args.transaction_cost).to_csv(
        out_dir / "benchmark_summary.csv", index=False
    )
    buckets_df.to_csv(out_dir / "bucket_returns.csv", index=False)
    bucket_summary(buckets_df).to_csv(out_dir / "bucket_summary.csv", index=False)
    ics_df.to_csv(out_dir / "ic_by_month.csv", index=False)
    ic_summary(ics_df).to_csv(out_dir / "ic_summary.csv", index=False)
    curves_df, drawdowns_df = equity_and_drawdown(
        pd.concat([portfolios_df, benchmarks_df], ignore_index=True),
        args.transaction_cost,
    )
    curves_df.to_csv(out_dir / "equity_curves.csv", index=False)
    drawdowns_df.to_csv(out_dir / "drawdowns.csv", index=False)
    portfolios_df[["model", "seed", "fold_year", "Dates", "turnover"]].to_csv(
        out_dir / "turnover.csv", index=False
    )
    avg_importance([r["rf_importance"] for r in results]).to_csv(
        out_dir / "feature_importance_rf.csv", index=False
    )
    avg_importance([r["xgb_importance"] for r in results]).to_csv(
        out_dir / "feature_importance_xgb.csv", index=False
    )

    sim_sel = metrics_df[(metrics_df["split"] == "sim") & (metrics_df["threshold_source"] == "held_out")]
    for label, mask in [
        ("RF", sim_sel["model"].str.contains("rf", case=False)),
        ("XGB", sim_sel["model"].str.contains("xgb", case=False)),
    ]:
        grp = sim_sel.loc[mask, "auc"]
        if not grp.empty:
            print(f"[aggregate {label}] sim AUC = {grp.mean():.4f} +/- {grp.std():.4f}")

    print(f"Saved diagnostics to: {out_dir}")
    print("Primary diagnostics: ic_summary.csv, bucket_summary.csv, portfolio_summary.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
