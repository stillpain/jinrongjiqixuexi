import argparse
import sys
from math import sqrt
from pathlib import Path

import pandas as pd


EXCLUDED_FEATURE_COLUMNS = {"Dates", "stkcd", "y", "y_next", "label", "split"}
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
            "pip install scikit-learn xgboost"
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
            "panel, select parameters on validation data, and predict simulation data."
        )
    )
    parser.add_argument(
        "--data",
        default="data/cleaned_panel.csv",
        help="Path to cleaned panel CSV created by clean_data.py.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory for metrics, predictions, and feature importance files.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for supported models.",
    )
    parser.add_argument(
        "--random-seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for training (e.g. --random-seeds 42 123 456). Multiple seeds enable robustness checks.",
    )
    parser.add_argument(
        "--risk-free-monthly",
        type=float,
        default=0.0,
        help="Monthly risk-free rate subtracted from returns when computing Sharpe ratio (e.g. 0.002 for ~2.4%% annualized).",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.0,
        help="One-way transaction cost per trade (e.g. 0.001 for 0.1%%). Applied as 2× per leg, 4× for long-short, per month.",
    )
    parser.add_argument(
        "--portfolio-quantile",
        type=float,
        default=0.1,
        help="Top/bottom quantile for long-short portfolio construction (e.g. 0.1 for top/bottom 10%%).",
    )
    return parser.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Dates", "stkcd", "y_next", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    valid_splits = {"train", "valid", "sim"}
    splits = set(df["split"].unique())
    if splits != valid_splits:
        raise ValueError(f"Expected splits {valid_splits}; got {splits}")

    labels = set(df["label"].unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected binary labels 0/1; got {sorted(labels)}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [col for col in df.columns if col not in EXCLUDED_FEATURE_COLUMNS]
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
    threshold_source: str = "",
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
        row = compute_metrics(name, "valid", y_valid, valid_prob, deps, threshold,
                              threshold_source="optimized_on_valid")
        validation_rows.append(row)
        candidate_auc = row["auc"]
        if pd.notna(candidate_auc) and candidate_auc > best_auc:
            best_auc = candidate_auc
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
    scale_factor: float = 1.0,
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
        scaled = int(raw_best * scale_factor)
        max_cap = int(model.n_estimators * 1.5)
        best_n_estimators = min(max(scaled, 1), max_cap)

        params = model.get_params()
        params.pop("early_stopping_rounds", None)
        params["n_estimators"] = best_n_estimators
        final_model = type(model)(**params)
        final_model.fit(X_train_full, y_train_full)

        valid_prob = predict_prob(final_model, X_valid)
        threshold = optimize_threshold(y_valid, valid_prob, deps)
        row = compute_metrics(name, "valid", y_valid, valid_prob, deps, threshold,
                              threshold_source="optimized_on_valid")
        row["model"] = f"{name}_best{best_n_estimators}"
        validation_rows.append(row)

        candidate_auc = row["auc"]
        if pd.notna(candidate_auc) and candidate_auc > best_auc:
            best_auc = candidate_auc
            best_name = row["model"]
            best_model = final_model
            best_threshold = threshold

    if best_model is None:
        raise ValueError("Could not select an XGBoost model; all validation AUC values are missing.")

    return best_name, best_model, best_threshold, validation_rows


def feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"Model {type(model).__name__} does not expose feature_importances_.")
    return (
        pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def portfolio_returns(
    sim: pd.DataFrame,
    prob: pd.Series,
    model_name: str,
    quantile: float = 0.1,
) -> pd.DataFrame:
    frame = sim[["Dates", "stkcd", "y_next"]].copy()
    frame["prob"] = prob
    rows = []
    for date, group in frame.groupby("Dates", sort=True):
        if len(group) < 2:
            continue
        bucket_size = max(1, int(len(group) * quantile))
        ranked = group.sort_values("prob", ascending=False)
        long_return = ranked.head(bucket_size)["y_next"].mean()
        short_return = ranked.tail(bucket_size)["y_next"].mean()
        rows.append(
            {
                "model": model_name,
                "Dates": date,
                "long_return": long_return,
                "short_return": short_return,
                "long_short_return": long_return - short_return,
                "n_long": bucket_size,
                "n_short": bucket_size,
            }
        )
    return pd.DataFrame(rows)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return drawdown.min()


def portfolio_summary(
    portfolio_df: pd.DataFrame,
    risk_free_monthly: float = 0.0,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    # cost per month per leg: 2× one-way for single legs, 4× for the combined long-short
    cost_map = {
        "long_return": 2 * transaction_cost,
        "short_return": 2 * transaction_cost,
        "long_short_return": 4 * transaction_cost,
    }
    has_seed = "seed" in portfolio_df.columns
    group_cols = ["model", "seed"] if has_seed else ["model"]
    rows = []
    for group_key, group in portfolio_df.groupby(group_cols, sort=False):
        if has_seed:
            model, seed_val = group_key
        else:
            model, seed_val = group_key, None
        for column in ["long_return", "short_return", "long_short_return"]:
            returns = group[column]
            excess = returns - risk_free_monthly
            std = excess.std()
            returns_ac = returns - cost_map[column]
            excess_ac = returns_ac - risk_free_monthly
            std_ac = excess_ac.std()
            row = {
                "model": model,
                "portfolio": column,
                "months": len(returns),
                "mean_monthly_return": returns.mean(),
                "annualized_return": (1 + returns.mean()) ** 12 - 1,
                "annualized_sharpe": float("nan") if std == 0 else excess.mean() / std * sqrt(12),
                "max_drawdown": max_drawdown(returns),
                "cumulative_return": (1 + returns).prod() - 1,
                "mean_monthly_return_ac": returns_ac.mean(),
                "annualized_return_ac": (1 + returns_ac.mean()) ** 12 - 1,
                "annualized_sharpe_ac": float("nan") if std_ac == 0 else excess_ac.mean() / std_ac * sqrt(12),
                "max_drawdown_ac": max_drawdown(returns_ac),
                "cumulative_return_ac": (1 + returns_ac).prod() - 1,
            }
            if seed_val is not None:
                row["seed"] = seed_val
            rows.append(row)
    return pd.DataFrame(rows)


def run_single_seed(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    sim: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    seed: int,
    deps: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = train[feature_cols]
    y_train = train["label"]
    X_valid = valid[feature_cols]
    y_valid = valid["label"]
    X_sim = sim[feature_cols]
    y_sim = sim["label"]

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    rf_name, rf_model, rf_threshold, rf_valid_rows = select_best_model(
        rf_candidates(deps["RandomForestClassifier"], seed, args.n_jobs),
        X_train, y_train, X_valid, y_valid, deps,
    )

    train_dates_sorted = sorted(train["Dates"].unique())
    n_full_dates = len(train_dates_sorted)
    n_inner_dates = max(1, int(n_full_dates * 0.8))
    inner_cutoff = train_dates_sorted[n_inner_dates - 1]
    early_stop_mask = train["Dates"] > inner_cutoff
    if early_stop_mask.all() or not early_stop_mask.any():
        raise ValueError("Cannot create internal XGBoost early-stopping split from train dates.")

    X_train_inner = train.loc[~early_stop_mask, feature_cols]
    y_train_inner = train.loc[~early_stop_mask, "label"]
    X_early_stop = train.loc[early_stop_mask, feature_cols]
    y_early_stop = train.loc[early_stop_mask, "label"]

    xgb_name, xgb_model, xgb_threshold, xgb_valid_rows = select_xgb_model(
        xgb_candidates(deps["XGBClassifier"], seed, args.n_jobs, scale_pos_weight),
        X_train_inner, y_train_inner,
        X_early_stop, y_early_stop,
        X_train, y_train,
        X_valid, y_valid,
        deps,
        scale_factor=n_full_dates / n_inner_dates,
    )

    rf_valid_prob = predict_prob(rf_model, X_valid)
    xgb_valid_prob = predict_prob(xgb_model, X_valid)
    rf_sim_prob = predict_prob(rf_model, X_sim)
    xgb_sim_prob = predict_prob(xgb_model, X_sim)

    metrics = []
    metrics.extend(rf_valid_rows)
    metrics.extend(xgb_valid_rows)
    metrics.append(
        compute_metrics(f"{rf_name}_selected", "valid", y_valid, rf_valid_prob, deps, rf_threshold,
                        threshold_source="optimized_on_valid")
    )
    metrics.append(
        compute_metrics(f"{rf_name}_selected", "sim", y_sim, rf_sim_prob, deps, rf_threshold,
                        threshold_source="held_out")
    )
    metrics.append(
        compute_metrics(
            f"{xgb_name}_selected", "valid", y_valid, xgb_valid_prob, deps, xgb_threshold,
            threshold_source="optimized_on_valid",
        )
    )
    metrics.append(
        compute_metrics(f"{xgb_name}_selected", "sim", y_sim, xgb_sim_prob, deps, xgb_threshold,
                        threshold_source="held_out")
    )

    metrics_df = pd.DataFrame(metrics, columns=METRIC_COLUMNS)
    metrics_df["seed"] = seed

    sim_predictions = sim[["Dates", "stkcd", "y_next", "label"]].copy()
    sim_predictions["rf_prob"] = rf_sim_prob
    sim_predictions["xgb_prob"] = xgb_sim_prob
    sim_predictions["rf_pred"] = (sim_predictions["rf_prob"] >= rf_threshold).astype(int)
    sim_predictions["xgb_pred"] = (sim_predictions["xgb_prob"] >= xgb_threshold).astype(int)
    sim_predictions["seed"] = seed

    portfolios = pd.concat(
        [
            portfolio_returns(sim, rf_sim_prob, f"{rf_name}_selected", quantile=args.portfolio_quantile),
            portfolio_returns(sim, xgb_sim_prob, f"{xgb_name}_selected", quantile=args.portfolio_quantile),
        ],
        ignore_index=True,
    )
    portfolios["seed"] = seed

    rf_imp = feature_importance(rf_model, feature_cols)
    xgb_imp = feature_importance(xgb_model, feature_cols)

    print(f"[seed={seed}] RF: {rf_name}, XGBoost: {xgb_name}")
    print(f"[seed={seed}] RF threshold: {rf_threshold:.2f}, XGBoost threshold: {xgb_threshold:.2f}")
    print(f"[seed={seed}] XGBoost scale_pos_weight: {scale_pos_weight:.4f}")

    return metrics_df, portfolios, sim_predictions, rf_imp, xgb_imp


def main() -> int:
    args = parse_args()
    deps = import_ml_dependencies()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_panel(data_path)
    feature_cols = get_feature_columns(df)

    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"]
    sim = df[df["split"] == "sim"]

    all_metrics: list[pd.DataFrame] = []
    all_portfolios: list[pd.DataFrame] = []
    all_sim_predictions: list[pd.DataFrame] = []
    all_rf_imps: list[pd.DataFrame] = []
    all_xgb_imps: list[pd.DataFrame] = []

    for seed in args.random_seeds:
        m, p, s, ri, xi = run_single_seed(train, valid, sim, feature_cols, args, seed, deps)
        all_metrics.append(m)
        all_portfolios.append(p)
        all_sim_predictions.append(s)
        all_rf_imps.append(ri)
        all_xgb_imps.append(xi)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    portfolios_df = pd.concat(all_portfolios, ignore_index=True)

    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    pd.concat(all_sim_predictions, ignore_index=True).to_csv(out_dir / "sim_predictions.csv", index=False)

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

    avg_importance(all_rf_imps).to_csv(out_dir / "feature_importance_rf.csv", index=False)
    avg_importance(all_xgb_imps).to_csv(out_dir / "feature_importance_xgb.csv", index=False)

    portfolios_df.to_csv(out_dir / "portfolio_returns.csv", index=False)
    summary_df = portfolio_summary(
        portfolios_df,
        risk_free_monthly=args.risk_free_monthly,
        transaction_cost=args.transaction_cost,
    )
    summary_df.to_csv(out_dir / "portfolio_summary.csv", index=False)

    if len(args.random_seeds) > 1:
        sim_sel = metrics_df[
            (metrics_df["split"] == "sim") & (metrics_df["threshold_source"] == "held_out")
        ]
        for label, mask in [
            ("RF", sim_sel["model"].str.contains("rf", case=False)),
            ("XGB", sim_sel["model"].str.contains("xgb", case=False)),
        ]:
            grp = sim_sel.loc[mask, "auc"]
            if not grp.empty:
                print(f"[aggregate {label}] sim AUC = {grp.mean():.4f} ± {grp.std():.4f}")

        ls_summary = summary_df[summary_df["portfolio"] == "long_short_return"]
        for label, mask in [
            ("RF", ls_summary["model"].str.contains("rf", case=False)),
            ("XGB", ls_summary["model"].str.contains("xgb", case=False)),
        ]:
            grp = ls_summary.loc[mask, "annualized_sharpe_ac"]
            if not grp.empty:
                print(
                    f"[aggregate {label}] long-short Sharpe (after-cost) = "
                    f"{grp.mean():.4f} ± {grp.std():.4f}"
                )

    print(f"Saved metrics: {metrics_path}")
    print(
        "Note: valid metrics use threshold optimized on the same split. "
        "sim metrics are held_out and unbiased."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
