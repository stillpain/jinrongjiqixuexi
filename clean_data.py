import argparse
from pathlib import Path

import pandas as pd


TRAIN_START = 201001
TRAIN_END = 201511
VALID_START = 201601
VALID_END = 201711
SIM_START = 201801
SIM_END = 201911

ID_COLUMNS = ["Dates", "stkcd"]
TARGET_COLUMN = "y"
MACRO_COLUMNS = ["Vol", "GDPgrowth", "CPIgrowth"]
NON_FEATURE_COLUMNS = {"Dates", "stkcd", "y", "y_next", "label", "split"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean the China A-share monthly panel, build next-month up/down "
            "labels, merge lagged macro predictors, and create time splits."
        )
    )
    parser.add_argument(
        "--stock",
        default="CHN_sample_data.csv",
        help="Path to the stock-level monthly panel CSV.",
    )
    parser.add_argument(
        "--macro",
        default="CHN_Marco_predictors.csv",
        help="Path to the monthly macro predictors CSV.",
    )
    parser.add_argument(
        "--out",
        default="data/cleaned_panel.csv",
        help="Output path for the cleaned panel CSV.",
    )
    parser.add_argument(
        "--keep-zero-return",
        action="store_true",
        help="Keep y_next == 0 observations and label them as 0. Default drops them.",
    )
    parser.add_argument(
        "--missing-strategy",
        choices=["monthly-median", "drop", "none"],
        default="monthly-median",
        help=(
            "How to handle feature missing values. monthly-median fills by each "
            "month's cross-sectional median and then train median; drop removes "
            "rows with any missing feature; none leaves missing values in place."
        ),
    )
    parser.add_argument("--train-start", type=int, default=TRAIN_START,
                        help="First YYYYMM of training period.")
    parser.add_argument("--train-end",   type=int, default=TRAIN_END,
                        help="Last YYYYMM of training period.")
    parser.add_argument("--valid-start", type=int, default=VALID_START,
                        help="First YYYYMM of validation period.")
    parser.add_argument("--valid-end",   type=int, default=VALID_END,
                        help="Last YYYYMM of validation period.")
    parser.add_argument("--sim-start",   type=int, default=SIM_START,
                        help="First YYYYMM of simulation period.")
    parser.add_argument("--sim-end",     type=int, default=SIM_END,
                        help="Last YYYYMM of simulation period.")
    return parser.parse_args()


def normalize_dates(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "Dates" not in df.columns:
        raise ValueError(f"{source_name} must contain a Dates column.")
    df = df.copy()
    df["Dates"] = pd.to_numeric(df["Dates"], errors="raise").astype(int)
    return df


def next_month(dates: pd.Series) -> pd.Series:
    years = dates // 100
    months = dates % 100
    next_years = years + (months == 12).astype(int)
    next_months = months.where(months < 12, 0) + 1
    return next_years * 100 + next_months


def load_stock_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_dates(df, str(path))

    missing = [col for col in ID_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    duplicate_count = df.duplicated(ID_COLUMNS).sum()
    if duplicate_count:
        raise ValueError(f"{path} has {duplicate_count} duplicate Dates-stkcd rows.")

    df = df.sort_values(["stkcd", "Dates"]).reset_index(drop=True)
    df["label_date"] = next_month(df["Dates"])
    next_returns = df[["stkcd", "Dates", TARGET_COLUMN]].rename(
        columns={"Dates": "label_date", TARGET_COLUMN: "y_next"}
    )
    df = df.merge(next_returns, on=["stkcd", "label_date"], how="left", validate="many_to_one")
    df = df.drop(columns=["label_date"])
    return df


def load_lagged_macro(path: Path) -> pd.DataFrame:
    macro = pd.read_csv(path)
    macro = normalize_dates(macro, str(path))

    missing = [col for col in ["Dates"] + MACRO_COLUMNS if col not in macro.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    duplicate_count = macro.duplicated(["Dates"]).sum()
    if duplicate_count:
        raise ValueError(f"{path} has {duplicate_count} duplicate Dates rows.")

    macro = macro.sort_values("Dates").reset_index(drop=True)
    lagged = macro[["Dates"] + MACRO_COLUMNS].copy()
    lagged["Dates"] = next_month(lagged["Dates"])
    lagged = lagged.rename(columns={col: f"macro_lag1_{col}" for col in MACRO_COLUMNS})
    return lagged


def assign_split(
    dates: pd.Series,
    train_start: int = TRAIN_START,
    train_end: int = TRAIN_END,
    valid_start: int = VALID_START,
    valid_end: int = VALID_END,
    sim_start: int = SIM_START,
    sim_end: int = SIM_END,
) -> pd.Series:
    split = pd.Series(pd.NA, index=dates.index, dtype="object")
    split.loc[dates.between(train_start, train_end)] = "train"
    split.loc[dates.between(valid_start, valid_end)] = "valid"
    split.loc[dates.between(sim_start, sim_end)] = "sim"
    return split


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]


def handle_feature_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    features = feature_columns(df)
    missing_before = int(df[features].isna().sum().sum())
    if missing_before == 0:
        print("Feature missing values: 0")
        return df

    df = df.copy()
    if strategy == "drop":
        before_rows = len(df)
        df = df.dropna(subset=features).copy()
        print(
            f"Feature missing values: {missing_before:,}; "
            f"dropped {before_rows - len(df):,} rows."
        )
        return df

    if strategy == "monthly-median":
        train_medians = df.loc[df["split"] == "train", features].median(numeric_only=True)
        df[features] = df.groupby("Dates", sort=False)[features].transform(
            lambda col: col.fillna(col.median())
        )
        df[features] = df[features].fillna(train_medians)
        missing_after = int(df[features].isna().sum().sum())
        print(
            f"Feature missing values: {missing_before:,}; "
            f"remaining after monthly/train median fill: {missing_after:,}"
        )
        return df

    print(f"Feature missing values left unchanged: {missing_before:,}")
    return df


def validate_cleaned(df: pd.DataFrame, allow_feature_na: bool) -> None:
    if df.empty:
        raise ValueError("Cleaned panel is empty after filtering.")

    if not pd.api.types.is_integer_dtype(df["Dates"]):
        raise ValueError("Dates must be integer YYYYMM values.")

    labels = set(df["label"].dropna().unique())
    if labels != {0, 1}:
        raise ValueError(f"label must contain exactly 0/1 classes; got {sorted(labels)}")

    splits = set(df["split"].dropna().unique())
    expected_splits = {"train", "valid", "sim"}
    if splits != expected_splits:
        raise ValueError(f"split must contain {expected_splits}; got {splits}")

    if df[["Dates", "stkcd"]].duplicated().any():
        raise ValueError("Cleaned panel contains duplicate Dates-stkcd rows.")

    required_no_na = ["Dates", "stkcd", "y", "y_next", "label", "split"]
    required_missing = df[required_no_na].isna().sum()
    required_missing = required_missing[required_missing > 0]
    if not required_missing.empty:
        raise ValueError(f"Required columns still have missing values:\n{required_missing}")

    features = feature_columns(df)
    feature_missing = df[features].isna().sum()
    feature_missing = feature_missing[feature_missing > 0].sort_values(ascending=False)
    if not allow_feature_na and not feature_missing.empty:
        raise ValueError(f"Feature columns still have missing values:\n{feature_missing}")

    non_feature_missing = df.drop(columns=features).isna().sum()
    non_feature_missing = non_feature_missing[non_feature_missing > 0]
    if not non_feature_missing.empty:
        missing = non_feature_missing.sort_values(ascending=False)
        missing = missing[missing > 0].sort_values(ascending=False)
        raise ValueError(f"Cleaned panel still has missing values:\n{missing}")


def build_cleaned_panel(
    stock_path: Path,
    macro_path: Path,
    keep_zero_return: bool,
    missing_strategy: str,
    train_start: int = TRAIN_START,
    train_end: int = TRAIN_END,
    valid_start: int = VALID_START,
    valid_end: int = VALID_END,
    sim_start: int = SIM_START,
    sim_end: int = SIM_END,
) -> pd.DataFrame:
    stock = load_stock_panel(stock_path)
    macro = load_lagged_macro(macro_path)

    cleaned = stock.merge(macro, on="Dates", how="left", validate="many_to_one")
    cleaned = cleaned.dropna(subset=["y_next"] + [f"macro_lag1_{col}" for col in MACRO_COLUMNS])

    if not keep_zero_return:
        cleaned = cleaned[cleaned["y_next"] != 0].copy()

    cleaned["label"] = (cleaned["y_next"] > 0).astype(int)
    cleaned["split"] = assign_split(
        cleaned["Dates"],
        train_start, train_end, valid_start, valid_end, sim_start, sim_end,
    )
    cleaned = cleaned.dropna(subset=["split"]).copy()
    cleaned = cleaned.sort_values(["Dates", "stkcd"]).reset_index(drop=True)
    cleaned = handle_feature_missing(cleaned, missing_strategy)
    cleaned = cleaned.sort_values(["Dates", "stkcd"]).reset_index(drop=True)

    validate_cleaned(cleaned, allow_feature_na=missing_strategy == "none")
    return cleaned


def print_summary(df: pd.DataFrame, out_path: Path) -> None:
    print(f"Saved cleaned panel: {out_path}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print(f"Date range: {df['Dates'].min()} to {df['Dates'].max()}")
    print(f"Stocks: {df['stkcd'].nunique():,}")
    print(
        "Default embargo: train ends at 201511 and validation ends at 201711 "
        "because labels use the next calendar month."
    )
    print("\nSplit summary:")
    summary = (
        df.groupby("split", sort=False)
        .agg(
            rows=("label", "size"),
            dates=("Dates", "nunique"),
            stocks=("stkcd", "nunique"),
            first_date=("Dates", "min"),
            last_date=("Dates", "max"),
            positive_rate=("label", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    stock_path = Path(args.stock)
    macro_path = Path(args.macro)
    out_path = Path(args.out)

    cleaned = build_cleaned_panel(
        stock_path,
        macro_path,
        args.keep_zero_return,
        args.missing_strategy,
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        sim_start=args.sim_start,
        sim_end=args.sim_end,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    print_summary(cleaned, out_path)


if __name__ == "__main__":
    main()
