import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

BASE_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]
STATISTICS = ["mean", "se", "worst"]
FEATURE_COLUMNS = [f"{name}_{stat}" for stat in STATISTICS for name in BASE_FEATURES]
COLUMN_NAMES = ["id", "diagnosis"] + FEATURE_COLUMNS


def validate_raw_data(raw_wdbc: pd.DataFrame, data_validation: Dict[str, int]) -> pd.DataFrame:
    expected_rows = data_validation["expected_rows"]
    expected_columns = data_validation["expected_columns"]

    if raw_wdbc.shape != (expected_rows, expected_columns):
        raise ValueError(
            f"Unexpected shape {raw_wdbc.shape}, "
            f"expected ({expected_rows}, {expected_columns})"
        )
    if len(COLUMN_NAMES) != raw_wdbc.shape[1]:
        raise ValueError("Number of column names does not match the data")

    df = raw_wdbc.copy()
    df.columns = COLUMN_NAMES

    if df.isnull().any().any():
        raise ValueError("Raw data contains missing values")

    labels = set(df["diagnosis"].unique())
    if not labels <= {"M", "B"}:
        raise ValueError(f"Unexpected diagnosis labels: {labels}")

    non_numeric = [
        col for col in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")

    logger.info("Validated raw data: %d rows, %d columns", *df.shape)
    return df


def clean_data(validated_wdbc: pd.DataFrame) -> pd.DataFrame:
    df = validated_wdbc.drop(columns=["id"]).drop_duplicates()
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    if df["diagnosis"].isnull().any():
        raise ValueError("Diagnosis encoding produced missing values")

    logger.info("Cleaned data: %d rows, malignant share %.3f", len(df), df["diagnosis"].mean())
    return df


def split_data(cleaned_wdbc: pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = cleaned_wdbc.drop(columns=["diagnosis"])
    y = cleaned_wdbc[["diagnosis"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y["diagnosis"],
    )
    logger.info("Split data: %d train rows, %d test rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
