import logging
from typing import Any, Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_options: Dict[str, Any],
    random_state: int,
) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=model_options["C"],
                    max_iter=model_options["max_iter"],
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train["diagnosis"])
    logger.info("Trained model on %d rows", len(X_train))
    return model


def evaluate_model(
    trained_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Dict[str, Any]:
    y_true = y_test["diagnosis"]
    y_pred = trained_model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "n_test_samples": int(len(y_true)),
    }
    logger.info("Test accuracy %.3f, recall %.3f", metrics["accuracy"], metrics["recall"])
    return metrics
