import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path.cwd().resolve()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load California Housing data and validate its structure."""
    dataset = fetch_california_housing(as_frame=True, download_if_missing=True)
    features, target = dataset.data, dataset.target

    if features.empty:
        raise ValueError("Dataset contains no rows.")

    if features.isnull().any().any() or target.isnull().any():
        raise ValueError("Dataset contains missing values.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes
    ):
        raise ValueError("All input features must be numeric.")

    return features, target


def train_and_evaluate() -> dict[str, float]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    features, target = load_data()

    test_size = 0.2
    random_state = 42
    alpha = 1.0

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "train_rows": float(len(X_train)),
        "test_rows": float(len(X_test)),
    }

    model_path = ARTIFACTS_DIR / "baseline_model.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"

    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    tracking_dir = ARTIFACTS_DIR / "mlflow"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    database_path = tracking_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{database_path.resolve()}")

    mlflow.set_experiment("house-price-baseline")

    with mlflow.start_run(run_name="ridge-baseline"):
        mlflow.log_params(
            {
                "model_type": "Ridge",
                "alpha": alpha,
                "test_size": test_size,
                "random_state": random_state,
                "feature_count": features.shape[1],
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))

        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"MLflow tracking data saved to: {tracking_dir}")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()