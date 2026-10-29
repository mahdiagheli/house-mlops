import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECTROOT=Path(__file__).resolve().parents[3]
ARTIFACTS_DIR=PROJECTROOT/"artifacts"

def load_data() -> [pd.DataFrame,pd.Series]:
    dataset=fetch_california_housing(as_frame=True,download_if_missing=True)
    x,y=dataset.data,dataset.target
    if x.empty:
        raise ValueError("Dataset contain no row")
    if x.isnull().any().any() or y.isnull().any():
        raise ValueError("dataset has missing value")
    return x,y


def train_evaulate()-> dict[str, float]:
    ARTIFACTS_DIR.mkdir(parents=True,exist_ok=True)
    x,y=load_data()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    model=Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
            ("regression",Ridge(alpha=1.0))
        ]
    )
    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    metrics={
        "mae":float(mean_absolute_error(y_test,prediction)),
        "rmse":float(mean_squared_error(y_test,prediction)**0.5),
        "train_row":float(len(x_train)),
        "test_row":float(len(x_test))
    }
    joblib.dump(model, ARTIFACTS_DIR / "baseline_model.joblib")

    with (ARTIFACTS_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {ARTIFACTS_DIR / 'baseline_model.joblib'}")
    print(f"Metrics saved to: {ARTIFACTS_DIR / 'metrics.json'}")

    return metrics

if __name__=="__main__":
    train_evaulate()