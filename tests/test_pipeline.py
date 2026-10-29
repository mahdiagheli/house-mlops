
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from houseprice_mlops.pipelines.train import train_and_evaluate


def test_pipeline_trains_and_returns_valid_metrics(
    tmp_path, monkeypatch
) -> None:
    import houseprice_mlops.pipelines.train as train_module

    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        rng.normal(size=(120, 8)),
        columns=[f"feature_{i}" for i in range(8)],
    )
    target = pd.Series(
        2.0 * features["feature_0"]
        - 0.5 * features["feature_1"]
        + rng.normal(scale=0.1, size=120)
    )

    monkeypatch.setattr(train_module, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(
        train_module,
        "load_data",
        lambda: (features, target),
    )

    metrics = train_and_evaluate()

    assert set(metrics) == {"mae", "rmse", "train_rows", "test_rows"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["train_rows"] + metrics["test_rows"] == 120

    model_path = tmp_path / "baseline_model.joblib"
    metrics_path = tmp_path / "metrics.json"

    assert model_path.exists()
    assert metrics_path.exists()

    import joblib

    model = joblib.load(model_path)
    assert isinstance(model, Pipeline)
    assert len(model.predict(features.iloc[:5])) == 5
