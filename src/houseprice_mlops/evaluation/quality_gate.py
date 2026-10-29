import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MAX_MAE = 0.60
MAX_RMSE = 0.85
def evaluate_model() -> bool:
    """Return whether the model passes the quality gate."""
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    model_path = ARTIFACTS_DIR / "baseline_model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    for name in ("mae", "rmse"):
        value = metrics.get(name)

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid metric: {name}={value!r}")

    passed = metrics["mae"] < MAX_MAE and metrics["rmse"] < MAX_RMSE

    print(f"MAE:  {metrics['mae']:.4f} (limit: < {MAX_MAE})")
    print(f"RMSE: {metrics['rmse']:.4f} (limit: < {MAX_RMSE})")

    if passed:
        print("QUALITY GATE: PASSED")
    else:
        print("QUALITY GATE: FAILED")

    return passed


if __name__ == "__main__":
    try:
        passed = evaluate_model()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"QUALITY GATE ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0 if passed else 1)