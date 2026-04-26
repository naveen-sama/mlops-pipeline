"""
Evaluation script — loads a registered MLflow model by name + version,
runs inference on the test split, and logs evaluation metrics back to
a dedicated MLflow evaluation run linked to the original training run.

Usage:
    python evaluate.py --model-name MLOpsDemoModel --model-version 1
    python evaluate.py --model-name MLOpsDemoModel --model-version 1 --config ../configs/xgboost.yaml
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as fh:
        return yaml.safe_load(fh)


def get_test_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Recreate the exact same test split used during training."""
    housing = fetch_california_housing(as_frame=True)
    X = housing.frame.drop(columns=[housing.target.name])
    y = housing.target

    data_cfg = config["data"]
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"]
    )
    logger.info("Test set — rows: %d, features: %d", len(X_test), X_test.shape[1])
    return X_test, y_test


def compute_full_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Comprehensive regression metric suite."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
    }


def get_model_training_run_id(
    client: MlflowClient, model_name: str, model_version: str
) -> str | None:
    """Return the training run_id stored in the model version tags, if present."""
    try:
        mv = client.get_model_version(model_name, model_version)
        return mv.tags.get("run_id")
    except Exception:
        return None


def load_model_uri(model_name: str, model_version: str) -> str:
    """Build the MLflow model URI for a registered model version."""
    return f"models:/{model_name}/{model_version}"


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str,
    model_version: str,
    config_path: str,
    threshold_rmse: float | None = None,
) -> dict[str, float]:
    """
    Load the registered model, run evaluation, log results to MLflow.

    Parameters
    ----------
    model_name : str
        Registered model name in MLflow Model Registry.
    model_version : str
        Version number as a string (e.g. "1").
    config_path : str
        Path to the YAML config file.
    threshold_rmse : float | None
        If provided, raises RuntimeError when RMSE exceeds this value.

    Returns
    -------
    dict[str, float]
        Dictionary of evaluation metrics.
    """
    config = load_config(config_path)
    mlflow_cfg = config["mlflow"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    # Resolve model version if "latest" alias is used
    if model_version.lower() in ("latest", "none", ""):
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        model_version = str(max(int(v.version) for v in versions))
        logger.info("Resolved to latest version: %s", model_version)

    model_uri = load_model_uri(model_name, model_version)
    logger.info("Loading model from URI: %s", model_uri)

    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Model loaded successfully.")

    X_test, y_test = get_test_data(config)

    logger.info("Running predictions on %d samples …", len(X_test))
    y_pred = model.predict(X_test)

    metrics = compute_full_metrics(y_test.values, np.array(y_pred))

    # Pretty-print metrics
    logger.info("Evaluation Results:")
    for metric_name, metric_val in metrics.items():
        logger.info("  %-25s %.6f", metric_name, metric_val)

    # Threshold guard
    if threshold_rmse is not None and metrics["rmse"] > threshold_rmse:
        raise RuntimeError(
            f"Evaluation RMSE {metrics['rmse']:.4f} exceeds threshold {threshold_rmse:.4f}. "
            "Model does not meet quality gate."
        )

    # Retrieve the training run_id to link this eval run
    training_run_id = get_model_training_run_id(client, model_name, model_version)

    # Log evaluation metrics to a new MLflow run
    experiment_name = mlflow_cfg.get("experiment_name", "california-housing-xgboost")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"eval_{model_name}_v{model_version}") as run:
        eval_run_id = run.info.run_id

        mlflow.set_tags(
            {
                "evaluation_type": "offline",
                "model_name": model_name,
                "model_version": model_version,
                "linked_training_run": training_run_id or "unknown",
                "dataset": "california_housing_test",
            }
        )
        mlflow.log_params(
            {
                "model_name": model_name,
                "model_version": model_version,
                "test_size": config["data"]["test_size"],
                "n_test_samples": len(X_test),
            }
        )

        # Log all metrics prefixed with "eval_"
        prefixed = {f"eval_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed)

        # Save a JSON summary artifact
        summary = {
            "model_name": model_name,
            "model_version": model_version,
            "model_uri": model_uri,
            "n_test_samples": len(X_test),
            "metrics": metrics,
            "training_run_id": training_run_id,
            "eval_run_id": eval_run_id,
        }
        summary_path = Path("/tmp") / f"eval_summary_{eval_run_id[:8]}.json"
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        mlflow.log_artifact(str(summary_path), artifact_path="evaluation")

        # Residuals artifact
        residuals_df = X_test.copy()
        residuals_df["y_true"] = y_test.values
        residuals_df["y_pred"] = y_pred
        residuals_df["residual"] = residuals_df["y_true"] - residuals_df["y_pred"]
        residuals_path = Path("/tmp") / f"residuals_{eval_run_id[:8]}.csv"
        residuals_df.to_csv(residuals_path, index=False)
        mlflow.log_artifact(str(residuals_path), artifact_path="evaluation")

        logger.info("Evaluation run logged. Run ID: %s", eval_run_id)

    # Annotate the model version with the evaluation run
    try:
        client.set_model_version_tag(
            model_name, model_version, "eval_run_id", eval_run_id
        )
        client.set_model_version_tag(
            model_name, model_version, "eval_rmse", str(round(metrics["rmse"], 5))
        )
        client.set_model_version_tag(
            model_name, model_version, "eval_r2", str(round(metrics["r2"], 5))
        )
    except Exception as exc:
        logger.warning("Could not set model version tags: %s", exc)

    return metrics


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a registered MLflow model on the California Housing test set"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("MODEL_NAME", "MLOpsDemoModel"),
        help="Registered model name in MLflow",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="latest",
        help="Model version to evaluate (number or 'latest')",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "xgboost.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--threshold-rmse",
        type=float,
        default=None,
        help="Fail evaluation if RMSE exceeds this value",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = evaluate(
        model_name=args.model_name,
        model_version=args.model_version,
        config_path=args.config,
        threshold_rmse=args.threshold_rmse,
    )
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.6f}")
