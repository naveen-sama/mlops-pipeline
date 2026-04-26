"""
Training script for XGBoost on the California Housing dataset.

Usage:
    python train.py --config ../configs/xgboost.yaml
    python train.py --config ../configs/xgboost.yaml --experiment-name my-experiment
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from mlflow.models.signature import infer_signature
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    """Load and return the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as fh:
        config = yaml.safe_load(fh)
    logger.info("Loaded config from %s", config_path)
    return config


def load_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch the California Housing dataset and return (X, y)."""
    logger.info("Fetching California Housing dataset …")
    housing = fetch_california_housing(as_frame=True)
    X: pd.DataFrame = housing.frame.drop(columns=[housing.target.name])
    y: pd.Series = housing.target
    logger.info("Dataset shape: X=%s, y=%s", X.shape, y.shape)
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into train / validation / test sets."""
    data_cfg = config["data"]
    test_size: float = data_cfg["test_size"]
    val_size: float = data_cfg["val_size"]
    random_state: int = data_cfg["random_state"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # Derive validation fraction relative to the training portion
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_fraction, random_state=random_state
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a dict of regression evaluation metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mse": mse, "mape": mape}


def build_xgb_params(config: dict[str, Any]) -> dict[str, Any]:
    """Build the XGBoost parameter dict from config, excluding sklearn-level keys."""
    hp = config["hyperparameters"]
    xgb_params = {k: v for k, v in hp.items() if k not in ("n_estimators", "random_state", "eval_metric")}
    xgb_params["seed"] = hp.get("random_state", 42)
    return xgb_params


def log_feature_importance(
    model: xgb.XGBRegressor, feature_names: list[str], run_id: str
) -> None:
    """Log feature importances as an MLflow artifact (CSV)."""
    importances = model.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    tmp_path = Path("/tmp") / f"feature_importance_{run_id[:8]}.csv"
    fi_df.to_csv(tmp_path, index=False)
    mlflow.log_artifact(str(tmp_path), artifact_path="feature_importance")
    logger.info("Feature importance logged (%d features).", len(feature_names))


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(config_path: str, experiment_name: str | None = None) -> str:
    """
    Full training pipeline.

    Returns
    -------
    str
        The MLflow run ID of the completed training run.
    """
    config = load_config(config_path)

    # --- MLflow setup -------------------------------------------------------
    mlflow_cfg = config["mlflow"]
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", mlflow_cfg["experiment_name"]
    )
    mlflow.set_experiment(exp_name)

    # --- Data ---------------------------------------------------------------
    X, y = load_data(config)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)

    # Persist reference data for drift detection later
    ref_data_dir = Path(__file__).parent.parent / "data"
    ref_data_dir.mkdir(parents=True, exist_ok=True)
    ref_df = X_train.copy()
    ref_df[config["data"]["target_column"]] = y_train.values
    ref_df.to_parquet(ref_data_dir / "reference.parquet", index=False)
    logger.info("Reference data saved to %s", ref_data_dir / "reference.parquet")

    # --- Training -----------------------------------------------------------
    hp = config["hyperparameters"]
    training_cfg = config["training"]

    model = xgb.XGBRegressor(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"],
        reg_alpha=hp["reg_alpha"],
        reg_lambda=hp["reg_lambda"],
        objective=hp["objective"],
        eval_metric=hp["eval_metric"],
        random_state=hp["random_state"],
        early_stopping_rounds=training_cfg["early_stopping_rounds"],
        verbosity=1,
    )

    logger.info("Starting XGBoost training …")
    t0 = time.time()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run ID: %s", run_id)

        # Log all hyperparameters
        flat_params: dict[str, Any] = {
            **hp,
            "experiment_name": exp_name,
            "tracking_uri": tracking_uri,
            "test_size": config["data"]["test_size"],
            "val_size": config["data"]["val_size"],
            "data_random_state": config["data"]["random_state"],
        }
        mlflow.log_params(flat_params)

        # Tags
        mlflow.set_tags(
            {
                "framework": "xgboost",
                "dataset": "california_housing",
                "task": "regression",
                "python_version": sys.version.split()[0],
                "xgboost_version": xgb.__version__,
            }
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=training_cfg["verbose_eval"],
        )

        training_time = time.time() - t0
        logger.info("Training complete in %.1f s", training_time)

        # Evaluate on val + test
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        val_metrics = {f"val_{k}": v for k, v in compute_metrics(y_val.values, val_preds).items()}
        test_metrics = {f"test_{k}": v for k, v in compute_metrics(y_test.values, test_preds).items()}
        train_preds = model.predict(X_train)
        train_metrics = {f"train_{k}": v for k, v in compute_metrics(y_train.values, train_preds).items()}

        all_metrics = {**train_metrics, **val_metrics, **test_metrics, "training_time_s": training_time}
        mlflow.log_metrics(all_metrics)

        for split, metrics in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
            logger.info(
                "%s — RMSE=%.4f  MAE=%.4f  R²=%.4f",
                split,
                metrics.get(f"{split.lower()}_rmse", metrics.get("train_rmse", 0)),
                metrics.get(f"{split.lower()}_mae", metrics.get("train_mae", 0)),
                metrics.get(f"{split.lower()}_r2", metrics.get("train_r2", 0)),
            )

        # Log best iteration
        if hasattr(model, "best_iteration"):
            mlflow.log_metric("best_iteration", model.best_iteration)

        # Feature importance artifact
        if training_cfg.get("log_feature_importance", True):
            log_feature_importance(model, list(X_train.columns), run_id)

        # Infer model signature
        sample_input = X_test.iloc[:5]
        sample_output = model.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)

        # Log and register model
        model_name = os.getenv("MODEL_NAME", mlflow_cfg["model_name"])
        artifact_path = mlflow_cfg["artifact_path"]

        mlflow.xgboost.log_model(
            model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=sample_input,
            registered_model_name=model_name,
        )
        logger.info("Model logged and registered as '%s'.", model_name)

        # Apply tags to registered model
        client = mlflow.tracking.MlflowClient()
        registered_versions = client.search_model_versions(f"name='{model_name}'")
        if registered_versions:
            latest = max(registered_versions, key=lambda v: int(v.version))
            for tag_key, tag_val in mlflow_cfg.get("registered_model_tags", {}).items():
                client.set_model_version_tag(model_name, latest.version, tag_key, tag_val)
            client.set_model_version_tag(model_name, latest.version, "run_id", run_id)
            client.set_model_version_tag(
                model_name, latest.version, "test_rmse", str(round(test_metrics["test_rmse"], 5))
            )
            logger.info("Tags applied to model version %s.", latest.version)

    logger.info("Run %s finished successfully.", run_id)
    return run_id


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost on California Housing")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "xgboost.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override MLflow experiment name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_id = train(args.config, args.experiment_name)
    print(f"\nCompleted. MLflow run ID: {run_id}")
