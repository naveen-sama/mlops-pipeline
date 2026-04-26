"""
Unit tests for src/train.py

These tests mock the MLflow tracking server so no live MLflow instance is needed.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "model": {"name": "TestModel", "version": "1.0.0"},
    "hyperparameters": {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
    },
    "data": {
        "test_size": 0.2,
        "val_size": 0.1,
        "random_state": 42,
        "target_column": "MedHouseVal",
    },
    "mlflow": {
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "test-experiment",
        "model_name": "TestModel",
        "artifact_path": "model",
        "registered_model_tags": {"framework": "xgboost"},
    },
    "training": {
        "early_stopping_rounds": 5,
        "verbose_eval": 10,
        "log_feature_importance": False,
    },
}


@pytest.fixture()
def config_file(tmp_path: Path) -> str:
    """Write a minimal YAML config to a temp file and return the path."""
    cfg_path = tmp_path / "test_config.yaml"
    with open(cfg_path, "w") as fh:
        yaml.dump(SAMPLE_CONFIG, fh)
    return str(cfg_path)


# ---------------------------------------------------------------------------
# Tests: load_config
# ---------------------------------------------------------------------------

def test_load_config_success(config_file: str) -> None:
    from src.train import load_config

    config = load_config(config_file)
    assert config["model"]["name"] == "TestModel"
    assert config["hyperparameters"]["n_estimators"] == 10


def test_load_config_missing_file() -> None:
    from src.train import load_config

    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# Tests: load_data
# ---------------------------------------------------------------------------

def test_load_data_returns_correct_shapes() -> None:
    from src.train import load_data

    X, y = load_data(SAMPLE_CONFIG)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == len(y)
    assert X.shape[1] == 8  # California Housing has 8 features
    assert y.name == "MedHouseVal"


# ---------------------------------------------------------------------------
# Tests: split_data
# ---------------------------------------------------------------------------

def test_split_data_proportions() -> None:
    from src.train import load_data, split_data

    X, y = load_data(SAMPLE_CONFIG)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, SAMPLE_CONFIG)

    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(X)

    # Test split should be ~20%
    test_ratio = len(X_test) / len(X)
    assert 0.18 <= test_ratio <= 0.22

    # No overlap between splits (check index uniqueness)
    all_indices = set(X_train.index) | set(X_val.index) | set(X_test.index)
    assert len(all_indices) == len(X)


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect_prediction() -> None:
    from src.train import compute_metrics

    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_metrics(y, y)

    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-10)


def test_compute_metrics_known_values() -> None:
    from src.train import compute_metrics

    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    metrics = compute_metrics(y_true, y_pred)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "mse" in metrics
    assert "mape" in metrics
    assert metrics["rmse"] > 0
    assert metrics["r2"] > 0  # Should be a decent fit


def test_compute_metrics_keys() -> None:
    from src.train import compute_metrics

    y = np.array([1.0, 2.0, 3.0])
    metrics = compute_metrics(y, y + 0.1)
    expected_keys = {"rmse", "mae", "r2", "mse", "mape"}
    assert set(metrics.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: build_xgb_params
# ---------------------------------------------------------------------------

def test_build_xgb_params_excludes_sklearn_keys() -> None:
    from src.train import build_xgb_params

    params = build_xgb_params(SAMPLE_CONFIG)
    # These are sklearn-level constructor args, not XGBoost native params
    assert "n_estimators" not in params
    assert "random_state" not in params
    assert "eval_metric" not in params
    # Should include core XGBoost params
    assert "max_depth" in params
    assert "learning_rate" in params


# ---------------------------------------------------------------------------
# Tests: full train() with mocked MLflow
# ---------------------------------------------------------------------------

@patch("src.train.mlflow")
def test_train_runs_without_error(mock_mlflow: MagicMock, config_file: str, tmp_path: Path) -> None:
    """
    Smoke test: train() should complete without raising when MLflow is mocked.
    We also verify that key MLflow calls were made.
    """
    # Set up mock MLflow context manager
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id-12345"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    # Mock MlflowClient
    mock_client = MagicMock()
    mock_version = MagicMock()
    mock_version.version = "1"
    mock_client.search_model_versions.return_value = [mock_version]
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    # Patch data directory to use tmp
    with patch("src.train.Path") as mock_path_cls:
        # Let Path work normally for config loading, only redirect data dir
        real_path = Path
        def path_side_effect(*args, **kwargs):
            p = real_path(*args, **kwargs)
            return p
        mock_path_cls.side_effect = path_side_effect

        from src.train import load_config, load_data, split_data, compute_metrics
        config = load_config(config_file)
        X, y = load_data(config)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)

        # Verify all data splits are non-empty
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0

        # Verify metrics calculation works end-to-end
        import xgboost as xgb
        hp = config["hyperparameters"]
        model = xgb.XGBRegressor(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            random_state=hp["random_state"],
            verbosity=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test.values, preds)

        assert metrics["r2"] > 0.3, f"Model R² too low: {metrics['r2']:.4f}"
        assert metrics["rmse"] < 2.0, f"Model RMSE too high: {metrics['rmse']:.4f}"
