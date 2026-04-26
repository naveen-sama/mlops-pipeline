"""
Data drift detector using the Evidently library.

Compares a reference dataset against a current (production) dataset using
Evidently's DataDriftPreset. Saves an HTML report and returns True if drift
is detected above the configured threshold.

Usage:
    python drift_detector.py --reference data/reference.parquet --current data/current.parquet
    python drift_detector.py --reference data/reference.parquet --current data/current.parquet \
        --threshold 0.5 --report-path reports/drift_report.html
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_presets import DataDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default drift share threshold — if more than this fraction of features drift,
# the function returns True (drift detected).
DEFAULT_DRIFT_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load a Parquet or CSV dataset from `path`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .parquet or .csv")

    logger.info("Loaded dataset from %s — shape: %s", path, df.shape)
    return df


def generate_synthetic_current_data(
    reference: pd.DataFrame,
    drift_magnitude: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic 'current' dataset from the reference by adding controlled noise.
    Used for demonstration / testing when no current data file is available.
    """
    rng = np.random.default_rng(random_state)
    current = reference.copy()

    numeric_cols = current.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        col_std = current[col].std()
        noise = rng.normal(loc=drift_magnitude * col_std, scale=col_std * 0.1, size=len(current))
        current[col] = current[col] + noise

    logger.info(
        "Generated synthetic current dataset with drift_magnitude=%.2f.", drift_magnitude
    )
    return current


# ---------------------------------------------------------------------------
# Core drift detection
# ---------------------------------------------------------------------------

def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: str | None = None,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    report_path: str | None = None,
    return_report_dict: bool = False,
) -> bool | tuple[bool, dict[str, Any]]:
    """
    Run Evidently DataDriftPreset and determine if significant drift occurred.

    Parameters
    ----------
    reference_data : pd.DataFrame
        Training / baseline dataset.
    current_data : pd.DataFrame
        Recent production / scoring dataset.
    target_column : str | None
        Name of the target column (excluded from feature drift calculation).
    drift_threshold : float
        Fraction of drifted features that triggers a True return value.
        Default is 0.3 (30 % of features).
    report_path : str | None
        If provided, the HTML report is saved to this path.
    return_report_dict : bool
        If True, also return the raw Evidently result dict.

    Returns
    -------
    bool  (or tuple[bool, dict] when return_report_dict=True)
        True  → drift detected above threshold.
        False → data is within acceptable bounds.
    """
    # Align columns between reference and current
    common_cols = [c for c in reference_data.columns if c in current_data.columns]
    if target_column and target_column in common_cols:
        feature_cols = [c for c in common_cols if c != target_column]
    else:
        feature_cols = common_cols

    ref = reference_data[common_cols].copy()
    cur = current_data[common_cols].copy()

    logger.info(
        "Running drift detection — %d feature columns, %d reference rows, %d current rows.",
        len(feature_cols),
        len(ref),
        len(cur),
    )

    # Column mapping
    column_mapping = ColumnMapping()
    if target_column and target_column in common_cols:
        column_mapping.target = target_column
    column_mapping.numerical_features = (
        ref[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    )
    column_mapping.categorical_features = (
        ref[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    )

    # Build Evidently report
    report = Report(
        metrics=[
            DataDriftPreset(drift_share=drift_threshold),
            DatasetDriftMetric(drift_share_threshold=drift_threshold),
            DatasetMissingValuesMetric(),
        ]
    )

    report.run(
        reference_data=ref,
        current_data=cur,
        column_mapping=column_mapping,
    )

    # Save HTML report
    if report_path:
        output_path = Path(report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_path))
        logger.info("HTML report saved to: %s", output_path)

    # Extract results
    report_dict = report.as_dict()
    drift_detected = _parse_drift_result(report_dict, drift_threshold, len(feature_cols))

    logger.info(
        "Drift detection result: %s (threshold=%.2f)",
        "DRIFT DETECTED" if drift_detected else "No significant drift",
        drift_threshold,
    )

    if return_report_dict:
        return drift_detected, report_dict
    return drift_detected


def _parse_drift_result(
    report_dict: dict[str, Any],
    drift_threshold: float,
    n_features: int,
) -> bool:
    """
    Parse Evidently's output dict and determine if drift exceeded the threshold.
    Returns True when drift is detected.
    """
    try:
        metrics_list = report_dict.get("metrics", [])
        for metric_entry in metrics_list:
            metric_id = metric_entry.get("metric", "")

            if "DatasetDriftMetric" in metric_id:
                result = metric_entry.get("result", {})
                dataset_drift = result.get("dataset_drift", False)
                share_drifted = result.get("share_drifted_columns", 0.0)
                n_drifted = result.get("number_of_drifted_columns", 0)

                logger.info(
                    "DatasetDriftMetric — dataset_drift=%s, drifted_features=%d/%d (%.1f%%)",
                    dataset_drift,
                    n_drifted,
                    n_features,
                    share_drifted * 100,
                )
                return bool(dataset_drift) or (share_drifted > drift_threshold)

        # Fallback: check DataDriftPreset results
        for metric_entry in metrics_list:
            metric_id = metric_entry.get("metric", "")
            if "DataDriftTable" in metric_id or "DatasetDrift" in metric_id:
                result = metric_entry.get("result", {})
                share = result.get("share_drifted_columns", result.get("drift_share", 0.0))
                if share > drift_threshold:
                    logger.info(
                        "Drift share %.2f exceeds threshold %.2f → drift detected.",
                        share,
                        drift_threshold,
                    )
                    return True

    except Exception as exc:
        logger.warning("Could not parse Evidently report dict: %s", exc)

    return False


# ---------------------------------------------------------------------------
# Summary & MLflow integration
# ---------------------------------------------------------------------------

def log_drift_to_mlflow(
    drift_detected: bool,
    report_dict: dict[str, Any],
    model_name: str,
    report_path: str | None,
    tracking_uri: str = "http://localhost:5000",
) -> None:
    """Optionally log drift metrics to an MLflow run."""
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("data-drift-monitoring")

        with mlflow.start_run(run_name=f"drift_check_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"):
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("drift_detected", str(drift_detected))
            mlflow.log_metric("drift_detected_flag", int(drift_detected))

            # Attempt to extract per-feature drift scores
            try:
                metrics_list = report_dict.get("metrics", [])
                for metric_entry in metrics_list:
                    if "DatasetDriftMetric" in metric_entry.get("metric", ""):
                        result = metric_entry.get("result", {})
                        mlflow.log_metric("share_drifted_columns", result.get("share_drifted_columns", 0))
                        mlflow.log_metric("number_of_drifted_columns", result.get("number_of_drifted_columns", 0))
            except Exception:
                pass

            if report_path and Path(report_path).exists():
                mlflow.log_artifact(report_path, artifact_path="drift_reports")

        logger.info("Drift metrics logged to MLflow.")
    except Exception as exc:
        logger.warning("Could not log drift to MLflow: %s", exc)


def print_drift_summary(report_dict: dict[str, Any]) -> None:
    """Print a concise per-feature drift summary to stdout."""
    print("\n" + "=" * 60)
    print("DATA DRIFT SUMMARY")
    print("=" * 60)
    try:
        metrics_list = report_dict.get("metrics", [])
        for metric_entry in metrics_list:
            if "DataDriftTable" in metric_entry.get("metric", ""):
                result = metric_entry.get("result", {})
                columns = result.get("drift_by_columns", {})
                if columns:
                    print(f"\n{'Feature':<30} {'Drift Score':>12} {'Drifted':>8}")
                    print("-" * 52)
                    for col_name, col_result in sorted(columns.items()):
                        score = col_result.get("drift_score", 0.0)
                        drifted = col_result.get("drift_detected", False)
                        marker = " !" if drifted else ""
                        print(f"{col_name:<30} {score:>12.4f} {str(drifted):>8}{marker}")
    except Exception as exc:
        logger.debug("Could not print per-feature summary: %s", exc)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect data drift using Evidently between reference and current datasets"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "reference.parquet"),
        help="Path to reference (training) dataset (.parquet or .csv)",
    )
    parser.add_argument(
        "--current",
        type=str,
        default=None,
        help="Path to current (production) dataset. If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DRIFT_THRESHOLD,
        help=f"Drift share threshold (default: {DEFAULT_DRIFT_THRESHOLD})",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(Path(__file__).parent.parent / "reports" / "drift_report.html"),
        help="Output path for the HTML drift report",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="MedHouseVal",
        help="Name of the target column to exclude from feature drift",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        default=False,
        help="Log drift metrics to MLflow",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking URI (when --log-mlflow is set)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("MODEL_NAME", "MLOpsDemoModel"),
        help="Model name tag for MLflow logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load reference data
    reference_df = load_dataset(args.reference)

    # Load or generate current data
    if args.current:
        current_df = load_dataset(args.current)
    else:
        logger.warning(
            "No --current dataset provided. Generating synthetic drifted data for demonstration."
        )
        current_df = generate_synthetic_current_data(reference_df, drift_magnitude=0.25)

    # Run drift detection
    drift_detected, report_dict = detect_drift(
        reference_data=reference_df,
        current_data=current_df,
        target_column=args.target_column,
        drift_threshold=args.threshold,
        report_path=args.report_path,
        return_report_dict=True,
    )

    print_drift_summary(report_dict)

    if args.log_mlflow:
        log_drift_to_mlflow(
            drift_detected=drift_detected,
            report_dict=report_dict,
            model_name=args.model_name,
            report_path=args.report_path,
            tracking_uri=args.tracking_uri,
        )

    print(f"\nResult: {'DRIFT DETECTED' if drift_detected else 'No significant drift detected'}")
    print(f"Report saved to: {args.report_path}")

    # Exit with non-zero code if drift is detected (useful in CI pipelines)
    raise SystemExit(1 if drift_detected else 0)
