"""
Model promotion CLI — transitions a registered MLflow model version to a
target stage (Staging / Production / Archived) after running validation
checks.

Usage:
    python promote.py --model-name MLOpsDemoModel --version 1 --stage Production
    python promote.py --model-name MLOpsDemoModel --version 2 --stage Staging --no-archive-previous
    python promote.py --model-name MLOpsDemoModel --version 1 --stage Archived
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Allowed stage values (MLflow canonical names)
VALID_STAGES = {"Staging", "Production", "Archived", "None"}

# Minimum quality thresholds for promotion to Production
PRODUCTION_QUALITY_GATES: dict[str, float] = {
    "eval_rmse": 1.5,   # RMSE must be below this
    "eval_r2": 0.4,     # R² must be above this (checked separately)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri)


def get_model_version(
    client: MlflowClient, model_name: str, version: str
) -> Any:
    """Fetch and return the ModelVersion object; raise if not found."""
    try:
        mv = client.get_model_version(model_name, version)
    except Exception as exc:
        raise ValueError(
            f"Model '{model_name}' version '{version}' not found in registry. "
            f"Original error: {exc}"
        ) from exc
    return mv


def check_model_exists(client: MlflowClient, model_name: str) -> None:
    """Raise ValueError if the registered model does not exist."""
    try:
        client.get_registered_model(model_name)
    except Exception as exc:
        raise ValueError(
            f"Registered model '{model_name}' not found. "
            f"Original error: {exc}"
        ) from exc


def validate_for_production(
    client: MlflowClient,
    model_name: str,
    version: str,
) -> tuple[bool, list[str]]:
    """
    Run quality gate checks required before promoting to Production.

    Returns
    -------
    (passed: bool, messages: list[str])
        `passed` is True only when ALL checks pass.
    """
    messages: list[str] = []
    passed = True

    mv = get_model_version(client, model_name, version)
    tags = mv.tags or {}

    logger.info("Running production quality gates for %s v%s …", model_name, version)

    # --- Check 1: evaluation run must have been performed ------------------
    eval_run_id = tags.get("eval_run_id")
    if not eval_run_id:
        msg = "No evaluation run found (tag 'eval_run_id' missing). Run evaluate.py first."
        messages.append(msg)
        passed = False
        logger.warning(msg)
    else:
        logger.info("Evaluation run ID present: %s", eval_run_id)

    # --- Check 2: RMSE threshold -------------------------------------------
    eval_rmse_str = tags.get("eval_rmse")
    if eval_rmse_str is not None:
        eval_rmse = float(eval_rmse_str)
        threshold = PRODUCTION_QUALITY_GATES["eval_rmse"]
        if eval_rmse > threshold:
            msg = f"RMSE {eval_rmse:.4f} exceeds production threshold {threshold:.4f}."
            messages.append(msg)
            passed = False
            logger.warning(msg)
        else:
            logger.info("RMSE gate passed: %.4f <= %.4f", eval_rmse, threshold)
    else:
        msg = "Tag 'eval_rmse' missing — cannot verify RMSE quality gate."
        messages.append(msg)
        passed = False
        logger.warning(msg)

    # --- Check 3: R² threshold ---------------------------------------------
    eval_r2_str = tags.get("eval_r2")
    if eval_r2_str is not None:
        eval_r2 = float(eval_r2_str)
        threshold = PRODUCTION_QUALITY_GATES["eval_r2"]
        if eval_r2 < threshold:
            msg = f"R² {eval_r2:.4f} is below production threshold {threshold:.4f}."
            messages.append(msg)
            passed = False
            logger.warning(msg)
        else:
            logger.info("R² gate passed: %.4f >= %.4f", eval_r2, threshold)
    else:
        msg = "Tag 'eval_r2' missing — cannot verify R² quality gate."
        messages.append(msg)
        passed = False
        logger.warning(msg)

    # --- Check 4: model run must be in FINISHED state ----------------------
    run_id = mv.run_id
    if run_id:
        try:
            run = client.get_run(run_id)
            if run.info.status != "FINISHED":
                msg = f"Training run {run_id} is not FINISHED (status: {run.info.status})."
                messages.append(msg)
                passed = False
                logger.warning(msg)
            else:
                logger.info("Training run status: FINISHED.")
        except Exception as exc:
            msg = f"Could not retrieve training run {run_id}: {exc}"
            messages.append(msg)
            logger.warning(msg)

    return passed, messages


def archive_current_production(
    client: MlflowClient, model_name: str, promoting_version: str
) -> list[str]:
    """Move all current Production versions (except the one being promoted) to Archived."""
    archived: list[str] = []
    production_versions = client.search_model_versions(
        f"name='{model_name}'",
    )
    for mv in production_versions:
        if mv.current_stage == "Production" and mv.version != promoting_version:
            logger.info(
                "Archiving previous Production model %s v%s …", model_name, mv.version
            )
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Archived",
                archive_existing_versions=False,
            )
            client.set_model_version_tag(
                model_name, mv.version, "archived_at",
                datetime.now(tz=timezone.utc).isoformat()
            )
            archived.append(mv.version)
    return archived


# ---------------------------------------------------------------------------
# Core promotion function
# ---------------------------------------------------------------------------

def promote_model(
    model_name: str,
    version: str,
    stage: str,
    tracking_uri: str,
    archive_previous: bool = True,
    force: bool = False,
) -> None:
    """
    Transition a model version to the specified stage with validation.

    Parameters
    ----------
    model_name : str
        Registered model name.
    version : str
        Version number as a string.
    stage : str
        Target stage: Staging | Production | Archived | None
    tracking_uri : str
        MLflow tracking server URI.
    archive_previous : bool
        When True, archive existing Production versions when promoting to Production.
    force : bool
        Skip quality gate checks (use with caution).
    """
    stage = stage.strip().capitalize()
    if stage == "Staging":
        stage = "Staging"
    elif stage == "Production":
        stage = "Production"
    elif stage == "Archived":
        stage = "Archived"
    elif stage == "None":
        stage = "None"

    if stage not in VALID_STAGES:
        raise ValueError(
            f"Invalid stage '{stage}'. Must be one of: {sorted(VALID_STAGES)}"
        )

    client = build_client(tracking_uri)

    # Verify model + version exist
    check_model_exists(client, model_name)
    mv = get_model_version(client, model_name, version)

    logger.info(
        "Preparing to promote '%s' v%s  %s → %s",
        model_name,
        version,
        mv.current_stage,
        stage,
    )

    if mv.current_stage == stage:
        logger.info("Model is already in stage '%s'. Nothing to do.", stage)
        return

    # Quality gates for Production
    if stage == "Production" and not force:
        passed, messages = validate_for_production(client, model_name, version)
        if not passed:
            logger.error("Production quality gates FAILED:")
            for msg in messages:
                logger.error("  • %s", msg)
            logger.error(
                "Use --force to bypass checks (not recommended for production workloads)."
            )
            sys.exit(1)
        logger.info("All production quality gates PASSED.")

    # Archive existing Production versions if needed
    archived_versions: list[str] = []
    if stage == "Production" and archive_previous:
        archived_versions = archive_current_production(client, model_name, version)
        if archived_versions:
            logger.info("Archived previous production versions: %s", archived_versions)

    # Perform the transition
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=False,  # We handle this manually above
    )

    # Apply promotion audit tags
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    client.set_model_version_tag(model_name, version, f"promoted_to_{stage.lower()}_at", now_iso)
    client.set_model_version_tag(model_name, version, "current_stage", stage)

    # Add a description note
    description = (
        f"Promoted to {stage} on {now_iso[:10]} by MLOps pipeline. "
        f"Previous Production versions archived: {archived_versions or 'none'}."
    )
    client.update_model_version(name=model_name, version=version, description=description)

    logger.info(
        "Successfully promoted '%s' v%s to '%s'.",
        model_name,
        version,
        stage,
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a registered MLflow model to a target stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python promote.py --model-name MLOpsDemoModel --version 1 --stage Production
  python promote.py --model-name MLOpsDemoModel --version 2 --stage Staging
  python promote.py --model-name MLOpsDemoModel --version 1 --stage Archived
  python promote.py --model-name MLOpsDemoModel --version 3 --stage Production --force
        """,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Registered model name in MLflow Model Registry",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version number to promote",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["Staging", "Production", "Archived", "None"],
        help="Target model stage",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking server URI (or set MLFLOW_TRACKING_URI env var)",
    )
    parser.add_argument(
        "--no-archive-previous",
        action="store_true",
        default=False,
        help="Do not archive existing Production versions when promoting to Production",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Skip quality gate validation checks (use with caution)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    promote_model(
        model_name=args.model_name,
        version=args.version,
        stage=args.stage,
        tracking_uri=args.tracking_uri,
        archive_previous=not args.no_archive_previous,
        force=args.force,
    )
