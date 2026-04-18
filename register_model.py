"""
register_model.py
-----------------
One-time script: logs the Rail Madad MLflow model, registers it in the
Databricks Unity Catalog Model Registry, and creates a Model Serving endpoint.

Run once from your terminal or a Databricks notebook:
    python3 register_model.py

Prerequisites:
    pip install mlflow databricks-sdk
    databricks configure --token        # (already done if verify.py works)
"""

from __future__ import annotations

import logging
import os

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("register_model")

# ── Config ────────────────────────────────────────────────────────────────────
# Unity Catalog path:  <catalog>.<schema>.<model_name>
# Override with env var UC_MODEL_NAME if you use a different catalog/schema.
UC_MODEL_NAME    = os.environ.get(
    "UC_MODEL_NAME",
    "workspace.default.rail_madad_complaint_generator",
)
ENDPOINT_NAME    = "rail-madad-api"
RUN_NAME         = "rail-madad-v1"
# Databricks workspace path for the MLflow experiment.
# Override with env var EXPERIMENT_PATH, otherwise defaults to the caller's
# /Users/<me>/rail_madad_experiment.
EXPERIMENT_PATH  = os.environ.get("EXPERIMENT_PATH")
# Code files that the serving container must have
CODE_PATHS       = [
    "complaint_engine.py",
    "sarvam_client.py",
    "rail_madad_model.py",
    "chat_agent.py",
]


def log_and_register() -> str:
    """
    Log the model to MLflow, register in Unity Catalog, alias as Champion.
    Returns: registered model version string.
    """
    # Point MLflow at the Databricks-hosted tracking server
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")   # Use Unity Catalog registry

    # Ensure an experiment exists (Databricks tracking requires a workspace path)
    experiment_path = EXPERIMENT_PATH
    if not experiment_path:
        user_name = WorkspaceClient().current_user.me().user_name
        experiment_path = f"/Users/{user_name}/rail_madad_experiment"
    logger.info("Using MLflow experiment: %s", experiment_path)
    mlflow.set_experiment(experiment_path)

    from rail_madad_model import RailMadadModel  # noqa: PLC0415

    logger.info("Logging MLflow model (run: %s) ...", RUN_NAME)
    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.pyfunc.log_model(
            artifact_path="rail_madad_model",
            python_model=RailMadadModel(),
            # Bundle source files so the serving container can import them
            code_paths=CODE_PATHS,
            pip_requirements=[
                "databricks-sdk>=0.20.0",
                "databricks-sql-connector>=3.0.0",
                "databricks-vectorsearch>=0.40",
                "requests>=2.28.0",
                "pandas>=1.5.0",
                "mlflow>=2.10.0",
            ],
            # Declare input/output signature for the serving UI and validation.
            # Columns cover BOTH request types — complaint fields + ticket fields.
            # Clients supply the columns relevant to their request; the model
            # dispatches based on whether `ticket_image_base64` is set.
            signature=mlflow.models.infer_signature(
                model_input={
                    "request_type":         "chat",
                    "query":                "ट्रेन 12 घंटे लेट है",
                    "train_number":         "12345",
                    "pnr":                  "1234567890",
                    "issue_type":           "delay",
                    "language_code":        "hi-IN",
                    "ticket_image_base64":  "",
                    "user_id":              "",
                },
                model_output={
                    "request_type":        "complaint",
                    "status":              "success",
                    "message":             "Complaint generated",
                    "sms_uri":             "sms:139?body=MADAD...",
                    "formatted_complaint": "Delay Train 12345 | ...",
                    "char_count":          62,
                    "detected_language":   "hi-IN",
                    "engine_notes":        "Translated from hi-IN to English.",
                    "journey_id":          "",
                    "extracted_data":      "",
                    "chat_reply":          "",
                    "tool_log":            "",
                },
            ),
        )
        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/rail_madad_model"
    logger.info("Registering model '%s' from %s ...", UC_MODEL_NAME, model_uri)

    mv = mlflow.register_model(
        model_uri=model_uri,
        name=UC_MODEL_NAME,
    )
    version = mv.version
    logger.info("Registered as version %s", version)

    # Tag as Champion so the serving endpoint can pin to alias
    client = MlflowClient()
    client.set_registered_model_alias(
        name=UC_MODEL_NAME,
        alias="Champion",
        version=version,
    )
    logger.info("Alias 'Champion' → version %s", version)

    return version


def create_serving_endpoint(version: str) -> None:
    """Create (or update) the Databricks Model Serving endpoint."""
    w = WorkspaceClient()
    existing = {e.name for e in w.serving_endpoints.list()}

    served_model = ServedModelInput(
        model_name=UC_MODEL_NAME,
        model_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        # Add environment variables for authentication.
        # DATABRICKS_SQL_HTTP_PATH is required for the ticket-upload route
        # (the pyfunc writes journey rows via the Databricks SQL connector).
        environment_vars={
            "DATABRICKS_HOST":          w.config.host,
            "DATABRICKS_TOKEN":         w.config.token,
            "DATABRICKS_SQL_HTTP_PATH": os.environ.get(
                "DATABRICKS_SQL_HTTP_PATH", ""
            ),
            "JOURNEY_TABLE": os.environ.get(
                "JOURNEY_TABLE", "workspace.default.journey"
            ),
            "TICKET_VISION_ENDPOINT": os.environ.get(
                "TICKET_VISION_ENDPOINT", "databricks-llama-4-maverick"
            ),
            # Chat path — Databricks Vector Search + Tavily web search
            "VECTOR_SEARCH_ENDPOINT": os.environ.get(
                "VECTOR_SEARCH_ENDPOINT", "try1"
            ),
            "VECTOR_SEARCH_INDEX": os.environ.get(
                "VECTOR_SEARCH_INDEX", "workspace.rail_adhikar.rules_vs_index"
            ),
            "VECTOR_SEARCH_EMBEDDING_MODEL": os.environ.get(
                "VECTOR_SEARCH_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
            "TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY", ""),
        },
    )

    config = EndpointCoreConfigInput(
        name=ENDPOINT_NAME,
        served_models=[served_model],
    )

    if ENDPOINT_NAME in existing:
        logger.info("Endpoint '%s' exists — updating ...", ENDPOINT_NAME)
        w.serving_endpoints.update_config_and_wait(
            name=ENDPOINT_NAME,
            served_models=config.served_models,
        )
    else:
        logger.info("Creating endpoint '%s' (scale-to-zero, Small) ...", ENDPOINT_NAME)
        w.serving_endpoints.create_and_wait(
            name=ENDPOINT_NAME,
            config=config,
        )

    logger.info("✅ Endpoint '%s' is live.", ENDPOINT_NAME)

    # Print a ready-to-use test snippet
    host = w.config.host.rstrip("/")
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║   Rail Madad API — ready                                        ║
╚══════════════════════════════════════════════════════════════════╝

Endpoint URL:
  {host}/serving-endpoints/{ENDPOINT_NAME}/invocations

Test (complaint) with Python:
  import requests
  from databricks.sdk import WorkspaceClient
  w = WorkspaceClient()
  resp = requests.post(
      "{host}/serving-endpoints/{ENDPOINT_NAME}/invocations",
      headers={{"Authorization": f"Bearer {{w.config.token}}"}},
      json={{
          "dataframe_records": [{{
              "query":        "ट्रेन 12 घंटे लेट है, बच्चे भूखे हैं",
              "train_number": "12345",
              "pnr":          "1234567890",
              "issue_type":   "delay"
          }}]
      }},
  )
  print(resp.json())

Test (ticket upload) with Python:
  import base64, requests
  from databricks.sdk import WorkspaceClient
  w = WorkspaceClient()
  img_b64 = base64.b64encode(open("ticket.jpg", "rb").read()).decode()
  resp = requests.post(
      "{host}/serving-endpoints/{ENDPOINT_NAME}/invocations",
      headers={{"Authorization": f"Bearer {{w.config.token}}"}},
      json={{
          "dataframe_records": [{{
              "ticket_image_base64": img_b64,
              "user_id":             "user123"
          }}]
      }},
  )
  print(resp.json())

Test (chat) with Python:
  import requests
  from databricks.sdk import WorkspaceClient
  w = WorkspaceClient()
  resp = requests.post(
      "{host}/serving-endpoints/{ENDPOINT_NAME}/invocations",
      headers={{"Authorization": f"Bearer {{w.config.token}}"}},
      json={{
          "dataframe_records": [{{
              "request_type": "chat",
              "query":        "My train was 4 hours late, can I get a refund?",
              "user_id":      "user123"
          }}]
      }},
  )
  print(resp.json())
""")


if __name__ == "__main__":
    version = log_and_register()
    create_serving_endpoint(version)
