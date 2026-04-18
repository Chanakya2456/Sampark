"""
setup_endpoint.py
-----------------
One-time setup: registers Sarvam AI as a Databricks External Model
(Custom Provider) so all LLM calls go through the Databricks serving endpoint.
Run once from your terminal or a Databricks notebook:
    python setup_endpoint.py
After this, your Databricks workspace has an endpoint called
'sarvam-30b-gateway' that:
  - Routes chat requests to https://api.sarvam.ai/v1
  - Reads the Sarvam key from Databricks Secrets (never hardcoded)
  - Logs every request/response to an inference table
  - Is queryable via w.serving_endpoints.query() in any Databricks workload
Prereq: secret must exist first:
    databricks secrets create-scope bharatbricks
    databricks secrets put-secret bharatbricks sarvam_api_key
"""
import logging
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ExternalModel,
    ExternalModelProvider,
    CustomProviderConfig,
    BearerTokenAuth,
    AiGatewayConfig,
    AiGatewayUsageTrackingConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("setup_endpoint")

ENDPOINT_NAME = "sarvam-30b-gateway"
SECRET_SCOPE  = "bharatbricks"
SECRET_KEY    = "sarvam_api_key"

def _build_config():
    return EndpointCoreConfigInput(
        name="sarvam-30b",
        served_entities=[
            ServedEntityInput(
                name="sarvam-30b",
                external_model=ExternalModel(
                    name="sarvam-30b",
                    provider=ExternalModelProvider.CUSTOM,
                    task="llm/v1/chat",
                    custom_provider_config=CustomProviderConfig(
                        custom_provider_url="https://api.sarvam.ai/v1",
                        bearer_token_auth=BearerTokenAuth(
                            token=f"{{{{secrets/{SECRET_SCOPE}/{SECRET_KEY}}}}}",
                        ),
                    ),
                ),
            )
        ]
    )


def _build_ai_gateway():
    return AiGatewayConfig(
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        # Inference table disabled — enable once you run:
        # CREATE SCHEMA main.rail_madad;
    )


def create_or_update_endpoint(w: WorkspaceClient) -> None:
    existing = {e.name: e for e in w.serving_endpoints.list()}

    if ENDPOINT_NAME in existing:
        ep = w.serving_endpoints.get(name=ENDPOINT_NAME)
        state = str(ep.state.config_update) if ep.state else "UNKNOWN"
        logger.info("Endpoint '%s' exists. State: %s", ENDPOINT_NAME, state)

        # If the endpoint is in a failed/error state, delete and recreate
        if "NOT_UPDATING" not in state and "READY" not in state:
            logger.warning(
                "Endpoint is in state '%s' — deleting and recreating...", state
            )
            w.serving_endpoints.delete(name=ENDPOINT_NAME)
            logger.info("Deleted. Waiting before recreating...")
            import time; time.sleep(5)
        else:
            logger.info("Endpoint is healthy. Updating config...")
            w.serving_endpoints.update_config_and_wait(
                name=ENDPOINT_NAME,
                served_entities=_build_config().served_entities,
            )
            logger.info("Config updated.")
            return

    logger.info("Creating endpoint '%s'...", ENDPOINT_NAME)
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=_build_config(),
        ai_gateway=_build_ai_gateway(),
    )
    logger.info("Endpoint created and ready.")


def verify_endpoint(w: WorkspaceClient) -> None:
    import requests

    # ── Full state diagnostics ───────────────────────────────────────────────
    ep = w.serving_endpoints.get(name=ENDPOINT_NAME)
    config_update_state = str(ep.state.config_update) if ep.state else "UNKNOWN"
    ready_state         = str(ep.state.ready)          if ep.state else "UNKNOWN"
    logger.info("config_update state : %s", config_update_state)
    logger.info("ready state         : %s", ready_state)

    # Log any pending config errors
    if ep.pending_config:
        logger.warning("Pending config errors: %s", ep.pending_config)

    # Log served entities status
    if ep.config and ep.config.served_entities:
        for se in ep.config.served_entities:
            logger.info(
                "Served entity '%s' state: %s",
                se.name,
                getattr(se, "state", "N/A"),
            )

    if ready_state != "EndpointStateReady.READY":
        logger.error(
            "Endpoint is NOT ready (ready=%s). Check the Databricks UI → "
            "Serving → %s for error details.", ready_state, ENDPOINT_NAME
        )
        raise RuntimeError(
            f"Endpoint '{ENDPOINT_NAME}' is not ready: {ready_state}"
        )

    # ── Test invocation ──────────────────────────────────────────────────────
    logger.info("Sending test query to '%s'...", ENDPOINT_NAME)
    host = w.config.host.rstrip("/")
    url  = f"{host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    logger.info("POST %s", url)

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {w.config.token}"},
        json={
            "messages": [{"role": "user", "content": "Reply with: OK"}],
            "max_tokens": 10,
            "temperature": 0.0,
        },
        timeout=30,
    )
    if not resp.ok:
        logger.error("Invocation failed (%s): %s", resp.status_code, resp.text)
        resp.raise_for_status()
    reply = resp.json()["choices"][0]["message"]["content"]
    logger.info("Test reply: %r", reply)
    assert reply, "Empty reply from endpoint"
    print(f"\n✅ Endpoint '{ENDPOINT_NAME}' is live. Test reply: {reply!r}\n")


if __name__ == "__main__":
    w = WorkspaceClient()
    create_or_update_endpoint(w)
    verify_endpoint(w)