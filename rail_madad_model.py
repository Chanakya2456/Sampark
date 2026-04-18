"""
rail_madad_model.py
-------------------
MLflow PythonModel wrapper for the Rail Madad pipeline.

The single endpoint routes two request types based on the payload shape:

1) Complaint generation (default)
   Input columns:
       query          str   Required. Passenger's complaint in any Indian language.
       pnr            str   Optional. 10-digit PNR number.
       train_number   str   Optional. Train number.
       issue_type     str   Optional. One of: delay, food, cleanliness, staff,
                                     safety, medical, water, electricity,
                                     bedroll, other. Default: "other".
       language_code  str   Optional. BCP-47 code (e.g. "hi-IN"). Auto-detect if absent.

2) Ticket upload (if `ticket_image_base64` is present on the row)
   Input columns:
       ticket_image_base64  str  Required. Base64-encoded JPEG/PNG of the ticket.
       user_id              str  Optional. Identifier for the passenger. Default "unknown".

3) Grievance chat (if `request_type == "chat"`, or if `query` + `user_id`
   are present and no complaint-specific fields are set)
   Input columns:
       request_type  str   "chat" (explicit opt-in)
       query         str   Required. Free-form user message (any Indian language).
       user_id       str   Required. Used to look up the passenger's journey row.

   The chat path is served by `chat_agent.ChatAgent`, which drives sarvam-30b
   with four tools:
     • get_user_journey  — SQL lookup on the journey table
     • submit_grievance  — wraps ComplaintEngine to produce an SMS URI
     • rag_search        — Databricks Vector Search over the rules PDFs
     • web_search        — Tavily for general / live queries

Output (pandas DataFrame) — unified superset of columns. Fields not relevant
for a given request type are left blank / zero.
    request_type         str   "complaint" | "ticket" | "chat"
    status               str   "success" | "error"
    message              str   Human-readable status note.
    # Complaint fields
    sms_uri              str
    formatted_complaint  str
    char_count           int
    detected_language    str
    engine_notes         str
    # Ticket fields
    journey_id           str
    extracted_data       str   JSON-encoded ticket fields.
    # Chat fields
    chat_reply           str   Final assistant message for the user.
    tool_log             str   JSON-encoded trace of tools invoked.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import mlflow.pyfunc

logger = logging.getLogger(__name__)

# ── Ticket-extraction config ─────────────────────────────────────────────────
VISION_ENDPOINT_NAME = os.environ.get(
    "TICKET_VISION_ENDPOINT", "databricks-llama-4-maverick"
)
JOURNEY_TABLE = os.environ.get(
    "JOURNEY_TABLE", "workspace.default.journey"
)
TICKET_VOLUME_PATH = os.environ.get(
    "TICKET_VOLUME_PATH", "/Volumes/workspace/default/ticket_uploads"
)

_TICKET_PROMPT = """Extract these details from this Indian railway ticket:
- PNR number
- Train number
- Train name
- Source station code
- Destination station code
- Journey date (YYYY-MM-DD)
- Departure time (HH:MM)
- Arrival time (HH:MM)
- Class (3A/2A/SL etc)
- Seat/berth number

Return ONLY a JSON object with these keys: pnr, train_number, train_name,
source_station, destination_station, journey_date, departure_time,
arrival_time, class, seat_number. Use null if field not visible.
"""

_EMPTY_ROW: dict[str, Any] = {
    "request_type":        "",
    "status":              "",
    "message":             "",
    "sms_uri":             "",
    "formatted_complaint": "",
    "char_count":          0,
    "detected_language":   "",
    "engine_notes":        "",
    "journey_id":          "",
    "extracted_data":      "",
    "chat_reply":          "",
    "tool_log":            "",
}


def _clean_row(row: pd.Series) -> dict[str, Any]:
    """Convert a DataFrame row to a plain dict, dropping NaNs/empty strings."""
    out: dict[str, Any] = {}
    for k, v in row.to_dict().items():
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out[k] = v
    return out


class RailMadadModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel that routes to either ComplaintEngine.generate()
    or the ticket-extraction pipeline based on payload shape.

    Loaded once per serving replica (load_context),
    then predict() is called for each batch of requests.
    """

    def load_context(self, context) -> None:  # noqa: ANN001
        """Initialise the ComplaintEngine (and its SarvamClient) at load time."""
        # complaint_engine.py + sarvam_client.py + chat_agent.py are bundled
        # via code_paths in register_model.py.
        from complaint_engine import ComplaintEngine  # noqa: PLC0415
        from chat_agent import ChatAgent              # noqa: PLC0415

        self.engine = ComplaintEngine()
        # Share the SarvamClient + WorkspaceClient with the chat agent so
        # auth is initialised exactly once per serving replica.
        self.chat_agent = ChatAgent(sarvam_client=self.engine.client)

    def predict(self, context, model_input) -> pd.DataFrame:  # noqa: ANN001
        """
        Dispatch each row to the complaint, ticket, or chat pipeline.

        Accepts either a pandas DataFrame (standard MLflow serving input)
        or a plain dict (for local testing).
        """
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])

        rows: list[dict[str, Any]] = []
        for _, raw in model_input.iterrows():
            row = _clean_row(raw)
            rows.append(self._dispatch(row))

        return pd.DataFrame(rows)

    def _dispatch(self, row: dict[str, Any]) -> dict[str, Any]:
        """Route a single cleaned row to the correct sub-pipeline."""
        request_type = str(row.get("request_type", "") or "").strip().lower()

        if request_type == "chat":
            return self._chat_respond(row)
        if request_type == "ticket" or row.get("ticket_image_base64"):
            return self._process_ticket(row)
        if request_type == "complaint":
            return self._generate_complaint(row)

        # Heuristic fallback — a query + user_id without complaint metadata is a
        # chat message (keeps the Streamlit client simple).
        if (
            row.get("query")
            and row.get("user_id")
            and not any(row.get(k) for k in ("issue_type", "pnr", "train_number"))
        ):
            return self._chat_respond(row)

        return self._generate_complaint(row)

    # ── Chat path ─────────────────────────────────────────────────────────

    def _chat_respond(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(_EMPTY_ROW)
        out["request_type"] = "chat"

        query = str(row.get("query", "")).strip()
        user_id = str(row.get("user_id", "") or "")
        if not query:
            out["status"]  = "error"
            out["message"] = "empty query"
            return out

        try:
            result = self.chat_agent.respond(query=query, user_id=user_id)
            out.update({
                "status":     "success",
                "message":    "Chat response generated",
                "chat_reply": result.get("reply", ""),
                "sms_uri":    result.get("sms_uri", ""),
                "tool_log":   json.dumps(
                    result.get("tool_log", []), ensure_ascii=False, default=str
                ),
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("Chat pipeline failed")
            out["status"]  = "error"
            out["message"] = f"Chat pipeline failed: {exc}"
        return out

    # ── Complaint path ──────────────────────────────────────────────────

    def _generate_complaint(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(_EMPTY_ROW)
        out["request_type"] = "complaint"

        query = str(row.get("query", "")).strip()
        if not query:
            out["status"]       = "error"
            out["message"]      = "empty query"
            out["engine_notes"] = "ERROR: empty query"
            return out

        try:
            result = self.engine.generate(
                raw_input=query,
                issue_type=str(row.get("issue_type", "other") or "other"),
                train_number=str(row.get("train_number", "") or ""),
                pnr=str(row.get("pnr", "") or ""),
                language_code=row.get("language_code") or None,
            )
            out.update({
                "status":              "success",
                "message":             "Complaint generated",
                "sms_uri":             result.sms_uri,
                "formatted_complaint": result.formatted_complaint,
                "char_count":          result.char_count,
                "detected_language":   result.detected_language,
                "engine_notes":        ", ".join(result.engine_notes),
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("Complaint generation failed")
            out["status"]       = "error"
            out["message"]      = "Failed to generate complaint"
            out["engine_notes"] = f"ERROR: {exc}"
        return out

    # ── Ticket path ───────────────────────────────────────────────────────────

    def _process_ticket(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Extract ticket fields from a base64 image via a vision LLM and persist
        the journey row to a Unity Catalog table.
        """
        out = dict(_EMPTY_ROW)
        out["request_type"] = "ticket"

        image_base64 = str(row.get("ticket_image_base64", ""))
        user_id      = str(row.get("user_id", "unknown") or "unknown")

        if image_base64.startswith("data:"):
            image_base64 = image_base64.split(",", 1)[-1]
        image_base64 = "".join(image_base64.split())

        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            out["status"]  = "error"
            out["message"] = f"Failed to process ticket: invalid base64 image ({exc})"
            return out
        mime_type = _detect_image_mime(image_bytes)

        try:
            # Reuse the WorkspaceClient already held by SarvamClient so we
            # don't re-authenticate on every request.
            w = self.engine.client.w

            # NOTE: w.serving_endpoints.query(...) can't be used here because
            # its typed ChatMessage schema doesn't support OpenAI-style
            # multi-part content blocks (text + image_url). We POST directly
            # to the endpoint's invocations URL instead.
            host = w.config.host.rstrip("/")
            token = w.config.token
            vision_response = requests.post(
                f"{host}/serving-endpoints/{VISION_ENDPOINT_NAME}/invocations",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _TICKET_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_base64}",
                                },
                            },
                        ],
                    }],
                    "max_tokens": 500,
                },
                timeout=60,
            )
            if not vision_response.ok:
                raise RuntimeError(
                    f"Vision endpoint {VISION_ENDPOINT_NAME} returned "
                    f"{vision_response.status_code}: {vision_response.text}"
                )
            payload = vision_response.json()
            extracted_text = payload["choices"][0]["message"]["content"]
            ticket_data    = _parse_ticket_json(extracted_text)

            journey_id = str(uuid.uuid4())
            _insert_journey(
                journey_id=journey_id,
                user_id=user_id,
                ticket_data=ticket_data,
                extracted_text=extracted_text,
            )

            out.update({
                "status":         "success",
                "message":        "Ticket processed and journey details saved",
                "journey_id":     journey_id,
                "extracted_data": json.dumps(ticket_data, ensure_ascii=False),
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ticket processing failed")
            out["status"]  = "error"
            out["message"] = f"Failed to process ticket: {exc}"
        return out


# ── Helpers ──────────────────────────────────────────────────────────────────

def _detect_image_mime(image_bytes: bytes) -> str:
    """Sniff the image MIME type from the first bytes of the decoded payload.

    Databricks' vision endpoints require a matching data URI prefix
    (e.g. ``data:image/webp;base64,...``). Using the wrong MIME makes the
    endpoint reject the request with ``Invalid base64 string for image``.
    """
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    # Fall back to jpeg; most tickets from mobile capture are JPEG.
    return "image/jpeg"


def _parse_ticket_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from the vision model response."""
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    blob  = match.group() if match else text
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        logger.warning("Could not parse ticket JSON from: %r", text[:200])
        return {}


def _insert_journey(
    journey_id: str,
    user_id: str,
    ticket_data: dict[str, Any],
    extracted_text: str,
) -> None:
    """
    Insert a journey record into the Unity Catalog `journey` table via
    the Databricks SQL connector.

    Expects these env vars on the serving replica:
        DATABRICKS_HOST          e.g. https://<workspace>.cloud.databricks.com
        DATABRICKS_TOKEN         PAT / OAuth token
        DATABRICKS_SQL_HTTP_PATH e.g. /sql/1.0/warehouses/<warehouse-id>
    """
    from databricks import sql  # noqa: PLC0415 — lazy import (only on ticket path)

    host = os.environ.get("DATABRICKS_HOST", "").replace("https://", "").rstrip("/")
    http_path = os.environ.get("DATABRICKS_SQL_HTTP_PATH")
    token     = os.environ.get("DATABRICKS_TOKEN")
    if not (host and http_path and token):
        raise RuntimeError(
            "Missing one of DATABRICKS_HOST / DATABRICKS_SQL_HTTP_PATH / "
            "DATABRICKS_TOKEN in the serving environment."
        )

    ticket_image_path = f"{TICKET_VOLUME_PATH.rstrip('/')}/{journey_id}.jpg"
    now = datetime.now(timezone.utc)

    connection = sql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {JOURNEY_TABLE}
                    (journey_id, user_id, pnr, train_number, train_name,
                     source_station, destination_station, journey_date,
                     departure_time, arrival_time, `class`, seat_number,
                     ticket_image_path, extracted_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    journey_id,
                    user_id,
                    ticket_data.get("pnr"),
                    ticket_data.get("train_number"),
                    ticket_data.get("train_name"),
                    ticket_data.get("source_station"),
                    ticket_data.get("destination_station"),
                    ticket_data.get("journey_date"),
                    ticket_data.get("departure_time"),
                    ticket_data.get("arrival_time"),
                    ticket_data.get("class"),
                    ticket_data.get("seat_number"),
                    ticket_image_path,
                    extracted_text,
                    now,
                ),
            )
    finally:
        connection.close()
