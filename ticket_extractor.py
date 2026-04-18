"""
ticket_extractor.py
-------------------
Databricks-native helpers for the ticket-upload route.

Flow:
    JPG/PNG bytes
      │
      ▼  base64 + MIME sniff
    POST {host}/serving-endpoints/{TICKET_VISION_ENDPOINT}/invocations
      │  (databricks-llama-4-maverick by default — a vision model)
      ▼
    JSON → parsed dict (pnr, train_number, etc.)
      │
      ▼
    INSERT INTO workspace.default.journey  (Databricks SQL warehouse)

This module is imported by both the FastAPI app and the legacy MLflow
pyfunc (`rail_madad_model.py`) so the extraction logic stays in one place.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import requests
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# ── Shared Databricks SQL connection ────────────────────────────────────────
# The Databricks SQL connector is expensive to construct (TLS + Thrift
# handshake ≈ 0.5–2 s) and is NOT thread-safe, so we keep a single process-wide
# connection guarded by a lock and reconnect transparently if it's closed.
_SQL_CONNECTION = None
_SQL_CONNECTION_LOCK = threading.Lock()

# Per-table "CREATE TABLE IF NOT EXISTS" flag. Issuing DDL on every write
# costs another round-trip to the warehouse, so we run it at most once per
# process/table.
_TABLES_READY: set[str] = set()


def _sql_config() -> tuple[str, str, str]:
    host = os.environ.get("DATABRICKS_HOST", "").replace("https://", "").rstrip("/")
    http_path = os.environ.get("DATABRICKS_SQL_HTTP_PATH")
    token = os.environ.get("DATABRICKS_TOKEN")
    if not (host and http_path and token):
        raise RuntimeError(
            "Missing DATABRICKS_HOST / DATABRICKS_SQL_HTTP_PATH / DATABRICKS_TOKEN "
            "in the environment."
        )
    return host, http_path, token


def _reset_sql_connection() -> None:
    """Close and forget the cached SQL connection (used on stale-connection retry)."""
    global _SQL_CONNECTION
    try:
        if _SQL_CONNECTION is not None:
            _SQL_CONNECTION.close()
    except Exception:  # noqa: BLE001
        pass
    _SQL_CONNECTION = None


@contextmanager
def _sql_cursor() -> Iterator[Any]:
    """
    Yield a cursor from the shared Databricks SQL connection under a lock.

    Thread-safety: the Databricks SQL connector is not thread-safe, so we
    serialize all access behind `_SQL_CONNECTION_LOCK`. For the Rail Madad
    workload (register + ticket insert, a handful of RPS) this is plenty; if
    we ever need more throughput we can promote this to a small pool.
    """
    global _SQL_CONNECTION
    from databricks import sql  # noqa: PLC0415

    host, http_path, token = _sql_config()
    with _SQL_CONNECTION_LOCK:
        if _SQL_CONNECTION is None:
            _SQL_CONNECTION = sql.connect(
                server_hostname=host,
                http_path=http_path,
                access_token=token,
            )
        try:
            with _SQL_CONNECTION.cursor() as cur:
                yield cur
        except Exception:
            # Most likely the warehouse recycled the session; drop the cached
            # handle so the next call reconnects cleanly.
            _reset_sql_connection()
            raise

VISION_ENDPOINT_NAME = os.environ.get(
    "TICKET_VISION_ENDPOINT", "databricks-llama-4-maverick"
)
JOURNEY_TABLE = os.environ.get("JOURNEY_TABLE", "workspace.default.journey")
TICKET_VOLUME_PATH = os.environ.get(
    "TICKET_VOLUME_PATH", "/Volumes/workspace/default/ticket_uploads"
)

TICKET_PROMPT = """Extract these details from this Indian railway ticket:
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


# ── Public entry point ───────────────────────────────────────────────────────

def process_ticket(
    image_bytes: bytes,
    user_id: str,
    workspace_client: WorkspaceClient | None = None,
) -> dict[str, Any]:
    """
    Run a ticket image through the Databricks vision endpoint and persist
    the extracted fields to the journey table.

    Returns:
        {
            "status":        "success" | "error",
            "message":       str,
            "journey_id":    str,
            "extracted":     {... ticket fields ...},
        }
    """
    if not image_bytes:
        return {"status": "error", "message": "No image bytes received."}

    mime_type = _detect_image_mime(image_bytes)
    import base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    w = workspace_client or WorkspaceClient()

    try:
        extracted_text = _call_vision_endpoint(w, image_b64, mime_type)
        ticket_data = _parse_ticket_json(extracted_text)

        journey_id = str(uuid.uuid4())
        _insert_journey(
            journey_id=journey_id,
            user_id=user_id or "unknown",
            ticket_data=ticket_data,
            extracted_text=extracted_text,
        )
        return {
            "status":     "success",
            "message":    "Ticket processed and journey details saved",
            "journey_id": journey_id,
            "extracted":  ticket_data,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ticket processing failed")
        return {"status": "error", "message": f"Ticket processing failed: {exc}"}


# ── Internals ────────────────────────────────────────────────────────────────

def _call_vision_endpoint(w: WorkspaceClient, image_b64: str, mime_type: str) -> str:
    """POST to the vision model-serving endpoint and return the raw content."""
    host = w.config.host.rstrip("/")
    token = w.config.token
    resp = requests.post(
        f"{host}/serving-endpoints/{VISION_ENDPOINT_NAME}/invocations",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": TICKET_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                        },
                    },
                ],
            }],
            "max_tokens": 500,
        },
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Vision endpoint {VISION_ENDPOINT_NAME} returned "
            f"{resp.status_code}: {resp.text}"
        )
    return resp.json()["choices"][0]["message"]["content"]


def _detect_image_mime(image_bytes: bytes) -> str:
    """Sniff MIME from the first bytes. Matches what Databricks vision expects."""
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
    return "image/jpeg"


def _parse_ticket_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    blob = match.group() if match else text
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
    """Insert a journey row via the Databricks SQL connector."""
    from databricks import sql  # noqa: PLC0415 — lazy

    host = os.environ.get("DATABRICKS_HOST", "").replace("https://", "").rstrip("/")
    http_path = os.environ.get("DATABRICKS_SQL_HTTP_PATH")
    token = os.environ.get("DATABRICKS_TOKEN")
    if not (host and http_path and token):
        raise RuntimeError(
            "Missing DATABRICKS_HOST / DATABRICKS_SQL_HTTP_PATH / DATABRICKS_TOKEN "
            "in the environment."
        )

    ticket_image_path = f"{TICKET_VOLUME_PATH.rstrip('/')}/{journey_id}.jpg"
    now = datetime.now(timezone.utc)

    connection = sql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token,
    )
    try:
        with connection.cursor() as cur:
            cur.execute(
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


def insert_user(user_id: str, phone: str, users_table: str | None = None) -> None:
    """Create (if missing) and insert a row into the users table.

    Uses the shared pooled SQL connection and runs the table-creation DDL at
    most once per process. On a stale connection it retries exactly once with
    a fresh connection.
    """
    table = users_table or os.environ.get("USERS_TABLE", "workspace.default.users")

    def _do() -> None:
        with _sql_cursor() as cur:
            if table not in _TABLES_READY:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        user_id STRING,
                        phone_number STRING,
                        created_at TIMESTAMP
                    )
                    """
                )
                _TABLES_READY.add(table)
            cur.execute(
                f"INSERT INTO {table} (user_id, phone_number, created_at) VALUES (?, ?, ?)",
                (user_id, phone, datetime.now(timezone.utc)),
            )

    try:
        _do()
    except Exception:  # noqa: BLE001
        logger.warning(
            "insert_user failed on pooled connection; retrying once with a fresh session",
            exc_info=True,
        )
        _reset_sql_connection()
        _do()
