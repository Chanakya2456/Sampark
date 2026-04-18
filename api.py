"""
api.py
------
Rail Madad backend — FastAPI app deployable as a Databricks App.

Routes:
    GET  /healthz              Liveness probe.
    POST /users/register       Phone number → user_id (inserted into users table).
    POST /tickets/upload       Multipart image → vision model + journey insert.
    POST /chat                 {query, user_id} → ChatAgent reply + optional SMS URI.
    POST /complaints           Direct complaint-engine call (optional, legacy).

All Databricks integrations (Sarvam via Databricks Secrets, Vector Search on
Mosaic AI, SQL writes via the Databricks SQL connector, vision via the
existing `databricks-llama-4-maverick` serving endpoint) are unchanged — only
the outer transport is now FastAPI instead of MLflow model serving.

Run locally:
    uvicorn api:app --reload

Databricks App:
    See `app.yaml` for the startup command.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from functools import lru_cache
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chat_agent import ChatAgent
from complaint_engine import ComplaintEngine
from sarvam_client import SarvamClient
from ticket_extractor import insert_user, process_ticket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("rail_madad_api")

app = FastAPI(
    title="Rail Madad Sahayak API",
    version="1.0.0",
    description=(
        "Grievance-resolution backend for elderly Indian Railways passengers. "
        "Deployable as a Databricks App."
    ),
)

# Open CORS so the Streamlit frontend (wherever it's hosted) can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy singletons ──────────────────────────────────────────────────────────
# Heavy clients are created on first use so uvicorn boot stays fast, and the
# singletons make sure the Vector Search index + SentenceTransformer warm-start
# exactly once per worker.

@lru_cache(maxsize=1)
def _sarvam_client() -> SarvamClient:
    logger.info("Initialising SarvamClient (first request)...")
    return SarvamClient()


@lru_cache(maxsize=1)
def _complaint_engine() -> ComplaintEngine:
    return ComplaintEngine(client=_sarvam_client())


@lru_cache(maxsize=1)
def _chat_agent() -> ChatAgent:
    return ChatAgent(sarvam_client=_sarvam_client())


# ── Request / response schemas ───────────────────────────────────────────────

class RegisterRequest(BaseModel):
    phone: str = Field(..., description="10-digit Indian mobile number.")


class RegisterResponse(BaseModel):
    user_id: str
    phone: str
    status: str = "success"
    message: str = ""


class ChatRequest(BaseModel):
    query: str
    user_id: str = ""
    lang: str = "en"  # "en" | "hi" — controls reply language
    history: list[dict] = Field(
        default_factory=list,
        description="Prior turns as [{role:'user'|'assistant', content:'...'}, ...].",
    )


class ChatResponse(BaseModel):
    reply: str
    sms_uri: str = ""
    tool_log: list = Field(default_factory=list)


class ComplaintRequest(BaseModel):
    query: str
    issue_type: str = "other"
    train_number: Optional[str] = ""
    pnr: Optional[str] = ""
    language_code: Optional[str] = None


class ComplaintResponse(BaseModel):
    status: str
    sms_uri: str
    formatted_complaint: str
    char_count: int
    detected_language: str
    engine_notes: list[str]


class TicketResponse(BaseModel):
    status: str
    message: str
    journey_id: str = ""
    extracted: dict = Field(default_factory=dict)


class TranscribeResponse(BaseModel):
    text: str
    source_language_code: str = ""


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe."""
    return {"ok": True, "service": "rail-madad-api"}


def _background_insert_user(user_id: str, phone: str) -> None:
    """Wrapper so background task failures are logged instead of silently dropped."""
    try:
        insert_user(user_id=user_id, phone=phone)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Users table insert failed in background for user_id=%s: %s",
            user_id, exc,
        )


@app.post("/users/register", response_model=RegisterResponse)
def register_user(
    body: RegisterRequest,
    background: BackgroundTasks,
) -> RegisterResponse:
    """
    Validate a 10-digit phone number, mint a user_id, and return immediately.

    The Delta INSERT is dispatched as a background task so the HTTP response
    is bounded only by UUID minting + regex validation (milliseconds) rather
    than by SQL-warehouse round-trip + Delta commit (seconds).
    """
    phone = (body.phone or "").strip()
    if not re.fullmatch(r"\d{10}", phone):
        raise HTTPException(
            status_code=400,
            detail="Phone number must be exactly 10 digits.",
        )
    user_id = str(uuid.uuid4())
    background.add_task(_background_insert_user, user_id, phone)
    return RegisterResponse(user_id=user_id, phone=phone)


@app.post("/speech/transcribe", response_model=TranscribeResponse)
async def transcribe_speech(
    file: UploadFile = File(...),
    lang: str = Form("en"),
) -> TranscribeResponse:
    """
    Accept a short audio clip (wav/webm/mp3) captured from the browser mic
    and return its transcript in the caller's selected UI language ("en" or
    "hi"). Auto-detects the spoken language and translates if needed.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio.")
    try:
        result = _sarvam_client().transcribe_to_language(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.wav",
            content_type=file.content_type or "audio/wav",
            target_lang=(lang or "en").lower(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Speech transcription failed")
        raise HTTPException(status_code=502, detail=f"Transcription failed: {exc}") from exc
    return TranscribeResponse(
        text=result.get("text", ""),
        source_language_code=result.get("source_language_code", ""),
    )


@app.post("/tickets/upload", response_model=TicketResponse)
async def upload_ticket(
    file: UploadFile = File(...),
    user_id: str = Form(...),
) -> TicketResponse:
    """
    Accept a ticket image (JPG/PNG), push it through the Databricks vision
    serving endpoint, and persist the extracted fields into the journey table.
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    result = process_ticket(image_bytes=image_bytes, user_id=user_id)
    if result.get("status") != "success":
        raise HTTPException(status_code=502, detail=result.get("message", "Failed"))
    return TicketResponse(**result)


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    """
    Grievance chat agent. Drives sarvam-30b with four tools:
      - get_user_journey  (SQL on journey table)
      - submit_grievance  (wraps ComplaintEngine → SMS URI)
      - rag_search        (Databricks Mosaic Vector Search)
      - web_search        (Tavily)
    """
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required.")

    try:
        result = _chat_agent().respond(
            query=query,
            user_id=body.user_id or "",
            lang=body.lang or "en",
            history=body.history or [],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat pipeline failed")
        raise HTTPException(status_code=500, detail=f"Chat pipeline failed: {exc}") from exc

    return ChatResponse(
        reply=result.get("reply", ""),
        sms_uri=result.get("sms_uri", ""),
        tool_log=result.get("tool_log", []),
    )


@app.post("/complaints", response_model=ComplaintResponse)
def generate_complaint(body: ComplaintRequest) -> ComplaintResponse:
    """
    Optional direct access to the complaint pipeline (bypasses the chat agent).
    """
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required.")
    try:
        result = _complaint_engine().generate(
            raw_input=query,
            issue_type=body.issue_type or "other",
            train_number=body.train_number or "",
            pnr=body.pnr or "",
            language_code=body.language_code,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Complaint generation failed")
        raise HTTPException(status_code=500, detail=f"Complaint failed: {exc}") from exc

    return ComplaintResponse(
        status="success",
        sms_uri=result.sms_uri,
        formatted_complaint=result.formatted_complaint,
        char_count=result.char_count,
        detected_language=result.detected_language,
        engine_notes=result.engine_notes,
    )


# ── Local dev entrypoint ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
