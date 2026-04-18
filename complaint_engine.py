"""
complaint_engine.py
-------------------
Builds Rail Madad complaint SMS messages from raw user input using
Sarvam AI (sarvam-30b) with a rural-context-aware system prompt.

Output:
  - A structured English complaint string (≤ 160 chars for single SMS)
  - A click-to-send SMS URI: sms:139?body=<url-encoded-message>

The engine:
  1. Auto-detects the user's language (or accepts an explicit code).
  2. Translates the input to English if needed.
  3. Sends the translated text + metadata to sarvam-30b with a
     carefully engineered system prompt that prioritises rural context.
  4. Returns the final message + click-to-send link.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote

from sarvam_client import SarvamClient, COMPLAINT_MODEL

logger = logging.getLogger(__name__)

# ── Rail Madad complaint categories (from the official 139 / RailMadad menu) ─
ISSUE_TYPES = {
    "delay":        "Train Delay",
    "food":         "Food/Catering Quality",
    "cleanliness":  "Coach Cleanliness",
    "staff":        "Staff Misconduct/Behaviour",
    "safety":       "Security/Safety Concern",
    "medical":      "Medical Assistance Required",
    "water":        "Water Availability",
    "electricity":  "Electrical Fittings",
    "bedroll":      "Bedroll/Linen Complaint",
    "other":        "General Grievance",
}

# Language code → human label (subset; matches Sarvam-Translate support)
LANGUAGE_LABELS = {
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "bn-IN": "Bengali",
    "te-IN": "Telugu",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "pa-IN": "Punjabi",
    "od-IN": "Odia",
    "as-IN": "Assamese",
    "ur-IN": "Urdu",
    "en-IN": "English",
}

SMS_MAX_CHARS = 160  # single SMS segment target


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class ComplaintResult:
    """Encapsulates everything produced by the complaint engine."""

    # Raw inputs
    raw_input: str
    detected_language: str
    issue_type: str
    train_number: str
    pnr: str

    # Intermediate
    english_translation: str = ""

    # Outputs
    formatted_complaint: str = ""
    sms_uri: str = ""          # sms:139?body=<url-encoded>
    char_count: int = 0

    # Metadata / diagnostics
    truncated: bool = False
    engine_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw_input": self.raw_input,
            "detected_language": LANGUAGE_LABELS.get(self.detected_language, self.detected_language),
            "issue_type": ISSUE_TYPES.get(self.issue_type, self.issue_type),
            "train_number": self.train_number,
            "pnr": self.pnr,
            "english_translation": self.english_translation,
            "formatted_complaint": self.formatted_complaint,
            "sms_uri": self.sms_uri,
            "char_count": self.char_count,
            "truncated": self.truncated,
            "engine_notes": self.engine_notes,
        }


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a Rail Madad SMS complaint writer. "
    "Output ONE plain-text line only — the complaint text itself. "
    "No markdown, no asterisks, no bullet points, no explanation, no label. "
    "Format: [Category] Train [number] | [short issue description]. "
    "Max 155 characters. If urgent (medical/safety), start with URGENT:."
)

# ── Engine ─────────────────────────────────────────────────────────────────────

class ComplaintEngine:
    """
    Orchestrates translation + LLM complaint generation.

    Args:
        client: An initialised SarvamClient. If None, one is created (reads
                Databricks secret automatically when running on Databricks).
    """

    def __init__(self, client: Optional[SarvamClient] = None):
        self.client = client or SarvamClient()

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        raw_input: str,
        issue_type: str = "other",
        train_number: str = "",
        pnr: str = "",
        language_code: Optional[str] = None,
    ) -> ComplaintResult:
        """
        Full pipeline: detect language → translate → generate complaint → build SMS URI.

        Args:
            raw_input:      User's description in any Indian language (or English).
            issue_type:     Key from ISSUE_TYPES dict. Default 'other'.
            train_number:   Train number (optional, helps the LLM).
            pnr:            10-digit PNR (optional).
            language_code:  Force a BCP-47 code. If None, auto-detect.

        Returns:
            ComplaintResult with all fields populated.
        """
        result = ComplaintResult(
            raw_input=raw_input,
            detected_language="",
            issue_type=issue_type,
            train_number=train_number,
            pnr=pnr,
        )

        # Step 1 — Detect language
        if language_code:
            result.detected_language = language_code
            result.engine_notes.append(f"Language forced to: {language_code}")
        else:
            result.detected_language = self.client.identify_language(raw_input)
            result.engine_notes.append(
                f"Language auto-detected: {result.detected_language}"
            )

        # Step 2 — Translate to English (skip if already English)
        if result.detected_language == "en-IN":
            result.english_translation = raw_input
            result.engine_notes.append("Input is English — no translation needed.")
        else:
            result.english_translation = self.client.translate_to_english(
                text=raw_input,
                source_language_code=result.detected_language,
            )
            result.engine_notes.append(
                f"Translated from {result.detected_language} to English."
            )

        # Step 3 — Build user message for the LLM
        user_message = self._build_user_message(result)

        # Step 4 — LLM complaint generation
        reply = self.client.chat_completion(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=2048,  # sarvam-m reasoning needs headroom before the output line
            model=COMPLAINT_MODEL,
        )
        logger.debug("Raw LLM reply (len=%d): %s", len(reply), reply[:300])

        # Step 5 — Extract complaint from completion (strip prefix + take first line)
        complaint = self._extract_complaint(reply)

        # Fallback: if the model returned nothing usable, build a minimal complaint
        # from the structured fields so the SMS is never empty.
        if not complaint:
            category = ISSUE_TYPES.get(result.issue_type, "General Grievance")
            desc = (result.english_translation or result.raw_input)[:80].strip()
            if result.train_number:
                complaint = f"{category} Train {result.train_number} | {desc}"
            else:
                complaint = f"{category} | {desc}"
            result.engine_notes.append("LLM returned empty output — used structured fallback.")
            logger.warning("[complaint_engine] LLM output was empty; using fallback complaint.")
        if len(complaint) > SMS_MAX_CHARS:
            complaint = complaint[: SMS_MAX_CHARS - 1] + "…"
            result.truncated = True
            result.engine_notes.append(
                f"Complaint truncated to {SMS_MAX_CHARS} characters."
            )

        # Append PNR to the complaint string itself
        if result.pnr:
            complaint = f"{complaint} PNR {result.pnr}"

        result.formatted_complaint = complaint
        result.char_count = len(complaint)

        # Step 6 — Build SMS URI  (format: MADAD <complaint with PNR>)
        result.sms_uri = _build_sms_uri(complaint)

        return result

    def chat(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Freeform chat with the Rail Madad assistant (for follow-up questions).

        Args:
            user_message:          The user's latest message (any language).
            conversation_history:  Previous turns as [{"role":..., "content":...}].

        Returns:
            Assistant reply string.
        """
        system = (
            "You are a helpful Rail Madad grievance assistant for Indian Railways passengers. "
            "Respond in the same language the user writes in. "
            "Help them understand how to file complaints, track status, and get assistance. "
            "Be especially helpful to rural passengers who may be unfamiliar with digital processes. "
            "Rail Madad helpline: 139. Website: railmadad.indianrailways.gov.in"
        )
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        return self.client.chat_completion(
            messages=messages,
            system_prompt=system,
            temperature=0.5,
            max_tokens=400,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_user_message(self, result: ComplaintResult) -> str:
        parts = [
            f"Category: {ISSUE_TYPES.get(result.issue_type, result.issue_type)}",
        ]
        if result.train_number:
            parts.append(f"Train: {result.train_number}")
        parts.append(f"Issue: {result.english_translation}")
        return "\n".join(parts)

    @staticmethod
    def _extract_complaint(text: str) -> str:
        """Strip reasoning tags and return the first clean output line.
        sarvam-m wraps its chain-of-thought in <think>...</think>;
        the actual complaint is the first non-empty line after </think>.
        """
        text = text.strip()
        # Remove <think>...</think> block
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Also strip unclosed <think> (when max_tokens cuts off before </think>)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
        # Strip echoed labels the model sometimes prepends
        text = re.sub(r"(?i)^(sms\s+complaint|complaint|output)\s*:\s*", "", text)
        # Return first non-empty line
        for line in text.splitlines():
            line = line.strip().strip('"').strip("*").strip()
            if len(line) > 5:
                return line
        return text.strip()





# ── SMS URI builder ────────────────────────────────────────────────────────────

def _build_sms_uri(complaint: str, number: str = "139") -> str:
    """
    Build a click-to-send SMS URI.

    Format sent to Rail Madad 139:
        MADAD <complaint>   (PNR already embedded in complaint string)

    Example:
        sms:139?body=MADAD%20Electricity%20Train%2011045%20...%20PNR%209876543210

    On Android/iOS this opens the SMS app pre-populated so the user
    only needs to press Send.
    """
    body = f"MADAD {complaint}"
    return f"sms:{number}?body={quote(body, safe='')}"
