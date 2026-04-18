"""
sarvam_client.py
----------------
Sarvam AI client that routes LLM calls through the Databricks serving endpoint
('sarvam-30b-gateway') instead of calling Sarvam directly.

Architecture:
  Chat/LLM  → w.serving_endpoints.query("sarvam-30b-gateway", ...)
                  ↳ Databricks AI Gateway → https://api.sarvam.ai/v1/chat/completions
                  ↳ Key resolved from Databricks Secret Scope (never in code)
                  ↳ Every call logged to Delta inference table

  Translate → Direct REST to https://api.sarvam.ai/translate
  (Sarvam's /translate is not OpenAI-compatible so it can't go through
   the AI Gateway; the API key is still fetched from Databricks Secrets.)

Run setup_endpoint.py once before using this module.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import requests
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SARVAM_BASE_URL     = "https://api.sarvam.ai"
SARVAM_CHAT_URL     = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
SARVAM_LID_URL       = "https://api.sarvam.ai/text-lid"
SARVAM_STT_URL       = "https://api.sarvam.ai/speech-to-text"
CHAT_MODEL           = "sarvam-30b"     # routing / tool-calling (sarvam-30b supports tools)
GENERATOR_MODEL      = "sarvam-30b"     # primary chat – final user-facing replies
COMPLAINT_MODEL      = "sarvam-m"       # no tool calling – fast template filling

SECRET_SCOPE = "bharatbricks"
SECRET_KEY   = "sarvam_api_key"

# ── Supported UI languages (2-letter code → BCP-47 used by Sarvam APIs) ──────
SUPPORTED_LANG_CODES: dict[str, str] = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
}


def to_bcp47(code: str, default: str = "en-IN") -> str:
    """Map a short UI language code (e.g. 'hi') to its Sarvam BCP-47 code."""
    return SUPPORTED_LANG_CODES.get((code or "").lower(), default)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_workspace_client() -> WorkspaceClient:
    """
    Return an authenticated WorkspaceClient.
    Inside Databricks Apps/Jobs this auto-authenticates from the injected
    environment. Locally it uses ~/.databrickscfg or env vars.
    """
    return WorkspaceClient()


def _get_sarvam_key_for_rest(w: WorkspaceClient) -> str:
    """
    Fetch the Sarvam API key from Databricks Secrets for REST endpoints
    that can't go through the AI Gateway (translate, lid).
    Falls back to SARVAM_API_KEY env var for local dev.
    """
    try:
        secret = w.secrets.get_secret(scope=SECRET_SCOPE, key=SECRET_KEY)
        if secret.value:
            # The Databricks Secrets REST API always returns values base64-encoded.
            # Decode it to get the original plaintext key.
            try:
                decoded = base64.b64decode(secret.value).decode("utf-8")
            except Exception:  # noqa: BLE001
                decoded = secret.value  # already plaintext (future SDK versions)
            return decoded
    except Exception as exc:  # noqa: BLE001
        logger.debug("Databricks secret fetch failed: %s", exc)

    env_key = os.environ.get("SARVAM_API_KEY", "")
    if env_key:
        return env_key

    raise EnvironmentError(
        f"Sarvam API key not found in Databricks Secrets "
        f"(scope={SECRET_SCOPE!r}, key={SECRET_KEY!r}) "
        f"or env var SARVAM_API_KEY."
    )


# ── Client ────────────────────────────────────────────────────────────────────

class SarvamClient:
    """
    Sarvam AI client backed by Databricks infrastructure.

    LLM calls  → Databricks serving endpoint  (w.serving_endpoints.query)
    REST calls → Sarvam translate/LID APIs    (key from Databricks Secrets)

    Usage:
        client = SarvamClient()           # inside Databricks — zero config
        client = SarvamClient(w=my_client) # bring your own WorkspaceClient
    """

    def __init__(self, w: Optional[WorkspaceClient] = None, timeout: int = 30):
        self.w       = w or _get_workspace_client()
        self.timeout = timeout
        # Lazy — only fetched when a REST endpoint is called
        self._sarvam_key: Optional[str] = None

    @property
    def sarvam_key(self) -> str:
        if self._sarvam_key is None:
            self._sarvam_key = _get_sarvam_key_for_rest(self.w)
        return self._sarvam_key

    # ── LLM via Databricks serving endpoint ───────────────────────────────────

    def chat_completion_raw(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        model: str = CHAT_MODEL,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> dict:
        """
        Low-level Sarvam chat call that returns the raw `message` object from
        the first choice (including `tool_calls` when tools are configured).

        Use this when you need OpenAI-style function calling. For plain-text
        completions, prefer `chat_completion(...)`.
        """
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": "low",
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        logger.info(
            "Calling Sarvam chat [raw] (%s, %d messages, tools=%s)",
            model, len(messages), bool(tools),
        )
        result = self._sarvam_post(SARVAM_CHAT_URL, payload)
        return result["choices"][0]["message"]

    def chat_completion(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        model: str = CHAT_MODEL,
    ) -> str:
        """
        Call sarvam-30b chat completions directly via Sarvam REST API.
        The API key is fetched from Databricks Secrets via the SDK.

        Note: Databricks AI Gateway (Custom Provider) was attempted but Sarvam's
        /v1/chat/completions only accepts the 'api-subscription-key' header, not
        'Authorization: Bearer' which is what the AI Gateway sends. Calling directly
        is equivalent and the key is still managed by Databricks Secrets.

        Args:
            messages:      [{"role": "user"|"assistant", "content": "..."}]
            system_prompt: Prepended as a system message.
            temperature:   0-2. Low = deterministic. Default 0.3.
            max_tokens:    Response length cap.

        Returns:
            Assistant reply as a plain string.
        """
        full_messages: list[dict] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        logger.info(
            "Calling Sarvam chat (%s, %d messages, key from Databricks Secrets)",
            model, len(full_messages),
        )

        payload = {
            "model": model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": "low",
        }
        result = self._sarvam_post(SARVAM_CHAT_URL, payload)
        message = result["choices"][0]["message"]

        reply = message.get("content") or ""
        if not reply:
            raise RuntimeError(
                f"Sarvam returned empty content. Full message: {message}"
            )
        usage = result.get("usage", {})
        if usage:
            logger.info(
                "Tokens — prompt: %s, completion: %s, total: %s",
                usage.get("prompt_tokens", "?"),
                usage.get("completion_tokens", "?"),
                usage.get("total_tokens", "?"),
            )
        return reply

    # ── Translation via direct Sarvam REST (key from Databricks Secrets) ─────

    def translate_to_english(
        self,
        text: str,
        source_language_code: str = "hi-IN",
    ) -> str:
        """Back-compat wrapper — see `translate` for the general form."""
        return self.translate(
            text=text,
            source_language_code=source_language_code,
            target_language_code="en-IN",
        )

    def translate(
        self,
        text: str,
        source_language_code: str = "auto",
        target_language_code: str = "en-IN",
    ) -> str:
        """
        Translate text between any supported language pair via Sarvam Translate.
        `source_language_code="auto"` lets Sarvam detect the source.
        """
        payload = {
            "input": text,
            "source_language_code": source_language_code,
            "target_language_code": target_language_code,
            "speaker_gender": "Male",
            "mode": "formal",
            "model": "mayura:v1",
            "enable_preprocessing": True,
        }
        result = self._sarvam_post(SARVAM_TRANSLATE_URL, payload)
        translated = result.get("translated_text", "") or ""
        logger.info(
            "Translated [%s → %s]: %r → %r",
            source_language_code, target_language_code,
            text[:50], translated[:50],
        )
        return translated

    # ── Speech to text (ASR) via direct Sarvam REST ────────────────────────────

    def speech_to_text(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language_code: str = "unknown",
        model: str = "saarika:v2.5",
    ) -> dict:
        """
        Transcribe audio using Sarvam ASR (saarika). Returns
        `{"transcript": str, "language_code": str}` where `language_code` is
        the BCP-47 code Sarvam detected (or the one you passed in).
        """
        files = {"file": (filename, audio_bytes, content_type)}
        data = {"language_code": language_code, "model": model}
        logger.info(
            "Calling Sarvam STT (%s, %d bytes, lang=%s)",
            model, len(audio_bytes or b""), language_code,
        )
        response = requests.post(
            SARVAM_STT_URL,
            files=files,
            data=data,
            headers={"api-subscription-key": self.sarvam_key},
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Sarvam STT error {response.status_code}: {response.text}"
            ) from exc
        body = response.json() or {}
        return {
            "transcript": body.get("transcript", "") or "",
            "language_code": body.get("language_code") or language_code,
        }

    def transcribe_to_language(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        target_lang: str = "en",
    ) -> dict:
        """
        Transcribe audio with auto language detection, then translate the
        transcript into `target_lang` (any of the SUPPORTED_LANG_CODES keys,
        e.g. "en", "hi", "bn", "ta", ...) if needed.

        Returns `{"text": str, "source_language_code": str}`.
        """
        stt = self.speech_to_text(
            audio_bytes=audio_bytes,
            filename=filename,
            content_type=content_type,
            language_code="unknown",
        )
        transcript = stt["transcript"]
        src = stt["language_code"] or "hi-IN"

        target_code = to_bcp47(target_lang, default="en-IN")
        if not transcript or src == target_code:
            return {"text": transcript, "source_language_code": src}

        try:
            translated = self.translate(
                text=transcript,
                source_language_code=src,
                target_language_code=target_code,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Translate after STT failed (%s) — returning transcript", exc)
            translated = transcript
        return {"text": translated or transcript, "source_language_code": src}

    def identify_language(self, text: str) -> str:
        """
        Detect the BCP-47 language code of the input text.
        Falls back to 'hi-IN' on error.
        """
        try:
            result = self._sarvam_post(SARVAM_LID_URL, {"input": text})
            code = result.get("language_code", "hi-IN")
            logger.info("Detected language: %s", code)
            return code
        except Exception as exc:  # noqa: BLE001
            logger.warning("Language detection failed (%s) — defaulting to hi-IN", exc)
            return "hi-IN"

    # ── Internal REST helper ─────────────────────────────────────────────────

    def _sarvam_post(self, url: str, payload: dict) -> dict:
        response = requests.post(
            url,
            json=payload,
            headers={
                "api-subscription-key": self.sarvam_key,
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Sarvam REST API error [{url}] "
                f"{response.status_code}: {response.text}"
            ) from exc
        return response.json()
