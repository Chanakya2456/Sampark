"""
verify.py
---------
End-to-end verification script for the Rail Madad complaint pipeline.

Architecture under test:
  LLM calls   → w.serving_endpoints.query("sarvam-30b-gateway")
                    ↳ Databricks AI Gateway → Sarvam /v1/chat/completions
  REST calls  → Sarvam /translate + /text-lid directly
                    ↳ Key fetched from Databricks Secret Scope 'bharatbricks'

Prereqs:
  1. python setup_endpoint.py          # register Sarvam as Databricks endpoint
  2. export SARVAM_API_KEY=sk_xxx      # only for local dev (not needed on Databricks)

Run:
    python verify.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Callable

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify")

# ── Test harness ──────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    detail: str = ""
    error: str = ""


@dataclass
class Suite:
    results: list[TestResult] = field(default_factory=list)

    def run(self, name: str, fn: Callable) -> TestResult:
        print(f"\n{'─'*60}")
        print(f"▶  {name}")
        start = time.monotonic()
        r = TestResult(name=name, passed=False, duration_ms=0.0)
        try:
            detail = fn()
            r.passed = True
            r.detail = str(detail or "")
        except Exception as exc:  # noqa: BLE001
            r.error = f"{type(exc).__name__}: {exc}"
            logger.error("FAIL — %s", r.error)
        r.duration_ms = (time.monotonic() - start) * 1000
        self.results.append(r)
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"   {status}  ({r.duration_ms:.0f} ms)")
        if r.detail:
            wrapped = textwrap.indent(
                textwrap.fill(r.detail, width=72), prefix="   │  "
            )
            print(wrapped)
        if r.error:
            print(f"   └─ {r.error}")
        return r

    def summary(self) -> int:
        """Print summary table; return exit code (0=all pass, 1=some fail)."""
        print(f"\n{'═'*60}")
        print("  SUMMARY")
        print(f"{'═'*60}")
        passed = sum(1 for r in self.results if r.passed)
        total  = len(self.results)
        for r in self.results:
            icon = "✅" if r.passed else "❌"
            print(f"  {icon} {r.name:<40} {r.duration_ms:>6.0f} ms")
        print(f"{'─'*60}")
        print(f"  {passed}/{total} tests passed")
        print(f"{'═'*60}\n")
        return 0 if passed == total else 1


# ── Individual test functions ─────────────────────────────────────────────────

def test_databricks_sdk_available() -> str:
    """Confirm databricks-sdk is importable."""
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415,F401
    from databricks.sdk.service.compute import State  # noqa: PLC0415,F401
    return "databricks-sdk imported successfully"


def test_databricks_identity() -> str:
    """
    Use the Databricks SDK to fetch the current workspace identity.
    Works automatically inside Databricks Apps / Jobs (uses env-injected token).
    Falls back gracefully when running locally without a configured profile.
    """
    try:
        from databricks.sdk import WorkspaceClient  # noqa: PLC0415
        w = WorkspaceClient()
        me = w.current_user.me()
        return f"Authenticated as: {me.user_name} | workspace: {w.config.host}"
    except Exception as exc:  # noqa: BLE001
        # Not fatal — local dev without Databricks creds
        return f"Databricks auth not configured (local dev mode): {exc}"


def test_databricks_secret_scope() -> str:
    """
    Verify the secret scope 'bharatbricks' exists and contains 'sarvam_api_key'.
    This key is used for the non-OpenAI REST endpoints (translate, lid).
    """
    try:
        from databricks.sdk import WorkspaceClient  # noqa: PLC0415
        w = WorkspaceClient()
        secret = w.secrets.get_secret(scope="bharatbricks", key="sarvam_api_key")
        if not secret.value:
            raise ValueError("Secret value is empty")
        masked = secret.value[:6] + "..." + secret.value[-4:]
        return f"Secret key found. Value (masked): {masked}"
    except Exception as exc:  # noqa: BLE001
        env_key = os.environ.get("SARVAM_API_KEY", "")
        if env_key:
            masked = env_key[:6] + "..." + env_key[-4:]
            return (
                f"Databricks secret not accessible ({exc}). "
                f"Falling back to env var SARVAM_API_KEY (masked): {masked}"
            )
        raise RuntimeError(
            "No Sarvam API key found. Set SARVAM_API_KEY env var or "
            "configure Databricks secret scope 'bharatbricks/sarvam_api_key'."
        ) from exc


def test_sarvam_api_reachable() -> str:
    """
    Confirm the Sarvam API base URL is reachable and the key is valid
    by calling the language-id endpoint with a trivial payload.
    API key fetched from Databricks Secrets.
    """
    from sarvam_client import SarvamClient  # noqa: PLC0415
    client = SarvamClient()
    # A minimal call — cheap (no LLM involved)
    detected = client.identify_language("hello")
    assert detected, "Empty language code returned"
    masked = client.sarvam_key[:6] + "..." + client.sarvam_key[-4:]
    return f"Sarvam API reachable. Key (masked): {masked} | Detected: {detected}"


def test_sarvam_client_init() -> str:
    """Instantiate SarvamClient — confirms WorkspaceClient auth works."""
    from sarvam_client import SarvamClient  # noqa: PLC0415
    client = SarvamClient()
    host = client.w.config.host or "(local)"
    return f"SarvamClient initialised. Databricks host: {host}"


def test_language_detection_hindi() -> str:
    from sarvam_client import SarvamClient  # noqa: PLC0415
    client = SarvamClient()
    sample = "मेरी ट्रेन बहुत देर से चल रही है।"
    detected = client.identify_language(sample)
    assert detected, "Empty language code returned"
    return f"Input: '{sample}' → detected: {detected}"


def test_translate_hindi_to_english() -> str:
    from sarvam_client import SarvamClient  # noqa: PLC0415
    client = SarvamClient()
    hindi_text = "ट्रेन में खाना बेकार था और पानी भी नहीं था। S3 कोच में बहुत गंदगी है।"
    translated = client.translate_to_english(hindi_text, source_language_code="hi-IN")
    assert translated, "Translation returned empty string"
    assert len(translated) > 10, f"Translation suspiciously short: {translated!r}"
    return f"Hindi → English:\n  IN:  {hindi_text}\n  OUT: {translated}"


def test_translate_tamil_to_english() -> str:
    from sarvam_client import SarvamClient  # noqa: PLC0415
    client = SarvamClient()
    tamil_text = "என் ரயில் 3 மணி நேரம் தாமதமாக உள்ளது. உணவு தரமற்றது."
    translated = client.translate_to_english(tamil_text, source_language_code="ta-IN")
    assert translated, "Translation returned empty string"
    return f"Tamil → English:\n  IN:  {tamil_text}\n  OUT: {translated}"


def test_sarvam_chat() -> str:
    """
    Calls sarvam-30b chat directly via Sarvam REST API.
    API key is fetched from Databricks Secrets (SDK).
    """
    from sarvam_client import SarvamClient, CHAT_MODEL  # noqa: PLC0415
    client = SarvamClient()
    reply = client.chat_completion(
        messages=[{"role": "user", "content": "What is Rail Madad and how do I file a complaint?"}],
        system_prompt="You are a Rail Madad grievance assistant. Respond in English. Keep it brief.",
        temperature=0.4,
        max_tokens=150,
    )
    assert reply and len(reply) > 20, f"Short/empty reply: {reply!r}"
    return f"[{CHAT_MODEL}] Reply: {reply[:200]}"


def test_complaint_generation_hindi() -> str:
    from complaint_engine import ComplaintEngine, ISSUE_TYPES  # noqa: PLC0415
    engine = ComplaintEngine()
    result = engine.generate(
        raw_input="मेरी ट्रेन 12 घंटे लेट है। बच्चे भूखे हैं और पानी नहीं है। प्लेटफार्म पर कोई मदद नहीं।",
        issue_type="delay",
        train_number="12345",
        pnr="1234567890",
        language_code="hi-IN",   # force to skip auto-detect API call in test
    )
    assert result.formatted_complaint, "Empty complaint"
    assert result.sms_uri.startswith("sms:139"), "SMS URI malformed"
    assert result.char_count <= 160 or result.truncated, "Complaint exceeds 160 chars without truncation flag"
    return (
        f"Complaint ({result.char_count} chars):\n  {result.formatted_complaint}\n"
        f"SMS URI prefix: {result.sms_uri[:80]}..."
    )


def test_complaint_generation_bengali() -> str:
    from complaint_engine import ComplaintEngine  # noqa: PLC0415
    engine = ComplaintEngine()
    result = engine.generate(
        raw_input="খাবার পচা ছিল, আমার পেট খারাপ হয়ে গেছে। কোচ S2 এ অনেক ময়লা।",
        issue_type="food",
        train_number="12530",
        language_code="bn-IN",
    )
    assert result.formatted_complaint, "Empty complaint"
    return (
        f"Bengali complaint ({result.char_count} chars):\n  {result.formatted_complaint}\n"
        f"Notes: {result.engine_notes}"
    )


def test_sms_uri_encoding() -> str:
    """Verify SMS URI has correct MADAD <complaint> PNR <pnr> format."""
    from complaint_engine import _build_sms_uri  # noqa: PLC0415
    complaint = "Delay Train 12345 | Train 12h late, children hungry, no water."
    pnr = "1234567890"
    uri = _build_sms_uri(complaint, pnr=pnr)
    assert uri.startswith("sms:139?body="), f"Unexpected URI prefix: {uri}"
    assert "MADAD" in uri, "Body missing MADAD prefix"
    assert "PNR" in uri, "Body missing PNR suffix"
    assert " " not in uri, "Spaces found in URI — not URL-encoded"
    return f"URI (first 120 chars): {uri[:120]}"


def test_chatbot_hindi() -> str:
    from complaint_engine import ComplaintEngine  # noqa: PLC0415
    engine = ComplaintEngine()
    reply = engine.chat(
        "मुझे Rail Madad पर कैसे शिकायत करनी है?",
        conversation_history=[],
    )
    assert reply and len(reply) > 10, f"Short/empty reply: {reply!r}"
    return f"Chatbot reply (first 200 chars): {reply[:200]}"


def test_full_pipeline_json_output() -> str:
    """
    Full pipeline test — prints the final ComplaintResult as JSON.
    This is the integration test that exercises all components together.
    """
    from complaint_engine import ComplaintEngine  # noqa: PLC0415
    engine = ComplaintEngine()
    result = engine.generate(
        raw_input="बिजली नहीं है पूरी बोगी में। रात को सफर कर रहे हैं, बहुत गर्मी है।",
        issue_type="electricity",
        train_number="11045",
        pnr="9876543210",
        # language_code not set → auto-detect
    )
    output = result.to_dict()
    json_str = json.dumps(output, ensure_ascii=False, indent=2)
    print(f"\n   Full JSON result:\n{textwrap.indent(json_str, '   ')}")
    return f"Pipeline complete. Char count: {result.char_count}"


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Rail Madad Assistant — End-to-End Verification     ║")
    print("╚══════════════════════════════════════════════════════╝")

    suite = Suite()

    # ── Databricks SDK / infrastructure (no API cost) ─────────────────────────
    suite.run("1.  databricks-sdk importable",       test_databricks_sdk_available)
    suite.run("2.  Databricks workspace identity",   test_databricks_identity)
    suite.run("3.  Databricks secret scope",         test_databricks_secret_scope)
    suite.run("4.  SarvamClient initialisation",     test_sarvam_client_init)

    # ── Sarvam API — key fetched from Databricks Secrets ─────────────────────
    suite.run("5.  Sarvam API reachable + key valid", test_sarvam_api_reachable)
    suite.run("6.  Language detection (Hindi)",       test_language_detection_hindi)
    suite.run("7.  Translation Hindi → English",      test_translate_hindi_to_english)
    suite.run("8.  Translation Tamil → English",      test_translate_tamil_to_english)
    suite.run("9.  Chat — sarvam-30b",                test_sarvam_chat)

    # ── Complaint engine ──────────────────────────────────────────────────────
    suite.run("10. SMS URI encoding",                 test_sms_uri_encoding)
    suite.run("11. Complaint gen (Hindi/delay)",      test_complaint_generation_hindi)
    suite.run("12. Complaint gen (Bengali/food)",     test_complaint_generation_bengali)
    suite.run("13. Chatbot (Hindi)",                 test_chatbot_hindi)
    suite.run("14. Full pipeline + JSON output",      test_full_pipeline_json_output)

    sys.exit(suite.summary())


if __name__ == "__main__":
    main()
