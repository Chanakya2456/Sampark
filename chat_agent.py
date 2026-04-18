"""
chat_agent.py
-------------
Rail Madad grievance chat agent served through the same Databricks model
serving endpoint as the complaint / ticket pipelines.

Architecture (matches the reference notebook in `chatbot/main (1).ipynb`):

    User query + user_id
           │
           ▼
    ┌───────────────────┐   tools
    │  Sarvam routing   │──────────────────────────┐
    │  (sarvam-30b)     │                          │
    └───────────────────┘                          │
           │                                       ▼
           │                 ┌─────────────────────────────────────┐
           │                 │ Tools available to the agent:       │
           │                 │  • get_user_journey(user_id)        │
           │                 │  • submit_grievance(...)            │
           │                 │  • rag_search(query)                │
           │                 │  • web_search(query)                │
           │                 └─────────────────────────────────────┘
           ▼
    ┌───────────────────┐
    │  Sarvam generator │   (empathetic rural-India persona)
    │  (sarvam-30b)     │   composes final answer from tool context
    └───────────────────┘
           │
           ▼
      assistant reply
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import requests

# Sarvam-m is a reasoning model that emits chain-of-thought wrapped in
# <think>...</think> tags inside `content`/`reasoning_content`. We never want
# that meta-commentary surfaced to the elderly-facing UI, so strip it.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    if not text:
        return text
    return _THINK_RE.sub("", text).strip()

from sarvam_client import SarvamClient, CHAT_MODEL, GENERATOR_MODEL

# (name, native script) for the persona prompt. Order matches the UI toggle.
LANGUAGE_INFO: dict[str, tuple[str, str]] = {
    "en": ("English", "Latin"),
    "hi": ("Hindi", "Devanagari"),
    "bn": ("Bengali", "Bengali"),
    "ta": ("Tamil", "Tamil"),
    "te": ("Telugu", "Telugu"),
    "mr": ("Marathi", "Devanagari"),
    "gu": ("Gujarati", "Gujarati"),
    "kn": ("Kannada", "Kannada"),
    "ml": ("Malayalam", "Malayalam"),
    "pa": ("Punjabi", "Gurmukhi"),
}

logger = logging.getLogger(__name__)

# ── Config (all overridable via env) ─────────────────────────────────────────
VS_ENDPOINT_NAME = os.environ.get("VECTOR_SEARCH_ENDPOINT", "try1")
VS_INDEX_NAME = os.environ.get(
    "VECTOR_SEARCH_INDEX", "workspace.rail_adhikar.rules_vs_index"
)
JOURNEY_TABLE = os.environ.get("JOURNEY_TABLE", "workspace.default.journey")

TAVILY_URL = "https://api.tavily.com/search"


# ── Tool schemas (OpenAI-compatible, consumed by sarvam-30b) ─────────────────
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_user_journey",
            "description": (
                "Fetch the most recent booking/journey details (PNR, train number, "
                "train name, source, destination, date, class, seat) for the current "
                "user. Call this whenever the user asks about 'my train', 'my ticket', "
                "'my booking', or when you need PNR/train context to answer."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_grievance",
            "description": (
                "Submit an official Rail Madad grievance on behalf of the user. Only "
                "call this AFTER you have collected: (1) a clear description of the "
                "problem, (2) the issue category, and (3) a train number / PNR (use "
                "get_user_journey first if needed). Returns a ready-to-send SMS link "
                "to Rail Madad (139)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Passenger's description of the problem, any language.",
                    },
                    "issue_type": {
                        "type": "string",
                        "description": (
                            "One of: delay, food, cleanliness, staff, safety, medical, "
                            "water, electricity, bedroll, other. "
                            "Use 'electricity' for AC not working, fan, lights, or any "
                            "electrical/air-conditioning issue. "
                            "Use 'cleanliness' for dirty coach/toilet. "
                            "Use 'food' for catering complaints. "
                            "Use 'delay' for late running trains. "
                            "Use 'staff' for misbehaviour by railway staff. "
                            "Use 'safety' for security threats. "
                            "Use 'medical' for health emergencies. "
                            "Only use 'other' if none of the above fit."
                        ),
                    },
                    "train_number": {"type": "string"},
                    "pnr": {"type": "string"},
                    "language_code": {
                        "type": "string",
                        "description": "BCP-47 code e.g. hi-IN, ta-IN, en-IN. Optional.",
                    },
                },
                "required": ["description", "issue_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the official Indian Railways rulebook (G&SR, IRCTC cancellation "
                "rules, refund rules) for questions about ticketing rules, refund "
                "policies, cancellation fees, TDR filing, and passenger grievances."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific rule/policy to look up.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "General web search via Tavily. Use for live information: city Metro "
                "timings, weather, news, or anything not in the Railways rulebook."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."}
                },
                "required": ["query"],
            },
        },
    },
]


# ── Prompts ──────────────────────────────────────────────────────────────────
ROUTING_SYSTEM_PROMPT = """You are a tool-dispatch agent for the Rail Madad grievance assistant.
Your ONLY job is to decide which tools to call — do NOT answer the user yourself.

Use the FULL conversation (including earlier user turns) to infer intent. If
the latest user message is a short follow-up like "generate it / file it /
yes, do it / इसे भेजदो / जनरेट करो", and the earlier turns already describe
a grievance (AC, delay, food, cleanliness, staff, electricity, etc.), treat
this as a confirmation and call `submit_grievance` now.

Mandatory rules — follow exactly:

1. COMPLAINT / GRIEVANCE (any problem with the train, coach, staff, food,
   cleanliness, AC, delay, safety, medical, water, bedroll, electricity, etc.)
   → You MUST call `submit_grievance` — this is the ONLY way to raise a
     complaint. Never describe how to file manually; never say "call 139" or
     "visit the station". If train number / PNR is not yet known, call
     `get_user_journey` first in this round; then call `submit_grievance` in
     the next round once you have the journey details.

1a. SECOND-ROUND MANDATE — if the conversation already contains a tool result
    from `get_user_journey` (status=success) AND the user's request is a
    complaint, you MUST call `submit_grievance` in THIS round with:
      - description: the user's own words about the problem (from their
        earlier message, translated into English if needed),
      - issue_type: best-matching category,
      - train_number and pnr: from the journey tool result.
    Do NOT call `get_user_journey` again, and do NOT return zero tool calls.

2. JOURNEY DETAILS (user mentions their train, ticket, seat, PNR, booking,
   destination, or "my train")
   → Call `get_user_journey` to fetch the data.

3. LIVE / REAL-TIME information (weather, train running status, Metro timings,
   news, platform info, anything not in the static rulebook)
   → ALWAYS call `web_search`. Never claim you cannot search.
   → If the query depends on a LOCATION (weather, metro timings, local news,
     "mere jagah", "my city", "yahaan", "there"), infer the location from
     the journey context. If a `get_user_journey` tool result (or prior
     assistant message that mentions the train/PNR/source/destination) is
     already present in the conversation, build the `web_search` query using
     that location — typically the destination station, or source if the
     user says "where I'm departing from". Do NOT ask the user to re-type
     their city. If no journey is known yet, call `get_user_journey` first,
     then `web_search` in the next round.

4. RAILWAY RULES / POLICY (cancellation fees, refund, TDR, ticket rules)
   → Call `rag_search`.

5. PARALLEL calls — you MAY call multiple tools at once if the query covers
   multiple needs.

6. GREETINGS / SMALL-TALK with no actionable question — call no tools.
"""

PERSONA_SYSTEM_PROMPT = """You are 'Rail Adhikar Sahayak', an empathetic grievance-resolution
assistant for Indian Railways. Your users are elderly / rural passengers.

Tone:
- Empathetic and respectful. Use "Namaskar", "Sir/Madam", "Ji".
- Simple language; explain jargon (TDR, RAC) when needed.
- Concise — do NOT write long lists of manual steps.

Language:
- The user's preferred language is given at the bottom of this system prompt
  on a `Reply language:` line (e.g. `Reply language: Hindi (script: Devanagari)`).
- Write the ENTIRE reply — including the complaint summary line and the
  SMS-send instruction — in that language and its native script.
- Proper nouns (train names, station codes, PNR, SMS short-codes like 139,
  English railway acronyms like TDR/RAC/IRCTC) stay in their original form.

COMPLAINT RESPONSES — strict rules:
- If `submit_grievance` tool result is present in the conversation:
  • Confirm the complaint was filed on their behalf.
  • Prominently share the SMS link and say: "Please just press Send on this
    SMS and your complaint will be registered with Rail Madad (139)."
  • Give the complaint summary in 1–2 lines.
  • Do NOT add any other instructions (no app steps, no 139 call, no visit
    station).
- If a complaint was requested but `submit_grievance` was NOT called:
  • Apologise that the complaint could not be filed automatically and ask the
    user to describe their problem more clearly so it can be submitted.
  • Do NOT give manual filing instructions.

FACTUAL / RULE QUESTIONS:
- Base answers strictly on `rag_search` or `web_search` results.
- If the search returned no useful data, say so briefly and suggest calling
  the Rail Madad helpline 139 for that specific factual query only.

General grounding: never make up facts; cite tool results."""


# ── Agent ────────────────────────────────────────────────────────────────────

class ChatAgent:
    """
    Tool-using chat agent driven by Sarvam-30b, with Databricks-native
    Vector Search for RAG and a SQL-backed user-journey tool.

    The agent is instantiated once per model-serving replica (see
    `RailMadadModel.load_context`) so heavy clients (VectorSearchClient,
    SentenceTransformer) warm-start exactly once.
    """

    def __init__(self, sarvam_client: SarvamClient):
        self.sarvam = sarvam_client
        # Lazy init — VectorSearchClient is slow to construct
        self._vs_index = None
        # Complaint engine is lazy-initialised too to avoid import cycles on load
        self._complaint_engine = None

    # ── Lazy resource accessors ─────────────────────────────────────────────

    def _get_vs_index(self):
        if self._vs_index is None:
            from databricks.vector_search.client import VectorSearchClient  # noqa: PLC0415

            client = VectorSearchClient(disable_notice=True)
            self._vs_index = client.get_index(
                endpoint_name=VS_ENDPOINT_NAME,
                index_name=VS_INDEX_NAME,
            )
            logger.info("Vector Search index ready: %s", VS_INDEX_NAME)
        return self._vs_index

    def _get_complaint_engine(self):
        if self._complaint_engine is None:
            from complaint_engine import ComplaintEngine  # noqa: PLC0415

            self._complaint_engine = ComplaintEngine(client=self.sarvam)
        return self._complaint_engine

    # ── Tool implementations ────────────────────────────────────────────────

    def tool_get_user_journey(self, user_id: str) -> dict:
        """Return the latest journey row from the SQL journey table."""
        logger.info("[tool] get_user_journey user_id=%s", user_id)
        host = os.environ.get("DATABRICKS_HOST", "").replace("https://", "").rstrip("/")
        http_path = os.environ.get("DATABRICKS_SQL_HTTP_PATH", "")
        token = os.environ.get("DATABRICKS_TOKEN", "")
        if not (host and http_path and token and user_id):
            return {"status": "error", "message": "Journey lookup not available."}

        from databricks import sql  # noqa: PLC0415

        try:
            connection = sql.connect(
                server_hostname=host,
                http_path=http_path,
                access_token=token,
            )
            try:
                with connection.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT pnr, train_number, train_name, source_station,
                               destination_station, journey_date, departure_time,
                               arrival_time, `class`, seat_number
                        FROM {JOURNEY_TABLE}
                        WHERE user_id = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (user_id,),
                    )
                    row = cur.fetchone()
            finally:
                connection.close()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Journey lookup failed")
            return {"status": "error", "message": f"Journey lookup failed: {exc}"}

        if not row:
            return {"status": "not_found", "message": "No booking on file for this user."}

        cols = [
            "pnr", "train_number", "train_name", "source_station",
            "destination_station", "journey_date", "departure_time",
            "arrival_time", "class", "seat_number",
        ]
        return {"status": "success", "journey": dict(zip(cols, row))}

    def tool_submit_grievance(
        self,
        description: str,
        issue_type: str,
        train_number: str = "",
        pnr: str = "",
        language_code: Optional[str] = None,
    ) -> dict:
        """Generate a Rail Madad SMS via the existing ComplaintEngine."""
        logger.info("[tool] submit_grievance issue=%s train=%s pnr=%s",
                    issue_type, train_number, pnr)
        try:
            engine = self._get_complaint_engine()
            result = engine.generate(
                raw_input=description,
                issue_type=issue_type or "other",
                train_number=train_number or "",
                pnr=pnr or "",
                language_code=language_code,
            )
            return {
                "status": "success",
                "sms_uri": result.sms_uri,
                "formatted_complaint": result.formatted_complaint,
                "char_count": result.char_count,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Grievance submission failed")
            return {"status": "error", "message": f"Grievance submission failed: {exc}"}

    def tool_rag_search(self, query: str, k: int = 4) -> dict:
        """
        Query the Databricks Mosaic Vector Search index of railway rule PDFs.

        The index is built with `embedding_model_endpoint_name=...` so Databricks
        computes embeddings server-side — we just pass `query_text`.
        """
        logger.info("[tool] rag_search query=%r", query[:80])
        try:
            index = self._get_vs_index()
            results = index.similarity_search(
                query_text=query,
                columns=["text", "source", "page"],
                num_results=k,
            )
            rows = results.get("result", {}).get("data_array", []) or []
            docs = [
                {"text": r[0], "source": r[1], "page": r[2]}
                for r in rows
            ]
            return {"status": "success", "docs": docs}
        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG search failed")
            return {"status": "error", "message": f"RAG search failed: {exc}"}

    def tool_web_search(self, query: str) -> dict:
        """Tavily web search for general/live queries."""
        logger.info("[tool] web_search query=%r", query[:80])
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return {
                "status": "error",
                "message": "Web search is not configured (TAVILY_API_KEY missing).",
            }
        try:
            resp = requests.post(
                TAVILY_URL,
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": True,
                    "max_results": 5,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "status": "success",
                "answer": data.get("answer", ""),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                    }
                    for r in (data.get("results") or [])[:5]
                ],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Web search failed")
            return {"status": "error", "message": f"Web search failed: {exc}"}

    # ── Orchestration ───────────────────────────────────────────────────────

    def _dispatch_tool(self, name: str, args: dict, user_id: str) -> Any:
        """Route a tool_call to the corresponding Python implementation."""
        if name == "get_user_journey":
            return self.tool_get_user_journey(user_id)
        if name == "submit_grievance":
            return self.tool_submit_grievance(
                description=args.get("description", ""),
                issue_type=args.get("issue_type", "other"),
                train_number=args.get("train_number", ""),
                pnr=args.get("pnr", ""),
                language_code=args.get("language_code"),
            )
        if name == "rag_search":
            return self.tool_rag_search(args.get("query", ""))
        if name == "web_search":
            return self.tool_web_search(args.get("query", ""))
        return {"status": "error", "message": f"Unknown tool: {name}"}

    def respond(
        self,
        query: str,
        user_id: str = "",
        lang: str = "en",
        history: Optional[list[dict]] = None,
    ) -> dict:
        """
        Agentic pipeline:
          1) Routing call with tools=auto to decide which tools to run.
          2) Execute each tool call and append results to message history.
          3) Re-run routing with updated history (up to MAX_ROUNDS total).
             This lets the router call `get_user_journey` first, then
             `submit_grievance` in the next round once it has the PNR/train.
          4) Final generator call (empathetic persona) with tool_choice=none
             to compose a plain-text reply grounded in the tool results.

        `lang` ("en" | "hi") controls the language of the final user-facing
        reply only — tool routing and internal reasoning remain in English.
        """
        MAX_ROUNDS = 3
        tool_log: list[dict] = []

        # Build a clean history — only role/content user+assistant turns, capped
        # to the most recent few so the context window stays small.
        clean_history: list[dict] = []
        for turn in (history or [])[-8:]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                clean_history.append({"role": role, "content": content})

        messages: list[dict] = [
            {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
            *clean_history,
            {"role": "user", "content": query},
        ]

        # ── Agentic loop ────────────────────────────────────────────────────
        for round_num in range(MAX_ROUNDS):
            route_msg = self.sarvam.chat_completion_raw(
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=4096,
                model=CHAT_MODEL,
            )
            tool_calls = route_msg.get("tool_calls") or []
            if not tool_calls:
                break  # nothing more to do — proceed to generator

            messages.append({
                "role": "assistant",
                "content": route_msg.get("content") or "",
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                fn = tc.get("function", {}) or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._dispatch_tool(name, args, user_id)
                tool_log.append({"tool": name, "args": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
            logger.info("[agent] round %d complete — tools called: %s",
                        round_num + 1, [e["tool"] for e in tool_log])

        had_tool_calls = bool(tool_log)

        # ── Generator ───────────────────────────────────────────────────────
        # NOTE: Sarvam requires `tools` to be re-sent whenever the message
        # history contains role="tool" entries, otherwise it 400s with
        # "Tool messages found but no tools provided".
        # tool_choice="none" forces a plain-text reply (prevents the generator
        # from making another tool call and returning content=None).
        lang_code = (lang or "en").lower()
        name, script = LANGUAGE_INFO.get(lang_code, LANGUAGE_INFO["en"])
        messages[0] = {
            "role": "system",
            "content": (
                f"{PERSONA_SYSTEM_PROMPT}\n\n"
                f"Reply language: {name} (script: {script})"
            ),
        }
        final = self.sarvam.chat_completion_raw(
            messages=messages,
            tools=TOOLS if had_tool_calls else None,
            tool_choice="none" if had_tool_calls else None,
            temperature=0.4,
            max_tokens=4096,
            model=GENERATOR_MODEL,
        )
        raw_reply = final.get("content") or ""
        reply = _strip_think(raw_reply)

        # Surface the submit_grievance SMS link (if any) for the UI to use.
        sms_uri = ""
        for entry in tool_log:
            if entry["tool"] == "submit_grievance":
                res = entry.get("result") or {}
                if isinstance(res, dict) and res.get("sms_uri"):
                    sms_uri = res["sms_uri"]
                    break

        return {
            "reply": reply,
            "sms_uri": sms_uri,
            "tool_log": tool_log,
        }
