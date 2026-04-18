<div align="center">

# 🚆 SAMPARK — सम्पर्क

```
░██████╗░█████╗░███╗░░░███╗██████╗░░█████╗░██████╗░██╗░░██╗
██╔════╝██╔══██╗████╗░████║██╔══██╗██╔══██╗██╔══██╗██║░██╔╝
╚█████╗░███████║██╔████╔██║██████╔╝███████║██████╔╝█████═╝░
░╚═══██╗██╔══██║██║╚██╔╝██║██╔═══╝░██╔══██║██╔══██╗██╔═██╗░
██████╔╝██║░░██║██║░╚═╝░██║██║░░░░░██║░░██║██║░░██║██║░╚██╗
╚═════╝░╚═╝░░╚═╝╚═╝░░░░░╚═╝╚═╝░░░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝
```

### *One Stop. Every Grievance. Gone.*
#### **Agentic AI-Powered Railway Grievance Resolution for Rural India**

<br/>

[![Built with Databricks](https://img.shields.io/badge/Powered%20by-Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)](https://databricks.com)
[![Sarvam AI](https://img.shields.io/badge/Language-Sarvam%20API-6C3BF5?style=for-the-badge&logo=translate&logoColor=white)](https://sarvam.ai)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-FF6600?style=for-the-badge&logo=python&logoColor=white)](https://xgboost.ai)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![10 Languages](https://img.shields.io/badge/Languages-10%2B%20Supported-00C851?style=for-the-badge&logo=googletranslate&logoColor=white)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br/>

> **700 million** railway passengers. **Millions** of unresolved complaints.
> **One** agentic AI to handle them all.

<br/>

---

</div>

## 🔥 What the Hell is Sampark?

Forget chatbots. Forget helplines. Forget filing a complaint and praying to god it gets resolved before your hair turns grey.

**Sampark** is an end-to-end **agentic AI system** that doesn't just *listen* to your railway grievance — it *understands* it, *reasons* about it, and **resolves it** — automatically, in your language, whether you're in Mumbai or the remotest village in Bihar.

Built for the **1.4 billion Indians** riding Indian Railways, with a laser focus on **rural accessibility** — because world-class service shouldn't be a privilege of the English-speaking, app-downloading elite.

---

## 🧠 The Agent Squad — Meet Your Problem Solvers

Sampark runs a **multi-agent orchestration pipeline** where specialized AI agents collaborate to handle any grievance — from a basic refund query to a full automated complaint submission, no forms required.

```
┌──────────────────────────────────────────────────────────────────────┐
│                           USER INPUT                                  │
│          📸 Ticket Photo  /  🔢 PNR  /  🎙️ Voice  /  ⌨️ Text        │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       👁️  VISION AGENT                                │
│           Parses ticket images → Extracts all train details           │
│           (Databricks-deployed vision model · endpoint inference)     │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    🎯  MAIN ORCHESTRATOR AGENT                         │
│          Understands intent → Routes to the right specialist          │
└─────────────┬────────────────────────────────────┬───────────────────┘
              │                                    │
              ▼                                    ▼
┌─────────────────────────┐          ┌─────────────────────────────────┐
│    📚  QnA AGENT         │          │    ⚡  AUTO GRIEVANCE RESOLVER   │
│                          │          │                                  │
│  RAG over Railway rules  │          │  Understands the problem in      │
│  + Refund regulations    │          │  depth → Auto-drafts & submits   │
│  + Live web search for   │          │  refund request via SMS with     │
│  real-time queries       │          │  user review before submit       │
└─────────────────────────┘          └─────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    📊  PREDICTIVE DELAY AGENT                          │
│     XGBoost model trained on historical grievance data                │
│     → Predicts how long before your complaint gets resolved           │
│     → No more "we'll look into it" without a timeline                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Feature Breakdown

### `[01]` 📸 Smart Ticket Ingestion
- **Upload a photo** of your physical paper ticket OR simply **enter your PNR**
- A **Databricks-deployed Vision Model** reads the ticket like a hawk — train number, seat class, journey date, passenger details, PNF status, the works
- OCR + semantic understanding = **zero manual data entry** for the user
- Works even with crumpled, low-light ticket photos

---

### `[02]` 🎯 Master Orchestrator
The brain of the entire system — a **LangGraph-powered orchestrator agent** that:
- Classifies user intent (query? complaint? refund? status check?)
- Decides which downstream agent gets the task
- Maintains **full context memory** across multi-turn conversations
- Handles ambiguous inputs gracefully with smart fallbacks

---

### `[03]` 📚 QnA Agent — Know Everything Railway, Instantly
Two modes, one powerhouse:

| Mode | What It Does |
|------|-------------|
| 🔎 **RAG Mode** | Semantic search over Databricks Vector Store containing Indian Railway rulebooks — TDR policies, refund regulations, cancellation charts, fare rules, quota policies |
| 🌐 **Web Search Mode** | Real-time answers for train running status, PNR status, platform info — anything outside the static knowledge corpus |

The agent decides on-the-fly which mode to use (or both) based on the query type.

---

### `[04]` ⚡ Auto Grievance Resolver — The Star of the Show

This is where Sampark **goes beyond** every other chatbot ever built for Indian Railways:

```
1. 🧠  Agent deeply understands the user's complaint
2. 📖  Pulls relevant policy context via RAG
          → Is the user eligible for a refund?
          → What form/category should be filed?
          → What's the SLA?
3. 📝  Auto-drafts a complete, accurate complaint/refund request
4. 📲  Sends an SMS preview to the user for approval
5. ✅  On user approval → Submits the request automatically
          No forms. No portals. No queues.
```

> *"File a TDR grievance in under 60 seconds. Without touching a single website."*

---

### `[05]` 🌍 Built for Bharat — Radical Inclusivity

Because 500+ million Indians don't communicate in English — and **deserve the same access**:

| Feature | Details |
|---------|---------|
| 🗣️ **10+ Regional Languages** | Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Odia, Punjabi |
| 🎙️ **Speech-to-Text (STT)** | Sarvam API — speak your complaint in your language, we understand it perfectly |
| 🔊 **Text-to-Speech (TTS)** | Responses are read back aloud — designed for low-literacy users |
| 📱 **SMS-First Workflow** | No app needed. No data plan needed. Works on a ₹800 Nokia. |

This isn't an accessibility checkbox. **This is the core design principle.**

---

### `[06]` 📊 Predictive Resolution Agent
- **XGBoost classifier** trained on historical Indian Railways grievance resolution data stored in **Databricks Delta Tables**
- **Inputs**: complaint type, railway zone, issue category, time of filing, priority level
- **Output**: Predicted resolution time in days with confidence interval
- **MLOps**: Fully tracked with **Databricks MLflow** — every model version logged, metrics captured, auto-retrain triggered on data drift detection
- **Why it matters**: Sets realistic expectations upfront → Builds genuine trust with rural users who've been gaslit by the system their whole lives

---

## 🛠️ Tech Stack — The Full Arsenal

```
╔═══════════════════════════════════════════════════════════════════════╗
║                          🏗️  INFRASTRUCTURE                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   🟠  DATABRICKS                                                      ║
║       ├── Delta Tables        →  Historical grievance data storage    ║
║       ├── Vector Search       →  Embedding store + semantic RAG       ║
║       ├── Delta Lake Indexing →  Real-time index sync & updates       ║
║       ├── MLflow              →  Model tracking, versioning,          ║
║       │                          experiment logging, auto-retrain     ║
║       └── Model Serving       →  Vision model deployment endpoint     ║
║                                                                       ║
║   🟣  SARVAM AI                                                       ║
║       ├── STT                 →  Voice input, 10 Indian languages     ║
║       ├── TTS                 →  Audio response for low-lit users     ║
║       └── Translation         →  Cross-lingual semantic bridge        ║
║                                                                       ║
║   🟡  ML / AI LAYER                                                   ║
║       ├── XGBoost             →  Resolution delay predictor           ║
║       ├── LangGraph           →  Multi-agent orchestration graph      ║
║       ├── LangChain           →  Tool use, RAG chains, memory         ║
║       └── Vision LLM          →  Ticket OCR & structured parsing      ║
║                                                                       ║
║   🔵  COMMUNICATION                                                   ║
║       └── SMS Gateway         →  User-approved auto-submission        ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
sampark/
│
├── 🎯 orchestrator/                # Main intent routing agent
│   ├── agent.py                    # LangGraph orchestrator graph
│   └── tools.py                    # Agent tool definitions
│
├── 👁️ vision_agent/                # Ticket image parsing
│   ├── parser.py                   # Image → structured train data
│   └── databricks_endpoint.py      # Vision model API calls
│
├── 📚 qna_agent/                   # QnA with RAG + live web search
│   ├── rag_pipeline.py             # RAG chain implementation
│   ├── vector_search.py            # Databricks Vector Search client
│   └── web_search.py               # Real-time web search tool
│
├── ⚡ grievance_resolver/           # Auto complaint filing agent
│   ├── resolver_agent.py           # Core resolution agent logic
│   ├── sms_gateway.py              # SMS preview + submission
│   └── policy_checker.py           # RAG-based eligibility check
│
├── 📊 predictive_agent/            # XGBoost delay predictor
│   ├── model.py                    # Inference wrapper
│   ├── train.py                    # Training pipeline
│   └── mlflow_logger.py            # Experiment tracking
│
├── 🌍 language/                    # Sarvam multilingual stack
│   ├── stt.py                      # Speech-to-Text pipeline
│   ├── tts.py                      # Text-to-Speech pipeline
│   └── translate.py                # Input/output translation
│
├── 🗄️ data/                        # Delta table schemas & seeds
├── 📝 notebooks/                   # Databricks training notebooks
├── 🧪 tests/                       # Test suite
├── .env.example
├── requirements.txt
└── README.md
```

---

## ⚡ Getting Started

### Prerequisites
```bash
Python 3.10+
Databricks workspace with Unity Catalog enabled
Sarvam AI API key
SMS Gateway credentials
```

### Installation
```bash
git clone https://github.com/yourteam/sampark.git
cd sampark
pip install -r requirements.txt
cp .env.example .env
```

### Environment Variables
```env
# Databricks
DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
DATABRICKS_TOKEN=your_personal_access_token
VECTOR_SEARCH_ENDPOINT=your_vector_search_endpoint_name
VISION_MODEL_ENDPOINT=your_vision_serving_endpoint

# Sarvam AI
SARVAM_API_KEY=your_sarvam_api_key

# SMS Gateway
SMS_GATEWAY_KEY=your_sms_gateway_key
SMS_GATEWAY_URL=your_sms_gateway_url
```

### Run
```bash
# Launch the Sampark agent API
python -m sampark.app

# Train / retrain the predictive model (Databricks notebook)
databricks jobs run-now --job-id <predictive_model_job_id>

# Trigger RAG index rebuild
python -m sampark.data.rebuild_index
```

---

## 🧪 End-to-End Flow — Watch It Work

```
👤  Rural user speaks in Hindi:
    "बाबू, मेरी ट्रेन 5 घंटे लेट थी, पैसे वापस मिलेंगे?"

         │
         ▼  [Sarvam STT converts speech → text]

🔤  Transcribed: "मेरी ट्रेन 5 घंटे लेट थी, पैसे वापस मिलेंगे?"

         │
         ▼  [Orchestrator: Intent = Refund Query + Potential Grievance]

         ├──► 📚 QnA Agent (RAG): "TDR Rule 54A — refund eligible for delays > 3 hrs"
         │
         ├──► ⚡ Auto Resolver: Drafts TDR application
         │                     → "Review your complaint. Reply YES to submit."
         │                     → SMS sent to user's number 📲
         │
         └──► 📊 Predictive Agent: "Expected resolution: 7–10 business days"

         │
         ▼  [Sarvam TTS converts response → Hindi audio]

🔊  User hears complete answer in their language. Done. 🎉
```

---

## 📈 The Impact We're Chasing

| Metric | Current Reality | With Sampark |
|--------|----------------|--------------|
| Avg. complaint resolution time | 30–45 days | **< 10 days** (predicted) |
| Rural users who can file digitally | ~5% | **100%** |
| Language support | English + Hindi only | **10 Indian languages** |
| Manual form filling required | Always | **Never** |
| Ticket parsing time | 10 minutes manually | **< 5 seconds** |
| Awareness of refund eligibility | Near zero | **Instant, automated** |

---

## 🏆 Why Sampark Wins

> Sampark is not another chatbot slapped on a government website.

- ✅ **Truly Agentic** — multi-agent, tool-using, self-routing, memory-aware system
- ✅ **RAG done right** — Databricks Vector Search + Delta Lake, not a weekend demo
- ✅ **Real inclusivity** — not a checkbox, the literal core architecture
- ✅ **MLOps baked in** — MLflow experiment tracking, drift detection, auto-retraining pipeline
- ✅ **Production-grade vision** — Databricks-served model, not a colab notebook
- ✅ **User consent first** — nothing submitted without explicit SMS approval
- ✅ **Offline-first comms** — SMS workflow means zero dependency on smartphones or internet

---

## 👥 The Team

> Built on chai, GitHub commits, and the quiet rage of watching grandparents struggle with IRCTC.

<!-- Add your team members here -->
<!-- | Name | Role | GitHub | -->
<!-- |------|------|--------| -->

---

## 📜 License

MIT License — because knowledge should be free.
Just like railways should work on time.

---

<div align="center">

**🚆 Sampark — सम्पर्क**

*"Because every Indian deserves to be heard — in their own language, on their own terms."*

<br/>

⭐ Star this repo if you believe technology should work for everyone, not just the privileged few.

</div>
