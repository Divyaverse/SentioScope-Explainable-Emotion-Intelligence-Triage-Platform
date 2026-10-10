# SentioScope: Explainable Emotion Intelligence & Triage Platform

An end-to-end, academic- and enterprise-grade Natural Language Processing (NLP) system that performs multi-class emotion classification, clinical/support distress triage, and token-level explainability (XAI) using Deep Sequence Modeling and LIME.

---

## 🌟 Overview & Key Highlights

Standard sentiment classifiers behave as opaque black boxes that fail on informal text (slang, emojis) and offer no visibility into their decisions. **SentioScope** bridges this gap by combining:
1. **Colloquial Normalization**: Preprocessing slang (`idk`, `fml`, `smh`) and demojizing Unicode emojis (`😭` $\rightarrow$ `crying face`) into semantic tokens.
2. **Deep Sequence Modeling**: A Bidirectional Gated Recurrent Unit (**Bi-GRU**) with custom 300-dimensional dense embeddings to process forward and backward syntactic context simultaneously.
3. **Model Interpretability (XAI via LIME)**: Real-time token-level feature attribution to highlight words that drive each classification.
4. **Clinical & Urgency Triage Scoring**: An automated heuristic calculating an Urgency Index (0–100) and risk tier (`CRITICAL`, `ELEVATED`, `NORMAL`) for distress monitoring.
5. **Full-Stack Persistence & Dockerized Delivery**: Powered by FastAPI Lifespan context management, SQLite with SQLAlchemy ORM, and single-command deployment via Docker Compose.

---

## 🏗️ Architecture & Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser UI)                             │
│  Interactive Single-Page Dashboard (Tailwind CSS + Chart.js via CDN)   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ HTTP / JSON
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND                                 │
│                                                                        │
│  POST /api/v1/analyze                                                  │
│  ┌───────────────────────┐     ┌──────────────────────────────────┐    │
│  │ 1. Text Normalizer    │ ──> │ 2. Vectorization & Padding       │    │
│  │    • Emoji Demojizer  │     │    • Keras Tokenizer             │    │
│  │    • Slang Expander   │     │    • Post-Padding (maxlen = 50)  │    │
│  └───────────────────────┘     └─────────────────┬────────────────┘    │
│                                                  ▼                     │
│  ┌───────────────────────┐     ┌──────────────────────────────────┐    │
│  │ 4. LIME Explainability│ <── │ 3. Deep Bi-GRU Neural Engine     │    │
│  │    • Local Saliency   │     │    • 300D Embeddings + Softmax   │    │
│  │    • Word Attributions│     │    • 6 Emotion Classes           │    │
│  └───────────┬───────────┘     └─────────────────┬────────────────┘    │
│              │                                   │                     │
│              └─────────────────┬─────────────────┘                     │
│                                ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ 5. Triage Engine (Urgency Index & Risk Tier Calculation)       │    │
│  └─────────────────────────────┬──────────────────────────────────┘    │
│                                ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ 6. Storage & Analytics (SQLite + SQLAlchemy ORM)               │    │
│  └────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Component | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Deep Learning** | TensorFlow / Keras | $\ge 2.15$ | Bidirectional GRU model inference, Tokenizer sequence padding |
| **Model Interpretability**| LIME | $\ge 0.2.0$ | Local Interpretable Model-agnostic Explanations for token attribution |
| **Backend Framework** | FastAPI | $\ge 0.110$ | Asynchronous REST API, Swagger UI (`/docs`), static file serving |
| **ASGI Server** | Uvicorn | $\ge 0.28$ | Production-grade asynchronous server |
| **Persistence / Database**| SQLAlchemy + SQLite | $\ge 2.0$ | Historical query persistence, triage logging, and session tracking |
| **Validation & Schema** | Pydantic v2 | $\ge 2.6$ | Strict request/response typing and error validation |
| **NLP Utilities** | `emoji`, `re` | $\ge 2.10$ | Emoji verbalization and colloquial chat acronym expansion |
| **Testing** | Pytest + HTTPX | $\ge 8.0$ | Automated API, normalizer, and edge-case integration testing |
| **DevOps & Containers** | Docker + Docker Compose | $\ge 3.8$ | Multi-platform, reproducible isolated Linux deployment |
| **Frontend UI** | HTML5, Tailwind CSS, Chart.js | CDN | Responsive editorial dashboard, probability bar charts, LIME badge chips |

---

## 📁 Repository Structure

```text
SentioScope/
├── artifacts/
│   ├── bi_gru_model.keras   # Pretrained Bidirectional GRU Keras model
│   └── tokenizer.pkl        # Pretrained Keras text tokenizer
├── app/
│   ├── __init__.py
│   ├── database.py          # SQLite engine and session factory
│   ├── models.py            # SQLAlchemy QueryHistory ORM model
│   ├── schemas.py           # Pydantic v2 schemas
│   ├── normalizer.py        # Emoji & slang normalization engine
│   ├── triage.py            # Distress weighting & urgency scoring
│   ├── explainer.py         # LIME token attribution wrapper
│   └── main.py              # FastAPI app with lifespan management
├── static/
│   └── index.html           # Dark-mode single page dashboard
├── tests/
│   ├── __init__.py
│   └── test_sentioscope.py  # Pytest automated test suite
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 How to Run the Project

### Option 1: Running with Docker (Recommended)
Make sure **Docker Desktop** is running, then in your project root:
```bash
docker compose up --build
```
- **Dashboard UI**: http://localhost:8000
- **Interactive Swagger Docs**: http://localhost:8000/docs
- **To stop the container**: `docker compose down`

### Option 2: Running Locally (Native Python)
Create and activate a virtual environment:

```bash
# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\activate

# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Start the FastAPI server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Access the application: Open your browser at http://localhost:8000.

---

## 🧪 Running Automated Tests
Run the full automated test suite using Pytest:
```bash
python -m pytest tests/ -v
```

**Verifies:**
- Text sanitization, slang mapping, and emoji demojization
- Model health check and artifact verification
- Urgency triage heuristic computation (`NORMAL`, `ELEVATED`, `CRITICAL`)
- Input validation (empty text strings $\rightarrow$ HTTP 422)
- SQLite query history insertion and deletion operations

---

## 📡 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/analyze` | Evaluates text, returns dominant emotion, LIME attributions, and triage score |
| `GET` | `/api/v1/history` | Fetches recent query history from SQLite |
| `DELETE` | `/api/v1/history` | Purges all historical query records |
| `GET` | `/api/v1/health` | Verifies server status, artifact availability, and loaded classes |
| `GET` | `/docs` | Auto-generated interactive Swagger UI |

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
