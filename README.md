<<<<<<< HEAD
# 🤖 RAG Assistant

A **Retrieval-Augmented Generation (RAG)** web application that lets users upload their own PDF, CSV, or webpage URL, then ask questions about the content — powered by **Llama 3 (via Groq)** and a three-retriever pipeline with live latency/accuracy metrics.

---

## 📸 Features

- 📄 Upload **PDF**, **CSV**, or paste a **URL** — any combination
- ⚡ Three-retriever fusion pipeline (FAISS + Sentence Window)
- 🔁 **Reciprocal Rank Fusion (RRF)** + **CrossEncoder reranking**
- 📊 **Live pipeline metrics** after every answer:
  - Latency per stage (retrieval / reranking / LLM)
  - Accuracy proxies (rerank score, context coverage, retriever agreement)
  - Human-readable trade-off label (e.g. *⚡ Fast & Accurate*)
- 🌐 FastAPI backend with auto-generated Swagger docs at `/docs`
- 💬 Clean dark-themed chat UI with drag-and-drop file upload

---

## 🗂️ Project Structure

```
3GenAIProj/
├── app.py                  # FastAPI app — all routes & metric assembly
├── store_indexes.py        # Builds FAISS, Sentence-Window
├── template.py             # Utility to scaffold empty project files
├── requirements.txt
├── .env                    # API keys (never commit this)
│
├── src/
│   ├── __init__.py
│   ├── helper.py           # Document loaders, chunker, embedding setup
│   ├── prompt.py           # LangChain PromptTemplate + build_prompt()
│   ├── retrievers.py       # RRF fusion, CrossEncoder reranking, timing
│   └── metrics.py          # StageTimer, PipelineMetrics, accuracy proxies
│
├── data/
│   ├── SDG.pdf
│   ├── stats.pdf
│   └── mnist_train.csv
│
├── static/
│   └── style.css           # Dark theme UI styles
│
├── templates/
│   └── index.html          # Full chat + upload + metrics UI
│
├── uploads/                # Created at runtime — stores user-uploaded files
└── experiment/
    └── sample.ipynb        # Exploratory notebook
```

---

## ⚙️ Setup

### 1. Git Clone 

```bash
cd MultiSourceRAG
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root (or edit the existing one):

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com

### 5. Run the server

```bash
python app.py
```

The server starts at **http://localhost:8000**

---

## 🚀 How to Use

| Step | Action |
|------|--------|
| 1 | Open **http://localhost:8000** in your browser |
| 2 | Upload a PDF and/or CSV using the drag-and-drop zones in the sidebar |
| 3 | Optionally paste a webpage URL |
| 4 | Click **⚡ Index Documents** and wait for indexing to complete |
| 5 | Type a question in the chat box and press **Enter** |
| 6 | Read the answer — a **📊 Pipeline Metrics** card appears below each response |

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI |
| `POST` | `/upload` | Upload files + build indexes |
| `POST` | `/ask` | Ask a question, get answer + metrics |
| `GET` | `/health` | Server health + active session count |
| `GET` | `/docs` | Swagger interactive API docs |
| `GET` | `/redoc` | ReDoc API documentation |

### `POST /upload`

Accepts `multipart/form-data`:

| Field | Type | Required |
|-------|------|----------|
| `pdf` | File (.pdf) | Optional |
| `csv` | File (.csv) | Optional |
| `url` | String (URL) | Optional |

At least one field must be provided.

**Response:**
```json
{
  "session_id": "abc-123-...",
  "sources": ["report.pdf", "data.csv"],
  "indexing_ms": 4210.5,
  "message": "Successfully indexed 2 source(s)."
}
```

### `POST /ask`

```json
{
  "session_id": "abc-123-...",
  "question": "What are the 17 SDGs?"
}
```

**Response:**
```json
{
  "answer": "The 17 Sustainable Development Goals are...",
  "latency": {
    "indexing_ms": 4210.5,
    "retrieval_ms": 320.1,
    "reranking_ms": 180.4,
    "llm_ms": 890.2,
    "total_ms": 5601.2
  },
  "accuracy_proxies": {
    "rerank_score": 0.823,
    "context_coverage": 0.714,
    "retriever_agreement": 0.667
  },
  "trade_off_label": "⚡ Fast & Accurate"
}
```

---

## 📊 Understanding the Metrics

After every answer, the UI shows a **Pipeline Metrics** card with two sections:

### ⏱ Latency

Measures real wall-clock time for each stage of the pipeline:

| Stage | What it measures |
|-------|-----------------|
| **Retrieval** | Time to query all 2 retrievers in parallel |
| **Reranking** | Time for CrossEncoder to score candidate chunks |
| **LLM (Groq)** | Time for Llama 3 to generate the answer |
| **Total** | Sum of all stages (indexing is shown at upload time) |

### 🎯 Accuracy Proxies

These are **proxy scores** — they don't require labelled ground truth but correlate strongly with answer quality:

| Proxy | How it's calculated | What it means |
|-------|---------------------|---------------|
| **Rerank Score** | CrossEncoder confidence score (0–1) of the top-ranked chunk | Higher = the retrieved chunk is genuinely relevant to the query |
| **Context Coverage** | % of meaningful answer words (length > 3) found in the retrieved context | Higher = the LLM used the documents, not its training memory |
| **Retriever Agreement** | Fraction of the 3 retrievers that returned the same top chunk | Higher = multiple independent retrievers agree → stronger signal |

### 🏷 Trade-off Labels

| Label | Meaning |
|-------|---------|
| ⚡ **Fast & Accurate** | Total < 2 s and avg accuracy proxy ≥ 65% — best case |
| ⚡ **Fast but Low Confidence** | Total < 2 s but accuracy proxies are low — check your documents |
| 🎯 **Accurate but Slow** | High accuracy but slow — consider smaller chunk sizes or fewer retrievers |
| ⚠️ **Slow & Low Confidence** | Both poor — document may not contain the answer, or indexing needs tuning |

### How to Improve Latency vs Accuracy

| If you want... | Change this |
|----------------|-------------|
| **Faster responses** | Reduce `chunk_size` (e.g. 300) and `k` (top-k = 3) in `store_indexes.py` |
| **More accurate answers** | Increase `chunk_size` (e.g. 800) and `window_size` in `SentenceWindowNodeParser` |
| **Faster reranking** | Swap CrossEncoder for a lighter model (`ms-marco-TinyBERT-L-2-v2`) |
| **Skip KG retriever** | Remove `retriever3` from `store_indexes.py` — saves 30–60% of index time |
| **Faster LLM** | Switch Groq model to `gemma-7b-it` (faster, slightly less accurate) |

---

## 🧱 Pipeline Architecture

```
User Question
     │
     ▼
┌───────────────────────────────┐
│         Two Retrievers        │
│  ┌──────────┐ ┌────────────┐  │
│  │  FAISS   │ │  Sentence  │  │
│  │  Vector  │ │   Window   │  │
│  │(LangChain│ │(LlamaIndex)│  │
│  └──────────┘ └────────────┘  │
└───────────────────────────────┘
     │ up to 5 results each
     ▼
┌──────────────────────┐
│  RRF Fusion (top 20) │  ← merges + deduplicates
└──────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│ CrossEncoder Reranking (→5) │  ← scores relevance
└─────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ LangChain PromptTemplate     │  ← fills context + question
└──────────────────────────────┘
     │
     ▼
┌───────────────────────┐
│  Groq LLM (Llama 3)   │  ← generates grounded answer
└───────────────────────┘
     │
     ▼
Answer + Metrics returned to UI
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (Jinja2 templates) |
| LLM | Llama 3 8B via Groq API |
| Vector store | FAISS (LangChain) |
| Sentence-window index | LlamaIndex `VectorStoreIndex` + `SentenceWindowNodeParser` |
| Embeddings (LangChain) | `sentence-transformers/all-MiniLM-L6-v2` |
| Embeddings (LlamaIndex) | `BAAI/bge-small-en-v1.5` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Prompt management | LangChain `PromptTemplate` |
| Document loaders | LangChain (`PyPDFLoader`, `CSVLoader`, `WebBaseLoader`) |

---

## 🔒 Security Notes

- **Never commit your `.env` file** — it contains your Groq API key.
- The `uploads/` folder stores user files locally. For production, use cloud storage (S3, GCS).
- `INDEX_STORE` is in-memory — sessions are lost on server restart. For production, persist indexes to disk with `FAISS.save_local()`.
- Add authentication before deploying publicly.

---

## 📄 License

- This project is for educational and research purposes.
