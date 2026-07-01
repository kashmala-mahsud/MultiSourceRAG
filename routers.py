import json
import os
import time
import traceback
import uuid

from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    File,
    Form,
    UploadFile,
    HTTPException,
    Request,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain_ollama import ChatOllama
from store_indexes import build_indexes
from src.retrievers import retrievers
from src.prompt import PROMPT
from models import UploadResponse, AskRequest, AskResponse
from metrics import PipelineMetrics, compute_context_coverage, compute_retriever_agreement

load_dotenv()

router = APIRouter()

templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store retrievers for each session
INDEX_STORE = {}

import os as _os

# Reuse a single ChatOllama client instead of constructing a new one on
# every /ask call.
_llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.2,
    num_predict=400,                       # cap max output tokens -- bounds
                                            # worst-case generation time instead
                                            # of letting the model ramble on
    num_thread=_os.cpu_count(),            # use all available CPU cores
                                            # instead of Ollama's default guess
    num_ctx=2048,                          # smaller context window than the
                                            # 4096+ default -- less KV-cache
                                            # overhead per generated token on CPU
)


def warm_up_llm():
    """
    Sends a trivial prompt to force Ollama to load the model into memory
    now, rather than on whichever request happens to hit it first. Ollama
    unloads idle models from memory after a few minutes, so without this,
    any /ask call after a period of inactivity pays a hidden model-reload
    cost inside its llm_ms timing, on top of actual generation time. Call
    this at FastAPI startup, and optionally on a periodic timer if you want
    to keep the model warm during long idle stretches too.
    """
    try:
        _llm.invoke("Hi")
    except Exception:
        # Don't crash server startup if Ollama isn't reachable yet -- the
        # first real /ask call will just pay the load cost as before.
        traceback.print_exc()


# -----------------------------
# Home Page
# -----------------------------
@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


# -----------------------------
# Upload Route
# -----------------------------
@router.post("/upload", response_model=UploadResponse)
async def upload(
    pdf: UploadFile | None = File(default=None),
    csv: UploadFile | None = File(default=None),
    url: str = Form(default=""),
):

    pdf_path = None
    csv_path = None
    sources = []

    url = url.strip()

    # Save PDF
    if pdf and pdf.filename:

        if not pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed.",
            )

        pdf_path = os.path.join(
            UPLOAD_FOLDER,
            pdf.filename,
        )

        with open(pdf_path, "wb") as f:
            f.write(await pdf.read())

        sources.append(pdf.filename)

    # Save CSV
    if csv and csv.filename:

        if not csv.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are allowed.",
            )

        csv_path = os.path.join(
            UPLOAD_FOLDER,
            csv.filename,
        )

        with open(csv_path, "wb") as f:
            f.write(await csv.read())

        sources.append(csv.filename)

    # URL
    if url:
        sources.append(url)
    else:
        url = None

    if not sources:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF, CSV, or provide a URL.",
        )

    try:
        # build_indexes() is CPU-bound and blocking (document parsing,
        # embedding, FAISS/LlamaIndex construction). Running it directly
        # in this async endpoint would block the whole event loop, so any
        # other request hitting the server during an upload would just
        # hang. run_in_threadpool moves it to a worker thread instead.
        vector_index, sentence_index = await run_in_threadpool(
            build_indexes,
            pdf_path=pdf_path,
            csv_path=csv_path,
            url=url,
        )

    except Exception as e:
            traceback.print_exc()      # <-- add this
            raise HTTPException(
                status_code=500,
                detail=str(e),
            )

    session_id = str(uuid.uuid4())

    INDEX_STORE[session_id] = (
        vector_index,
        sentence_index,
    )

    return UploadResponse(
        session_id=session_id,
        sources=sources,
    )


# -----------------------------
# Ask Route
# -----------------------------
@router.post("/ask")
async def ask(request: AskRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if request.session_id not in INDEX_STORE:
        print("Available Sessions:", INDEX_STORE.keys())
        print("Received Session:", request.session_id)

        raise HTTPException(
            status_code=400,
            detail="Please upload documents first.",
        )

    vector_index, sentence_index = INDEX_STORE[request.session_id]
    metrics = PipelineMetrics(question=request.question)

    # --- Retrieval + fusion + rerank (retrievers.py measures its own
    # stage timing/counts now -- more precise than timing the whole
    # threadpool round-trip from out here) ---
    try:
        docs, retrieval_stats = await run_in_threadpool(
            retrievers,
            vector_index,
            sentence_index,
            request.question,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    metrics.retrieval_ms         = retrieval_stats["retrieval_ms"]
    metrics.reranking_ms         = retrieval_stats["reranking_ms"]
    metrics.num_chunks_retrieved = retrieval_stats["num_retrieved"]
    metrics.num_chunks_after_rrf = retrieval_stats["num_after_rrf"]
    if docs:
        metrics.rerank_score = docs[0]["rerank_score"]
        metrics.top_source = docs[0]["source"]
    metrics.retriever_agreement = compute_retriever_agreement(docs)

    # Trim what goes to the LLM: fewer chunks + a per-chunk char cap. Less
    # prompt for the model to read before it can even start generating --
    # a real time saving, not just a perceived one, on top of streaming.
    TOP_N_CONTEXT       = 3
    MAX_CHARS_PER_CHUNK = 500
    context = "\n\n".join(doc["text"][:MAX_CHARS_PER_CHUNK] for doc in docs[:TOP_N_CONTEXT])

    prompt = PROMPT.format(
        context=context,
        question=request.question,
    )

    async def token_stream():
        t0 = time.perf_counter()
        answer_parts = []
        try:
            async for chunk in _llm.astream(prompt):
                token = chunk.content
                if token:
                    answer_parts.append(token)
                    yield token
        except Exception as e:
            traceback.print_exc()
            yield f"\n\n[Error while generating: {e}]"
            return

        metrics.llm_ms = round((time.perf_counter() - t0) * 1000, 1)
        metrics.context_coverage = compute_context_coverage("".join(answer_parts), docs)

        # Trailing marker the frontend splits on to pull out trade-off/latency
        # info, without needing a second round trip to the server.
        yield "\n\n<<<METRICS>>>" + json.dumps({
            "trade_off": metrics.trade_off_label(),
            "metrics": metrics.summary(),
        })

    return StreamingResponse(token_stream(), media_type="text/plain")