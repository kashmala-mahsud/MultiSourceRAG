from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from routers import router, warm_up_llm
from src.retrievers import warm_up as warm_up_reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the CrossEncoder reranker and the Ollama LLM into memory now,
    # at server startup, instead of on whichever request happens to hit
    # them first. This trades a few extra seconds of startup time for
    # consistent, honest per-request latency numbers -- no more "first
    # /ask call is mysteriously 30-50s slower than the rest" surprises.
    print("Warming up reranker and LLM...")
    await run_in_threadpool(warm_up_reranker)
    await run_in_threadpool(warm_up_llm)
    print("Warm-up complete.")
    yield
    # (nothing to clean up on shutdown)


app = FastAPI(
    title="RAG Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)