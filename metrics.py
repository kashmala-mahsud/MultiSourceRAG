"""
metrics.py — Latency & accuracy-proxy tracking for the RAG pipeline.

Measures wall-clock time for each stage and computes three
accuracy-proxy scores that don't require labelled ground truth:

  1. rerank_score   — CrossEncoder confidence (higher = more relevant)
  2. context_coverage — fraction of answer tokens found in the context
  3. retriever_agreement — how many of the 3 retrievers returned the top chunk
"""

import time
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Data classes ──────────────────────────────────────────────

@dataclass
class StageTimer:
    name: str
    start: float = field(default_factory=time.perf_counter)
    end:   Optional[float] = None

    def stop(self) -> float:
        self.end = time.perf_counter()
        return round(self.end - self.start, 3)

    @property
    def elapsed_ms(self) -> float:
        if self.end is None:
            return round((time.perf_counter() - self.start) * 1000, 1)
        return round((self.end - self.start) * 1000, 1)


@dataclass
class PipelineMetrics:
    # Latency (milliseconds)
    indexing_ms:   float = 0.0
    retrieval_ms:  float = 0.0
    reranking_ms:  float = 0.0
    llm_ms:        float = 0.0

    # Accuracy proxies (0-1)
    rerank_score:          float = 0.0   # CrossEncoder score of best chunk
    context_coverage:      float = 0.0   # % answer words found in context
    retriever_agreement:   float = 0.0   # fraction of retrievers that returned top-1 chunk

    # Meta
    num_chunks_retrieved:  int   = 0
    num_chunks_after_rrf:  int   = 0
    top_source:            str   = ""
    question:              str   = ""

    @property
    def total_ms(self) -> float:
        return round(self.indexing_ms + self.retrieval_ms + self.reranking_ms + self.llm_ms, 1)

    def summary(self) -> dict:
        return {
            "latency": {
                "indexing_ms":  self.indexing_ms,
                "retrieval_ms": self.retrieval_ms,
                "reranking_ms": self.reranking_ms,
                "llm_ms":       self.llm_ms,
                "total_ms":     self.total_ms,
            },
            "accuracy_proxies": {
                "rerank_score":        round(self.rerank_score, 4),
                "context_coverage":    round(self.context_coverage, 4),
                "retriever_agreement": round(self.retriever_agreement, 4),
            },
            "retrieval_info": {
                "chunks_retrieved":   self.num_chunks_retrieved,
                "chunks_after_rrf":   self.num_chunks_after_rrf,
                "top_source":         self.top_source,
            },
        }

    def trade_off_label(self) -> str:
        """
        Returns a human-readable trade-off label based on the metrics.
        Useful for displaying to the user in the UI.
        """
        avg_acc = (self.rerank_score + self.context_coverage + self.retriever_agreement) / 3

        if self.total_ms < 2000 and avg_acc >= 0.65:
            return "⚡ Fast & Accurate"
        elif self.total_ms < 2000 and avg_acc < 0.65:
            return "⚡ Fast but Low Confidence"
        elif self.total_ms >= 2000 and avg_acc >= 0.65:
            return "🎯 Accurate but Slow"
        else:
            return "⚠️  Slow & Low Confidence"


# ── Accuracy proxy helpers ────────────────────────────────────

def compute_context_coverage(answer: str, top_docs: list) -> float:
    """
    Fraction of meaningful answer words (len > 3) found in the context.
    A rough proxy: high coverage means the LLM grounded its answer in
    the retrieved chunks rather than hallucinating.
    """
    context = " ".join(d["text"] for d in top_docs).lower()
    words   = [w for w in re.findall(r"\b\w+\b", answer.lower()) if len(w) > 3]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in context)
    return round(hits / len(words), 4)


NUM_RETRIEVERS = 2  # vector + sentence_window (knowledge-graph retriever removed)


def compute_retriever_agreement(top_docs: list) -> float:
    """
    If the top-ranked chunk appears in results from multiple retrievers,
    the retrievers 'agree' — a sign the chunk is genuinely relevant.
    Returns the fraction of the retrievers that returned the top chunk.
    """
    if not top_docs:
        return 0.0
    top_text     = top_docs[0]["text"]
    all_sources  = [d.get("source", "") for d in top_docs]
    unique_srcs  = set(all_sources)
    # Count how many retrievers returned a chunk with this exact text
    contributors = sum(1 for d in top_docs if d["text"] == top_text)
    return round(min(contributors / NUM_RETRIEVERS, 1.0), 4)