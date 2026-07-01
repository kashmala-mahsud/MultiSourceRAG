import math
import time
from collections import defaultdict

# The CrossEncoder model is fairly heavy to load (a few hundred MB). Lazy-load
# it on first use instead of at import time, so simply importing this module
# (e.g. when FastAPI starts up) doesn't pay that cost up front.
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def warm_up():
    """
    Forces the CrossEncoder to load now, rather than lazily on whichever
    request happens to hit it first. Without this, the first /ask call
    after the server starts (or reloads) eats the full model-load cost
    inside its timed reranking_ms window, making that one request look
    artificially slow (e.g. 14s+ instead of the real sub-second rerank
    cost) with no indication it was a one-time load, not real work.
    Call this once at FastAPI startup.
    """
    _get_reranker()


def _normalize_rerank_score(raw_score: float) -> float:
    """
    CrossEncoder (ms-marco-MiniLM) outputs an UNBOUNDED relevance logit, not
    a 0-1 probability -- raw scores commonly range roughly -10 to +10+
    depending on match quality (a strong match can easily score >10). If
    that raw number gets averaged with genuinely 0-1 metrics (like
    context_coverage, retriever_agreement) in metrics.trade_off_label(), it
    dominates the average and makes almost everything look "accurate"
    regardless of real quality. Squashing it through a sigmoid gives a
    proper 0-1 relevance probability that's safe to average with the rest.
    """
    return 1 / (1 + math.exp(-raw_score))


def retrievers(retriever1, retriever2, query: str):
    """
    Runs the query through the vector retriever and the sentence-window
    retriever, fuses results with RRF, reranks with a CrossEncoder, and
    returns (top_5_results, stats).

    stats contains per-stage timing and candidate counts:
      - num_retrieved:  raw candidates returned by the 2 retrievers combined
                         (before RRF dedup/fusion)
      - num_after_rrf:  candidates remaining after RRF fusion (fed to rerank)
      - retrieval_ms:   time spent in the 2 retriever calls
      - reranking_ms:   time spent on RRF fusion + CrossEncoder reranking

    (The knowledge-graph retriever that used to be the 3rd source here has
    been removed from the default query path -- it was the most expensive
    part of every /ask call for comparatively little quality gain. See git
    history if you want to add it back in for specific sessions.)
    """
    t0 = time.perf_counter()
    vector_results   = retriever1.invoke(query)
    sentence_results = retriever2.retrieve(query)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

    retrieved_docs = []

    for rank, doc in enumerate(vector_results, start=1):
        retrieved_docs.append({
            "text": doc.page_content,
            "source": "vector",
            "rank": rank,
            "metadata": doc.metadata,
        })

    for rank, node in enumerate(sentence_results, start=1):
        retrieved_docs.append({
            "text": node.node.text,
            "source": "sentence_window",
            "rank": rank,
            "metadata": node.node.metadata,
        })

    t0 = time.perf_counter()

    # RRF fusion
    k = 60
    rrf_scores = defaultdict(lambda: {"score": 0, "metadata": None, "source": None})
    for doc in retrieved_docs:
        t = doc["text"]
        rrf_scores[t]["score"]    += 1 / (k + doc["rank"])
        rrf_scores[t]["metadata"]  = doc["metadata"]
        rrf_scores[t]["source"]    = doc["source"]

    # Fewer candidates go into the reranker (10 instead of 20) since we now
    # have 2 sources feeding it instead of 3 -- keeps rerank latency down.
    fused = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:10]

    reranker = _get_reranker()
    raw_scores = reranker.predict([(query, doc[0]) for doc in fused])

    reranked = sorted(
        [
            {
                "text": t,
                "rerank_score": _normalize_rerank_score(float(s)),
                "rerank_score_raw": float(s),
                "fusion_score": info["score"],
                "source": info["source"],
                "metadata": info["metadata"],
            }
            for (t, info), s in zip(fused, raw_scores)
        ],
        key=lambda x: x["rerank_score"],
        reverse=True,
    )
    reranking_ms = round((time.perf_counter() - t0) * 1000, 1)

    stats = {
        "num_retrieved": len(retrieved_docs),
        "num_after_rrf": len(fused),
        "retrieval_ms": retrieval_ms,
        "reranking_ms": reranking_ms,
    }

    return reranked[:5], stats