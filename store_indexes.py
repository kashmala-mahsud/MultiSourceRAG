import hashlib
import os
import re
import time

from langchain_community.vectorstores import FAISS
from llama_index.core import (
    Document as LlamaDocument,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama

from src.helper import chunking, embedding as get_embedding, load_documents

# Where cached indexes live. Re-uploading the same file(s) will reuse
# whatever was built last time instead of rebuilding from scratch.
CACHE_DIR = os.path.join(os.path.dirname(__file__), "storage", "indexes")
os.makedirs(CACHE_DIR, exist_ok=True)


def cleaning(all_docs):
    for doc in all_docs:
        text = doc.page_content
        text = re.sub(r"\s+", " ", text)
        doc.page_content = text.strip()
    return all_docs


def _fingerprint(pdf_path=None, csv_path=None, url=None):
    """
    Build a stable cache key from the *content* of the uploaded file(s)
    (or the url string), so the same document uploaded twice reuses the
    cached indexes instead of paying the full build cost again.
    """
    hasher = hashlib.sha256()

    for path in (pdf_path, csv_path):
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                hasher.update(f.read())

    if url:
        hasher.update(url.encode("utf-8"))

    return hasher.hexdigest()[:16]


def _load_cached(cache_key, embed_model):
    faiss_dir = os.path.join(CACHE_DIR, cache_key, "faiss")
    llama_dir = os.path.join(CACHE_DIR, cache_key, "llama")

    if not (os.path.isdir(faiss_dir) and os.path.isdir(llama_dir)):
        return None

    try:
        db = FAISS.load_local(
            faiss_dir, embed_model, allow_dangerous_deserialization=True
        )
        storage_context = StorageContext.from_defaults(persist_dir=llama_dir)
        sentence_index = load_index_from_storage(storage_context)
        return db, sentence_index
    except Exception:
        # Cache is corrupt/incompatible (e.g. embedding model changed) --
        # fall through and rebuild instead of crashing the request.
        return None


def _save_cache(cache_key, db, sentence_index):
    faiss_dir = os.path.join(CACHE_DIR, cache_key, "faiss")
    llama_dir = os.path.join(CACHE_DIR, cache_key, "llama")
    os.makedirs(faiss_dir, exist_ok=True)
    os.makedirs(llama_dir, exist_ok=True)

    db.save_local(faiss_dir)
    sentence_index.storage_context.persist(persist_dir=llama_dir)


def build_indexes(pdf_path=None, csv_path=None, url=None):
    """
    Builds (or loads from cache) a FAISS vector index and a sentence-window
    LlamaIndex vector index, and returns retrievers for both.

    NOTE: the previous version of this function also built a
    KnowledgeGraphIndex here, which calls the LLM once PER CHUNK to extract
    triplets. That's the main reason uploads were slow (and, if pointed at
    a hosted API like Groq, the main reason you'd blow through rate
    limits). It's removed from the default path -- see build_graph_index()
    below if you want to build one explicitly/lazily for a specific
    session instead of on every single upload.
    """
    embed_model = get_embedding()

    Settings.llm = Ollama(
        model="qwen2.5:1.5b",
        base_url="http://127.0.0.1:11434",
        request_timeout=300,
    )

    cache_key = _fingerprint(pdf_path=pdf_path, csv_path=csv_path, url=url)

    cached = _load_cached(cache_key, embed_model)
    if cached is not None:
        print(f"Loaded cached indexes for key {cache_key}")
        db, sentence_index = cached
        return (
            db.as_retriever(search_kwargs={"k": 5}),
            sentence_index.as_retriever(similarity_top_k=5),
        )

    start = time.time()
    extracted_data = load_documents(pdf_path=pdf_path, csv_path=csv_path, url=url)
    print(f"Load Documents: {time.time() - start:.2f} sec")

    start = time.time()
    cleaned = cleaning(extracted_data)
    print(f"Cleaning: {time.time() - start:.2f} sec")

    start = time.time()
    chunks = chunking(cleaned)
    print(f"Chunking: {time.time() - start:.2f} sec ({len(chunks)} chunks)")

    start = time.time()
    db = FAISS.from_documents(chunks, embed_model)
    print(f"FAISS Index: {time.time() - start:.2f} sec")

    llama_docs = [
        LlamaDocument(text=doc.page_content, metadata=doc.metadata)
        for doc in chunks
    ]

    # NOTE: this used to be SentenceWindowNodeParser, which creates ONE NODE
    # PER SENTENCE (so 60 chunks could easily explode into 300-600+ nodes,
    # each needing its own embedding -- that's why this step took 22s while
    # FAISS, embedding the same 60 chunks, took 5.5s). It also stores extra
    # "window" context in each node's metadata for a query-time postprocessor
    # to expand around -- but retrievers.py was reading node.node.text
    # directly, never using that window, so all that extra granularity was
    # pure overhead with no retrieval benefit.
    #
    # SentenceSplitter here uses a smaller chunk size than the FAISS chunks
    # (512 vs 1000 chars) so this index still adds genuine retrieval
    # diversity for the RRF fusion step, without one-node-per-sentence blowup.
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = node_parser.get_nodes_from_documents(llama_docs)

    # Raise the embedding batch size (default is 10) so LlamaIndex sends
    # bigger batches to the underlying model per forward pass instead of
    # looping in small groups.
    if Settings.embed_model is not None:
        Settings.embed_model.embed_batch_size = 64

    start = time.time()
    sentence_index = VectorStoreIndex(nodes)
    print(f"Sentence Index: {time.time() - start:.2f} sec ({len(nodes)} nodes)")

    start = time.time()
    _save_cache(cache_key, db, sentence_index)
    print(f"Cache Save: {time.time() - start:.2f} sec")

    retriever1 = db.as_retriever(search_kwargs={"k": 5})
    retriever2 = sentence_index.as_retriever(similarity_top_k=5)

    return retriever1, retriever2