from langchain_community.document_loaders import CSVLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings

# PyMuPDFLoader (backed by the `pymupdf` package) parses PDFs considerably
# faster than PyPDFLoader. Fall back to PyPDFLoader if pymupdf isn't
# installed, so this doesn't break anything -- just run
# `pip install pymupdf` to get the speed-up.
try:
    from langchain_community.document_loaders import PyMuPDFLoader as _PDFLoader
except ImportError:
    from langchain_community.document_loaders import PyPDFLoader as _PDFLoader


def load_documents(pdf_path=None, csv_path=None, url=None):
    """Load any combination of PDF, CSV, and URL sources."""
    all_docs = []

    if pdf_path:
        loader   = _PDFLoader(pdf_path)
        pdf_docs = loader.load()
        for doc in pdf_docs:
            doc.metadata["source_type"] = "pdf"
        all_docs.extend(pdf_docs)
 
    if url:
        loader   = WebBaseLoader(url)
        url_docs = loader.load()
        for doc in url_docs:
            doc.metadata["source_type"] = "url"
        all_docs.extend(url_docs)
 
    if csv_path:
        loader   = CSVLoader(file_path=csv_path)
        csv_docs = loader.load()
        for doc in csv_docs:
            doc.metadata["source_type"] = "csv"  
        all_docs.extend(csv_docs)
    
    if not all_docs:
        raise ValueError("At least one source (PDF, CSV, or URL) must be provided.")
 
    return all_docs



def chunking(all_docs):

    splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100)
    return splitter.split_documents(all_docs)
   

# Cache the embedding model at module load time so repeated calls to
# embedding() (e.g. across multiple /upload requests) don't reload the
# model from disk/HF hub every time.
_lc_embedding = None


def embedding():
    """
    Loads ONE embedding model and shares it between LangChain (FAISS) and
    LlamaIndex (VectorStoreIndex), instead of loading two separate models.
    Previously this loaded both `sentence-transformers/all-MiniLM-L6-v2`
    (via langchain) AND `BAAI/bge-small-en-v1.5` (via llama-index) into
    memory at the same time -- wasted RAM and load time for no benefit.
    """
    global _lc_embedding

    if _lc_embedding is None:
        _lc_embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    # Wrap the single LangChain embedding model so LlamaIndex uses the
    # exact same model/vectors instead of loading its own.
    Settings.embed_model = LangchainEmbedding(_lc_embedding)

    return _lc_embedding