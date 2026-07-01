from pydantic import BaseModel
# -----------------------------
# Pydantic Models
# -----------------------------
class AskRequest(BaseModel):
    session_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    trade_off: str | None = None      # e.g. "⚡ Fast & Accurate"
    metrics: dict | None = None       # latency_ms / accuracy_proxies / retrieval_info


class UploadResponse(BaseModel):
    session_id: str
    sources: list[str]