"""
api_service.py — FastAPI Backend
==================================
Run:  uvicorn api_service:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import shutil
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Local modules ──────────────────────────────────────────────────────────
from config import setup_output_dirs, OUTPUT_DIR, CHAT_LOG_DIR
from llm    import correct_text, summarize_text, generate_response
from ocr    import setup_tesseract, run_ocr, extract_text
from rag    import build_or_load_index, retrieve_passages, auto_scan_text, save_chat_log
from image  import extract_and_analyze_graphs

# ── Bootstrap ──────────────────────────────────────────────────────────────
setup_output_dirs()
setup_tesseract()

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not set — add it to your .env file")

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="EduScan API",
    description="Intelligent Study Scanner & Tutor",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ← غيّري لـ domain الفرونت-إند في production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global RAG collection (loaded once on startup) ─────────────────────────
_collection = None

def get_collection():
    global _collection
    if _collection is None:
        _collection = build_or_load_index(folder=OUTPUT_DIR)
    return _collection


# ══════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    session_id: str


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "EduScan API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok", "hf_token_set": bool(HF_TOKEN)}


# ── 1. Upload & Analyze Document ──────────────────────────────────────────
@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    """
    Accepts a PDF or image file.
    Returns: OCR text, corrected text, summary, and graph analysis.
    """
    allowed = {".pdf", ".jpg", ".jpeg", ".png"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # ── Save uploaded file to temp ────────────────────────────────────────
    temp_path = os.path.join("temp", f"{uuid.uuid4().hex}{ext}")
    os.makedirs("temp", exist_ok=True)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ── OCR ───────────────────────────────────────────────────────────
        raw_text = extract_text(run_ocr(temp_path))
        if not raw_text:
            raise HTTPException(status_code=422, detail="Could not extract text from file.")

        # ── Correct & Summarize ───────────────────────────────────────────
        corrected = correct_text(raw_text)
        summary   = summarize_text(corrected)

        # ── Graph Analysis ────────────────────────────────────────────────
        graph_results = extract_and_analyze_graphs(temp_path, hf_token=HF_TOKEN)
        graphs = [
            {"image": os.path.basename(r["image_path"]), "analysis": r["analysis"]}
            for r in graph_results
        ]

        # ── Auto-scan for rules & definitions ────────────────────────────
        from llm import _QW_MODEL_ID
        auto_scan_text(raw_text, HF_TOKEN, _QW_MODEL_ID)

        # ── Save to output for RAG ────────────────────────────────────────
        stem = Path(file.filename).stem.replace(" ", "_")
        with open(os.path.join(OUTPUT_DIR, f"{stem}_corrected.md"), "w", encoding="utf-8") as f:
            f.write(corrected)
        with open(os.path.join(OUTPUT_DIR, f"{stem}_summary.md"), "w", encoding="utf-8") as f:
            f.write(summary)

        # Reload RAG index with new docs
        global _collection
        _collection = build_or_load_index(folder=OUTPUT_DIR)

        return {
            "status":          "success",
            "document_name":   file.filename,
            "raw_text":        raw_text[:500] + "…" if len(raw_text) > 500 else raw_text,
            "corrected_text":  corrected,
            "summary":         summary,
            "graphs_analyzed": len(graphs),
            "graphs":          graphs,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── 2. Chat with uploaded document ────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question about previously uploaded documents.
    Uses RAG retrieval + persistent memory.
    """
    collection = get_collection()

    # Retrieve relevant passages
    passages = retrieve_passages(request.question, collection, top_k=5)
    context  = "\n\n".join(passages) if passages else "No context found."

    # Load memory log for this session
    log_path = os.path.join(CHAT_LOG_DIR, f"session_{request.session_id}.md")
    memory   = ""
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            memory = f.read()[-3000:]

    prompt = (
        "You are a helpful study assistant.\n"
        f"[MEMORY]:\n{memory}\n\n"
        f"[PDF CONTEXT]:\n{context}\n\n"
        f"[QUESTION]: {request.question}\n\n"
        "Answer from context first. Format laws as [RULE: Name] Formula. English only."
    )

    answer = generate_response(prompt)
    save_chat_log(request.question, answer, log_filename=f"session_{request.session_id}.md")

    return ChatResponse(answer=answer, session_id=request.session_id)


# ── 3. Get summary of uploaded document ───────────────────────────────────
@app.get("/summary")
def get_summary(filename: str = Query(..., description="Document filename")):
    """Returns the saved summary for a given document."""
    stem     = Path(filename).stem.replace(" ", "_")
    summary_path = os.path.join(OUTPUT_DIR, f"{stem}_summary.md")
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Summary not found. Upload the document first.")
    with open(summary_path, "r", encoding="utf-8") as f:
        return {"filename": filename, "summary": f.read()}


# ── 4. Get extracted rules & definitions ──────────────────────────────────
@app.get("/rules")
def get_rules():
    """Returns all extracted physics rules."""
    path = os.path.join(OUTPUT_DIR, "rules", "All_rules.txt")
    if not os.path.exists(path):
        return {"rules": []}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return {"rules": lines}


@app.get("/definitions")
def get_definitions():
    """Returns all extracted definitions."""
    path = os.path.join(OUTPUT_DIR, "definitions", "All_definitions.txt")
    if not os.path.exists(path):
        return {"definitions": []}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return {"definitions": lines}


# ── 5. Get graph analysis results ─────────────────────────────────────────
@app.get("/graphs")
def get_graphs():
    """Returns all graph analysis results."""
    import glob
    results = []
    for path in glob.glob(os.path.join(OUTPUT_DIR, "graph_analysis", "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        results.append({"file": os.path.basename(path), "analysis": content})
    return {"graphs": results}
