"""
api_service.py — FastAPI Backend
==================================
Run:  uvicorn api_service:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import re
import glob
import shutil
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import setup_output_dirs, OUTPUT_DIR, CHAT_LOG_DIR
from llm    import correct_text, summarize_text, generate_response
from ocr    import setup_tesseract, run_ocr, extract_text
from rag    import build_or_load_index, retrieve_passages, auto_scan_text, save_chat_log, MEMORY_DIR
from image  import extract_and_analyze_graphs
from study_plan import generate_study_plan

setup_output_dirs()
setup_tesseract()

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not set — add it to your .env file")

app = FastAPI(
    title="MindEdge API",
    description="Intelligent Study Scanner & Tutor",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_collection = None
_last_pdf_name: Optional[str] = None
_last_corrected_text: Optional[str] = None
_LAST_PDF_FILE = os.path.join(os.path.dirname(__file__), "last_pdf.txt")

if os.path.exists(_LAST_PDF_FILE):
    with open(_LAST_PDF_FILE, "r", encoding="utf-8") as f:
        _last_pdf_name = f.read().strip()
    print(f"[📂] Restored last PDF: '{_last_pdf_name}'")

def get_collection():
    global _collection
    if _collection is None:
        _collection = build_or_load_index(folder=OUTPUT_DIR)
    return _collection


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    session_id: str

class StudyPlanRequest(BaseModel):
    days: int
    hours_per_day: float
    level: Optional[str] = "Intermediate"


@app.get("/")
def root():
    return {"status": "MindEdge API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok", "hf_token_set": bool(HF_TOKEN)}


# ── 1. Upload & Analyze Document ──────────────────────────────────────────
@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".jpg", ".jpeg", ".png"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    temp_path = os.path.join("temp", f"{uuid.uuid4().hex}{ext}")
    os.makedirs("temp", exist_ok=True)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_text = extract_text(run_ocr(temp_path))
        if not raw_text:
            raise HTTPException(status_code=422, detail="Could not extract text from file.")

        corrected = correct_text(raw_text)
        summary   = summarize_text(corrected)

        graph_results = extract_and_analyze_graphs(temp_path, hf_token=HF_TOKEN)
        graphs = [
            {"image": os.path.basename(r["image_path"]), "analysis": r["analysis"]}
            for r in graph_results
        ]

        # ── تنظيف اسم الـ PDF ────────────────────────────────────────────
        from llm import _QW_MODEL_ID
        raw_name = os.path.splitext(os.path.basename(file.filename))[0]
        pdf_name = re.sub(r'[^\w\u0600-\u06FF]', '_', raw_name).strip('_')

        global _last_pdf_name, _last_corrected_text
        _last_pdf_name = pdf_name
        _last_corrected_text = corrected

        with open(_LAST_PDF_FILE, "w", encoding="utf-8") as f:
            f.write(pdf_name)

        # ── حفظ الـ corrected text على disk عشان study-plan بعد الـ reload ──
        corrected_cache_path = os.path.join(os.path.dirname(__file__), "last_corrected.txt")
        with open(corrected_cache_path, "w", encoding="utf-8") as f:
            f.write(corrected)

        auto_scan_text(corrected, HF_TOKEN, _QW_MODEL_ID, pdf_name=pdf_name)

        # ── حفظ index الـ graphs بتاعت الـ PDF ده فقط ───────────────────
        graph_analysis_dir = os.path.join(OUTPUT_DIR, "graph_analysis")
        os.makedirs(graph_analysis_dir, exist_ok=True)
        graph_index_path = os.path.join(graph_analysis_dir, f"{pdf_name}_index.txt")
        with open(graph_index_path, "w", encoding="utf-8") as f:
            for r in graph_results:
                fname = os.path.basename(r["image_path"])
                fname = fname.replace(".png", "_analysis.txt").replace(".jpg", "_analysis.txt")
                f.write(fname + "\n")

        stem = pdf_name
        with open(os.path.join(OUTPUT_DIR, f"{stem}_corrected.md"), "w", encoding="utf-8") as f:
            f.write(corrected)
        with open(os.path.join(OUTPUT_DIR, f"{stem}_summary.md"), "w", encoding="utf-8") as f:
            f.write(summary)

        global _collection
        _collection = build_or_load_index(folder=OUTPUT_DIR)

        return {
            "status":          "success",
            "document_name":   file.filename,
            "pdf_name":        pdf_name,
            "raw_text":        raw_text[:500] + "…" if len(raw_text) > 500 else raw_text,
            "corrected_text":  corrected,
            "summary":         summary,
            "graphs_analyzed": len(graphs),
            "graphs":          graphs,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── 2. Chat ────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    collection = get_collection()
    passages = retrieve_passages(request.question, collection, top_k=5)
    context  = "\n\n".join(passages) if passages else "No context found."

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


# ── 3. Summary ────────────────────────────────────────────────────────────
@app.get("/summary")
def get_summary():
    """Returns summary of the last uploaded PDF — no parameter needed."""
    name = _last_pdf_name
    if not name:
        return {"error": "No PDF uploaded yet"}
    stem = name
    summary_path = os.path.join(OUTPUT_DIR, f"{stem}_summary.md")
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Summary not found.")
    with open(summary_path, "r", encoding="utf-8") as f:
        return {"pdf": name, "summary": f.read()}


# ── 4. Rules ───────────────────────────────────────────────────────────────
@app.get("/rules")
def get_rules(pdf_name: Optional[str] = Query(None)):
    """Returns extracted rules — uses last uploaded PDF if no pdf_name given."""
    name = pdf_name or _last_pdf_name
    if not name:
        return {"error": "No PDF uploaded yet and no pdf_name provided"}
    folder = os.path.join(MEMORY_DIR, name, "rules")
    if not os.path.exists(folder):
        return {"pdf": name, "rules": []}
    lines = []
    for filepath in glob.glob(os.path.join(folder, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            lines.append(f.read().strip())
    return {"pdf": name, "rules": lines}


# ── 5. Definitions ────────────────────────────────────────────────────────
@app.get("/definitions")
def get_definitions(pdf_name: Optional[str] = Query(None)):
    """Returns extracted definitions — uses last uploaded PDF if no pdf_name given."""
    name = pdf_name or _last_pdf_name
    if not name:
        return {"error": "No PDF uploaded yet and no pdf_name provided"}
    folder = os.path.join(MEMORY_DIR, name, "definitions")
    if not os.path.exists(folder):
        return {"pdf": name, "definitions": []}
    lines = []
    for filepath in glob.glob(os.path.join(folder, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            lines.append(f.read().strip())
    return {"pdf": name, "definitions": lines}


# ── 6. Study Plan ─────────────────────────────────────────────────────────
@app.post("/study-plan")
def get_study_plan(request: StudyPlanRequest):
    """
    Generates a study plan from the last uploaded PDF.
    Body: { "days": 7, "hours_per_day": 2, "level": "Intermediate" }
    No pdf_name needed — uses last uploaded PDF automatically.
    """
    # جيب الـ corrected text — من الـ memory أو من الـ disk
    corrected_cache_path = os.path.join(os.path.dirname(__file__), "last_corrected.txt")

    text = _last_corrected_text
    if not text and os.path.exists(corrected_cache_path):
        with open(corrected_cache_path, "r", encoding="utf-8") as f:
            text = f.read()

    if not text:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet. Please upload a PDF first.")

    # استخرج التوبكس بالـ AI
    topic_prompt = (
        f"Extract a list of the main study topics from the following academic text. "
        f"Return only a numbered list of topic names, nothing else.\n\nText:\n{text[:4000]}"
    )
    topics_raw = generate_response(topic_prompt)

    # حوّل الرد لـ list
    topics = []
    for line in topics_raw.strip().splitlines():
        line = re.sub(r'^\s*\d+[\.\)]\s*', '', line).strip()
        if line:
            topics.append(line)

    if not topics:
        raise HTTPException(status_code=500, detail="Could not extract topics from the document.")

    # ولّد الخطة
    plan = generate_study_plan(
        topics=topics,
        days=request.days,
        hours_per_day=request.hours_per_day,
        level=request.level,
    )

    return {
        "status": "success",
        "pdf":    _last_pdf_name or "unknown",
        "topics_found": len(topics),
        "plan":   plan,
    }


# ── 7. Graphs ─────────────────────────────────────────────────────────────
@app.get("/graphs")
def get_graphs(pdf_name: Optional[str] = Query(None)):
    """
    Returns graph analyses for the last uploaded PDF only.
    Uses pdf_name_index.txt to know which files belong to this PDF.
    """
    name = pdf_name or _last_pdf_name
    if not name:
        return {"error": "No PDF uploaded yet and no pdf_name provided"}

    graph_analysis_dir = os.path.join(OUTPUT_DIR, "graph_analysis")
    index_path = os.path.join(graph_analysis_dir, f"{name}_index.txt")

    if not os.path.exists(index_path):
        return {"pdf": name, "graphs": []}

    with open(index_path, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f if line.strip()]

    results = []
    for fname in filenames:
        fpath = os.path.join(graph_analysis_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            results.append({"file": fname, "analysis": content})

    return {"pdf": name, "graphs": results}
