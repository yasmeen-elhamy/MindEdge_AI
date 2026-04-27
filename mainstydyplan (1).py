"""
╔══════════════════════════════════════════════════════════════╗
║          AI-POWERED STUDY PLAN GENERATOR                     ║
║  PDF / Image → OCR → RAG → LLM → Personalised Study Plan    ║
╚══════════════════════════════════════════════════════════════╝

Single-file application.  Run:  python main.py

Dependencies:
    pip install openai pillow numpy                              \
                sentence-transformers chromadb                   \
                python-doctr[torch] PyMuPDF
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import sys
import re
import math
import uuid
import textwrap
import logging
import datetime
from pathlib import Path
from typing import Optional

# ── OpenAI client (used for HuggingFace Router)
from openai import OpenAI

# ── Standard-library GUI
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Third-party
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None

# ── Lazy imports (checked inside functions that need them)
doctr_available                 = False
sentence_transformers_available = False
chromadb_available              = False
fitz_available                  = False

# ── Detect available optional packages at startup (no auto-install)
try:
    import doctr  # noqa: F401
    doctr_available = True
except ImportError:
    pass

try:
    import fitz  # noqa: F401
    fitz_available = True
except ImportError:
    pass

try:
    import sentence_transformers  # noqa: F401
    sentence_transformers_available = True
except ImportError:
    pass

try:
    import chromadb  # noqa: F401
    chromadb_available = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("StudyPlan")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # Embedding model (local, no key needed)
    EMBED_MODEL   : str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG
    CHUNK_SIZE    : int = 400       # tokens (words approx.)
    CHUNK_OVERLAP : int = 50
    TOP_K         : int = 8         # chunks returned per query

    # ChromaDB
    CHROMA_PATH   : str = "./study_plan_chroma"
    COLLECTION    : str = "study_materials"

    # Output
    OUTPUT_FILE   : str = "study_plan.md"

CFG = Config()

# ─────────────────────────────────────────────────────────────────────────────
# LLM  —  HuggingFace Router via OpenAI-compatible client
# ─────────────────────────────────────────────────────────────────────────────
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError
import time

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
HF_TOKEN = "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"
_QW_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
_QW_BASE_URL      = "https://router.huggingface.co/v1"
_QW_MAX_TOKENS    = 800
_QW_TEMPERATURE   = 0.5
_QW_TOP_P         = 0.9
_QW_TIMEOUT       = 60
_QW_MAX_RETRIES   = 4
_QW_BACKOFF_BASE  = 2.0
_QW_SYSTEM_PROMPT = "You are Qwen, a helpful AI assistant."
_QW_CLIENT        = None


# ─────────────────────────────────────────────────────────
# CLIENT LOADER
# ─────────────────────────────────────────────────────────
def _get_client():
    global _QW_CLIENT
    if _QW_CLIENT is None:
        _QW_CLIENT = OpenAI(
            base_url=_QW_BASE_URL,
            api_key=HF_TOKEN
        )
    return _QW_CLIENT


# ─────────────────────────────────────────────────────────
# MAIN LLM FUNCTION (UPGRADED)
# ─────────────────────────────────────────────────────────
def llm_chat(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = _QW_MAX_TOKENS,
    temperature: float = _QW_TEMPERATURE,
) -> str:

    client = _get_client()
    print(f"🚀 Calling HF Router → {_QW_MODEL_ID}")

    for attempt in range(1, _QW_MAX_RETRIES + 1):
        backoff = _QW_BACKOFF_BASE ** (attempt - 1)

        try:
            response = client.chat.completions.create(
                model=_QW_MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=_QW_TOP_P,
                timeout=_QW_TIMEOUT,
            )

            answer = response.choices[0].message.content.strip()

            usage = getattr(response, "usage", None)
            if usage:
                print(f"✅ Tokens → prompt:{usage.prompt_tokens} | completion:{usage.completion_tokens}")
            else:
                print(f"✅ Response received ({len(answer)} chars)")

            return answer

        # ─────────────── API ERRORS ───────────────
        except APIStatusError as e:
            code = e.status_code

            if code == 503:
                wait = backoff
                print(f"⏳ Model loading... retry in {wait}s ({attempt}/{_QW_MAX_RETRIES})")

            elif code == 429:
                wait = backoff
                print(f"🚦 Rate limit... retry in {wait}s ({attempt}/{_QW_MAX_RETRIES})")

            else:
                msg = f"HTTP {code}: {str(e)[:200]}"
                print(f"❌ {msg}")
                return f"Error: {msg}"

            if attempt < _QW_MAX_RETRIES:
                time.sleep(wait)
                continue
            return f"Error: HTTP {code} after retries"

        # ─────────────── TIMEOUT ───────────────
        except APITimeoutError:
            print(f"⏱ Timeout ({attempt}/{_QW_MAX_RETRIES})")
            if attempt < _QW_MAX_RETRIES:
                time.sleep(backoff)
                continue
            return "Error: Timeout"

        # ─────────────── CONNECTION ───────────────
        except APIConnectionError as e:
            print(f"❌ Connection error: {e}")
            if attempt < _QW_MAX_RETRIES:
                time.sleep(backoff)
                continue
            return f"Error: Connection failed"

        # ─────────────── UNKNOWN ───────────────
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return f"Error: {str(e)}"

    return "Error: All retries failed"

# ─────────────────────────────────────────────────────────────────────────────
# FILE SELECTION  (tkinter)
# ─────────────────────────────────────────────────────────────────────────────
def select_files() -> list[str]:
    """Open a tkinter dialog and return selected file paths."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    paths = filedialog.askopenfilenames(
        title="Select study materials (PDF / JPG / PNG)",
        filetypes=[
            ("Study materials", "*.pdf *.jpg *.jpeg *.png"),
            ("PDF files",       "*.pdf"),
            ("Image files",     "*.jpg *.jpeg *.png"),
            ("All files",       "*.*"),
        ],
    )
    root.destroy()

    if not paths:
        log.warning("No files selected.")
        return []

    log.info("Selected %d file(s):", len(paths))
    for p in paths:
        log.info("  • %s", p)
    return list(paths)

# ─────────────────────────────────────────────────────────────────────────────
# OCR & TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def _ocr_image_doctr(path: str) -> str:
    """Run doctr OCR on an image file."""
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        log.info("  Running doctr OCR on %s ...", Path(path).name)
        model  = ocr_predictor(pretrained=True)
        doc    = DocumentFile.from_images(path)
        result = model(doc)

        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = [w.value for w in line.words]
                    lines.append(" ".join(words))
        return "\n".join(lines)
    except Exception as exc:
        log.error("doctr OCR failed for %s: %s", path, exc)
        return ""


def _ocr_image_pil_fallback(path: str) -> str:
    """Fallback: just return a note that OCR wasn't available."""
    log.warning("OCR not available; could not extract text from image %s", path)
    return f"[Image file: {Path(path).name} — install doctr for OCR]"


def _extract_pdf_fitz(path: str) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        log.info("  Extracting PDF text (fitz): %s ...", Path(path).name)
        doc        = fitz.open(path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(pages_text)
    except Exception as exc:
        log.error("fitz PDF extraction failed for %s: %s", path, exc)
        return ""


def _extract_pdf_doctr(path: str) -> str:
    """Extract text from a PDF using doctr (OCR-based)."""
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        log.info("  Running doctr OCR on PDF: %s ...", Path(path).name)
        model  = ocr_predictor(pretrained=True)
        doc    = DocumentFile.from_pdf(path)
        result = model(doc)

        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = [w.value for w in line.words]
                    lines.append(" ".join(words))
        return "\n".join(lines)
    except Exception as exc:
        log.error("doctr PDF OCR failed for %s: %s", path, exc)
        return ""


def extract_text_from_file(path: str) -> str:
    """Route to the appropriate extraction method by file extension."""
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        text = _extract_pdf_fitz(path) if fitz_available else ""
        if len(text.strip()) < 100 and doctr_available:
            log.info("  PDF appears scanned — switching to doctr OCR ...")
            text = _extract_pdf_doctr(path)
        elif not fitz_available and doctr_available:
            text = _extract_pdf_doctr(path)
        return text

    elif ext in {".jpg", ".jpeg", ".png"}:
        if doctr_available:
            return _ocr_image_doctr(path)
        return _ocr_image_pil_fallback(path)

    else:
        log.warning("Unsupported file type: %s", ext)
        return ""


def extract_all_texts(file_paths: list[str]) -> str:
    """Extract and combine text from all selected files."""
    if not file_paths:
        raise ValueError("No files provided for extraction.")

    all_parts: list[str] = []
    for fp in file_paths:
        log.info("Processing: %s", fp)
        text = extract_text_from_file(fp)
        if text.strip():
            all_parts.append(f"### Source: {Path(fp).name}\n\n{text}")
        else:
            log.warning("No text extracted from %s", fp)

    if not all_parts:
        raise RuntimeError(
            "Could not extract any text from the provided files.\n"
            "  • For PDFs:  install PyMuPDF  (pip install PyMuPDF)\n"
            "  • For images: install doctr   (pip install 'python-doctr[torch]')"
        )

    combined = "\n\n" + "─" * 60 + "\n\n".join(all_parts)
    log.info("Total extracted text: %d characters", len(combined))
    return combined

# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING VIA LLM
# ─────────────────────────────────────────────────────────────────────────────
def clean_and_structure_text(raw_text: str) -> str:
    """Use the LLM to clean OCR noise and improve formatting."""
    log.info("Cleaning and structuring extracted text with LLM ...")

    snippet = raw_text[:6000]

    system = (
        "You are a text-cleaning assistant. "
        "You receive raw text that may have been extracted by OCR. "
        "Your task is to:\n"
        "  1. Fix OCR errors and typos.\n"
        "  2. Restore proper sentence and paragraph structure.\n"
        "  3. Remove duplicate or garbled lines.\n"
        "  4. Keep ALL academic/technical content intact.\n"
        "Return ONLY the cleaned text, no commentary."
    )
    user = f"Clean and structure the following study material:\n\n{snippet}"

    try:
        cleaned = llm_chat(system, user, max_tokens=3000)
        if len(raw_text) > 6000:
            cleaned += "\n\n" + raw_text[6000:]
        return cleaned
    except Exception as exc:
        log.warning("Text cleaning skipped (LLM error): %s", exc)
        return raw_text

# ─────────────────────────────────────────────────────────────────────────────
# TEXT CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = CFG.CHUNK_SIZE,
    overlap: int = CFG.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words  = text.split()
    chunks : list[str] = []
    start  = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    log.info(
        "Created %d text chunks (size≈%d words, overlap=%d)",
        len(chunks), chunk_size, overlap,
    )
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# RAG — EMBEDDINGS + CHROMADB
# ─────────────────────────────────────────────────────────────────────────────
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    if not sentence_transformers_available:
        raise RuntimeError("sentence-transformers not installed.")
    from sentence_transformers import SentenceTransformer
    log.info("Loading embedding model: %s ...", CFG.EMBED_MODEL)
    _embed_model = SentenceTransformer(CFG.EMBED_MODEL)
    return _embed_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return L2-normalised embeddings as nested Python lists."""
    model      = _get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def build_rag_store(chunks: list[str]) -> object:
    """Create a ChromaDB collection and populate it with chunk embeddings."""
    if not chromadb_available:
        raise RuntimeError("chromadb not installed. Run: pip install chromadb")

    import chromadb

    log.info("Building RAG vector store (%d chunks) ...", len(chunks))
    client_db = chromadb.PersistentClient(path=CFG.CHROMA_PATH)

    try:
        client_db.delete_collection(CFG.COLLECTION)
    except Exception:
        pass

    collection = client_db.create_collection(
        name=CFG.COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch  = chunks[i : i + batch_size]
        ids    = [str(uuid.uuid4()) for _ in batch]
        embeds = embed_texts(batch)
        collection.add(ids=ids, embeddings=embeds, documents=batch)
        log.info("  Stored chunks %d–%d", i, i + len(batch) - 1)

    log.info("RAG store built successfully.")
    return collection


def rag_query(collection, query: str, top_k: int = CFG.TOP_K) -> str:
    """Retrieve the most relevant chunks for a query and return them joined."""
    query_embed = embed_texts([query])[0]
    results     = collection.query(
        query_embeddings=[query_embed],
        n_results=min(top_k, collection.count()),
    )
    docs = results.get("documents", [[]])[0]
    return "\n\n---\n\n".join(docs)

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_topics(study_text: str, collection=None) -> list[str]:
    """
    Use the LLM to extract main topics and subtopics from the study material.
    If a RAG collection exists, retrieve the most relevant chunks first.
    """
    log.info("Extracting topics from study material ...")

    if collection is not None:
        context = rag_query(
            collection,
            "main topics, subjects, chapters, key concepts",
            top_k=10,
        )
    else:
        context = study_text[:5000]

    system = (
        "You are an expert curriculum designer. "
        "Your task is to read study material and output a clean, structured list of topics."
    )
    user = (
        "Extract the main topics and subtopics from the following study material.\n"
        "Return ONLY a numbered list, one topic per line, ordered from fundamental to advanced.\n"
        "Example format:\n"
        "1. Topic A\n"
        "2. Topic B\n"
        "   2a. Subtopic B1\n"
        "   2b. Subtopic B2\n\n"
        f"Study material:\n\n{context}"
    )

    try:
        raw = llm_chat(system, user, max_tokens=1024, temperature=0.2)
    except Exception as exc:
        log.error("Topic extraction failed: %s", exc)
        return ["Topic 1", "Topic 2", "Topic 3"]

    topics: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r"^[\d]+[a-z]?[.)]\s*|^-\s*", "", stripped).strip()
        if cleaned:
            topics.append(cleaned)

    if not topics:
        topics = [ln.strip() for ln in raw.splitlines() if ln.strip()][:10]

    log.info("Extracted %d topic(s).", len(topics))
    return topics

# ─────────────────────────────────────────────────────────────────────────────
# USER PREFERENCES  (CLI)
# ─────────────────────────────────────────────────────────────────────────────
class StudyPreferences:
    subject      : str
    duration_days: int
    daily_hours  : int
    level        : str


def _prompt(msg: str, choices: list[str] | None = None, cast=str, default=None):
    """CLI prompt with optional validation."""
    choices_str = f"  [{' / '.join(choices)}]" if choices else ""
    while True:
        raw = input(f"  {msg}{choices_str}: ").strip()
        if not raw and default is not None:
            return default
        if choices and raw not in choices:
            print(f"    ⚠ Please enter one of: {', '.join(choices)}")
            continue
        try:
            return cast(raw)
        except ValueError:
            print("    ⚠ Invalid input, please try again.")


def collect_user_preferences() -> StudyPreferences:
    """Interactively ask the user for their study parameters."""
    print("\n" + "═" * 58)
    print("   STUDY PLAN PREFERENCES")
    print("═" * 58)

    prefs               = StudyPreferences()
    prefs.subject       = _prompt("Subject name")
    prefs.duration_days = _prompt("Study duration (days)", cast=int, default=30)
    prefs.daily_hours   = _prompt(
        "Daily study hours", choices=["1", "2", "3"], cast=int, default=2
    )
    prefs.level         = _prompt(
        "Difficulty level",
        choices=["Beginner", "Intermediate", "Advanced"],
        default="Intermediate",
    )

    print("\n" + "─" * 58)
    print(f"  Subject   : {prefs.subject}")
    print(f"  Duration  : {prefs.duration_days} days")
    print(f"  Daily hrs : {prefs.daily_hours}h/day")
    print(f"  Level     : {prefs.level}")
    print("─" * 58 + "\n")

    return prefs

# ─────────────────────────────────────────────────────────────────────────────
# STUDY PLAN GENERATOR  (CORE FEATURE)
# ─────────────────────────────────────────────────────────────────────────────
def _build_plan_prompt(
    subject      : str,
    duration_days: int,
    daily_hours  : int,
    level        : str,
    topics       : list[str],
    rag_context  : str,
) -> tuple[str, str]:
    """Build the system + user prompt pair for study-plan generation."""

    total_days    = duration_days
    study_days    = math.ceil(total_days * 0.85)
    revision_days = total_days - study_days
    topics_str    = "\n".join(f"  - {t}" for t in topics)

    level_guidance = {
        "Beginner": (
            "Start from absolute basics. "
            "Spend extra time on foundational concepts. "
            "Include many practical examples. "
            "Avoid overwhelming the learner."
        ),
        "Intermediate": (
            "Assume basic familiarity. "
            "Balance theory and practice equally. "
            "Introduce progressively more complex topics mid-way."
        ),
        "Advanced": (
            "Move quickly through fundamentals. "
            "Focus on depth, edge cases, and advanced techniques. "
            "Include challenging exercises and real-world applications."
        ),
    }.get(level, "")

    system = (
        "You are an expert curriculum designer and study coach. "
        "You create detailed, realistic, personalized study plans. "
        "Every plan you produce is:\n"
        "  • Divided into individual DAYS (Day 1, Day 2, …)\n"
        "  • Progressive: easy → medium → hard\n"
        "  • Realistic: respects the daily hour constraint\n"
        "  • Complete: covers all provided topics\n"
        "  • Includes periodic REVISION days\n"
        "  • Formatted in clean Markdown"
    )

    user = (
        f"Create a COMPLETE, DAY-BY-DAY study plan with the following parameters:\n\n"
        f"Subject       : {subject}\n"
        f"Total duration: {total_days} days\n"
        f"Study days    : {study_days}\n"
        f"Revision days : {revision_days}\n"
        f"Daily hours   : {daily_hours} hour(s) per day\n"
        f"Level         : {level}\n\n"
        f"Level guidance: {level_guidance}\n\n"
        f"Topics to cover:\n{topics_str}\n\n"
        f"Relevant study material context:\n{rag_context}\n\n"
        f"STRICT OUTPUT FORMAT (Markdown, repeat for every day):\n\n"
        f"## Day 1\n"
        f"- **Topic**: <topic name>\n"
        f"- **Subtopics**: <comma-separated list>\n"
        f"- **Tasks**: <numbered list of concrete learning tasks>\n"
        f"- **Resources**: <what to read/watch/practice>\n"
        f"- **Estimated Time**: {daily_hours}h\n"
        f"- **Notes**: <any special tips for this day>\n\n"
        f"(Continue for ALL {total_days} days. "
        f"Mark revision days clearly as '## Day N — REVISION'.)\n\n"
        f"End with a short **Summary** section."
    )

    return system, user


def generate_study_plan(
    subject      : str,
    duration_days: int,
    daily_hours  : int,
    level        : str,
    topics       : list[str],
    collection   = None,
) -> str:
    """
    Generate a complete day-by-day study plan using the LLM.
    If a RAG collection is available it is used to ground the plan.
    """
    log.info(
        "Generating study plan (%d days, %dh/day, %s) ...",
        duration_days, daily_hours, level,
    )

    if collection is not None:
        query   = f"{subject} study plan {level} topics schedule"
        rag_ctx = rag_query(collection, query, top_k=6)
    else:
        rag_ctx = "\n".join(topics[:30])

    system, user = _build_plan_prompt(
        subject, duration_days, daily_hours, level, topics, rag_ctx
    )

    total_days = duration_days
    max_tok    = min(3000, 120 * total_days)

    try:
        plan_text = llm_chat(system, user, max_tokens=max_tok, temperature=0.5)
    except Exception as exc:
        log.error("Study plan generation failed: %s", exc)
        plan_text = _fallback_plan(subject, duration_days, daily_hours, level, topics)

    return plan_text


def _fallback_plan(subject, duration_days, hours, level, topics) -> str:
    """Minimal local fallback when LLM is unavailable."""
    lines = [
        f"# Study Plan: {subject}",
        f"**Level**: {level}  |  **Duration**: {duration_days} days  |  **{hours}h/day**",
        "",
        "> ⚠ Generated locally (LLM API unavailable).",
        "",
    ]

    # Insert a revision day every 7 days; remaining days get topics
    revision_interval = 7
    day_index         = 1
    topic_index       = 0

    for day_index in range(1, duration_days + 1):
        if day_index % revision_interval == 0:
            lines += [
                f"## Day {day_index} — REVISION",
                f"- **Topic**: Review previous topics",
                f"- **Tasks**: Revisit notes, solve practice problems, summarise key points",
                f"- **Estimated Time**: {hours}h",
                "",
            ]
        else:
            topic = topics[topic_index % len(topics)]
            topic_index += 1
            lines += [
                f"## Day {day_index}",
                f"- **Topic**: {topic}",
                f"- **Tasks**: Read and take notes, practice exercises",
                f"- **Estimated Time**: {hours}h",
                "",
            ]

    lines.append("---\n*Study hard and stay consistent!*")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT — PRINT + SAVE
# ─────────────────────────────────────────────────────────────────────────────
def _build_header(prefs: StudyPreferences) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"# 📚 AI Study Plan: {prefs.subject}\n\n"
        f"| Parameter      | Value                    |\n"
        f"|----------------|--------------------------|\n"
        f"| Generated      | {now}                    |\n"
        f"| Subject        | {prefs.subject}           |\n"
        f"| Duration       | {prefs.duration_days} days |\n"
        f"| Daily hours    | {prefs.daily_hours}h/day  |\n"
        f"| Level          | {prefs.level}             |\n\n"
        f"---\n\n"
    )


def print_and_save_plan(plan_text: str, prefs: StudyPreferences) -> None:
    """Print the plan to stdout and save it to a Markdown file."""
    header   = _build_header(prefs)
    full_doc = header + plan_text

    separator = "═" * 58
    print(f"\n{separator}")
    print(f"  YOUR STUDY PLAN")
    print(separator)

    for line in full_doc.splitlines():
        if line.startswith("## Day"):
            print("")
        print(line)

    print(f"\n{separator}\n")

    out_path = Path(CFG.OUTPUT_FILE)
    try:
        out_path.write_text(full_doc, encoding="utf-8")
        log.info("Study plan saved to: %s", out_path.resolve())
        print(f"✅  Plan saved to:  {out_path.resolve()}\n")
    except OSError as exc:
        log.error("Could not save plan: %s", exc)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("╔══════════════════════════════════════════════════════╗")
    print("║        AI-Powered Study Plan Generator               ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── 1. File selection ────────────────────────────────────────────────────
    print("\n[1/6] Select your study materials (PDF / JPG / PNG) ...")
    file_paths = select_files()

    if not file_paths:
        print("No files chosen via dialog. Enter file path(s) manually.")
        raw        = input("  Paths (comma-separated): ").strip()
        file_paths = [p.strip() for p in raw.split(",") if p.strip()]

    if not file_paths:
        print("❌  No study materials provided. Exiting.")
        sys.exit(1)

    # ── 2. Text extraction ───────────────────────────────────────────────────
    print("\n[2/6] Extracting text from files ...")
    try:
        raw_text = extract_all_texts(file_paths)
    except RuntimeError as exc:
        print(f"❌  Extraction failed: {exc}")
        sys.exit(1)

    if not raw_text.strip():
        print("❌  No text could be extracted. Check file format / quality.")
        sys.exit(1)

    # ── 3. Clean text ────────────────────────────────────────────────────────
    print("\n[3/6] Cleaning and structuring text with LLM ...")
    try:
        clean_text = clean_and_structure_text(raw_text)
    except Exception as exc:
        log.warning("Text cleaning skipped: %s", exc)
        clean_text = raw_text

    # ── 4. Build RAG ─────────────────────────────────────────────────────────
    print("\n[4/6] Building RAG vector store ...")
    collection = None
    if sentence_transformers_available and chromadb_available:
        try:
            chunks     = chunk_text(clean_text)
            collection = build_rag_store(chunks)
        except Exception as exc:
            log.error("RAG build failed (continuing without RAG): %s", exc)
    else:
        log.warning("sentence-transformers or chromadb unavailable — skipping RAG.")

    # ── 5. User preferences ──────────────────────────────────────────────────
    print("\n[5/6] Please enter your study preferences ...")
    prefs = collect_user_preferences()

    # ── 6. Extract topics ────────────────────────────────────────────────────
    print("\n[6/6] Extracting topics from study material ...")
    try:
        topics = extract_topics(clean_text, collection)
    except Exception as exc:
        log.error("Topic extraction failed: %s", exc)
        topics = ["Introduction", "Core Concepts", "Advanced Topics", "Revision"]

    print("\n  Topics identified:")
    for t in topics:
        print(f"    • {t}")

    # ── 7. Generate study plan ───────────────────────────────────────────────
    print("\n  Generating your personalised study plan (this may take ~30 s) ...")
    try:
        plan = generate_study_plan(
            subject       = prefs.subject,
            duration_days = prefs.duration_days,
            daily_hours   = prefs.daily_hours,
            level         = prefs.level,
            topics        = topics,
            collection    = collection,
        )
    except Exception as exc:
        log.error("Plan generation error: %s", exc)
        plan = _fallback_plan(
            prefs.subject, prefs.duration_days,
            prefs.daily_hours, prefs.level, topics,
        )

    # ── 8. Output ────────────────────────────────────────────────────────────
    print_and_save_plan(plan, prefs)

    print("Done! 🎓  Good luck with your studies!\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(0)
    except Exception as exc:
        log.error("Unexpected error: %s", exc, exc_info=True)
        sys.exit(1)
