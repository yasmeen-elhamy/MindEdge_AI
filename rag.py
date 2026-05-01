"""
rag.py — RAG Index & Memory Module
=====================================
Embedding, indexing, retrieval, and persistent chat memory.
"""
 
import os
import glob
import shutil
from huggingface_hub import snapshot_download, InferenceClient
from sentence_transformers import SentenceTransformer
import chromadb
from config import OUTPUT_DIR, RAG_DIR, CHAT_LOG_DIR, MODEL_DIR
from llm import generate_response
 
# ── مسار الميموري ──────────────────────────────────────────────────────────
MEMORY_DIR = os.path.join(OUTPUT_DIR, "memory")
 
# ── Load embedding model ──────────────────────────────────────────────────
print("\n[🔄] Loading sentence-transformer embedding model…")
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print("     Downloading from HF Hub…")
    snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir=MODEL_DIR)
 
_EMBEDDING_MODEL = SentenceTransformer(MODEL_DIR)
print("[✅] Embedding model loaded.")
 
 
# ── Index ──────────────────────────────────────────────────────────────────
 
def build_or_load_index(folder: str = OUTPUT_DIR, persist_dir: str = RAG_DIR):
    """Builds or loads a ChromaDB vector index from .md files in folder."""
    print("[🗂️]  Building / loading RAG index…")
    docs = []
    for filepath in glob.glob(os.path.join(folder, "*.md")):
        with open(filepath, "r", encoding="utf-8") as fh:
            docs.append({"id": filepath, "text": fh.read()})
 
    client     = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("edu_docs")
 
    if docs:
        embeddings = [_EMBEDDING_MODEL.encode(d["text"]).tolist() for d in docs]
        collection.upsert(
            documents =[d["text"]           for d in docs],
            metadatas =[{"source": d["id"]} for d in docs],
            ids       =[d["id"]             for d in docs],
            embeddings=embeddings,
        )
        print(f"     Upserted {len(docs)} document(s).")
    else:
        print(f"     No .md files found in '{folder}'.")
 
    print(f"[✅] Index ready — {collection.count()} document(s) total.")
    return collection
 
 
def retrieve_passages(query: str, collection, top_k: int = 7) -> list:
    """Retrieves top-k relevant passages for the given query."""
    print(f"[🔎] Retrieving top {top_k} passages…")
    try:
        q_emb   = _EMBEDDING_MODEL.encode(query).tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)
        flat    = []
        for group in results["documents"]:
            flat.extend(group if isinstance(group, list) else [group])
        print(f"     Retrieved {len(flat)} passage(s).")
        return flat
    except Exception as exc:
        print(f"[❌] Retrieval error: {exc}")
        return []
 
 
# ── Helper: حفظ في فولدر الـ PDF ──────────────────────────────────────────
 
def _save_to_pdf_folder(pdf_name: str, category: str, title: str, content: str):
    """Saves a definition or rule inside memory/<pdf_name>/<category>/"""
    folder = os.path.join(MEMORY_DIR, pdf_name, category)
    os.makedirs(folder, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title).strip()
    filepath = os.path.join(folder, f"{safe_title}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{title}: {content}\n")
    print(f"[✅] Saved {category}/{safe_title}")
 
 
# ── Memory ──────────────────────────────────────────────────────────────────
 
def auto_scan_text(text_content: str, hf_token: str, model_id: str, pdf_name: str):
    """Scans text for definitions and rules using the LLM, then saves them."""
    print("\n[🔍] EduScan is deep-scanning for Definitions and Rules...")
 
    prompt = (
        "Extract all terms and formulas from the text below.\n\n"
        "FORMAT (use exactly):\n"
        "- For concepts: [DEFINITION: Name] full_description\n"
        "- For math/formulas: [RULE: Name] formula_only\n\n"
        f"TEXT:\n{text_content[:8000]}\n\nRESULTS (English Only):"
    )
    try:
        result = generate_response(prompt, max_tokens=1500)
 
        # ── امسح الفولدر القديم للـ PDF ده لو موجود ──────────────────
        pdf_folder = os.path.join(MEMORY_DIR, pdf_name)
        if os.path.exists(pdf_folder):
            shutil.rmtree(pdf_folder)
            print(f"[🗑️]  Cleared old data for '{pdf_name}'")
 
        os.makedirs(os.path.join(pdf_folder, "definitions"), exist_ok=True)
        os.makedirs(os.path.join(pdf_folder, "rules"),       exist_ok=True)
        print(f"[📁] Created fresh folders for '{pdf_name}'")
 
        for line in result.split('\n'):
            line = line.strip()
            if not line or "]" not in line:
                continue
            try:
                header, body = line.split("]", 1)
                header = header.replace("[", "").upper()
                body   = body.strip().lstrip(":").strip()
                if "DEFINITION" in header:
                    name = header.replace("DEFINITION:", "").replace("DEFINITION", "").strip() or "Concept"
                    _save_to_pdf_folder(pdf_name, "definitions", name, body)
                elif "RULE" in header:
                    name = header.replace("RULE:", "").replace("RULE", "").strip() or "Physics Law"
                    _save_to_pdf_folder(pdf_name, "rules", name, body)
            except Exception:
                if "=" in line:
                    _save_to_pdf_folder(pdf_name, "rules", "Equation", line)
 
    except Exception as e:
        print(f"[❌] Auto-scan error: {e}")
 
 
def save_chat_log(question: str, answer: str, log_filename: str = "persistent_memory.md"):
    """Appends a Q&A pair to the persistent log file."""
    filepath = os.path.join(CHAT_LOG_DIR, log_filename)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"### ❓ Question\n{question}\n\n### 🤖 Answer\n{answer}\n\n---\n")
 
 
def load_previous_summary(log_filename: str = "persistent_memory.md") -> str:
    """Loads and summarises past chat history."""
    filepath = os.path.join(CHAT_LOG_DIR, log_filename)
    if not os.path.exists(filepath):
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()[-4000:]
        print("[📜] Restoring long-term memory…")
        return generate_response(
            "Summarize the main technical topics from this past history briefly.\n\n" + content,
            max_tokens=200,
        )
    except Exception:
        return ""
 
 
def summarize_chat_history(chat_history: list) -> str:
    """Compresses in-memory chat history to keep context small."""
    if not chat_history:
        return ""
    history_text = "".join(
        [f"User: {e['question']}\nAI: {e['answer']}\n" for e in chat_history]
    )
    return generate_response(
        f"Summarize this conversation briefly:\n\n{history_text}",
        max_tokens=300,
    )