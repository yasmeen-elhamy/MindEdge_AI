"""
rag.py — RAG Index & Memory Module
=====================================
Embedding, indexing, retrieval, and persistent chat memory.
"""

import os
import glob
from huggingface_hub import snapshot_download, InferenceClient
from sentence_transformers import SentenceTransformer
import chromadb
from config import OUTPUT_DIR, RAG_DIR, CHAT_LOG_DIR, MODEL_DIR
from llm import generate_response
from config import export_to_folder

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


# ── Memory ──────────────────────────────────────────────────────────────────

def auto_scan_text(text_content: str, hf_token: str, model_id: str):
    """Scans text for definitions and rules using the LLM, then saves them."""
    print("\n[🔍] EduScan is deep-scanning for Definitions and Rules...")

    prompt = f"""
[SYSTEM: PHYSICS DATA EXTRACTOR]
Extract all terms and formulas from the text.

FORMAT:
- For concepts: [DEFINITION: Name] full_description
- For math: [RULE: Name] formula_only

TEXT:
{text_content[:8000]}

RESULTS (English Only):
"""
    try:
        client   = InferenceClient(model=model_id, token=hf_token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        result = response.choices[0].message.content

        for line in result.split('\n'):
            line = line.strip()
            if not line or "]" not in line:
                continue
            try:
                header, body = line.split("]", 1)
                header = header.replace("[", "").upper()
                body   = body.strip().lstrip(":").strip()
                if "DEFINITION" in header:
                    name = header.replace("DEFINITION", "").strip() or "Concept"
                    export_to_folder("definitions", name, body)
                elif "RULE" in header:
                    name = header.replace("RULE", "").strip() or "Physics Law"
                    export_to_folder("rules", name, body)
            except Exception:
                if "=" in line:
                    export_to_folder("rules", "Equation", line)

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
