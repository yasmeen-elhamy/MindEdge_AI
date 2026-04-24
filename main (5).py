"""
EduScan – Intelligent Study Scanner & Tutor
============================================
Run:  python main.py
      python main.py file1.pdf file2.png

Structure:
  main.py   ← entry point (this file)
  config.py ← paths, token, file picker, helpers
  llm.py    ← Qwen API calls (correct, summarize, generate)
  ocr.py    ← OCR pipeline (pytesseract + pdf2image)
  rag.py    ← RAG index, retrieval, memory, auto-scan
  chat.py   ← interactive chat loop
  image.py  ← vision model (graph/figure analysis)
"""

import os
import sys
import glob
import zipfile
import platform
import subprocess
import nltk
from datetime import datetime
from pathlib import Path

# ── Local modules ──────────────────────────────────────────────────────────
from config  import (setup_output_dirs, load_token, pick_files,
                     PROJECT_ROOT, OUTPUT_DIR, CHAT_LOG_DIR)
from llm     import summarize_text, test_connection
from ocr     import setup_tesseract, run_ocr_pipeline
from rag     import (build_or_load_index, auto_scan_text)
from chat    import run_chat_loop
from image   import extract_and_analyze_graphs

# ── Bootstrap ──────────────────────────────────────────────────────────────
setup_output_dirs()
HF_TOKEN   = load_token()
FILE_PATHS = pick_files()

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

print(f"\n[✅] Configuration complete.")
print(f"     OUTPUT_DIR : {OUTPUT_DIR}")
print(f"     FILE_PATHS : {len(FILE_PATHS)} file(s)")

# ── LLM connection test ────────────────────────────────────────────────────
if not test_connection():
    print("[❌] Connection test failed. Check your HF_TOKEN and try again.")
    sys.exit(1)

# ── OCR setup ─────────────────────────────────────────────────────────────
setup_tesseract()

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 — OCR
# ══════════════════════════════════════════════════════════════════════════
all_corrected_texts, last_raw_text = run_ocr_pipeline(FILE_PATHS)

if not all_corrected_texts:
    print("[❌] No text could be extracted from any file. Exiting.")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1b — GRAPH / FIGURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("🖼️   STAGE 1b — GRAPH / FIGURE ANALYSIS")
print("═" * 60)

graph_results = extract_and_analyze_graphs(FILE_PATHS[0] if len(FILE_PATHS) == 1 else FILE_PATHS[0])

graphs_md_path = os.path.join(OUTPUT_DIR, "graphs.md")
with open(graphs_md_path, "w", encoding="utf-8") as fh:
    fh.write("# Graph Analysis Results\n\n")
    for r in graph_results:
        fh.write(f"## {os.path.basename(r['image_path'])}\n\n")
        fh.write(f"{r['analysis']}\n\n---\n\n")

print(f"[✅] {len(graph_results)} graph(s) analyzed → {graphs_md_path}")

# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 — SUMMARIZATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("📄  STAGE 2 — SUMMARIZATION")
print("═" * 60)

for path, corrected in all_corrected_texts.items():
    print(f"\n[📌] Summarizing: {os.path.basename(path)}")
    summary = summarize_text(corrected)
    if summary.startswith("Error:"):
        summary = "No summary generated."
    stem = os.path.basename(path).replace(" ", "_")
    with open(os.path.join(OUTPUT_DIR, f"{stem}_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary)

print(f"\n[✅] Summarization done. {len(all_corrected_texts)} summary(ies) saved.")

# ══════════════════════════════════════════════════════════════════════════
# STAGE 3 — RAG INDEX
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("🗂️   STAGE 3 — RAG INDEX")
print("═" * 60)

collection = build_or_load_index(folder=OUTPUT_DIR)
print(f"[✅] RAG index contains {collection.count()} document(s).")

# ── Auto-scan for rules & definitions ────────────────────────────────────
if last_raw_text:
    from llm import _QW_MODEL_ID
    auto_scan_text(last_raw_text, HF_TOKEN, _QW_MODEL_ID)

# ══════════════════════════════════════════════════════════════════════════
# STAGE 4 — CHAT
# ══════════════════════════════════════════════════════════════════════════
chat_history = run_chat_loop(collection)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if chat_history:
    session_path = os.path.join(OUTPUT_DIR, f"chat_session_{timestamp}.md")
    with open(session_path, "w", encoding="utf-8") as fh:
        for entry in chat_history:
            fh.write(f"### User:\n{entry['question']}\n\n### Assistant:\n{entry['answer']}\n\n---\n")
    print(f"\n[✅] {len(chat_history)} exchange(s) saved → {session_path}")
else:
    print("\n[ℹ️]  No questions were asked.")

# ══════════════════════════════════════════════════════════════════════════
# STAGE 5 — LIST & EXPORT
# ══════════════════════════════════════════════════════════════════════════
print(f"\n📂 {OUTPUT_DIR}")
for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", "*"), recursive=True)):
    if os.path.isfile(f):
        print(f"   {os.path.relpath(f, OUTPUT_DIR):60s}  {os.path.getsize(f):>8,} bytes")

print("\n" + "═" * 60)
print("📦  STAGE 5 — EXPORT")
print("═" * 60)

EXPORT_PATH = str(PROJECT_ROOT / f"EduScan_Export_{timestamp}.zip")
collected   = []

with zipfile.ZipFile(EXPORT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder_name, folder_path in [("output", OUTPUT_DIR), ("chat_logs", CHAT_LOG_DIR)]:
        for fpath in sorted(glob.glob(os.path.join(folder_path, "**", "*"), recursive=True)):
            if os.path.isfile(fpath):
                arcname = os.path.join(folder_name, os.path.relpath(fpath, folder_path))
                zf.write(fpath, arcname)
                collected.append((arcname, os.path.getsize(fpath)))

zip_size = os.path.getsize(EXPORT_PATH)
print(f"  📁  {len(collected)} files  |  💾  {zip_size/1024:.1f} KB  |  📍  {EXPORT_PATH}")

_sys = platform.system()
try:
    if _sys == "Windows":   subprocess.run(["explorer", "/select,", EXPORT_PATH])
    elif _sys == "Darwin":  subprocess.run(["open", "-R", EXPORT_PATH])
    else:                   subprocess.run(["xdg-open", os.path.dirname(EXPORT_PATH)])
    print("[📂] File explorer opened.")
except Exception as exc:
    print(f"[⚠️]  Navigate to: {EXPORT_PATH}")

print("\n[✅] EduScan finished successfully.")
