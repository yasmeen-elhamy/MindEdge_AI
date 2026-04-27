"""
EduScan – Intelligent Study Scanner & Tutor
============================================
Run:  python main.py
      python main.py file1.pdf file2.png

Structure:
  main.py        ← entry point (this file)
  config.py      ← paths, token, file picker, helpers
  llm.py         ← Qwen API calls (correct, summarize, generate)
  ocr.py         ← OCR pipeline (pytesseract + pdf2image)
  rag.py         ← RAG index, retrieval, memory, auto-scan
  chat.py        ← interactive chat loop
  image.py       ← vision model (graph/figure analysis)
  study_plan.py  ← Study Plan Generator
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
from config     import (setup_output_dirs, load_token, pick_files,
                        PROJECT_ROOT, OUTPUT_DIR, CHAT_LOG_DIR)
from llm        import summarize_text, test_connection
from ocr        import setup_tesseract, run_ocr_pipeline
from rag        import (build_or_load_index, auto_scan_text)
from chat       import run_chat_loop
from image      import extract_and_analyze_graphs
from study_plan import generate_study_plan          # ← NEW


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

graph_results = extract_and_analyze_graphs(FILE_PATHS[0], hf_token=HF_TOKEN)

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

# Collect topics from file names while summarizing
detected_topics: list[str] = []

for path, corrected in all_corrected_texts.items():
    print(f"\n[📌] Summarizing: {os.path.basename(path)}")
    summary = summarize_text(corrected)
    if summary.startswith("Error:"):
        summary = "No summary generated."
    stem = os.path.basename(path).replace(" ", "_")
    with open(os.path.join(OUTPUT_DIR, f"{stem}_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary)
    # Use the file stem as a topic hint
    detected_topics.append(Path(path).stem)

print(f"\n[✅] Summarization done. {len(all_corrected_texts)} summary(ies) saved.")

# ══════════════════════════════════════════════════════════════════════════
# STAGE 3 — STUDY PLAN
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("📅  STAGE 3 — STUDY PLAN GENERATION")
print("═" * 60)

# ── Ask the user for study plan parameters ────────────────────────────────
print("\n[📚] Let's build your personalised study plan.")
try:
    sp_days  = int(input("   How many days do you want to study? [default: 7] ").strip() or 7)
    sp_hours = int(input("   How many hours per day?             [default: 2] ").strip() or 2)
    sp_level = input(
        "   Your level? (Beginner / Intermediate / Advanced) [default: Intermediate] "
    ).strip() or "Intermediate"
    if sp_level not in ("Beginner", "Intermediate", "Advanced"):
        sp_level = "Intermediate"
    sp_subject = input(
        f"   Subject name? [default: {detected_topics[0] if detected_topics else 'Study Material'}] "
    ).strip() or (detected_topics[0] if detected_topics else "Study Material")
except (ValueError, EOFError):
    # Non-interactive environment — use safe defaults
    sp_days, sp_hours, sp_level = 7, 2, "Intermediate"
    sp_subject = detected_topics[0] if detected_topics else "Study Material"
    print("[ℹ️]  Using default study plan parameters.")

print(
    f"\n[🔧] Parameters → subject='{sp_subject}' | "
    f"{sp_days} days | {sp_hours}h/day | {sp_level}"
)

study_plan_dict = generate_study_plan(
    topics        = detected_topics if detected_topics else [sp_subject],
    days          = sp_days,
    hours_per_day = sp_hours,
    subject       = sp_subject,
    level         = sp_level,
    collection    = None,   # RAG index not yet built at this stage
)

# ── Save the plan to Markdown ─────────────────────────────────────────────
sp_md_path = os.path.join(OUTPUT_DIR, "study_plan.md")
with open(sp_md_path, "w", encoding="utf-8") as fh:
    fh.write(f"# Study Plan — {sp_subject}\n\n")
    for day_label, details in study_plan_dict.items():
        fh.write(f"## {day_label}\n\n{details}\n\n---\n\n")

print(f"[✅] {len(study_plan_dict)} day(s) generated → {sp_md_path}")

# ══════════════════════════════════════════════════════════════════════════
# STAGE 4 — RAG INDEX
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("🗂️   STAGE 4 — RAG INDEX")
print("═" * 60)

collection = build_or_load_index(folder=OUTPUT_DIR)
print(f"[✅] RAG index contains {collection.count()} document(s).")

# ── Auto-scan for rules & definitions ────────────────────────────────────
if last_raw_text:
    from llm import _QW_MODEL_ID
    auto_scan_text(last_raw_text, HF_TOKEN, _QW_MODEL_ID)

# ══════════════════════════════════════════════════════════════════════════
# STAGE 5 — CHAT
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
# STAGE 6 — LIST & EXPORT
# ══════════════════════════════════════════════════════════════════════════
print(f"\n📂 {OUTPUT_DIR}")
for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", "*"), recursive=True)):
    if os.path.isfile(f):
        print(f"   {os.path.relpath(f, OUTPUT_DIR):60s}  {os.path.getsize(f):>8,} bytes")

print("\n" + "═" * 60)
print("📦  STAGE 6 — EXPORT")
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