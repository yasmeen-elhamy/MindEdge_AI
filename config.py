"""
config.py — Configuration & Setup
===================================
All paths, tokens, and directory setup.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import tkinter as tk
from tkinter import filedialog

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR   = str(PROJECT_ROOT / "output")
RAG_DIR      = str(PROJECT_ROOT / "rag_index")
CHAT_LOG_DIR = str(PROJECT_ROOT / "chat_logs")
MODEL_DIR    = str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")
TEMP_DIR     = str(PROJECT_ROOT / "temp")


def setup_output_dirs():
    """Creates all required directories on startup."""
    dirs = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, "rules"),
        os.path.join(OUTPUT_DIR, "definitions"),
        RAG_DIR,
        CHAT_LOG_DIR,
        MODEL_DIR,
        TEMP_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_token() -> str:
    """Loads HF_TOKEN from .env or prompts user."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        print("[✅] HF_TOKEN loaded from .env")
    else:
        token = input("⚠️  HF_TOKEN not found. Paste your token: ").strip()
        os.environ["HF_TOKEN"] = token
        print("[✅] HF_TOKEN set for this session (not saved).")
    return token


def pick_files() -> list:
    """Opens file picker or reads from CLI args."""
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
        print(f"[✅] {len(paths)} file(s) provided via command line:")
        for p in paths:
            print(f"     • {p}")
        return paths

    print("\n[📂] Opening file picker — select your PDFs / images…")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    paths = list(filedialog.askopenfilenames(
        parent=root,
        title="Select study files (PDF / JPG / PNG)",
        filetypes=[
            ("Study materials", "*.pdf *.jpg *.jpeg *.png"),
            ("PDF files",       "*.pdf"),
            ("Image files",     "*.jpg *.jpeg *.png"),
            ("All files",       "*.*"),
        ],
    ))
    root.destroy()

    if paths:
        print(f"[✅] {len(paths)} file(s) selected:")
        for p in paths:
            print(f"     • {p}")
    else:
        print("[⚠️]  No files selected — exiting.")
        sys.exit(0)

    return paths


def export_to_folder(category: str, title: str, content: str):
    """Appends a titled entry to All_<category>.txt inside output/<category>/."""
    folder_path = os.path.join(OUTPUT_DIR, category)
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, f"All_{category}.txt")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"{title}: {content}\n")
    print(f"[✅] Saved to {category}: {title}")
