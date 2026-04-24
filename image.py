"""
image.py - PDF Graph Extractor & Analyzer
==========================================
1. Extracts embedded images from PDF pages using PyMuPDF (fitz)
2. Detects if a page region looks like a graph (not pure text)
3. Sends each graph to Qwen/Qwen2.5-VL-7B-Instruct via HF InferenceClient (auto provider)
4. Saves structured analysis reports

Folder structure:
    output/
    images/
        input/      <- extracted raw crops saved here
        analyzed/   <- JSON + TXT reports saved here
"""

import os
import sys
import re
import json
import base64
import io
from pathlib import Path
from datetime import datetime

import fitz
from PIL import Image
from huggingface_hub import InferenceClient


# ==============================================================================
# CONFIG
# ==============================================================================

PROJECT_ROOT  = Path(__file__).parent.resolve()
IMAGES_DIR    = PROJECT_ROOT / "output" / "images"
INPUT_DIR     = IMAGES_DIR / "input"
ANALYZED_DIR  = IMAGES_DIR / "analyzed"

# InferenceClient auto-selects available provider -- no manual config needed
VL_MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_TOKENS    = 2048
TIMEOUT_SEC   = 60
MAX_RETRIES   = 4

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MIN_WIDTH  = 80
MIN_HEIGHT = 80


def setup_image_dirs():
    for d in [INPUT_DIR, ANALYZED_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"[OK] Image folders ready:")
    print(f"     input    -> {INPUT_DIR}")
    print(f"     analyzed -> {ANALYZED_DIR}")


setup_image_dirs()


# ==============================================================================
# STEP 1 -- EXTRACT IMAGES FROM PDF
# ==============================================================================

def _looks_like_graph(img: Image.Image) -> bool:
    try:
        rgb    = img.convert("RGB")
        pixels = list(rgb.getdata())
        total  = len(pixels)
        if total == 0:
            return False
        # Improved detection: higher threshold and check for colored pixels
        colored = sum(
            1 for r, g, b in pixels
            if not (r > 240 and g > 240 and b > 240)
            and not (r < 15  and g < 15  and b < 15)
        )
        return (colored / total) > 0.1
    except Exception:
        return True


def extract_images_from_pdf(pdf_path: str) -> list:
    print(f"\n[Extracting images from: {os.path.basename(pdf_path)}]")
    saved_paths = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ERROR] Cannot open PDF: {e}")
        return []

    stem = Path(pdf_path).stem.replace(" ", "_")

    for page_num in range(len(doc)):
        page       = doc[page_num]
        image_list = page.get_images(full=True)

        if not image_list:
            pix       = page.get_pixmap(dpi=150)
            page_path = str(INPUT_DIR / f"{stem}_page{page_num+1}_full.png")
            pix.save(page_path)
            img = Image.open(page_path)
            if _looks_like_graph(img):
                saved_paths.append(page_path)
                print(f"     [OK] Page {page_num+1}: vector graph detected")
            else:
                os.remove(page_path)
            continue

        for idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                img_ext    = base_image["ext"]
                width      = base_image["width"]
                height     = base_image["height"]

                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    continue

                filename  = f"{stem}_page{page_num+1}_img{idx+1}.{img_ext}"
                save_path = str(INPUT_DIR / filename)
                with open(save_path, "wb") as f:
                    f.write(img_bytes)

                print(f"     [OK] Page {page_num+1} img{idx+1}: {width}x{height} -> {filename}")
                saved_paths.append(save_path)

            except Exception as e:
                print(f"     [WARN] Page {page_num+1} img{idx+1} error: {e}")

    doc.close()
    print(f"[OK] Extracted {len(saved_paths)} image(s).")
    return saved_paths


# ==============================================================================
# STEP 2 -- ENCODE & COMPRESS IMAGE
# ==============================================================================

def compress_image_to_base64(image_path: str, max_size=(512, 512), quality=70) -> tuple:
    """
    Resizes and compresses image to reduce payload size for Vision API.
    Returns (base64_string, mime_type)
    """
    ext = Path(image_path).suffix.lower()
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
        
    # Resize keeping aspect ratio
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Save to buffer
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    return b64, "image/jpeg"


# ==============================================================================
# STEP 3 -- ANALYZE GRAPH WITH VISION API (HF InferenceClient)
# ==============================================================================

def safe_vision_call(client, messages, max_retries=MAX_RETRIES):
    """Wrapper for vision API calls with exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"     [🚀] Calling Vision API (attempt {attempt}/{max_retries})")
            response = client.chat_completion(
                messages=messages,
                max_tokens=MAX_TOKENS,
            )
            raw = response.choices[0].message.content.strip()
            print(f"     [✅] Analysis received ({len(raw)} chars)")
            return raw
        except Exception as e:
            import time
            wait = 2 ** attempt
            print(f"     [⚠️] Error: {e} → retrying in {wait}s...")
            if attempt == max_retries:
                print(f"     [❌] Error: All {max_retries} attempts failed.")
                raise e
            else:
                time.sleep(wait)
    return None


def analyze_graph(image_path: str, hf_token: str, extra_prompt: str = "") -> dict:
    """
    Uses HuggingFace InferenceClient which automatically selects
    an available provider -- no manual provider setup needed.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Use compressed image to reduce payload size
    b64_data, mime_type = compress_image_to_base64(image_path)
    image_url = f"data:{mime_type};base64,{b64_data}"

    user_text = (
        "Analyze this graph or figure in detail. Provide:\n"
        "1. DESCRIPTION: What type of graph is this and what does it show?\n"
        "2. AXES: What are the x and y axes, their labels and units?\n"
        "3. TREND: What is the overall trend or pattern?\n"
        "4. KEY VALUES: Any notable points, intersections, or values?\n"
        "5. CONCLUSION: What is the physical or scientific meaning?\n"
    )
    if extra_prompt:
        user_text += f"\nAdditional focus: {extra_prompt}\n"

    client = InferenceClient(model=VL_MODEL_ID, token=hf_token)

    messages = [
        {
            "role": "system",
            "content": "You are an expert physics and mathematics tutor. Analyze graphs and figures clearly. English only.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    try:
        raw = safe_vision_call(client, messages)
    except Exception as e:
        raw = f"Error: Vision API failed after {MAX_RETRIES} attempts. {e}"

    def _extract(label: str) -> str:
        pattern = rf"{label}[:\.]?\s*(.*?)(?=\n\d[.\)]|$)"
        match   = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    return {
        "image_path"  : image_path,
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description" : _extract(r"1[.\)]?\s*DESCRIPTION"),
        "axes"        : _extract(r"2[.\)]?\s*AXES"),
        "trend"       : _extract(r"3[.\)]?\s*TREND"),
        "key_values"  : _extract(r"4[.\)]?\s*KEY VALUES"),
        "conclusion"  : _extract(r"5[.\)]?\s*CONCLUSION"),
        "raw_response": raw,
    }


# ==============================================================================
# STEP 4 -- SAVE RESULTS
# ==============================================================================

def save_analysis(result: dict) -> tuple:
    stem      = Path(result["image_path"]).stem
    json_path = str(ANALYZED_DIR / f"{stem}_analysis.json")
    txt_path  = str(ANALYZED_DIR / f"{stem}_analysis.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines = [
        "GRAPH ANALYSIS REPORT",
        "=" * 50,
        f"File      : {result['image_path']}",
        f"Timestamp : {result['timestamp']}",
        "-" * 50,
        f"1. DESCRIPTION\n{result['description']}",
        "-" * 50,
        f"2. AXES\n{result['axes']}",
        "-" * 50,
        f"3. TREND\n{result['trend']}",
        "-" * 50,
        f"4. KEY VALUES\n{result['key_values']}",
        "-" * 50,
        f"5. CONCLUSION\n{result['conclusion']}",
        "=" * 50,
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[SAVED] {os.path.basename(json_path)}")
    print(f"[SAVED] {os.path.basename(txt_path)}")
    return json_path, txt_path


# ==============================================================================
# MAIN ENTRY POINT -- called from main.py
# ==============================================================================

def extract_and_analyze_graphs(
    file_paths: list,
    hf_token: str,
    extra_prompt: str = "",
) -> list:
    """
    Called from main.py after OCR stage.
    Uses HF_TOKEN only -- no Anthropic key required.
    """
    print(f"\n{'='*60}")
    print(f"GRAPH ANALYSIS PIPELINE -- {len(file_paths)} file(s)")
    print(f"{'='*60}")

    all_image_paths = []

    for path in file_paths:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            extracted = extract_images_from_pdf(path)
            all_image_paths.extend(extracted)
        elif ext in SUPPORTED_EXT:
            all_image_paths.append(path)
        else:
            print(f"[SKIP] {path}")

    if not all_image_paths:
        print("[WARN] No images found to analyze.")
        return []

    print(f"\n[INFO] Images to analyze: {len(all_image_paths)}")

    results = []
    for img_path in all_image_paths:
        try:
            result = analyze_graph(img_path, hf_token, extra_prompt=extra_prompt)
            save_analysis(result)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed on {img_path}: {e}")

    print(f"\n[OK] Graph analysis complete -- {len(results)} analyzed.")
    return results


# ==============================================================================
# STANDALONE MODE  (python image.py)
# ==============================================================================

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        token = input("HF_TOKEN not found. Paste your token: ").strip()

    _root = tk.Tk()
    _root.withdraw()
    _root.attributes("-topmost", True)
    paths = list(filedialog.askopenfilenames(
        title="Select PDF or image files",
        filetypes=[
            ("Study materials", "*.pdf *.jpg *.jpeg *.png *.bmp *.webp"),
            ("PDF",    "*.pdf"),
            ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All",    "*.*"),
        ],
    ))
    _root.destroy()

    if not paths:
        print("[WARN] No files selected.")
        sys.exit(0)

    results = extract_and_analyze_graphs(paths, token)

    print(f"\nResults saved in: {ANALYZED_DIR}")
    for r in results:
        print(f"   OK: {Path(r['image_path']).name}")
        if r.get("conclusion"):
            print(f"      -> {r['conclusion'][:120]}...")