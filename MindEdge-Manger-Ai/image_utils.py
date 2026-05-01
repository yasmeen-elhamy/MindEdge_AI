import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
from autocrop import Cropper
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def select_files():
    print("Opening file selection dialog for multiple files...")
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select Image or PDF files",
        filetypes=[("Image and PDF files", "*.jpg *.jpeg *.png *.pdf"), ("All files", "*.*")]
    )
    if file_paths:
        print(f"Selected files: {file_paths}")
    else:
        print("No files selected.")
    return list(file_paths)

def auto_crop_image(image_path):
    print(f"Starting auto-cropping for {image_path}...")
    try:
        cropper = Cropper()
        cropped_image = cropper.crop(image_path)
        if cropped_image is not None:
            if isinstance(cropped_image, np.ndarray):
                cropped_image = Image.fromarray(cropped_image)
            print("Image auto-cropped successfully.")
            return cropped_image
        else:
            print("Auto-cropping did not yield a result. Proceeding with original image.")
            return Image.open(image_path)
    except Exception as e:
        print(f"Error in auto-cropping: {e}. Proceeding with original image.")
        try:
            return Image.open(image_path)
        except Exception as e2:
            print(f"Error opening original image: {e2}")
            return None

def run_ocr(path):
    print(f"Running OCR on {path}...")
    try:
        if not os.path.exists(path):
            print(f"Error: File {path} does not exist.")
            return None
        if path.lower().endswith(('jpg', 'jpeg', 'png')):
            doc = DocumentFile.from_images(path)
        else:
            doc = DocumentFile.from_pdf(path)
        ocr_result = ocr_predictor(pretrained=True)(doc)
        print("OCR completed.")
        return ocr_result
    except Exception as e:
        print(f"Error in OCR: {e}")
        return None

def extract_text(result):
    print("Extracting text from OCR result...")
    try:
        if result is None:
            print("No OCR result to extract text from.")
            return ""
        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    lines.append(" ".join(w.value for w in line.words))
        extracted_text = "\n".join(lines)
        if not extracted_text:
            print("Warning: No text extracted from OCR result.")
        else:
            print(f"Extracted text: {extracted_text[:100]}...")
        return extracted_text
    except Exception as e:
        print(f"Error in text extraction: {e}")
        return ""
