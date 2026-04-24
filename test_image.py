#!/usr/bin/env python3
"""
Test script for EduScan image analysis
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from image import extract_images_from_pdf

def main():
    # Test with the PDF
    pdf_path = "Electric conductivity and ohm's law 2-26.pdf"
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return

    print("Testing image extraction...")
    image_paths = extract_images_from_pdf(pdf_path)
    print(f"Extraction complete: {len(image_paths)} images found")

    for path in image_paths:
        print(f"  - {path}")

if __name__ == "__main__":
    main()