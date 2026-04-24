from fastapi import FastAPI, UploadFile, File
import os
from main import process_pdf  # افترضنا إن دي الدالة اللي بتلم كل شغلك
from image import extract_and_analyze_graphs

app = FastAPI()

@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    # 1. حفظ الملف المؤقت اللي جاي من الباك-إند
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # 2. تشغيل الـ OCR (النصوص والقوانين)
        text_results = process_pdf(file_path) 
        
        # 3. تشغيل تحليل الجرافات (شغلنا في image.py)
        # ملحوظة: تأكدي إنك بتمرري الـ Token هنا
        token = os.getenv("HF_TOKEN")
        graph_results = extract_and_analyze_graphs([file_path], token)
        
        # 4. تجميع كل النتائج في JSON واحد
        final_response = {
            "status": "success",
            "document_name": file.filename,
            "text_analysis": text_results,
            "visual_analysis": graph_results
        }
        return final_response

    finally:
        # تنظيف الجهاز وحذف الملف المؤقت
        if os.path.exists(file_path):
            os.remove(file_path)