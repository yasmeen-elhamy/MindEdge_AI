from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from image_utils import run_ocr, extract_text
from text_utils import segment_text
from index_utils import build_or_load_index, retrieve_passages
from mistral_api import ask_mistral
from chat_utils import save_chat_log
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatQuery(BaseModel):
    prompt: str

@app.post("/api/ai/upload-file")
async def process_file(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ocr_result = run_ocr(temp_path)
    original_text = extract_text(ocr_result)

    segments = segment_text(original_text)

    os.remove(temp_path)

    return {
        "original": original_text,
        "augmented": segments
    }


@app.post("/api/ai")
async def ask_question(chat: ChatQuery):
    collection = build_or_load_index()
    passages = retrieve_passages(chat.prompt, collection)
    answer = ask_mistral(chat.prompt, passages)

    save_chat_log(chat.prompt, answer)

    return {"answer": answer}

