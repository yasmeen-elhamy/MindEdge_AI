ROM python:3.10-slim

# System dependencies فقط اللي محتاجينه
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# نثبت CPU-only versions عشان تاخد مساحة أقل
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p output/rules output/definitions output/graphs \
    output/graph_analysis rag_index chat_logs temp models

EXPOSE 8000

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]

