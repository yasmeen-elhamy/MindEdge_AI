FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Create output directories
RUN mkdir -p output/rules output/definitions output/graphs \
    output/graph_analysis rag_index chat_logs temp models

EXPOSE 8000

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]
