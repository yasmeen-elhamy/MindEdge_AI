# MindEdge 🧠📘
### Intelligent Study Scanner & Adaptive AI Tutor

**MindEdge** is an AI-powered educational platform that transforms handwritten or printed study materials into **structured, adaptive, and multimodal learning experiences.**

---

## 🚀 Overview
MindEdge is **not** a traditional Q&A system. Its core intelligence lies in the deep coupling between **Retrieval-Augmented Generation (RAG)** and an **Intelligent Educational Recommendation Engine**, which together decide *what* knowledge to retrieve and *how* it should be taught.

---

## 🧠 Core Intelligence (The Heart of MindEdge ❤️)

| Feature | Description |
| :--- | :--- |
| **RAG Layer** | Ensures answers are grounded in the student’s actual study materials. |
| **Recommendation Engine** | The pedagogical brain that selects the optimal teaching strategy. |
| **Intent Awareness** | Understands student intent (summarize, explain, solve) to adapt difficulty. |

---

## 🎯 What Makes MindEdge Different?
* **Beyond Questions**: Understands the *intent* behind the query.
* **Adaptive Explanations**: Changes tone and depth based on content difficulty.
* **Multimodal Support**: Treats **Audio-Based Learning** as a first-class citizen for accessibility.

---

## 🛠️ Project Structure (Professional Overview)

```text
EduScan/
├── main.py              # Central orchestration & full pipeline coordination
├── config.py            # System paths, tokens, and global settings
├── llm.py               # LLM interaction layer (Qwen/GPT)
├── ocr.py               # OCR pipeline & academic post-processing
├── rag.py               # Knowledge indexing & semantic retrieval
├── chat.py              # Interactive chat loop & session management
├── image.py             # Vision model & image preprocessing
🧩 Module Responsibilities
📡 Backend & API
/upload: Process educational documents (PDF/Images).

/query: Intent-aware adaptive learning interaction.

/health: System monitoring.

🔊 Audio-Based Learning
AI-generated spoken explanations for auditory learners.

Fully aligned with pedagogical decisions.

📊 Platform Positioning
MindEdge is a bridge between AI Reasoning and Learning Science. It is a scalable, research-friendly platform designed for production-ready educational assistance.

🛠️ Installation & Setup
Clone the repo: git clone https://github.com/yasmeen-elhamy/MindEdge_AI.git

Install Deps: pip install -r requirements.txt

Run Server: uvicorn api:app --reload


└── api.py               # FastAPI-based backend services
