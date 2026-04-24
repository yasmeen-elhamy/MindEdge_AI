"""
chat.py — Interactive Chat Loop
=================================
Persistent memory chat session with RAG retrieval.
"""

import os
from config import CHAT_LOG_DIR, OUTPUT_DIR, export_to_folder
from llm import generate_response
from rag import retrieve_passages, save_chat_log, load_previous_summary, summarize_chat_history


def run_chat_loop(collection) -> list:
    """Main interactive chat loop with persistent memory and RAG retrieval."""
    log_filename  = "persistent_memory.md"
    chat_history  = []
    all_exchanges = []

    current_summary = load_previous_summary(log_filename)
    if current_summary:
        print(f"[✅] Long-term memory restored: {current_summary[:120]}…")

    print("\n" + "═" * 60)
    print("🤖  STAGE 4 — PERSISTENT MEMORY CHAT")
    print("     Type 'exit' or 'quit' to end the session.")
    print("═" * 60)

    while True:
        try:
            query = input("\n❓ Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[ℹ️]  Session interrupted.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break

        # ── RAG retrieval ─────────────────────────────────────────────────
        passages = retrieve_passages(query, collection, top_k=5)
        context  = "\n\n".join(passages) if passages else "No context found."

        # ── Reload persistent memory ──────────────────────────────────────
        memory_path = os.path.join(CHAT_LOG_DIR, log_filename)
        if os.path.exists(memory_path):
            with open(memory_path, "r", encoding="utf-8") as f:
                current_summary = f.read()[-3000:]

        # ── Build prompt ──────────────────────────────────────────────────
        final_prompt = (
            "You are a helpful assistant with long-term memory.\n"
            f"[MEMORY FROM PAST SESSIONS]:\n{current_summary}\n\n"
            f"[CURRENT PDF CONTEXT]:\n{context}\n\n"
            f"[USER QUESTION]: {query}\n\n"
            "INSTRUCTION: Answer from the PDF context first. "
            "If not found, use memory. "
            "Format laws/formulas as [RULE: Name] Formula. English only."
        )

        answer = generate_response(final_prompt)
        print("\n" + "─" * 60 + f"\n💬 Answer:\n{answer}\n" + "─" * 60)

        # ── Auto-classify answer ──────────────────────────────────────────
        if "[DEFINITION:" in answer:
            try:
                parts = answer.split("[DEFINITION:")[1].split("]")
                export_to_folder("definitions", parts[0].strip(), parts[1].strip())
            except Exception:
                pass

        if "[RULE:" in answer:
            try:
                parts = answer.split("[RULE:")[1].split("]")
                export_to_folder("rules", parts[0].strip(), parts[1].strip())
            except Exception:
                pass

        # ── Persist ───────────────────────────────────────────────────────
        save_chat_log(query, answer, log_filename=log_filename)
        chat_history.append({"question": query, "answer": answer})
        all_exchanges.append({"question": query, "answer": answer})

        # ── Summarise every 5 turns ───────────────────────────────────────
        if len(chat_history) >= 5:
            current_summary = summarize_chat_history(chat_history)
            chat_history = []

    return all_exchanges
