import os

def save_processed_outputs(summary):
    print("Saving processed outputs...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    if not summary:
        print("Warning: Summary is empty, writing empty summary.md")
        summary = "No summary generated."
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as sum_file:
        sum_file.write(f"# Summary of \n\n{summary}\n")
    print("[✅] Outputs saved successfully in folders.")

def save_chat_log(query, answer, log_directory="chat_logs", log_filename="chat_history.md"):
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_directory)
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_filename)
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"### User:\n{query}\n\n### Assistant:\n{answer}\n\n---\n")
        print("[✅] Chat log saved successfully.")
    except PermissionError:
        print(f"Error: Insufficient file permissions to write to {file_path}.")
    except IOError as e:
        print(f"An I/O error occurred while saving chat log: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving chat log: {e}")
