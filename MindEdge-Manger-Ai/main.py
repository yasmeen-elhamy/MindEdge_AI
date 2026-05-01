from datetime import datetime
import os
from image_utils import select_files, auto_crop_image, run_ocr, extract_text
from text_utils import correct_text, summarize_text, segment_text,segment_text
from index_utils import build_or_load_index
from chat_utils import save_chat_log
from mistral_api import ask_mistral
from index_utils import retrieve_passages

def main():
    print("📌 Starting main application flow...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"chat_history_{timestamp}.md"

    collection = build_or_load_index()
    paths = select_files()

    if not paths:
        print("No file(s) selected or process was canceled. Exiting.")
        return

    for path in paths:
        print(f"Processing file: {path}")
        if path.lower().endswith(('jpg', 'png', 'jpeg')):
            cr = auto_crop_image(path)
            if cr:
                if cr.mode == 'RGBA':
                    cr = cr.convert('RGB')
                cr.save('cropped_temp.jpg')
                path = 'cropped_temp.jpg'
                print("Image was auto-cropped and saved as cropped_temp.jpg.")
            else:
                print(f"Warning: Failed to process image {path}. Skipping.")
                continue

        ocr_res = run_ocr(path)
        raw_text = extract_text(ocr_res)
        if not raw_text:
            print(f"Warning: No text extracted from {path}. Skipping further processing.")
            continue

        corrected = correct_text(raw_text)
        if corrected.startswith("Error:"):
            print(f"Warning: Text correction failed: {corrected}")
            corrected = raw_text  

        os.makedirs('output', exist_ok=True)
        fname = os.path.basename(path) + '.md'
        with open(os.path.join('output', fname), 'w', encoding='utf-8') as f:
            f.write(corrected if corrected else "No corrected text.")
        print(f"Saved corrected text to {fname}")

        collection = build_or_load_index()
        print("[✅] Content ingested.")

        summary = summarize_text(corrected)
        if summary.startswith("Error:"):
            print(f"Warning: Summarization failed: {summary}")
            summary = "No summary generated."
      

    segmented = segment_text(corrected)
    print("Segmented text:", segmented)


    with open(os.path.join(output_dir, 'raw_text.md'), 'w', encoding='utf-8') as f:
        f.write(raw_text if raw_text else "No text extracted.")
    with open(os.path.join(output_dir, 'corrected_text.md'), 'w', encoding='utf-8') as f:
        f.write(corrected if corrected else "No corrected text.")
    with open(os.path.join(output_dir, 'summarize_text.md'), 'w', encoding='utf-8') as f:
        f.write(summary if summary else "No summary generated.")







    chat_history = []
    while True:
        q = input("Ask Any question (or 'exit'): ")
        if q.lower() == 'exit':
            print("Exiting interactive chat.")
            break
        passages = retrieve_passages(q, collection)
        answer = ask_mistral(q, passages)
        print("Assistant:", answer)
        chat_history.append(f"### User:\n{q}\n\n### Assistant:\n{answer}\n")
        save_chat_log(q, answer, log_filename=log_filename)

    with open(os.path.join(output_dir, 'chat_history.md'), 'w', encoding='utf-8') as f:
        for entry in chat_history:
            f.write(entry + "\n---\n")


    print("📌 Main application flow finished.")

if __name__ == '__main__':
    main()
