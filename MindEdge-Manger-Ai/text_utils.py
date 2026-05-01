import nltk
import re
from mistral_api import generate_mistral_response

nltk.download('punkt', quiet=True)

def correct_text(text):
    print("[✏️] Correcting text...")
    prompt = (
        f"Correct the spelling and grammar of this educational content. "
        f"Also, expand on ideas and add more detail if possible to make the text clearer and richer. "
        f"Return the output in well-structured Markdown format with headings, bullet points, or numbered lists where appropriate:\n\n{text}"
    )
    return generate_mistral_response(prompt)

def summarize_text(text):
    print("[📌] Summarizing text...")
    prompt = (
        f"Please write a detailed and comprehensive summary of the following educational text. "
        f"Include all important points, explain concepts where possible, and keep it longer and informative. "
        f"Return the output in well-structured Markdown format with headings, bullet points, or numbered lists where appropriate:\n\n{text}"
    )
    return generate_mistral_response(prompt)


def segment_text(text):
    print("Segmenting text into sections...")
    if not text:
        print("Warning: No text provided for segmentation.")
        return {"headings": [], "bullets": [], "other_text": []}
    sentences = nltk.tokenize.sent_tokenize(text)
    headings = [s for s in sentences if s.isupper()]
    bullets = [s for s in sentences if re.match(r'^[-*•]', s.strip())]
    other = [s for s in sentences if s not in headings and s not in bullets]
    print("Text segmentation completed.")
    return {"headings": headings, "bullets": bullets, "other_text": other}
