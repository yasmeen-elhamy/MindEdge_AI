"""
llm.py — LLM API Module (Qwen via HF Router)
=============================================
All calls to the Qwen language model.
"""

import os
import time
from typing import Optional
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError

_QW_BASE_URL = "https://api.openai.com/v1"
_QW_MODEL_ID = "gpt-4o-mini"
_QW_MAX_TOKENS    = 2048
_QW_TEMPERATURE   = 0.7
_QW_TOP_P         = 0.9
_QW_TIMEOUT       = 90
_QW_MAX_RETRIES   = 4
_QW_BACKOFF_BASE  = 2.0
_QW_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
_QW_CLIENT: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _QW_CLIENT
    if _QW_CLIENT is None:
        token = os.environ.get("OPENAI_API_KEY", "").strip()

        if not token:
            raise EnvironmentError(
                "\n[llm] ❌  OPENAI_API_KEY is not set!\n"
"  Fix: create a .env file with: OPENAI_API_KEY=sk-your_key_here"
            )
        _QW_CLIENT = OpenAI(base_url=_QW_BASE_URL, api_key=token)
    return _QW_CLIENT


def generate_response(
    prompt: str,
    max_tokens: int = _QW_MAX_TOKENS,
    system_prompt: str = _QW_SYSTEM_PROMPT,
) -> str:
    """Sends a prompt to Qwen and returns the response string."""
    client = _get_client()
    print(f"[🚀] Calling HF Router → {_QW_MODEL_ID}")

    for attempt in range(1, _QW_MAX_RETRIES + 1):
        backoff = _QW_BACKOFF_BASE ** (attempt - 1)
        try:
            completion = client.chat.completions.create(
                model=_QW_MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=_QW_TEMPERATURE,
                top_p=_QW_TOP_P,
                timeout=_QW_TIMEOUT,
            )
            answer = completion.choices[0].message.content.strip()
            usage  = getattr(completion, "usage", None)
            if usage:
                print(f"[✅] Done — {usage.total_tokens} tokens used.")
            else:
                print(f"[✅] Response received ({len(answer)} chars).")
            return answer

        except APIStatusError as exc:
            code = exc.status_code
            if code == 503:
                try:    wait = max(float(exc.body.get("estimated_time", backoff)), backoff)
                except: wait = backoff
                print(f"[⏳] 503 – Model loading, waiting {wait:.0f}s… ({attempt}/{_QW_MAX_RETRIES})")
            elif code == 429:
                print(f"[🚦] 429 – Rate limited, backing off {backoff:.0f}s… ({attempt}/{_QW_MAX_RETRIES})")
                wait = backoff
            else:
                msg = f"HTTP {code}: {str(exc)[:300]}"
                print(f"[❌] {msg}")
                return f"Error: {msg}"
            if attempt < _QW_MAX_RETRIES:
                time.sleep(wait)
                continue
            return f"Error: HTTP {code} after {_QW_MAX_RETRIES} attempts."

        except APITimeoutError:
            print(f"[⏱️]  Timeout ({attempt}/{_QW_MAX_RETRIES})")
            if attempt < _QW_MAX_RETRIES:
                time.sleep(backoff)
                continue
            return f"Error: Timeout after {_QW_MAX_RETRIES} attempts."

        except APIConnectionError as exc:
            msg = f"Connection error: {exc}"
            print(f"[❌] {msg}")
            if attempt < _QW_MAX_RETRIES:
                time.sleep(backoff)
                continue
            return f"Error: {msg}"

    return f"Error: All {_QW_MAX_RETRIES} attempts failed."


def correct_text(text: str) -> str:
    """Corrects and enriches OCR-extracted text."""
    print("[✏️]  Correcting text…")
    prompt = (
        "Correct the spelling and grammar of this educational content. "
        "Expand on ideas and add more detail to make the text clearer and richer. "
        "Return the output in well-structured Markdown format:\n\n"
        f"{text}"
    )
    return generate_response(prompt)


def summarize_text(text: str) -> str:
    """Generates a comprehensive summary of the given text."""
    print("[📌] Summarizing text…")
    prompt = (
        "Write a detailed and comprehensive summary of the following educational text. "
        "Include all important points and explain concepts where possible. "
        "Return the output in well-structured Markdown format:\n\n"
        f"{text}"
    )
    return generate_response(prompt)


def test_connection() -> bool:
    """Tests the connection to HF Router."""
    print("[🔬] Testing connection to HF Inference Router…")
    result = generate_response("Reply with the single word: OK", max_tokens=10)
    ok = not result.startswith("Error:")
    print(f"[{'✅' if ok else '❌'}] Test {'PASSED' if ok else 'FAILED'} → {result[:80]}")
    return ok
