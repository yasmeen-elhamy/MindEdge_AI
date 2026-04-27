"""
study_plan.py  ─  Study Plan Generator module
==============================================
Extracted from mainstydyplan.py to work as a module imported by main.py.

main.py calls:
    from study_plan import generate_study_plan
    plan = generate_study_plan(topics, days, hours_per_day)

This module also exposes the richer signature used internally:
    generate_study_plan(topics, days, hours_per_day,
                        subject=None, level="Intermediate", collection=None)
"""

from __future__ import annotations

import math
import logging
import datetime
import re
import time
from pathlib import Path
from typing import Optional

# ── OpenAI-compatible client (HuggingFace Router)
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("StudyPlan")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (shared with main.py — values kept identical)
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN          = "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"
_QW_MODEL_ID      = "Qwen/Qwen2.5-7B-Instruct"
_QW_BASE_URL      = "https://router.huggingface.co/v1"
_QW_MAX_TOKENS    = 800
_QW_TEMPERATURE   = 0.5
_QW_TOP_P         = 0.9
_QW_TIMEOUT       = 60
_QW_MAX_RETRIES   = 4
_QW_BACKOFF_BASE  = 2.0
_QW_SYSTEM_PROMPT = "You are Qwen, a helpful AI assistant."
_QW_CLIENT        = None

OUTPUT_FILE = "study_plan.md"

# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────────────────────────
def _get_client() -> OpenAI:
    global _QW_CLIENT
    if _QW_CLIENT is None:
        _QW_CLIENT = OpenAI(base_url=_QW_BASE_URL, api_key=HF_TOKEN)
    return _QW_CLIENT


def llm_chat(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = _QW_MAX_TOKENS,
    temperature: float = _QW_TEMPERATURE,
) -> str:
    """Call the HuggingFace-routed LLM with retry logic."""
    client = _get_client()
    print(f"🚀 Calling HF Router → {_QW_MODEL_ID}")

    for attempt in range(1, _QW_MAX_RETRIES + 1):
        backoff = _QW_BACKOFF_BASE ** (attempt - 1)
        try:
            response = client.chat.completions.create(
                model=_QW_MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=_QW_TOP_P,
                timeout=_QW_TIMEOUT,
            )
            answer = response.choices[0].message.content.strip()
            usage  = getattr(response, "usage", None)
            if usage:
                print(f"✅ Tokens → prompt:{usage.prompt_tokens} | completion:{usage.completion_tokens}")
            else:
                print(f"✅ Response received ({len(answer)} chars)")
            return answer

        except APIStatusError as e:
            code = e.status_code
            if code in (503, 429):
                print(f"⏳ HTTP {code}, retry in {backoff}s ({attempt}/{_QW_MAX_RETRIES})")
                if attempt < _QW_MAX_RETRIES:
                    time.sleep(backoff)
                    continue
                return f"Error: HTTP {code} after retries"
            return f"Error: HTTP {code}: {str(e)[:200]}"

        except APITimeoutError:
            print(f"⏱ Timeout ({attempt}/{_QW_MAX_RETRIES})")
            if attempt < _QW_MAX_RETRIES:
                time.sleep(backoff)
                continue
            return "Error: Timeout"

        except APIConnectionError as e:
            print(f"❌ Connection error: {e}")
            if attempt < _QW_MAX_RETRIES:
                time.sleep(backoff)
                continue
            return "Error: Connection failed"

        except Exception as e:
            return f"Error: {str(e)}"

    return "Error: All retries failed"

# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK PLAN  (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────
def _fallback_plan(
    subject: str,
    duration_days: int,
    hours: int,
    level: str,
    topics: list[str],
) -> str:
    """Minimal local fallback when the LLM is unavailable."""
    lines = [
        f"# Study Plan: {subject}",
        f"**Level**: {level}  |  **Duration**: {duration_days} days  |  **{hours}h/day**",
        "",
        "> ⚠ Generated locally (LLM API unavailable).",
        "",
    ]
    revision_interval = 7
    topic_index = 0
    for day_index in range(1, duration_days + 1):
        if day_index % revision_interval == 0:
            lines += [
                f"## Day {day_index} — REVISION",
                "- **Topic**: Review previous topics",
                "- **Tasks**: Revisit notes, solve practice problems, summarise key points",
                f"- **Estimated Time**: {hours}h",
                "",
            ]
        else:
            topic = topics[topic_index % len(topics)]
            topic_index += 1
            lines += [
                f"## Day {day_index}",
                f"- **Topic**: {topic}",
                "- **Tasks**: Read and take notes, practice exercises",
                f"- **Estimated Time**: {hours}h",
                "",
            ]
    lines.append("---\n*Study hard and stay consistent!*")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def _build_plan_prompt(
    subject: str,
    duration_days: int,
    daily_hours: int,
    level: str,
    topics: list[str],
    rag_context: str,
) -> tuple[str, str]:
    total_days    = duration_days
    study_days    = math.ceil(total_days * 0.85)
    revision_days = total_days - study_days
    topics_str    = "\n".join(f"  - {t}" for t in topics)

    level_guidance = {
        "Beginner":     "Start from absolute basics. Include many practical examples.",
        "Intermediate": "Assume basic familiarity. Balance theory and practice equally.",
        "Advanced":     "Move quickly through fundamentals. Focus on depth and edge cases.",
    }.get(level, "")

    system = (
        "You are an expert curriculum designer and study coach. "
        "You create detailed, realistic, personalized study plans. "
        "Every plan you produce is:\n"
        "  • Divided into individual DAYS (Day 1, Day 2, …)\n"
        "  • Progressive: easy → medium → hard\n"
        "  • Realistic: respects the daily hour constraint\n"
        "  • Complete: covers all provided topics\n"
        "  • Includes periodic REVISION days\n"
        "  • Formatted in clean Markdown"
    )
    user = (
        f"Create a COMPLETE, DAY-BY-DAY study plan with the following parameters:\n\n"
        f"Subject       : {subject}\n"
        f"Total duration: {total_days} days\n"
        f"Study days    : {study_days}\n"
        f"Revision days : {revision_days}\n"
        f"Daily hours   : {daily_hours} hour(s) per day\n"
        f"Level         : {level}\n\n"
        f"Level guidance: {level_guidance}\n\n"
        f"Topics to cover:\n{topics_str}\n\n"
        f"Relevant study material context:\n{rag_context}\n\n"
        f"STRICT OUTPUT FORMAT (Markdown, repeat for every day):\n\n"
        f"## Day 1\n"
        f"- **Topic**: <topic name>\n"
        f"- **Subtopics**: <comma-separated list>\n"
        f"- **Tasks**: <numbered list of concrete learning tasks>\n"
        f"- **Resources**: <what to read/watch/practice>\n"
        f"- **Estimated Time**: {daily_hours}h\n"
        f"- **Notes**: <any special tips for this day>\n\n"
        f"(Continue for ALL {total_days} days. "
        f"Mark revision days clearly as '## Day N — REVISION'.)\n\n"
        f"End with a short **Summary** section."
    )
    return system, user

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def generate_study_plan(
    topics       : list[str],
    days         : int,
    hours_per_day: int,
    subject      : Optional[str] = None,
    level        : str = "Intermediate",
    collection   = None,
) -> dict:
    """
    Generate a study plan and return a dict  {day_label: details_string}.

    Compatible with main.py's call:
        plan = generate_study_plan(topics, days, hours_per_day)
        for day, details in plan.items():
            print(day, details)

    Parameters
    ----------
    topics        : list of topic strings
    days          : total number of days
    hours_per_day : daily study hours
    subject       : optional subject name (defaults to first topic)
    level         : "Beginner" | "Intermediate" | "Advanced"
    collection    : optional ChromaDB collection for RAG grounding
    """
    if subject is None:
        subject = topics[0] if topics else "Study Material"

    log.info("Generating study plan (%d days, %dh/day, %s) ...", days, hours_per_day, level)

    # RAG context
    if collection is not None:
        try:
            from study_plan import rag_query          # works if RAG helpers present
            rag_ctx = rag_query(collection, f"{subject} study plan {level} topics schedule", top_k=6)
        except Exception:
            rag_ctx = "\n".join(topics[:30])
    else:
        rag_ctx = "\n".join(topics[:30])

    system, user = _build_plan_prompt(subject, days, hours_per_day, level, topics, rag_ctx)

    max_tok = min(3000, 120 * days)
    try:
        plan_text = llm_chat(system, user, max_tokens=max_tok, temperature=0.5)
        if plan_text.startswith("Error:"):
            raise RuntimeError(plan_text)
    except Exception as exc:
        log.warning("LLM unavailable (%s) — using fallback plan.", exc)
        plan_text = _fallback_plan(subject, days, hours_per_day, level, topics)

    # ── Parse Markdown into {day_label: details} dict ──────────────────────
    result: dict[str, str] = {}
    current_day = None
    current_lines: list[str] = []

    for line in plan_text.splitlines():
        if line.startswith("## Day") or line.startswith("## **Day"):
            # save previous day
            if current_day is not None:
                result[current_day] = "\n".join(current_lines).strip()
            current_day   = line.lstrip("# ").strip()
            current_lines = []
        elif current_day is not None:
            current_lines.append(line)

    # save last day
    if current_day is not None:
        result[current_day] = "\n".join(current_lines).strip()

    # If parsing yielded nothing, store the raw text under one key
    if not result:
        result["Study Plan"] = plan_text

    return result