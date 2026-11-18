"""Prompt templates shared by all vision providers."""

from __future__ import annotations

FRAME_SYSTEM_PROMPT = (
    "You are an expert UI analyst. Look at each provided frame from a "
    "screen-recording and describe in 1-3 sentences what is shown: focus on "
    "interactive UI elements, on-screen text, and the user's likely action."
)

FRAME_USER_INSTRUCTION = (
    "Describe the following frame. Return a JSON object with fields:\n"
    "  description: a short paragraph describing the frame.\n"
    "  labels: a list of short tags identifying UI elements visible.\n"
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are summarising a recorded product walkthrough. You will be given "
    "per-frame descriptions and (optionally) a transcript. Produce a fluent "
    "narrative walkthrough plus a bullet list of the UI elements that "
    "appeared, in order. Output JSON with keys 'walkthrough' and 'ui_notes'."
)


def build_synthesis_prompt(
    *,
    frame_descriptions: list[str],
    transcript_text: str | None,
) -> str:
    """Compose the synthesis user prompt from frame descriptions + transcript."""
    parts: list[str] = ["Frame descriptions (in order):"]
    for index, desc in enumerate(frame_descriptions, start=1):
        parts.append(f"{index}. {desc}")
    if transcript_text:
        parts.append("")
        parts.append("Audio transcript:")
        parts.append(transcript_text)
    parts.append("")
    parts.append("Return JSON with keys 'walkthrough' and 'ui_notes'.")
    return "\n".join(parts)
