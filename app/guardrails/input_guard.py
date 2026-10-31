"""Screen incoming questions before any retrieval or LLM call.

Rejects empty, garbage, and obvious prompt-injection input with a typed error
the API maps to a clean 400. This is a cheap first line of defence, not a
complete injection filter: the answer prompt also tells the model to treat
book text as data, and the output guard checks grounding.
"""

import re
import unicodedata

from app.config import Settings

# Phrasings that only make sense as attempts to override the system prompt.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|prompts?|rules|directions)\b",
        r"\b(reveal|show|print|repeat|output|leak)\b.{0,30}\b(system|hidden|initial|original)\s+"
        r"(prompt|instructions?|message)",
        r"\byou are now\b|\bfrom now on,? you (are|will|must)\b",
        r"\b(jailbreak|jailbroken|developer mode|DAN mode)\b",
        r"</?\s*(system|assistant|instructions?)\s*>",
        r"^\s*(system|assistant)\s*:",
    )
]


class InputRejectedError(ValueError):
    """The question was refused before retrieval; *reason* is a stable code for clients."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _normalize(question: str) -> str:
    """Drop control/format characters and collapse runs of whitespace."""
    kept = (
        ch
        for ch in question
        if ch in "\n\t" or unicodedata.category(ch) not in {"Cc", "Cf", "Co", "Cs"}
    )
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def screen_question(question: str, settings: Settings) -> str:
    """Return the normalized question, or raise if it must not reach the pipeline.

    Raises:
        InputRejectedError: reason is one of "empty", "too_short", "garbage", "injection".
    """
    cleaned = _normalize(question)
    if not cleaned:
        raise InputRejectedError("empty", "The question is empty.")
    if len(cleaned) < settings.input_min_chars:
        raise InputRejectedError(
            "too_short", f"The question must be at least {settings.input_min_chars} characters."
        )
    if not re.search(r"\w", cleaned):
        raise InputRejectedError("garbage", "The question contains no words.")
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            raise InputRejectedError(
                "injection",
                "The question looks like an attempt to change the assistant's instructions. "
                "Ask about the content of your books instead.",
            )
    return cleaned
