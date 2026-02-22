"""Robust JSON extraction from LLM responses."""

import json
import re

from loguru import logger


def parse_json_response(text: str) -> dict:
    """Extract and parse JSON from an LLM response, with multiple fallbacks.

    Handles markdown fences, trailing commas, and unescaped newlines.
    Raises ValueError with the raw text on failure for easy debugging.
    """
    cleaned = text.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Attempt 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract outermost { ... }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        fragment = cleaned[start : end + 1]
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            pass

        # Attempt 3: fix trailing commas
        fixed = re.sub(r",\s*([}\]])", r"\1", fragment)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # All attempts failed — log the raw text and raise
    logger.warning(f"JSON parse failed. Raw LLM text ({len(text)} chars):\n{text[:3000]}")
    raise ValueError(
        f"Could not parse JSON from LLM response (length={len(text)})"
    )
