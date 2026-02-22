"""Robust JSON extraction from LLM responses."""

import json
import re

from loguru import logger


def _repair_truncated_json(text: str) -> str | None:
    """Try to close a truncated JSON string by balancing brackets.

    Walks the string tracking open braces/brackets/quotes, then appends
    the necessary closing characters.  Returns the repaired string or
    None if the text doesn't look like truncated JSON.
    """
    stack: list[str] = []  # tracks open { [ "
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            if in_string:
                in_string = False
                if stack and stack[-1] == '"':
                    stack.pop()
            else:
                in_string = True
                stack.append('"')
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    if not stack:
        return None  # already balanced, nothing to repair

    # Close everything that's still open
    closers = {"{": "}", "[": "]", '"': '"'}
    # If we're mid-string, close the string first, then close containers
    suffix = ""
    for opener in reversed(stack):
        suffix += closers[opener]

    return text + suffix


def parse_json_response(text: str) -> dict:
    """Extract and parse JSON from an LLM response, with multiple fallbacks.

    Handles markdown fences, trailing commas, truncated output, etc.
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

    # Attempt 4: repair truncated JSON (output cut off by max_tokens)
    fragment = cleaned[start:] if start != -1 else cleaned
    repaired = _repair_truncated_json(fragment)
    if repaired:
        # Remove trailing commas that might exist right before our added closers
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            result = json.loads(repaired)
            logger.warning(
                f"JSON was truncated (input {len(text)} chars). "
                f"Repaired by closing {len(repaired) - len(fragment)} chars of brackets. "
                f"Data may be incomplete."
            )
            return result
        except json.JSONDecodeError:
            pass

    # All attempts failed — log the raw text and raise
    logger.warning(f"JSON parse failed. Raw LLM text ({len(text)} chars):\n{text[:3000]}")
    raise ValueError(
        f"Could not parse JSON from LLM response (length={len(text)})"
    )
