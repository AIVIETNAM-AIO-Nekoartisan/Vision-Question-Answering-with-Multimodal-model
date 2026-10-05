"""Query expansion via DeepSeek.

Off by default. It adds a network round trip to every query, so the ablation in
spec section 8 measures whether it earns that cost before it becomes standard.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_PROMPT = (
    "Rewrite this video-search query as a short, concrete visual description. "
    "Name objects, actions and setting. No preamble, one line, under 40 words.\n\n"
    "Query: {query}"
)


def expand_query(query: str, timeout: float = 20.0) -> Optional[str]:
    """Return an expanded query, or None if expansion is unavailable."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.info("DEEPSEEK_API_KEY not set, skipping expansion")
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=timeout,
        )
        resp = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[{"role": "user", "content": _PROMPT.format(query=query)}],
            max_tokens=120,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception as exc:
        # A failed expansion must not fail the search; fall back to the raw query.
        logger.warning("query expansion failed: %s", exc)
        return None
