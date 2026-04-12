"""
DeepSeek API summarization backend.

Moved from summarizer.py — handles chunked API calls, deduplication,
and cost tracking for the DeepSeek-V3 model.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv

from summarizer import (
    SYSTEM_PROMPT,
    _format_messages_for_prompt,
    _parse_response,
    _named_to_numbered,
)

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

CHUNK_SIZE = 1000

# Pricing for deepseek-chat (DeepSeek-V3). Verify at api-docs.deepseek.com/quick_start/pricing
INPUT_COST_PER_TOKEN = 0.28 / 1_000_000
OUTPUT_COST_PER_TOKEN = 0.42 / 1_000_000

MODEL = "deepseek-chat"


def _compute_cost(input_tokens: int, output_tokens: int) -> dict:
    input_cost = input_tokens * INPUT_COST_PER_TOKEN
    output_cost = output_tokens * OUTPUT_COST_PER_TOKEN
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


def _dedup_events(
    events: list[dict], input_tokens: int, output_tokens: int
) -> tuple[list[dict], int, int]:
    """Ask the model to deduplicate a merged event list from multiple chunks."""
    if len(events) <= 1:
        return events, input_tokens, output_tokens

    payload = json.dumps(events, ensure_ascii=False)
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": (
                    "The following is a JSON array of events collected from multiple time chunks. "
                    "Each event has text with inline named citations like [Reuters] or [@channel]. "
                    "Remove exact and near-duplicate entries. Merge entries that describe the same real-world event, "
                    "preserving all unique details and all source citations. "
                    "Preserve the union of all countries. Keep the same citation format — named sources in brackets. "
                    "Return only the deduplicated JSON array, no markdown, no commentary.\n\n"
                    + payload
                ),
            }
        ],
    )
    input_tokens += response.usage.prompt_tokens
    output_tokens += response.usage.completion_tokens
    try:
        deduped = json.loads(response.choices[0].message.content.strip())
    except Exception:
        deduped = events
    return deduped, input_tokens, output_tokens


def summarize_deepseek(
    messages: list[dict],
    from_ts: datetime | None,
    to_ts: datetime,
    on_progress: callable | None = None,
) -> dict:
    """Extract events via DeepSeek API. Chunks large message sets."""
    if not messages:
        return {
            "period": {"from": from_ts.isoformat() if from_ts else None, "to": to_ts.isoformat()},
            "events": [],
            "message_count": 0,
            "cost": _compute_cost(0, 0),
            "model": MODEL,
        }

    chunks = [messages[i : i + CHUNK_SIZE] for i in range(0, len(messages), CHUNK_SIZE)]

    if on_progress:
        on_progress(f"Processing {len(messages)} messages in {len(chunks)} chunk(s)...")

    all_events: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0

    def _call_chunk(i: int, chunk: list[dict]):
        if on_progress:
            on_progress(f"Sending chunk {i + 1}/{len(chunks)} to DeepSeek...")
        formatted = _format_messages_for_prompt(chunk)
        chunk_from = chunk[0]["sent_at"]
        chunk_to = chunk[-1]["sent_at"]
        return client.chat.completions.create(
            model=MODEL,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Extract events from the following {len(chunk)} messages "
                        f"(period: {chunk_from[:19]} to {chunk_to[:19]} UTC).\n\n"
                        + formatted
                    ),
                },
            ],
        )

    if len(chunks) == 1:
        response = _call_chunk(0, chunks[0])
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens
        parsed = _parse_response(response.choices[0].message.content)
        all_events.extend(parsed.get("events", []))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = {pool.submit(_call_chunk, i, c): i for i, c in enumerate(chunks)}
            for future in as_completed(futures):
                response = future.result()
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
                parsed = _parse_response(response.choices[0].message.content)
                all_events.extend(parsed.get("events", []))

    if len(chunks) > 1:
        if on_progress:
            on_progress("Deduplicating events across chunks...")
        all_events, total_input_tokens, total_output_tokens = _dedup_events(
            all_events, total_input_tokens, total_output_tokens
        )

    if on_progress:
        on_progress(f"Linking {len(all_events)} events to sources...")
    numbered_events = _named_to_numbered(all_events, messages)

    return {
        "period": {
            "from": from_ts.isoformat() if from_ts else None,
            "to": to_ts.isoformat(),
        },
        "events": numbered_events,
        "message_count": len(messages),
        "cost": _compute_cost(total_input_tokens, total_output_tokens),
        "model": MODEL,
    }
