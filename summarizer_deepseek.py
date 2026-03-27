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

SYSTEM_PROMPT = """You are a news aggregation assistant. Read raw messages from multiple Telegram channels and compress them into a flat list of events with inline source citations.

Your primary goal is compression. Hundreds of messages should collapse into a small number of events — one entry per distinct real-world event, not one entry per message.

Rules:
- COLLAPSE THREADS: Messages about the same event must become one entry. If an airstrike is reported, then follow-up messages add the location, then further messages add a casualty count — that is ONE entry with all details merged.
- DEDUPLICATE ACROSS CHANNELS: The same event reported by multiple channels is still one event.
- For each event, list ALL nations meaningfully involved in the "countries" field.
- Keep event text factual and concise: who, what, where, when, how many. No analysis or commentary.
- Preserve specifics: locations, actor names, numbers, dates.
- Translate any foreign-language messages into English.
- Ignore forwarding notices, channel promotions, and anything with no geopolitical relevance.

CITATION RULES:
- Every factual claim MUST have an inline citation using the SOURCE NAME in square brackets, e.g. [Reuters] or [@clashreport].
- SOURCE PRIORITY: If a Telegram channel quotes or cites a news organization (e.g. "Reuters reports...", "according to AP..."), the citation must be the NEWS ORGANIZATION, not the Telegram channel. Only use the Telegram channel name (prefixed with @) when the channel is reporting on its own without citing another source.
- If multiple sources report the same claim, cite all of them: [Reuters][@clashreport].
- Place citations immediately after the claim they support.
- NEVER invent sources. Only cite sources present in the input messages.
- Use the exact channel name as shown in the input (prefixed with @), and the exact news org name as mentioned.

Output format — return ONLY valid JSON, no markdown fences, no commentary:
{
  "events": [
    {
      "text": "Turkish drone strikes killed 12 PKK fighters in Duhok[Reuters][@clashreport], with 3 additional casualties reported in Erbil[@middleeastspectator]. The strikes targeted a weapons depot and ammunition facility[Reuters].",
      "countries": ["Turkey", "Iraq"]
    },
    {
      "text": "France called for an emergency UN Security Council meeting and urged all parties to de-escalate[AP][@middleeastspectator].",
      "countries": ["France"]
    }
  ]
}

Use full country names (e.g., "United States", not "US" or "USA").
Do NOT use numbered citations. Use the actual source name inside the brackets.
"""


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
