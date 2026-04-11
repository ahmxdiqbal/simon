"""
Local two-model summarization backend using MLX.

Pipeline:
  1. Qwen 3.5 0.8B (triage) — per-message fact extraction
  2. Python glue — inject channel metadata, filter irrelevant, format
  3. Qwen 3.5 4B (synthesis) — deduplicate, merge, produce final events with citations
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from summarizer import _parse_response, _named_to_numbered

TRIAGE_MODEL = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
SYNTH_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"

TRIAGE_PROMPT = """Extract facts from this Telegram message. Return JSON with: details (array of facts), countries (array of full country names involved), news_source (name of news org if cited, else null). If irrelevant, return {"skip": true}.

Example input: "Reuters reports Turkey struck PKK positions in Iraq, killing 12"
Example output: {"details": ["Turkey struck PKK positions in Iraq, killing 12"], "countries": ["Turkey", "Iraq"], "news_source": "Reuters"}

Example input: "BREAKING: France calls emergency UN Security Council meeting over Syria crisis, per AP"
Example output: {"details": ["France called emergency UN Security Council meeting over Syria crisis"], "countries": ["France", "Syria"], "news_source": "AP"}

Example input: "Join our channel for more updates! t.me/example"
Example output: {"skip": true}

Example input: "3 Ukrainian drones shot down over Crimea overnight"
Example output: {"details": ["3 Ukrainian drones shot down over Crimea overnight"], "countries": ["Ukraine", "Russia"], "news_source": null}

Message: """

SYNTH_PROMPT = """You are a news aggregation assistant. The following are pre-extracted facts from Telegram channels. Each line has: channel name, fact, countries involved, and source (a news org, or "none" meaning the channel itself reported it).

Deduplicate and merge facts about the same event into single entries. Use inline named citations:
- If Source is a news org name, cite it: [Reuters]
- If Source is "none", cite the channel: [@channelname]
- If multiple sources report the same fact, cite all: [Reuters][@clashreport]

Return ONLY valid JSON, no markdown fences:
{"events": [{"text": "event text with [citations]", "countries": ["Country"]}]}

Facts:
"""


class ModelManager:
    """Lazy-loads and caches both MLX models for the process lifetime."""

    def __init__(self):
        self._triage_model = None
        self._triage_tokenizer = None
        self._synth_model = None
        self._synth_tokenizer = None

    def get_triage(self):
        if self._triage_model is None:
            from mlx_lm import load
            self._triage_model, self._triage_tokenizer = load(TRIAGE_MODEL)
        return self._triage_model, self._triage_tokenizer

    def get_synth(self):
        if self._synth_model is None:
            from mlx_lm import load
            self._synth_model, self._synth_tokenizer = load(SYNTH_MODEL)
        return self._synth_model, self._synth_tokenizer


_manager = ModelManager()


def _get_models():
    """Force-load both models. Useful for verifying setup."""
    _manager.get_triage()
    _manager.get_synth()
    return True


def _generate(model, tokenizer, user_content: str, max_tokens: int) -> str:
    """Generate using the model's chat template with thinking disabled.

    Streams tokens and stops early if the model enters a repetition loop.
    """
    from mlx_lm import stream_generate

    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Qwen3.5 templates start a <think> block in the generation prompt.
    # Close it immediately so the model skips reasoning and outputs directly.
    if prompt.rstrip().endswith("<think>"):
        prompt = prompt.rstrip() + "\n</think>\n\n"

    # Stream tokens and detect repetition loops
    text = ""
    segment_size = 200

    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
    ):
        text += response.text

        # Check for repetition: compare last two segments of equal length
        if len(text) > segment_size * 3:
            current = text[-segment_size:]
            previous = text[-segment_size * 2 : -segment_size]
            if current == previous:
                print(f"[generate] Repetition loop at {len(text)} chars, stopping")
                break

    # Safety: strip any residual think blocks from output
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _repair_truncated_json(raw: str) -> list[dict] | None:
    """Try to salvage events from truncated JSON output.

    The model often produces valid events then loops and gets cut off mid-JSON.
    Strategy: find the last complete event object and close the array/object.
    """
    # Find the events array start
    match = re.search(r'\{\s*"events"\s*:\s*\[', raw)
    if not match:
        return None

    # Find all complete event objects (those ending with a closing brace
    # followed by a comma or the array close)
    events_start = match.end()
    events = []
    depth = 0
    obj_start = None

    for i in range(events_start, len(raw)):
        ch = raw[i]
        if ch == '{' and depth == 0:
            obj_start = i
            depth = 1
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    obj = json.loads(raw[obj_start : i + 1])
                    if "text" in obj:
                        events.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None

    if events:
        # Deduplicate by text since the model was looping
        seen = set()
        unique = []
        for e in events:
            t = e["text"]
            if t not in seen:
                seen.add(t)
                unique.append(e)
        return unique
    return None


def _parse_triage_output(raw: str) -> dict | None:
    """Extract JSON from triage model output, tolerating noise."""
    raw = raw.strip()
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in the output
    match = re.search(r'\{[^{}]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try to find JSON with nested arrays
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _triage_messages(
    messages: list[dict], on_progress: callable | None = None
) -> list[dict]:
    """Run 0.8B model on each message to extract structured facts.

    Returns a list of dicts, one per message, with keys:
      - details: list[str]
      - countries: list[str]
      - news_source: str | None
      - channel: str (username)
      - channel_title: str
      - msg_id: int
      - raw_text: str (fallback)
    Skipped messages are excluded.
    """
    model, tokenizer = _manager.get_triage()
    total = len(messages)
    results = []

    for i, msg in enumerate(messages):
        if on_progress and i % 50 == 0:
            on_progress(f"Extracting facts: {i}/{total} messages...")

        text = msg["text"].replace("\n", " ").strip()
        if not text:
            continue

        # Skip recap/digest messages — they're redundant since individual
        # events already appear in other messages, and they overwhelm the
        # synthesis model with too much content per line.
        if len(text) > 800:
            continue

        prompt = TRIAGE_PROMPT + text
        raw_output = _generate(model, tokenizer, prompt, max_tokens=150)
        parsed = _parse_triage_output(raw_output)

        channel = msg.get("channel", "unknown")
        channel_title = msg.get("channel_title", channel)
        msg_id = msg.get("id", 0)

        if parsed and parsed.get("skip"):
            continue

        if parsed and "details" in parsed:
            results.append({
                "details": parsed["details"],
                "countries": parsed.get("countries", []),
                "news_source": parsed.get("news_source"),
                "channel": channel,
                "channel_title": channel_title,
                "msg_id": msg_id,
                "raw_text": text,
            })
        else:
            # Fallback: include raw text as a single detail
            results.append({
                "details": [text],
                "countries": [],
                "news_source": None,
                "channel": channel,
                "channel_title": channel_title,
                "msg_id": msg_id,
                "raw_text": text,
            })

    if on_progress:
        on_progress(f"Extraction complete: {len(results)} relevant messages from {total} total.")

    return results


def _assemble_for_synthesis(extractions: list[dict]) -> str:
    """Format triage outputs into a structured text block for the 4B model.

    Each line: [@channel] fact | Countries: X, Y | Source: org_or_none
    """
    lines = []
    for ext in extractions:
        channel = ext["channel"]
        source = ext["news_source"] if ext["news_source"] else "none"
        countries = ", ".join(ext["countries"]) if ext["countries"] else "unknown"

        for detail in ext["details"]:
            # Truncate overly long details to prevent synthesis overload
            if len(detail) > 300:
                detail = detail[:300]
            lines.append(
                f"[@{channel}] {detail} | Countries: {countries} | Source: {source}"
            )
    return "\n".join(lines)


def _try_parse_events(raw_output: str) -> list[dict] | None:
    """Try all parsing strategies on raw model output. Returns events or None."""
    # Strategy 1: clean JSON parse
    try:
        parsed = _parse_response(raw_output)
        events = parsed.get("events", [])
        if events:
            return events
    except Exception:
        pass

    # Strategy 2: regex for complete JSON
    match = re.search(r'\{"events"\s*:\s*\[.*\]\s*\}', raw_output, re.DOTALL)
    if match:
        try:
            events = json.loads(match.group()).get("events", [])
            if events:
                return events
        except json.JSONDecodeError:
            pass

    # Strategy 3: repair truncated/looping JSON
    repaired = _repair_truncated_json(raw_output)
    if repaired:
        return repaired

    return None


SYNTH_CHUNK_SIZE = 15  # lines per synthesis chunk


def _synthesize_events(
    assembled_text: str, on_progress: callable | None = None
) -> list[dict]:
    """Run 4B model on assembled facts in chunks to produce events.

    The 4B model degenerates into repetition loops on large inputs.
    Chunking into ~15-line batches keeps each call within the model's
    reliable output range, then we merge results.
    """
    model, tokenizer = _manager.get_synth()
    lines = assembled_text.splitlines()
    chunks = [
        lines[i : i + SYNTH_CHUNK_SIZE]
        for i in range(0, len(lines), SYNTH_CHUNK_SIZE)
    ]

    all_events: list[dict] = []

    for i, chunk in enumerate(chunks):
        if on_progress:
            on_progress(f"Synthesizing chunk {i + 1}/{len(chunks)}...")

        chunk_text = "\n".join(chunk)
        raw_output = _generate(model, tokenizer, SYNTH_PROMPT + chunk_text, max_tokens=4096)

        events = _try_parse_events(raw_output)
        if events:
            all_events.extend(events)
        else:
            # Retry once for this chunk
            print(f"[synth] Chunk {i + 1} failed, retrying. "
                  f"Output length: {len(raw_output)}")
            raw_output = _generate(model, tokenizer, SYNTH_PROMPT + chunk_text, max_tokens=4096)
            events = _try_parse_events(raw_output)
            if events:
                all_events.extend(events)
            else:
                print(f"[synth] Chunk {i + 1} retry also failed, skipping")

    if on_progress:
        on_progress(f"Synthesis complete: {len(all_events)} events from {len(chunks)} chunks")

    return all_events


def summarize_local(
    messages: list[dict],
    from_ts: datetime | None,
    to_ts: datetime,
    on_progress: callable | None = None,
) -> dict:
    """Two-model local inference pipeline: 0.8B triage + 4B synthesis."""
    if not messages:
        return {
            "period": {"from": from_ts.isoformat() if from_ts else None, "to": to_ts.isoformat()},
            "events": [],
            "message_count": 0,
            "cost": {"input_tokens": 0, "output_tokens": 0, "input_cost_usd": 0, "output_cost_usd": 0, "total_cost_usd": 0},
            "model": f"{TRIAGE_MODEL} + {SYNTH_MODEL}",
        }

    # Phase 1: Per-message extraction with 0.8B
    if on_progress:
        on_progress(f"Phase 1: Extracting facts from {len(messages)} messages...")
    extractions = _triage_messages(messages, on_progress)

    if not extractions:
        return {
            "period": {"from": from_ts.isoformat() if from_ts else None, "to": to_ts.isoformat()},
            "events": [],
            "message_count": len(messages),
            "cost": {"input_tokens": 0, "output_tokens": 0, "input_cost_usd": 0, "output_cost_usd": 0, "total_cost_usd": 0},
            "model": f"{TRIAGE_MODEL} + {SYNTH_MODEL}",
        }

    # Phase 2: Assemble structured input for synthesis
    if on_progress:
        on_progress("Phase 2: Assembling structured input...")
    assembled = _assemble_for_synthesis(extractions)

    # Phase 3: Synthesize with 4B
    if on_progress:
        on_progress(f"Phase 3: Synthesizing {len(extractions)} facts into events...")
    raw_events = _synthesize_events(assembled, on_progress)

    # Phase 4: Convert named citations to numbered with URLs
    if on_progress:
        on_progress(f"Linking {len(raw_events)} events to sources...")
    numbered_events = _named_to_numbered(raw_events, messages)

    return {
        "period": {
            "from": from_ts.isoformat() if from_ts else None,
            "to": to_ts.isoformat(),
        },
        "events": numbered_events,
        "message_count": len(messages),
        "cost": {"input_tokens": 0, "output_tokens": 0, "input_cost_usd": 0, "output_cost_usd": 0, "total_cost_usd": 0},
        "model": f"{TRIAGE_MODEL} + {SYNTH_MODEL}",
    }
