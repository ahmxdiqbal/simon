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
    """Generate using the model's chat template with thinking disabled."""
    from mlx_lm import generate
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Qwen3.5 templates start a <think> block in the generation prompt.
    # Close it immediately so the model skips reasoning and outputs directly.
    if prompt.rstrip().endswith("<think>"):
        prompt = prompt.rstrip() + "\n</think>\n\n"
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    # Safety: strip any residual think blocks from output
    if "</think>" in output:
        output = output.split("</think>", 1)[1]
    return output.strip()


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
            lines.append(
                f"[@{channel}] {detail} | Countries: {countries} | Source: {source}"
            )
    return "\n".join(lines)


def _synthesize_events(
    assembled_text: str, on_progress: callable | None = None
) -> list[dict]:
    """Run 4B model on assembled facts to produce deduplicated events."""
    if on_progress:
        on_progress("Synthesizing events with 4B model...")

    model, tokenizer = _manager.get_synth()
    prompt = SYNTH_PROMPT + assembled_text

    raw_output = _generate(model, tokenizer, prompt, max_tokens=4096)

    # Try parsing the response
    try:
        parsed = _parse_response(raw_output)
        events = parsed.get("events", [])
        if events:
            return events
    except (json.JSONDecodeError, Exception):
        pass

    # Retry once on failure
    if on_progress:
        on_progress("Retrying synthesis...")
    raw_output = _generate(model, tokenizer, prompt, max_tokens=4096)

    try:
        parsed = _parse_response(raw_output)
        return parsed.get("events", [])
    except (json.JSONDecodeError, Exception):
        pass

    # Last resort: regex-based JSON extraction
    match = re.search(r'\{"events"\s*:\s*\[.*\]\s*\}', raw_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()).get("events", [])
        except json.JSONDecodeError:
            pass

    return []


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
