"""
Local summarization backend using MLX.

Mirrors summarizer_deepseek.py: chunk messages, send each chunk to the
model with the shared SYSTEM_PROMPT, run a dedup pass if there are multiple
chunks, then convert named citations to numbered ones.

Generation routes through model_server.py if it's running (saves ~20s of
model loading per invocation). Falls back to loading the model directly.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from summarizer import (
    SYSTEM_PROMPT,
    _format_messages_for_prompt,
    _named_to_numbered,
    _parse_response,
)

MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"

CHUNK_SIZE = 150
MAX_TOKENS = 8192
REPETITION_WINDOW = 600
PREFILL_STEP_SIZE = 4096

# Sampling: small temperature + repetition penalty to prevent loops.
# temp=0.0 is greedy (deterministic), which gets stuck in repetition.
# A light touch keeps output coherent while letting the model escape loops.
TEMPERATURE = 0.1
REPETITION_PENALTY = 1.1
REPETITION_PENALTY_CONTEXT = 256

SERVER_URL = "http://127.0.0.1:8321/generate"
DEBUG_DIR = Path(__file__).parent / "tests" / "debug"

# Cached after first check so we don't retry a dead server on every call.
_server_checked = False
_server_up = False


# ---------------------------------------------------------------------------
# Generation: server path vs direct path
# ---------------------------------------------------------------------------

def _try_server(system_prompt: str, user_content: str, max_tokens: int) -> str | None:
    """POST to the persistent model server. Returns None if unreachable."""
    global _server_checked, _server_up

    if _server_checked and not _server_up:
        return None

    try:
        payload = json.dumps({
            "system_prompt": system_prompt,
            "user_content": user_content,
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            SERVER_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())

        if not _server_checked:
            _server_checked = True
            _server_up = True
            print("[model] Using persistent server at localhost:8321")
        return result["content"]
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        if not _server_checked:
            _server_checked = True
            _server_up = False
            print("[model] Server not running, loading model directly")
        return None


class ModelManager:
    """Lazy-load the MLX model once per process (fallback path only)."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def get(self):
        if self._model is None:
            from mlx_lm import load
            self._model, self._tokenizer = load(MODEL)
        return self._model, self._tokenizer


_manager = ModelManager()


def _generate_direct(system_prompt: str, user_content: str, max_tokens: int) -> str:
    """Generate using direct mlx_lm. Loads the model if not already cached."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

    model, tokenizer = _manager.get()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if prompt.rstrip().endswith("<think>"):
        prompt = prompt.rstrip() + "\n</think>\n\n"

    sampler = make_sampler(temp=TEMPERATURE)
    rep_penalty = make_repetition_penalty(
        penalty=REPETITION_PENALTY, context_size=REPETITION_PENALTY_CONTEXT,
    )

    text = ""
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens,
        prefill_step_size=PREFILL_STEP_SIZE,
        sampler=sampler,
        logits_processors=[rep_penalty],
    ):
        text += response.text
        if len(text) > REPETITION_WINDOW * 2:
            tail = text[-REPETITION_WINDOW:]
            if tail in text[:-REPETITION_WINDOW]:
                print(f"[generate] Repetition loop at {len(text)} chars, stopping")
                break

    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _generate(system_prompt: str, user_content: str, max_tokens: int) -> str:
    """Try the persistent server, fall back to direct loading."""
    result = _try_server(system_prompt, user_content, max_tokens)
    if result is not None:
        return result
    return _generate_direct(system_prompt, user_content, max_tokens)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _dedup_by_text(events: list[dict]) -> list[dict]:
    """Drop events with identical text (repetition loop artifacts)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for e in events:
        t = e.get("text", "")
        if t not in seen:
            seen.add(t)
            unique.append(e)
    return unique


def _dump_debug(label: str, content: str) -> Path:
    """Write raw model output to tests/debug/ for inspection."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_DIR / f"{label}.txt"
    path.write_text(content, encoding="utf-8")
    print(f"[debug] wrote {len(content)} chars to {path}")
    return path


def _repair_truncated_json(raw: str) -> list[dict] | None:
    """Salvage complete event objects from truncated JSON output.

    When the model hits max_tokens mid-JSON, we walk the string brace-by-brace
    and pull out every complete {"text": ..., "countries": ...} object.
    """
    match = re.search(r'\{\s*"events"\s*:\s*\[', raw)
    if not match:
        return None

    events = []
    depth = 0
    obj_start = None

    for i in range(match.end(), len(raw)):
        ch = raw[i]
        if ch == "{" and depth == 0:
            obj_start = i
            depth = 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    obj = json.loads(raw[obj_start : i + 1])
                    if "text" in obj:
                        events.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None

    return events if events else None


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _call_chunk(
    chunk_idx: int,
    chunk: list[dict],
    total_chunks: int,
    on_progress: callable | None,
) -> list[dict]:
    """Run one chunk through the model and return its events."""
    if on_progress:
        on_progress(f"Sending chunk {chunk_idx + 1}/{total_chunks} to local model...")

    formatted = _format_messages_for_prompt(chunk)
    chunk_from = chunk[0]["sent_at"]
    chunk_to = chunk[-1]["sent_at"]
    user_content = (
        f"Extract events from the following {len(chunk)} messages "
        f"(period: {chunk_from[:19]} to {chunk_to[:19]} UTC).\n\n"
        + formatted
    )

    raw_output = _generate(SYSTEM_PROMPT, user_content, max_tokens=MAX_TOKENS)
    _dump_debug(f"chunk_{chunk_idx + 1}_raw", raw_output)

    # Strategy 1: clean parse
    try:
        parsed = _parse_response(raw_output)
        events = _dedup_by_text(parsed.get("events", []))
        print(f"[chunk {chunk_idx + 1}] parsed OK: {len(events)} events from {len(raw_output)} chars")
        return events
    except Exception:
        pass

    # Strategy 2: repair truncated JSON (model hit max_tokens mid-output)
    repaired = _repair_truncated_json(raw_output)
    if repaired:
        events = _dedup_by_text(repaired)
        print(f"[chunk {chunk_idx + 1}] repaired truncated JSON: {len(events)} events from {len(raw_output)} chars")
        return events

    print(f"[chunk {chunk_idx + 1}] all parse strategies failed, {len(raw_output)} chars of output")
    return []


def _dedup_events(events: list[dict]) -> list[dict]:
    """Ask the model to deduplicate a merged event list across chunks."""
    if len(events) <= 1:
        return events

    payload = json.dumps(events, ensure_ascii=False)
    user_content = (
        "The following is a JSON array of events collected from multiple time chunks. "
        "Each event has text with inline named citations like [Reuters] or [@channel]. "
        "Remove exact and near-duplicate entries. Merge entries that describe the same real-world event, "
        "preserving all unique details and all source citations. "
        "Preserve the union of all countries. Keep the same citation format — named sources in brackets. "
        "Return only the deduplicated JSON array, no markdown, no commentary.\n\n"
        + payload
    )

    raw_output = _generate(SYSTEM_PROMPT, user_content, max_tokens=MAX_TOKENS)
    _dump_debug("dedup_raw", raw_output)

    try:
        parsed = _parse_response(raw_output)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "events" in parsed:
            return parsed["events"]
        return events
    except Exception as e:
        print(f"[dedup] parse failed ({e}), returning original events")
        return events


def _empty_result(from_ts: datetime | None, to_ts: datetime, message_count: int) -> dict:
    return {
        "period": {
            "from": from_ts.isoformat() if from_ts else None,
            "to": to_ts.isoformat(),
        },
        "events": [],
        "message_count": message_count,
        "cost": {
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost_usd": 0,
            "output_cost_usd": 0,
            "total_cost_usd": 0,
        },
        "model": MODEL,
    }


def summarize_local(
    messages: list[dict],
    from_ts: datetime | None,
    to_ts: datetime,
    on_progress: callable | None = None,
) -> dict:
    """Local single-model pipeline mirroring the DeepSeek flow."""
    if not messages:
        return _empty_result(from_ts, to_ts, 0)

    chunks = [messages[i : i + CHUNK_SIZE] for i in range(0, len(messages), CHUNK_SIZE)]

    if on_progress:
        on_progress(f"Processing {len(messages)} messages in {len(chunks)} chunk(s)...")

    all_events: list[dict] = []
    for i, chunk in enumerate(chunks):
        all_events.extend(_call_chunk(i, chunk, len(chunks), on_progress))

    if len(chunks) > 1:
        if on_progress:
            on_progress("Deduplicating events across chunks...")
        all_events = _dedup_events(all_events)

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
        "cost": {
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost_usd": 0,
            "output_cost_usd": 0,
            "total_cost_usd": 0,
        },
        "model": MODEL,
    }
