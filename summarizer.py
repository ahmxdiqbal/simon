"""
Summarization dispatcher and shared utilities.

Shared functions used by both backends:
  - _format_messages_for_prompt()  (DeepSeek backend)
  - _parse_response()
  - _named_to_numbered()           (both backends)
  - URL_PATTERN, NAMED_CITE_PATTERN

The summarize() entry point dispatches to the configured backend
based on the SUMMARIZER_BACKEND env var ("local" or "deepseek").
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

BACKEND = os.getenv("SUMMARIZER_BACKEND", "local")

URL_PATTERN = re.compile(r'https?://[^\s<>\[\]()\"\']+')
NAMED_CITE_PATTERN = re.compile(r'\[([^\[\]]+)\]')


def _format_messages_for_prompt(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        sent = msg.get("sent_at", "")[:19].replace("T", " ")
        channel = msg.get("channel_title") or msg.get("channel", "unknown")
        text = msg["text"].replace("\n", " ").strip()
        lines.append(f"[{sent}] [{channel}] {text}")
    return "\n".join(lines)


def _parse_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _word_set(text: str) -> set[str]:
    """Extract lowercase alphanumeric words for overlap scoring."""
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def _overlap_score(event_words: set[str], msg_words: set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not event_words or not msg_words:
        return 0.0
    return len(event_words & msg_words) / len(event_words | msg_words)


def _match_source_url(source_name: str, urls: list[str]) -> str | None:
    """Try to find a URL whose domain matches a news org name."""
    normalized = re.sub(r'[^a-z0-9]', '', source_name.lower())
    for url in urls:
        domain = url.split('/')[2].lower() if len(url.split('/')) > 2 else ""
        domain_normalized = re.sub(r'[^a-z0-9]', '', domain)
        if normalized in domain_normalized:
            return url
    return None


def _build_message_index(messages: list[dict]) -> dict[str, list[dict]]:
    """Group messages by channel username for fast lookup."""
    index: dict[str, list[dict]] = {}
    for msg in messages:
        channel = msg.get("channel", "")
        index.setdefault(channel, []).append(msg)
    return index


def _named_to_numbered(events: list[dict], messages: list[dict]) -> list[dict]:
    """Convert inline named citations [SourceName] to numbered [N] with source objects.

    Each source becomes {"name": "...", "url": "..."} where url may be null.
    For @channel sources: url is the Telegram post link (t.me/channel/msgid).
    For news org sources: url is extracted from the matching Telegram message if available.
    """
    msg_index = _build_message_index(messages)

    # Pre-compute word sets for all messages
    msg_words_cache: dict[int, set[str]] = {}
    for msg in messages:
        msg_words_cache[id(msg)] = _word_set(msg.get("text", ""))

    # Build a reverse map from channel_title to username
    title_to_username: dict[str, str] = {}
    for msg in messages:
        title = msg.get("channel_title", "")
        username = msg.get("channel", "")
        if title and username:
            title_to_username[title] = username

    result = []
    for event in events:
        text = event.get("text", "")
        countries = event.get("countries", [])

        # Strip the citation brackets to get clean text for matching
        clean_text = NAMED_CITE_PATTERN.sub('', text)
        event_words = _word_set(clean_text)

        # First pass: collect unique source names in order of appearance
        source_names: list[str] = []
        for match in NAMED_CITE_PATTERN.finditer(text):
            name = match.group(1)
            if name.isdigit():
                continue
            if name not in source_names:
                source_names.append(name)

        # Find the best matching messages for this event across all channels
        scored_messages: list[tuple[float, dict]] = []
        for msg in messages:
            score = _overlap_score(event_words, msg_words_cache[id(msg)])
            if score > 0.05:
                scored_messages.append((score, msg))
        scored_messages.sort(key=lambda x: x[0], reverse=True)

        # Collect all URLs from the top matching messages (take top 10)
        top_messages = [msg for _, msg in scored_messages[:10]]
        all_urls: list[str] = []
        for msg in top_messages:
            all_urls.extend(URL_PATTERN.findall(msg.get("text", "")))

        # For each @channel source, find the best matching message from that channel
        channel_best_msg: dict[str, dict] = {}
        for name in source_names:
            if not name.startswith("@"):
                continue
            username = name[1:]
            channel_msgs = msg_index.get(username, [])
            if not channel_msgs:
                resolved = title_to_username.get(username)
                if resolved:
                    channel_msgs = msg_index.get(resolved, [])
                    username = resolved
            if not channel_msgs:
                continue
            best_score = -1.0
            best_msg = None
            for msg in channel_msgs:
                score = _overlap_score(event_words, msg_words_cache[id(msg)])
                if score > best_score:
                    best_score = score
                    best_msg = msg
            if best_msg and best_score > 0.05:
                channel_best_msg[username] = best_msg
                all_urls.extend(URL_PATTERN.findall(best_msg.get("text", "")))

        # Build source objects
        source_index: dict[str, int] = {}
        sources_list: list[dict] = []
        for name in source_names:
            source_index[name] = len(sources_list) + 1

            url = None
            if name.startswith("@"):
                username = name[1:]
                best = channel_best_msg.get(username)
                if not best:
                    resolved = title_to_username.get(username)
                    if resolved:
                        best = channel_best_msg.get(resolved)
                        username = resolved
                if best:
                    url = f"https://t.me/{username}/{best['id']}"
            else:
                url = _match_source_url(name, all_urls)

            sources_list.append({"name": name, "url": url})

        # Replace named citations with numbered ones
        def replace_citation(m):
            name = m.group(1)
            if name.isdigit():
                return ""
            n = source_index.get(name)
            if n is not None:
                return f"[{n}]"
            return ""

        new_text = NAMED_CITE_PATTERN.sub(replace_citation, text)
        new_text = re.sub(r'  +', ' ', new_text).strip()

        result.append({
            "text": new_text,
            "countries": countries,
            "sources": sources_list,
        })

    return result


def summarize(
    messages: list[dict],
    from_ts: datetime | None,
    to_ts: datetime,
    on_progress: callable | None = None,
) -> dict:
    """Dispatch to the configured summarization backend."""
    if BACKEND == "deepseek":
        from summarizer_deepseek import summarize_deepseek
        return summarize_deepseek(messages, from_ts, to_ts, on_progress)
    else:
        from summarizer_local import summarize_local
        return summarize_local(messages, from_ts, to_ts, on_progress)
