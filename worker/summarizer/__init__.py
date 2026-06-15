"""
Summarization entry points and shared citation utilities.

summarize() and summarize_incremental() delegate to the DeepSeek backend.
Shared helpers live here: _format_messages_for_prompt, _parse_response,
_named_to_numbered, _numbered_to_named, _prior_source_map.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

URL_PATTERN = re.compile(r'https?://[^\s<>\[\]()\"\']+')
NAMED_CITE_PATTERN = re.compile(r'\[([^\[\]]+)\]')


SYSTEM_PROMPT = """You are a news aggregation assistant. Read raw messages from multiple Telegram channels and compress them into a flat list of events with inline source citations.

Your primary goal is compression. Hundreds of messages should collapse into a small number of events — one entry per distinct real-world event, not one entry per message.

Rules:
- COLLAPSE THREADS: Messages about the same event must become one entry. If an airstrike is reported, then follow-up messages add the location, then further messages add a casualty count — that is ONE entry with all details merged. Multiple developments within the same story arc — an incident, reactions to it, follow-up statements, official responses — are ONE event, not separate entries.
- DEDUPLICATE ACROSS CHANNELS: The same event reported by multiple channels is still one event.
- COMPRESS AGGRESSIVELY: A 6-hour window of messages should produce roughly 10-20 events. If you are producing more than 25, you are not compressing enough. Merge related sub-stories.
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


def _numbered_to_named(events: list[dict]) -> list[dict]:
    """Inverse of _named_to_numbered: expand [N] back to [SourceName].

    Feeds a stored report (numbered citations + sources list) back into the
    model for incremental merging.
    """
    result = []
    for event in events:
        sources = event.get("sources", [])

        def replace(m):
            n = int(m.group(1))
            if 1 <= n <= len(sources):
                return f"[{sources[n - 1]['name']}]"
            return ""

        text = re.sub(r'\[(\d+)\]', replace, event.get("text", ""))
        result.append({"text": text, "countries": event.get("countries", [])})
    return result


def _prior_source_map(events: list[dict]) -> dict[str, str]:
    """Map source name -> url from already-resolved events, for carry-forward."""
    mapping: dict[str, str] = {}
    for event in events:
        for source in event.get("sources", []):
            name, url = source.get("name"), source.get("url")
            if name and url and name not in mapping:
                mapping[name] = url
    return mapping


def _named_to_numbered(
    events: list[dict],
    messages: list[dict],
    prior_sources: dict[str, str] | None = None,
) -> list[dict]:
    """Convert inline named citations [SourceName] to numbered [N] with source objects.

    Each source becomes {"name": "...", "url": "..."} where url may be null.
    For @channel sources: url is the Telegram post link (t.me/channel/msgid).
    For news org sources: url is extracted from the matching Telegram message if available.
    prior_sources carries forward urls for citations not resolvable from `messages`
    (incremental merges, where the old messages are no longer in hand).
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

            if url is None and prior_sources:
                url = prior_sources.get(name)

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
) -> dict:
    """Summarize a batch of messages into a fresh report."""
    from .deepseek import summarize_deepseek
    return summarize_deepseek(messages, from_ts, to_ts)


def summarize_incremental(
    prior: dict,
    new_messages: list[dict],
    to_ts: datetime,
) -> dict:
    """Merge new messages into an existing report. Returns the updated report.

    Folds new information into existing events and appends genuinely new ones,
    carrying forward citation urls already resolved in the prior report.
    """
    if not new_messages:
        return prior
    from .deepseek import summarize_incremental_deepseek
    return summarize_incremental_deepseek(prior, new_messages, to_ts)
