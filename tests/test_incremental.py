"""Incremental merge + citation carry-forward tests (LLM mocked)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from summarizer import _named_to_numbered, _numbered_to_named, _prior_source_map
from summarizer import deepseek


def _prior_report() -> dict:
    return {
        "period": {"from": "2026-06-15T00:00:00+00:00", "to": "2026-06-15T03:00:00+00:00"},
        "events": [
            {
                "text": "Initial strike reported [1][2].",
                "countries": ["X"],
                "sources": [
                    {"name": "Reuters", "url": "https://reuters.com/x"},
                    {"name": "@oldchan", "url": "https://t.me/oldchan/5"},
                ],
            }
        ],
        "message_count": 10,
        "cost": {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_cost_usd": 0.0001,
            "output_cost_usd": 0.0001,
            "total_cost_usd": 0.0002,
        },
        "model": "deepseek-chat",
    }


def test_numbered_to_named_roundtrip():
    named = _numbered_to_named(_prior_report()["events"])
    assert named == [{"text": "Initial strike reported [Reuters][@oldchan].", "countries": ["X"]}]


def test_prior_source_map():
    mapping = _prior_source_map(_prior_report()["events"])
    assert mapping == {
        "Reuters": "https://reuters.com/x",
        "@oldchan": "https://t.me/oldchan/5",
    }


def test_named_to_numbered_carries_forward_unresolved_urls():
    events = [{"text": "Restated event [Reuters].", "countries": ["X"]}]
    # No messages to resolve from, but a prior url exists.
    out = _named_to_numbered(events, [], prior_sources={"Reuters": "https://reuters.com/x"})
    assert out[0]["sources"] == [{"name": "Reuters", "url": "https://reuters.com/x"}]


def _fake_client(merged_events: list[dict], prompt_tokens=200, completion_tokens=80):
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"events": merged_events})
        ))],
    )
    create = lambda **kwargs: response
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_incremental_merge_carries_old_sources_and_resolves_new(monkeypatch):
    merged = [
        {"text": "Initial strike reported, casualties rising [Reuters][@oldchan].", "countries": ["X"]},
        {"text": "New event from channel happening now [@newchan].", "countries": ["Y"]},
    ]
    monkeypatch.setattr(deepseek, "client", _fake_client(merged))

    new_messages = [{
        "id": 42,
        "channel": "newchan",
        "channel_title": "New Chan",
        "sent_at": "2026-06-15T04:00:00+00:00",
        "text": "New event from channel happening now",
    }]
    to_ts = datetime(2026, 6, 15, 4, 30, tzinfo=timezone.utc)

    result = deepseek.summarize_incremental_deepseek(_prior_report(), new_messages, to_ts)

    # Old citations carry forward their resolved urls.
    e0 = result["events"][0]
    by_name = {s["name"]: s["url"] for s in e0["sources"]}
    assert by_name["Reuters"] == "https://reuters.com/x"
    assert by_name["@oldchan"] == "https://t.me/oldchan/5"

    # New channel citation resolves to a fresh telegram link from the new message.
    e1 = result["events"][1]
    assert e1["sources"] == [{"name": "@newchan", "url": "https://t.me/newchan/42"}]

    # Cost and message count accumulate; period extends to the new boundary.
    assert result["message_count"] == 11
    assert result["cost"]["input_tokens"] == 300
    assert result["cost"]["output_tokens"] == 130
    assert result["period"]["from"] == "2026-06-15T00:00:00+00:00"
    assert result["period"]["to"] == to_ts.isoformat()


def test_incremental_noop_on_empty(monkeypatch):
    monkeypatch.setattr(deepseek, "client", _fake_client([]))
    prior = _prior_report()
    assert deepseek.summarize_incremental_deepseek(prior, [], datetime.now(timezone.utc)) is prior
