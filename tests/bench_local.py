"""
Benchmark summarize_local() against the saved fixture.

Loads the 6h message fixture and the DeepSeek baseline, runs the local pipeline
end-to-end, prints phase timings, then writes both event lists to markdown files
under tests/summaries/ for manual side-by-side quality comparison.

Filenames:
  - DeepSeek:  <model>.md               (e.g. deepseek-chat.md)
  - Local:     <name>_<size>_<quant>.md (e.g. qwen3.5_9b_4.md)

Swap the local model by editing MODEL in summarizer_local.py and rerunning
this script.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SUMMARIES_DIR = Path(__file__).parent / "summaries"
MESSAGES_FILE = FIXTURE_DIR / "messages_6h.json"
BASELINE_FILE = FIXTURE_DIR / "deepseek_baseline.json"


class PhaseTimer:
    """Records timestamps of on_progress callbacks to derive phase durations."""

    def __init__(self) -> None:
        self.start = time.monotonic()
        self.events: list[tuple[float, str]] = []

    def __call__(self, msg: str) -> None:
        elapsed = time.monotonic() - self.start
        self.events.append((elapsed, msg))
        print(f"  [{elapsed:6.1f}s] {msg}")

    def _first_with(self, needle: str) -> float | None:
        for ts, msg in self.events:
            if needle in msg:
                return ts
        return None

    def chunks_duration(self) -> float | None:
        """Time from first chunk send to dedup (or linking if no dedup)."""
        start = self._first_with("Sending chunk 1")
        end = self._first_with("Deduplicating") or self._first_with("Linking")
        if start is None or end is None:
            return None
        return end - start

    def dedup_duration(self) -> float | None:
        start = self._first_with("Deduplicating")
        end = self._first_with("Linking")
        if start is None or end is None:
            return None
        return end - start


def _load_json(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: {path.name} missing. Run: python tests/collect_fixture.py")
        sys.exit(1)
    with path.open() as f:
        return json.load(f)


def _local_slug(model_id: str) -> str:
    """
    Convert an mlx-community model id to the filename stem.
    Example: "mlx-community/Qwen3.5-9B-MLX-4bit" -> "qwen3.5_9b_4"
    """
    short = model_id.split("/")[-1]
    parts = short.split("-")
    if len(parts) < 4:
        return short.lower().replace("-", "_")
    name = parts[0].lower()
    size = parts[1].lower()
    quant = parts[-1].lower().replace("bit", "")
    return f"{name}_{size}_{quant}"


def _format_events_md(events: list[dict]) -> str:
    if not events:
        return "_No events._\n"
    lines: list[str] = []
    for i, ev in enumerate(events, 1):
        countries = ", ".join(ev.get("countries", [])) or "—"
        lines.append(f"### {i}. {countries}\n")
        lines.append(f"{ev.get('text', '').strip()}\n")
    return "\n".join(lines)


def _write_deepseek_md(baseline: dict, fixture: dict) -> Path:
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    model = baseline.get("model", "deepseek-chat")
    path = SUMMARIES_DIR / f"{model}.md"

    duration = baseline.get("duration_seconds")
    duration_str = (
        f"{duration:.1f}s"
        if duration is not None
        else "— (re-run collect_fixture.py to capture timing)"
    )
    cost = baseline.get("cost", {}).get("total_cost_usd", 0.0)

    header = [
        f"# DeepSeek — {model}",
        "",
        f"- Messages:  {fixture['message_count']}",
        f"- Duration:  {duration_str}",
        f"- Events:    {len(baseline.get('events', []))}",
        f"- Cost:      ${cost:.4f}",
        f"- Window:    {fixture['from_ts'][:19]} → {fixture['to_ts'][:19]}",
        "",
        "## Events",
        "",
    ]
    with path.open("w") as f:
        f.write("\n".join(header))
        f.write(_format_events_md(baseline.get("events", [])))
    return path


def _write_local_md(
    result: dict,
    fixture: dict,
    model: str,
    total_time: float,
    chunks_time: float | None,
    dedup_time: float | None,
) -> Path:
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARIES_DIR / f"{_local_slug(model)}.md"

    if chunks_time is not None and chunks_time > 0:
        mps = fixture["message_count"] / chunks_time
        chunks_str = f"{chunks_time:.1f}s  ({mps:.2f} msgs/sec)"
    else:
        chunks_str = "—"
    dedup_str = f"{dedup_time:.1f}s" if dedup_time is not None else "—"

    header = [
        f"# Local — {model}",
        "",
        f"- Model:     {model}",
        f"- Messages:  {fixture['message_count']}",
        f"- Total:     {total_time:.1f}s",
        f"- Chunks:    {chunks_str}",
        f"- Dedup:     {dedup_str}",
        f"- Events:    {len(result.get('events', []))}",
        f"- Window:    {fixture['from_ts'][:19]} → {fixture['to_ts'][:19]}",
        "",
        "## Events",
        "",
    ]
    with path.open("w") as f:
        f.write("\n".join(header))
        f.write(_format_events_md(result.get("events", [])))
    return path


def main() -> None:
    fixture = _load_json(MESSAGES_FILE)
    baseline = _load_json(BASELINE_FILE)

    messages = fixture["messages"]
    from_ts = datetime.fromisoformat(fixture["from_ts"])
    to_ts = datetime.fromisoformat(fixture["to_ts"])

    from summarizer_local import MODEL, summarize_local

    print(f"Fixture: {len(messages)} messages")
    print(f"Window:  {fixture['from_ts'][:19]}  →  {fixture['to_ts'][:19]}")
    print(f"\nModel:   {MODEL}\n")
    print("Running summarize_local (first run includes model load)...\n")

    timer = PhaseTimer()
    t0 = time.monotonic()
    result = summarize_local(messages, from_ts, to_ts, on_progress=timer)
    total_time = time.monotonic() - t0

    chunks_time = timer.chunks_duration()
    dedup_time = timer.dedup_duration()

    deepseek_path = _write_deepseek_md(baseline, fixture)
    local_path = _write_local_md(result, fixture, MODEL, total_time, chunks_time, dedup_time)

    bar = "-" * 70
    print(f"\n{bar}\n  TIMING\n{bar}")
    print(f"  Messages:     {len(messages)}")
    print(f"  Local total:  {total_time:7.1f}s")
    if chunks_time is not None:
        mps = len(messages) / chunks_time if chunks_time > 0 else 0.0
        print(f"  Chunks:       {chunks_time:7.1f}s   ({mps:.2f} msgs/sec)")
    if dedup_time is not None:
        print(f"  Dedup:        {dedup_time:7.1f}s")

    ds_duration = baseline.get("duration_seconds")
    if ds_duration is not None:
        print(f"  DeepSeek:     {ds_duration:7.1f}s")
    else:
        print(
            "  DeepSeek:     — (delete deepseek_baseline.json and rerun "
            "collect_fixture.py to capture)"
        )

    print(f"\n  Events: local={len(result['events'])}  deepseek={len(baseline['events'])}")
    print(f"\n  Wrote: {local_path.relative_to(ROOT)}")
    print(f"  Wrote: {deepseek_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
