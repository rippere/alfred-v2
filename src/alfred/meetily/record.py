"""Render a Meetily `Meeting` into an Alfred `type: meeting` inbox record.

Field mapping (Meetily → Alfred frontmatter):

    meetings.title        → name
    meetings.created_at   → created / meeting_date
    transcripts.duration  → duration_min
    summary_processes     → status (summarized vs captured)
    meetings.id           → meetily_id   (dedup key; never re-ingested twice)

The Curator daemon already short-circuits LLM classification when a note
declares a known `type` in its frontmatter, so writing `type: meeting`
explicitly means these land in `meeting/` deterministically and for free.
"""
from __future__ import annotations

import re
from datetime import date

import frontmatter

from alfred.meetily.reader import Meeting

# Keep the embedded transcript bounded — full transcripts can be huge and would
# dominate the embedding corpus. The summary is what you query on; the raw
# transcript is kept truncated with a pointer back to Meetily for the full text.
_TRANSCRIPT_MAX_CHARS = 6000


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:80] or "untitled"


def inbox_filename(m: Meeting) -> str:
    """Stable, collision-resistant inbox filename for a meeting."""
    return f"meetily-{_slugify(m.title)}-{m.id[:8]}.md"


def _iso_date(ts: str) -> str:
    """Best-effort YYYY-MM-DD from Meetily's created_at (ISO-ish) string."""
    if ts and len(ts) >= 10 and ts[4] == "-" and ts[7] == "-":
        return ts[:10]
    return date.today().isoformat()


def _render_structured(struct: dict) -> str:
    """Render Meetily's structured summary JSON as markdown sections.

    Meetily's `result` is a dict of {SectionName: block-or-list}. Its exact keys
    vary by version, so we render generically: dicts → subsections, lists → bullets.
    """
    out: list[str] = []
    for key, val in struct.items():
        if key.lower() in ("meetingname", "title"):
            continue
        heading = re.sub(r"(?<!^)(?=[A-Z])", " ", str(key)).strip()  # CamelCase → words
        out.append(f"## {heading}")
        out.append(_render_block(val))
    return "\n\n".join(out)


def _render_block(val) -> str:
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        lines = []
        for item in val:
            if isinstance(item, dict):
                text = item.get("content") or item.get("text") or item.get("title") or str(item)
                lines.append(f"- {str(text).strip()}")
            else:
                lines.append(f"- {str(item).strip()}")
        return "\n".join(lines)
    if isinstance(val, dict):
        # {title, blocks:[...]} shape used by some Meetily versions
        if "blocks" in val:
            return _render_block(val["blocks"])
        return "\n".join(f"- **{k}**: {v}" for k, v in val.items())
    return str(val)


def build_body(m: Meeting) -> str:
    """Assemble the markdown body from Meetily's summary + transcript."""
    sections: list[str] = [f"# {m.title}"]

    if m.structured_summary:
        sections.append(_render_structured(m.structured_summary))
    else:
        if m.summary:
            sections.append(f"## Summary\n\n{m.summary}")
        if m.key_points:
            sections.append(f"## Key Points\n\n{m.key_points}")
        if m.action_items:
            sections.append(f"## Action Items\n\n{m.action_items}")

    if m.transcript:
        clipped = m.transcript[:_TRANSCRIPT_MAX_CHARS]
        truncated = len(m.transcript) > _TRANSCRIPT_MAX_CHARS
        note = "\n\n*(transcript truncated — full text in Meetily)*" if truncated else ""
        sections.append(f"## Transcript\n\n{clipped}{note}")

    return "\n\n".join(s for s in sections if s.strip()) + "\n"


def to_markdown(m: Meeting) -> str:
    """Full inbox note (frontmatter + body) as a string ready to write."""
    status = "summarized" if (m.structured_summary or m.summary) else "captured"
    fm: dict = {
        "type": "meeting",
        "name": m.title,
        "created": _iso_date(m.created_at),
        "meeting_date": _iso_date(m.created_at),
        "status": status,
        "source": "meetily",
        "meetily_id": m.id,
        "tags": ["meeting"],
    }
    if m.duration_s:
        fm["duration_min"] = round(m.duration_s / 60.0, 1)

    post = frontmatter.Post(build_body(m), **fm)
    return frontmatter.dumps(post) + "\n"
