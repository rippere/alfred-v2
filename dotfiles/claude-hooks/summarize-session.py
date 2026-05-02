#!/usr/bin/env python3
"""Extract conversation from a Claude Code session JSONL and summarize via Ollama Mistral.

Usage: summarize-session.py <session.jsonl> [--raw-fallback]

Outputs structured markdown summary (Goals, Key Decisions, etc.) to stdout.
If Ollama fails or --raw-fallback is set, outputs truncated raw conversation instead.
"""

import json
import subprocess
import sys
from pathlib import Path

OLLAMA_MODEL = "mistral:latest"
OLLAMA_TIMEOUT = 120  # seconds (mistral is larger, needs more time)
MAX_CONTEXT_CHARS = 16000  # truncate conversation fed to LLM
MAX_RAW_CHARS = 4000  # fallback raw dump limit

SUMMARY_PROMPT = """You are summarizing a software engineering conversation between a human and an AI assistant (Claude Code). Produce a structured summary in markdown with these sections:

## Goals
What the human wanted to accomplish (bullet points)

## Key Decisions
Important choices made during the session (bullet points)

## What Was Done
Concrete actions taken — files created/edited, commands run, architecture decisions (bullet points)

## Current State
Where things stand at the end of the session

## Blockers / Open Questions
Anything unresolved or needing follow-up

## Next Steps
What should happen next session

Be concise. Focus on substance — skip greetings, tool mechanics, and routine back-and-forth. If the conversation is thin, say so briefly rather than padding.

Here is the conversation:

"""


def extract_conversation(jsonl_path: str) -> list[dict]:
    """Pull user and assistant text messages from session JSONL."""
    messages = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            msg = obj.get("message", {})
            content = msg.get("content", "")

            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            parts.append(f"[tool: {block.get('name', '?')}]")
                    elif isinstance(block, str):
                        parts.append(block)
                text = "\n".join(parts).strip()
            else:
                continue

            if not text:
                continue

            role = "Human" if msg_type == "user" else "Assistant"
            messages.append({"role": role, "text": text})

    return messages


def format_conversation(messages: list[dict], max_chars: int) -> str:
    """Format messages into a readable conversation, truncated to max_chars."""
    lines = []
    total = 0
    for msg in messages:
        if msg["role"] == "Human" and msg["text"].startswith("[tool_result]"):
            continue
        line = f"**{msg['role']}:** {msg['text']}"
        if total + len(line) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                lines.append(line[:remaining] + "...[truncated]")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n\n".join(lines)


def summarize_with_ollama(conversation: str) -> str | None:
    """Send conversation to Ollama for summarization. Returns None on failure."""
    prompt = SUMMARY_PROMPT + conversation
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=OLLAMA_TIMEOUT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: summarize-session.py <session.jsonl>", file=sys.stderr)
        sys.exit(1)

    jsonl_path = sys.argv[1]
    raw_fallback = "--raw-fallback" in sys.argv

    if not Path(jsonl_path).exists():
        print(f"Session file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    messages = extract_conversation(jsonl_path)
    if not messages:
        print("_No conversation content found in session._")
        sys.exit(0)

    conversation = format_conversation(messages, MAX_CONTEXT_CHARS)

    if raw_fallback:
        print(format_conversation(messages, MAX_RAW_CHARS))
        sys.exit(0)

    summary = summarize_with_ollama(conversation)
    if summary:
        print(summary)
    else:
        print("_LLM summary unavailable. Raw conversation extract:_\n")
        print(format_conversation(messages, MAX_RAW_CHARS))


if __name__ == "__main__":
    main()
