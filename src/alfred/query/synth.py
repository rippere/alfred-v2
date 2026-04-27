"""Synthesis backend chain: Anthropic → OpenRouter → Ollama."""
from __future__ import annotations

import os

import httpx

SYSTEM_PROMPT = (
    "You are Alfred, a personal knowledge assistant with access to a private vault "
    "of notes, projects, decisions, assumptions, conversations, and learnings. "
    "Answer the query using ONLY the provided vault context. Be specific and "
    "cite which vault documents support each point. If context is insufficient, say so."
)


def synthesize(
    query: str,
    context: str,
    anthropic_model: str,
    openrouter_model: str,
    ollama_base_url: str,
    ollama_model: str,
    preamble: str = "",
) -> tuple[str, str, str]:
    """Returns (answer, backend_label, model_label). Tries backends in order."""
    system = (preamble + "\n\n" + SYSTEM_PROMPT).strip() if preamble else SYSTEM_PROMPT

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _anthropic(query, context, system, anthropic_model), "Anthropic", anthropic_model
        except Exception as e:
            _warn(f"Anthropic: {e.__class__.__name__}, trying next...")

    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            return _openrouter(query, context, system, openrouter_model), "OpenRouter", openrouter_model
        except Exception as e:
            msg = str(e)
            reason = "insufficient credits" if "402" in msg or "credits" in msg.lower() else e.__class__.__name__
            _warn(f"OpenRouter: {reason} — falling back to Ollama")

    return _ollama(query, context, system, ollama_base_url, ollama_model), "Ollama (local)", ollama_model


def _anthropic(query: str, context: str, system: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": f"Query: {query}\n\nVault context:\n\n{context}"}],
    )
    return resp.content[0].text


def _openrouter(query: str, context: str, system: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Query: {query}\n\nVault context:\n\n{context}"},
        ],
    )
    return resp.choices[0].message.content or ""


def _ollama(query: str, context: str, system: str, base_url: str, model: str) -> str:
    resp = httpx.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Query: {query}\n\nVault context:\n\n{context}"},
            ],
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _warn(msg: str) -> None:
    from rich.console import Console
    Console().print(f"      [yellow]{msg}[/yellow]")
