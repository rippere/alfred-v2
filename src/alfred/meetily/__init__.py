"""Meetily ingestion — pull meeting transcripts + summaries from Meetily's
local SQLite store and drop them into Alfred's vault inbox as `type: meeting`
records.

Meetily (github.com/Zackriya-Solutions/meeting-minutes) is a local-first
meeting-notes tool: it captures mic + system audio, transcribes with
Whisper/Parakeet, and summarises with a pluggable LLM (Ollama/Claude/Groq/…).
Everything it produces is persisted to a local SQLite database with three
tables — `meetings`, `transcripts`, and `summary_processes`.

This package is a thin *reader*: Alfred never talks to Meetily's app, it only
reads that SQLite file. New meetings are rendered to markdown and written to
`<vault>/inbox/`, where the existing Curator daemon files them into `meeting/`.
From there the rest of Alfred's pipeline (surveyor embed → distiller learnings
→ consolidator project-linking → query/MCP) runs unchanged.

Design mirrors `alfred.ledger`: a Typer sub-app (`alfred meetily …`) plus an
optional systemd timer for periodic sync.
"""
