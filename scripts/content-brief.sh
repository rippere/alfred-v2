#!/usr/bin/env bash
# content-brief.sh — Generate a content creation brief from Alfred's knowledge base
#
# Usage:
#   ./scripts/content-brief.sh "dopamine and habit formation"
#   ./scripts/content-brief.sh "why AI agents fail" --platform reels
#   ./scripts/content-brief.sh "behavioral finance traps" --platform thread
#
# Platforms: reels (default), thread, carousel, newsletter

set -euo pipefail

TOPIC="${1:-}"
PLATFORM="reels"

# Parse flags
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "$TOPIC" ]]; then
    echo "Usage: content-brief.sh \"your topic\" [--platform reels|thread|carousel|newsletter]"
    exit 1
fi

VAULT_CONTENT="/mnt/external/vault-personal/content"
BRIEFS_DIR="$VAULT_CONTENT/briefs"
DATE=$(date +%Y-%m-%d)
SLUG=$(echo "$TOPIC" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | cut -c1-60)
OUTPUT="$BRIEFS_DIR/${DATE}-${SLUG}.md"

if [[ ! -d "$BRIEFS_DIR" ]]; then
    mkdir -p "$BRIEFS_DIR"
fi

echo "Querying Alfred for: $TOPIC"
echo "Platform: $PLATFORM"
echo ""

# Query Alfred for context.
#
# This used to POST to http://localhost:8765/query, which had two separate
# problems: there has never been a /query route (the server speaks MCP JSON-RPC
# at /mcp), and since the tailnet-only bind there is nothing listening on
# loopback at all. Both failures were invisible — curl's output went to
# /dev/null and the fallback was an empty string.
#
# The CLI needs no HTTP, no auth and no network: this script runs on the same
# machine as the vault.
ALFRED_BIN="${ALFRED_BIN:-$(dirname "$0")/../.venv/bin/alfred}"

CONTEXT=$("$ALFRED_BIN" query "$TOPIC" -k 10 --json 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
except ValueError:
    sys.exit(1)
if data.get('error'):
    print('ALFRED_ERROR:' + data['error'], file=sys.stderr)
    sys.exit(1)
passages = []
for hit in data.get('hits', []):
    preview = (hit.get('preview') or '')[:400]
    if preview:
        passages.append(f\"[{hit.get('rel_path', '')}]\n{preview}\")
print('\n\n---\n\n'.join(passages[:8]))
" 2>/tmp/alfred-brief-err.$$) || true

if [[ -z "$CONTEXT" ]]; then
    # Say which of the two it was. "No context" and "the backend is down" call
    # for completely different responses from whoever is reading this.
    if grep -q ALFRED_ERROR /tmp/alfred-brief-err.$$ 2>/dev/null; then
        echo "Warning: Alfred query failed — $(sed 's/^ALFRED_ERROR://' /tmp/alfred-brief-err.$$)"
        echo "         (ollama-game-guard stops ollama while a game is running.)"
    else
        echo "Warning: Alfred returned no matching context for this topic."
    fi
    echo "         Continuing with topic only."
fi
rm -f /tmp/alfred-brief-err.$$

# Platform-specific format instructions
case "$PLATFORM" in
    reels)
        FORMAT_HINT="Instagram Reel (60-90 seconds spoken). Hook in first 3 seconds. One core insight. CTA at end."
        ;;
    thread)
        FORMAT_HINT="Twitter/X thread. Tweet 1 = hook. Tweets 2-8 = one point each. Final tweet = CTA + summary."
        ;;
    carousel)
        FORMAT_HINT="Instagram carousel (7-10 slides). Slide 1 = hook. Each slide = one sentence insight. Last slide = CTA."
        ;;
    newsletter)
        FORMAT_HINT="Newsletter section (300-500 words). Opening hook, research context, practical takeaway, one question to leave reader with."
        ;;
    *)
        FORMAT_HINT="General content piece."
        ;;
esac

# Generate the brief on the local model, same backend as the rest of Alfred.
# The Anthropic call that used to live here shared the key whose exhausted
# credit balance stalled ingestion; it also ran under the system python3,
# which has no `anthropic` module installed, so it could only ever have failed.
BRIEF=$("$(dirname "$0")/../.venv/bin/python" - <<PYEOF
import sys

from alfred.config import AlfredConfig
from alfred.core.local_llm import LocalLLMUnavailable, complete

cfg = AlfredConfig.load("$(dirname "$0")/../config.yaml")

topic = """$TOPIC"""
platform_hint = """$FORMAT_HINT"""
context = """$CONTEXT"""

system = """You are a content strategist for a creator in the neuroscience × AI × behavioral finance niche.
Your audience: knowledge workers, ambitious professionals, people who want to understand their own minds.
Your creator's edge: real projects (alfred-v2 personal AI, EMM, tribe-social), academic neuroscience background, practical AI systems experience.
Tone: insightful but accessible. Smart but not academic. Real but not bro-ish."""

prompt = f"""Create a content brief for this topic: {topic}
Platform format: {platform_hint}

Context from the creator's knowledge base:
---
{context[:3000] if context else "(no prior knowledge on this topic — use general expertise)"}
---

Write a content brief with these sections:

## Core Insight
One sentence that captures the main point (this is what the content delivers)

## Why This Matters to My Audience
2-3 sentences on why this hits for knowledge workers / ambitious people

## Hook Options (write 3)
Three different opening lines for {platform_hint.split('.')[0]}. Each should grab attention differently: curiosity gap / counter-intuitive / personal story.

## Content Outline
The actual structure: what goes where, in what order. Be specific.

## Evidence / Examples I Can Use
Pull from the knowledge base context above. List 2-3 concrete things to reference.

## CTA
What do I want them to do at the end? (comment, save, DM, follow)

## Related Topics to Mine Next
3 adjacent topics that connect naturally from this one"""

try:
    print(complete(
        system, prompt,
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_llm_model,
        max_tokens=1200,
    ))
except LocalLLMUnavailable as e:
    print(f"BACKEND_UNAVAILABLE: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
)

if [[ -z "$BRIEF" ]]; then
    echo "Error: could not generate the brief — the local model is unreachable."
    echo "       Check: systemctl is-active ollama"
    echo "       (ollama-game-guard stops it while a game is running.)"
    exit 1
fi

# Write the brief to vault-content/briefs/
cat > "$OUTPUT" <<MDEOF
---
type: note
status: draft
topic: $TOPIC
platform: $PLATFORM
created: $DATE
tags: [content-brief, $PLATFORM, content]
---

# Content Brief: $TOPIC

*Platform: $PLATFORM | Generated: $DATE*

$BRIEF
MDEOF

echo ""
echo "Brief written to: $OUTPUT"
echo ""
echo "Next steps:"
echo "  1. Review and adjust the brief in Obsidian"
echo "  2. Move to vault-content/drafts/ when scripting"
echo "  3. Move to vault-content/published/ after posting"
echo "  4. Log engagement in vault-content/analytics/"
