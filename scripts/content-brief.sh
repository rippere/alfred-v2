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

# Query Alfred HTTP API for context across all vaults
CONTEXT=$(curl -s http://localhost:8765/query \
    -H 'Content-Type: application/json' \
    -d "{\"query\": \"$TOPIC\", \"top_k\": 10, \"include_synthesis\": false}" \
    2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
passages = []
for hit in data.get('hits', []):
    path = hit.get('rel_path', '')
    preview = hit.get('preview', hit.get('text', ''))[:400]
    if preview:
        passages.append(f'[{path}]\n{preview}')
print('\n\n---\n\n'.join(passages[:8]))
" 2>/dev/null || echo "")

if [[ -z "$CONTEXT" ]]; then
    echo "Warning: Alfred returned no context. Is the daemon running? Continuing with topic only."
fi

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

# Generate brief via Claude
BRIEF=$(python3 - <<PYEOF
import anthropic, os, sys

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

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

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1200,
    system=system,
    messages=[{"role": "user", "content": prompt}]
)
print(resp.content[0].text)
PYEOF
)

if [[ -z "$BRIEF" ]]; then
    echo "Error: Claude API call failed. Check ANTHROPIC_API_KEY."
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
