#!/usr/bin/env bash
# Claude Code PreCompact Hook
# Saves a context snapshot before compaction so critical state isn't lost
# Drops a checkpoint to Alfred inbox + writes HANDOFF.md (if git repo)

set -euo pipefail

CONFIG_FILE="$HOME/.config/claude/handoff-config.json"
VAULT_PATH=$(jq -r '.output.vault_path // empty' "$CONFIG_FILE" 2>/dev/null || echo "")
ALFRED_INBOX="${VAULT_PATH}/inbox"
HANDOFF_FILE="HANDOFF.md"

HOOK_INPUT=$(cat)
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null || echo "unknown")
TRIGGER=$(echo "$HOOK_INPUT" | jq -r '.trigger // "auto"' 2>/dev/null || echo "auto")
CWD=$(echo "$HOOK_INPUT" | jq -r '.cwd // empty' 2>/dev/null || echo "")

if [ -n "$CWD" ]; then
    PROJECT_DIR="$CWD"
elif [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
    PROJECT_DIR="$CLAUDE_PROJECT_DIR"
else
    PROJECT_DIR="$(pwd)"
fi

# Guard: skip Curator sub-sessions
if [ -n "${ALFRED_VAULT_SCOPE:-}" ] || [[ "$PROJECT_DIR" == */obsidian-vault* ]]; then
    exit 0
fi

log() { echo "[pre-compact] $*" >&2; }
timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
slugify() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//'; }

PROJECT_NAME=$(basename "$PROJECT_DIR")
NOW=$(timestamp)
SLUG=$(slugify "$PROJECT_NAME")

# Gather git state if available
IS_GIT=false
BRANCH="n/a"
DIFF_STAT=""
RECENT_COMMITS=""
DIRTY_FILES=""

if git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree &>/dev/null; then
    IS_GIT=true
    cd "$PROJECT_DIR"
    BRANCH=$(git branch --show-current 2>/dev/null || echo "detached")
    DIFF_STAT=$(git diff --stat 2>/dev/null || true)
    RECENT_COMMITS=$(git log --oneline -5 2>/dev/null || true)
    DIRTY_FILES=$(git status --short 2>/dev/null || true)
fi

# Write checkpoint HANDOFF.md only in git repos
if [ "$IS_GIT" = true ]; then
    cat > "$PROJECT_DIR/$HANDOFF_FILE" << HANDOFF
# Pre-Compaction Checkpoint

**Project:** $PROJECT_NAME
**Branch:** $BRANCH
**Session:** $SESSION_ID
**Trigger:** $TRIGGER compaction
**Timestamp:** $NOW

## Dirty Files (uncommitted)

\`\`\`
${DIRTY_FILES:-none}
\`\`\`

## Diff Summary

\`\`\`
${DIFF_STAT:-no changes}
\`\`\`

## Recent Commits

\`\`\`
${RECENT_COMMITS:-no git history}
\`\`\`

---
_Pre-compaction checkpoint. Read this to restore context after compaction._
HANDOFF

    log "Wrote $HANDOFF_FILE (compaction checkpoint)"
fi

# Always drop to Alfred inbox (if vault is configured)
if [ -n "$VAULT_PATH" ] && [ -d "$ALFRED_INBOX" ]; then
    INBOX_FILE="$ALFRED_INBOX/compact_${SLUG}_$(date +%Y%m%d_%H%M%S).md"

    if [ "$IS_GIT" = true ]; then
        CONTEXT_SECTION="## Uncommitted Work

\`\`\`
${DIRTY_FILES:-none}
\`\`\`

## Diff Summary

\`\`\`
${DIFF_STAT:-no changes}
\`\`\`

## Recent Commits

\`\`\`
${RECENT_COMMITS:-no git history}
\`\`\`"
    else
        CONTEXT_SECTION="## Context

_No git repository. Conversation-only session being compacted._
_Directory: ${PROJECT_DIR}_"
    fi

    cat > "$INBOX_FILE" << INBOX
# Context Compaction: $PROJECT_NAME

<!-- alfred:source claude_code_hook -->
<!-- alfred:source_id ${SESSION_ID}_compact -->
<!-- alfred:project $PROJECT_NAME -->
<!-- alfred:branch $BRANCH -->
<!-- alfred:timestamp $NOW -->
<!-- alfred:event pre_compact -->
<!-- alfred:trigger $TRIGGER -->

**Project:** $PROJECT_NAME
**Branch:** $BRANCH
**Session:** $SESSION_ID
**Event:** Pre-compaction ($TRIGGER)

$CONTEXT_SECTION

---
_Auto-generated pre-compaction snapshot. Curator: file as session checkpoint._
INBOX

    log "Dropped compaction snapshot to Alfred inbox: $INBOX_FILE"
fi

# Inject reminder into post-compaction context via stdout
echo "IMPORTANT: Context was just compacted. Read HANDOFF.md in project root for pre-compaction state."

exit 0
