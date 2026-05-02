#!/usr/bin/env bash
# Claude Code SessionEnd Hook
# 1. Extracts conversation from session JSONL
# 2. Summarizes via local LLM (Ollama)
# 3. Auto-commits + pushes git changes
# 4. Drops rich handoff to Alfred's inbox

set -euo pipefail

# --- Config ---
CONFIG_FILE="$HOME/.config/claude/handoff-config.json"
VAULT_PATH=$(jq -r '.output.vault_path // empty' "$CONFIG_FILE" 2>/dev/null || echo "")
if [ -z "$VAULT_PATH" ]; then
    echo "[session-end] vault_path not configured in $CONFIG_FILE, skipping" >&2
    exit 0
fi
AI_DIALOGUE_DIR="$VAULT_PATH/ai-dialogue"
MIN_TURNS_FOR_CAPTURE=5
HANDOFF_FILE="HANDOFF.md"
SUMMARIZER="$HOME/.config/claude/hooks/summarize-session.py"

# --- Read hook input from stdin ---
HOOK_INPUT=$(cat)
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null || echo "unknown")
CWD=$(echo "$HOOK_INPUT" | jq -r '.cwd // empty' 2>/dev/null || echo "")

# Use CWD from hook input, fall back to env
if [ -n "$CWD" ]; then
    PROJECT_DIR="$CWD"
elif [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
    PROJECT_DIR="$CLAUDE_PROJECT_DIR"
else
    PROJECT_DIR="$(pwd)"
fi

# --- Guard: skip Curator sub-sessions (prevents feedback loop) ---
if [ -n "${ALFRED_VAULT_SCOPE:-}" ]; then
    echo "[session-end] Skipping — this is a Curator sub-session (ALFRED_VAULT_SCOPE=$ALFRED_VAULT_SCOPE)" >&2
    exit 0
fi
# Also skip if CWD is inside the vault itself
if [[ "$PROJECT_DIR" == */obsidian-vault* ]]; then
    echo "[session-end] Skipping — CWD is inside the vault" >&2
    exit 0
fi

# --- Helpers ---
log() { echo "[session-end] $*" >&2; }
timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
slugify() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//'; }

PROJECT_NAME=$(basename "$PROJECT_DIR")
NOW=$(timestamp)
SLUG=$(slugify "$PROJECT_NAME")

# --- Find session JSONL ---
TURN_COUNT=0
CLAUDE_PROJECTS_DIR="$HOME/.claude/projects"
PROJECT_SLUG=$(echo "$PROJECT_DIR" | sed 's|/|-|g')
SESSION_JSONL=""

if [ -n "$SESSION_ID" ] && [ "$SESSION_ID" != "unknown" ]; then
    CANDIDATE="$CLAUDE_PROJECTS_DIR/$PROJECT_SLUG/$SESSION_ID.jsonl"
    if [ -f "$CANDIDATE" ]; then
        SESSION_JSONL="$CANDIDATE"
    fi
fi

# Fallback: most recently modified JSONL
if [ -z "$SESSION_JSONL" ] && [ -d "$CLAUDE_PROJECTS_DIR/$PROJECT_SLUG" ]; then
    SESSION_JSONL=$(find "$CLAUDE_PROJECTS_DIR/$PROJECT_SLUG" -maxdepth 1 -name "*.jsonl" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
fi

if [ -n "$SESSION_JSONL" ] && [ -f "$SESSION_JSONL" ]; then
    TURN_COUNT=$(grep -c '"role":"user"' "$SESSION_JSONL" 2>/dev/null || echo "0")
    log "Found session JSONL: $SESSION_JSONL ($TURN_COUNT user turns)"
fi

# --- Summarize conversation ---
SUMMARY=""
if [ -n "$SESSION_JSONL" ] && [ -f "$SESSION_JSONL" ] && [ -f "$SUMMARIZER" ]; then
    log "Summarizing conversation via LLM..."
    SUMMARY=$(python3 "$SUMMARIZER" "$SESSION_JSONL" 2>/dev/null || true)
    if [ -z "$SUMMARY" ]; then
        log "LLM summary failed, falling back to raw extract"
        SUMMARY=$(python3 "$SUMMARIZER" "$SESSION_JSONL" --raw-fallback 2>/dev/null || echo "_Could not extract conversation._")
    fi
else
    SUMMARY="_No session data available for summarization._"
fi

# --- Detect git repo ---
IS_GIT=false
BRANCH="n/a"
DIFF_STAT=""
STAGED_STAT=""
UNTRACKED=""
RECENT_COMMITS=""
HAS_CHANGES=false
FILES_CHANGED=""
DIFF_SUMMARY=""
REMOTE_NAME=""

if git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree &>/dev/null; then
    IS_GIT=true
    cd "$PROJECT_DIR"
    BRANCH=$(git branch --show-current 2>/dev/null || echo "detached")
    DIFF_STAT=$(git diff --stat HEAD 2>/dev/null || true)
    STAGED_STAT=$(git diff --cached --stat 2>/dev/null || true)
    UNTRACKED=$(git ls-files --others --exclude-standard 2>/dev/null || true)
    RECENT_COMMITS=$(git log --oneline -5 2>/dev/null || true)
    REMOTE_NAME=$(git remote 2>/dev/null | head -1 || true)

    if [ -n "$DIFF_STAT" ] || [ -n "$STAGED_STAT" ] || [ -n "$UNTRACKED" ]; then
        HAS_CHANGES=true
    fi
fi

# --- Gate: skip truly trivial sessions ---
if [ "$HAS_CHANGES" = false ] && [ "$TURN_COUNT" -lt "$MIN_TURNS_FOR_CAPTURE" ]; then
    log "Trivial session (no changes, $TURN_COUNT turns), skipping"
    exit 0
fi

# --- Secret scanning before any git operations ---
secret_scan() {
    local dirty_files
    dirty_files=$(git diff --cached --name-only 2>/dev/null || true)
    if [ -z "$dirty_files" ]; then
        return 0
    fi

    local found_secrets=false
    local secret_report=""

    while IFS= read -r file; do
        [ -z "$file" ] && continue
        file -b --mime "$file" 2>/dev/null | grep -q "text/" || continue

        local matches
        matches=$(grep -nEi \
            '(ANTHROPIC_API_KEY|OPENAI_API_KEY|CANVAS_TOKEN|NOTION_TOKEN|NOTION_DATABASE_ID|GRAFANA_ADMIN_PASSWORD|LANGFUSE_|NEXTAUTH_SECRET|SALT)=' \
            "$file" 2>/dev/null || true)

        local generic
        generic=$(grep -nE \
            '(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36}|xox[bpsa]-[a-zA-Z0-9-]{10,}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA |EC |DSA )?PRIVATE KEY)' \
            "$file" 2>/dev/null || true)

        local assignments
        assignments=$(grep -nEi \
            '(password|secret|token|api_key|apikey|auth_key)\s*[:=]\s*["\x27]?[a-zA-Z0-9/+]{8,}' \
            "$file" 2>/dev/null | grep -viE '(example|placeholder|your_|changeme|xxx|TODO)' || true)

        if [ -n "$matches" ] || [ -n "$generic" ] || [ -n "$assignments" ]; then
            found_secrets=true
            secret_report+="  BLOCKED: $file"$'\n'
            [ -n "$matches" ] && secret_report+="    known keys: $(echo "$matches" | head -3)"$'\n'
            [ -n "$generic" ] && secret_report+="    generic secrets: $(echo "$generic" | head -3)"$'\n'
            [ -n "$assignments" ] && secret_report+="    credential assignments: $(echo "$assignments" | head -3)"$'\n'
        fi
    done <<< "$dirty_files"

    if [ "$found_secrets" = true ]; then
        log "SECRET SCAN FAILED — blocking commit + push"
        log "$secret_report"
        echo "$secret_report"
        return 1
    fi
    return 0
}

# --- Git: auto-commit + push ---
if [ "$IS_GIT" = true ] && [ "$HAS_CHANGES" = true ]; then
    if [[ "$BRANCH" =~ ^(main|master)$ ]]; then
        log "On $BRANCH, skipping auto-commit (stage and commit manually)"
    else
        git add -A

        SCAN_RESULT=""
        if ! SCAN_RESULT=$(secret_scan); then
            log "SECRETS DETECTED — aborting commit + push:"
            log "$SCAN_RESULT"
            git reset HEAD 2>/dev/null || true
        else
            COMMIT_MSG="wip: auto-save from Claude Code session

Session: $SESSION_ID
Branch: $BRANCH
Timestamp: $NOW

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

            git commit -m "$COMMIT_MSG" --no-verify 2>/dev/null || true

            if [ -n "$REMOTE_NAME" ] && [ -n "$BRANCH" ] && [ "$BRANCH" != "detached" ]; then
                if git rev-parse --abbrev-ref --symbolic-full-name @{u} &>/dev/null; then
                    git push 2>/dev/null || log "Push failed (may need manual push)"
                else
                    git push -u "$REMOTE_NAME" "$BRANCH" 2>/dev/null || log "Push with -u failed (may need manual push)"
                fi
            fi
        fi
    fi

    FILES_CHANGED=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || echo "(no commit to diff)")
    DIFF_SUMMARY=$(git diff --stat HEAD~1 HEAD 2>/dev/null || echo "")
fi

# --- Build git section (only if relevant) ---
GIT_SECTION=""
if [ "$IS_GIT" = true ]; then
    GIT_SECTION="## Git State

**Branch:** $BRANCH

### Changes
\`\`\`
${DIFF_SUMMARY:-no changes}
\`\`\`

### Files Touched
\`\`\`
${FILES_CHANGED:-none}
\`\`\`

### Recent Commits
\`\`\`
${RECENT_COMMITS:-none}
\`\`\`"
fi

# --- Write HANDOFF.md to project dir (git repos only) ---
if [ "$IS_GIT" = true ]; then
    cat > "$PROJECT_DIR/$HANDOFF_FILE" << HANDOFF
# Session Handoff

**Project:** $PROJECT_NAME
**Branch:** $BRANCH
**Session:** $SESSION_ID
**Timestamp:** $NOW

## Session Summary

$SUMMARY

$GIT_SECTION

HANDOFF

    log "Wrote $HANDOFF_FILE"
fi

# --- Map project name to vault project link ---
DATE_SHORT=$(date +%Y-%m-%d)
PROJECT_LINK=""
for candidate in "$VAULT_PATH/project/"*.md; do
    [ -f "$candidate" ] || continue
    candidate_name=$(basename "$candidate" .md)
    if echo "$candidate_name" | grep -qi "$(echo "$PROJECT_NAME" | sed 's/-/ /g')"; then
        PROJECT_LINK="[[project/$candidate_name]]"
        break
    fi
done

# --- Write to inbox/ for Alfred Curator ---
INBOX_DIR="$VAULT_PATH/inbox"
if [ -d "$INBOX_DIR" ]; then
    INBOX_FILE="$INBOX_DIR/session-${SLUG}-${DATE_SHORT}-${SESSION_ID:0:8}.md"
    cat > "$INBOX_FILE" << INBOX
---
type: session
status: active
project: "${PROJECT_LINK:-}"
session_id: "$SESSION_ID"
turns: $TURN_COUNT
created: "$DATE_SHORT"
tags: []
<!-- alfred:source claude_code_hook -->
---

# $PROJECT_NAME — $DATE_SHORT

$SUMMARY

$GIT_SECTION
INBOX
    log "Wrote session record to inbox: $(basename "$INBOX_FILE")"
else
    log "inbox/ not found at $INBOX_DIR, skipping Alfred drop"
fi

# --- Write raw archive to ai-dialogue/ ---
if [ -d "$AI_DIALOGUE_DIR" ]; then
    DIALOGUE_FILE="$AI_DIALOGUE_DIR/${PROJECT_NAME} ${DATE_SHORT} ${SESSION_ID:0:8}.md"
    cat > "$DIALOGUE_FILE" << DIALOGUE
---
type: session
status: active
project: "${PROJECT_LINK:-}"
session_id: "$SESSION_ID"
turns: $TURN_COUNT
created: "$DATE_SHORT"
---

# $PROJECT_NAME — $DATE_SHORT (${SESSION_ID:0:8})

$SUMMARY

$GIT_SECTION
DIALOGUE
    log "Wrote raw archive to ai-dialogue: $(basename "$DIALOGUE_FILE")"
fi

exit 0
