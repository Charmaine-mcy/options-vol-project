#!/bin/bash
# auto_push.sh — runs the pipeline, then commits and pushes any resulting
# changes to GitHub automatically. Meant to be called by cron instead of
# calling run_pipeline.py directly.

set -e  # stop on any unexpected error in this script itself

PROJECT_DIR="/Users/charmaine/options-vol-project"
LOG_FILE="$PROJECT_DIR/logs/auto_push.log"

export PATH="/usr/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"

cd "$PROJECT_DIR"

echo "=== auto_push run: $(date) ===" >> "$LOG_FILE"

.venv/bin/python run_pipeline.py >> "$LOG_FILE" 2>&1

if [[ -n $(git status --porcelain) ]]; then
    git add .
    git commit -m "Automated pipeline update: $(date '+%Y-%m-%d %H:%M %Z')" >> "$LOG_FILE" 2>&1
    git push >> "$LOG_FILE" 2>&1
    echo "Pushed changes to GitHub." >> "$LOG_FILE"
else
    echo "No changes to commit — skipping push." >> "$LOG_FILE"
fi

echo "=== auto_push run complete ===" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
