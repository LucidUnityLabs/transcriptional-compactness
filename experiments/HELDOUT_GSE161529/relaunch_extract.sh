#!/bin/bash
# Relaunch wrapper for extract_labels.py — logs to extract.log, records exit reason.
cd "$(dirname "$0")" || exit 1
LOG=../../data/GSE161529/figshare_tmp/extract.log
echo "$(date) extract launch (pid $$)" >> "$LOG"
python3 extract_labels.py >> "$LOG" 2>&1
echo "$(date) extract exited rc=$?" >> "$LOG"
