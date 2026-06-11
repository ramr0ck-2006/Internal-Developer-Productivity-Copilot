import csv
import os
from datetime import datetime

FEEDBACK_FILE = "feedback_log.csv"

def log_feedback(query, answer, sources, rating):
    """Appends a feedback entry to the CSV file."""
    exists = os.path.isfile(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "query", "answer", "sources", "rating"])
        writer.writerow([
            datetime.now().isoformat(),
            query,
            answer[:200],               # truncate to avoid huge cells
            "|".join(sources),
            rating
        ])