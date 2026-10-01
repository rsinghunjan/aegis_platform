  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
"""
Simple CLI helper to export and escalate old pending items.

Example uses:
- Run periodically (cron) to export pending items older than N days to a secure bucket / send email to reviewers.
"""
import argparse
import logging
import time

from . import db as review_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_export(cutoff_days: int = 7, out_path: str = "logs/pending_export.json"):
    cutoff_ts = time.time() - cutoff_days * 24 * 3600
    items = review_db.list_pending(limit=10000)
    old_items = [i for i in items if i["created_at"] < cutoff_ts]
    if not old_items:
        logger.info("No old pending items")
        return None
    review_db.export_pending(out_path=out_path)
    logger.info("Exported %d old items to %s", len(old_items), out_path)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff-days", type=int, default=7)
    parser.add_argument("--out", type=str, default="logs/pending_export.json")
    args = parser.parse_args()
    run_export(cutoff_days=args.cutoff_days, out_path=args.out)

if __name__ == "__main__":
    main()
