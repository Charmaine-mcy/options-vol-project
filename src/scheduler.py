"""
Recurring option-chain snapshots.  Two ways to run it:

(a) Long-running loop (uses the `schedule` library; stop with Ctrl-C):

        python -m src.scheduler --loop --every 30 --ticker SPY

(b) One pull per invocation, driven by cron / Task Scheduler:

        python -m src.scheduler --once --ticker SPY

    --once checks the US market clock first and exits quietly if the market
    is closed. That keeps the cron line trivial — schedule it every 30
    minutes around the clock and let the script decide:

        */30 * * * * cd "/path/to/options-vol-project" && .venv/bin/python -m src.scheduler --once --ticker SPY >> data/snapshots/cron.log 2>&1

    (crontab -e, paste the line above; quote the path if it contains
    spaces.) macOS caveats worth knowing:
      - cron jobs don't fire while the laptop is asleep; there is no catch-up.
      - macOS privacy (TCC) can block cron from reading ~/Desktop; if the log
        shows "Operation not permitted", grant /usr/sbin/cron Full Disk
        Access in System Settings, or move the project out of Desktop.

Use --force to pull despite a closed market (the pipeline handles overnight
lastPrice data, but back-to-back closed-market snapshots are duplicates —
the underlying quotes don't change until the next session).

Snapshots are immutable timestamped files in data/snapshots/ (see
data_fetch.save_snapshot); this module never deletes or rewrites them.
"""

import argparse
import sys
import time
from datetime import datetime

import pandas as pd

from src.data_fetch import save_snapshot, NY


def market_is_open(now=None):
    """
    True during regular US equity hours: Mon-Fri 9:30-16:00 New York time.
    Deliberately simple — it does not know exchange holidays, so a holiday
    pull just produces one extra stale-quote snapshot, which the loader's
    stale_quotes flag identifies anyway.
    """
    now = now or pd.Timestamp.now(tz=NY)
    if now.weekday() >= 5:                      # Saturday/Sunday
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


def pull_once(ticker, force=False):
    """One guarded snapshot. Returns True if a snapshot was taken."""
    now = pd.Timestamp.now(tz=NY)
    if not market_is_open(now) and not force:
        print(f"[{now:%Y-%m-%d %H:%M %Z}] market closed — skipping "
              f"(use --force to pull anyway)")
        return False
    try:
        save_snapshot(ticker)
        return True
    except Exception as exc:
        # Never let one bad pull kill a long-running loop / cron history.
        print(f"[{now:%Y-%m-%d %H:%M %Z}] pull failed: {exc}", file=sys.stderr)
        return False


def run_loop(ticker, every_minutes, force=False, max_pulls=0):
    """
    Pull immediately, then every `every_minutes`. max_pulls > 0 stops after
    that many successful pulls (handy for testing); 0 means run until killed.
    """
    import schedule

    done = 0

    def job():
        nonlocal done
        if pull_once(ticker, force=force):
            done += 1

    print(f"snapshot loop: {ticker} every {every_minutes} min "
          f"(market-hours guard {'OFF' if force else 'on'}; Ctrl-C to stop)")
    job()
    schedule.every(every_minutes).minutes.do(job)
    try:
        while max_pulls == 0 or done < max_pulls:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopped.")
    print(f"loop finished: {done} snapshot(s) taken")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true",
                      help="single guarded pull (for cron/Task Scheduler)")
    mode.add_argument("--loop", action="store_true",
                      help="run forever, pulling on a cadence")
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--every", type=int, default=30, metavar="MIN",
                    help="loop cadence in minutes (default 30)")
    ap.add_argument("--force", action="store_true",
                    help="pull even when the US market is closed")
    ap.add_argument("--max-pulls", type=int, default=0, metavar="N",
                    help="loop mode: stop after N snapshots (0 = never)")
    args = ap.parse_args()

    if args.once:
        pull_once(args.ticker.upper(), force=args.force)
    else:
        run_loop(args.ticker.upper(), args.every, force=args.force,
                 max_pulls=args.max_pulls)


if __name__ == "__main__":
    main()
