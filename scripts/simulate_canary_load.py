#!/usr/bin/env python3
"""
Simple canary traffic generator.

- Sends requests to a model endpoint to simulate traffic.
- Optionally introduce a fraction of error requests (bad payloads) to simulate a bad canary.
- Useful for testing rollout alert/rollback rules.

Usage:
  python3 scripts/simulate_canary_load.py --url http://aegis-staging.example.com/infer --qps 50 --duration 60 --error-rate 0.05
"""
from __future__ import annotations
import argparse
import httpx
import time
import random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED

def send_request(url: str, payload: dict, timeout: float = 5.0):
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:
        return 0, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--qps", type=int, default=10)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--error-rate", type=float, default=0.0)
    args = ap.parse_args()

    def make_payload(bad=False):
        if bad:
            return {"input": "MALFORMED"}  # deliberately cause error in model
        # example happy path payload for image model: base64 or small synthetic
        return {"input": [[[[0.0]*1]*28]*28]}  # adjust to model input spec

    end = time.time() + args.duration
    pool = ThreadPoolExecutor(max_workers=100)
    # Only keep in-flight/not-yet-harvested futures in memory; completed ones
    # are tallied and discarded immediately below instead of accumulating the
    # full run's futures list (which can grow very large for long/high-qps
    # runs and needlessly hides client-side memory pressure behind the test).
    pending = set()
    sent = 0
    ok = 0
    total = 0

    def harvest(block: bool):
        nonlocal ok, total
        if not pending:
            return
        done, still_pending = wait(
            pending,
            timeout=None if block else 0,
            return_when=FIRST_COMPLETED if block else ALL_COMPLETED,
        )
        for f in done:
            status, _body = f.result()
            total += 1
            if 200 <= status < 300:
                ok += 1
        pending.clear()
        pending.update(still_pending)

    while time.time() < end:
        for _ in range(args.qps):
            bad = random.random() < args.error_rate
            payload = make_payload(bad=bad)
            pending.add(pool.submit(send_request, args.url, payload))
            sent += 1
        # sleep 1 second per QPS loop
        time.sleep(1)
        # drain any requests that already completed so `pending` doesn't grow
        # unbounded over a long run
        harvest(block=False)

    # drain whatever is left after the run ends
    while pending:
        harvest(block=True)
    pool.shutdown(wait=True)
    print(f"Sent {sent} requests, successful: {ok}/{total}")

if __name__ == "__main__":
    main()
