#!/usr/bin/env python3
"""
Simple canary traffic generator for canary tests.

Usage:
  python3 scripts/simulate_canary_load.py --url http://aegis-staging.example.com/infer --qps 20 --duration 60 --error-rate 0.05
"""
from __future__ import annotations
import argparse
import time
import random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
import httpx

def send_request(url: str, payload: dict, timeout: float = 5.0):
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
        return r.status_code
    except Exception:
        return 0

def make_payload(bad=False):
    if bad:
        return {"input": "MALFORMED"}
    # default synthetic example for image model
    return {"input": [[[[0.0]] * 1] * 28] * 28}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--qps", type=int, default=10)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--error-rate", type=float, default=0.0)
    args = ap.parse_args()

    end = time.time() + args.duration
    pool = ThreadPoolExecutor(max_workers=100)
    # See simulate_canary_load.py: harvest completed futures incrementally so
    # `pending` doesn't grow across the whole run for long/high-qps tests.
    pending = set()
    total = 0
    ok = 0

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
            status = f.result()
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
        time.sleep(1)
        harvest(block=False)

    while pending:
        harvest(block=True)
    pool.shutdown(wait=True)
    print(f"Sent {total} requests, successful: {ok}/{total}")

if __name__ == "__main__":
    main()
