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
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
#!/usr/bin/env python3
"""
Simple async load tester for the inference endpoint to help tune batching/queue depth.

Usage:
  python scripts/load_test_inference.py --url http://localhost:9000/predict --qps 50 --duration 30

Sends concurrent requests and reports achieved QPS and P50/P95/P99 latencies.
"""
import argparse
import asyncio
import json
import random
import time
from statistics import median

import aiohttp

async def worker(url, qps, stop_at):
    async with aiohttp.ClientSession() as sess:
        interval = 1.0 / qps if qps > 0 else 0
        latencies = []
        cnt = 0
        while time.time() < stop_at:
            payload = {"items":[{"id": str(random.randint(1,1000000)), "text": "hello world"}]}
            t0 = time.time()
            try:
                async with sess.post(url, json=payload, timeout=10) as r:
                    await r.text()
            except Exception:
                latencies.append(None)
            else:
                latencies.append(time.time() - t0)
            cnt += 1
            await asyncio.sleep(interval)
        return latencies, cnt

async def main(args):
    stop_at = time.time() + args.duration
    tasks = []
    # spawn N workers each at qps_per_worker
    qps_per_worker = args.qps / max(1, args.workers)
    for _ in range(args.workers):
        tasks.append(asyncio.create_task(worker(args.url, qps_per_worker, stop_at)))
    results = await asyncio.gather(*tasks)
    lat_all = []
    total = 0
    for lat, cnt in results:
        total += cnt
        lat_all.extend([l for l in lat if l is not None])
    lat_all.sort()
    def pct(p):
        if not lat_all:
            return None
        idx = int(len(lat_all) * p / 100)
        idx = max(0, min(idx, len(lat_all)-1))
        return lat_all[idx]
    print("Total requests:", total)
    print("P50:", pct(50), "P95:", pct(95), "P99:", pct(99))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--qps", type=int, default=10)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(main(args))
scripts/load_test_inference.py
