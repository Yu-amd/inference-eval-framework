"""
soak_test.py — 24-hour reliability soak test for Mangoboost
Sends requests at a steady rate and records error rates, latency drift, and availability.

Usage:
    python soak_test.py --base-url https://api.mangoboost.ai/v1 \
                        --api-key $MANGOBOOST_API_KEY \
                        --model <model-id> \
                        --rps 2 \
                        --duration-hours 24
"""

import argparse
import json
import signal
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROMPT = "What is the capital of France? Answer in one word."
_stop = threading.Event()


def _signal_handler(sig, frame):
    print("\nInterrupt received — stopping soak test...")
    _stop.set()


signal.signal(signal.SIGINT, _signal_handler)


def send_one(base_url: str, api_key: str, model: str) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 20,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30, verify=False)
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": resp.status_code,
            "latency_ms": round(latency_ms, 1),
            "ok": resp.status_code == 200,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": 0,
            "latency_ms": round(latency_ms, 1),
            "ok": False,
            "error": str(exc),
        }


def run_soak(base_url, api_key, model, rps, duration_hours, output_path):
    interval = 1.0 / rps
    end_time = time.time() + duration_hours * 3600
    results = []
    hourly_buckets: dict[int, list] = defaultdict(list)
    request_count = 0
    error_count = 0

    print(f"Starting soak: {rps} RPS for {duration_hours}h | target ~{int(rps * duration_hours * 3600):,} requests")

    while not _stop.is_set() and time.time() < end_time:
        t_loop = time.time()
        result = send_one(base_url, api_key, model)
        results.append(result)
        request_count += 1
        if not result["ok"]:
            error_count += 1
        hour_bucket = request_count // (rps * 3600)
        hourly_buckets[int(hour_bucket)].append(result["latency_ms"])

        # Print progress every 100 requests
        if request_count % 100 == 0:
            error_rate = error_count / request_count * 100
            recent = [r["latency_ms"] for r in results[-100:] if r["ok"]]
            avg_lat = sum(recent) / len(recent) if recent else 0
            elapsed_h = (time.time() - (end_time - duration_hours * 3600)) / 3600
            print(f"  [{elapsed_h:.1f}h] requests={request_count:,}  errors={error_rate:.2f}%  avg_latency={avg_lat:.0f}ms")

        sleep_time = interval - (time.time() - t_loop)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Write results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    availability = (request_count - error_count) / request_count * 100 if request_count else 0
    all_latencies = [r["latency_ms"] for r in results if r["ok"]]

    def pct(data, p):
        if not data:
            return None
        s = sorted(data)
        return s[int(len(s) * p / 100)]

    summary = {
        "total_requests": request_count,
        "successful": request_count - error_count,
        "errors": error_count,
        "availability_pct": round(availability, 4),
        "error_rate_pct": round(100 - availability, 4),
        "latency_ms": {
            "p50": pct(all_latencies, 50),
            "p95": pct(all_latencies, 95),
            "p99": pct(all_latencies, 99),
        },
        "hourly_avg_latency": {
            str(h): round(sum(lats) / len(lats), 1) if lats else None
            for h, lats in sorted(hourly_buckets.items())
        },
    }

    with open(output_path, "w") as f:
        json.dump({"summary": summary, "raw_samples": results[-1000:]}, f, indent=2)

    print(f"\nSoak complete — availability: {availability:.3f}%  errors: {error_count}")
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rps", type=float, default=2.0, help="Requests per second")
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--output", default="results/reliability/soak_results.json")
    args = parser.parse_args()

    run_soak(args.base_url, args.api_key, args.model,
             args.rps, args.duration_hours, args.output)


if __name__ == "__main__":
    main()
