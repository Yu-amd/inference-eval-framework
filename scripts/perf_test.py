"""
perf_test.py — Mangoboost inference performance harness
Measures TTFT, token generation speed, and end-to-end latency.

Usage:
    python perf_test.py --base-url https://api.mangoboost.ai/v1 \
                        --api-key $MANGOBOOST_API_KEY \
                        --model <model-id> \
                        --concurrency 1 5 10 25 50 \
                        --iterations 50
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class RequestResult:
    concurrency: int
    iteration: int
    ttft_ms: float | None       # time to first token
    total_ms: float             # end-to-end latency
    prompt_tokens: int
    completion_tokens: int
    tokens_per_sec: float | None
    status_code: int
    error: str | None = None


def send_request(base_url: str, api_key: str, model: str,
                 prompt: str, concurrency: int, iteration: int) -> RequestResult:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 200,
    }

    ttft_ms = None
    t_start = time.perf_counter()

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60, verify=False) as resp:
            status_code = resp.status_code
            if status_code != 200:
                return RequestResult(
                    concurrency=concurrency, iteration=iteration,
                    ttft_ms=None, total_ms=(time.perf_counter() - t_start) * 1000,
                    prompt_tokens=0, completion_tokens=0, tokens_per_sec=None,
                    status_code=status_code, error=resp.text[:200],
                )

            completion_tokens = 0
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            completion_tokens += len(delta.split())  # rough approximation
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

        total_ms = (time.perf_counter() - t_start) * 1000
        gen_time_s = (total_ms - (ttft_ms or 0)) / 1000
        tokens_per_sec = completion_tokens / gen_time_s if gen_time_s > 0 else None

        return RequestResult(
            concurrency=concurrency, iteration=iteration,
            ttft_ms=ttft_ms, total_ms=total_ms,
            prompt_tokens=0, completion_tokens=completion_tokens,
            tokens_per_sec=tokens_per_sec, status_code=status_code,
        )

    except Exception as exc:
        return RequestResult(
            concurrency=concurrency, iteration=iteration,
            ttft_ms=None, total_ms=(time.perf_counter() - t_start) * 1000,
            prompt_tokens=0, completion_tokens=0, tokens_per_sec=None,
            status_code=0, error=str(exc),
        )


def run_concurrency_level(base_url: str, api_key: str, model: str,
                          prompt: str, concurrency: int, iterations: int) -> list[RequestResult]:
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(send_request, base_url, api_key, model, prompt, concurrency, i)
            for i in range(iterations)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def percentile(data: list[float], p: int) -> float:
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def summarize(results: list[RequestResult], concurrency: int) -> dict:
    successes = [r for r in results if r.status_code == 200]
    errors = [r for r in results if r.status_code != 200]

    ttfts = [r.ttft_ms for r in successes if r.ttft_ms is not None]
    totals = [r.total_ms for r in successes]
    tps = [r.tokens_per_sec for r in successes if r.tokens_per_sec is not None]

    def stats(data: list[float]) -> dict:
        if not data:
            return {}
        return {
            "p50": round(percentile(data, 50), 1),
            "p95": round(percentile(data, 95), 1),
            "p99": round(percentile(data, 99), 1),
            "mean": round(statistics.mean(data), 1),
            "min": round(min(data), 1),
            "max": round(max(data), 1),
        }

    return {
        "concurrency": concurrency,
        "total_requests": len(results),
        "success_count": len(successes),
        "error_count": len(errors),
        "error_rate_pct": round(len(errors) / len(results) * 100, 2),
        "ttft_ms": stats(ttfts),
        "total_latency_ms": stats(totals),
        "tokens_per_sec": stats(tps),
        "errors": [{"status": r.status_code, "error": r.error} for r in errors[:5]],
    }


def main():
    parser = argparse.ArgumentParser(description="Mangoboost performance tester")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10, 25])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--prompt", default="Explain the concept of gradient descent in 3 sentences.")
    parser.add_argument("--output", default="results/performance/perf_results.json")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for c in args.concurrency:
        print(f"\nRunning concurrency={c}, iterations={args.iterations}...")
        results = run_concurrency_level(
            args.base_url, args.api_key, args.model,
            args.prompt, c, args.iterations,
        )
        summary = summarize(results, c)
        all_summaries.append(summary)
        print(f"  TTFT p50={summary['ttft_ms'].get('p50')}ms  "
              f"p99={summary['ttft_ms'].get('p99')}ms  "
              f"errors={summary['error_rate_pct']}%")

    with open(args.output, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
