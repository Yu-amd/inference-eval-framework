# Inference Provider Evaluation Framework

A reusable framework for evaluating third-party inference API providers on service quality and technical due diligence.

## What's Included

| Path | Description |
|---|---|
| `report/inference_provider_eval_framework.html` | Full framework document (open in browser) |
| `report/inference_provider_eval_framework.pdf` | Print-ready PDF version |
| `scripts/perf_test.py` | TTFT, tok/s, latency percentiles across concurrency levels |
| `scripts/accuracy_test.py` | Factuality, instruction following, safety refusals, consistency |
| `scripts/cost_estimator.py` | Per-request and monthly cost vs. GPT-4o and other comparators |
| `scripts/soak_test.py` | Long-running availability and latency drift test |

## Evaluation Categories

| # | Category | Weight |
|---|---|---|
| 1 | Inference Performance & Latency | 25% |
| 2 | Reliability & Availability | 20% |
| 3 | API Completeness & Compatibility | 15% |
| 4 | Security & Compliance | 15% |
| 5 | Cost Structure & Commercial Terms | 12% |
| 6 | Output Quality & Safety | 8% |
| 7 | Support & Operations | 5% |

## Quick Start

```bash
pip install requests urllib3

# Performance test
python scripts/perf_test.py \
  --base-url https://api.provider.io/v1 \
  --api-key $API_KEY \
  --model model-id \
  --concurrency 1 5 10 25 \
  --iterations 50

# Accuracy & safety
python scripts/accuracy_test.py \
  --base-url https://api.provider.io/v1 \
  --api-key $API_KEY \
  --model model-id

# Cost modeling
python scripts/cost_estimator.py \
  --base-url https://api.provider.io/v1 \
  --api-key $API_KEY \
  --model model-id \
  --input-cost-per-1m 1.50 \
  --output-cost-per-1m 3.00

# Soak test (2 hours @ 2 RPS)
python scripts/soak_test.py \
  --base-url https://api.provider.io/v1 \
  --api-key $API_KEY \
  --model model-id \
  --rps 2 \
  --duration-hours 2
```

## Decision Tiers

| Weighted Score | Dealbreakers | Recommendation |
|---|---|---|
| ≥ 4.0 | None | ✅ Recommend |
| 3.0 – 3.9 | None | ⚠ Conditional |
| 2.0 – 2.9 | None | ⏸ Defer |
| Any | Any triggered | 🚫 Reject |
