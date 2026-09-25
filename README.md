# Inference Provider Evaluation Framework

A reusable framework for evaluating third-party inference API providers on service quality and technical due diligence. Produces a scored HTML report with a weighted composite score and decision tier recommendation.

---

## How to Use — GitHub Actions (Recommended)

The easiest way to run an evaluation is via the reusable GitHub Actions workflow. No local Python setup required — just a GitHub repo and an API key.

### Step 1 — Create a caller workflow in your repo

Copy `.github/workflows/example-caller.yml` from this repo into your own repo and edit the provider details:

```yaml
# .github/workflows/eval-acme.yml
name: Evaluate — Acme Inference

on:
  workflow_dispatch:   # trigger manually from GitHub Actions UI

jobs:
  evaluate:
    uses: Yu-amd/inference-eval-framework/.github/workflows/eval.yml@main
    with:
      base-url: "https://api.acme-inference.io/v1"
      model: "acme/model-name"
      provider-name: "Acme Inference"
      input-cost-per-1m: "1.50"    # USD per 1M input tokens
      output-cost-per-1m: "3.00"   # USD per 1M output tokens
      concurrency-levels: "1 5 10 25"
      iterations: "50"
      run-soak: true               # set false to skip the 2h soak test
      soak-duration-hours: "2"
      soak-rps: "2"
    secrets:
      api-key: ${{ secrets.API_KEY }}
```

### Step 2 — Add your API key as a secret

In your GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**

- Name: `API_KEY`
- Value: your inference provider API key

### Step 3 — Run the workflow

Go to **Actions → your workflow → Run workflow**. No inputs needed — everything is configured in the YAML.

### Step 4 — Download the report

When the run completes, go to the **Summary** tab of the run and download the artifact:

| Artifact | Contains | Available |
|---|---|---|
| `evaluation-report` | Interim report (perf, accuracy, cost) | ~3 minutes |
| `evaluation-report-final` | Full report including reliability/soak | ~2 hours (if `run-soak: true`) |

Open the downloaded `.html` file in any browser.

---

## Workflow Inputs Reference

| Input | Required | Default | Description |
|---|---|---|---|
| `base-url` | Yes | — | Provider API base URL (must be OpenAI-compatible) |
| `model` | Yes | — | Model ID to evaluate |
| `provider-name` | No | `Inference Provider` | Display name used in the report |
| `input-cost-per-1m` | No | `0` | Input token price in USD per 1M tokens |
| `output-cost-per-1m` | No | `0` | Output token price in USD per 1M tokens |
| `concurrency-levels` | No | `1 5 10 25` | Space-separated concurrency levels for perf test |
| `iterations` | No | `50` | Requests per concurrency level |
| `run-soak` | No | `false` | Whether to run the 2h reliability soak test |
| `soak-duration-hours` | No | `2` | Soak test duration in hours |
| `soak-rps` | No | `2` | Soak test requests per second |

---

## How to Use — Local / CI

If you prefer to run scripts directly, install dependencies and run each script in order:

```bash
pip install requests urllib3
```

```bash
# 1. Performance — TTFT, tok/s, latency percentiles
python scripts/perf_test.py \
  --base-url https://api.provider.io/v1 \
  --api-key $API_KEY \
  --model model-id \
  --concurrency 1 5 10 25 \
  --iterations 50 \
  --output results/performance/perf_results.json

# 2. Accuracy & safety
python scripts/accuracy_test.py \
  --base-url https://api.provider.io/v1 \
  --api-key $API_KEY \
  --model model-id \
  --output results/accuracy/accuracy_results.json

# 3. Cost modeling
python scripts/cost_estimator.py \
  --base-url https://api.provider.io/v1 \
  --api-key $API_KEY \
  --model model-id \
  --input-cost-per-1m 1.50 \
  --output-cost-per-1m 3.00 \
  --output results/cost/cost_results.json

# 4. Soak test — optional, runs for --duration-hours
python scripts/soak_test.py \
  --base-url https://api.provider.io/v1 \
  --api-key $API_KEY \
  --model model-id \
  --rps 2 \
  --duration-hours 2 \
  --output results/reliability/soak_results.json

# 5. Generate report
python scripts/generate_report.py \
  --results-dir results \
  --provider-name "Acme Inference" \
  --model model-id \
  --base-url https://api.provider.io/v1 \
  --input-cost 1.50 \
  --output-cost 3.00 \
  --output report/evaluation_report.html
```

The report generator works with any combination of results — missing test results are noted as "deferred" in the report rather than causing an error.

---

## Evaluation Categories

| # | Category | Weight | What's Measured |
|---|---|---|---|
| 1 | Inference Performance & Latency | 25% | TTFT p50/p99, tok/s, error rate across concurrency levels |
| 2 | Reliability & Availability | 20% | Availability %, error rate, latency drift over soak test |
| 3 | API Completeness & Compatibility | 15% | OpenAI compat, streaming, tool calling, token reporting |
| 4 | Security & Compliance | 15% | SOC 2, data retention, encryption, SSL cert, GDPR |
| 5 | Cost Structure & Commercial Terms | 12% | Savings vs GPT-4o, volume discounts, rate limit transparency |
| 6 | Output Quality & Safety | 8% | Factuality, instruction following, safety refusals, consistency |
| 7 | Support & Operations | 5% | Support SLA, status page, deprecation policy |

> Categories 3, 4, and 7 require manual assessment — the report flags these with a checklist and links to scoring rubrics in the framework document.

---

## Decision Tiers

| Weighted Score | Dealbreakers | Recommendation |
|---|---|---|
| ≥ 4.0 | None | ✅ Recommend |
| 3.0 – 3.9 | None | ⚠ Recommend with Conditions |
| 2.0 – 2.9 | None | ⏸ Defer |
| Any | Any triggered | 🚫 Reject |

Scores are normalized against the weight of evaluated categories only — missing categories do not penalize the score.

---

## What's Included

| Path | Description |
|---|---|
| `.github/workflows/eval.yml` | Reusable workflow — the engine |
| `.github/workflows/example-caller.yml` | Copy-paste caller template |
| `scripts/perf_test.py` | Performance test |
| `scripts/accuracy_test.py` | Accuracy & safety test |
| `scripts/cost_estimator.py` | Cost modeling |
| `scripts/soak_test.py` | Reliability soak test |
| `scripts/generate_report.py` | HTML report generator |
| `report/inference_provider_eval_framework.html` | Full framework document with rubrics and due diligence checklist |
| `report/inference_provider_eval_framework.pdf` | Print-ready PDF version |
| `report_final/` | Example output — Mangoboost MiniMax-M3 evaluation |
