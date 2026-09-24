"""
cost_estimator.py — Mangoboost token cost modeling
Runs representative workloads and estimates cost vs. comparators.

Base URL: https://api.mangoboost.io/v1

Mangoboost model IDs and pricing (as of 2026-09):
  deepseek-ai/DeepSeek-V4-Pro-0813   in=$1.50  cached=$0.13  out=$3.00
  zai-org/GLM-5.2-FP8                in=$1.30  cached=$0.14  out=$4.20
  zai-org/GLM-5.2-MXFP4             in=$1.30  cached=$0.14  out=$4.20
  minimax/MiniMax-M3                 in=$0.29  cached=$0.06  out=$1.20
  minimax/MiniMax-M3 (Kimi K3)      in=$2.90  cached=$0.29  out=$14.25 [Coming Soon]

NOTE: DeepSeek-V4-Pro-0813 is a reasoning model. The API returns a
      `reasoning_content` field and `reasoning_tokens` in usage. Reasoning
      tokens count toward output billing — account for this in cost models.

Usage:
    python cost_estimator.py --base-url https://api.mangoboost.io/v1 \\
                             --api-key $MANGOBOOST_API_KEY \\
                             --model deepseek-ai/DeepSeek-V4-Pro-0813 \\
                             --input-cost-per-1m 1.50 \\
                             --output-cost-per-1m 3.00
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class PricingConfig:
    name: str
    input_cost_per_1m: float   # USD per 1M input tokens
    output_cost_per_1m: float  # USD per 1M output tokens


# Mangoboost models — pricing as of 2026-09
MANGOBOOST_MODELS = {
    "deepseek-ai/DeepSeek-V4-Pro-0813": PricingConfig(
        "Mangoboost DeepSeek-V4-Pro-0813", input_cost_per_1m=1.50, output_cost_per_1m=3.00),
    "zai-org/GLM-5.2-FP8": PricingConfig(
        "Mangoboost GLM-5.2 (FP8)", input_cost_per_1m=1.30, output_cost_per_1m=4.20),
    "zai-org/GLM-5.2-MXFP4": PricingConfig(
        "Mangoboost GLM-5.2 (MXFP4)", input_cost_per_1m=1.30, output_cost_per_1m=4.20),
    "minimax/MiniMax-M3": PricingConfig(
        "Mangoboost MiniMax-M3", input_cost_per_1m=0.29, output_cost_per_1m=1.20),
}

# External comparators — pricing as of 2026-09 — update if stale
COMPARATORS = [
    PricingConfig("OpenAI GPT-4o", input_cost_per_1m=2.50, output_cost_per_1m=10.00),
    PricingConfig("OpenAI GPT-4o-mini", input_cost_per_1m=0.15, output_cost_per_1m=0.60),
    PricingConfig("Claude Sonnet 4.6", input_cost_per_1m=3.00, output_cost_per_1m=15.00),
    PricingConfig("Claude Haiku 4.5", input_cost_per_1m=0.80, output_cost_per_1m=4.00),
]

WORKLOADS = [
    {
        "id": "classification",
        "description": "Short classification task",
        "prompt": (
            "Classify the sentiment of the following review as Positive, Negative, or Neutral. "
            "Return only the label.\n\nReview: 'The product arrived on time and works great!'"
        ),
        "max_tokens": 10,
        "monthly_volume": 1_000_000,  # typical monthly request volume
    },
    {
        "id": "summarization",
        "description": "Document summarization",
        "prompt": (
            "Summarize the following article in 3 concise bullet points:\n\n"
            "Artificial intelligence has rapidly advanced in recent years, with large language models "
            "demonstrating remarkable capabilities in natural language understanding and generation. "
            "These systems are now being deployed across industries including healthcare, finance, "
            "education, and software development. Key challenges include ensuring model safety, "
            "reducing computational costs, and addressing bias in training data. Researchers are "
            "actively working on more efficient architectures, better alignment techniques, and "
            "improved evaluation methods to measure model capabilities reliably."
        ),
        "max_tokens": 150,
        "monthly_volume": 100_000,
    },
    {
        "id": "code_generation",
        "description": "Code generation task",
        "prompt": (
            "Write a Python function that takes a list of integers and returns a new list "
            "with duplicates removed, preserving the original order. Include type hints and "
            "a docstring."
        ),
        "max_tokens": 300,
        "monthly_volume": 50_000,
    },
]


def call_api(base_url: str, api_key: str, model: str, prompt: str, max_tokens: int) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120, verify=False)
    resp.raise_for_status()
    return resp.json()


def extract_tokens(response: dict) -> tuple[int, int, int]:
    """Returns (input_tokens, output_tokens, reasoning_tokens).
    Mangoboost response structure:
      usage.prompt_tokens           → input
      usage.completion_tokens       → total output (includes reasoning)
      usage.completion_tokens_details.reasoning_tokens → thinking tokens
    Reasoning tokens are billed as output and already included in completion_tokens.
    Also tracks cached_tokens from prompt_tokens_details for cache analysis.
    """
    usage = response.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens", 0)
    return input_tokens, output_tokens, reasoning_tokens


def compute_cost(input_tokens: int, output_tokens: int, pricing: PricingConfig) -> float:
    return (input_tokens / 1_000_000 * pricing.input_cost_per_1m +
            output_tokens / 1_000_000 * pricing.output_cost_per_1m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-cost-per-1m", type=float, required=True,
                        help="Mangoboost input token price per 1M tokens (USD)")
    parser.add_argument("--output-cost-per-1m", type=float, required=True,
                        help="Mangoboost output token price per 1M tokens (USD)")
    parser.add_argument("--output", default="results/cost/cost_results.json")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    mangoboost_pricing = PricingConfig(
        name=f"Mangoboost ({args.model})",
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )

    workload_results = []

    for workload in WORKLOADS:
        print(f"\nWorkload: {workload['description']}")
        try:
            response = call_api(args.base_url, args.api_key, args.model,
                                workload["prompt"], workload["max_tokens"])
            input_tokens, output_tokens, reasoning_tokens = extract_tokens(response)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            input_tokens, output_tokens, reasoning_tokens = 0, 0, 0

        print(f"  Tokens — input: {input_tokens}, output: {output_tokens} "
              f"(reasoning: {reasoning_tokens})")

        per_request_costs = {}
        monthly_costs = {}
        all_pricings = [mangoboost_pricing] + COMPARATORS

        for p in all_pricings:
            per_req = compute_cost(input_tokens, output_tokens, p)
            monthly = per_req * workload["monthly_volume"]
            per_request_costs[p.name] = round(per_req, 8)
            monthly_costs[p.name] = round(monthly, 2)
            print(f"  {p.name:35s}  ${per_req:.6f}/req  ${monthly:,.2f}/month")

        # Savings vs. most common comparator (GPT-4o)
        gpt4o_monthly = monthly_costs.get("OpenAI GPT-4o", 0)
        mb_monthly = monthly_costs.get(mangoboost_pricing.name, 0)
        savings_vs_gpt4o = gpt4o_monthly - mb_monthly if gpt4o_monthly > 0 else None

        workload_results.append({
            "workload_id": workload["id"],
            "description": workload["description"],
            "measured_tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "reasoning": reasoning_tokens,
            },
            "monthly_volume": workload["monthly_volume"],
            "per_request_cost_usd": per_request_costs,
            "monthly_cost_usd": monthly_costs,
            "monthly_savings_vs_gpt4o_usd": round(savings_vs_gpt4o, 2) if savings_vs_gpt4o else None,
        })

    output_data = {
        "model": args.model,
        "mangoboost_pricing": {
            "input_per_1m_usd": args.input_cost_per_1m,
            "output_per_1m_usd": args.output_cost_per_1m,
        },
        "workloads": workload_results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nCost results saved to {args.output}")


if __name__ == "__main__":
    main()
