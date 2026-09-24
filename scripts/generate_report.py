"""
generate_report.py — Auto-generate an HTML evaluation report from test results.

Reads JSON output files from perf_test.py, accuracy_test.py, cost_estimator.py,
and soak_test.py, then produces a self-contained HTML report scored against the
Inference Provider Evaluation Framework rubric.

Usage:
    python scripts/generate_report.py \
        --results-dir results \
        --provider-name "Acme Inference" \
        --model "acme/model-name" \
        --base-url "https://api.acme-inference.io/v1" \
        --input-cost 1.50 \
        --output-cost 3.00 \
        --output report/evaluation_report.html \
        --date 2026-09-24
"""

import argparse
import json
import math
import os
import statistics
from datetime import date
from pathlib import Path


# ── Scoring helpers ───────────────────────────────────────────────────────────

def score_performance(perf: dict | None) -> tuple[float, dict]:
    """Score performance category (1–5) from perf_results.json."""
    if not perf:
        return 1.0, {"note": "No performance data available"}

    # Use c=1 entry for primary scoring
    c1 = next((r for r in perf if r.get("concurrency") == 1), perf[0] if perf else None)
    if not c1:
        return 1.0, {"note": "No c=1 data"}

    ttft_p50 = c1.get("ttft_ms", {}).get("p50") or 9999
    ttft_p99 = c1.get("ttft_ms", {}).get("p99") or 9999
    tps_p50  = c1.get("tokens_per_sec", {}).get("p50") or 0
    err_rate = c1.get("error_rate_pct", 100)

    # TTFT p50 scoring
    if ttft_p50 < 300:    ttft_score = 5
    elif ttft_p50 < 600:  ttft_score = 4
    elif ttft_p50 < 1000: ttft_score = 3
    elif ttft_p50 < 2000: ttft_score = 2
    else:                 ttft_score = 1

    # p99 scoring
    if ttft_p99 < 800:    p99_score = 5
    elif ttft_p99 < 2000: p99_score = 4
    elif ttft_p99 < 5000: p99_score = 3
    elif ttft_p99 < 10000: p99_score = 2
    else:                  p99_score = 1

    # Throughput scoring
    if tps_p50 > 80:    tps_score = 5
    elif tps_p50 > 30:  tps_score = 4
    elif tps_p50 > 15:  tps_score = 3
    elif tps_p50 > 5:   tps_score = 2
    else:               tps_score = 1

    # Error rate scoring
    if err_rate == 0:        err_score = 5
    elif err_rate < 0.5:     err_score = 4
    elif err_rate < 1.0:     err_score = 3
    elif err_rate < 3.0:     err_score = 2
    else:                    err_score = 1

    score = round((ttft_score * 0.35 + p99_score * 0.25 + tps_score * 0.25 + err_score * 0.15), 2)
    return score, {
        "ttft_p50_ms": ttft_p50, "ttft_p99_ms": ttft_p99,
        "tps_p50": tps_p50, "error_rate_pct": err_rate,
        "concurrency_levels": [r.get("concurrency") for r in perf],
    }


def score_reliability(soak: dict | None) -> tuple[float, dict]:
    """Score reliability category (1–5) from soak_results.json."""
    if not soak:
        return None, {"note": "Soak test not run — reliability score deferred"}

    summary = soak.get("summary", soak)
    avail = summary.get("availability_pct", 0)
    err   = summary.get("error_rate_pct", 100)

    if avail >= 99.9:   score = 5
    elif avail >= 99.5: score = 4
    elif avail >= 99.0: score = 3
    elif avail >= 98.0: score = 2
    else:               score = 1

    return float(score), {
        "availability_pct": avail,
        "error_rate_pct": err,
        "total_requests": summary.get("total_requests"),
        "latency_p50_ms": summary.get("latency_ms", {}).get("p50"),
        "latency_p99_ms": summary.get("latency_ms", {}).get("p99"),
    }


def score_accuracy(acc: dict | None) -> tuple[float, dict]:
    """Score accuracy/quality category (1–5) from accuracy_results.json."""
    if not acc:
        return 1.0, {"note": "No accuracy data available"}

    factuality = acc.get("factuality", {})
    instruction = acc.get("instruction_following", {})
    safety = acc.get("safety", {})
    consistency = acc.get("consistency", {})

    fact_score  = (factuality.get("correct", 0) / max(factuality.get("total", 1), 1)) * 5
    inst_score  = (instruction.get("passed", 0) / max(instruction.get("total", 1), 1)) * 5
    safe_score  = (safety.get("refused", 0) / max(safety.get("total", 5), 1)) * 5
    sem_overlap = consistency.get("semantic_overlap", 0)
    cons_score  = sem_overlap * 5

    score = round(fact_score * 0.30 + inst_score * 0.25 + safe_score * 0.30 + cons_score * 0.15, 2)
    return score, {
        "factuality": f"{factuality.get('correct', '?')}/{factuality.get('total', '?')}",
        "instruction_following": f"{instruction.get('passed', '?')}/{instruction.get('total', '?')}",
        "safety_refusals": f"{safety.get('refused', '?')}/{safety.get('total', '?')}",
        "semantic_overlap": round(sem_overlap, 3),
    }


def score_cost(cost: dict | None) -> tuple[float, dict]:
    """Score cost category (1–5) by comparing to GPT-4o on classification workload."""
    if not cost:
        return None, {"note": "No cost data available"}

    workloads = cost.get("workloads", [])
    class_wl = next((w for w in workloads if w.get("workload_id") == "classification"), None)
    if not class_wl:
        return None, {"note": "Classification workload missing from cost data"}

    monthly = class_wl.get("monthly_cost_usd", {})
    provider_key = next((k for k in monthly if "GPT" not in k and "Claude" not in k and "OpenAI" not in k and "Anthropic" not in k), None)
    gpt4o_monthly = monthly.get("OpenAI GPT-4o", 0)
    provider_monthly = monthly.get(provider_key, 0) if provider_key else 0

    if gpt4o_monthly and provider_monthly:
        savings_pct = (gpt4o_monthly - provider_monthly) / gpt4o_monthly * 100
    else:
        savings_pct = 0

    if savings_pct >= 70:    score = 5
    elif savings_pct >= 40:  score = 4
    elif savings_pct >= 10:  score = 3
    elif savings_pct >= -10: score = 2
    else:                    score = 1

    return float(score), {
        "provider_monthly_usd": round(provider_monthly, 2),
        "gpt4o_monthly_usd": round(gpt4o_monthly, 2),
        "savings_vs_gpt4o_pct": round(savings_pct, 1),
        "workload": "Classification @ 1M req/month",
    }


def compute_weighted_score(scores: dict) -> float:
    """Compute weighted score, skipping None (deferred) categories."""
    weights = {
        "performance": 0.25,
        "reliability": 0.20,
        "api_compat":  0.15,
        "security":    0.15,
        "cost":        0.12,
        "accuracy":    0.08,
        "support":     0.05,
    }
    total_weight = 0
    total_score  = 0
    for cat, weight in weights.items():
        val = scores.get(cat)
        if val is not None:
            total_score  += val * weight
            total_weight += weight

    return round(total_score / total_weight * (total_weight / sum(weights.values())), 2) if total_weight else 0


def decision_tier(score: float, has_dealbreaker: bool) -> str:
    if has_dealbreaker:
        return "Reject"
    if score >= 4.0:
        return "Recommend"
    if score >= 3.0:
        return "Recommend with Conditions"
    if score >= 2.0:
        return "Defer"
    return "Reject"


# ── Pill / color helpers ──────────────────────────────────────────────────────

def score_color(s):
    if s is None:   return "#94a3b8"
    if s >= 4.0:    return "#16a34a"
    if s >= 3.0:    return "#d97706"
    return "#dc2626"


def score_bar_pct(s):
    if s is None: return 0
    return round(s / 5 * 100)


def fmt(v, suffix=""):
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:,.1f}{suffix}"
    return f"{v}{suffix}"


# ── HTML generation ───────────────────────────────────────────────────────────

def build_html(provider_name, model, base_url, input_cost, output_cost,
               eval_date, perf_data, perf_score, perf_detail,
               rel_score, rel_detail, acc_score, acc_detail,
               cost_score, cost_detail, weighted_score, tier) -> str:

    tier_colors = {
        "Recommend": ("#f0fdf4", "#86efac", "#15803d"),
        "Recommend with Conditions": ("#fffbeb", "#fde68a", "#b45309"),
        "Defer": ("#fff7ed", "#fed7aa", "#ea580c"),
        "Reject": ("#fff1f2", "#fecaca", "#dc2626"),
    }
    tier_bg, tier_border, tier_text = tier_colors.get(tier, ("#f8fafc", "#e2e8f0", "#1e293b"))

    # Build performance table rows
    perf_rows = ""
    if perf_data:
        for entry in perf_data:
            c = entry.get("concurrency", "?")
            t50  = entry.get("ttft_ms", {}).get("p50")
            t99  = entry.get("ttft_ms", {}).get("p99")
            tps  = entry.get("tokens_per_sec", {}).get("p50")
            err  = entry.get("error_rate_pct", 0)
            err_pill = f'<span style="background:#dcfce7;color:#15803d;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600;">{err}%</span>' if err == 0 else f'<span style="background:#fee2e2;color:#b91c1c;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600;">{err:.1f}%</span>'
            perf_rows += f"<tr><td><strong>c = {c}</strong></td><td>{fmt(t50,'ms')}</td><td>{fmt(t99,'ms')}</td><td>{fmt(tps)}</td><td>{err_pill}</td></tr>\n"

    # Build chart data
    conc_labels = json.dumps([f"c = {r.get('concurrency')}" for r in (perf_data or [])])
    ttft_p50_data = json.dumps([r.get("ttft_ms", {}).get("p50") for r in (perf_data or [])])
    ttft_p99_data = json.dumps([r.get("ttft_ms", {}).get("p99") for r in (perf_data or [])])
    tps_data = json.dumps([r.get("tokens_per_sec", {}).get("p50") for r in (perf_data or [])])

    # Scores (display N/A for deferred)
    def score_str(s): return f"{s:.1f} / 5.0" if s is not None else "Deferred"

    scores = {
        "performance": perf_score,
        "reliability": rel_score,
        "accuracy":    acc_score,
        "cost":        cost_score,
    }

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{provider_name} — Inference Evaluation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#f8fafc;--surface:#fff;--border:#e2e8f0;--text:#1e293b;--muted:#64748b;
    --green:#16a34a;--yellow:#d97706;--red:#dc2626;--blue:#2563eb;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.6;}}
  .header{{background:linear-gradient(135deg,#1a1d27 0%,#2d3149 100%);color:#fff;padding:48px 64px 40px;}}
  .header h1{{font-size:26px;font-weight:700;}}
  .header .sub{{color:#94a3b8;font-size:13px;margin-top:6px;}}
  .header .meta{{display:flex;gap:32px;margin-top:24px;flex-wrap:wrap;}}
  .header .meta-item strong{{color:#fff;display:block;font-size:13px;}}
  .header .meta-item{{font-size:12px;color:#cbd5e1;}}
  .container{{max-width:1080px;margin:0 auto;padding:40px 48px;}}
  section{{margin-bottom:44px;}}
  h2{{font-size:18px;font-weight:700;border-bottom:2px solid var(--border);padding-bottom:10px;margin-bottom:20px;}}
  h3{{font-size:14px;font-weight:600;margin-bottom:10px;}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px 24px;margin-bottom:14px;}}
  .card-grid{{display:grid;gap:14px;}}
  .card-grid-2{{grid-template-columns:1fr 1fr;}}
  table{{width:100%;border-collapse:collapse;font-size:13px;}}
  th{{background:#f1f5f9;color:var(--muted);font-weight:600;text-align:left;padding:9px 12px;border-bottom:2px solid var(--border);font-size:11px;text-transform:uppercase;letter-spacing:.3px;}}
  td{{padding:9px 12px;border-bottom:1px solid var(--border);}}
  tr:last-child td{{border-bottom:none;}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px;}}
  .stat{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center;}}
  .stat .num{{font-size:26px;font-weight:700;}}
  .stat .lbl{{font-size:11px;color:var(--muted);margin-top:4px;}}
  .stat.green{{border-top:3px solid var(--green);}} .stat.green .num{{color:var(--green);}}
  .stat.yellow{{border-top:3px solid var(--yellow);}} .stat.yellow .num{{color:var(--yellow);}}
  .stat.red{{border-top:3px solid var(--red);}} .stat.red .num{{color:var(--red);}}
  .stat.blue{{border-top:3px solid var(--blue);}} .stat.blue .num{{color:var(--blue);}}
  .verdict{{background:var(--surface);border:1px solid {tier_border};border-left:6px solid {tier_text};border-radius:10px;padding:28px 32px;display:flex;gap:32px;align-items:center;margin-bottom:28px;}}
  .verdict .big{{font-size:52px;font-weight:800;color:{tier_text};line-height:1;}}
  .verdict .lbl{{font-size:12px;color:var(--muted);margin-top:4px;}}
  .verdict h3{{font-size:18px;font-weight:700;color:{tier_text};margin-bottom:8px;}}
  .score-row{{display:grid;grid-template-columns:200px 60px 1fr 70px;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border);}}
  .score-row:last-child{{border-bottom:none;}}
  .score-row.total{{font-weight:700;border-top:2px solid var(--border);padding-top:12px;}}
  .bar-wrap{{background:#f1f5f9;border-radius:99px;height:8px;}}
  .bar{{height:8px;border-radius:99px;}}
  .chart-wrap{{position:relative;height:240px;}}
  .info-box{{background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid var(--blue);border-radius:8px;padding:12px 16px;font-size:13px;margin-bottom:14px;}}
  .warn-box{{background:#fffbeb;border:1px solid #fde68a;border-left:4px solid var(--yellow);border-radius:8px;padding:12px 16px;font-size:13px;margin-bottom:14px;}}
  .na-box{{background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:14px 18px;font-size:13px;color:var(--muted);margin-bottom:14px;}}
  .page-footer{{background:#1a1d27;color:#6b7280;text-align:center;padding:24px;font-size:12px;margin-top:48px;}}
  code{{background:#f1f5f9;border-radius:4px;padding:1px 6px;font-size:12px;font-family:monospace;}}
  @media print {{
    *{{-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important;}}
    section{{page-break-inside:avoid;}}
    .card{{page-break-inside:avoid;}}
    .verdict{{page-break-inside:avoid;}}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>{provider_name} — Inference Provider Evaluation</h1>
  <div class="sub">AMD ISV Partnership Team · Inference Provider Evaluation Framework v1.0</div>
  <div class="meta">
    <div class="meta-item"><strong>Model</strong>{model}</div>
    <div class="meta-item"><strong>Base URL</strong>{base_url}</div>
    <div class="meta-item"><strong>Evaluation Date</strong>{eval_date}</div>
    <div class="meta-item"><strong>Input Price</strong>${input_cost}/1M tokens</div>
    <div class="meta-item"><strong>Output Price</strong>${output_cost}/1M tokens</div>
  </div>
</div>

<div class="container">

<!-- VERDICT -->
<section>
  <div class="verdict">
    <div>
      <div class="big">{weighted_score:.2f}</div>
      <div class="lbl">out of 5.0</div>
    </div>
    <div>
      <h3>{tier}</h3>
      <p style="font-size:13px;color:var(--muted);">
        Weighted composite score across {sum(1 for s in scores.values() if s is not None)} evaluated categories.
        {"Soak test deferred — reliability score not included in weighted total." if rel_score is None else ""}
      </p>
    </div>
  </div>
</section>

<!-- SCORECARD -->
<section>
  <h2>Overall Scorecard</h2>
  <div class="card">
    <div class="score-row">
      <div>🚀 Inference Performance</div><div style="color:var(--muted);font-size:12px;">25%</div>
      <div class="bar-wrap"><div class="bar" style="width:{score_bar_pct(perf_score)}%;background:{score_color(perf_score)};"></div></div>
      <div style="font-weight:700;color:{score_color(perf_score)};">{score_str(perf_score)}</div>
    </div>
    <div class="score-row">
      <div>🛡️ Reliability &amp; Availability</div><div style="color:var(--muted);font-size:12px;">20%</div>
      <div class="bar-wrap"><div class="bar" style="width:{score_bar_pct(rel_score)}%;background:{score_color(rel_score)};"></div></div>
      <div style="font-weight:700;color:{score_color(rel_score)};">{score_str(rel_score)}</div>
    </div>
    <div class="score-row">
      <div>🎯 Output Quality &amp; Safety</div><div style="color:var(--muted);font-size:12px;">8%</div>
      <div class="bar-wrap"><div class="bar" style="width:{score_bar_pct(acc_score)}%;background:{score_color(acc_score)};"></div></div>
      <div style="font-weight:700;color:{score_color(acc_score)};">{score_str(acc_score)}</div>
    </div>
    <div class="score-row">
      <div>💰 Cost &amp; Economics</div><div style="color:var(--muted);font-size:12px;">12%</div>
      <div class="bar-wrap"><div class="bar" style="width:{score_bar_pct(cost_score)}%;background:{score_color(cost_score)};"></div></div>
      <div style="font-weight:700;color:{score_color(cost_score)};">{score_str(cost_score)}</div>
    </div>
    <div class="score-row total">
      <div>Total Weighted Score</div><div style="color:var(--muted);font-size:12px;">100%</div>
      <div class="bar-wrap"><div class="bar" style="width:{score_bar_pct(weighted_score)}%;background:{score_color(weighted_score)};"></div></div>
      <div style="font-weight:700;color:{score_color(weighted_score)};">{weighted_score:.2f} / 5.0</div>
    </div>
  </div>
  <div class="info-box">API compatibility, security/compliance, and support scores require manual assessment — see the <strong>Evaluation Framework</strong> for rubrics and due diligence checklist.</div>
</section>

<!-- PERFORMANCE -->
<section>
  <h2>1 — Inference Performance &amp; Latency <span style="font-size:13px;font-weight:400;color:var(--muted);margin-left:8px;">Score: {score_str(perf_score)}</span></h2>
  {"" if perf_data else '<div class="na-box">No performance data — run perf_test.py to populate this section.</div>'}

  {"" if not perf_data else f'''
  <div class="stat-grid">
    <div class="stat {"green" if (perf_detail.get("ttft_p50_ms") or 9999) < 600 else "yellow"}">
      <div class="num">{fmt(perf_detail.get("ttft_p50_ms"), "ms")}</div><div class="lbl">TTFT p50 @ c=1</div>
    </div>
    <div class="stat {"green" if (perf_detail.get("ttft_p99_ms") or 9999) < 2000 else "yellow"}">
      <div class="num">{fmt(perf_detail.get("ttft_p99_ms"), "ms")}</div><div class="lbl">TTFT p99 @ c=1</div>
    </div>
    <div class="stat green"><div class="num">{fmt(perf_detail.get("error_rate_pct"), "%")}</div><div class="lbl">Error Rate</div></div>
    <div class="stat blue"><div class="num">{fmt(perf_detail.get("tps_p50"))}</div><div class="lbl">Tok/s p50 @ c=1</div></div>
  </div>

  <div class="card-grid card-grid-2">
    <div class="card">
      <h3>TTFT by Concurrency</h3>
      <div class="chart-wrap"><canvas id="ttftChart"></canvas></div>
    </div>
    <div class="card">
      <h3>Token Generation Speed (tok/s)</h3>
      <div class="chart-wrap"><canvas id="tpsChart"></canvas></div>
    </div>
  </div>

  <div class="card">
    <h3>Full Latency Breakdown</h3>
    <table>
      <thead><tr><th>Concurrency</th><th>TTFT p50</th><th>TTFT p99</th><th>Tok/s p50</th><th>Error Rate</th></tr></thead>
      <tbody>{perf_rows}</tbody>
    </table>
  </div>
  '''}
</section>

<!-- RELIABILITY -->
<section>
  <h2>2 — Reliability &amp; Availability <span style="font-size:13px;font-weight:400;color:var(--muted);margin-left:8px;">Score: {score_str(rel_score)}</span></h2>
  {f'''
  <div class="stat-grid">
    <div class="stat {"green" if (rel_detail.get("availability_pct") or 0) >= 99.9 else "yellow"}">
      <div class="num">{fmt(rel_detail.get("availability_pct"), "%")}</div><div class="lbl">Availability</div>
    </div>
    <div class="stat {"green" if (rel_detail.get("error_rate_pct") or 100) < 0.5 else "yellow"}">
      <div class="num">{fmt(rel_detail.get("error_rate_pct"), "%")}</div><div class="lbl">Error Rate</div>
    </div>
    <div class="stat blue"><div class="num">{fmt(rel_detail.get("total_requests"))}</div><div class="lbl">Total Requests</div></div>
    <div class="stat blue"><div class="num">{fmt(rel_detail.get("latency_p50_ms"), "ms")}</div><div class="lbl">Latency p50 (soak)</div></div>
  </div>
  ''' if rel_score is not None else '<div class="na-box">Soak test not run. Run soak_test.py with <code>--duration-hours 2</code> to populate this section. Reliability score excluded from weighted total.</div>'}
</section>

<!-- ACCURACY -->
<section>
  <h2>3 — Output Quality &amp; Safety <span style="font-size:13px;font-weight:400;color:var(--muted);margin-left:8px;">Score: {score_str(acc_score)}</span></h2>
  {f'''
  <div class="stat-grid">
    <div class="stat {"green" if acc_detail.get("factuality", "0/0").split("/")[0] == acc_detail.get("factuality", "0/0").split("/")[1] else "yellow"}">
      <div class="num">{acc_detail.get("factuality", "N/A")}</div><div class="lbl">Factuality</div>
    </div>
    <div class="stat blue"><div class="num">{acc_detail.get("instruction_following", "N/A")}</div><div class="lbl">Instruction Following</div></div>
    <div class="stat {"green" if acc_detail.get("safety_refusals", "0/5").split("/")[0] == acc_detail.get("safety_refusals", "0/5").split("/")[1] else "yellow"}">
      <div class="num">{acc_detail.get("safety_refusals", "N/A")}</div><div class="lbl">Safety Refusals</div>
    </div>
    <div class="stat {"green" if (acc_detail.get("semantic_overlap") or 0) >= 0.6 else "yellow"}">
      <div class="num">{acc_detail.get("semantic_overlap", "N/A")}</div><div class="lbl">Semantic Overlap</div>
    </div>
  </div>
  ''' if acc_score is not None else '<div class="na-box">No accuracy data. Run accuracy_test.py to populate this section.</div>'}
</section>

<!-- COST -->
<section>
  <h2>4 — Cost &amp; Economics <span style="font-size:13px;font-weight:400;color:var(--muted);margin-left:8px;">Score: {score_str(cost_score)}</span></h2>
  {f'''
  <div class="stat-grid">
    <div class="stat {"green" if (cost_detail.get("savings_vs_gpt4o_pct") or 0) > 30 else "yellow"}">
      <div class="num">{fmt(cost_detail.get("savings_vs_gpt4o_pct"), "%")}</div><div class="lbl">Savings vs GPT-4o</div>
    </div>
    <div class="stat blue"><div class="num">${fmt(cost_detail.get("provider_monthly_usd"))}</div><div class="lbl">Provider/month (class., 1M req)</div></div>
    <div class="stat blue"><div class="num">${fmt(cost_detail.get("gpt4o_monthly_usd"))}</div><div class="lbl">GPT-4o/month (same workload)</div></div>
  </div>
  ''' if cost_score is not None else '<div class="na-box">No cost data. Run cost_estimator.py to populate this section.</div>'}
</section>

<!-- MANUAL ASSESSMENT REMINDER -->
<section>
  <h2>5 — Manual Assessment Required</h2>
  <div class="warn-box">
    The following categories require vendor documentation review and/or direct conversation with the provider. Use the scoring rubrics in the <strong>Evaluation Framework</strong> document to complete these scores.
  </div>
  <div class="card">
    <table>
      <thead><tr><th>Category</th><th>Weight</th><th>Items to Assess</th><th>Score</th></tr></thead>
      <tbody>
        <tr><td>API Completeness &amp; Compatibility</td><td>15%</td><td>Tool calling, JSON mode, model pinning, batch API, SDK</td><td style="color:var(--muted);">___ / 5</td></tr>
        <tr><td>Security &amp; Compliance</td><td>15%</td><td>SOC 2, data retention policy, encryption, SSL cert, GDPR DPA</td><td style="color:var(--muted);">___ / 5</td></tr>
        <tr><td>Cost (commercial terms)</td><td>(12%)</td><td>Volume discounts, rate limit documentation, contract terms, SLA credits</td><td style="color:var(--muted);">___ / 5</td></tr>
        <tr><td>Support &amp; Operations</td><td>5%</td><td>Support SLA, status page, deprecation policy, TAM availability</td><td style="color:var(--muted);">___ / 5</td></tr>
      </tbody>
    </table>
  </div>
</section>

</div>

<div class="page-footer">
  Generated by Inference Provider Evaluation Framework v1.0 · AMD ISV Partnership Team · {eval_date}
</div>

<script>
{"" if not perf_data else f"""
const LABELS = {conc_labels};
const ORANGE = '#ff6b35';

new Chart(document.getElementById('ttftChart'), {{
  type: 'line',
  data: {{
    labels: LABELS,
    datasets: [
      {{ label: 'TTFT p50', data: {ttft_p50_data}, borderColor: ORANGE, backgroundColor: ORANGE + '20', tension: 0.3, fill: false, pointRadius: 5, pointBackgroundColor: ORANGE }},
      {{ label: 'TTFT p99', data: {ttft_p99_data}, borderColor: '#94a3b8', backgroundColor: '#94a3b820', tension: 0.3, fill: false, pointRadius: 5, pointBackgroundColor: '#94a3b8', borderDash: [4,4] }}
    ]
  }},
  options: {{ responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ position:'top', labels:{{ boxWidth:12 }} }} }},
    scales:{{ y:{{ title:{{ display:true, text:'ms' }}, beginAtZero:true }}, x:{{ grid:{{ display:false }} }} }}
  }}
}});

new Chart(document.getElementById('tpsChart'), {{
  type: 'bar',
  data: {{
    labels: LABELS,
    datasets: [{{ label: 'Tok/s p50', data: {tps_data}, backgroundColor: ORANGE, borderRadius: 4 }}]
  }},
  options: {{ responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ position:'top', labels:{{ boxWidth:12 }} }} }},
    scales:{{ y:{{ title:{{ display:true, text:'tokens / second' }}, beginAtZero:true }}, x:{{ grid:{{ display:false }} }} }}
  }}
}});
"""}
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict | list | None:
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--provider-name", default="Inference Provider")
    parser.add_argument("--model", default="unknown-model")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--input-cost", default="0")
    parser.add_argument("--output-cost", default="0")
    parser.add_argument("--output", default="report/evaluation_report.html")
    parser.add_argument("--date", default=str(date.today()))
    args = parser.parse_args()

    results_dir = args.results_dir
    perf_path  = os.path.join(results_dir, "performance", "perf_results.json")
    acc_path   = os.path.join(results_dir, "accuracy",    "accuracy_results.json")
    cost_path  = os.path.join(results_dir, "cost",        "cost_results.json")
    soak_path  = os.path.join(results_dir, "reliability", "soak_results.json")

    perf_data = load_json(perf_path)
    acc_data  = load_json(acc_path)
    cost_data = load_json(cost_path)
    soak_data = load_json(soak_path)

    perf_score, perf_detail = score_performance(perf_data)
    rel_score,  rel_detail  = score_reliability(soak_data)
    acc_score,  acc_detail  = score_accuracy(acc_data)
    cost_score, cost_detail = score_cost(cost_data)

    all_scores = {
        "performance": perf_score,
        "reliability": rel_score,
        "accuracy":    acc_score,
        "cost":        cost_score,
    }
    weighted = compute_weighted_score(all_scores)
    tier = decision_tier(weighted, has_dealbreaker=False)

    html = build_html(
        provider_name=args.provider_name,
        model=args.model,
        base_url=args.base_url,
        input_cost=args.input_cost,
        output_cost=args.output_cost,
        eval_date=args.date,
        perf_data=perf_data,
        perf_score=perf_score,
        perf_detail=perf_detail,
        rel_score=rel_score,
        rel_detail=rel_detail,
        acc_score=acc_score,
        acc_detail=acc_detail,
        cost_score=cost_score,
        cost_detail=cost_detail,
        weighted_score=weighted,
        tier=tier,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{'='*60}")
    print(f"  Provider:  {args.provider_name}")
    print(f"  Model:     {args.model}")
    print(f"  Score:     {weighted:.2f} / 5.0")
    print(f"  Tier:      {tier}")
    print(f"  Report:    {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
