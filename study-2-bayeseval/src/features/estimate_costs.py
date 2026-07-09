#!/usr/bin/env python3
"""Estimate the cost of a BayesEval run from config.yaml without calling any APIs.

Usage:
    python -m src.features.estimate_costs
    python -m src.features.estimate_costs --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOMAINS_DIR = REPO_ROOT / "domains"

# OpenRouter pricing per million tokens: (input, output)
# https://openrouter.ai/models
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "google/gemini-2.0-flash": (0.10, 0.40),
    "google/gemini-2.5-flash": (0.15, 0.60),
    "google/gemini-2.5-flash-preview": (0.15, 0.60),
    "google/gemini-2.5-pro": (1.25, 10.00),
    "google/gemini-2.5-pro-preview": (1.25, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4.1": (2.00, 8.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-4.1-nano": (0.10, 0.40),
    "openai/gpt-5.4-mini": (0.75, 4.50),
    "anthropic/claude-sonnet-4": (3.00, 15.00),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3.5-haiku": (0.80, 4.00),
    "anthropic/claude-haiku-4.5": (0.80, 4.00),
    "meta-llama/llama-4-maverick": (0.20, 0.60),
    "meta-llama/llama-4-scout": (0.15, 0.35),
    "qwen/qwen3.6-plus": (0.325, 1.95),
}

CHARS_PER_TOKEN = 4
OUTPUT_TOKENS_ESTIMATE = 25


def _normalize_domains(raw) -> list[dict]:
    if raw == "all" or raw == ["all"]:
        return [{"name": p.name} for p in sorted(DOMAINS_DIR.iterdir()) if p.is_dir()]
    out = []
    for entry in raw:
        if isinstance(entry, str):
            out.append({"name": entry})
        else:
            out.append(entry)
    return out


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["domains"] = _normalize_domains(cfg.get("domains", []))
    cfg.setdefault("max_questions", None)
    return cfg


def estimate_input_tokens(benchmark: pd.DataFrame) -> float:
    overhead = len(
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"Answer": "<your estimate>", "Confidence": "<probability between 0 and 1>"}\n'
        "No other text."
    )
    total_chars = 0
    for _, row in benchmark.iterrows():
        total_chars += len(str(row["question_prompt"])) + len(str(row["confidence_prompt"])) + overhead + 4
    return (total_chars / len(benchmark)) / CHARS_PER_TOKEN


def short_model(model: str) -> str:
    return model.split("/")[-1]


def main():
    ap = argparse.ArgumentParser(description="Estimate cost of a BayesEval run.")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = cfg.get("models", [])
    domains = cfg.get("domains", [])
    max_q = cfg.get("max_questions")
    console = Console()

    # -- gather domain stats --------------------------------------------------
    domain_stats: list[dict] = []
    for d in domains:
        bench_file = d.get("benchmark_file", "benchmark.csv")
        bench_path = DOMAINS_DIR / d["name"] / "Data" / bench_file
        if not bench_path.exists():
            console.print(f"[red]  {d['name']}: benchmark not found, skipping[/red]")
            continue
        bench = pd.read_csv(bench_path)
        n = min(len(bench), max_q) if max_q else len(bench)
        bench = bench.head(n)
        mean_in = estimate_input_tokens(bench)
        tok_in_total = int(mean_in * n)
        tok_out_total = OUTPUT_TOKENS_ESTIMATE * n
        domain_stats.append({
            "domain": d["name"], "questions": n,
            "mean_input_tok": mean_in,
            "tok_in": tok_in_total, "tok_out": tok_out_total,
        })

    # -- model rates table (left) ---------------------------------------------
    rates_table = Table(
        title="Model Rates", show_header=True,
        header_style="bold magenta", border_style="dim", expand=True,
    )
    rates_table.add_column("Model", style="cyan", no_wrap=True)
    rates_table.add_column("Input $/M", justify="right")
    rates_table.add_column("Output $/M", justify="right")
    for model in models:
        pricing = MODEL_PRICING.get(model)
        if pricing:
            rates_table.add_row(
                short_model(model),
                f"${pricing[0]:.2f}",
                f"${pricing[1]:.2f}",
            )
        else:
            rates_table.add_row(short_model(model), "[red]?[/red]", "[red]?[/red]")

    # -- domain tokens table (right) ------------------------------------------
    tokens_table = Table(
        title="Domain Tokens", show_header=True,
        header_style="bold magenta", border_style="dim", expand=True,
    )
    tokens_table.add_column("Domain", style="cyan", no_wrap=True)
    tokens_table.add_column("Questions", justify="right")
    tokens_table.add_column("Tok In", justify="right")
    tokens_table.add_column("Tok Out", justify="right")
    for ds in domain_stats:
        tokens_table.add_row(
            ds["domain"],
            f"{ds['questions']:,}",
            f"{ds['tok_in']:,}",
            f"{ds['tok_out']:,}",
        )

    # -- cost matrix: domains (rows) x models (columns) -----------------------
    cost_table = Table(
        title="Estimated Cost by Domain × Model",
        show_header=True, header_style="bold magenta",
        border_style="dim", expand=True,
    )
    cost_table.add_column("Domain", style="cyan", no_wrap=True)
    for model in models:
        cost_table.add_column(short_model(model), justify="right")
    cost_table.add_column("Total", justify="right", style="bold")

    model_totals = {m: 0.0 for m in models}
    for ds in domain_stats:
        row_values = [ds["domain"]]
        row_total = 0.0
        for model in models:
            pricing = MODEL_PRICING.get(model)
            if pricing:
                cost = (ds["tok_in"] * pricing[0] + ds["tok_out"] * pricing[1]) / 1_000_000
                model_totals[model] += cost
                row_total += cost
                row_values.append(f"${cost:.4f}")
            else:
                row_values.append("[red]?[/red]")
        row_values.append(f"${row_total:.4f}")
        cost_table.add_row(*row_values)

    # totals footer row
    footer = ["[bold]Total[/bold]"]
    grand_total = 0.0
    for model in models:
        footer.append(f"[bold]${model_totals[model]:.4f}[/bold]")
        grand_total += model_totals[model]
    footer.append(f"[bold green]${grand_total:.4f}[/bold green]")
    cost_table.add_row(*footer, end_section=True)

    # -- assemble panel --------------------------------------------------------
    max_q_str = f"{max_q}" if max_q else "all"
    panel = Panel(
        Group(
            Columns([rates_table, tokens_table], expand=True, padding=(0, 2)),
            Text(""),
            cost_table,
        ),
        title="[bold cyan]BayesEval Cost Estimate[/bold cyan]",
        border_style="cyan",
        subtitle=f"[dim]{len(models)} model(s) · {len(domain_stats)} domain(s) · max_questions: {max_q_str}[/dim]",
    )
    console.print(panel)


if __name__ == "__main__":
    main()
