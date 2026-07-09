#!/usr/bin/env python3
"""BayesEval cross-domain runner. Driven entirely by a YAML config.

Usage:
    python eval.py --config config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import yaml

from analysis.scoring import get_scorer
from src.runner.dashboard import Dashboard
from src.runner.executor import run_task
from src.runner.openrouter_client import OpenRouterClient

REPO_ROOT = Path(__file__).resolve().parent
DOMAINS_DIR = REPO_ROOT / "domains"


def _normalize_domains(raw) -> list[dict]:
    """Normalize domain entries to dicts with 'name' and optional 'benchmark_file'."""
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
    cfg.setdefault("mode", "batch")
    cfg.setdefault("concurrency", 8)
    cfg.setdefault("max_questions", None)
    cfg.setdefault("output_dir", "results")
    cfg.setdefault("openrouter", {})
    cfg["openrouter"].setdefault("api_key_env", "OPENROUTER_API_KEY")
    cfg["openrouter"].setdefault("base_url", "https://openrouter.ai/api/v1")
    cfg["openrouter"].setdefault("timeout_s", 60)
    cfg["openrouter"].setdefault("max_retries", 3)
    return cfg


def load_domain(name: str, benchmark_file: str = "benchmark.csv"):
    """Return (benchmark_df, score_fn) for a domain."""
    dir_name = name.removesuffix("_SPD")
    domain_dir = DOMAINS_DIR / dir_name
    benchmark_path = domain_dir / "Data" / benchmark_file
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"{name}: missing benchmark at {benchmark_path}. "
            f"Run its build_benchmark.py first."
        )
    score_fn = get_scorer(name)
    benchmark = pd.read_csv(benchmark_path)

    if "photo" in benchmark.columns:
        encoded_dir = domain_dir / "Data" / "Photos-encoded"
        manifest_path = domain_dir / "Data" / "image_sizes.json"
        photos = benchmark["photo"].tolist()
        manifest = _read_manifest(manifest_path)
        need = [p for p in set(photos) if Path(p).stem not in manifest]
        if need:
            print(f"Encoding {len(need)} photos for {name}…")
            enc_mod = _load_encoder(domain_dir)
            enc_mod.encode_all(
                domain_dir / "Data" / "Photos", encoded_dir, manifest_path,
            )
        benchmark["image_path"] = benchmark["photo"].apply(
            lambda p: str(encoded_dir / f"{Path(p).stem}.txt")
        )

    return benchmark, score_fn


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    import json
    return json.loads(path.read_text())


def _load_encoder(domain_dir: Path):
    encoder_path = domain_dir / "image_encoder.py"
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Domain has 'photo' column but no image_encoder.py at {encoder_path}"
        )
    spec = importlib.util.spec_from_file_location("_img_enc", encoder_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def slugify(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)


async def _main(cfg: dict) -> None:
    api_key = os.environ.get(cfg["openrouter"]["api_key_env"])
    if not api_key:
        sys.exit(f"Missing env var {cfg['openrouter']['api_key_env']}")

    output_dir = REPO_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_data: dict[str, tuple[pd.DataFrame, object, bool]] = {}
    for d in cfg["domains"]:
        bench_file = d.get("benchmark_file", "benchmark.csv")
        bench, score_fn = load_domain(d["name"], bench_file)
        if cfg["max_questions"]:
            bench = bench.head(cfg["max_questions"]).reset_index(drop=True)
        spd = d.get("spd", False)
        domain_data[d["name"]] = (bench, score_fn, spd)

    client = OpenRouterClient(
        api_key=api_key,
        base_url=cfg["openrouter"]["base_url"],
        timeout_s=cfg["openrouter"]["timeout_s"],
        max_retries=cfg["openrouter"]["max_retries"],
    )

    title = "BayesEval"
    summary_rows: list[dict] = []
    try:
        with Dashboard(title) as dash:
            for domain, (bench, _, _) in domain_data.items():
                for model in cfg["models"]:
                    dash.register(domain, model, total=len(bench))

            def on_event(key, row_result):
                dash.record(
                    key[0], key[1],
                    error=row_result.error is not None,
                    brier=None,
                    question_id=str(row_result.question_id),
                    answer=row_result.answer,
                    tok_in=row_result.tok_in,
                    tok_out=row_result.tok_out,
                )

            async def process_domain(domain, bench, score_fn, spd=False):
                out_domain = output_dir / domain
                out_domain.mkdir(parents=True, exist_ok=True)
                tasks = [
                    run_task(client, domain, model, bench, cfg["mode"],
                             cfg["concurrency"], on_event, spd=spd)
                    for model in cfg["models"]
                ]
                results_list = list(await asyncio.gather(*tasks))

                meta_cols = [c for c in bench.columns
                             if c not in {"question_prompt", "confidence_prompt",
                                          "gold_response", "image_path"}]
                domain_rows = []
                for model, results in zip(cfg["models"], results_list):
                    out_path = out_domain / f"{slugify(model)}.csv"
                    merged = results.merge(bench[meta_cols], on="question_id", how="left")
                    merged.to_csv(out_path, index=False)
                    scored = score_fn(results, bench)
                    mean_brier = float(scored["brier"].dropna().mean()) if "brier" in scored else float("nan")
                    err_count = int(results["error"].notna().sum())
                    domain_rows.append({
                        "domain": domain,
                        "model": model,
                        "n": len(results),
                        "errors": err_count,
                        "mean_brier": mean_brier,
                        "results_csv": str(out_path.relative_to(REPO_ROOT)),
                    })
                return domain_rows

            all_domain_rows = await asyncio.gather(*[
                process_domain(domain, bench, score_fn, spd=spd)
                for domain, (bench, score_fn, spd) in domain_data.items()
            ])
            for rows in all_domain_rows:
                summary_rows.extend(rows)
            dash.refresh()
    finally:
        await client.aclose()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


def _scan_errors(cfg: dict, output_dir: Path) -> dict[str, dict[str, set[str]]]:
    """Lightweight scan: read result CSVs and collect errored question_ids per (domain, model).

    Returns {domain: {model: {question_id, ...}}} only for pairs with errors.
    Does NOT load benchmarks or images.
    """
    errors_by_domain: dict[str, dict[str, set[str]]] = {}
    for d in cfg["domains"]:
        domain = d["name"]
        for model in cfg["models"]:
            results_path = output_dir / domain / f"{slugify(model)}.csv"
            if not results_path.exists():
                continue
            prev = pd.read_csv(results_path, usecols=["question_id", "error"])
            errored = prev[prev["error"].notna()]
            if errored.empty:
                continue
            errors_by_domain.setdefault(domain, {})[model] = set(errored["question_id"])
    return errors_by_domain


async def _retry_main(cfg: dict) -> None:
    """Load existing results, retry only errored questions, update files.

    Processes one domain at a time to avoid holding all benchmarks + encoded
    images in memory simultaneously (important for image-heavy domains like WGD).
    """
    api_key = os.environ.get(cfg["openrouter"]["api_key_env"])
    if not api_key:
        sys.exit(f"Missing env var {cfg['openrouter']['api_key_env']}")

    output_dir = REPO_ROOT / cfg["output_dir"]

    errors_by_domain = _scan_errors(cfg, output_dir)
    if not errors_by_domain:
        print("No errors to retry.")
        return

    total_retries = sum(
        len(ids) for models in errors_by_domain.values() for ids in models.values()
    )
    n_pairs = sum(len(models) for models in errors_by_domain.values())
    print(f"Retrying {total_retries} errored questions across {n_pairs} model/domain pairs")

    domain_cfgs = {d["name"]: d for d in cfg["domains"]}

    client = OpenRouterClient(
        api_key=api_key,
        base_url=cfg["openrouter"]["base_url"],
        timeout_s=cfg["openrouter"]["timeout_s"],
        max_retries=cfg["openrouter"]["max_retries"],
    )

    summary_rows: list[dict] = []
    try:
        with Dashboard("BayesEval · Retry") as dash:
            for domain, model_errors in errors_by_domain.items():
                for model, errored_ids in model_errors.items():
                    dash.register(domain, model, total=len(errored_ids))

            def on_event(key, row_result):
                dash.record(
                    key[0], key[1],
                    error=row_result.error is not None,
                    brier=None,
                    question_id=str(row_result.question_id),
                    answer=row_result.answer,
                    tok_in=row_result.tok_in,
                    tok_out=row_result.tok_out,
                )

            for domain, model_errors in errors_by_domain.items():
                bench_file = domain_cfgs[domain].get("benchmark_file", "benchmark.csv")
                bench, score_fn = load_domain(domain, bench_file)
                spd = domain_cfgs[domain].get("spd", False)

                retry_tasks = []
                models_to_retry = []
                for model, errored_ids in model_errors.items():
                    retry_bench = bench[bench["question_id"].isin(errored_ids)].reset_index(drop=True)
                    retry_tasks.append(
                        run_task(client, domain, model, retry_bench, cfg["mode"],
                                 cfg["concurrency"], on_event, spd=spd)
                    )
                    models_to_retry.append(model)

                retry_results = await asyncio.gather(*retry_tasks)

                meta_cols = [c for c in bench.columns
                             if c not in {"question_prompt", "confidence_prompt",
                                          "gold_response", "image_path"}]
                results_cols = ["question_id", "Answer", "Confidence", "raw", "error"]
                for model, retry_res in zip(models_to_retry, retry_results):
                    out_path = output_dir / domain / f"{slugify(model)}.csv"
                    prev = pd.read_csv(out_path)
                    ok = prev[prev["error"].isna()][results_cols]
                    merged = pd.concat([ok, retry_res], ignore_index=True)
                    merged_with_meta = merged.merge(bench[meta_cols], on="question_id", how="left")
                    merged_with_meta.to_csv(out_path, index=False)
                    scored = score_fn(merged, bench)
                    mean_brier = float(scored["brier"].dropna().mean()) if "brier" in scored else float("nan")
                    err_count = int(merged["error"].notna().sum())
                    summary_rows.append({
                        "domain": domain,
                        "model": model,
                        "n": len(merged),
                        "errors": err_count,
                        "mean_brier": mean_brier,
                        "results_csv": str(out_path.relative_to(REPO_ROOT)),
                    })

            dash.refresh()
    finally:
        await client.aclose()

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        print("\n" + summary.to_string(index=False))


def _dummy_main() -> None:
    """Show dashboard with fake data for screenshots — no API calls."""
    import random

    from rich.console import Console

    random.seed(42)

    domains = {
        "LifeEval": 186, "LifeEval_SPD": 186,
        "WGD": 928, "WGD_SPD": 928,
        "MedEval": 312, "MedEval_SPD": 312,
    }
    models = [
        "google/gemini-2.5-flash",
        "openai/gpt-5.4-mini",
        "anthropic/claude-haiku-4.5",
        "meta-llama/llama-4-maverick",
    ]
    pricing = {
        "google/gemini-2.5-flash": (0.15e-6, 0.60e-6),
        "openai/gpt-5.4-mini": (0.30e-6, 1.20e-6),
        "anthropic/claude-haiku-4.5": (0.80e-6, 4.00e-6),
        "meta-llama/llama-4-maverick": (0.20e-6, 0.60e-6),
    }

    dash = Dashboard("BayesEval", show_price=True, pricing=pricing, static=True)
    for domain, total in domains.items():
        for model in models:
            dash.register(domain, model, total=total)

    for (domain, model), st in dash._tasks.items():
        done = random.randint(int(st.total * 0.4), st.total)
        errors = random.randint(0, max(1, int(done * 0.02)))
        st.done = done
        st.errors = errors
        st.tok_in = done * random.randint(900, 1800)
        st.tok_out = done * random.randint(150, 350)
        dash._progress.update(
            st.progress_task_id, completed=done,
            speed=random.uniform(8.0, 15.0),
        )

    done_total = sum(s.done for s in dash._tasks.values())
    dash._progress.update(
        dash._overall_task, completed=done_total,
        speed=random.uniform(10.0, 14.0),
    )

    sample_answers = ["Yes", "No", "0.73", "Positive", "A", "B", "0.85", "Negative"]
    for _ in range(6):
        model = random.choice(models)
        short = model.split("/")[-1]
        qid = f"Q{random.randint(1, 500):04d}"
        if random.random() < 0.15:
            dash._recent.append(
                f"[red]✗[/red] [dim]{qid}[/dim] [cyan]{short}[/cyan] → [red]error[/red]"
            )
        else:
            ans = random.choice(sample_answers)
            dash._recent.append(
                f"[green]✓[/green] [dim]{qid}[/dim] [cyan]{short}[/cyan] → [bold]{ans}[/bold]"
            )

    Console().print(dash)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-run only errored questions from existing results")
    ap.add_argument("--dummy", action="store_true",
                    help="Show dashboard with fake data for screenshots (no API calls)")
    args = ap.parse_args()
    if args.dummy:
        _dummy_main()
        return
    cfg = load_config(args.config)
    if args.retry_errors:
        asyncio.run(_retry_main(cfg))
    else:
        asyncio.run(_main(cfg))


if __name__ == "__main__":
    main()
