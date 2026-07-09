"""Run one (domain, model) task against a benchmark DataFrame."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .openrouter_client import OpenRouterClient


@dataclass
class RowResult:
    question_id: str
    answer: str | None
    confidence: float | None
    raw: str
    error: str | None
    tok_in: int = 0
    tok_out: int = 0


# ---------------------------------------------------------------------------
# SPD response parsers
# ---------------------------------------------------------------------------

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_spd_bins(raw: str) -> tuple[str | None, float | None, str | None]:
    """Extract modal bin from a JSON array of {min, max, confidence} objects."""
    if not raw:
        return None, None, "empty response"
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        return None, None, "no JSON array found"
    try:
        arr = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return None, None, f"JSON decode: {e}"
    if not isinstance(arr, list) or len(arr) == 0:
        return None, None, "empty or non-list JSON"
    best = None
    for item in arr:
        if not isinstance(item, dict):
            continue
        try:
            lo = float(item["min"])
            hi = float(item["max"])
            conf = float(item["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        if hi <= lo:
            continue
        if best is None or conf > best[2]:
            best = (lo, hi, conf)
    if best is None:
        return None, None, "no valid bin objects found"
    center = (best[0] + best[1]) / 2
    return str(center), best[2], None


def parse_spd_candidates(raw: str) -> tuple[str | None, float | None, str | None]:
    """Extract modal candidate from a JSON array of {pathology, confidence} objects."""
    if not raw:
        return None, None, "empty response"
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        return None, None, "no JSON array found"
    try:
        arr = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return None, None, f"JSON decode: {e}"
    if not isinstance(arr, list) or len(arr) == 0:
        return None, None, "empty or non-list JSON"
    best = None
    for item in arr:
        if not isinstance(item, dict):
            continue
        pathology = item.get("pathology")
        conf = item.get("confidence")
        if pathology is None or conf is None:
            continue
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            continue
        if best is None or conf_f > best[1]:
            best = (str(pathology), conf_f)
    if best is None:
        return None, None, "no valid candidate objects found"
    return best[0], best[1], None


# ---------------------------------------------------------------------------
# Standard runner
# ---------------------------------------------------------------------------

async def _run_one(
    client: OpenRouterClient,
    model: str,
    row: pd.Series,
    on_event,
    task_key: tuple[str, str],
) -> RowResult:
    image_path = row.get("image_path")
    image_uri = Path(image_path).read_text() if image_path else None
    resp = await client.complete(model, row["question_prompt"], row["confidence_prompt"],
                                 image_uri=image_uri)
    rr = RowResult(row["question_id"], resp.answer, resp.confidence, resp.raw, resp.error,
                   resp.tok_in, resp.tok_out)
    on_event(task_key, rr)
    return rr


# ---------------------------------------------------------------------------
# SPD runner
# ---------------------------------------------------------------------------

async def _run_one_spd(
    client: OpenRouterClient,
    model: str,
    row: pd.Series,
    on_event,
    task_key: tuple[str, str],
) -> RowResult:
    image_path = row.get("image_path")
    image_uri = Path(image_path).read_text() if image_path else None
    resp = await client.complete_raw(model, row["question_prompt"], row["confidence_prompt"],
                                     image_uri=image_uri)
    if resp.error:
        rr = RowResult(row["question_id"], None, None, resp.raw, resp.error,
                       resp.tok_in, resp.tok_out)
    else:
        has_bins = "bin_width" in row.index
        if has_bins:
            ans, conf, err = parse_spd_bins(resp.raw)
        else:
            ans, conf, err = parse_spd_candidates(resp.raw)
        rr = RowResult(row["question_id"], ans, conf, resp.raw, err,
                       resp.tok_in, resp.tok_out)
    on_event(task_key, rr)
    return rr


# ---------------------------------------------------------------------------
# Task entry point
# ---------------------------------------------------------------------------

async def run_task(
    client: OpenRouterClient,
    domain: str,
    model: str,
    benchmark: pd.DataFrame,
    mode: str,
    concurrency: int,
    on_event,
    spd: bool = False,
) -> pd.DataFrame:
    task_key = (domain, model)
    runner = _run_one_spd if spd else _run_one
    if mode == "seq":
        rows = [await runner(client, model, r, on_event, task_key) for _, r in benchmark.iterrows()]
    else:
        sem = asyncio.Semaphore(concurrency)

        async def guarded(r):
            async with sem:
                return await runner(client, model, r, on_event, task_key)

        rows = await asyncio.gather(*(guarded(r) for _, r in benchmark.iterrows()))
    return pd.DataFrame(
        [
            {
                "question_id": r.question_id,
                "Answer": r.answer,
                "Confidence": r.confidence,
                "raw": r.raw,
                "error": r.error,
            }
            for r in rows
        ]
    )
