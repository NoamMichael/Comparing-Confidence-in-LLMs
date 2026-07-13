"""Async OpenRouter client. Chat Completions with JSON answer parsing."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

import httpx


@dataclass
class ModelResponse:
    answer: str | None
    confidence: float | None
    raw: str
    error: str | None
    tok_in: int = 0
    tok_out: int = 0


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_JSON_GREEDY_RE = re.compile(r"\{.*\}", re.DOTALL)
# Last-resort extraction when the JSON is malformed (e.g. unescaped quotes
# inside a Reasoning string): pull the trailing Answer/Confidence fields.
_ANSWER_RE = re.compile(r'"Answer"\s*:\s*"([^"]*)"\s*,')
_CONF_RE = re.compile(r'"Confidence"\s*:\s*"?([0-9.eE+-]+)"?\s*\}')
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", raw)


def _parse(raw: str) -> tuple[str | None, float | None, str | None]:
    if not raw:
        return None, None, "empty response"
    raw = _strip_fences(raw)
    match = _JSON_RE.search(raw)
    if not match:
        return None, None, "no JSON object found"
    # strict=False: models emit literal newlines inside the Reasoning string
    try:
        obj = json.loads(match.group(0), strict=False)
    except json.JSONDecodeError:
        # A Reasoning field containing "}" breaks the non-greedy match;
        # retry with the widest brace span before giving up.
        try:
            obj = json.loads(_JSON_GREEDY_RE.search(raw).group(0), strict=False)
        except (json.JSONDecodeError, AttributeError) as e:
            ans_m = _ANSWER_RE.search(raw)
            conf_m = _CONF_RE.search(raw.rstrip())
            if ans_m and conf_m:
                try:
                    return ans_m.group(1), float(conf_m.group(1)), None
                except ValueError:
                    pass
            return None, None, f"JSON decode: {e}"
    ans = obj.get("Answer")
    conf = obj.get("Confidence")
    if ans is None or conf is None:
        # e.g. Gemini sometimes returns {"Reasoning": "..."} with the answer
        # buried in prose — flag it so --retry-errors re-asks.
        missing = [k for k in ("Answer", "Confidence") if obj.get(k) is None]
        return ans, None, f"missing key(s): {', '.join(missing)}"
    try:
        conf_f = float(conf)
    except (TypeError, ValueError):
        return ans, None, f"bad confidence: {conf!r}"
    return ans, conf_f, None


class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str, timeout_s: float, max_retries: int,
                 ignore_providers: list[str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._provider = {"ignore": ignore_providers} if ignore_providers else None
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete_raw(self, model: str, question: str, confidence_prompt: str,
                           image_uri: str | None = None) -> ModelResponse:
        """Send prompt without appending standard JSON format. Returns raw text unparsed."""
        text = f"{question}\n\n{confidence_prompt}"
        if image_uri:
            content = [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": text},
            ]
        else:
            content = text
        messages = [{"role": "user", "content": content}]
        payload = {"model": model, "messages": messages}
        if self._provider:
            payload["provider"] = self._provider
        last_err: str | None = None
        for attempt in range(self.max_retries):
            try:
                r = await self._client.post(f"{self.base_url}/chat/completions", json=payload)
                if r.status_code >= 500 or r.status_code == 429:
                    last_err = f"http {r.status_code}: {r.text[:200]}"
                    await asyncio.sleep(1.5 ** attempt)
                    continue
                if r.status_code >= 400:
                    return ModelResponse(None, None, "", f"http {r.status_code}: {r.text[:500]}")
                r.raise_for_status()
                data = r.json()
                try:
                    raw = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    return ModelResponse(None, None, str(data)[:500],
                                         f"unexpected response: {str(data)[:200]}")
                usage = data.get("usage", {})
                tok_in = usage.get("prompt_tokens", 0)
                tok_out = usage.get("completion_tokens", 0)
                return ModelResponse(None, None, raw, None, tok_in, tok_out)
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(1.5 ** attempt)
        return ModelResponse(None, None, "", last_err or "exhausted retries")

    async def complete(self, model: str, question: str, confidence_prompt: str,
                       image_uri: str | None = None, reasoning: bool = False) -> ModelResponse:
        if reasoning:
            text = (
                f"{question}\n\n{confidence_prompt}\n\n"
                "Respond with ONLY a JSON object in this exact format:\n"
                '{"Reasoning": "<your step-by-step reasoning>", '
                '"Answer": "<your estimate>", '
                '"Confidence": "<probability between 0 and 1>"}\n'
                "No other text."
            )
        else:
            text = (
                f"{question}\n\n{confidence_prompt}\n\n"
                "Respond with ONLY a JSON object in this exact format:\n"
                '{"Answer": "<your estimate>", '
                '"Confidence": "<probability between 0 and 1>"}\n'
                "No other text."
            )
        if image_uri:
            content = [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": text},
            ]
        else:
            content = text
        messages = [{"role": "user", "content": content}]
        payload = {"model": model, "messages": messages}
        if self._provider:
            payload["provider"] = self._provider
        last_err: str | None = None
        for attempt in range(self.max_retries):
            try:
                r = await self._client.post(f"{self.base_url}/chat/completions", json=payload)
                if r.status_code >= 500 or r.status_code == 429:
                    last_err = f"http {r.status_code}: {r.text[:200]}"
                    await asyncio.sleep(1.5 ** attempt)
                    continue
                if r.status_code >= 400:
                    return ModelResponse(None, None, "", f"http {r.status_code}: {r.text[:500]}")
                r.raise_for_status()
                data = r.json()
                raw = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tok_in = usage.get("prompt_tokens", 0)
                tok_out = usage.get("completion_tokens", 0)
                ans, conf, perr = _parse(raw)
                return ModelResponse(ans, conf, raw, perr, tok_in, tok_out)
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(1.5 ** attempt)
        return ModelResponse(None, None, "", last_err or "exhausted retries")
