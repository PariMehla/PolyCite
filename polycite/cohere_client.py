"""Budget-capped, cached, rate-limited wrapper around cohere.ClientV2.

Every Cohere call in this repo must go through `CohereClient`, never through
a bare `cohere.ClientV2()`. This is the one piece that keeps a 1,000
call/month trial key alive across a full day of development: it caches every
response on disk (so re-running a script after a crash costs nothing), hard
stops at BUDGET, rate-limits per endpoint, and makes you confirm before any
batch that would spend more than a handful of calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cohere

DEFAULT_BUDGET = int(os.environ.get("POLYCITE_COHERE_BUDGET", "1000"))
DEFAULT_CACHE_DIR = Path(os.environ.get("POLYCITE_CACHE_DIR", "cache"))

# Requests/minute per endpoint on a Cohere trial key, per Cohere's published
# trial rate limits (Sept 2025). If your key's actual limits differ, override
# via CohereClient(rate_limits={...}).
DEFAULT_RATE_LIMITS = {"chat": 20, "embed": 100, "rerank": 10, "tokenize": 20}

EMBED_MAX_TEXTS_PER_CALL = 96


class BudgetExceededError(RuntimeError):
    pass


class BatchNotConfirmedError(RuntimeError):
    pass


def _hash_key(endpoint: str, model: str, params: dict) -> str:
    payload = json.dumps({"endpoint": endpoint, "model": model, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _RateLimiter:
    """Simple per-endpoint sliding window limiter with 429 backoff."""

    def __init__(self, calls_per_minute: dict[str, int]):
        self._limits = calls_per_minute
        self._history: dict[str, list[float]] = {k: [] for k in calls_per_minute}

    def wait_for_slot(self, endpoint: str) -> None:
        limit = self._limits.get(endpoint)
        if not limit:
            return
        history = self._history.setdefault(endpoint, [])
        now = time.monotonic()
        window_start = now - 60.0
        while history and history[0] < window_start:
            history.pop(0)
        if len(history) >= limit:
            sleep_for = 60.0 - (now - history[0]) + 0.05
            if sleep_for > 0:
                time.sleep(sleep_for)
        history.append(time.monotonic())

    def record_429(self, endpoint: str, attempt: int) -> float:
        # Cohere enforces a token-volume limit (e.g. 100k tokens/min for
        # embed on a trial key) as well as the per-endpoint request-count
        # limit this class otherwise paces against. A burst of large
        # embed batches can blow the token budget while staying well under
        # the request-count cap, so backoff here needs to be able to ride
        # out a full ~60s token-bucket reset, not just a few seconds.
        backoff = min(2**attempt, 60)
        time.sleep(backoff)
        return backoff


@dataclass
class CallCounter:
    path: Path
    _count: int = field(init=False, default=0)

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._count = json.loads(self.path.read_text()).get("count", 0)
        else:
            self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps({"count": self._count}))

    @property
    def count(self) -> int:
        return self._count

    def increment(self) -> int:
        self._count += 1
        self._save()
        return self._count


class DiskCache:
    def __init__(self, cache_dir: Path):
        self.dir = cache_dir / "responses"
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[dict]:
        f = self.dir / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())
        return None

    def set(self, key: str, value: dict) -> None:
        f = self.dir / f"{key}.json"
        f.write_text(json.dumps(value))


def estimate_and_confirm(n_calls: int, description: str, auto_confirm: Optional[bool] = None) -> None:
    """Print the call cost of a batch job and require confirmation.

    Set POLYCITE_AUTO_CONFIRM=1 (or pass auto_confirm=True) for non-interactive
    runs, e.g. CI or `make reproduce`. Anything else must be confirmed with
    'y' on stdin so a runaway loop can't silently drain a trial key.
    """
    print(f"[polycite] About to make ~{n_calls} Cohere calls for: {description}", file=sys.stderr)
    if auto_confirm is None:
        auto_confirm = os.environ.get("POLYCITE_AUTO_CONFIRM") == "1"
    if auto_confirm:
        return
    if not sys.stdin.isatty():
        raise BatchNotConfirmedError(
            f"Refusing to spend ~{n_calls} calls on '{description}' without confirmation. "
            "Set POLYCITE_AUTO_CONFIRM=1 or run interactively."
        )
    reply = input(f"Proceed with {n_calls} calls? [y/N] ").strip().lower()
    if reply != "y":
        raise BatchNotConfirmedError(f"User declined batch: {description}")


class CohereClient:
    """Cached, budgeted, rate-limited facade over cohere.ClientV2.

    Usage:
        client = CohereClient()
        resp = client.embed(model="embed-multilingual-v3.0", input_type="search_document", texts=[...])
        resp = client.rerank(model="rerank-v3.5", query=q, documents=docs, top_n=5)
        resp = client.chat(model="command-a", messages=[...], documents=[...], temperature=0)

    Every call is cached on (endpoint, model, params) — a re-run of an
    identical call costs zero budget and zero latency.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        budget: int = DEFAULT_BUDGET,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        rate_limits: Optional[dict[str, int]] = None,
        transport: Optional[Callable[[str, str, dict], dict]] = None,
    ):
        self.budget = budget
        cache_dir = Path(cache_dir)
        self.cache = DiskCache(cache_dir)
        self.counter = CallCounter(cache_dir / "call_count.json")
        self.limiter = _RateLimiter(rate_limits or DEFAULT_RATE_LIMITS)
        self._transport = transport  # injected in tests; bypasses the real SDK
        self._client: Optional[cohere.ClientV2] = None
        self._api_key = api_key

    def _sdk_client(self) -> cohere.ClientV2:
        if self._client is None:
            key = self._api_key or os.environ.get("COHERE_API_KEY")
            if not key:
                raise RuntimeError(
                    "COHERE_API_KEY is not set. Get a free trial key at "
                    "https://dashboard.cohere.com/api-keys (no payment method required, "
                    "1,000 calls/month) and `export COHERE_API_KEY=...`."
                )
            self._client = cohere.ClientV2(api_key=key)
        return self._client

    def _call(self, endpoint: str, model: str, params: dict, live_fn: Callable[[], Any]) -> dict:
        key = _hash_key(endpoint, model, params)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if self.counter.count >= self.budget:
            raise BudgetExceededError(
                f"Cohere call budget of {self.budget} exhausted "
                f"({self.counter.count} calls made this month per cache/call_count.json). "
                "Wait for the trial key to reset, upgrade, or apply for a Cohere Labs Catalyst Grant."
            )

        if self._transport is not None:
            result = self._transport(endpoint, model, params)
        else:
            self.limiter.wait_for_slot(endpoint)
            attempt = 0
            max_attempts = 7  # cumulative backoff (capped at 60s/try) comfortably exceeds a 60s token-bucket reset window
            while True:
                try:
                    result = live_fn()
                    break
                except Exception as e:  # cohere SDK raises typed errors incl. 429
                    status = getattr(e, "status_code", None)
                    if status == 429 and attempt < max_attempts:
                        backoff = self.limiter.record_429(endpoint, attempt)
                        print(
                            f"[polycite] {endpoint} rate-limited (429), waited {backoff:.0f}s "
                            f"(retry {attempt + 1}/{max_attempts})",
                            file=sys.stderr,
                        )
                        attempt += 1
                        continue
                    raise
            result = result.dict() if hasattr(result, "dict") else json.loads(result.json())

        self.counter.increment()
        self.cache.set(key, result)
        return result

    def embed(self, *, model: str, input_type: str, texts: Sequence[str], **kwargs) -> dict:
        if len(texts) > EMBED_MAX_TEXTS_PER_CALL:
            raise ValueError(
                f"embed() called with {len(texts)} texts; batch at "
                f"{EMBED_MAX_TEXTS_PER_CALL} max per call (see polycite/retrieval/dense.py)."
            )
        params = {"input_type": input_type, "texts": list(texts), **kwargs}
        client = None if self._transport else self._sdk_client()
        return self._call(
            "embed",
            model,
            params,
            live_fn=(lambda: client.embed(model=model, input_type=input_type, texts=list(texts), **kwargs)),
        )

    def rerank(self, *, model: str, query: str, documents: Sequence[str], top_n: int = 5, **kwargs) -> dict:
        params = {"query": query, "documents": list(documents), "top_n": top_n, **kwargs}
        client = None if self._transport else self._sdk_client()
        return self._call(
            "rerank",
            model,
            params,
            live_fn=(lambda: client.rerank(model=model, query=query, documents=list(documents), top_n=top_n, **kwargs)),
        )

    def tokenize(self, *, text: str, model: str) -> dict:
        """offline=False forces a real API call instead of a local-tokenizer
        download, since this sandbox has no HF hub access. Used by
        analysis/tokenizer_fertility.py."""
        params = {"text": text}
        client = None if self._transport else self._sdk_client()
        return self._call(
            "tokenize",
            model,
            params,
            live_fn=(lambda: client.tokenize(text=text, model=model, offline=False)),
        )

    def chat(self, *, model: str, messages: list, documents: Optional[list] = None, temperature: float = 0.0, **kwargs) -> dict:
        params = {"messages": messages, "documents": documents, "temperature": temperature, **kwargs}
        client = None if self._transport else self._sdk_client()
        return self._call(
            "chat",
            model,
            params,
            live_fn=(lambda: client.chat(model=model, messages=messages, documents=documents, temperature=temperature, **kwargs)),
        )
