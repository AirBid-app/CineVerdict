import json
import os
import threading
import time

from dotenv import load_dotenv
from parallel import Parallel


load_dotenv("cineverdict_agent/.env", override=True)

DEFAULT_PARALLEL_TIMEOUT_SECONDS = 30.0
MAX_PARALLEL_SEARCHES_PER_RESEARCH_BURST = 6
PARALLEL_BUDGET_IDLE_RESET_SECONDS = 15.0

_budget_lock = threading.Lock()
_budget_count = 0
_budget_last_completed_at = 0.0


def _parallel_timeout_seconds() -> float:
    raw_value = os.getenv("PARALLEL_SEARCH_TIMEOUT_SECONDS")
    if not raw_value:
        return DEFAULT_PARALLEL_TIMEOUT_SECONDS

    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_PARALLEL_TIMEOUT_SECONDS

    return max(5.0, min(value, 120.0))


def _claim_search_budget() -> tuple[bool, int]:
    """Reserve one search slot for the current active Research burst.

    A quiet gap resets the counter so later evaluations get a fresh budget.
    The counter is enforced in code, not only by agent instructions.
    """
    global _budget_count, _budget_last_completed_at

    now = time.monotonic()
    with _budget_lock:
        if (
            _budget_last_completed_at
            and now - _budget_last_completed_at >= PARALLEL_BUDGET_IDLE_RESET_SECONDS
        ):
            _budget_count = 0

        if _budget_count >= MAX_PARALLEL_SEARCHES_PER_RESEARCH_BURST:
            return False, _budget_count

        _budget_count += 1
        return True, _budget_count


def _mark_search_completed() -> None:
    global _budget_last_completed_at
    with _budget_lock:
        _budget_last_completed_at = time.monotonic()


def parallel_search(query: str, domain: str | None = None) -> str:
    """Search the live web with Parallel and optionally restrict results to one domain.

    Each request has a hard client timeout, and each active Research burst has a
    hard six-call budget. Tool failures and budget exhaustion are returned as
    structured JSON so Research can mark the evidence unresolved instead of
    hanging or silently exceeding the contract.
    """
    allowed, search_number = _claim_search_budget()
    if not allowed:
        print("[Parallel] search budget exhausted (max=6)")
        return json.dumps(
            {
                "ok": False,
                "error": "Parallel search budget exhausted for this research burst.",
                "query": query,
                "domain": domain,
                "max_searches": MAX_PARALLEL_SEARCHES_PER_RESEARCH_BURST,
            }
        )

    timeout_seconds = _parallel_timeout_seconds()
    print(
        f"[Parallel] search invoked "
        f"(call={search_number}/{MAX_PARALLEL_SEARCHES_PER_RESEARCH_BURST}, "
        f"timeout={timeout_seconds:g}s"
        + (f", domain={domain}" if domain else "")
        + ")"
    )

    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        _mark_search_completed()
        return json.dumps(
            {
                "ok": False,
                "error": "Parallel API key missing.",
                "query": query,
                "domain": domain,
            }
        )

    client = Parallel(api_key=api_key)
    search_kwargs = {
        "search_queries": [query],
    }

    if domain:
        search_kwargs["advanced_settings"] = {
            "source_policy": {
                "include_domains": [domain],
            }
        }

    try:
        result = client.search(**search_kwargs, timeout=timeout_seconds)
    except Exception as exc:
        print(f"[Parallel] search failed: {type(exc).__name__}: {exc}")
        return json.dumps(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "query": query,
                "domain": domain,
                "timeout_seconds": timeout_seconds,
            }
        )
    finally:
        _mark_search_completed()

    return result.model_dump_json(indent=2, exclude_none=True)
