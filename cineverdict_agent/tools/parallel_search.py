"""CineVerdict Parallel Search Integration.

This module provides the core web search capability for CineVerdict by integrating
the Parallel Web Systems Search API. It includes a thread-safe search budget tracker
limiting queries to 6 per research burst, idle-based budget resetting, custom
timeout configuration, and robust fallback/error logging returned as structured JSON.
"""

import json
import os
import threading
import time

from dotenv import load_dotenv
from parallel import Parallel

# Securely load env file containing Parallel and Google Cloud keys
load_dotenv("cineverdict_agent/.env", override=True)

# Safe boundary settings for search execution
MAX_BURST_QUERIES = 6
IDLE_RESET_THRESHOLD_SEC = 15.0
DEFAULT_TIMEOUT_SEC = 30.0

# Thread-safe synchronization and budget metrics
_state_lock = threading.Lock()
_burst_query_count = 0
_last_completed_timestamp = 0.0


def _get_configured_timeout() -> float:
    """Read, parse, and clamp the search timeout from configuration settings."""
    env_timeout = os.getenv("PARALLEL_SEARCH_TIMEOUT_SECONDS")
    if not env_timeout:
        return DEFAULT_TIMEOUT_SEC
    try:
        parsed = float(env_timeout)
        # Enforce safe engineering limits
        return max(5.0, min(parsed, 120.0))
    except ValueError:
        return DEFAULT_TIMEOUT_SEC


def _acquire_search_slot() -> tuple[bool, int]:
    """Reserve a slot for the search call, resetting the tracking count if idle.

    Enforces a strict budget of max 6 queries per active burst to protect against API loops.
    """
    global _burst_query_count, _last_completed_timestamp
    now = time.monotonic()

    with _state_lock:
        # Check if the pipeline was idle long enough to reset the search budget
        if (
            _last_completed_timestamp > 0.0
            and now - _last_completed_timestamp >= IDLE_RESET_THRESHOLD_SEC
        ):
            _burst_query_count = 0

        if _burst_query_count >= MAX_BURST_QUERIES:
            return False, _burst_query_count

        _burst_query_count += 1
        return True, _burst_query_count


def _record_search_completion() -> None:
    """Track the exact timestamp when a search query completes."""
    global _last_completed_timestamp
    with _state_lock:
        _last_completed_timestamp = time.monotonic()


def parallel_search(query: str, domain: str | None = None) -> str:
    """Execute a query against the live web using the Parallel Search SDK.

    Supports optional domain filtering for primary-source verification. All outcomes,
    including exceptions, budget limits, or missing credentials, are returned as
    serialized JSON to let the downstream agent adapt to uncertainty without crashing.
    """
    is_budget_ok, current_call_index = _acquire_search_slot()
    if not is_budget_ok:
        print(f"[Parallel Search] Budget exhausted (Limit: {MAX_BURST_QUERIES})")
        return json.dumps(
            {
                "ok": False,
                "error": "Parallel search budget exhausted for this research burst.",
                "query": query,
                "domain": domain,
                "max_searches": MAX_BURST_QUERIES,
            }
        )

    timeout = _get_configured_timeout()
    domain_msg = f" restricted to {domain}" if domain else ""
    print(
        f"[Parallel Search] Initiating query #{current_call_index}/{MAX_BURST_QUERIES} "
        f"(Timeout: {timeout:g}s{domain_msg})"
    )

    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        _record_search_completion()
        return json.dumps(
            {
                "ok": False,
                "error": "Parallel API key missing.",
                "query": query,
                "domain": domain,
            }
        )

    # Initialize SDK client and search parameters
    client = Parallel(api_key=api_key)
    search_params = {
        "search_queries": [query],
    }

    if domain:
        search_params["advanced_settings"] = {
            "source_policy": {
                "include_domains": [domain],
            }
        }

    try:
        search_result = client.search(**search_params, timeout=timeout)
    except Exception as error:
        print(f"[Parallel Search] Execution failed: {type(error).__name__}: {error}")
        return json.dumps(
            {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "query": query,
                "domain": domain,
                "timeout_seconds": timeout,
            }
        )
    finally:
        _record_search_completion()

    return search_result.model_dump_json(indent=2, exclude_none=True)
