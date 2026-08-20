import json
import os

from dotenv import load_dotenv
from parallel import Parallel


load_dotenv("cineverdict_agent/.env", override=True)

DEFAULT_PARALLEL_TIMEOUT_SECONDS = 30.0


def _parallel_timeout_seconds() -> float:
    raw_value = os.getenv("PARALLEL_SEARCH_TIMEOUT_SECONDS")
    if not raw_value:
        return DEFAULT_PARALLEL_TIMEOUT_SECONDS

    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_PARALLEL_TIMEOUT_SECONDS

    return max(5.0, min(value, 120.0))


def parallel_search(query: str, domain: str | None = None) -> str:
    """Search the live web with Parallel and optionally restrict results to one domain.

    The request is bounded by a hard client timeout so one stalled search cannot
    block the CineVerdict pipeline indefinitely. Tool failures are returned as
    structured JSON for the Research Agent to mark as unresolved evidence.
    """
    timeout_seconds = _parallel_timeout_seconds()
    print(
        f"[Parallel] search invoked (timeout={timeout_seconds:g}s"
        + (f", domain={domain}" if domain else "")
        + ")"
    )

    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
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

    return result.model_dump_json(indent=2, exclude_none=True)
