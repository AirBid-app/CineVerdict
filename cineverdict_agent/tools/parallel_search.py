import os

from dotenv import load_dotenv
from parallel import Parallel


load_dotenv("cineverdict_agent/.env", override=True)


def parallel_search(query: str, domain: str | None = None) -> str:
    """Search the live web with Parallel and optionally restrict results 
to one domain."""
    print("[Parallel] search invoked")
    api_key = os.getenv("PARALLEL_API_KEY")

    if not api_key:
        return "Parallel API key missing."

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

    result = client.search(**search_kwargs)

    return result.model_dump_json(indent=2, exclude_none=True)
