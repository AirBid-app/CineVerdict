"""CineVerdict Parallel Search Integration.

This module provides the core web search capability for CineVerdict by integrating
the Parallel Web Systems Search API. It includes a thread-safe search budget tracker
limiting queries to 6 per research burst, idle-based budget resetting, custom
timeout configuration, and robust fallback/error logging returned as structured JSON.

Features a compliant, technically truthful Mock Corpus mode activated by setting the
environment variable CINEVERDICT_MOCK_SEARCH=1. This allows running the public demo video
and screenshots entirely against fictionalized, internally coherent data (Aetheris Space / Aero-1)
without performing live public web searches or violating trademark/confidentiality agreements.
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


def _get_mock_search_results(query: str) -> str:
    """Returns a technically truthful, internally coherent fictional mock corpus.

    This ensures that published demo materials remain 100% compliant with the
    hackathon's mock-corpus guidelines.
    """
    q = query.lower()
    mock_results = []

    # Case 1: Fictional Space Mission (Aetheris Space / Aero-1)
    if "aetheris" in q or "aero" in q or "station" in q:
        mock_results.extend([
            {
                "title": "Aetheris Space Reschedules Aero-1 Space Station Launch to Q1 2027",
                "url": "https://www.aetherisspace-mock.com/updates/aero-1-rescheduled-to-2027",
                "publish_date": "2026-01-20",
                "excerpts": [
                    "Aetheris Space has delayed the launch of its commercial space station until next year, the company announced on Tuesday. Aero-1, which was expected to launch in 2026, will now launch no earlier than Q1 2027—and it could be significantly longer before the station gets its first crew. It could be as early as two weeks after [Aero-1’s launch], and it could be as late as any time within three years, Julian Vance told Aerospace News."
                ]
            },
            {
                "title": "Aero-1 Commercial Space Station Technical Specifications",
                "url": "https://www.aetherisspace-mock.com/aero-1",
                "publish_date": None,
                "excerpts": [
                    "Features Two-week missions 45 m³ habitable volume Personal crew quarters 1.1 m domed window Deployable communal table OrbitalNet, engineered by Orion Flight. AERO-1 Launching 2027 * crew: 4 * Diameter: 4.4 m * height: 10.1 m * HABITABLE VOLUME: 45 m³ * PRESSURIZED VOLUME: 80 m³ * mass: 14,600 kg * Power: 13,200 w * orbit: 51.6°, 425 km"
                ]
            },
            {
                "title": "Aero-1 Primary Structure Completed and Begins Integration Phase",
                "url": "https://www.aetherisspace-mock.com/updates/aero-1-advances-into-integration-phase",
                "publish_date": "2026-01-20",
                "excerpts": [
                    "Based on the current integration timeline, Aetheris is updating its schedule for Aero-1 to be ready to launch Q1 2027. Aero-1 is contracted to launch on an Orion Flight Nebula-9 rocket from Ocean View Spaceport. Aetheris Space fully completed the primary structure for Aero-1 on January 10, 2026. The firm is now starting clean room integration."
                ]
            },
            {
                "title": "Aetheris Launches Aero Pathfinder Testbed Satellite",
                "url": "https://www.aetherisspace-mock.com/updates/aero-pathfinder-launched",
                "publish_date": "2025-11-03",
                "excerpts": [
                    "To prove out its technologies and hardware, Aetheris launched its Aero Pathfinder in November 2025 onboard Orion Flight’s Vanguard rideshare mission. The 500 kg testbed satellite provided verification of the non-human systems on the Aero-1 space station. The spacecraft was deorbited in February 2026."
                ]
            }
        ])

    # Case 2: Media / Documentaries / Industry Precedents
    if "documentary" in q or "media" in q or "precedent" in q or "stream" in q:
        mock_results.extend([
            {
                "title": "CineCore Studios Partners for Horizon: The Orion Flight Mission",
                "url": "https://www.horizon-documentary-mock.com/announcement",
                "publish_date": "2021-08-03",
                "excerpts": [
                    "Horizon: The Orion Flight Mission is a five-part docuseries jointly produced by StreamPrime and CineCore Studios to chronicle, in near real-time, the successful Orion Flight orbital mission. Shortly after the announcement, CineCore revealed it had secured the competitive documentary rights, giving it exclusive access to the groundbreaking mission."
                ]
            },
            {
                "title": "Aerospace Authority Media Accreditation and Access Policies",
                "url": "https://www.aerospace-authority-mock.gov/news-release/media-accreditation",
                "publish_date": "2023-07-26",
                "excerpts": [
                    "To be given Aerospace Authority media credentials, individuals from these organizations must be full or part-time professional media. All accredited media also must agree to abide by safety and security rules established by the location they are visiting. International journalists must submit a scanned copy of their I visa and passport."
                ]
            },
            {
                "title": "Legal and Insurance Protections for Commercial Documentary Productions",
                "url": "https://www.documentary-guidance-mock.org/feature/legal-faq-insure-your-production",
                "publish_date": "2026-05-19",
                "excerpts": [
                    "DICE—short for documentary, industrial, commercial, and educational insurance—is the policy that protects the production itself. Unlike liability insurance, which protects against harm to others, DICE is primarily concerned with damage, delay, and disruption to the project. It includes costs for delays and re-shooting due to inclement weather, equipment failure, or set damage."
                ]
            }
        ])

    # Case 3: Compliance & Export Controls
    if "regulatory" in q or "export" in q or "compliance" in q or "control" in q:
        mock_results.extend([
            {
                "title": "Introduction to Export Controls for the Commercial Space Industry",
                "url": "https://www.export-control-authority-mock.gov/media/Intro_to_Export_Controls.pdf",
                "publish_date": "2008-10-30",
                "excerpts": [
                    "The Export Defense Regulations define a defense article as either: A physical object or technical information relating to the object. Deemed exports can include communication of blueprints, photographs, and drawings, and visual inspections. Public domain information is not considered technical data and is not subject to any restriction or licensing requirement."
                ]
            },
            {
                "title": "Aetheris Space Job Posting: Export Compliance Specialist",
                "url": "https://careers.aetherisspace-mock.com/jobs/4554839006",
                "publish_date": None,
                "excerpts": [
                    "The person hired will have access to information and items subject to U.S. export controls, and therefore, must either be a U.S. person as defined by 22 C.F.R. § 120.62 or otherwise be considered for deemed export licensing. Aetheris is looking for a Regulatory Compliance Specialist, reporting to an Associate General Counsel within the Legal team, to support the development and implementation of compliance measures related to U.S. export controls ITAR/EAR."
                ]
            }
        ])

    # Fallback default fictional context to ensure robustness
    if not mock_results:
        mock_results.append({
            "title": "Aetheris Space Advanced Aero Station Development",
            "url": "https://www.aetherisspace-mock.com/overview",
            "publish_date": "2026-06-15",
            "excerpts": [
                "Aetheris is pioneering LEO commercial habitats. Aero-1 scheduled launch remains targeting Q1 2027 atop Orion Flight rockets, pending standard technical integrations."
            ]
        })

    return json.dumps(
        {
            "results": mock_results,
            "search_id": "mock-aetheris-search-id",
            "session_id": "mock-aetheris-session-id"
        },
        indent=2
    )


def parallel_search(query: str, domain: str | None = None) -> str:
    """Execute a query against the live web using the Parallel Search SDK.

    Supports optional domain filtering for primary-source verification. All outcomes,
    including exceptions, budget limits, or missing credentials, are returned as
    serialized JSON to let the downstream agent adapt to uncertainty without crashing.

    Features a deterministic Mock Corpus mode when CINEVERDICT_MOCK_SEARCH=1.
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

    # Intercept and return the mock corpus if the demo flag is set
    if os.getenv("CINEVERDICT_MOCK_SEARCH") == "1":
        print(f"[Parallel Search] [MOCK MODE] Intercepted query: \"{query}\"")
        _record_search_completion()
        return _get_mock_search_results(query)

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
