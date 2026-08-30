"""CineVerdict Orchestration Pipeline.

This module defines and instantiates the multi-agent sequential evaluation pipeline
using the Google Agent Development Kit (ADK). It sequentially runs the evaluation
from initial planning through live research, market validation, feasibility assessment,
and final strategic recommendation.
"""

from google.adk.agents.sequential_agent import SequentialAgent

# Import the individual specialist agents comprising the CineVerdict pipeline
from .agents.director import director_agent
from .agents.research import research_agent
from .agents.market import market_agent
from .agents.production_risk import production_risk_agent
from .agents.verdict import verdict_agent


# Orchestrate the five specialized agents in sequence.
# Each agent's output feeds the next, establishing a strict evidence-driven decision chain.
root_agent = SequentialAgent(
    name="cineverdict_pipeline",
    description="Sequential multi-agent pipeline for structured media and film evaluation.",
    sub_agents=[
        director_agent,
        research_agent,
        market_agent,
        production_risk_agent,
        verdict_agent,
    ],
)
