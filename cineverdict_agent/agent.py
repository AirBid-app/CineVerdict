from google.adk.agents.sequential_agent import SequentialAgent

from .agents.director import director_agent
from .agents.research import research_agent
from .agents.market import market_agent
from .agents.production_risk import production_risk_agent
from .agents.verdict import verdict_agent

root_agent = SequentialAgent(
    name="cineverdict_pipeline",
    description="CineVerdict multi-agent evaluation pipeline.",
    sub_agents=[
        director_agent,
        research_agent,
        market_agent,
        production_risk_agent,
        verdict_agent,
    ],
)
