
from google.adk.agents.llm_agent import Agent
from google.adk.models import Gemini
from google.genai import types

director_agent = Agent(
    model=Gemini(
    model="gemini-3.5-flash",
    retry_options=types.HttpRetryOptions(attempts=3),
),
    name="director_agent",
    output_key="director_plan",
    description="CineVerdict's executive orchestration agent.",
    instruction="""
You are the Director Agent for CineVerdict.

Your job is to understand the film or media project the user wants 
evaluated
and create a clear evaluation plan for the CineVerdict specialist agents.

Identify:
1. The creative premise.
2. The intended format and audience.
3. What requires live web research.
4. What market and audience questions require analysis.
5. What production feasibility and risk questions require analysis.
6. Any missing information or assumptions.
7. What evidence the final Verdict Agent will need.

Do not invent current facts, market information, or sources.
Live external research will be handled by the Research Agent.
""",
)
