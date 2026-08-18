# CineVerdict

CineVerdict is a multi-agent AI decision-support system for evaluating 
film, television, documentary, and other media projects using structured 
creative analysis, live web research, market assessment, production-risk 
analysis, and evidence-grounded final verdicts.

Built with Google Agent Development Kit (ADK), Gemini on Google Cloud, and 
Parallel Search.

## What CineVerdict Does

CineVerdict evaluates a project through five specialist AI agents.

### 1. Director Agent

Interprets the project and evaluation objective and establishes the 
creative and strategic analysis plan.

### 2. Research Agent

Performs live web research when current evidence is required.

The Research Agent:

- Uses Parallel Search for discovery.
- Verifies important factual claims against relevant first-party domains 
when possible.
- Separates primary-source verification from secondary-source evidence.
- Preserves source information.
- Identifies conflicting evidence.

### 3. Market Agent

Evaluates audience potential, positioning, competitive landscape, platform 
fit, and commercial considerations.

### 4. Production/Risk Agent

Evaluates production feasibility, scheduling, access, operational 
constraints, budget exposure, and material execution risks.

### 5. Verdict Agent

Synthesizes the preceding analysis into an evidence-grounded strategic 
recommendation.

## Architecture

CineVerdict currently operates as the following sequential pipeline:

User / Project Brief
        |
        v
Director Agent
        |
        v
Research Agent
        |
        +---- Parallel Search
        |       |
        |       +---- Live web discovery
        |       +---- Domain-restricted verification
        |
        v
Market Agent
        |
        v
Production/Risk Agent
        |
        v
Verdict Agent
        |
        v
CineVerdict Evaluation

The agents are orchestrated using Google ADK SequentialAgent.

## Technology

- Python
- Google Agent Development Kit (ADK)
- Gemini
- Google Cloud / Vertex AI
- Parallel Search API
- python-dotenv

## Evidence and Source Integrity

For important factual claims, CineVerdict is designed to:

- Prefer primary sources.
- Use unrestricted Parallel Search for discovery when necessary.
- Verify claims with domain-restricted Parallel searches against relevant 
first-party sources.
- Avoid marking a claim as primary-source verified when first-party 
evidence does not support it.
- Clearly label secondary-source evidence.
- Preserve source title, URL, publication date when available, and 
supporting excerpts.
- Identify conflicts when sources disagree.

The separation between discovery and verification is a core part of 
CineVerdict's research architecture.

## Repository Structure

cineverdict_agent/
    agent.py

    agents/
        director.py
        research.py
        market.py
        production_risk.py
        verdict.py

    state/

    tools/
        parallel_search.py

## Local Setup

Create a Python virtual environment:

python -m venv .venv

Activate it:

source .venv/bin/activate

Install the dependencies required by CineVerdict.

Create the following local environment file:

cineverdict_agent/.env

Configure the required Google Cloud and Parallel credentials in that file.

Never commit the .env file to GitHub.

The repository .gitignore protects environment files, the virtual 
environment, Google ADK session data, Python caches, and logs.

## Running CineVerdict

From the CineVerdict repository root with the virtual environment active, 
run:

adk run cineverdict_agent

IMPORTANT:

When testing through the interactive ADK terminal, submit the complete 
evaluation request as ONE user message.

Do not paste separate instruction lines as separate submissions because 
ADK will treat each submission as a new pipeline invocation.

## Current Status

Completed:

- Five-agent sequential orchestration
- Director Agent
- Research Agent
- Market Agent
- Production/Risk Agent
- Verdict Agent
- Gemini-backed specialist agents
- Gemini retry protection
- Parallel Search integration
- Live web research
- First-party domain verification
- Source-integrity rules
- Successful whole-package compilation
- Successful end-to-end runtime validation
- Git version control
- GitHub repository

## Development Roadmap

Next phases:

- Professional CineVerdict web interface
- Structured evidence and citation presentation
- Persistent project and evaluation history
- Production deployment on Google Cloud
- Expanded evaluation workflows
- Agentic Cinema hackathon demo experience

## Security

Secrets and credentials must never be committed to source control.

The repository excludes:

- .env files
- .venv
- Google ADK local session state
- Python caches
- Local logs

## License

A project license will be added before public release.
