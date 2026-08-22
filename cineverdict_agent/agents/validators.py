import re
from google.adk.models.llm_response import LlmResponse
from google.genai import types

COMMON_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "what", "where", "when", "why", "how",
    "this", "that", "these", "those", "then", "there", "their", "theirs", "they", "them", "he", "she", "it",
    "its", "his", "her", "hers", "him", "himself", "herself", "itself", "we", "us", "our", "ours", "you",
    "your", "yours", "i", "me", "my", "mine", "myself", "yourself", "yourselves", "ourselves",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "in", "on", "at", "to", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "from", "up", "down", "of", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "just", "don", "shouldn", "now", "anyway", "however", "therefore", "furthermore", "thus", "likewise",
}

SYSTEM_ALLOWED_WORDS = {
    "verified", "secondary", "evidence", "conflicting", "unresolved", "questions",
    "analysis", "assumption", "missing", "hypothesis", "exist", "size", "composition",
    "reachability", "engagement", "commercial", "viability", "unverified", "unknown",
    "dependency", "conditional", "consideration", "go", "modify", "no-go", "high",
    "medium", "low", "verify", "first", "strategic", "action", "format", "documentary",
    "short", "premise", "questions", "inputs", "status", "unspecified", "supplied",
    "lacks", "funding", "strategy", "timeline", "schedule", "external", "milestone",
    "alignment", "film", "filming", "alternative", "approach", "availability",
    "rights", "commitment", "unsupported", "excerpt", "reconstructable", "presupposed",
    "evaluative", "interpretation", "upgrades", "production", "risk", "director",
    "verdict", "final", "evaluation", "confidence", "decisive", "reasons", "uncertainties",
    "required", "next", "actions", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10",
    "align", "monitor", "confirm", "evaluate", "determine", "investigate",
    "monitoring", "confirming", "evaluating", "determining", "investigating"
}


def extract_supporting_excerpts(research_text: str) -> list[str]:
    """Extracts all text values following 'Supporting Excerpt:' in the research text."""
    excerpts = []
    pattern = re.compile(
        r"Supporting Excerpt:\s*\"?([^\"]+?)\"?(?=\n\s*(?:[A-Z][a-zA-Z\s\d]+:|E\d+\s*—|$))",
        re.IGNORECASE,
    )
    matches = pattern.findall(research_text)
    for m in matches:
        excerpts.append(m.strip())

    if not excerpts:
        parts = research_text.split("Supporting Excerpt:")
        for part in parts[1:]:
            val = part.strip()
            if val.startswith('"'):
                end_quote = val.find('"', 1)
                if end_quote != -1:
                    val = val[1:end_quote]
            else:
                lines = []
                for line in val.split("\n"):
                    if any(
                        line.strip().startswith(label)
                        for label in [
                            "Claim:",
                            "Verification Status:",
                            "Source Title:",
                            "Source URL:",
                            "Publish Date:",
                            "E",
                        ]
                    ):
                        break
                    lines.append(line)
                val = "\n".join(lines).strip()
            excerpts.append(val)
    return excerpts


def get_research_text(ctx) -> str:
    if not ctx or not ctx.session or not ctx.session.events:
        return ""
    for event in ctx.session.events:
        if event.author == "research_agent":
            if isinstance(event.output, dict):
                return event.output.get("research_evidence", "")
            if isinstance(event.output, str):
                return event.output
            if event.content and event.content.parts:
                return "".join(
                    part.text for part in event.content.parts if part.text
                )
    return ""


def get_director_text(ctx) -> str:
    if not ctx or not ctx.session or not ctx.session.events:
        return ""
    for event in ctx.session.events:
        if event.author == "director_agent":
            if isinstance(event.output, dict):
                return event.output.get("director_plan", "")
            if isinstance(event.output, str):
                return event.output
            if event.content and event.content.parts:
                return "".join(
                    part.text for part in event.content.parts if part.text
                )
    return ""


def get_user_text(ctx) -> str:
    if not ctx:
        return ""
    if hasattr(ctx, "user_content") and ctx.user_content and ctx.user_content.parts:
        return "".join(part.text for part in ctx.user_content.parts if part.text)
    return ""


def get_allowed_words(ctx) -> set[str]:
    allowed = set()
    # 1. From Supporting Excerpts
    research_text = get_research_text(ctx)
    excerpts = extract_supporting_excerpts(research_text)
    for exc in excerpts:
        words = re.findall(r"[a-zA-Z0-9\-]+", exc)
        for w in words:
            allowed.add(w.lower())

    # 2. From Director Plan
    director_text = get_director_text(ctx)
    if director_text:
        words = re.findall(r"[a-zA-Z0-9\-]+", director_text)
        for w in words:
            allowed.add(w.lower())

    # 3. From User content
    user_text = get_user_text(ctx)
    if user_text:
        words = re.findall(r"[a-zA-Z0-9\-]+", user_text)
        for w in words:
            allowed.add(w.lower())

    # 4. Add system vocabulary & common words
    for w in SYSTEM_ALLOWED_WORDS:
        allowed.add(w.lower())
    for w in COMMON_STOP_WORDS:
        allowed.add(w.lower())

    return allowed


def clean_and_validate_hidden_facts(text: str, allowed_words: set[str]) -> str:
    """Finds proper nouns and numbers that do not exist in the allowed words set and redacts them."""
    significant_words = set(re.findall(r"\b[A-Z][a-zA-Z0-9\-]*\b|\b\d+\b", text))
    unauthorized = []
    for w in significant_words:
        w_lower = w.lower()
        if (
            w_lower not in allowed_words
            and w_lower not in COMMON_STOP_WORDS
            and w_lower not in SYSTEM_ALLOWED_WORDS
            and len(w) > 1
        ):
            unauthorized.append(w)

    unauthorized.sort(key=len, reverse=True)
    for w in unauthorized:
        text = re.sub(r"\b" + re.escape(w) + r"\b", "[UNSUPPORTED]", text)
    return text


def neutralize_audience_assumptions(text: str) -> str:
    mappings = {
        r"\bpublic\s+interest\s+exists\b": "HYPOTHESIS: public interest may exist but remains unverified",
        r"\ba\s+viable\s+audience\s+is\s+reachable\b": "HYPOTHESIS: an audience may exist; its size, composition, reachability, engagement, and commercial viability remain unverified",
        r"\bthe\s+short\s+documentary\s+format\s+is\s+commercially\s+sustainable\b": "HYPOTHESIS: whether the short documentary format is commercially sustainable remains unverified",
        r"\bcommercially\s+sustainable\b": "commercially unverified",
        r"\baudience\s+demand\b": "unverified audience demand",
        r"\bpublic\s+interest\b": "unverified public interest",
        r"\bpopularity\b": "unverified popularity",
        r"\bwillingness\s+to\s+pay\b": "unverified willingness to pay",
        r"\bcommercial\s+viability\b": "unverified commercial viability",
        r"\bmarket\s+size\b": "unverified market size",
        r"\bcommercial\s+sustainability\b": "unverified commercial sustainability",
    }
    
    sorted_keys = sorted(mappings.keys(), key=len, reverse=True)
    placeholders = {}
    for i, pattern in enumerate(sorted_keys):
        placeholder = f"___AUD_PLACEHOLDER_{i}___"
        text, count = re.subn(pattern, placeholder, text, flags=re.IGNORECASE)
        if count > 0:
            placeholders[placeholder] = mappings[pattern]
            
    for placeholder, final_val in placeholders.items():
        text = text.replace(placeholder, final_val)
        
    return text


def neutralize_production_assumptions(text: str) -> str:
    mappings = {
        r"\bdesired\s+access\s+to\s+personnel\b": "unverified desired access to personnel",
        r"\bcan\s+be\s+structured\s+around\s+launch\s+uncertainty\b": "whether a format can be structured around launch uncertainty remains unverified and conditional",
        r"\bcan\s+be\s+coordinated\s+with\s+the\s+subject\s+company\b": "whether coordination with the subject company is possible remains unverified and conditional",
        r"\bcoordination\s+with\s+the\s+subject\s+company\b": "unverified coordination with the subject company",
        r"\bdesired\s+access\b": "unverified desired access",
        r"\baccess\s+to\s+personnel\b": "unverified access to personnel",
        r"\baccess\s+to\s+facilities\b": "unverified access to facilities",
        r"\baccess\s+to\s+hardware\b": "unverified access to hardware",
        r"\bproduction\s+feasibility\b": "unverified production feasibility",
        r"\bpersonnel\s+availability\b": "unverified personnel availability",
    }
    
    sorted_keys = sorted(mappings.keys(), key=len, reverse=True)
    placeholders = {}
    for i, pattern in enumerate(sorted_keys):
        placeholder = f"___PROD_PLACEHOLDER_{i}___"
        text, count = re.subn(pattern, placeholder, text, flags=re.IGNORECASE)
        if count > 0:
            placeholders[placeholder] = mappings[pattern]
            
    for placeholder, final_val in placeholders.items():
        text = text.replace(placeholder, final_val)
        
    return text


def neutralize_evaluative_words(text: str, allowed_words: set[str]) -> str:
    evaluative_mappings = {
        r"\bcommercially\s+successful\b": "existing",
        r"\bcommercially\s+viable\b": "unverified commercial viability",
        r"\bproven\s+demand\b": "unverified demand",
        r"\bhigh\s+engagement\b": "unverified engagement",
        r"\bmajor\s+audience\b": "unverified audience",
        r"\bstrong\s+market\b": "unverified market",
        r"\bstrong\s+precedent\b": "unverified precedent",
        r"\bsuccessful\b": "existing/distributed",
        r"\bpopular\b": "unverified popularity",
        r"\bproven\b": "unverified",
        r"\beffective\b": "unverified effectiveness",
        r"\bstrong\b": "unverified strength",
    }

    sorted_keys = sorted(evaluative_mappings.keys(), key=len, reverse=True)
    placeholders = {}
    for i, pattern in enumerate(sorted_keys):
        raw_words = re.findall(r"[a-z]+", pattern.lower())
        if all(w in allowed_words for w in raw_words):
            continue
        
        placeholder = f"___EVAL_PLACEHOLDER_{i}___"
        text, count = re.subn(pattern, placeholder, text, flags=re.IGNORECASE)
        if count > 0:
            placeholders[placeholder] = evaluative_mappings[pattern]
            
    for placeholder, final_val in placeholders.items():
        text = text.replace(placeholder, final_val)
        
    return text


def make_schedule_conditional(text: str) -> str:
    schedule_mappings = {
        r"\balign\s+(?:the\s+)?production's\s+release\s+timeline\b": "determine whether/how it affects the production's release timeline",
        r"\balign\s+(?:the\s+)?production's\s+timeline\b": "determine whether/how it affects the production's timeline",
        r"\balign\s+(?:the\s+)?release\s+timeline\b": "determine whether/how it affects the release timeline",
        r"\balign\s+(?:the\s+)?production\s+timeline\b": "determine whether/how it affects the production timeline",
        r"\balign\s+(?:the\s+)?filming\s+schedule\b": "determine whether/how it affects the filming schedule",
        r"\balign\s+(?:the\s+)?production\s+schedule\b": "determine whether/how it affects the production schedule",
        r"\balign\s+(?:the\s+)?delivery\s+schedule\b": "determine whether/how it affects the delivery schedule",
        r"\balign\s+(?:the\s+)?marketing\s+schedule\b": "determine whether/how it affects the marketing schedule",
        r"\balign\s+(?:the\s+)?festival\s+schedule\b": "determine whether/how it affects the festival schedule",
    }
    for pattern, replacement in schedule_mappings.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def market_after_model_callback(callback_context, llm_response: LlmResponse) -> LlmResponse | None:
    ctx = callback_context.get_invocation_context()
    if not llm_response.content or not llm_response.content.parts:
        return None

    allowed_words = get_allowed_words(ctx)
    modified = False

    for part in llm_response.content.parts:
        if part.text:
            orig = part.text
            text = clean_and_validate_hidden_facts(orig, allowed_words)
            text = neutralize_audience_assumptions(text)
            text = neutralize_evaluative_words(text, allowed_words)
            if text != orig:
                part.text = text
                modified = True

    return llm_response if modified else None


def production_risk_after_model_callback(
    callback_context, llm_response: LlmResponse
) -> LlmResponse | None:
    ctx = callback_context.get_invocation_context()
    if not llm_response.content or not llm_response.content.parts:
        return None

    allowed_words = get_allowed_words(ctx)
    modified = False

    for part in llm_response.content.parts:
        if part.text:
            orig = part.text
            text = clean_and_validate_hidden_facts(orig, allowed_words)
            text = neutralize_production_assumptions(text)
            text = neutralize_evaluative_words(text, allowed_words)
            if text != orig:
                part.text = text
                modified = True

    return llm_response if modified else None


def verdict_after_model_callback(callback_context, llm_response: LlmResponse) -> LlmResponse | None:
    ctx = callback_context.get_invocation_context()
    if not llm_response.content or not llm_response.content.parts:
        return None

    allowed_words = get_allowed_words(ctx)
    modified = False

    for part in llm_response.content.parts:
        if part.text:
            orig = part.text
            text = clean_and_validate_hidden_facts(orig, allowed_words)
            text = neutralize_evaluative_words(text, allowed_words)
            text = make_schedule_conditional(text)
            if text != orig:
                part.text = text
                modified = True

    return llm_response if modified else None
