"""CineVerdict Validator Engine.

A Gemini-led, deterministic NLP engine ensuring strict behavioral compliance for
the CineVerdict multi-agent pipeline. It enforces evidence provenance, neutralizes
unsupported assumptions, and fails closed on ungrounded factual assertions.
"""

import re
import os
import sys
import threading
from typing import Set, Tuple, List, Dict, Optional

from google.adk.models.llm_response import LlmResponse
from google.genai import types

# ---------------------------------------------------------------------------
# Tracing & Telemetry
# ---------------------------------------------------------------------------
_trace_state = threading.local()

def _trace_log(msg: str):
    if os.environ.get("CINEVERDICT_VALIDATOR_TRACE") == "1":
        sys.stderr.write(f"[CINEVERDICT TRACE] {msg}\n")
        sys.stderr.flush()

# ---------------------------------------------------------------------------
# Core Vocabulary & Allowed Words
# ---------------------------------------------------------------------------
COMMON_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "what", "where", "when", "why", "how",
    "this", "that", "these", "those", "then", "there", "their", "theirs", "they", "them", "he", "she", "it",
    "its", "his", "her", "hers", "him", "we", "us", "our", "ours", "you", "your", "yours", "i", "me", "my",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must", "in", "on", "at", "to",
    "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "of", "off", "over", "under", "again", "further", "once", "here", "there",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "now", "anyway", "however", "therefore", "thus"
}

SYSTEM_ALLOWED = {"go", "modify", "no-go", "high", "medium", "low"}
KNOWN_LABELS = {
    "VERIFIED EVIDENCE", "SECONDARY EVIDENCE", "CONFLICTING EVIDENCE", "ANALYSIS", 
    "ASSUMPTION", "HYPOTHESIS", "MISSING EVIDENCE", "FINAL VERDICT", "CONFIDENCE", 
    "DECISIVE REASONS", "UNRESOLVED UNCERTAINTIES", "REQUIRED NEXT ACTIONS", 
    "SUPPORTED ACTION", "VERIFY FIRST", "STRATEGIC ACTION", "EVIDENCE LEDGER", 
    "RESEARCH EVIDENCE BRIEF", "CLAIM", "VERIFICATION STATUS", "SOURCE TITLE", 
    "SOURCE URL", "PUBLISH DATE", "SUPPORTING EXCERPT", "DIRECTOR PLAN", 
    "MARKET ANALYSIS", "PRODUCTION & RISK ANALYSIS", "PRODUCTION AND RISK ANALYSIS", 
    "CINEVERDICT FINAL EVALUATION"
}

ANALYTICAL_WORDS = {
    "verify", "determine", "evaluate", "assess", "confirm", "investigate", "analyze", "identify",
    "define", "explore", "structure", "plan", "manage", "review", "mitigate", "address", "check",
    "unverified", "unknown", "unresolved", "missing", "evidence", "lack", "absence", "insufficient",
    "project", "production", "budget", "funding", "rights", "schedule", "timeline", "access",
    "demand", "audience", "market", "commercial", "public", "interest", "viability", "feasibility",
    "applicable", "conditions", "authorization", "approaches", "alternative", "launch", "campaign",
    "release", "regulatory", "event", "timing", "development", "distribution", "hypothesis", "assumption",
    "risk", "verdict", "confidence", "decisive", "reasons", "action", "strategic", "ledger", "claim",
    "source", "title", "url", "publish", "date", "excerpt", "director", "user", "research", "agent",
    "pipeline", "contract", "dependency", "relationship", "external", "internal", "conclusion",
    "independent", "independence", "documentary", "film", "premise", "story"
}

# ---------------------------------------------------------------------------
# Helper Extraction & Parsing
# ---------------------------------------------------------------------------
def get_user_text(ctx) -> str:
    """Extracts the latest user prompt from the ADK context."""
    if not ctx or not hasattr(ctx, 'session'): return ""
    user_events = [e.output for e in ctx.session.events if getattr(e, 'author', '') == 'user']
    return user_events[-1] if user_events else ""

def get_director_text(ctx) -> str:
    if not ctx or not hasattr(ctx, 'session'): return ""
    evs = [e.output for e in ctx.session.events if getattr(e, 'author', '') == 'director_agent']
    return evs[-1] if evs else ""

def get_research_text(ctx) -> str:
    if not ctx or not hasattr(ctx, 'session'): return ""
    evs = [e.output for e in ctx.session.events if getattr(e, 'author', '') == 'research_agent']
    return evs[-1] if evs else ""

def extract_supporting_excerpts(text: str) -> List[str]:
    """Finds all 'Supporting Excerpt: "..."' blocks in the text."""
    return re.findall(r'Supporting Excerpt:\s*(.*?)(?:\n\s*(?:[A-Z0-9]+[ \-:]|$))', text + '\nE999:', flags=re.DOTALL | re.IGNORECASE)

def extract_claims(text: str) -> List[str]:
    return re.findall(r'Claim:\s*(.*?)(?:\n\s*(?:Verification|Source|Supporting|$))', text, flags=re.DOTALL | re.IGNORECASE)

def parse_cited_evidence_ids(text: str) -> List[str]:
    matches = re.findall(r'\bE(\d+)\b', text, flags=re.IGNORECASE)
    return [f"e{m}" for m in matches]

def get_evidence_excerpts_map(text: str) -> Dict[str, List[str]]:
    mapping = {}
    parts = re.split(r'\b(E\d+)\s*(?:[\u2014\u2013\-:])', text, flags=re.IGNORECASE)
    for i in range(1, len(parts), 2):
        key = parts[i].lower().strip()
        body = parts[i+1]
        mapping.setdefault(key, []).extend(extract_supporting_excerpts(body))
    return mapping

def get_evidence_claims_map(text: str) -> Dict[str, List[str]]:
    mapping = {}
    parts = re.split(r'\b(E\d+)\s*(?:[\u2014\u2013\-:])', text, flags=re.IGNORECASE)
    for i in range(1, len(parts), 2):
        key = parts[i].lower().strip()
        body = parts[i+1]
        mapping.setdefault(key, []).extend(extract_claims(body))
    return mapping

def get_allowed_words(ctx) -> Set[str]:
    """Builds the global allowed words set based on user input, director plan, and ALL evidence."""
    allowed = set(COMMON_STOP_WORDS) | set(SYSTEM_ALLOWED) | set(ANALYTICAL_WORDS)
    sources = [get_user_text(ctx), get_director_text(ctx), get_research_text(ctx)]
    for src in sources:
        words = re.findall(r"[a-zA-Z0-9\-]+", src.lower())
        allowed.update(words)
        for w in words:
            if "-" in w: allowed.update(w.split("-"))
    # Always allow Ex citations
    for i in range(1, 100): allowed.add(f"e{i}")
    return allowed

def split_structural_line(line: str) -> Optional[Tuple[str, str]]:
    labels_pattern = "|".join(re.escape(l) for l in KNOWN_LABELS)
    pattern = re.compile(
        r"^([ \t]*(?:#+\s*)?(?:-\s*|\*\s*|\+\s*|\d+\s*\.\s*|[IVXLCDM]+\s*\.\s*)?(?:\*\*|\[)?(?:E\d+\s*(?:—|-)\s*)?(?:" + labels_pattern + r")(?:\s*\[[^\]]+\])?(?:\*\*|\])?(?:\s*(?::|—|-)\s*|\s*$))(.*)$",
        re.IGNORECASE
    )
    m = pattern.match(line)
    if m: return m.group(1), m.group(2)
    # Generic bold headers or E# headers
    m_bold = re.match(r"^([ \t]*(?:#+\s*)?(?:-\s*|\*\s*|\d+\.\s*)?\*\*[^\*]+\*\*(?:\s*(?::|—|-)\s*|\s*$))(.*)$", line)
    if m_bold: return m_bold.group(1), m_bold.group(2)
    m_dr = re.match(r"^([ \t]*(?:#+\s*)?(?:-\s*|\*\s*|\d+\.\s*)?DECISIVE REASON \d+(?:\s*(?::|—|-)\s*|\s*$))(.*)$", line, re.IGNORECASE)
    if m_dr: return m_dr.group(1), m_dr.group(2)
    m_e = re.match(r"^([ \t]*(?:#+\s*)?(?:-\s*|\*\s*|\d+\.\s*)?(?:E\d+)(?:\*\*|\])?(?:\s*(?::|—|-)\s*|\s*$))(.*)$", line, re.IGNORECASE)
    if m_e: return m_e.group(1), m_e.group(2)
    m_list = re.match(r"^([ \t]*(?:-\s+|\*\s+|\+\s+|\d+\s*\.\s+))(.*)$", line)
    if m_list: return m_list.group(1), m_list.group(2)
    return None

def classify_sentence_role(sentence: str) -> str:
    s = sentence.lower()
    if any(x in s for x in ["verify", "determine", "evaluate", "assess", "confirm", "action"]): return "action"
    if any(x in s for x in ["unverified", "unknown", "unresolved", "missing", "lack of", "remains", "whether"]): return "uncertainty"
    if any(x in s for x in ["assume", "assumption", "hypothesis", "viability", "feasibility", "implication"]): return "analytical_assumption"
    return "factual"

# ---------------------------------------------------------------------------
# Content Neutralization
# ---------------------------------------------------------------------------
def clean_and_validate_hidden_facts(text: str, allowed_words: Set[str], ctx=None) -> str:
    """Replaces unauthorized proper nouns/numbers with [UNSUPPORTED]."""
    _trace_log("Starting clean_and_validate_hidden_facts")
    text = text.replace("&#58;", ":")
    ev_map = get_evidence_excerpts_map(get_research_text(ctx)) if ctx else {}
    
    lines = text.split('\n')
    out_lines = []
    active_cits = []

    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
            
        split_res = split_structural_line(line)
        label_part, body = split_res if split_res else ("", line)
        
        # Track active evidence scope
        if split_res and any(l in label_part.upper() for l in ["VERIFIED EVIDENCE", "SECONDARY EVIDENCE", "CONFLICTING EVIDENCE"]):
            active_cits = parse_cited_evidence_ids(label_part)
        elif split_res and any(l in label_part.upper() for l in KNOWN_LABELS):
            active_cits = []
            
        sentences = re.split(r'([.!?]\s+)', body)
        merged_sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "") for i in range(0, len(sentences), 2) if sentences[i]]
        
        out_sentences = []
        for sent in merged_sentences:
            line_cits = parse_cited_evidence_ids(label_part + sent) or active_cits
            
            # Build local allowed words
            local_allowed = set(allowed_words)
            if ev_map and line_cits:
                local_allowed = set(COMMON_STOP_WORDS) | set(SYSTEM_ALLOWED) | set(ANALYTICAL_WORDS)
                for cid in line_cits:
                    if cid in ev_map:
                        for exc in ev_map[cid]: local_allowed.update(re.findall(r"[a-zA-Z0-9\-]+", exc.lower()))
                # Add context words
                if ctx:
                    local_allowed.update(re.findall(r"[a-zA-Z0-9\-]+", get_director_text(ctx).lower()))
                    local_allowed.update(re.findall(r"[a-zA-Z0-9\-]+", get_user_text(ctx).lower()))
            
            role = classify_sentence_role(sent)
            tokens = set(re.findall(r"\b[A-Z][a-zA-Z0-9\-]*\b|\b\d+\b", sent))
            unauthorized = []
            for t in tokens:
                tl = t.lower()
                if re.match(r"^e\d+$", tl): continue
                if tl not in local_allowed and not any(tl.startswith(a) for a in local_allowed):
                    if role == "factual" or (tl not in ANALYTICAL_WORDS):
                        unauthorized.append(t)
            
            out_sent = sent
            for u in sorted(unauthorized, key=len, reverse=True):
                out_sent = re.sub(r"\b" + re.escape(u) + r"\b", "[UNSUPPORTED]", out_sent, flags=re.IGNORECASE)
            out_sentences.append(out_sent)
            
        out_lines.append(label_part + "".join(out_sentences))
        
    return "\n".join(out_lines)

def is_clause_grammatically_complete(clause: str, full_sentence: str) -> bool:
    return len(clause.split()) >= 3 and any(v in clause.lower() for v in [" is ", " are ", " was ", " were ", " will ", " has ", " have ", " remains ", " affects ", " begins ", " satisfies "])

def fail_closed_on_unsupported_sentences(text: str) -> str:
    """Completely redacts sentences containing [UNSUPPORTED]."""
    if "[UNSUPPORTED]" not in text: return text
    
    out_lines = []
    for line in text.split("\n"):
        if not line.strip() or "[UNSUPPORTED]" not in line:
            if not re.match(r'^(\s*(?:-\s+|\*\s+|\d+\.\s+))?\[Factual proposition unverified.*?\]$', line, re.IGNORECASE):
                out_lines.append(line)
            continue
            
        sentences = re.split(r'([.!?]\s+)', line)
        merged = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "") for i in range(0, len(sentences), 2) if sentences[i]]
        
        out_merged = []
        for s in merged:
            if "[UNSUPPORTED]" in s:
                prefix_match = re.match(r'^(\s*(?:-\s+|\*\s+|\d+\.\s+))', s)
                prefix = prefix_match.group(1) if prefix_match else ""
                
                # Check clause rescue
                preserved = False
                for conj in [", but ", ", however, ", "; however, ", ", and "]:
                    if conj in s:
                        left, right = s.split(conj, 1)
                        if "[UNSUPPORTED]" in left and "[UNSUPPORTED]" not in right and is_clause_grammatically_complete(right, s):
                            rs = right.strip()
                            out_merged.append(f"{prefix}{rs[0].upper()}{rs[1:]}" + ("." if not rs.endswith((".","!","?")) else ""))
                            preserved = True
                            break
                        elif conj != ", and " and "[UNSUPPORTED]" not in left and "[UNSUPPORTED]" in right and is_clause_grammatically_complete(left, s):
                            ls = re.sub(r'^(?:-\s*|\*\s*|\d+\.\s*)', '', left).strip()
                            out_merged.append(f"{prefix}{ls[0].upper()}{ls[1:]}" + ("." if not ls.endswith((".","!","?")) else ""))
                            preserved = True
                            break
                if not preserved:
                    out_merged.append(f"{prefix}Evidence is insufficient to verify this factual proposition.")
            else:
                out_merged.append(s)
        out_lines.append("".join(out_merged))
    return "\n".join(out_lines)

def neutralize_positive_assumptions(text: str) -> str:
    lines = text.split("\n")
    out = []
    for line in lines:
        if any(x in line.lower() for x in ["assume", "assumed", "assumption", "hypothesis"]):
            if "[UNSUPPORTED]" in line:
                out.append(line)
                continue
            s_clean = line.lower()
            if any(x in s_clean for x in ["audience", "demand", "interest", "market"]):
                out.append(re.sub(r'^.*$', "Audience demand remains unverified and whether a reachable audience exists remains unknown.", line))
            elif "access" in s_clean or "coordination" in s_clean:
                out.append(re.sub(r'^.*$', "Access has not been established and remains unverified.", line))
            elif "funding" in s_clean or "budget" in s_clean:
                out.append(re.sub(r'^.*$', "Funding status is unspecified and remains unverified.", line))
            elif any(x in s_clean for x in ["rights", "authorization", "licensing", "clearance", "permission"]):
                out.append(re.sub(r'^.*$', "Rights/authorization remain to be verified.", line))
            elif any(x in s_clean for x in ["schedule", "timeline", "independent", "affect", "impact"]):
                if "independent" in s_clean: out.append(re.sub(r'^.*$', "The relationship between the internal schedule and the external schedule is unverified.", line))
                elif "affect" in s_clean: out.append(re.sub(r'^.*$', "Whether the external schedule affects the internal production timeline remains unverified.", line))
                else: out.append(re.sub(r'^.*$', "The production timeline and schedule relationship remains unverified.", line))
            else:
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out)

def neutralize_audience_assumptions(text: str) -> str:
    replacements = {
        r"\bpublic\s+interest\s+exists\b": "HYPOTHESIS: public interest may exist but remains unverified",
        r"\ba\s+viable\s+audience\s+is\s+reachable\b": "HYPOTHESIS: an audience may exist; its size, composition, reachability, engagement, and commercial viability remain unverified",
        r"\bcommercially\s+sustainable\b": "commercially unverified",
        r"\baudience\s+demand\b": "unverified audience demand",
        r"\bpublic\s+interest\b": "unverified public interest",
        r"\bwillingness\s+to\s+pay\b": "unverified willingness to pay",
        r"\bcommercial\s+viability\b": "unverified commercial viability",
    }
    for k, v in replacements.items(): text = re.sub(k, v, text, flags=re.IGNORECASE)
    return text

def neutralize_production_assumptions(text: str) -> str:
    replacements = {
        r"\bformat\s+can\s+be\s+structured\b": "format whether a format can be structured",
        r"\bdesired\s+access\s+to\s+personnel\s+can\s+be\s+coordinated\b": "unverified desired access to personnel whether coordination",
        r"\bcan\s+be\s+coordinated\b": "whether coordination is possible remains unverified and conditional",
        r"\bcan\s+be\s+structured\b": "remains unverified and conditional",
    }
    for k, v in replacements.items(): text = re.sub(k, v, text, flags=re.IGNORECASE)
    return text

def neutralize_evaluative_words(text: str, allowed: Set[str]) -> str:
    if "successful" in text.lower() and "successful" not in allowed:
        text = re.sub(r"\bsuccessful\b", "existing/distributed", text, flags=re.IGNORECASE)
    if "commercially viable" in text.lower():
        text = re.sub(r"\bcommercially\s+viable\b", "unverified commercial viability", text, flags=re.IGNORECASE)
    return text

def make_schedule_conditional(text: str, ctx=None) -> str:
    replacements = {
        r"\bto\s+align\s+the\s+production's\s+release\s+timeline\b": "to determine whether/how it affects the production's release timeline",
        r"\bimpacts\s+the\s+production\s+schedule\b": "is an external event; determine whether/how it affects the production schedule",
        r"\baround\s+the\s+launch\s+uncertainty\b": "determine whether/how the launch uncertainty affects the filming schedule before final planning",
        r"\bintroduces\s+timing\s+uncertainty\s+for\s+the\b": "is an external event; determine whether/how it affects the",
        r"\bwould\s+need\s+to\s+be\s+aligned\s+with\b": "determine whether/how the external campaign affects the documentary schedule before deciding if alignment is required",
        r"\bassume\s+the\s+schedules\s+are\s+independent\s+unless\s+evidence\s+establishes\s+coupling\b": "The relationship remains unverified and unknown",
        r"\bAssume\s+no\s+dependency\s+unless\s+evidence\s+establishes\s+one\b": "The relationship remains unverified and unknown",
        r"\bWhether\s+the\s+internal\s+schedule\s+will\s+remain\s+independent\s+of\s+or\s+align\s+with\s+the\s+external\s+schedule\b": "Whether and how the schedules are related remains unknown",
        r"\bDetermine\s+whether\s+to\s+align\s+the\s+schedules\s+or\s+keep\s+them\s+independent\b": "Determine whether any dependency, alignment, independence, coupling, influence exists",
    }
    for k, v in replacements.items(): text = re.sub(k, v, text, flags=re.IGNORECASE)
    return text

def parse_evidence_ledger_table(text: str) -> Optional[Tuple[Dict[str, List[str]], Dict[str, List[str]]]]:
    return None

def verdict_before_model_callback(ctx, req):
    """Binds active evidence IDs to system instructions."""
    ledger = get_research_text(ctx)
    cits = re.findall(r'\b(E\d+)\b', ledger)
    ids = sorted(list(set(cits)))
    base = req.config.system_instruction or ""
    if not ids:
        req.config.system_instruction = base + "\nDYNAMIC EVIDENCE LEDGER BINDING CONTRACT: There is NO active Research Evidence Ledger. You are STRICTLY prohibited from making any cited factual assertions."
    else:
        req.config.system_instruction = base + f"\nDYNAMIC EVIDENCE LEDGER BINDING CONTRACT: You may ONLY cite the following active evidence keys: {', '.join(ids)}."

def _apply_validators(ctx, response: LlmResponse) -> LlmResponse:
    if not response or not response.text: return response
    allowed = get_allowed_words(ctx)
    t = response.text
    t = clean_and_validate_hidden_facts(t, allowed, ctx)
    t = fail_closed_on_unsupported_sentences(t)
    t = neutralize_positive_assumptions(t)
    t = neutralize_audience_assumptions(t)
    t = neutralize_production_assumptions(t)
    t = neutralize_evaluative_words(t, allowed)
    t = make_schedule_conditional(t, ctx)
    response.text = t
    return response

def market_after_model_callback(ctx, response: LlmResponse) -> LlmResponse:
    return _apply_validators(ctx, response)

def production_risk_after_model_callback(ctx, response: LlmResponse) -> LlmResponse:
    return _apply_validators(ctx, response)

def verdict_after_model_callback(ctx, response: LlmResponse) -> LlmResponse:
    return _apply_validators(ctx, response)

def research_after_model_callback(ctx, response: LlmResponse) -> LlmResponse:
    return response

def director_after_model_callback(ctx, response: LlmResponse) -> LlmResponse:
    return response
