import re
import os
import sys
import threading
from google.adk.models.llm_response import LlmResponse
from google.genai import types

_trace_state = threading.local()

def _is_trace_enabled() -> bool:
    return os.environ.get("CINEVERDICT_VALIDATOR_TRACE") == "1"

def _trace_log(msg: str):
    if _is_trace_enabled():
        role = getattr(_trace_state, "role", "unknown")
        sys.stderr.write(f"[CINEVERDICT TRACE][{role}] {msg}\n")
        sys.stderr.flush()

def _trace_raw_callback(role: str, text: str):
    if os.environ.get("CINEVERDICT_RAW_CALLBACK_TRACE") == "1":
        sys.stderr.write(f"[CINEVERDICT RAW][{role}] === START RAW CALLBACK ===\n")
        sys.stderr.write(text)
        if not text.endswith("\n"):
            sys.stderr.write("\n")
        sys.stderr.write(f"[CINEVERDICT RAW][{role}] === END RAW CALLBACK ===\n")
        sys.stderr.flush()

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
    "although", "since", "given", "while", "moreover", "consequently", "indeed", "unless", "until", "despite",
    "first", "second", "third", "finally", "hence", "otherwise", "instead", "specifically", "concerning",
    "regarding", "meanwhile", "whether", "either", "neither", "each", "every",
}

# Strictly minimal outcomes, technical indexes, and grammatical enums (no content-bearing words)
SYSTEM_ALLOWED_WORDS = {
    "go", "modify", "no-go", "high", "medium", "low",
    "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10"
}

# Explicit structural and metadata labels allowed by the CineVerdict contract
KNOWN_LABELS = {
    "VERIFIED EVIDENCE",
    "SECONDARY EVIDENCE",
    "CONFLICTING EVIDENCE",
    "ANALYSIS",
    "ASSUMPTION",
    "HYPOTHESIS",
    "MISSING EVIDENCE",
    "FINAL VERDICT",
    "CONFIDENCE",
    "DECISIVE REASONS",
    "UNRESOLVED UNCERTAINTIES",
    "REQUIRED NEXT ACTIONS",
    "SUPPORTED ACTION",
    "VERIFY FIRST",
    "STRATEGIC ACTION",
    "EVIDENCE LEDGER",
    "RESEARCH EVIDENCE BRIEF",
    "CLAIM",
    "VERIFICATION STATUS",
    "SOURCE TITLE",
    "SOURCE URL",
    "PUBLISH DATE",
    "SUPPORTING EXCERPT",
    "DIRECTOR PLAN",
    "MARKET ANALYSIS",
    "PRODUCTION & RISK ANALYSIS",
    "PRODUCTION AND RISK ANALYSIS",
    "CINEVERDICT FINAL EVALUATION"
}


ANALYTICAL_SUBSTANTIVE_WORDS = {
    # Action verbs and derivatives
    "verify", "verification", "verified", "verifies",
    "determine", "determination", "determined", "determines",
    "evaluate", "evaluation", "evaluates", "evaluated", "evaluative",
    "assess", "assessment", "assessed", "assesses",
    "confirm", "confirmation", "confirmed", "confirms",
    "investigate", "investigation", "investigative",
    "obtain", "obtaining", "obtained",
    "coordinate", "coordination", "coordinated",
    "align", "alignment", "aligned",
    "track", "tracking", "tracked",
    "clarify", "clarification", "clarified",
    "ensure", "ensuring", "ensured",
    "analyze", "analysis", "analyst", "analytical", "analyses",
    "identify", "identification", "identified",
    "explore", "exploration", "explored",
    "structure", "structured", "structuring",
    "plan", "planning", "planned",
    "manage", "management", "managed",
    "review", "reviewing", "reviewed",
    "mitigate", "mitigation", "mitigated",
    "address", "addressing", "addressed",
    "check", "checking", "checked",
    "commit", "commitment", "committed",
    "establish", "establishes", "established", "establishing",
    "compare", "compares", "compared", "comparing", "comparison", "comparisons",
    "reach", "reached",
    "affect", "affects", "affected", "affecting",
    "impact", "impacts", "impacted", "impacting",

    # Uncertainty/missing evidence nouns and adjectives
    "unverified", "unknown", "unresolved", "unspecified", "unclear", "undetermined",
    "missing", "evidence", "lack", "absence", "insufficient", "status",
    "exist", "exists", "existence", "potential", "potentially", "likely", "possible", "possibly",
    "uncertainty", "uncertainties", "unavailability", "unavailable", "adequacy", "adequate",
    "viability", "viable", "feasibility", "feasible", "sustainability", "sustainable",
    "suitability", "suitable", "applicability", "applicable",
    "unsupported", "unproven", "unconfirmed", "unsupplied",
    "sufficient", "sufficiency", "sufficiently",

    # Common project/domain substantive nouns
    "project", "production", "budget", "funding", "rights", "schedule", "timeline", "timelines", "schedules",
    "access", "conditions", "authorization", "approaches", "approach", "alternative", "alternatives",
    "demand", "audience", "market", "commercial", "public", "interest", "popularity", "willingness", "pay",
    "size", "personnel", "availability", "facilities", "hardware", "subject", "company", "launch",
    "milestone", "milestones", "campaign", "opening", "release", "regulatory", "event", "events",
    "timing", "development", "editorial", "focus", "post-production", "distribution", "festival",
    "delivery", "planning", "activity", "implications", "implication", "hypothesis", "hypotheses",
    "assumption", "assumptions", "risk", "risks", "verdict", "final", "evaluation", "confidence",
    "decisive", "reasons", "unresolved", "required", "next", "actions", "action", "strategic",
    "ledger", "brief", "claim", "claims", "source", "title", "url", "publish", "date", "supporting",
    "excerpt", "excerpts", "director", "user", "research", "agent", "pipeline", "contract",
    "dependency", "dependencies", "dependent",
    "relationship", "relationships",
    "external", "internal",
    "conclusion", "conclusions",
    "confident",
    "adjustment", "adjustments",
    "connection", "connections",
    "permission", "permissions", "requirement", "requirements",
    "agreement", "agreements", "clearance", "clearances",
    "licensing", "license", "licenses", "authority", "authorities",
    "regulator", "regulators", "software", "crew", "crews", "staff", "staffing",
    "partnership", "partnerships", "partner", "partners",
    "precedent", "precedents", "consequence", "consequences",
    "strategy", "strategies", "independent", "independence",
    "documentary", "documentaries", "film", "films", "premise", "premises", "story", "stories",

    # Grammatical/generic words that might get capitalized at sentence starts
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "what", "where", "when", "why", "how",
    "this", "that", "these", "those", "then", "there", "their", "theirs", "they", "them", "he", "she", "it",
    "its", "his", "her", "hers", "him", "we", "us", "our", "ours", "you", "your", "yours", "i", "me", "my",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must", "whether", "if", "either",
    "neither", "both", "each", "every", "all", "any", "some", "such", "no", "not", "only", "other", "another",
    "additional", "general", "specific", "proposed", "current", "currently", "future", "past", "historical", "contextual",
    "context", "factual", "assertion", "assertions"
}


def get_normalized_sentence_for_classification(sentence: str) -> str:
    """Strips recognized structural prefixes from the start of a sentence for classification purposes.

    Retains the original text for output and citation handling.
    """
    s = sentence.replace("&#58;", ":").strip()

    # Define KNOWN_LABELS prefix pattern
    labels_pattern = "|".join(re.escape(label) for label in KNOWN_LABELS)

    while True:
        orig = s

        # 1. Strip Markdown bullet markers or numbered list markers
        s = re.sub(r'^([ \t]*(?:-\s*|\*\s*|\+\s*|\d+\s*\.\s*|[IVXLCDM]+\s*\.\s*))', '', s, flags=re.IGNORECASE)

        # 2. Strip bold/markdown wrappers at start of prefix area
        s = re.sub(r'^([ \t]*\*\*|\*)', '', s)

        # 3. Strip explicit citation prefixes like [E1], [E1 and E2], [MISSING EVIDENCE], etc.
        s = re.sub(r'^([ \t]*\[(?:E\d+|MISSING EVIDENCE|based on E\d+|secondary evidence|verified evidence)[^\]]*\]:?\s*)', '', s, flags=re.IGNORECASE)

        # 4. Strip known labels, e.g. VERIFY FIRST, SUPPORTED ACTION, ANALYSIS, etc.
        s = re.sub(r'^([ \t]*(?:' + labels_pattern + r')(?:\s*\[[^\]]+\])?(?:\*\*|\])?(?:\s*(?::|—|-)\s*|\s*$))', '', s, flags=re.IGNORECASE)

        # 5. Strip any leftover punctuation/bold markers and whitespace from the start
        s = re.sub(r'^([ \t]*(?:\*\*|\*|:|\—|-)\s*)', '', s)

        s = s.strip()
        if s == orig:
            break

    return s


def classify_sentence_role(sentence: str) -> str:
    """Classifies a sentence/clause into one of the semantic roles:
    - 'structural': structural heading/label
    - 'action': Recommended action / verification task
    - 'uncertainty': Missing evidence / uncertainty
    - 'analytical_assumption': Analytical interpretation / assumption
    - 'factual': Factual evidence assertion
    """
    normalized_s = get_normalized_sentence_for_classification(sentence)
    s = normalized_s.strip().lower()
    if not s:
        return "structural"

    # E. Recommended action / verification task (Action)
    action_verbs = {
        "verify", "determine", "evaluate", "assess", "confirm", "investigate",
        "obtain", "coordinate", "align", "track", "clarify", "ensure", "analyze",
        "identify", "explore", "mitigate", "address", "check", "review", "compare",
        "establish", "define", "formulate", "schedule", "plan"
    }

    # Strip list markers
    clean_s = re.sub(r'^(?:-\s*|\*\s*|\d+\.\s*)', '', s).strip()
    first_word_match = re.match(r'^([a-z]+)', clean_s)
    if first_word_match:
        first_word = first_word_match.group(1)
        if first_word in action_verbs:
            return "action"

    # Check for modal verbs + action verbs
    action_patterns = [
        r"\b(?:should|must|needs?\s+to|need\s+to|to\s+be|required\s+to|planning\s+to|recommend\s+to|would\s+need\s+to|choose\s+to|decide\s+to|to|be\s+required\s+to)\s+(?:verify|determine|evaluate|assess|confirm|investigate|obtain|coordinate|align|track|clarify|ensure|analyze|identify|explore|mitigate|address|check|review|compare|establish)\b",
        r"\b(?:needs?|must|should|required|would\s+need)\s+be\s+(?:verified|determined|evaluated|assessed|confirmed|investigated|obtained|coordinated|aligned|tracked|clarified|ensured|analyzed|identified|explored|mitigated|addressed|checked|reviewed|compared|established)\b",
        r"\bnext\s+actions?\b",
        r"\bverify\s+first\b",
        r"\baction\s+items?\b",
        r"\brecommended\s+actions?\b"
    ]
    for pattern in action_patterns:
        if re.search(pattern, s):
            return "action"

    # D. Missing evidence / uncertainty
    uncertainty_patterns = [
        r"\bunverified\b",
        r"\bunknown\b",
        r"\bunresolved\b",
        r"\bunspecified\b",
        r"\bundetermined\b",
        r"\bunclear\b",
        r"\bunsupported\b",
        r"\bunavailable\b",
        r"\bunavailability\b",
        r"\bunproven\b",
        r"\b(?:no|lack\s+of|absence\s+of|insufficient|missing|unresolved|without)\s+evidence\b",
        r"\bnot\s+(?:been\s+)?(?:established|verified|supplied|confirmed|proven|supported|unveiled|specified)\b",
        r"\bnot\s+yet\s+(?:established|verified|supplied|confirmed|proven|supported)\b",
        r"\bnot\s+fully\s+(?:established|verified|supplied|confirmed|proven|supported)\b",
        r"\b(?:remains|remain)\s+(?:unverified|unknown|unresolved|unspecified|undetermined|unclear|unsupported|unavailable|unproven)\b",
        r"\b(?:remains|remain|is|are)\s+to\s+be\s+(?:verified|established|determined|confirmed|proven|supported)\b",
        r"\b(?:evidence|information|details?|clarification|verification)\s+(?:is|are|remain|remains)\s+(?:required|needed|necessary)\b",
        r"\b(?:no|not|does\s+not|do\s+not)\s+(?:establish|creates?|imply|implies|prove|proves)\s+(?:any\s+)?(?:dependency|relationship|connection|alignment)\b",
        r"\b(?:external|launch|campaign|event|schedule)\s+(?:dates?|timelines?|schedules?|uncertaint(?:y|ies)|changes?|movements?|timing|adjustments?)\b",
        r"\bwhether\s+[\w\s\-]+?\s+(?:exists|is|can|remains|affects)\b",
        r"\bviability\s+remains\b",
        r"\bfeasibility\s+remains\b",
        r"\bstatus\s+remains\b",
        r"\bnot\s+established\b",
        r"\bnot\s+supplied\b",
        r"\bnot\s+verified\b",
        r"\babsence\s+of\s+evidence\b",
        r"\bmissing\s+evidence\b",
        r"\black\s+of\s+evidence\b",
        r"\binsufficient\s+evidence\b",
        r"\bhas\s+not\s+been\s+established\b",
        r"\bnot\s+yet\s+verified\b",
        r"\bnot\s+fully\s+established\b",
        r"\bmissing\s+inputs?\b",
        r"\bunresolved\s+uncertainties\b"
    ]
    for pattern in uncertainty_patterns:
        if re.search(pattern, s):
            return "uncertainty"

    # B & C. Analytical interpretation / assumption
    analytical_patterns = [
        r"\bhypothesis\b",
        r"\bhypothesis:\b",
        r"\bassumption\b",
        r"\bassumption:\b",
        r"\bit\s+is\s+assumed\b",
        r"\bassumes\s+that\b",
        r"\bassuming\b",
        r"\bviability\b",
        r"\bfeasibility\b",
        r"\bsustainability\b",
        r"\bmarket\s+viability\b",
        r"\bcommercial\s+viability\b",
        r"\bproduction\s+feasibility\b",
        r"\bproject\s+viability\b",
        r"\bproject\s+feasibility\b",
        r"\bpotential\s+impact\b",
        r"\bpotential\s+effect\b",
        r"\bpotential\s+influence\b",
        r"\bpotential\s+implications\b",
        r"\bstrategic\s+implications\b",
        r"\bconditional\b",
        r"\bdependency\b",
        r"\bdependencies\b",
        r"\bcontingency\b",
        r"\bcontingencies\b",
        r"\bimplication\b",
        r"\bimplications\b",
        r"\btiming\s+adjustments\b",
        r"\bproduction\s+timeline\b",
        r"\bproposed\s+production\s+timeline\b",
        r"\banalysis\b"
    ]
    for pattern in analytical_patterns:
        if re.search(pattern, s):
            return "analytical_assumption"

    return "factual"


def neutralize_positive_assumptions(text: str) -> str:
    """Neutralizes any positive assumptions converting absence of evidence into positive claims.

    Ensures unknown or unverified conditions remain UNKNOWN/unverified, rather than positive assumptions.
    """
    lines = text.split("\n")
    processed_lines = []
    current_section = None

    for line in lines:
        if not line.strip():
            processed_lines.append(line)
            continue

        # Detect structural part vs body
        split_res = split_structural_line(line)
        if split_res:
            label_part, body_part = split_res
            for label in KNOWN_LABELS:
                if label.lower() in label_part.lower():
                    current_section = label
                    break
        else:
            label_part = ""
            body_part = line

        # Split body into sentences
        sentence_end = re.compile(r'([.!?]\s+)')
        parts = sentence_end.split(body_part)
        sentences = []
        i = 0
        while i < len(parts):
            s = parts[i]
            if i + 1 < len(parts):
                s += parts[i+1]
                i += 2
            else:
                i += 1
            if s:
                sentences.append(s)

        processed_sentences = []
        for sentence in sentences:
            s_clean = sentence.strip().lower()

            # Match positive assumption patterns
            is_in_assumption_sec = current_section and current_section.upper() in ("ASSUMPTION", "HYPOTHESIS")
            has_assumption_intro = (
                "assumed" in s_clean or
                "assumption" in s_clean or
                "assume" in s_clean or
                "hypothesis" in s_clean or
                is_in_assumption_sec
            )

            if has_assumption_intro:
                if "[unsupported]" in s_clean:
                    # Let factual grounding violations fail-closed rather than neutralizing them
                    processed_sentences.append(sentence)
                    continue

                # Audience
                if any(x in s_clean for x in ["audience", "demand", "interest", "popularity", "market"]):
                    if any(p in s_clean for p in ["viable", "exists", "exist", "reachable", "segment", "there is", "is a", "sufficient", "commercial", "high", "interested", "reach", "can reach", "willing"]):
                        sentence = "Audience demand remains unverified and audience viability remains unknown."
                    elif not any(x in s_clean for x in ["unverified", "unknown", "unspecified", "not established", "not been", "remains to be"]):
                        sentence = re.sub(
                            r'(?:(?:it\s+is\s+)?(?:assumed|hypothesized|assumes)(?:\s+that)?|(?:the\s+)?(?:assumption|hypothesis)(?:\s+is)?(?:\s+that)?)\s+(?:an?\s+)?(?:viable|commercially\s+viable|reachable|defined)?\s*(?:audience|demand|public\s+interest|market)(?:\s+(?:exists|is\s+reachable|is\s+viable|exists\s+and\s+is\s+reachable|is\s+defined|is\s+reachable\s+or\s+unverified|is\s+commercially\s+sustainable|is\s+commercially\s+viable))?[.\s]*',
                            "Audience demand remains unverified and whether a reachable audience exists remains unknown.",
                            sentence,
                            flags=re.IGNORECASE
                        )

                # Access
                elif "access" in s_clean or "coordination" in s_clean:
                    if not any(x in s_clean for x in ["unverified", "unknown", "unspecified", "not established", "not been", "remains to be"]):
                        sentence = re.sub(
                            r'(?:(?:it\s+is\s+)?(?:assumed|hypothesized|assumes)(?:\s+that)?|(?:the\s+)?(?:assumption|hypothesis)(?:\s+is)?(?:\s+that)?)\s+(?:desired|personnel|facility|hardware)?\s*(?:access|coordination)(?:\s+(?:is\s+available|can\s+be\s+obtained|exists|is\s+established))?[.\s]*',
                            "Access has not been established and remains unverified.",
                            sentence,
                            flags=re.IGNORECASE
                        )

                # Funding / Budget
                elif "funding" in s_clean or "budget" in s_clean:
                    if not any(x in s_clean for x in ["unverified", "unknown", "unspecified", "not established", "not been", "remains to be"]):
                        sentence = re.sub(
                            r'(?:(?:it\s+is\s+)?(?:assumed|hypothesized|assumes)(?:\s+that)?|(?:the\s+)?(?:assumption|hypothesis)(?:\s+is)?(?:\s+that)?)\s+(?:project|budget)?\s*(?:funding|budget)(?:\s+(?:exists|is\s+available))?[.\s]*',
                            "Funding status is unspecified and remains unverified.",
                            sentence,
                            flags=re.IGNORECASE
                        )

                # Rights / Authorization / Licensing / Clearance / Permission
                elif any(x in s_clean for x in ["rights", "authorization", "licensing", "clearance", "permission"]):
                    if not any(x in s_clean for x in ["unverified", "unknown", "unspecified", "not established", "not been", "remains to be"]):
                        sentence = re.sub(
                            r'(?:(?:it\s+is\s+)?(?:assumed|hypothesized|assumes)(?:\s+that)?|(?:the\s+)?(?:assumption|hypothesis)(?:\s+is)?(?:\s+that)?)\s+(?:applicable|necessary|custom)?\s*(?:rights|authorization|licensing|clearance|permission)(?:\s+(?:can\s+be\s+obtained|exists|is\s+available))?[.\s]*',
                            "Rights/authorization remain to be verified.",
                            sentence,
                            flags=re.IGNORECASE
                        )

                # Schedule / Timeline / Independence / Affect
                elif any(x in s_clean for x in ["schedule", "timeline", "timing", "independent", "independence", "affect", "affects", "impact", "impacts"]):
                    if not any(x in s_clean for x in ["unverified", "unknown", "unspecified", "not established", "not been", "remains to be"]):
                        if "independent" in s_clean or "independence" in s_clean:
                            sentence = "The relationship between the internal schedule and the external schedule is unverified."
                        elif any(x in s_clean for x in ["affect", "affects", "impact", "impacts"]):
                            sentence = "Whether the external schedule affects the internal production timeline remains unverified."
                        else:
                            sentence = "The production timeline and schedule relationship remains unverified."

            processed_sentences.append(sentence)

        processed_lines.append(label_part + "".join(processed_sentences))

    return "\n".join(processed_lines)


def split_structural_line(line: str) -> tuple[str, str] | None:
    """Detects and splits known structural labels, returning (label_prefix, body) or None."""
    labels_pattern = "|".join(re.escape(label) for label in KNOWN_LABELS)
    # Match structural headers like: "### 1. FINAL VERDICT" or "- **ANALYSIS**:" or "MISSING EVIDENCE:" or "E1 — Claim:"
    # Also matches labels followed by optional bracketed citation/explanatory suffixes
    pattern = re.compile(
        r"^([ \t]*(?:#+\s*)?(?:-\s*|\*\s*|\+\s*|\d+\s*\.\s*|[IVXLCDM]+\s*\.\s*)?(?:\*\*|\[)?(?:E\d+\s*(?:—|-)\s*)?(?:" + labels_pattern + r")(?:\s*\[[^\]]+\])?(?:\*\*|\])?(?:\s*(?::|—|-)\s*|\s*$))(.*)$",
        re.IGNORECASE
    )
    m = pattern.match(line)
    if m:
        return m.group(1), m.group(2)

    # Match arbitrary bold structural titles e.g. "* **Any Title**: "
    bold_pattern = re.compile(
        r"^([ \t]*(?:#+\s*)?(?:-\s*|\*\s*|\+\s*|\d+\s*\.\s*|[IVXLCDM]+\s*\.\s*)?\*\*[^\*]+\*\*(?:\s*(?::|—|-)\s*|\s*$))(.*)$",
        re.IGNORECASE
    )
    m_bold = bold_pattern.match(line)
    if m_bold:
        return m_bold.group(1), m_bold.group(2)

    # Match Decisive Reason N prefixes
    dr_pattern = re.compile(
        r"^([ \t]*(?:#+\s*)?(?:-\s*|\*\s*|\+\s*|\d+\s*\.\s*|[IVXLCDM]+\s*\.\s*)?DECISIVE REASON \d+(?:\s*(?::|—|-)\s*|\s*$))(.*)$",
        re.IGNORECASE
    )
    m_dr = dr_pattern.match(line)
    if m_dr:
        return m_dr.group(1), m_dr.group(2)

    # Fallback to match standalone index headers e.g. "### E1:" or "E1 —" or "E1"
    index_pattern = re.compile(
        r"^([ \t]*(?:#+\s*)?(?:-\s*|\*\s*|\+\s*|\d+\s*\.\s*|[IVXLCDM]+\s*\.\s*)?(?:E\d+)(?:\*\*|\])?(?:\s*(?::|—|-)\s*|\s*$))(.*)$",
        re.IGNORECASE
    )
    m2 = index_pattern.match(line)
    if m2:
        return m2.group(1), m2.group(2)

    # M7A.10 Fallback: Match general list markers/bullets, e.g., "1. ", "10. ", "- ", "* "
    list_marker_pattern = re.compile(
        r"^([ \t]*(?:-\s+|\*\s+|\+\s+|\d+\s*\.\s+))(.*)$",
        re.IGNORECASE
    )
    m3 = list_marker_pattern.match(line)
    if m3:
        return m3.group(1), m3.group(2)

    return None


def get_word_variations(word: str, expand_mappings: bool = False) -> set[str]:
    """Generates conservative, morphology-safe lowercase variations of a word.

    Ensures zero false matches for unrelated words (e.g., status/analysis/experimental).
    """
    w = word.lower().replace("’", "'")
    variations = {w}

    # Deterministic evidence-local lexical variations (M7A.16 revised, Source -> Variants)
    safe_local_mappings = {}

    if expand_mappings and w in safe_local_mappings:
        for val in safe_local_mappings[w]:
            variations.add(val)

    # 1. Strip possessives (fully safe)
    if w.endswith("'s"):
        variations.add(w[:-2])
    elif w.endswith("'"):
        variations.add(w[:-1])

    # 2. Safe Plural / Singular handling
    # Safeguard: Do not strip trailing 's' from words ending in non-plural 's' endings
    non_plural_s_endings = ("us", "is", "as", "os", "ss")

    if any(w.endswith(end) for end in non_plural_s_endings):
        pass
    elif w.endswith("ies") and len(w) > 4:
        variations.add(w[:-3] + "y")
    elif w.endswith("es") and len(w) > 4:
        # Strip 'es' if preceded by sibilants (ch, sh, x, s, z)
        base = w[:-2]
        if any(base.endswith(sib) for sib in ("ch", "sh", "x", "s", "z")):
            variations.add(base)
    elif w.endswith("s") and len(w) > 3:
        variations.add(w[:-1])

    # 3. Safe verbal inflections
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
        variations.add(base)
        variations.add(base + "e")  # e.g., timing -> time
        # Strip double consonants (e.g., planning -> plann -> plan)
        if len(base) > 3 and base[-1] == base[-2] and base[-1] in "bdfgklmnprstz":
            variations.add(base[:-1])

    if w.endswith("ed") and len(w) > 4:
        base = w[:-2]
        variations.add(base)
        variations.add(base + "e")  # e.g., measured -> measure
        # Strip double consonants (e.g., tapped -> tap)
        if len(base) > 3 and base[-1] == base[-2] and base[-1] in "bdfgklmnprstz":
            variations.add(base[:-1])

    if w.endswith("ied") and len(w) > 4:
        variations.add(w[:-3] + "y")  # e.g., verified -> verify

    return variations


def extract_supporting_excerpts(research_text: str) -> list[str]:
    """Robustly extracts all text values following 'Supporting Excerpt:' or 'Supporting Excerpts:' in the research text."""
    excerpts = []
    # Split by Supporting Excerpt(s) followed by optional bold markers and colon
    parts = re.split(r"Supporting Excerpts?\b(?:\s*\*\*)?\s*:", research_text, flags=re.IGNORECASE)
    # The first part is before the first Supporting Excerpt, so skip it
    for part in parts[1:]:
        val = part.strip()
        lines = val.split("\n")
        excerpt_lines = []
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            # Stop if the line starts with an evidence block marker or a standard label
            if (
                re.match(r"^E\d+\s*(?:[\u2014\u2013\-:])", line_stripped, re.IGNORECASE) or
                re.match(r"^#*\s*E\d+\b", line_stripped, re.IGNORECASE) or
                line_lower.startswith("claim:") or
                line_lower.startswith("- **claim**:") or
                line_lower.startswith("verification status:") or
                line_lower.startswith("- **verification status**:") or
                line_lower.startswith("source title:") or
                line_lower.startswith("- **source title**:") or
                line_lower.startswith("source url:") or
                line_lower.startswith("- **source url**:") or
                line_lower.startswith("publish date:") or
                line_lower.startswith("- **publish date**:") or
                "supporting excerpt" in line_lower
            ):
                break
            excerpt_lines.append(line)

        # Clean up blockquote markers and strip lines and quotes
        cleaned_lines = []
        for line in excerpt_lines:
            line_stripped = line.strip()
            if line_stripped.startswith(">"):
                line_stripped = line_stripped[1:].strip()
            # Strip quotes on a line level only if they fully wrap the line
            if line_stripped.startswith('"') and line_stripped.endswith('"'):
                line_stripped = line_stripped[1:-1].strip()
            cleaned_lines.append(line_stripped)

        excerpt_text = "\n".join(cleaned_lines).strip()
        # Strip surrounding double quotes if present on the entire block
        if excerpt_text.startswith('"') and excerpt_text.endswith('"'):
            excerpt_text = excerpt_text[1:-1].strip()
        elif excerpt_text.startswith('"'):
            excerpt_text = excerpt_text.strip('"').strip()

        if excerpt_text:
            excerpts.append(excerpt_text)

    return excerpts


def get_research_text(ctx) -> str:
    if not ctx or not ctx.session or not ctx.session.events:
        return ""
    for event in reversed(ctx.session.events):
        if event.author == "research_agent":
            text = ""
            if isinstance(event.output, dict):
                text = event.output.get("research_evidence", "")
            elif isinstance(event.output, str):
                text = event.output
            elif event.content and event.content.parts:
                text = "".join(
                    part.text for part in event.content.parts if part.text
                )
            if text.strip():
                return text
    return ""


def get_director_text(ctx) -> str:
    if not ctx or not ctx.session or not ctx.session.events:
        return ""
    for event in reversed(ctx.session.events):
        if event.author == "director_agent":
            text = ""
            if isinstance(event.output, dict):
                text = event.output.get("director_plan", "")
            elif isinstance(event.output, str):
                text = event.output
            elif event.content and event.content.parts:
                text = "".join(
                    part.text for part in event.content.parts if part.text
                )
            if text.strip():
                return text
    return ""


def get_user_text(ctx) -> str:
    if not ctx:
        return ""
    if hasattr(ctx, "user_content") and ctx.user_content and ctx.user_content.parts:
        text = "".join(part.text for part in ctx.user_content.parts if part.text)
        if text.strip():
            return text
    if hasattr(ctx, "session") and ctx.session and hasattr(ctx.session, "events"):
        for event in reversed(ctx.session.events):
            if getattr(event, "author", "") == "user":
                text = ""
                if isinstance(event.output, dict):
                    text = event.output.get("user", "")
                elif isinstance(event.output, str):
                    text = event.output
                elif getattr(event, "content", None) and event.content.parts:
                    text = "".join(part.text for part in event.content.parts if part.text)
                if text.strip():
                    return text
    return ""


# Conservative normalization maps and helper functions for M7A.14
NUMBER_WORDS_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10"
}

SUPERSCRIPTS_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

def extract_claims(research_text: str) -> list[str]:
    """Robustly extracts all text values following 'Claim:' or 'Claims:' in the research text."""
    claims = []
    # Split by Claim(s) followed by optional bold markers and colon
    parts = re.split(r"Claims?\b(?:\s*\*\*)?\s*:", research_text, flags=re.IGNORECASE)
    # The first part is before the first Claim, so skip it
    for part in parts[1:]:
        val = part.strip()
        lines = val.split("\n")
        claim_lines = []
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            # Stop if the line starts with an evidence block marker or a standard label
            if (
                re.match(r"^E\d+\s*(?:[\u2014\u2013\-:])", line_stripped, re.IGNORECASE) or
                re.match(r"^#*\s*E\d+\b", line_stripped, re.IGNORECASE) or
                line_lower.startswith("verification status:") or
                line_lower.startswith("- **verification status**:") or
                line_lower.startswith("source title:") or
                line_lower.startswith("- **source title**:") or
                line_lower.startswith("source url:") or
                line_lower.startswith("- **source url**:") or
                line_lower.startswith("publish date:") or
                line_lower.startswith("- **publish date**:") or
                line_lower.startswith("supporting excerpt:") or
                line_lower.startswith("- **supporting excerpt**:") or
                "supporting excerpt" in line_lower or
                "claim:" in line_lower or
                "claims:" in line_lower or
                "- **claim**:" in line_lower or
                "* **claim**:" in line_lower
            ):
                break
            claim_lines.append(line)

        # Clean up blockquote markers and strip lines and quotes
        cleaned_lines = []
        for line in claim_lines:
            line_stripped = line.strip()
            if line_stripped.startswith(">"):
                line_stripped = line_stripped[1:].strip()
            # Strip quotes on a line level only if they fully wrap the line
            if line_stripped.startswith('"') and line_stripped.endswith('"'):
                line_stripped = line_stripped[1:-1].strip()
            cleaned_lines.append(line_stripped)

        claim_text = "\n".join(cleaned_lines).strip()
        # Strip surrounding double quotes if present on the entire block
        if claim_text.startswith('"') and claim_text.endswith('"'):
            claim_text = claim_text[1:-1].strip()
        elif claim_text.startswith('"'):
            claim_text = claim_text.strip('"').strip()

        if claim_text:
            claims.append(claim_text)

    return claims


def extract_and_normalize_words(text: str) -> set[str]:
    """Helper to extract words from a text block and apply conservative normalizations."""
    normalized_text = text.translate(SUPERSCRIPTS_MAP)
    words_set = set()

    # 1. Standard word find
    found_words = re.findall(r"[a-zA-Z0-9\-]+", normalized_text)
    for w in found_words:
        words_set.add(w.lower())
        if "-" in w:
            for sub in w.split("-"):
                if sub:
                    words_set.add(sub.lower())

    # 2. Find numbers with commas (e.g. 14,600) and add their comma-less version
    for num_with_commas in re.findall(r"\b\d{1,3}(?:,\d{3})+\b", normalized_text):
        words_set.add(num_with_commas.replace(",", ""))

    # 3. Split mixed alphanumeric words (e.g. 45m3 or 10m)
    for w in list(words_set):
        if re.search(r"\d", w) and re.search(r"[a-zA-Z]", w):
            parts = re.findall(r"\d+|[a-zA-Z]+", w)
            for p in parts:
                words_set.add(p.lower())

    # 4. Add number-word to digit mappings and vice versa
    for word_num, digit in NUMBER_WORDS_MAP.items():
        if word_num in words_set:
            words_set.add(digit)
        if digit in words_set:
            words_set.add(word_num)

    return words_set


def split_table_row_cells(line: str) -> list[str]:
    cells = []
    current_cell = []
    escaped = False

    for char in line:
        if escaped:
            current_cell.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
            current_cell.append(char)
        elif char == "|":
            cells.append("".join(current_cell))
            current_cell = []
        else:
            current_cell.append(char)
    cells.append("".join(current_cell))

    if cells and cells[0] == "":
        cells.pop(0)
    if cells and cells[-1].strip() == "":
        cells.pop()

    return [c.strip() for c in cells]


def parse_evidence_ledger_table(research_text: str) -> tuple[dict[str, list[str]], dict[str, list[str]]] | None:
    if not research_text:
        return None

    lines = [line.strip() for line in research_text.split("\n") if line.strip()]

    header_idx = -1
    for idx, line in enumerate(lines):
        if "|" in line:
            cells = [c.lower() for c in split_table_row_cells(line)]
            if any("e#" in c for c in cells) and any("claim" in c for c in cells) and any("supporting excerpt" in c for c in cells):
                header_idx = idx
                break

    if header_idx == -1:
        return None

    if header_idx + 1 >= len(lines):
        return None

    sep_line = lines[header_idx + 1]
    if not sep_line.startswith("|"):
        return None
    if not re.match(r"^[|:\-\s]+$", sep_line):
        return None

    header_cells = [c.lower() for c in split_table_row_cells(lines[header_idx])]
    col_e = -1
    col_claim = -1
    col_excerpt = -1

    for c_idx, cell in enumerate(header_cells):
        if "e#" in cell:
            col_e = c_idx
        elif "claim" in cell:
            col_claim = c_idx
        elif "supporting excerpt" in cell:
            col_excerpt = c_idx

    if col_e == -1 or col_claim == -1 or col_excerpt == -1:
        return None

    excerpts_map = {}
    claims_map = {}

    for line in lines[header_idx + 2:]:
        if not line.startswith("|"):
            continue

        cells = split_table_row_cells(line)
        if len(cells) <= max(col_e, col_claim, col_excerpt):
            # Malformed row with inconsistent cell counts
            return {}, {}

        e_raw = cells[col_e]
        e_clean = re.sub(r"[*\s_`]", "", e_raw).lower()
        if not re.match(r"^e\d+$", e_clean):
            if not e_clean:
                continue
            # Malformed row with invalid E#
            return {}, {}

        claim_val = cells[col_claim].strip()
        claim_val = claim_val.replace("\\|", "|")
        if claim_val.startswith(">"):
            claim_val = claim_val[1:].strip()
        if claim_val.startswith('"') and claim_val.endswith('"'):
            claim_val = claim_val[1:-1].strip()
        elif claim_val.startswith('"'):
            claim_val = claim_val.strip('"').strip()

        excerpt_val = cells[col_excerpt].strip()
        excerpt_val = excerpt_val.replace("\\|", "|")
        if excerpt_val.startswith(">"):
            excerpt_val = excerpt_val[1:].strip()
        if excerpt_val.startswith('"') and excerpt_val.endswith('"'):
            excerpt_val = excerpt_val[1:-1].strip()
        elif excerpt_val.startswith('"'):
            excerpt_val = excerpt_val.strip('"').strip()

        if claim_val:
            if e_clean not in claims_map:
                claims_map[e_clean] = []
            claims_map[e_clean].append(claim_val)

        if excerpt_val:
            if e_clean not in excerpts_map:
                excerpts_map[e_clean] = []
            excerpts_map[e_clean].append(excerpt_val)

    return excerpts_map, claims_map


def merge_evidence_maps(map1: dict[str, list[str]], map2: dict[str, list[str]]) -> dict[str, list[str]]:
    merged = {}
    for k in set(map1.keys()) | set(map2.keys()):
        merged[k] = []
        if k in map1:
            merged[k].extend(map1[k])
        if k in map2:
            merged[k].extend(map2[k])
    return merged


def get_allowed_words(ctx) -> set[str]:
    allowed = set()
    # 1. From Supporting Excerpts & Claims
    research_text = get_research_text(ctx)
    if research_text:
        # Direct sequential/bullet extraction for backward compatibility with headerless/mock ledgers
        excerpts = extract_supporting_excerpts(research_text)
        for exc in excerpts:
            allowed.update(extract_and_normalize_words(exc))

        claims = extract_claims(research_text)
        for clm in claims:
            allowed.update(extract_and_normalize_words(clm))

        # Map-based extraction to support Markdown tables and sequential formats with E# headers
        excerpts_map = get_evidence_excerpts_map(research_text)
        for excerpts in excerpts_map.values():
            for exc in excerpts:
                allowed.update(extract_and_normalize_words(exc))

        claims_map = get_evidence_claims_map(research_text)
        for claims in claims_map.values():
            for clm in claims:
                allowed.update(extract_and_normalize_words(clm))

    # 2. From Director Plan
    director_text = get_director_text(ctx)
    if director_text:
        words = re.findall(r"[a-zA-Z0-9\-]+", director_text)
        for w in words:
            allowed.add(w.lower())
            if "-" in w:
                for sub in w.split("-"):
                    if sub:
                        allowed.add(sub.lower())

    # 3. From User content
    user_text = get_user_text(ctx)
    if user_text:
        words = re.findall(r"[a-zA-Z0-9\-]+", user_text)
        for w in words:
            allowed.add(w.lower())
            if "-" in w:
                for sub in w.split("-"):
                    if sub:
                        allowed.add(sub.lower())

    # 4. Add system vocabulary & common words
    for w in SYSTEM_ALLOWED_WORDS:
        allowed.add(w.lower())
    for w in COMMON_STOP_WORDS:
        allowed.add(w.lower())

    return allowed


def get_evidence_excerpts_map(research_text: str) -> dict[str, list[str]]:
    """Maps evidence keys (e.g. 'e1', 'e2') to their list of Supporting Excerpts."""
    if not research_text:
        return {}

    table_excerpts = {}
    has_table = False
    if "|" in research_text and "E#" in research_text:
        table_parsed = parse_evidence_ledger_table(research_text)
        if table_parsed is not None:
            table_excerpts, _ = table_parsed
            has_table = True
            _trace_log("Ledger format detected: table")

    seq_excerpts = {}
    # Regex to find evidence headers: E# at start of lines (possibly with Markdown decoration)
    header_pattern = re.compile(
        r'(?:^|\n)[ \t]*(?:'
        r'(#+)[ \t]*(E\d+)\b|'  # Case 1: Markdown heading, e.g., ### E1
        r'(?:-|\*|\d+\.)?[ \t]*(E\d+)[ \t]*([\u2014\u2013\-:])|'  # Case 2: Separators, e.g., E1:, E1 -
        r'(?:-|\*|\d+\.)?[ \t]*(E\d+)[ \t]*(?:\n|$)'  # Case 3: Standalone line, e.g., E1 or - E1
        r')',
        re.IGNORECASE
    )

    matches = list(header_pattern.finditer(research_text))
    if not matches:
        # Fallback to the original split behavior just in case
        parts = re.split(r'\b(E\d+)\s*(?:[\u2014\u2013\-:])', research_text, flags=re.IGNORECASE)
        for i in range(1, len(parts), 2):
            key = parts[i].lower().strip()
            body = parts[i+1]
            excerpts = extract_supporting_excerpts(body)
            if key not in seq_excerpts:
                seq_excerpts[key] = []
            seq_excerpts[key].extend(excerpts)
    else:
        for idx, match in enumerate(matches):
            key = (match.group(2) or match.group(3) or match.group(5)).lower().strip()
            start_idx = match.end()
            end_idx = matches[idx + 1].start() if idx + 1 < len(matches) else len(research_text)
            body = research_text[start_idx:end_idx]
            excerpts = extract_supporting_excerpts(body)
            if key not in seq_excerpts:
                seq_excerpts[key] = []
            seq_excerpts[key].extend(excerpts)

    if has_table:
        return merge_evidence_maps(table_excerpts, seq_excerpts)
    _trace_log("Ledger format detected: sequential")
    return seq_excerpts


def get_evidence_claims_map(research_text: str) -> dict[str, list[str]]:
    """Maps evidence keys (e.g. 'e1', 'e2') to their list of Claims."""
    if not research_text:
        return {}

    table_claims = {}
    has_table = False
    if "|" in research_text and "E#" in research_text:
        table_parsed = parse_evidence_ledger_table(research_text)
        if table_parsed is not None:
            _, table_claims = table_parsed
            has_table = True

    seq_claims = {}
    header_pattern = re.compile(
        r'(?:^|\n)[ \t]*(?:'
        r'(#+)[ \t]*(E\d+)\b|'  # Case 1: Markdown heading, e.g., ### E1
        r'(?:-|\*|\d+\.)?[ \t]*(E\d+)[ \t]*([\u2014\u2013\-:])|'  # Case 2: Separators, e.g., E1:, E1 -
        r'(?:-|\*|\d+\.)?[ \t]*(E\d+)[ \t]*(?:\n|$)'  # Case 3: Standalone line, e.g., E1 or - E1
        r')',
        re.IGNORECASE
    )

    matches = list(header_pattern.finditer(research_text))
    if not matches:
        parts = re.split(r'\b(E\d+)\s*(?:[\u2014\u2013\-:])', research_text, flags=re.IGNORECASE)
        for i in range(1, len(parts), 2):
            key = parts[i].lower().strip()
            body = parts[i+1]
            claims = extract_claims(body)
            if key not in seq_claims:
                seq_claims[key] = []
            seq_claims[key].extend(claims)
    else:
        for idx, match in enumerate(matches):
            key = (match.group(2) or match.group(3) or match.group(5)).lower().strip()
            start_idx = match.end()
            end_idx = matches[idx + 1].start() if idx + 1 < len(matches) else len(research_text)
            body = research_text[start_idx:end_idx]
            claims = extract_claims(body)
            if key not in seq_claims:
                seq_claims[key] = []
            seq_claims[key].extend(claims)

    if has_table:
        return merge_evidence_maps(table_claims, seq_claims)
    return seq_claims


def parse_cited_evidence_ids(line: str) -> list[str]:
    """Finds all E# references cited in a line (e.g., 'E1', 'E2')."""
    matches = re.findall(r'\bE(\d+)\b', line, flags=re.IGNORECASE)
    return [f"e{m}" for m in matches]


def is_analytical_or_uncertainty_line(line: str) -> bool:
    """Detects if a line is an analytical statement, uncertainty, or next action."""
    neutral_patterns = [
        r"\bremains\s+unverified\b",
        r"\bremains\s+unknown\b",
        r"\bmissing\s+evidence\b",
        r"\bremains\s+unresolved\b",
        r"\bis\s+not\s+established\b",
        r"\bis\s+unverified\b",
        r"\bnot\s+supplied\b",
        r"\bunspecified\b",
        r"\bverify\s+first\b",
        r"\bstrategic\s+action\b",
        r"\bdetermine\s+whether\b",
        r"\bwhether\s+[\w\s\-]+?\s+(?:exists|is|can|remains|affects)\b",
        r"\bviability\s+remains\s+unverified\b",
        r"\bsustainability\s+remains\s+unverified\b",
        r"\bviability\s+is\s+not\s+established\b",
        r"\bviability\s+is\s+unverified\b",
        r"\bproject-specific\s+viability\b"
    ]
    for pattern in neutral_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def neutralize_unauthorized_in_action(sentence: str, unauthorized_words: list[str]) -> str:
    """Attempts to deterministically neutralize unauthorized proper nouns or numbers in an action sentence.

    Returns the neutralized sentence if successful, or None if it cannot be safely neutralized
    (in which case it will fail closed). Modifies the unauthorized_words list in place if neutralized.
    """
    ns = sentence
    unauth_set = set(unauthorized_words)

    # 1. Regulators (e.g. "FAA licensing requirements")
    for w in list(unauth_set):
        if w.isupper() or w[0].isupper():
            verb_match = re.match(r'^(\s*(?:-\s*|\*\s*|\d+\.\s*)?)(verify|determine|evaluate|assess|confirm|investigate|obtain|coordinate|align|track|clarify|ensure|analyze|identify|explore|mitigate|address|check|review|compare|establish)\b', ns, re.IGNORECASE)
            if verb_match:
                prefix = verb_match.group(1)
                verb = verb_match.group(2)
                verb_body = ns[len(prefix) + len(verb):].strip()

                reg_pattern = re.compile(
                    r"\b" + re.escape(w) + r"\s+(licens(?:ing|e)|clearance|authorization|approval|permit|pathway|requirement)s?\b",
                    re.IGNORECASE
                )
                if reg_pattern.search(verb_body):
                    verb_body_replaced = reg_pattern.sub(r"the applicable \1", verb_body)
                    ns = f"{prefix}Determine which regulator, if any, applies and verify {verb_body_replaced}"
                    unauth_set.discard(w)
                    break

    # 2. Locations with location word after (e.g. "access at Long Beach", "permissions in Seattle")
    for w in list(unauth_set):
        if w[0].isupper():
            verb_match = re.match(r'^(\s*(?:-\s*|\*\s*|\d+\.\s*)?)(verify|determine|evaluate|assess|confirm|investigate|obtain|coordinate|align|track|clarify|ensure|analyze|identify|explore|mitigate|address|check|review|compare|establish)\b', ns, re.IGNORECASE)
            if verb_match:
                prefix = verb_match.group(1)
                verb = verb_match.group(2)
                verb_body = ns[len(prefix) + len(verb):].strip()

                loc_pattern = re.compile(
                    r"\b(access|facility|facilities|permission|permissions|conditions|coordination|filming|production)\s+(?:at|in|within|for)\s+([^.\?!,]+)\b",
                    re.IGNORECASE
                )
                m = loc_pattern.search(verb_body)
                if m:
                    noun_part = m.group(1)
                    loc_part = m.group(2).strip()
                    if w in loc_part or w.lower() in loc_part.lower():
                        if noun_part.lower() == "access":
                            generic_phrase = "the applicable access conditions"
                        elif noun_part.lower() in ("facility", "facilities"):
                            generic_phrase = "the applicable facility conditions"
                        elif noun_part.lower() in ("permission", "permissions"):
                            generic_phrase = "the applicable permissions"
                        else:
                            generic_phrase = f"the applicable {noun_part.lower()}"

                        whole_phrase_pattern = re.compile(re.escape(m.group(0)))
                        verb_body_replaced = whole_phrase_pattern.sub(generic_phrase, verb_body)
                        ns = f"{prefix}Confirm which location, if any, is relevant and verify {verb_body_replaced}"

                        loc_words = re.findall(r"\b[A-Za-z0-9\-]+\b", loc_part)
                        for lw in loc_words:
                            unauth_set.discard(lw)
                        break

    # 3. Locations with location word before (e.g. "Seattle facility permissions")
    for w in list(unauth_set):
        if w[0].isupper():
            verb_match = re.match(r'^(\s*(?:-\s*|\*\s*|\d+\.\s*)?)(verify|determine|evaluate|assess|confirm|investigate|obtain|coordinate|align|track|clarify|ensure|analyze|identify|explore|mitigate|address|check|review|compare|establish)\b', ns, re.IGNORECASE)
            if verb_match:
                prefix = verb_match.group(1)
                verb = verb_match.group(2)
                verb_body = ns[len(prefix) + len(verb):].strip()

                loc_pattern_before = re.compile(
                    r"\b([A-Z][a-zA-Z0-9\s\-]+)\s+(access|facility|facilities|permission|permissions|conditions|coordination|filming|production)s?\b",
                    re.IGNORECASE
                )
                m2 = loc_pattern_before.search(verb_body)
                if m2:
                    loc_part = m2.group(1).strip()
                    noun_part = m2.group(2)
                    if w in loc_part or w.lower() in loc_part.lower():
                        if noun_part.lower() == "access":
                            generic_phrase = "the applicable access conditions"
                        elif noun_part.lower() in ("facility", "facilities"):
                            generic_phrase = "the applicable facility permissions"
                        elif noun_part.lower() in ("permission", "permissions"):
                            generic_phrase = "the applicable permissions"
                        else:
                            generic_phrase = f"the applicable {noun_part.lower()}"

                        whole_phrase_pattern = re.compile(re.escape(m2.group(0)))
                        verb_body_replaced = whole_phrase_pattern.sub(generic_phrase, verb_body)
                        ns = f"{prefix}Confirm which location, if any, is relevant and verify {verb_body_replaced}"

                        loc_words = re.findall(r"\b[A-Za-z0-9\-]+\b", loc_part)
                        for lw in loc_words:
                            unauth_set.discard(lw)
                        break

    unauthorized_words.clear()
    unauthorized_words.extend(sorted(list(unauth_set)))
    return ns


def lowercase_sentence_starts(text: str) -> str:
    # Split text into sentences using standard sentence separators (. ! ?)
    sentence_end = re.compile(r'([.!?]\s+)')
    parts = sentence_end.split(text)
    processed = []
    for part in parts:
        if re.match(r'^[.!?]\s+$', part):
            processed.append(part)
        else:
            # Find the first alphabetic character and lowercase it
            m = re.search(r'[a-zA-Z]', part)
            if m:
                idx = m.start()
                processed_part = part[:idx] + part[idx].lower() + part[idx+1:]
                processed.append(processed_part)
            else:
                processed.append(part)
    return "".join(processed)


def clean_and_validate_hidden_facts(text: str, allowed_words: set[str], ctx=None) -> str:
    """Finds proper nouns and numbers that do not exist in the allowed words set and redacts them.

    Processes line-by-line:
    - Bypasses explicitly recognized CineVerdict structural labels.
    - Validates only body content following the label.
    - Uses conservative morphological variations for checking.
    """
    text = text.replace("&#58;", ":")
    research_text = get_research_text(ctx) if ctx else ""
    ev_map = get_evidence_excerpts_map(research_text) if research_text else {}
    ev_claims_map = get_evidence_claims_map(research_text) if research_text else {}
    _trace_log(f"[Stage 4] Evidence-scope construction. Research text length: {len(research_text)}. Evidence map keys: {list(ev_map.keys())}")

    lines = text.split("\n")
    processed_lines = []

    current_section = None
    active_evidence_scope = None

    for line in lines:
        if not line.strip():
            processed_lines.append(line)
            continue

        # Detect and split known structural label
        split_res = split_structural_line(line)
        matched_label = None
        if split_res:
            label_part, body_part = split_res
            _trace_log(f"[Stage 2] Structural splitting: Line has known label. Label: '{label_part}', Body: '{body_part}'")

            # M7A.10: Update current_section if a known label is found in label_part
            for label in KNOWN_LABELS:
                if label.lower() in label_part.lower():
                    current_section = label
                    matched_label = label
                    _trace_log(f"  [M7A.10 Section Tracking] Active section updated to: '{current_section}'")
                    break
        else:
            label_part = ""
            body_part = line
            _trace_log(f"[Stage 2] Structural splitting: No known label found. Body: '{body_part}'")

        # Track active evidence scope hierarchically
        if matched_label:
            if matched_label.upper() in ("VERIFIED EVIDENCE", "SECONDARY EVIDENCE", "CONFLICTING EVIDENCE"):
                parent_citations = parse_cited_evidence_ids(line)
                active_evidence_scope = parent_citations
                _trace_log(f"  [Evidence Scope Tracking] Active evidence scope updated to: {active_evidence_scope}")
            else:
                active_evidence_scope = None
                _trace_log(f"  [Evidence Scope Tracking] Active evidence scope cleared (section: {matched_label})")

        # Split body_part into sentences
        sentence_end = re.compile(r'([.!?]\s+)')
        parts = sentence_end.split(body_part)
        sentences = []
        i = 0
        while i < len(parts):
            s = parts[i]
            if i + 1 < len(parts):
                s += parts[i+1]
                i += 2
            else:
                i += 1
            if s:
                sentences.append(s)

        processed_sentences = []
        label_citations = parse_cited_evidence_ids(label_part) if split_res else []

        for sentence in sentences:
            # Construct the allowed words set for this sentence
            sentence_citations = parse_cited_evidence_ids(sentence)

            # Combine label citations and sentence citations, preserving order and removing duplicates
            combined_cits = []
            for cid in label_citations + sentence_citations:
                if cid not in combined_cits:
                    combined_cits.append(cid)
            explicit_citations = combined_cits

            _trace_log(f"[Stage 3] Citation parsing: Cited IDs on line: {explicit_citations}")

            if ev_map:
                # We have dynamic evidence mapping
                effective_cited_ids = None
                if explicit_citations:
                    effective_cited_ids = explicit_citations
                    _trace_log(f"  [Stage 4] Using explicit line citations: {effective_cited_ids}")
                elif active_evidence_scope is not None:
                    effective_cited_ids = active_evidence_scope
                    _trace_log(f"  [Stage 4] Inheriting parent evidence scope: {effective_cited_ids}")

                if effective_cited_ids is not None:
                    line_allowed = set()
                    for cid in effective_cited_ids:
                        excerpts = ev_map.get(cid, [])
                        _trace_log(f"  [Stage 4] Selected excerpts for {cid}: {excerpts}")
                        for exc in excerpts:
                            line_allowed.update(extract_and_normalize_words(exc))
                        claims = ev_claims_map.get(cid, [])
                        _trace_log(f"  [Stage 4] Selected claims for {cid}: {claims}")
                        for clm in claims:
                            line_allowed.update(extract_and_normalize_words(clm))

                    # Add Director Plan words
                    director_text = get_director_text(ctx)
                    if director_text:
                        words = re.findall(r"[a-zA-Z0-9\-]+", director_text)
                        for w in words:
                            line_allowed.add(w.lower())
                            if "-" in w:
                                for sub in w.split("-"):
                                    if sub:
                                        line_allowed.add(sub.lower())

                    # Add User content words
                    user_text = get_user_text(ctx)
                    if user_text:
                        words = re.findall(r"[a-zA-Z0-9\-]+", user_text)
                        for w in words:
                            line_allowed.add(w.lower())
                            if "-" in w:
                                for sub in w.split("-"):
                                    if sub:
                                        line_allowed.add(sub.lower())

                    # Add system vocabulary & common words
                    for w in SYSTEM_ALLOWED_WORDS:
                        line_allowed.add(w.lower())
                    for w in COMMON_STOP_WORDS:
                        line_allowed.add(w.lower())
                    _trace_log(f"  [Stage 4] Custom line-level allowed words constructed. Size: {len(line_allowed)}")
                else:
                    # No explicit citations and no active_evidence_scope. Fall back to global allowed words.
                    _trace_log(f"  [Stage 4] Falling back to global allowed words.")
                    line_allowed = allowed_words
            else:
                _trace_log(f"  [Stage 4] Falling back to global allowed words.")
                line_allowed = allowed_words

            # Pre-expand allowed words to include all of their conservative variations
            expanded_allowed = set()
            for w in line_allowed:
                for var in get_word_variations(w, expand_mappings=True):
                    expanded_allowed.add(var)

            content_role = classify_sentence_role(sentence)
            sentence_role = content_role

            # M7A.10 Section-based role override for semantic accuracy
            if current_section:
                sec_upper = current_section.upper()
                if sec_upper in ("MISSING EVIDENCE", "UNRESOLVED UNCERTAINTIES"):
                    sentence_role = "uncertainty"
                elif sec_upper in ("REQUIRED NEXT ACTIONS", "VERIFY FIRST", "SUPPORTED ACTION", "STRATEGIC ACTION"):
                    sentence_role = "action"
                elif sec_upper in ("ASSUMPTION", "HYPOTHESIS"):
                    sentence_role = "analytical_assumption"

            _trace_log(f"  [Stage 5] Semantic proposition classification: Sentence: '{sentence.strip()}' -> Role: '{sentence_role}' (Content Role: '{content_role}', Section: '{current_section}')")

            sentence_for_extraction = sentence
            _trace_log(f"    [Stage 6] Keeping original sentence capitalization: '{sentence_for_extraction.strip()}'")

            raw_tokens = re.findall(r"\b[A-Z][a-zA-Z0-9\-]*\b|\b\d+\b", sentence_for_extraction)
            significant_words = set(raw_tokens)

            # Exclude genuine introductory non-substantive discourse markers from significant_words
            discourse_markers = {"additionally", "furthermore", "moreover"}
            clean_s = re.sub(r'^(?:-\s*|\*\s*|\d+\.\s*)', '', sentence).strip()
            first_word_match = re.match(r'^([a-zA-Z]+)\s*,', clean_s)
            if first_word_match:
                intro_word = first_word_match.group(1).lower()
                if intro_word in discourse_markers:
                    significant_words.discard(first_word_match.group(1))

            _trace_log(f"    [Stage 7/8] Extracted tokens: {list(significant_words)}")

            unauthorized = []
            if ev_map and explicit_citations:
                nonexistent_cits = [cid for cid in explicit_citations if cid not in ev_map]
                if nonexistent_cits:
                    unauthorized.append(nonexistent_cits[0])

            for w in significant_words:
                w_lower = w.lower()

                # Check conservative morphological variations of the word
                w_vars = get_word_variations(w_lower)
                is_authorized = (
                    any(var in expanded_allowed for var in w_vars)
                    or w_lower in COMMON_STOP_WORDS
                    or w_lower in SYSTEM_ALLOWED_WORDS
                    or any(var in SYSTEM_ALLOWED_WORDS for var in w_vars)
                    or bool(re.match(r"^[eE]\d+$", w_lower))
                )
                _trace_log(f"      [Stage 9] Word check for '{w}': Variations checked: {list(w_vars)}")
                _trace_log(f"        Match in expanded_allowed: {any(var in expanded_allowed for var in w_vars)}")
                _trace_log(f"        Match in system/common: {w_lower in COMMON_STOP_WORDS or w_lower in SYSTEM_ALLOWED_WORDS or any(var in SYSTEM_ALLOWED_WORDS for var in w_vars)}")

                if not is_authorized and sentence_role != "factual":
                    is_authorized_analytical = (
                        w_lower in ANALYTICAL_SUBSTANTIVE_WORDS
                        or any(var in ANALYTICAL_SUBSTANTIVE_WORDS for var in w_vars)
                    )
                    _trace_log(f"        Match in ANALYTICAL_SUBSTANTIVE_WORDS: {is_authorized_analytical}")
                    if is_authorized_analytical:
                        is_authorized = True
                    else:
                        if "-" in w_lower:
                            parts_list = [p for p in w_lower.split("-") if p]
                            if parts_list and all(
                                sub in ANALYTICAL_SUBSTANTIVE_WORDS or any(var in ANALYTICAL_SUBSTANTIVE_WORDS for var in get_word_variations(sub))
                                for sub in parts_list
                            ):
                                is_authorized = True
                                _trace_log(f"        Sub-parts check in ANALYTICAL_SUBSTANTIVE_WORDS for '{w_lower}': Authorized")

                if not is_authorized and len(w) > 1:
                    unauthorized.append(w)
                    _trace_log(f"        Result for '{w}': UNAUTHORIZED (Rule: token/entity/value judged unsupported)")
                else:
                    _trace_log(f"        Result for '{w}': AUTHORIZED")

            if sentence_role == "action" and unauthorized:
                _trace_log(f"    [Stage: Action Neutralization] Attempting to neutralize unauthorized tokens in action sentence: {unauthorized}")
                neutralized_sentence = neutralize_unauthorized_in_action(sentence, unauthorized)
                if neutralized_sentence != sentence:
                    _trace_log(f"    [Stage: Action Neutralization] Neutralized action sentence.\n      Before: '{sentence.strip()}'\n      After: '{neutralized_sentence.strip()}'\n      Remaining unauthorized tokens: {unauthorized}")
                    sentence = neutralized_sentence

            unauthorized.sort(key=len, reverse=True)
            validated_sentence = sentence
            if unauthorized:
                _trace_log(f"    [Stage 10] Unsupported tokens to redact in sentence: {unauthorized}")
            for w in unauthorized:
                validated_sentence = re.sub(r"\b" + re.escape(w) + r"\b", "[UNSUPPORTED]", validated_sentence, flags=re.IGNORECASE)

            if validated_sentence != sentence:
                _trace_log(f"    [Stage 10] Unsupported marker insertion: Sentence after redaction: '{validated_sentence.strip()}'")
            processed_sentences.append(validated_sentence)

        processed_lines.append(label_part + "".join(processed_sentences))

    return "\n".join(processed_lines)


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

    sentence_end = re.compile(r'([.!?]\s+)')
    parts = sentence_end.split(text)
    processed = []
    i = 0
    while i < len(parts):
        s = parts[i]
        if i + 1 < len(parts):
            s += parts[i+1]
            i += 2
        else:
            i += 1
        if s:
            if is_analytical_or_uncertainty_line(s):
                processed.append(s)
            else:
                placeholders = {}
                for idx, pattern in enumerate(sorted_keys):
                    placeholder = f"___AUD_PLACEHOLDER_{idx}___"
                    s, count = re.subn(pattern, placeholder, s, flags=re.IGNORECASE)
                    if count > 0:
                        placeholders[placeholder] = mappings[pattern]
                for placeholder, final_val in placeholders.items():
                    s = s.replace(placeholder, final_val)
                processed.append(s)
    return "".join(processed)


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

    sentence_end = re.compile(r'([.!?]\s+)')
    parts = sentence_end.split(text)
    processed = []
    i = 0
    while i < len(parts):
        s = parts[i]
        if i + 1 < len(parts):
            s += parts[i+1]
            i += 2
        else:
            i += 1
        if s:
            if is_analytical_or_uncertainty_line(s):
                processed.append(s)
            else:
                placeholders = {}
                for idx, pattern in enumerate(sorted_keys):
                    placeholder = f"___PROD_PLACEHOLDER_{idx}___"
                    s, count = re.subn(pattern, placeholder, s, flags=re.IGNORECASE)
                    if count > 0:
                        placeholders[placeholder] = mappings[pattern]
                for placeholder, final_val in placeholders.items():
                    s = s.replace(placeholder, final_val)
                processed.append(s)
    return "".join(processed)


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
        clean_pattern = pattern.replace(r"\b", "")
        raw_words = re.findall(r"[a-z]+", clean_pattern.lower())
        if all(w in allowed_words for w in raw_words):
            continue

        placeholder = f"___EVAL_PLACEHOLDER_{i}___"
        text, count = re.subn(pattern, placeholder, text, flags=re.IGNORECASE)
        if count > 0:
            placeholders[placeholder] = evaluative_mappings[pattern]

    for placeholder, final_val in placeholders.items():
        text = text.replace(placeholder, final_val)

    return text


def neutralize_evidence_strength_upgrades(text: str) -> str:
    """Neutralizes upgrades of historical/contextual evidence into positive viability evidence for the proposed project."""
    mappings = {
        # 1. Historical/other demand multiples/success upgraded to project demand/viability
        r"\b(?:historical|existing|other|measured)?\s*(?:demand|success|multiples|measurements|data|examples|titles|observations)(?:\s+(?:of|for|about|concerning)\s+[\w\s\-]+?)?\s+(?:demonstrate|demonstrates|prove|proves|indicate|indicates|show|shows|establish|establishes|support|supports|suggest|suggests)\s+(?:the\s+)?(?:commercial|market|target-audience)?\s*(?:viability|success|demand|interest|appeal|potential|feasibility)\s+(?:of|for)\s+(?:the\s+)?(?:proposed\s+)?(?:project|documentary|film|premise)\b":
            "is historical/contextual evidence, but project-specific viability remains unverified",

        # 2. Historical demand multiples demonstrating notable/strong demand for the project
        r"\b(?:demand\s+multiples|historical\s+demand|historical\s+success|measured\s+demand)\s+(?:demonstrate|demonstrates|show|shows|prove|proves)\s+(?:that\s+)?(?:[\w\s\-/]+?)\s+(?:can\s+)?(?:achieve|have|demonstrate|attain)\s+(?:notable|strong|high|significant)\s+(?:demand|success|viability|multiples|demand\s+multiples)\b":
            "demonstrate historical metrics for those specific examples, which is historical/contextual evidence only; project-specific viability remains unverified",

        # 3. Market demand viability or similar labels used as decisive project viability proof
        r"\bmarket\s+demand\s+viability\b":
            "historical/contextual demand evidence (project-specific viability remains unverified)",

        # 4. Transferring demand from other/existing works to this project
        r"\b(?:transferable\s+demand|demand\s+transferability|likely\s+success|project's\s+viability|project's\s+demand)\s+(?:is\s+)?(?:demonstrated|proven|supported|indicated)\s+by\s+(?:historical|other|existing)\b":
            "remains unverified as historical evidence does not automatically transfer to the proposed project",

        # 5. General upgrades of historical context to positive viability
        r"\b(?:historical\s+observations|historical\s+evidence|evidence\s+concerning\s+other\s+works)\s+(?:proves|demonstrates|establishes|shows)\s+(?:the\s+)?(?:proposed\s+)?(?:project|documentary|film|premise)'s\s+(?:market|commercial)?\s*(?:viability|success|demand)\b":
            "is historical/contextual evidence only; the proposed project's market viability remains unverified",
    }

    sorted_keys = sorted(mappings.keys(), key=len, reverse=True)
    placeholders = {}
    for i, pattern in enumerate(sorted_keys):
        placeholder = f"___EV_STR_PLACEHOLDER_{i}___"
        text, count = re.subn(pattern, placeholder, text, flags=re.IGNORECASE)
        if count > 0:
            placeholders[placeholder] = mappings[pattern]

    for placeholder, final_val in placeholders.items():
        text = text.replace(placeholder, final_val)

    return text


def is_relationship_supported(relationship_type: str, cited_ids: list[str], ctx) -> bool:
    if not ctx or not cited_ids:
        return False
    research_text = get_research_text(ctx)
    if not research_text:
        return False
    ev_map = get_evidence_excerpts_map(research_text)
    ev_claims_map = get_evidence_claims_map(research_text)

    words_to_check = []
    if relationship_type == "dependency":
        words_to_check = ["dependent", "depends", "dependency", "contractually tied", "tied", "dictated", "influence", "influences", "impact", "impacts", "shape", "shapes", "affect", "affects"]
    elif relationship_type == "alignment":
        words_to_check = ["align", "alignment", "aligned", "must align", "coordinate"]
    elif relationship_type == "independence":
        words_to_check = ["independent", "independently", "independence", "unrelated", "decoupled"]

    for cid in cited_ids:
        excerpts = ev_map.get(cid, [])
        for exc in excerpts:
            exc_lower = exc.lower()
            if any(re.search(rf"\b{re.escape(w)}\b", exc_lower) for w in words_to_check):
                return True
        claims = ev_claims_map.get(cid, [])
        for clm in claims:
            clm_lower = clm.lower()
            if any(re.search(rf"\b{re.escape(w)}\b", clm_lower) for w in words_to_check):
                return True
    return False


def make_schedule_conditional(text: str, ctx=None) -> str:
    """Enforces conditional treatment of external schedules to prevent unverified internal project dependencies."""
    # 1. Backward-compatible classic mappings
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
        r"\balign\s+(?:the\s+)?internal\s+schedule\b": "determine whether/how it affects the internal schedule",
    }

    # 2. Advanced conditionalization mappings for schedule dependency creation
    internal_sched = r"(?:internal|documentary's|project's|film's|production|post-production|filming|delivery|release|marketing|festival|distribution|project|delivery's|documentary|film|proposed)(?:,\s*(?:internal|documentary's|project's|film's|production|post-production|filming|delivery|release|marketing|festival|distribution|project|documentary|film|proposed)|,\s*and\s+(?:internal|documentary's|project's|film's|production|post-production|filming|delivery|release|marketing|festival|distribution|project|documentary|film|proposed)|\s+and\s+(?:internal|documentary's|project's|film's|production|post-production|filming|delivery|release|marketing|festival|distribution|project|documentary|film|proposed)|\s+(?:internal|documentary's|project's|film's|production|post-production|filming|delivery|release|marketing|festival|distribution|project|documentary|film|proposed))*\s+(?:schedule|timeline|planning|plan|schedules|timelines|window|windows|date|dates|activities|activity|focus|risk|risks)"
    external_timing = r"(?:external|launch|conflicting|subject's|third-party|industry|subject|company's|campaign|timing|timeline)\s+(?:[\w\-]+\s+)?(?:date|dates|schedule|timeline|timing|event|events|uncertainty|uncertainties|launch\s+date|launch\s+schedule|launch\s+uncertainty|campaign\s+schedule|campaign\s+timeline|campaign|adjustments?|delays?|changes?|slips?|movements?|history|history\s+of\s+timing\s+adjustments|historical\s+schedule\s+changes|timing\s+adjustments)"

    advanced_mappings = {
        # Pattern: formulate/make/define internal schedule conditionally or independently of external timing
        rf"\b(?:formulate|schedule|define|make|align|tie|base|adjust|structure|organize)\s+(?:the\s+)?(?:any\s+)?(?:proposed\s+)?(?:({internal_sched})\s+)?(?:conditionally\s+or\s+independently\s+of|independently\s+of|independent\s+of|independent\s+from|conditionally\s+on|dependent\s+on)\s+(?:the\s+)?({external_timing})\b":
            r"define the \1 without presupposing any dependency or independence relationship to the \2, and separately determine whether such a relationship is intended or required",

        # Pattern: external timing introduces timing uncertainty for internal schedule
        rf"\b({external_timing})\s+(?:introduces|creates|causes|leads\s+to)\s+(?:timing\s+)?uncertainty\s+(?:for|in)\s+(?:the\s+)?(?:any\s+)?(?:proposed\s+)?({internal_sched})\b":
            r"\1 is an external event; determine whether/how it ___TEMP_AFFECTS___ the \2",

        # Pattern: internal schedule depends on/is dictated by external timing
        rf"\b(?:the\s+)?(?:any\s+)?(?:proposed\s+)?({internal_sched})\s+(?:depends\s+on|depend\s+on|is\s+dictated\s+by|are\s+dictated\s+by|is\s+impacted\s+by|are\s+impacted\s+by|is\s+governed\s+by|are\s+governed\s+by|creates\s+a\s+dependency\s+on|has\s+a\s+dependency\s+on)\s+(?:the\s+)?({external_timing})\b":
            r"whether the \1 depends on the \2 remains unverified; verify the external schedule and determine whether/how it ___TEMP_AFFECTS___ the \1",

        # Pattern: external timing impacts/dictates/determines/governs internal schedule
        rf"\b({external_timing})\s+(?:[\w\-]+\s+)?(?:impacts|impact|dictates|dictate|determines|determine|governs|govern|shapes|shape|affects|affect|influences|influence)\s+(?:the\s+)?(?:any\s+)?(?:proposed\s+)?({internal_sched})\b":
            r"\1 is an external event; determine whether/how it ___TEMP_AFFECTS___ the \2",

        # Pattern: build/structure/plan internal schedule around external timing/uncertainty
        rf"\b(?:the\s+)?(?:build|structure|plan|schedule|organize)\s+(?:the\s+)?(?:any\s+)?(?:proposed\s+)?({internal_sched})\s+(?:around|based\s+on)\s+(?:the\s+)?({external_timing})\b":
            r"determine whether/how the \2 ___TEMP_AFFECTS___ the \1 before final planning",

        # Pattern: internal schedule must/should/needs to align with external timing
        rf"\b(?:the\s+)?(?:any\s+)?(?:proposed\s+)?({internal_sched})\s+(?:must|should|needs\s+to|need\s+to|would\s+need|would\s+need\s+to|is\s+required\s+to)\s+(?:align\s+with|align|be\s+aligned\s+with|be\s+aligned\s+to|coordinate\s+with|be\s+coordinated\s+with)\s+(?:the\s+)?({external_timing})\b":
            r"determine whether/how the \2 ___TEMP_AFFECTS___ the \1 before deciding if alignment is required",

        # Pattern: external timing adjustments must be accounted for in internal schedule
        rf"\b({external_timing})\s+(?:that\s+)?(?:must|should|needs?\s+to|is\s+required\s+to|has\s+to)\s+be\s+accounted\s+for\s+(?:in|within)\s+(?:the\s+)?(?:any\s+)?(?:proposed\s+)?({internal_sched})\b":
            r"\1; determine whether/how those timing adjustments ___TEMP_AFFECTS___ the \2",

        # Pattern: internal schedule must account for external timing adjustments
        rf"\b(?:the\s+)?(?:any\s+)?(?:proposed\s+)?({internal_sched})\s+(?:must|should|needs?\s+to|is\s+required\s+to|has\s+to)\s+account\s+for\s+(?:the\s+)?({external_timing})\b":
            r"determine whether/how the \1 needs to account for the \2 before final planning",

        # Pattern: external timing adjustments requires/forces internal schedule to shift/move
        rf"\b({external_timing})\s+(?:requires|compels|dictates|forces|demands)\s+(?:the\s+)?(?:any\s+)?(?:proposed\s+)?({internal_sched})\s+to\s+(?:move|change|shift|align|slip|be\s+delayed|be\s+adjusted)\b":
            r"whether the \1 requires the \2 to shift remains unverified; determine whether/how it ___TEMP_AFFECTS___ the \2",
    }

    sorted_keys = sorted(advanced_mappings.keys(), key=len, reverse=True)

    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        if not line.strip():
            processed_lines.append(line)
            continue

        split_res = split_structural_line(line)
        if split_res:
            label_part, body_part = split_res
        else:
            label_part = ""
            body_part = line

        sentence_end = re.compile(r'([.!?]\s+)')
        parts = sentence_end.split(body_part)
        sentences = []
        i = 0
        while i < len(parts):
            s = parts[i]
            if i + 1 < len(parts):
                s += parts[i+1]
                i += 2
            else:
                i += 1
            if s:
                sentences.append(s)

        processed_sentences = []
        for sentence in sentences:
            s_lower = sentence.lower()

            # Skip if sentence is already neutralized/conditionalized to prevent double neutralization
            if "without presupposing" in s_lower or s_lower.strip().startswith("whether") or "remains unverified" in s_lower or "remains unknown" in s_lower or "remains to be verified" in s_lower:
                if s_lower.strip().startswith("whether") or "without presupposing" in s_lower:
                    processed_sentences.append(sentence)
                    continue
                # Do not skip if it contains a relationship assertion that needs validation
                has_alignment_word = any(re.search(rf"\b{re.escape(w)}\b", s_lower) for w in ["align", "alignment", "aligning", "aligned", "coordinate", "coordinating", "coordination"])
                has_independence_word = any(re.search(rf"\b{re.escape(w)}\b", s_lower) for w in ["independent", "independently", "independence", "unrelated", "decoupled"])
                has_dependency_word = any(re.search(rf"\b{re.escape(w)}\b", s_lower) for w in ["tie", "tying", "tied", "depend", "depends", "dependency", "dependent", "dictate", "dictates", "govern", "governs", "shape", "shapes", "affect", "affects", "influence", "influences"])
                if not (has_alignment_word or has_independence_word or has_dependency_word):
                    processed_sentences.append(sentence)
                    continue

            # Dynamic schedule relationship neutralization
            has_alignment_word = any(re.search(rf"\b{re.escape(w)}\b", s_lower) for w in ["align", "alignment", "aligning", "aligned", "coordinate", "coordinating", "coordination"])
            has_independence_word = any(re.search(rf"\b{re.escape(w)}\b", s_lower) for w in ["independent", "independently", "independence", "unrelated", "decoupled"])
            has_dependency_word = any(re.search(rf"\b{re.escape(w)}\b", s_lower) for w in ["tie", "tying", "tied", "depend", "depends", "dependency", "dependent", "dictate", "dictates", "govern", "governs", "shape", "shapes", "affect", "affects", "influence", "influences"])

            coupling_terms = [
                "couple", "coupling", "decouple", "decoupling",
                "sequence", "sequencing", "synchronize", "synchronizing", "synchronization",
                "accommodate", "accommodates", "accommodating",
                "conditional", "conditionalize", "flexible", "flexibility",
                "schedule around", "time around", "scheduled around", "timed around"
            ]
            has_coupling_word = any(re.search(rf"\b{re.escape(w)}\b", s_lower) for w in coupling_terms)
            has_qual_dep = any(p in s_lower for p in ["potential dependency", "possible dependency", "may depend", "could depend"])

            has_relation_action = has_alignment_word or has_independence_word or has_dependency_word or has_coupling_word or has_qual_dep

            if ctx is not None and has_relation_action:
                has_schedule_terms = any(re.search(rf"\b{re.escape(w)}\b", s_lower) for w in ["schedule", "timeline", "timelines", "schedules", "delivery", "release", "production", "post-production", "editorial", "documentary", "project", "film", "planning"])
                has_external_terms = any(re.search(rf"\b{re.escape(w)}\b", s_lower) for w in ["external", "launch", "milestone", "milestones", "event", "events", "q1", "2026", "2027", "timeline", "timelines", "adjustment", "adjustments"])

                if has_schedule_terms and has_external_terms:
                    cited_ids = parse_cited_evidence_ids(sentence)

                    is_supported = True
                    if has_alignment_word and not is_relationship_supported("alignment", cited_ids, ctx):
                        is_supported = False
                    if has_independence_word and not is_relationship_supported("independence", cited_ids, ctx):
                        is_supported = False
                    if (has_dependency_word or has_coupling_word or has_qual_dep) and not is_relationship_supported("dependency", cited_ids, ctx):
                        is_supported = False

                    if not is_supported:
                        bullet_match = re.match(r'^(\s*(?:-\s+|\*\s+|\d+\.\s+))', sentence)
                        bullet_prefix = bullet_match.group(1) if bullet_match else ""

                        trailing_ws = ""
                        m = re.search(r'(\s+)$', sentence)
                        if m:
                            trailing_ws = m.group(1)

                        is_action_sentence = classify_sentence_role(sentence) == "action" or "action" in s_lower or "should" in s_lower or "must" in s_lower or any(s_lower.strip().startswith(v) for v in ["establish", "define", "formulate", "align", "schedule", "create", "structure", "make", "organize", "plan", "coordinate", "build", "tie", "base", "adjust"])

                        if is_action_sentence:
                            sentence = f"{bullet_prefix}Define the project's internal production schedule, budget, and funding without presupposing a dependency, alignment, or independence relationship to the external event, and separately determine whether any such relationship is intended or required.{trailing_ws}"
                        else:
                            sentence = f"{bullet_prefix}The relationship between the internal schedule and the external schedule remains unverified and unknown.{trailing_ws}"
                        processed_sentences.append(sentence)
                        continue
                    else:
                        processed_sentences.append(sentence)
                        continue

            # Apply classic schedule mappings
            for pattern, replacement in schedule_mappings.items():
                sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)

            # Apply advanced mappings
            for pattern in sorted_keys:
                replacement_template = advanced_mappings[pattern]

                def sub_fn(match, template=replacement_template):
                    result = template
                    val1 = match.group(1) or ""
                    if not val1.strip():
                        result = result.replace("the \\1", "the internal schedule")
                    for g_num in range(1, len(match.groups()) + 1):
                        val = match.group(g_num) or ""
                        result = result.replace(f"\\{g_num}", val)
                    return result

                sentence = re.sub(pattern, sub_fn, sentence, flags=re.IGNORECASE)

            sentence = sentence.replace("___TEMP_AFFECTS___", "affects")
            s_lower = sentence.lower()

            # Epistemic neutralization logic (M7A.14): absence of evidence != independence/dependence
            has_independence = "independent" in s_lower or "independence" in s_lower
            has_unknown = any(x in s_lower for x in ["unknown", "unverified", "no evidence", "absence of evidence", "no public evidence", "not establish", "not link"])

            if has_independence and has_unknown:
                if any(x in s_lower for x in ["schedule", "timeline", "timing"]):
                    if "no public evidence establishing a relationship, dependency, or independence" not in s_lower:
                        sentence = "There is no public evidence establishing a relationship, dependency, or independence between the external schedule and the internal project timeline. The relationship remains unknown."
                elif any(x in s_lower for x in ["audience", "demand", "market", "viability"]):
                    if "no public evidence establishing a relationship, dependency, or independence" not in s_lower:
                        sentence = "There is no public evidence establishing a relationship, dependency, or independence between the external market factors and the internal project viability. The relationship remains unknown."
                elif any(x in s_lower for x in ["access", "permission", "coordination", "facility"]):
                    if "no public evidence establishing a relationship, dependency, or independence" not in s_lower:
                        sentence = "There is no public evidence establishing a relationship, dependency, or independence between the external entity and the internal project access. The relationship remains unknown."
                else:
                    if "no public evidence establishing a relationship, dependency, or independence" not in s_lower:
                        sentence = "There is no public evidence establishing a relationship, dependency, or independence between the external variables and the internal project variables. The relationship remains unknown."

            processed_sentences.append(sentence)

        processed_lines.append(label_part + "".join(processed_sentences))

    return "\n".join(processed_lines)


def is_clause_grammatically_complete(clause_text: str, sentence: str) -> bool:
    role = classify_sentence_role(sentence)
    clause_clean = clause_text.strip().strip(",;").strip()
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", clause_clean)]
    if not words:
        return False

    first_word = words[0]

    # 1. Check disallowed leading words for any clause (including coordinating/subordinating conjunctions and leading prepositions/adverbs)
    disallowed_starters = {
        "currently", "subsequently", "while", "because", "due", "moving", "to",
        "and", "but", "or", "since", "although", "if", "for", "with", "at",
        "from", "by", "under", "above", "below", "consequently"
    }
    if first_word in disallowed_starters:
        return False

    # 2. Check role-specific constraints
    if role == "action":
        action_verbs = {
            "define", "verify", "determine", "evaluate", "assess", "confirm", "investigate",
            "establish", "formulate", "schedule", "plan", "obtain", "coordinate", "align",
            "track", "clarify", "ensure", "analyze", "identify", "explore", "mitigate",
            "address", "check", "review", "compare"
        }
        if first_word not in action_verbs:
            return False
    else:
        # Factual, uncertainty, analytical_assumption, etc.
        # Must contain at least one valid finite verb
        factual_verbs = {
            "is", "are", "was", "were", "has", "have", "had", "been",
            "remains", "remain", "represents", "represent", "establishes", "establish",
            "contains", "contain", "prohibits", "prohibit", "restricts", "restrict",
            "begins", "begin", "aims", "aim", "delays", "delayed", "launches", "launched",
            "completed", "completes", "dictates", "dictate", "impacts", "impact",
            "shapes", "shape", "affects", "affect", "exists", "exist", "satisfies", "satisfy",
            "permits", "permit", "allows", "allow", "imposes", "impose"
        }
        has_verb = any(w in factual_verbs for w in words)
        if not has_verb:
            return False

    return True


def is_uncertainty_or_cited_sentence(s: str) -> bool:
    s_lower = s.lower()
    # Check for citation
    if re.search(r'\[based on e\d+|\[e\d+', s_lower):
        return True
    # Check for uncertainty or action keywords
    uncertainty_keywords = {
        "remains unknown", "unverified", "unresolved", "unspecified", "remains to be verified",
        "whether", "if", "determine", "verify", "evaluate", "assess", "confirm", "investigate",
        "unsupported", "uncertainty", "unspecified", "remains unresolved"
    }
    if any(k in s_lower for k in uncertainty_keywords):
        return True
    return False


def fail_closed_on_unsupported_sentences(text: str) -> str:
    """Splits text into sentences. Any sentence containing '[UNSUPPORTED]' is completely failed closed."""
    if "[UNSUPPORTED]" not in text:
        # Even if there's no [UNSUPPORTED], we still want to filter out literal model placeholders!
        lines = text.split("\n")
        processed_lines = []
        for line in lines:
            line_stripped = re.sub(r'^(\s*(?:-\s+|\*\s+|\d+\.\s+))', '', line).strip()
            if re.match(r'^\[Factual proposition unverified.*?\]$', line_stripped, re.IGNORECASE):
                continue
            processed_lines.append(line)
        return "\n".join(processed_lines)

    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        if not line.strip():
            processed_lines.append(line)
            continue

        # Drop literal placeholders immediately
        line_stripped = re.sub(r'^(\s*(?:-\s+|\*\s+|\d+\.\s+))', '', line).strip()
        if re.match(r'^\[Factual proposition unverified.*?\]$', line_stripped, re.IGNORECASE):
            continue

        # Split by sentence boundaries, preserving separators
        sentence_end = re.compile(r'([.!?]\s+)')
        parts = sentence_end.split(line)
        sentences = []
        i = 0
        while i < len(parts):
            s = parts[i]
            if i + 1 < len(parts):
                s += parts[i+1]
                i += 2
            else:
                i += 1
            if s:
                sentences.append(s)

        processed_sentences = []
        for sentence in sentences:
            if "[UNSUPPORTED]" in sentence:
                # Capture list markers or indentation to preserve layout structure
                bullet_match = re.match(r'^(\s*(?:-\s+|\*\s+|\d+\.\s+))', sentence)
                bullet_prefix = bullet_match.group(1) if bullet_match else ""

                # Check for trailing whitespace/newlines
                trailing_ws = ""
                m = re.search(r'(\s+)$', sentence)
                if m:
                    trailing_ws = m.group(1)

                preserved = False

                # M7A.16 Clause-Level Preservation
                # Split sentence by common conjunctions or punctuation separating factual preface and independent uncertainty
                conjunctions = [", but ", ", however, ", "; however, ", ", and ", "; "]
                for conj in conjunctions:
                    if conj in sentence:
                        parts_clause = sentence.split(conj, 1)
                        if len(parts_clause) == 2:
                            left, right = parts_clause
                            # The left clause contains [UNSUPPORTED], but the right clause has ZERO unsupported words!
                            if "[UNSUPPORTED]" in left and "[UNSUPPORTED]" not in right:
                                right_stripped = right.strip()
                                # Verify the right clause is grammatically complete
                                is_valid_right = is_clause_grammatically_complete(right_stripped, sentence)

                                if is_valid_right and len(right_stripped.split()) >= 3:
                                    valid_clause = right_stripped[0].upper() + right_stripped[1:]
                                    if not valid_clause.endswith((".", "!", "?")):
                                        valid_clause += "."
                                    preserved_sentence = f"{bullet_prefix}{valid_clause}{trailing_ws}"
                                    _trace_log(f"[Stage 11] Clause-level preservation: Replaced compound sentence with valid clause. Before: '{sentence.strip()}' -> After: '{preserved_sentence.strip()}'")
                                    processed_sentences.append(preserved_sentence)
                                    preserved = True
                                    break

                if not preserved:
                    neutral_marker = f"{bullet_prefix}Evidence is insufficient to verify this factual proposition.{trailing_ws}"
                    _trace_log(f"[Stage 11] Sentence-level fail-closed replacement: Sentence containing '[UNSUPPORTED]' replaced. Before: '{sentence.strip()}' -> After: '{neutral_marker.strip()}'")
                    processed_sentences.append(neutral_marker)
            else:
                processed_sentences.append(sentence)

        # Drop redundant generic failure messages if there's at least one valid, preserved sentence in the same line
        # that already communicates uncertainty or is a fully supported cited clause.
        has_valid_preserved = any(
            re.sub(r'^(?:-\s*|\*\s*|\d+\.\s*)', '', s).strip() and
            "[UNSUPPORTED]" not in s and
            "Evidence is insufficient to verify" not in s and
            is_uncertainty_or_cited_sentence(s)
            for s in processed_sentences
        )

        if has_valid_preserved:
            bullet_to_preserve = ""
            new_processed = []
            for idx, s in enumerate(processed_sentences):
                s_stripped_of_bullet = re.sub(r'^(\s*(?:-\s+|\*\s+|\d+\.\s+))', '', s).strip()
                if s_stripped_of_bullet == "Evidence is insufficient to verify this factual proposition.":
                    if not bullet_to_preserve:
                        bullet_match = re.match(r'^(\s*(?:-\s+|\*\s+|\d+\.\s+))', s)
                        if bullet_match:
                            bullet_to_preserve = bullet_match.group(1)
                else:
                    new_processed.append(s)

            if bullet_to_preserve and new_processed:
                first_s = new_processed[0]
                first_s_clean = re.sub(r'^(\s*(?:-\s+|\*\s+|\d+\.\s+))', '', first_s)
                new_processed[0] = bullet_to_preserve + first_s_clean

            processed_sentences = new_processed

        processed_lines.append("".join(processed_sentences))

    # Finally, remove lines that became purely the generic fail-closed placeholder if they originated as literal raw placeholders
    final_lines = []
    for line in processed_lines:
        if not line.strip():
            final_lines.append(line)
            continue
        final_lines.append(line)

    return "\n".join(final_lines)


def market_after_model_callback(callback_context, llm_response: LlmResponse) -> LlmResponse | None:
    _trace_state.role = "market_agent"
    _trace_log("=== START CALLBACK ===")
    try:
        ctx = callback_context.get_invocation_context()
        if not llm_response.content or not llm_response.content.parts:
            _trace_log("No content/parts in LLM response.")
            return None

        allowed_words = get_allowed_words(ctx)
        _trace_log(f"Global allowed words size: {len(allowed_words)}")
        modified = False

        for part in llm_response.content.parts:
            if part.text:
                orig = part.text
                _trace_raw_callback("market_agent", orig)
                _trace_log(f"[Stage 1] Raw downstream model output:\n{orig}\n")

                text = clean_and_validate_hidden_facts(orig, allowed_words, ctx=ctx)
                _trace_log(f"After clean_and_validate_hidden_facts:\n{text}\n")

                before_aud = text
                text = neutralize_audience_assumptions(text)
                if text != before_aud:
                    _trace_log(f"[Stage: Audience Neutralization] Modified text.\nBefore:\n{before_aud}\nAfter:\n{text}\n")

                before_pos = text
                text = neutralize_positive_assumptions(text)
                if text != before_pos:
                    _trace_log(f"[Stage: Positive Assumptions Neutralization] Modified text.\nBefore:\n{before_pos}\nAfter:\n{text}\n")

                before_eval = text
                text = neutralize_evaluative_words(text, allowed_words)
                if text != before_eval:
                    _trace_log(f"[Stage: Evaluative Words Neutralization] Modified text.\nBefore:\n{before_eval}\nAfter:\n{text}\n")

                before_ev_str = text
                text = neutralize_evidence_strength_upgrades(text)
                if text != before_ev_str:
                    _trace_log(f"[Stage: Evidence Strength Upgrades Neutralization] Modified text.\nBefore:\n{before_ev_str}\nAfter:\n{text}\n")

                before_sched = text
                text = make_schedule_conditional(text, ctx=ctx)
                if text != before_sched:
                    _trace_log(f"[Stage 12] Schedule semantic guard:\nBefore:\n{before_sched}\nAfter:\n{text}\n")

                before_fail = text
                text = fail_closed_on_unsupported_sentences(text)
                if text != before_fail:
                    _trace_log(f"[Stage 11] Sentence-level fail-closed replacement:\nBefore:\n{before_fail}\nAfter:\n{text}\n")

                _trace_log(f"[Stage 13] Final callback output:\n{text}\n")
                if text != orig:
                    part.text = text
                    modified = True

        return llm_response if modified else None
    finally:
        _trace_log("=== END CALLBACK ===")
        _trace_state.role = "unknown"


def production_risk_after_model_callback(
    callback_context, llm_response: LlmResponse
) -> LlmResponse | None:
    _trace_state.role = "production_risk_agent"
    _trace_log("=== START CALLBACK ===")
    try:
        ctx = callback_context.get_invocation_context()
        if not llm_response.content or not llm_response.content.parts:
            _trace_log("No content/parts in LLM response.")
            return None

        allowed_words = get_allowed_words(ctx)
        _trace_log(f"Global allowed words size: {len(allowed_words)}")
        modified = False

        for part in llm_response.content.parts:
            if part.text:
                orig = part.text
                _trace_raw_callback("production_risk_agent", orig)
                _trace_log(f"[Stage 1] Raw downstream model output:\n{orig}\n")

                text = clean_and_validate_hidden_facts(orig, allowed_words, ctx=ctx)
                _trace_log(f"After clean_and_validate_hidden_facts:\n{text}\n")

                before_prod = text
                text = neutralize_production_assumptions(text)
                if text != before_prod:
                    _trace_log(f"[Stage: Production Neutralization] Modified text.\nBefore:\n{before_prod}\nAfter:\n{text}\n")

                before_pos = text
                text = neutralize_positive_assumptions(text)
                if text != before_pos:
                    _trace_log(f"[Stage: Positive Assumptions Neutralization] Modified text.\nBefore:\n{before_pos}\nAfter:\n{text}\n")

                before_eval = text
                text = neutralize_evaluative_words(text, allowed_words)
                if text != before_eval:
                    _trace_log(f"[Stage: Evaluative Words Neutralization] Modified text.\nBefore:\n{before_eval}\nAfter:\n{text}\n")

                before_ev_str = text
                text = neutralize_evidence_strength_upgrades(text)
                if text != before_ev_str:
                    _trace_log(f"[Stage: Evidence Strength Upgrades Neutralization] Modified text.\nBefore:\n{before_ev_str}\nAfter:\n{text}\n")

                before_sched = text
                text = make_schedule_conditional(text, ctx=ctx)
                if text != before_sched:
                    _trace_log(f"[Stage 12] Schedule semantic guard:\nBefore:\n{before_sched}\nAfter:\n{text}\n")

                before_fail = text
                text = fail_closed_on_unsupported_sentences(text)
                if text != before_fail:
                    _trace_log(f"[Stage 11] Sentence-level fail-closed replacement:\nBefore:\n{before_fail}\nAfter:\n{text}\n")

                _trace_log(f"[Stage 13] Final callback output:\n{text}\n")
                if text != orig:
                    part.text = text
                    modified = True

        return llm_response if modified else None
    finally:
        _trace_log("=== END CALLBACK ===")
        _trace_state.role = "unknown"


def verdict_after_model_callback(callback_context, llm_response: LlmResponse) -> LlmResponse | None:
    _trace_state.role = "verdict_agent"
    _trace_log("=== START CALLBACK ===")
    try:
        ctx = callback_context.get_invocation_context()
        if not llm_response.content or not llm_response.content.parts:
            _trace_log("No content/parts in LLM response.")
            return None

        allowed_words = get_allowed_words(ctx)
        _trace_log(f"Global allowed words size: {len(allowed_words)}")
        modified = False

        for part in llm_response.content.parts:
            if part.text:
                orig = part.text
                _trace_raw_callback("verdict_agent", orig)
                _trace_log(f"[Stage 1] Raw downstream model output:\n{orig}\n")

                text = clean_and_validate_hidden_facts(orig, allowed_words, ctx=ctx)
                _trace_log(f"After clean_and_validate_hidden_facts:\n{text}\n")

                before_pos = text
                text = neutralize_positive_assumptions(text)
                if text != before_pos:
                    _trace_log(f"[Stage: Positive Assumptions Neutralization] Modified text.\nBefore:\n{before_pos}\nAfter:\n{text}\n")

                before_eval = text
                text = neutralize_evaluative_words(text, allowed_words)
                if text != before_eval:
                    _trace_log(f"[Stage: Evaluative Words Neutralization] Modified text.\nBefore:\n{before_eval}\nAfter:\n{text}\n")

                before_ev_str = text
                text = neutralize_evidence_strength_upgrades(text)
                if text != before_ev_str:
                    _trace_log(f"[Stage: Evidence Strength Upgrades Neutralization] Modified text.\nBefore:\n{before_ev_str}\nAfter:\n{text}\n")

                before_sched = text
                text = make_schedule_conditional(text, ctx=ctx)
                if text != before_sched:
                    _trace_log(f"[Stage 12] Schedule semantic guard:\nBefore:\n{before_sched}\nAfter:\n{text}\n")

                before_fail = text
                text = fail_closed_on_unsupported_sentences(text)
                if text != before_fail:
                    _trace_log(f"[Stage 11] Sentence-level fail-closed replacement:\nBefore:\n{before_fail}\nAfter:\n{text}\n")

                _trace_log(f"[Stage 13] Final callback output:\n{text}\n")
                if text != orig:
                    part.text = text
                    modified = True

        return llm_response if modified else None
    finally:
        _trace_log("=== END CALLBACK ===")
        _trace_state.role = "unknown"


def verdict_before_model_callback(callback_context, llm_request) -> None:
    """Callback executed before model call to dynamically bind Verdict to active Research Evidence Ledger."""
    _trace_state.role = "verdict_agent"
    _trace_log("=== START BEFORE-MODEL CALLBACK ===")
    try:
        ctx = callback_context.get_invocation_context()
        research_text = get_research_text(ctx)

        valid_ids = []
        if research_text:
            ev_map = get_evidence_excerpts_map(research_text)
            if ev_map:
                valid_ids = sorted(
                    list(ev_map.keys()),
                    key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
                )

        if valid_ids:
            ids_str = ", ".join(f"E{re.search(r'\d+', x).group()}" for x in valid_ids)

            dynamic_contract = (
                f"\n\nDYNAMIC EVIDENCE LEDGER BINDING CONTRACT:\n"
                f"- The Evidence Ledger supplied in THIS execution is the sole factual citation namespace.\n"
                f"- The active evidence IDs are exactly: {ids_str}.\n"
                f"- Cite ONLY evidence IDs present in this active ledger. Never cite E# IDs that are absent.\n"
                f"- Evidence IDs from prior runs, examples, memory, or other executions are invalid.\n"
                f"- Every factual proposition must cite the specific CURRENT evidence item whose Claim or Supporting Excerpt supports it.\n"
                f"- Do not cite an ID merely because it existed in another execution.\n"
                f"- If no current evidence item supports a factual proposition, express it as uncertainty/missing evidence or omit the proposition.\n"
                f"- Never translate or remap a remembered evidence ID by ordinal position."
            )
        else:
            dynamic_contract = (
                f"\n\nDYNAMIC EVIDENCE LEDGER BINDING CONTRACT:\n"
                f"- There is NO active Research Evidence Ledger or no valid evidence IDs present for the current execution.\n"
                f"- You are STRICTLY prohibited from making any cited factual assertions or citing any E# IDs because no active evidence exists."
            )

        if not llm_request.config.system_instruction:
            llm_request.config.system_instruction = dynamic_contract
        elif isinstance(llm_request.config.system_instruction, str):
            llm_request.config.system_instruction += dynamic_contract
        else:
            _trace_log(f"Unsupported system_instruction type: {type(llm_request.config.system_instruction)}")
    finally:
        _trace_log("=== END BEFORE-MODEL CALLBACK ===")
        _trace_state.role = "unknown"


def research_after_model_callback(callback_context, llm_response: LlmResponse) -> LlmResponse | None:
    _trace_state.role = "research_agent"
    _trace_log("=== START CALLBACK ===")
    try:
        ctx = callback_context.get_invocation_context()
        if not llm_response.content or not llm_response.content.parts:
            return None

        modified = False
        for part in llm_response.content.parts:
            if part.text:
                orig = part.text
                _trace_raw_callback("research_agent", orig)

                # Apply schedule semantic guard to neutralize schedule relationship presupposition
                text = make_schedule_conditional(orig, ctx=ctx)

                if text != orig:
                    part.text = text
                    modified = True

        return llm_response if modified else None
    finally:
        _trace_log("=== END CALLBACK ===")
        _trace_state.role = "unknown"
