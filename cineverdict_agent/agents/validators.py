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
    "DIRECTOR PLAN"
}


def split_structural_line(line: str) -> tuple[str, str] | None:
    """Detects and splits known structural labels, returning (label_prefix, body) or None."""
    labels_pattern = "|".join(re.escape(label) for label in KNOWN_LABELS)
    # Match structural headers like: "- **ANALYSIS**:" or "MISSING EVIDENCE:" or "E1 — Claim:"
    pattern = re.compile(
        r"^([ \t]*(?:-\s*|\*\s*|\d+\.\s*)?(?:\*\*|\[)?(?:E\d+\s*—\s*)?(?:" + labels_pattern + r")(?:\*\*|\])?\s*(?::|—|-)\s*)(.*)$",
        re.IGNORECASE
    )
    m = pattern.match(line)
    if m:
        return m.group(1), m.group(2)
        
    # Fallback to match standalone index headers e.g. "E1:" or "E1 —"
    index_pattern = re.compile(r"^([ \t]*(?:-\s*|\*\s*|\d+\.\s*)?(?:E\d+)(?:\*\*|\])?\s*(?::|—|-)\s*)(.*)$", re.IGNORECASE)
    m2 = index_pattern.match(line)
    if m2:
        return m2.group(1), m2.group(2)
        
    return None


def get_word_variations(word: str) -> set[str]:
    """Generates conservative, morphology-safe lowercase variations of a word.
    
    Ensures zero false matches for unrelated words (e.g., status/analysis/experimental).
    """
    w = word.lower().replace("’", "'")
    variations = {w}
    
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
    """Robustly extracts all text values following 'Supporting Excerpt:' in the research text."""
    excerpts = []
    # Split by Supporting Excerpt: (case insensitive)
    parts = re.split(r"Supporting Excerpt:", research_text, flags=re.IGNORECASE)
    # The first part is before the first Supporting Excerpt, so skip it
    for part in parts[1:]:
        val = part.strip()
        lines = val.split("\n")
        excerpt_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Stop if the line starts with an evidence block marker or a standard label
            if (
                re.match(r"^E\d+\s*—", line_stripped, re.IGNORECASE) or
                line_stripped.startswith("Claim:") or
                line_stripped.startswith("Verification Status:") or
                line_stripped.startswith("Source Title:") or
                line_stripped.startswith("Source URL:") or
                line_stripped.startswith("Publish Date:") or
                line_stripped.startswith("Supporting Excerpt:")
            ):
                break
            excerpt_lines.append(line)
        
        excerpt_text = "\n".join(excerpt_lines).strip()
        # Strip surrounding double quotes if present
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
            if "-" in w:
                for sub in w.split("-"):
                    if sub:
                        allowed.add(sub.lower())

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


def clean_and_validate_hidden_facts(text: str, allowed_words: set[str]) -> str:
    """Finds proper nouns and numbers that do not exist in the allowed words set and redacts them.
    
    Processes line-by-line:
    - Bypasses explicitly recognized CineVerdict structural labels.
    - Validates only body content following the label.
    - Uses conservative morphological variations for checking.
    """
    # Pre-expand allowed words to include all of their conservative variations
    expanded_allowed = set()
    for w in allowed_words:
        for var in get_word_variations(w):
            expanded_allowed.add(var)

    lines = text.split("\n")
    processed_lines = []
    
    for line in lines:
        if not line.strip():
            processed_lines.append(line)
            continue
            
        # Detect and split known structural label
        split_res = split_structural_line(line)
        if split_res:
            label_part, body_part = split_res
        else:
            label_part = ""
            body_part = line
            
        # Extract proper nouns (capitalized words) and numbers from the body part
        significant_words = set(re.findall(r"\b[A-Z][a-zA-Z0-9\-]*\b|\b\d+\b", body_part))
        unauthorized = []
        for w in significant_words:
            w_lower = w.lower()
            
            # Check conservative morphological variations of the word
            w_vars = get_word_variations(w_lower)
            is_authorized = (
                any(var in expanded_allowed for var in w_vars)
                or w_lower in COMMON_STOP_WORDS
                or w_lower in SYSTEM_ALLOWED_WORDS
                or any(var in SYSTEM_ALLOWED_WORDS for var in w_vars)
            )
            
            if not is_authorized and len(w) > 1:
                unauthorized.append(w)
                
        # Sort by length descending to replace larger words first (prevents substring issues)
        unauthorized.sort(key=len, reverse=True)
        validated_body = body_part
        for w in unauthorized:
            # Use word boundaries for precise replacement
            validated_body = re.sub(r"\b" + re.escape(w) + r"\b", "[UNSUPPORTED]", validated_body)
            
        processed_lines.append(label_part + validated_body)
        
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


def make_schedule_conditional(text: str) -> str:
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
    }
    for pattern, replacement in schedule_mappings.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 2. Advanced conditionalization mappings for schedule dependency creation
    internal_sched = r"(?:production|filming|delivery|release|marketing|festival|distribution|project|delivery's)\s+(?:schedule|timeline|planning|plan|schedules|timelines)"
    external_timing = r"(?:external|launch|conflicting|subject's|third-party|industry|subject|company's)\s+(?:date|dates|schedule|timeline|timing|event|events|uncertainty|uncertainties|launch\s+date|launch\s+schedule|launch\s+uncertainty)"

    advanced_mappings = {
        # Pattern: internal schedule depends on/is dictated by external timing
        rf"\b({internal_sched})\s+(?:depends\s+on|depend\s+on|is\s+dictated\s+by|are\s+dictated\s+by|is\s+impacted\s+by|are\s+impacted\s+by|is\s+governed\s+by|are\s+governed\s+by|creates\s+a\s+dependency\s+on|has\s+a\s+dependency\s+on)\s+(?:the\s+)?({external_timing})\b":
            r"whether the \1 depends on the \2 remains unverified; verify the external schedule and determine whether/how it affects the \1",

        # Pattern: external timing impacts/dictates/determines/governs internal schedule
        rf"\b({external_timing})\s+(?:impacts|impact|dictates|dictate|determines|determine|governs|govern|shapes|shape|affects|affect)\s+(?:the\s+)?({internal_sched})\b":
            r"\1 is an external event; determine whether/how it affects the \2",

        # Pattern: build/structure/plan internal schedule around external timing/uncertainty
        rf"\b(?:build|structure|plan|schedule|organize)\s+(?:the\s+)?({internal_sched})\s+(?:around|based\s+on)\s+(?:the\s+)?({external_timing})\b":
            r"determine whether/how the \2 affects the \1 before final planning",

        # Pattern: internal schedule must/should/needs to align with external timing
        rf"\b({internal_sched})\s+(?:must|should|needs\s+to|need\s+to)\s+align\s+with\s+(?:the\s+)?({external_timing})\b":
            r"determine whether/how the \2 affects the \1 before deciding if alignment is required",
    }

    sorted_keys = sorted(advanced_mappings.keys(), key=len, reverse=True)
    for pattern in sorted_keys:
        replacement_template = advanced_mappings[pattern]
        
        def sub_fn(match, template=replacement_template):
            result = template
            for g_num in range(1, len(match.groups()) + 1):
                val = match.group(g_num) or ""
                result = result.replace(f"\\{g_num}", val)
            return result

        text = re.sub(pattern, sub_fn, text, flags=re.IGNORECASE)
        
    return text


def fail_closed_on_unsupported_sentences(text: str) -> str:
    """Splits text into sentences. Any sentence containing '[UNSUPPORTED]' is completely failed closed."""
    if "[UNSUPPORTED]" not in text:
        return text
        
    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        if not line.strip():
            processed_lines.append(line)
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
                bullet_match = re.match(r'^(\s*(?:-\s*|\*\s*|\d+\.\s*))', sentence)
                bullet_prefix = bullet_match.group(1) if bullet_match else ""
                
                # Check for trailing whitespace/newlines
                trailing_ws = ""
                m = re.search(r'(\s+)$', sentence)
                if m:
                    trailing_ws = m.group(1)
                
                neutral_marker = f"{bullet_prefix}[Factual proposition unverified due to missing evidence.]{trailing_ws}"
                processed_sentences.append(neutral_marker)
            else:
                processed_sentences.append(sentence)
                
        processed_lines.append("".join(processed_sentences))
        
    return "\n".join(processed_lines)


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
            text = neutralize_evidence_strength_upgrades(text)
            text = make_schedule_conditional(text)
            text = fail_closed_on_unsupported_sentences(text)
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
            text = neutralize_evidence_strength_upgrades(text)
            text = make_schedule_conditional(text)
            text = fail_closed_on_unsupported_sentences(text)
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
            text = neutralize_evidence_strength_upgrades(text)
            text = make_schedule_conditional(text)
            text = fail_closed_on_unsupported_sentences(text)
            if text != orig:
                part.text = text
                modified = True

    return llm_response if modified else None
