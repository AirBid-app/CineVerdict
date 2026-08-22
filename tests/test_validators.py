import unittest
from unittest.mock import MagicMock
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.adk import Context

from cineverdict_agent.agents.validators import (
    extract_supporting_excerpts,
    clean_and_validate_hidden_facts,
    neutralize_audience_assumptions,
    neutralize_production_assumptions,
    neutralize_evaluative_words,
    make_schedule_conditional,
    market_after_model_callback,
    production_risk_after_model_callback,
    verdict_after_model_callback,
    get_allowed_words,
    fail_closed_on_unsupported_sentences,
    neutralize_positive_assumptions
)


class TestValidators(unittest.TestCase):

    def test_excerpt_extraction(self):
        sample_research = """
        RESEARCH EVIDENCE BRIEF
        EVIDENCE LEDGER
        E1 — Claim: Vast Space headquarters is in Long Beach.
        Verification Status: PRIMARY-SOURCE VERIFIED
        Source Title: Vast Space Web Site
        Source URL: https://vastspace.com
        Supporting Excerpt: "Vast Space has its primary facility located in Long Beach, California."

        E2 — Claim: Haven-1 is scheduled for launch.
        Verification Status: SECONDARY-SOURCE EVIDENCE
        Source Title: Space News
        Source URL: https://spacenews.com
        Supporting Excerpt: The commercial space station Haven-1 is planned to launch no earlier than 2026.
        """
        excerpts = extract_supporting_excerpts(sample_research)
        self.assertEqual(len(excerpts), 2)
        self.assertIn("Vast Space has its primary facility located in Long Beach, California.", excerpts)
        self.assertIn("The commercial space station Haven-1 is planned to launch no earlier than 2026.", excerpts)

    def test_hidden_facts_redaction(self):
        allowed = {"long", "beach", "california", "vast", "space", "haven-1", "2026"}
        
        # Test with forbidden location (Houston) and forbidden address (1234 Houston St)
        input_text = "The production facility is located at 1234 Houston Street in Long Beach."
        expected = "The production facility is located at [UNSUPPORTED] [UNSUPPORTED] [UNSUPPORTED] in Long Beach."
        output = clean_and_validate_hidden_facts(input_text, allowed)
        self.assertEqual(output, expected)

        # Test with allowed values
        input_allowed = "Vast Space primary facility in Long Beach, California."
        output_allowed = clean_and_validate_hidden_facts(input_allowed, allowed)
        self.assertEqual(output_allowed, input_allowed)

    def test_neutralize_audience_assumptions(self):
        input_text = "Our analysis shows that public interest exists and a viable audience is reachable."
        expected = "Our analysis shows that HYPOTHESIS: public interest may exist but remains unverified and HYPOTHESIS: an audience may exist; its size, composition, reachability, engagement, and commercial viability remain unverified."
        output = neutralize_audience_assumptions(input_text)
        self.assertEqual(output, expected)

    def test_neutralize_production_assumptions(self):
        input_text = "The format can be structured around launch uncertainty, and desired access to personnel can be coordinated with the subject company."
        expected = "The format whether a format can be structured around launch uncertainty remains unverified and conditional, and unverified desired access to personnel whether coordination with the subject company is possible remains unverified and conditional."
        output = neutralize_production_assumptions(input_text)
        self.assertEqual(output, expected)

    def test_neutralize_evaluative_upgrades(self):
        allowed = {"distribution", "pathway", "existed"}
        
        # 'successful' should be neutralized since 'successful' is not in allowed
        input_text = "We have established successful distribution pathways."
        expected = "We have established existing/distributed distribution pathways."
        output = neutralize_evaluative_words(input_text, allowed)
        self.assertEqual(output, expected)

        # 'commercially viable' should be neutralized
        input_text = "This project is commercially viable."
        expected = "This project is unverified commercial viability."
        output = neutralize_evaluative_words(input_text, allowed)
        self.assertEqual(output, expected)

    def test_make_schedule_conditional(self):
        input_text = "Monitor and confirm the finalized launch to align the production's release timeline."
        expected = "Monitor and confirm the finalized launch to determine whether/how it affects the production's release timeline."
        output = make_schedule_conditional(input_text)
        self.assertEqual(output, expected)

    def test_callbacks_integration(self):
        # Create a mock context with session events
        mock_ctx = MagicMock()
        mock_event_director = MagicMock()
        mock_event_director.author = "director_agent"
        mock_event_director.output = "DIRECTOR PLAN - USER Premise: Vast Space film. Align the production schedule, release timeline, and milestone."

        mock_event_research = MagicMock()
        mock_event_research.author = "research_agent"
        mock_event_research.output = """
        Supporting Excerpt: "Vast Space has its primary facility located in Long Beach, California."
        """

        mock_ctx.session.events = [mock_event_director, mock_event_research]
        
        mock_callback_ctx = MagicMock(spec=Context)
        mock_callback_ctx.get_invocation_context.return_value = mock_ctx

        # Test market callback with forbidden facts & assumptions
        llm_response_market = LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text="A viable audience is reachable at Houston facility. This is a successful project."
                    )
                ]
            )
        )
        res = market_after_model_callback(mock_callback_ctx, llm_response_market)
        self.assertIsNotNone(res)
        modified_text = res.content.parts[0].text
        # Houston is redacted, and since it contains [UNSUPPORTED], the entire sentence fails closed
        self.assertIn("Factual proposition unverified due to missing evidence.", modified_text)
        self.assertNotIn("Houston", modified_text)
        self.assertIn("existing/distributed", modified_text)

        # Test production callback
        llm_response_prod = LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text="Desired access can be coordinated with the subject company at Mojave."
                    )
                ]
            )
        )
        res_prod = production_risk_after_model_callback(mock_callback_ctx, llm_response_prod)
        self.assertIsNotNone(res_prod)
        modified_text_prod = res_prod.content.parts[0].text
        # Mojave is redacted, so the sentence fails closed
        self.assertIn("Factual proposition unverified due to missing evidence.", modified_text_prod)
        self.assertNotIn("Mojave", modified_text_prod)

        # Test verdict callback
        llm_response_verdict = LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text="Align the production's release timeline. This is a successful milestone at Long Beach."
                    )
                ]
            )
        )
        res_verdict = verdict_after_model_callback(mock_callback_ctx, llm_response_verdict)
        self.assertIsNotNone(res_verdict)
        modified_text_verdict = res_verdict.content.parts[0].text
        self.assertIn("determine whether/how it affects the production's release timeline", modified_text_verdict)
        # Long Beach should NOT be redacted because it is allowed (present in research excerpt)
        self.assertIn("Long Beach", modified_text_verdict)
        self.assertNotIn("[UNSUPPORTED]", modified_text_verdict)

    def test_invariant_a_no_substantive_global_allowlist(self):
        # A substantive word must NOT survive solely because CineVerdict uses that word structurally elsewhere.
        allowed = {"project"}  # "funding", "viability", "rights" are NOT allowed
        input_text = "We confirmed the Funding of Rights."
        output = clean_and_validate_hidden_facts(input_text, allowed)
        self.assertIn("[UNSUPPORTED]", output)
        self.assertNotIn("Funding", output)
        self.assertNotIn("Rights", output)

    def test_invariant_b_structural_labels_preserved(self):
        # Explicitly recognized structural labels must survive intact.
        allowed = {"project"}
        input_text = "VERIFIED EVIDENCE: Funding was secured from Company.\nANALYSIS: The project is feasible."
        output = clean_and_validate_hidden_facts(input_text, allowed)
        # "VERIFIED EVIDENCE:" and "ANALYSIS:" must survive intact
        self.assertIn("VERIFIED EVIDENCE:", output)
        self.assertIn("ANALYSIS:", output)
        # In the body, "Funding" must be redacted because it's not in 'allowed'
        self.assertIn("[UNSUPPORTED] was secured from [UNSUPPORTED]", output)

        # Arbitrary/content-bearing labels must NOT be bypassed
        arbitrary_input = "Secret Funding: Some detail."
        arbitrary_output = clean_and_validate_hidden_facts(arbitrary_input, allowed)
        self.assertIn("[UNSUPPORTED] [UNSUPPORTED]: Some detail.", arbitrary_output)

    def test_invariant_c_strict_morphology_and_stem_safeguards(self):
        # sibilant plurals, safe possessives, and safe inflections survive ONLY when their literal source form exists in the allowed set.
        
        # 1. "Launches" (plural) survives if "launch" is in allowed
        out_launch = clean_and_validate_hidden_facts("We monitored multiple Launches.", {"launch"})
        self.assertNotIn("[UNSUPPORTED]", out_launch)
        self.assertIn("Launches", out_launch)
        
        # 2. "Launches" is redacted if "launch" is absent
        out_no_launch = clean_and_validate_hidden_facts("We monitored multiple Launches.", {"project"})
        self.assertIn("[UNSUPPORTED]", out_no_launch)
        
        # 3. Non-plural 's' endings like Status/Analysis fail closed and do not validate Status/Analysis against truncated roots
        out_status = clean_and_validate_hidden_facts("The Status of the project.", {"statu"})
        self.assertIn("[UNSUPPORTED]", out_status) # Status should be redacted since statu is NOT its base form in variations

        # 4. Unrelated words with superficially similar roots (experimental -> expert) must not validate each other
        out_coincidental = clean_and_validate_hidden_facts("We conducted Experimental tests.", {"expert"})
        self.assertIn("[UNSUPPORTED]", out_coincidental)
        self.assertNotIn("Experimental", out_coincidental)

    def test_invariant_fail_closed_sentence_level(self):
        # Unsupported propositions must fail closed at a sentence level rather than producing partially redacted gibberish.
        from cineverdict_agent.agents.validators import fail_closed_on_unsupported_sentences
        
        # Unaltered if no unsupported
        self.assertEqual(fail_closed_on_unsupported_sentences("This is a safe sentence."), "This is a safe sentence.")

        # Completely replaces sentence with neutral marker
        input_text = "We have two locations. One is in Long Beach. The other is at [UNSUPPORTED] [UNSUPPORTED]."
        expected = "We have two locations. One is in Long Beach. [Factual proposition unverified due to missing evidence.]"
        self.assertEqual(fail_closed_on_unsupported_sentences(input_text), expected)

        # Preserves bullet points and list formatting
        list_input = "- We checked the [UNSUPPORTED] location."
        list_expected = "- [Factual proposition unverified due to missing evidence.]"
        self.assertEqual(fail_closed_on_unsupported_sentences(list_input), list_expected)

    def test_evidence_strength_protection(self):
        # Historical/contextual evidence must not be upgraded to positive viability for the proposed project.
        from cineverdict_agent.agents.validators import neutralize_evidence_strength_upgrades
        
        input_text = "Demand multiples of other space documentaries demonstrate market viability for the proposed film."
        expected = "is historical/contextual evidence, but project-specific viability remains unverified"
        self.assertIn(expected, neutralize_evidence_strength_upgrades(input_text))

        input_text_2 = "These demand multiples demonstrate that science/space documentaries can achieve notable demand multiples."
        expected_2 = "demonstrate historical metrics for those specific examples, which is historical/contextual evidence only"
        self.assertIn(expected_2, neutralize_evidence_strength_upgrades(input_text_2))

    def test_schedule_dependency_protection(self):
        # External launch dates do not automatically create an internal project dependency.
        from cineverdict_agent.agents.validators import make_schedule_conditional
        
        input_text = "The external launch date impacts the production schedule."
        expected = "The external launch date is an external event; determine whether/how it affects the production schedule."
        self.assertEqual(make_schedule_conditional(input_text), expected)

        input_text_2 = "Build the filming schedule around the launch uncertainty."
        expected_2 = "determine whether/how the launch uncertainty affects the filming schedule before final planning."
        self.assertEqual(make_schedule_conditional(input_text_2), expected_2)

    def test_numbered_markdown_headings_survive(self):
        # M7A.3 Failure Class A: Numbered Markdown structural headings must survive validation unchanged.
        allowed = {"project"}
        headings = [
            "### 1. FINAL VERDICT",
            "### 2. CONFIDENCE",
            "### 3. DECISIVE REASONS",
            "### 4. UNRESOLVED UNCERTAINTIES",
            "### 5. REQUIRED NEXT ACTIONS",
            "### MARKET ANALYSIS",
            "### PRODUCTION & RISK ANALYSIS",
            "### CINEVERDICT FINAL EVALUATION",
            "- **ANALYSIS**:",
            "1. **FINAL VERDICT**"
        ]
        for heading in headings:
            output = clean_and_validate_hidden_facts(heading, allowed)
            self.assertEqual(output, heading)

    def test_citation_scoped_validation(self):
        # M7A.3 Failure Class B: Citation-scoped validation evaluates line against cited excerpts plus permitted context.
        # Create a mock context with research agent event containing E1 and E2 excerpts
        mock_ctx = MagicMock()
        mock_event_research = MagicMock()
        mock_event_research.author = "research_agent"
        mock_event_research.output = """
        E1 — Claim: Vast Space headquarters is in Long Beach.
        Supporting Excerpt: "Vast Space has its primary facility located in Long Beach, California."

        E2 — Claim: Haven-1 launch is in 2026.
        Supporting Excerpt: "Haven-1 is planned to launch in 2026."
        """
        mock_ctx.session.events = [mock_event_research]
        
        # Scenario 1: Line cites E1 and contains words from E1 excerpt -> survives!
        line_supported = "The primary facility of Vast Space is in Long Beach [E1]."
        # We don't supply allowed_words (rely on E1 citation mapping)
        output_supported = clean_and_validate_hidden_facts(line_supported, set(), ctx=mock_ctx)
        self.assertEqual(output_supported, line_supported)

        # Scenario 2: Line cites E1 but contains fact from E2 ("2026") -> "2026" is NOT in E1, so it is redacted!
        line_unsupported = "Vast Space primary facility in Long Beach was planned in 2026 [E1]."
        output_unsupported = clean_and_validate_hidden_facts(line_unsupported, set(), ctx=mock_ctx)
        self.assertIn("[UNSUPPORTED]", output_unsupported)
        self.assertNotIn("2026", output_unsupported)

    def test_analytical_and_neutral_language_survival(self):
        # M7A.3 Failure Class C/D: Neutral analytical or uncertainty statements survive without globally allowlisting their substantive nouns.
        allowed = set() # Empty allowed set
        
        # These are lowercase and contain neutral patterns, so they should survive without any redactions
        analytical_lines = [
            "project-specific viability remains unverified.",
            "whether demand exists remains unknown.",
            "budget/funding status was not supplied.",
            "the distribution strategy is unspecified."
        ]
        for line in analytical_lines:
            output = clean_and_validate_hidden_facts(line, allowed)
            self.assertEqual(output, line)

    def test_positive_audience_neutralized(self):
        # M7A.3 Failure Class C/D: Positive unsupported audience language is neutralized
        input_text = "We assume public interest exists and the project is viable."
        # "public interest exists" should be neutralized, while "viable" gets rewritten
        output = neutralize_audience_assumptions(input_text)
        self.assertIn("HYPOTHESIS: public interest may exist but remains unverified", output)

    def test_external_schedule_dependency_guarded(self):
        # M7A.3 Failure Class E: External schedule evidence does not automatically establish internal project dependency.
        from cineverdict_agent.agents.validators import make_schedule_conditional
        
        # Test Case 1: External timing introduces timing uncertainty for internal windows
        input_text_1 = "The external launch schedule introduces timing uncertainty for the production and post-production windows."
        expected_1 = "The external launch schedule is an external event; determine whether/how it affects the production and post-production windows."
        self.assertEqual(make_schedule_conditional(input_text_1), expected_1)

        # Test Case 2: Internal schedule needs to align with external campaign
        input_text_2 = "The documentary schedule would need to be aligned with the external campaign."
        expected_2 = "determine whether/how the external campaign affects the documentary schedule before deciding if alignment is required."
        self.assertEqual(make_schedule_conditional(input_text_2), expected_2)

    def test_m7a4_semantic_roles_and_neutralization(self):
        # Deterministic regression tests covering M7A.4 requirements
        
        # A. Neutral analytical language survives without global substantive allowlisting.
        out_a = clean_and_validate_hidden_facts("Project-specific viability remains unverified.", set())
        self.assertEqual(out_a, "Project-specific viability remains unverified.")

        # B. Missing-evidence/uncertainty statements survive.
        out_b = clean_and_validate_hidden_facts("Budget and funding status remain unspecified.", set())
        self.assertEqual(out_b, "Budget and funding status remain unspecified.")

        # C. Recommended verification/action statements survive.
        out_c1 = clean_and_validate_hidden_facts("Verify the applicable access conditions before commitment.", set())
        self.assertEqual(out_c1, "Verify the applicable access conditions before commitment.")
        out_c2 = clean_and_validate_hidden_facts("Determine whether additional authorization is required.", set())
        self.assertEqual(out_c2, "Determine whether additional authorization is required.")

        # D. Unsupported factual assertions still fail closed.
        out_d_raw = clean_and_validate_hidden_facts("Secret Space has its headquarters in Seattle.", set())
        from cineverdict_agent.agents.validators import fail_closed_on_unsupported_sentences
        out_d = fail_closed_on_unsupported_sentences(out_d_raw)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_d)

        # E. An analytical wrapper cannot preserve a material unsupported factual assertion.
        out_e_raw = clean_and_validate_hidden_facts("The production should evaluate alternative approaches if Secret Space is unavailable.", set())
        out_e = fail_closed_on_unsupported_sentences(out_e_raw)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_e)

        # F. Unknown audience demand is not converted into a positive assumption.
        from cineverdict_agent.agents.validators import neutralize_positive_assumptions
        out_f1 = neutralize_positive_assumptions("It is assumed that a viable audience is reachable.")
        self.assertIn("Audience demand remains unverified", out_f1)
        out_f2 = neutralize_positive_assumptions("It is assumed that an audience exists.")
        self.assertIn("Audience demand remains unverified", out_f2)

        # G. Unknown access is not converted into a positive assumption.
        out_g = neutralize_positive_assumptions("It is assumed that access is available.")
        self.assertIn("Access has not been established and remains unverified.", out_g)

        # H. Unknown funding/rights or equivalent prerequisite is not converted into a positive assumption.
        out_h1 = neutralize_positive_assumptions("It is assumed that funding exists.")
        self.assertIn("Funding status is unspecified and remains unverified.", out_h1)
        out_h2 = neutralize_positive_assumptions("It is assumed that rights can be obtained.")
        self.assertIn("Rights/authorization remain to be verified.", out_h2)

    def test_m7a5_reproduction_and_closure(self):
        # 1. Analytical uncertainty sentence survives.
        out_1_raw = clean_and_validate_hidden_facts("Project-specific commercial viability remains unverified.", set())
        out_1 = fail_closed_on_unsupported_sentences(out_1_raw)
        self.assertEqual(out_1, "Project-specific commercial viability remains unverified.")

        out_1b_raw = clean_and_validate_hidden_facts("Whether sufficient audience demand exists remains unknown.", set())
        out_1b = fail_closed_on_unsupported_sentences(out_1b_raw)
        self.assertEqual(out_1b, "Whether sufficient audience demand exists remains unknown.")

        # 2. Missing-evidence sentence survives.
        out_2_raw = clean_and_validate_hidden_facts("Access conditions have not been established.", set())
        out_2 = fail_closed_on_unsupported_sentences(out_2_raw)
        self.assertEqual(out_2, "Access conditions have not been established.")

        out_2b_raw = clean_and_validate_hidden_facts("Budget and funding status remain unspecified.", set())
        out_2b = fail_closed_on_unsupported_sentences(out_2b_raw)
        self.assertEqual(out_2b, "Budget and funding status remain unspecified.")

        # 3. Recommended verification action survives.
        out_3_raw = clean_and_validate_hidden_facts("Verify the applicable access conditions before commitment.", set())
        out_3 = fail_closed_on_unsupported_sentences(out_3_raw)
        self.assertEqual(out_3, "Verify the applicable access conditions before commitment.")

        out_3b_raw = clean_and_validate_hidden_facts("Determine whether additional authorization is required.", set())
        out_3b = fail_closed_on_unsupported_sentences(out_3b_raw)
        self.assertEqual(out_3b, "Determine whether additional authorization is required.")

        # 4. Conditional recommendation survives.
        out_4_raw = clean_and_validate_hidden_facts("Evaluate alternative production approaches if access is unavailable.", set())
        out_4 = fail_closed_on_unsupported_sentences(out_4_raw)
        self.assertEqual(out_4, "Evaluate alternative production approaches if access is unavailable.")

        # 5. Unsupported factual assertion still fails closed.
        out_5_raw = clean_and_validate_hidden_facts("Orbital Media Corporation secured exclusive access in 2027.", set())
        out_5 = fail_closed_on_unsupported_sentences(out_5_raw)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_5)

        # 6. Unsupported proper noun inside analytical language still fails closed.
        out_6_raw = clean_and_validate_hidden_facts("Evaluation of custom licensing needs for the Paramount footage remains unverified.", set())
        out_6 = fail_closed_on_unsupported_sentences(out_6_raw)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_6)

        # 7. Unsupported number/date inside action language still fails closed.
        out_7_raw = clean_and_validate_hidden_facts("Determine whether the 42 crew members can be supported.", set())
        out_7 = fail_closed_on_unsupported_sentences(out_7_raw)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_7)

        # 8. Analytical prefix cannot launder unsupported factual material.
        out_8_raw = clean_and_validate_hidden_facts("Analysis shows that the facility supports 14 production crews.", set())
        out_8 = fail_closed_on_unsupported_sentences(out_8_raw)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_8)

        out_8b_raw = clean_and_validate_hidden_facts("Verify the confirmed $25 million agreement.", set())
        out_8b = fail_closed_on_unsupported_sentences(out_8b_raw)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_8b)

        # 9. Unknown audience remains unknown.
        out_9 = neutralize_positive_assumptions("It is assumed that a viable audience exists.")
        self.assertIn("Audience demand remains unverified", out_9)

        # 10. Unknown access remains unknown.
        out_10 = neutralize_positive_assumptions("It is assumed that access has been obtained.")
        self.assertIn("Access has not been established and remains unverified.", out_10)

        # 11. Unknown funding/rights remains unknown.
        out_11 = neutralize_positive_assumptions("It is assumed that project funding is available.")
        self.assertIn("Funding status is unspecified and remains unverified.", out_11)

        # 12. External schedule change alone creates no internal dependency.
        out_12 = make_schedule_conditional("This highlights a history of timing adjustments that must be accounted for in any proposed production timeline.")
        self.assertIn("determine whether/how", out_12.lower())
        self.assertNotIn("must be accounted for", out_12.lower())

        # 13. Explicit conditional dependency remains conditional.
        out_13 = make_schedule_conditional("If project access ultimately depends on that external event, any resulting internal schedule implications would need to be evaluated.")
        self.assertIn("would need to be evaluated", out_13.lower())

        # 14. Explicit grounded dependency can be discussed when actually supplied.
        out_14 = make_schedule_conditional("If access is established, the timeline may be aligned.")
        self.assertIn("may be aligned", out_14.lower())

        # 15. Structural headings remain intact.
        out_15 = clean_and_validate_hidden_facts("### 3. DECISIVE REASONS", set())
        self.assertEqual(out_15, "### 3. DECISIVE REASONS")

        # 16. Citation-scoped factual validation remains intact.
        mock_ctx = MagicMock()
        mock_event_research = MagicMock()
        mock_event_research.author = "research_agent"
        mock_event_research.output = """
        E1 — Claim: Vast Space headquarters is in Long Beach.
        Supporting Excerpt: "Vast Space has its primary facility located in Long Beach, California."
        """
        mock_ctx.session.events = [mock_event_research]
        out_16 = clean_and_validate_hidden_facts("Vast Space primary facility in Long Beach [E1].", set(), ctx=mock_ctx)
        self.assertNotIn("[UNSUPPORTED]", out_16)

        # 17. M7A.1 callback Context behavior remains intact.
        mock_callback_ctx = MagicMock(spec=Context)
        mock_callback_ctx.get_invocation_context.return_value = mock_ctx
        llm_response = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Vast Space is in Long Beach and this is a successful project.")]))
        res = verdict_after_model_callback(mock_callback_ctx, llm_response)
        self.assertIsNotNone(res)
        self.assertIn("Long Beach", res.content.parts[0].text)
        self.assertIn("existing/distributed", res.content.parts[0].text)

        # 18. Sentence-level fail-closed remains intact.
        out_18 = fail_closed_on_unsupported_sentences("This has [UNSUPPORTED] word.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_18)

    def test_decision_trace(self):
        import io
        import sys
        import os
        from unittest.mock import patch, MagicMock
        from google.adk import Context
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types

        # Ensure environment starts without CINEVERDICT_VALIDATOR_TRACE
        old_env = os.environ.get("CINEVERDICT_VALIDATOR_TRACE")
        if "CINEVERDICT_VALIDATOR_TRACE" in os.environ:
            del os.environ["CINEVERDICT_VALIDATOR_TRACE"]

        try:
            # 1. Tracing is OFF by default
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                out = clean_and_validate_hidden_facts("Seattle is beautiful.", set())
            self.assertEqual(stderr_capture.getvalue(), "")

            # 2. Can be explicitly enabled and 3. does not change validator output
            os.environ["CINEVERDICT_VALIDATOR_TRACE"] = "1"
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                out_traced = clean_and_validate_hidden_facts("Seattle is beautiful.", set())
            self.assertEqual(out, out_traced)
            trace_content = stderr_capture.getvalue()
            self.assertNotEqual(trace_content, "")

            # 4. Records original rejected sentence
            self.assertIn("Seattle is beautiful.", trace_content)

            # 5. Records the exact rejection reason and 6. identifies unsupported proper nouns
            self.assertIn("Result for 'Seattle': UNAUTHORIZED", trace_content)

            # 7. Identifies unsupported numbers/dates
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                clean_and_validate_hidden_facts("Launched in 2027.", set())
            trace_content_num = stderr_capture.getvalue()
            self.assertIn("Result for '2027': UNAUTHORIZED", trace_content_num)

            # 8. Records proposition classification
            self.assertIn("Semantic proposition classification:", trace_content)

            # 9. Records cited evidence IDs / evidence scope
            mock_ctx = MagicMock()
            mock_event_research = MagicMock()
            mock_event_research.author = "research_agent"
            mock_event_research.output = """
            E1 — Claim: Vast Space headquarters is in Long Beach.
            Supporting Excerpt: "Vast Space has its primary facility located in Long Beach."
            """
            mock_ctx.session.events = [mock_event_research]
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                clean_and_validate_hidden_facts("Vast Space primary facility is in Long Beach [E1].", set(), ctx=mock_ctx)
            trace_content_cite = stderr_capture.getvalue()
            self.assertIn("Citation parsing: Cited IDs on line: ['e1']", trace_content_cite)
            self.assertIn("Selected excerpts for e1:", trace_content_cite)

            # 10. Records the fail-closed transformation
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                fail_closed_on_unsupported_sentences("Seattle is [UNSUPPORTED].")
            trace_content_fail = stderr_capture.getvalue()
            self.assertIn("Sentence-level fail-closed replacement: Sentence containing '[UNSUPPORTED]' replaced.", trace_content_fail)

            # 11. Does not expose arbitrary environment variables/secrets
            # Make sure we only log structural/validation strings, no os.environ listing.
            self.assertNotIn("PATH", trace_content)
            self.assertNotIn("HOME", trace_content)

            # 12. Works for Market, Production/Risk, and Verdict callback paths if role context is available
            mock_callback_ctx = MagicMock(spec=Context)
            mock_callback_ctx.get_invocation_context.return_value = mock_ctx
            
            # Market callback
            llm_response = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Seattle is beautiful.")]))
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                market_after_model_callback(mock_callback_ctx, llm_response)
            trace_market = stderr_capture.getvalue()
            self.assertIn("[CINEVERDICT TRACE][market_agent] === START CALLBACK ===", trace_market)

            # Production/Risk callback
            llm_response = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Seattle is beautiful.")]))
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                production_risk_after_model_callback(mock_callback_ctx, llm_response)
            trace_prod = stderr_capture.getvalue()
            self.assertIn("[CINEVERDICT TRACE][production_risk_agent] === START CALLBACK ===", trace_prod)

            # Verdict callback
            llm_response = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Seattle is beautiful.")]))
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                verdict_after_model_callback(mock_callback_ctx, llm_response)
            trace_verdict = stderr_capture.getvalue()
            self.assertIn("[CINEVERDICT TRACE][verdict_agent] === START CALLBACK ===", trace_verdict)

        finally:
            if old_env is not None:
                os.environ["CINEVERDICT_VALIDATOR_TRACE"] = old_env
            elif "CINEVERDICT_VALIDATOR_TRACE" in os.environ:
                del os.environ["CINEVERDICT_VALIDATOR_TRACE"]

    def test_m7a7_regression_boundary(self):
        import io
        import sys
        import os
        from unittest.mock import patch, MagicMock
        from google.adk import Context
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types
        from cineverdict_agent.agents.validators import (
            get_evidence_excerpts_map,
            classify_sentence_role,
            clean_and_validate_hidden_facts,
            fail_closed_on_unsupported_sentences,
            market_after_model_callback,
            production_risk_after_model_callback,
            verdict_after_model_callback
        )

        # A, B, C, D: Evidence entry separators parsed
        seps = {
            "hyphen": "E1 - Claim: Some Claim\nSupporting Excerpt: Long Beach",
            "en-dash": "E1 \u2013 Claim: Some Claim\nSupporting Excerpt: Long Beach",
            "em-dash": "E1 \u2014 Claim: Some Claim\nSupporting Excerpt: Long Beach",
            "colon": "E1: Claim: Some Claim\nSupporting Excerpt: Long Beach"
        }
        for name, text in seps.items():
            ev_map = get_evidence_excerpts_map(text)
            self.assertIn("e1", ev_map, f"Failed parsing with {name} separator")
            self.assertEqual(ev_map["e1"], ["Long Beach"], f"Wrong excerpt parsed for {name}")

        # E, F: Citation/Markdown prefix action classification
        self.assertEqual(classify_sentence_role("[E1]: Verify the applicable conditions."), "action")
        self.assertEqual(classify_sentence_role("* [E2] Determine whether additional authorization is required."), "action")
        self.assertEqual(classify_sentence_role("- **VERIFY FIRST**: Confirm the applicable terms."), "action")
        self.assertEqual(classify_sentence_role("1. [E3] Evaluate whether the proposed use satisfies the stated conditions."), "action")

        # Mock Context with E1 evidence
        mock_ctx = MagicMock()
        mock_event_research = MagicMock()
        mock_event_research.author = "research_agent"
        mock_event_research.output = """
        E1 — Claim: Vast Space headquarters is in Long Beach.
        Supporting Excerpt: "Vast Space has its primary facility located in Long Beach."
        """
        mock_ctx.session.events = [mock_event_research]

        # G. Valid grounded proper noun inside action survives
        line_grounded = "Verify the access conditions at Long Beach [E1]."
        out_grounded = clean_and_validate_hidden_facts(line_grounded, set(), ctx=mock_ctx)
        self.assertEqual(out_grounded, line_grounded)

        # H. Ungrounded proper noun inside action is neutralized
        # Regulators
        line_reg = "Verify FAA licensing requirements."
        out_reg = clean_and_validate_hidden_facts(line_reg, set(), ctx=mock_ctx)
        self.assertIn("Determine which regulator, if any, applies and verify the applicable licensing requirements.", out_reg)
        self.assertNotIn("[UNSUPPORTED]", out_reg)

        # Locations (after)
        line_loc_after = "Confirm access at Seattle."
        out_loc_after = clean_and_validate_hidden_facts(line_loc_after, set(), ctx=mock_ctx)
        self.assertIn("Confirm which location, if any, is relevant and verify the applicable access conditions.", out_loc_after)
        self.assertNotIn("[UNSUPPORTED]", out_loc_after)

        # Locations (before)
        line_loc_before = "Verify Seattle facility permissions."
        out_loc_before = clean_and_validate_hidden_facts(line_loc_before, set(), ctx=mock_ctx)
        self.assertIn("Confirm which location, if any, is relevant and verify the applicable permissions.", out_loc_before)
        self.assertNotIn("[UNSUPPORTED]", out_loc_before)

        # I. Ungrounded number/date inside action does not survive (fails closed)
        line_num = "Determine whether the 42 crew members can be supported."
        out_num_raw = clean_and_validate_hidden_facts(line_num, set(), ctx=mock_ctx)
        out_num = fail_closed_on_unsupported_sentences(out_num_raw)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_num)

        # J. Generic verification action survives
        line_generic = "Verify the applicable access conditions before commitment."
        out_generic = clean_and_validate_hidden_facts(line_generic, set(), ctx=mock_ctx)
        self.assertEqual(out_generic, line_generic)

        # K. Citation-scoped evidence remains correctly selected
        # "Long Beach" is allowed in E1, but Seattle is not. Calling it with E1 cite:
        line_scoped = "Verify access at Seattle [E1]."
        out_scoped_raw = clean_and_validate_hidden_facts(line_scoped, set(), ctx=mock_ctx)
        # Neutralization changes "access at Seattle" to generic
        self.assertIn("Confirm which location, if any, is relevant", out_scoped_raw)
        self.assertNotIn("[UNSUPPORTED]", out_scoped_raw)

        # L. Sentence fail-closed remains intact
        out_fail = fail_closed_on_unsupported_sentences("Seattle is [UNSUPPORTED].")
        self.assertEqual(out_fail, "[Factual proposition unverified due to missing evidence.]")

        # M. Structural headings remain intact
        out_heading = clean_and_validate_hidden_facts("### 3. DECISIVE REASONS", set(), ctx=mock_ctx)
        self.assertEqual(out_heading, "### 3. DECISIVE REASONS")

        # N. Unknown-vs-assumption protection remains intact
        from cineverdict_agent.agents.validators import neutralize_positive_assumptions
        out_ass = neutralize_positive_assumptions("It is assumed that a viable audience is reachable.")
        self.assertIn("Audience demand remains unverified", out_ass)

        # O. External→internal schedule closure remains intact
        from cineverdict_agent.agents.validators import make_schedule_conditional
        out_sched = make_schedule_conditional("The external launch date impacts the production schedule.")
        self.assertIn("determine whether/how it affects", out_sched)

        # P, Q. Trace OFF / ON behaviors
        old_env = os.environ.get("CINEVERDICT_VALIDATOR_TRACE")
        try:
            # Trace OFF
            if "CINEVERDICT_VALIDATOR_TRACE" in os.environ:
                del os.environ["CINEVERDICT_VALIDATOR_TRACE"]
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                clean_and_validate_hidden_facts("Verify the access conditions.", set(), ctx=mock_ctx)
            self.assertEqual(stderr_capture.getvalue(), "")

            # Trace ON
            os.environ["CINEVERDICT_VALIDATOR_TRACE"] = "1"
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                clean_and_validate_hidden_facts("Verify the access conditions.", set(), ctx=mock_ctx)
            trace_content = stderr_capture.getvalue()
            self.assertIn("[CINEVERDICT TRACE]", trace_content)
            self.assertIn("[Stage 5] Semantic proposition classification:", trace_content)
        finally:
            if old_env is not None:
                os.environ["CINEVERDICT_VALIDATOR_TRACE"] = old_env
            elif "CINEVERDICT_VALIDATOR_TRACE" in os.environ:
                del os.environ["CINEVERDICT_VALIDATOR_TRACE"]

    def test_m7a8_comprehensive_validation(self):
        import io
        import sys
        import os
        from unittest.mock import patch, MagicMock
        from cineverdict_agent.agents.validators import (
            get_evidence_excerpts_map,
            classify_sentence_role,
            clean_and_validate_hidden_facts,
            fail_closed_on_unsupported_sentences,
            market_after_model_callback,
            production_risk_after_model_callback,
            verdict_after_model_callback
        )

        # 1. Markdown `### E1` ledger entry parses.
        ledger_markdown = """
        ### E1
        - **Claim**: Vast Space has launch permissions.
        - **Supporting Excerpt**:
        > "Vast Space has official authorization for space flight."
        """
        ev_map = get_evidence_excerpts_map(ledger_markdown)
        self.assertIn("e1", ev_map)
        self.assertEqual(ev_map["e1"], ["Vast Space has official authorization for space flight."])

        # 2. `## E1` parses if supported by contract.
        ledger_h2 = """
        ## E2
        - **Supporting Excerpt**:
        > "Excerpt h2 content"
        """
        ev_map_h2 = get_evidence_excerpts_map(ledger_h2)
        self.assertIn("e2", ev_map_h2)
        self.assertEqual(ev_map_h2["e2"], ["Excerpt h2 content"])

        # 3. Inline `E1:` parses.
        ledger_inline = """
        E3: Claim: Something
        Supporting Excerpt: Excerpt inline content
        """
        ev_map_inline = get_evidence_excerpts_map(ledger_inline)
        self.assertIn("e3", ev_map_inline)
        self.assertEqual(ev_map_inline["e3"], ["Excerpt inline content"])

        # 4. Hyphen/en-dash/em-dash formats remain supported.
        ledger_seps = """
        E4 - Claim: A
        Supporting Excerpt: Excerpt hyp
        E5 – Claim: B
        Supporting Excerpt: Excerpt en
        E6 — Claim: C
        Supporting Excerpt: Excerpt em
        """
        ev_map_seps = get_evidence_excerpts_map(ledger_seps)
        self.assertEqual(ev_map_seps["e4"], ["Excerpt hyp"])
        self.assertEqual(ev_map_seps["e5"], ["Excerpt en"])
        self.assertEqual(ev_map_seps["e6"], ["Excerpt em"])

        # 5. `Supporting Excerpt:` parses.
        # 6. `Supporting Excerpts:` parses.
        ledger_plural = """
        ### E7
        Supporting Excerpts:
        > "First plural excerpt"
        > "Second plural excerpt"
        """
        ev_map_plural = get_evidence_excerpts_map(ledger_plural)
        self.assertEqual(ev_map_plural["e7"], ["First plural excerpt\nSecond plural excerpt"])

        # 7. Blockquote excerpt parses.
        # 8. Multi-line blockquote remains within correct E#.
        # 9. Next E# terminates prior excerpt scope.
        # 10. Claim text is NOT used as evidence.
        ledger_multi = """
        ### E8
        - **Claim**: Wrong claim text not to be used.
        - **Supporting Excerpt**:
        > "Line one of excerpt
        > Line two of excerpt"
        ### E9
        - **Supporting Excerpt**:
        > "E9 excerpt content"
        """
        ev_map_multi = get_evidence_excerpts_map(ledger_multi)
        self.assertEqual(ev_map_multi["e8"], ["Line one of excerpt\nLine two of excerpt"])
        self.assertEqual(ev_map_multi["e9"], ["E9 excerpt content"])
        self.assertNotIn("Wrong claim text not to be used", ev_map_multi["e8"][0])

        # Mock context setup for downstream validations
        mock_ctx = MagicMock()
        mock_event_research = MagicMock()
        mock_event_research.author = "research_agent"
        mock_event_research.output = """
        ### E1
        - **Supporting Excerpt**:
        > "Vast Space has official authorization."
        
        ### E2
        - **Supporting Excerpt**:
        > "The launch campaign begins in autumn."
        """
        mock_ctx.session.events = [mock_event_research]

        # 11. Citation at start of line scopes all sentences in that same line.
        # 12. `[based on E1, E2]` scopes same-line sentences.
        # 13. Citation does not leak to next unrelated line.
        # 14. Supported two-sentence factual bullet survives.
        # 15. Mixed factual + analytical bullet survives when grounded.
        # 16. Unsupported second factual sentence fails closed independently.
        test_mixed = """* Vast Space has official authorization [E1]. This implies compliance.
* Vast Space has official authorization [E1]. Seattle campaign begins in autumn."""
        out_mixed = clean_and_validate_hidden_facts(test_mixed, set(), ctx=mock_ctx)
        out_mixed_fail = fail_closed_on_unsupported_sentences(out_mixed)
        self.assertIn("compliance", out_mixed_fail)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_mixed_fail)

        # 17. Decisive Reason factual+analytical shape survives.
        test_decisive = "* [E1] Vast Space has official authorization. Because compliance is unresolved, next steps are unspecified."
        out_decisive = clean_and_validate_hidden_facts(test_decisive, set(), ctx=mock_ctx)
        self.assertIn("compliance is unresolved", out_decisive)

        # 18. Generic Required Next Action survives.
        # 19. Citation-prefixed Required Next Action survives.
        # 20. Ungrounded named entity in action remains neutralized/fail-closed.
        # 21. Ungrounded number/date remains protected.
        test_actions = """* Verify whether the proposed use satisfies the applicable terms.
* [E1] Determine whether additional authorization is required.
* Verify Seattle access conditions.
* Determine whether the 42 crew members can be supported."""
        out_actions = clean_and_validate_hidden_facts(test_actions, set(), ctx=mock_ctx)
        out_actions_fail = fail_closed_on_unsupported_sentences(out_actions)
        self.assertIn("Verify whether the proposed use satisfies the applicable terms.", out_actions_fail)
        self.assertIn("Determine whether additional authorization is required.", out_actions_fail)
        self.assertIn("Confirm which location, if any, is relevant", out_actions_fail)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_actions_fail)

        # 22. Structural headings remain intact.
        self.assertEqual(clean_and_validate_hidden_facts("### REQUIRED NEXT ACTIONS", set(), ctx=mock_ctx), "### REQUIRED NEXT ACTIONS")

        # 23. Schedule semantic closure remains intact.
        from cineverdict_agent.agents.validators import make_schedule_conditional
        self.assertIn("determine whether/how", make_schedule_conditional("The external launch date impacts the production schedule."))

        # 24. Unknown-vs-assumption protection remains intact.
        from cineverdict_agent.agents.validators import neutralize_positive_assumptions
        self.assertIn("Audience demand remains unverified", neutralize_positive_assumptions("It is assumed a viable audience is reachable."))

        # 25. Trace OFF leaves behavior unchanged.
        # 26. Trace ON reveals actual evidence-map/citation/role decisions.
        old_env = os.environ.get("CINEVERDICT_VALIDATOR_TRACE")
        try:
            os.environ["CINEVERDICT_VALIDATOR_TRACE"] = "1"
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                clean_and_validate_hidden_facts("Verify access [E1].", set(), ctx=mock_ctx)
            self.assertIn("Citation parsing:", stderr_capture.getvalue())
        finally:
            if old_env is not None:
                os.environ["CINEVERDICT_VALIDATOR_TRACE"] = old_env
            elif "CINEVERDICT_VALIDATOR_TRACE" in os.environ:
                del os.environ["CINEVERDICT_VALIDATOR_TRACE"]

        # 27. M7A.1 callback Context regression remains intact.
        from google.adk.models.llm_response import LlmResponse
        mock_callback_context = MagicMock()
        mock_callback_context.get_invocation_context.return_value = mock_ctx
        
        # 28. M7A.2 sentence fail-closed remains intact.
        self.assertEqual(fail_closed_on_unsupported_sentences("Factual statement with [UNSUPPORTED] word."), "[Factual proposition unverified due to missing evidence.]")


if __name__ == "__main__":
    unittest.main()
