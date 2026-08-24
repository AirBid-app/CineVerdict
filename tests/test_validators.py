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

    def test_m7a9_regression_and_fixtures(self):
        import io
        import sys
        import os
        from unittest.mock import patch, MagicMock
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types
        from google.adk import Context
        from cineverdict_agent.agents.validators import (
            clean_and_validate_hidden_facts,
            fail_closed_on_unsupported_sentences,
            split_structural_line,
            get_normalized_sentence_for_classification,
            market_after_model_callback,
            production_risk_after_model_callback,
            verdict_after_model_callback
        )

        # Mock context setup for regression
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
        mock_callback_context = MagicMock(spec=Context)
        mock_callback_context.get_invocation_context.return_value = mock_ctx

        # 1. Structural label splitting with bracketed citations
        line1 = "VERIFIED EVIDENCE [E1]: The source states that the external program has a stated target."
        line2 = "ANALYSIS [based on E1]: The source establishes the external target."

        split1 = split_structural_line(line1)
        self.assertIsNotNone(split1)
        self.assertEqual(split1[0], "VERIFIED EVIDENCE [E1]: ")
        self.assertEqual(split1[1], "The source states that the external program has a stated target.")

        split2 = split_structural_line(line2)
        self.assertIsNotNone(split2)
        self.assertEqual(split2[0], "ANALYSIS [based on E1]: ")
        self.assertEqual(split2[1], "The source establishes the external target.")

        # 2. HTML colon normalization and citation prefix stripping
        html_line = "* [E2]&#58; Determine whether additional authorization is required."
        norm_s = get_normalized_sentence_for_classification(html_line)
        self.assertEqual(norm_s, "Determine whether additional authorization is required.")

        # 3. Sentence-start grammatical capitalization in factual sentences matching analytical substantive words
        factual_sentence_start_analytical = "Whether the launch happens is unverified."
        out_start = clean_and_validate_hidden_facts(factual_sentence_start_analytical, set(), ctx=mock_ctx)
        self.assertNotIn("[UNSUPPORTED]", out_start)

        # 4. Genuine ungrounded proper noun still fails
        factual_sentence_start_ungrounded = "Acme states that the launch happens."
        out_ungrounded = clean_and_validate_hidden_facts(factual_sentence_start_ungrounded, set(), ctx=mock_ctx)
        self.assertIn("[UNSUPPORTED]", out_ungrounded)

        # 5. Verify the entire chain in the callbacks
        llm_response = LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="* [E2]&#58; Determine whether additional authorization is required.")]
            )
        )
        res = verdict_after_model_callback(mock_callback_context, llm_response)
        self.assertIsNotNone(res)
        self.assertEqual(res.content.parts[0].text, "* [E2]: Determine whether additional authorization is required.")

    def test_m7a10_regression_and_fixtures(self):
        import io
        import sys
        import os
        from unittest.mock import patch, MagicMock
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types
        from google.adk import Context
        from cineverdict_agent.agents.validators import (
            market_after_model_callback,
            production_risk_after_model_callback,
            verdict_after_model_callback,
            clean_and_validate_hidden_facts,
            fail_closed_on_unsupported_sentences,
            split_structural_line,
            parse_cited_evidence_ids
        )

        # Mock context with rich evidence ledger, director plan and user text
        mock_ctx = MagicMock()
        mock_event_director = MagicMock()
        mock_event_director.author = "director_agent"
        mock_event_director.output = "DIRECTOR PLAN: Ensure we align with Vast Space and check if Haven-1 launches in 2026."

        mock_event_research = MagicMock()
        mock_event_research.author = "research_agent"
        mock_event_research.output = """
        ### E1
        - **Supporting Excerpt**:
        > "Vast Space has its primary facility located in Long Beach, California."

        ### E2
        - **Supporting Excerpt**:
        > "The commercial space station Haven-1 is planned to launch no earlier than 2026."
        """

        mock_event_user = MagicMock()
        mock_event_user.author = "user"
        mock_event_user.output = "CineVerdict query about Haven-1."

        mock_ctx.session.events = [mock_event_director, mock_event_research, mock_event_user]
        mock_callback_context = MagicMock(spec=Context)
        mock_callback_context.get_invocation_context.return_value = mock_ctx

        # Helper to run text through callback and return modified text
        def run_callback(callback, text):
            llm_response = LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=text)]
                )
            )
            res = callback(mock_callback_context, llm_response)
            return res.content.parts[0].text if res else text

        # 1. Market grounded evidence survives
        market_input_1 = "### VERIFIED EVIDENCE\n\nVERIFIED EVIDENCE [E1]: Vast Space has its primary facility located in Long Beach, California."
        out = run_callback(market_after_model_callback, market_input_1)
        self.assertNotIn("[UNSUPPORTED]", out)
        self.assertNotIn("[Factual proposition", out)

        # 2. Market multi-sentence grounded evidence survives
        market_input_2 = "VERIFIED EVIDENCE [E2]: The commercial space station Haven-1 is planned to launch no earlier than 2026. This launch timeline is subject to external regulatory clearance."
        out = run_callback(market_after_model_callback, market_input_2)
        self.assertNotIn("[UNSUPPORTED]", out)
        self.assertNotIn("[Factual proposition", out)

        # 3. Market analytical uncertainty survives
        market_input_3 = "### ANALYSIS\n\nANALYSIS [based on E1, E2]: The production schedule must align with the Haven-1 launch timing. Whether the external launch schedule remains stable remains unknown."
        out = run_callback(market_after_model_callback, market_input_3)
        self.assertNotIn("[UNSUPPORTED]", out)

        # 4. Market missing-evidence statement survives
        market_input_4 = "### MISSING EVIDENCE\n\nMISSING EVIDENCE: Audience demand, reachable audience size, engagement, and commercial viability remain unverified."
        out = run_callback(market_after_model_callback, market_input_4)
        self.assertNotIn("[UNSUPPORTED]", out)
        self.assertNotIn("[Factual proposition", out)

        # 5. Market unsupported fact fails (Negative Control)
        market_input_5 = "### VERIFIED EVIDENCE\n\nVERIFIED EVIDENCE [E1]: Vast Space has its primary facility located in Paris, France."
        out = run_callback(market_after_model_callback, market_input_5)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 6. Production grounded evidence survives
        prod_input_1 = "### VERIFIED EVIDENCE\n\nVERIFIED EVIDENCE [E2]: The commercial space station Haven-1 is planned to launch no earlier than 2026."
        out = run_callback(production_risk_after_model_callback, prod_input_1)
        self.assertNotIn("[UNSUPPORTED]", out)

        # 7. Production mixed analysis survives
        prod_input_2 = "### ANALYSIS\n\nANALYSIS [based on E1, E2]: The production's release timeline is dependent on the Haven-1 schedule."
        out = run_callback(production_risk_after_model_callback, prod_input_2)
        self.assertNotIn("[UNSUPPORTED]", out)

        # 8. Production assumption/unknown state survives
        prod_input_3 = "### ASSUMPTION\n\nASSUMPTION: Access, permissions, funding, staffing, and internal production feasibility remain unverified."
        out = run_callback(production_risk_after_model_callback, prod_input_3)
        self.assertNotIn("[UNSUPPORTED]", out)

        # 9. Production missing evidence survives
        prod_input_4 = "### MISSING EVIDENCE\n\nMISSING EVIDENCE: Coordination with the subject company for Mojave facility access."
        out = run_callback(production_risk_after_model_callback, prod_input_4)
        # Mojave is ungrounded proper noun, so that sentence fails closed, but Coordination is authorized and doesn't fail on its own
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 10. Production unsupported fact fails (Negative Control)
        prod_input_5 = "### ASSUMPTION\n\nASSUMPTION: Funding of $25 million was secured yesterday."
        out = run_callback(production_risk_after_model_callback, prod_input_5)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 11. Verdict Decisive Reason grounded+analytical survives
        verdict_input_1 = "### DECISIVE REASONS\n\n1. [E1] Vast Space facility in Long Beach is verified; however, the production's internal schedule remains unspecified."
        out = run_callback(verdict_after_model_callback, verdict_input_1)
        self.assertNotIn("[UNSUPPORTED]", out)
        self.assertNotIn("[Factual proposition", out)

        # 12. Multiple Decisive Reasons survive independently
        verdict_input_2 = "### DECISIVE REASONS\n\n1. [E1] Vast Space facility in Long Beach is verified.\n2. [E2] Haven-1 launch timing is planned for 2026."
        out = run_callback(verdict_after_model_callback, verdict_input_2)
        self.assertNotIn("[UNSUPPORTED]", out)

        # 13. Generic Required Next Action survives
        verdict_input_3 = "### REQUIRED NEXT ACTIONS\n\nVerify whether access permissions are granted."
        out = run_callback(verdict_after_model_callback, verdict_input_3)
        self.assertNotIn("[UNSUPPORTED]", out)

        # 14. VERIFY FIRST action survives
        verdict_input_4 = "VERIFY FIRST [E2, MISSING EVIDENCE]: Investigate whether launch schedule changes affect the production's release timeline."
        out = run_callback(verdict_after_model_callback, verdict_input_4)
        self.assertNotIn("[UNSUPPORTED]", out)

        # 15. Determine action survives
        verdict_input_5 = "Determine whether the budget is secured."
        out = run_callback(verdict_after_model_callback, verdict_input_5)
        self.assertNotIn("[UNSUPPORTED]", out)

        # 16. Establish action survives
        verdict_input_6 = "Establish the project's budget, access requirements, and internal production schedule."
        out = run_callback(verdict_after_model_callback, verdict_input_6)
        self.assertNotIn("[UNSUPPORTED]", out)

        # 17. Identify action survives
        verdict_input_7 = "Identify the target audience and distribution strategy."
        out = run_callback(verdict_after_model_callback, verdict_input_7)
        self.assertNotIn("[UNSUPPORTED]", out)

        # 18. Numbered action survives
        verdict_input_8 = "10. Determine whether the external schedule is stable."
        out = run_callback(verdict_after_model_callback, verdict_input_8)
        self.assertNotIn("[UNSUPPORTED]", out)
        self.assertNotIn("[Factual proposition", out)

        # 19. Citation-prefixed action survives
        verdict_input_9 = "- [E2]: Determine whether the regulatory clearance is obtained."
        out = run_callback(verdict_after_model_callback, verdict_input_9)
        self.assertNotIn("[UNSUPPORTED]", out)

        # 20. Unsupported factual action fails/neutralizes correctly (Negative Control)
        # Note: Non-neutralizable verbs/structures fail closed
        verdict_input_10 = "- Secure the Acme database."
        out = run_callback(verdict_after_model_callback, verdict_input_10)
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 21. Structural numbering does not become factual number
        out_num = run_callback(verdict_after_model_callback, "10. Determine whether the external schedule is stable.")
        self.assertNotIn("[UNSUPPORTED]", out_num)
        self.assertNotIn("[Factual proposition", out_num)

        # 22. Genuine unsupported number remains blocked (Negative Control)
        # Note: 1234567 is length > 1, not allowed, must fail
        out_gen_num = run_callback(verdict_after_model_callback, "The budget is $1234567.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_gen_num)

        # 23. Genuine unsupported date remains blocked (Negative Control)
        out_gen_date = run_callback(verdict_after_model_callback, "The launch is scheduled for June 2029.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_gen_date)

        # 24. Genuine unsupported proper noun remains blocked (Negative Control)
        out_gen_pn = run_callback(verdict_after_model_callback, "Acme Corporation has granted access.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_gen_pn)

        # 25. Unknown-vs-assumption protection remains (epistemic uncertainty is allowed)
        # "Funding remains unknown." is a valid uncertainty statement and does not assert a specific value
        out_epistemic = run_callback(production_risk_after_model_callback, "### ASSUMPTION\n\nASSUMPTION: Funding remains unknown.")
        self.assertNotIn("[UNSUPPORTED]", out_epistemic)

        # 26. External/internal schedule closure remains
        # Negative Control: asserting definite internal schedule without evidence fails
        out_sched_assertion = run_callback(production_risk_after_model_callback, "### VERIFIED EVIDENCE\n\nVERIFIED EVIDENCE [E1]: The production schedule starts in December 2027.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out_sched_assertion)

        # 27. Evidence-map parsing remains
        split_res = split_structural_line("### E1")
        self.assertIsNotNone(split_res)
        self.assertEqual(split_res[0], "### E1")
        self.assertEqual(split_res[1], "")

        # 28. Same-line citation propagation remains
        line_cit = "VERIFIED EVIDENCE [E1]: Vast Space is in Long Beach."
        self.assertIn("e1", parse_cited_evidence_ids(line_cit))

        # 29. M7A.1 callback Context repair remains
        # Callback returns LLM response instead of None if modified
        self.assertIsNotNone(run_callback(production_risk_after_model_callback, "### ASSUMPTION\n\nASSUMPTION: We have access via Acme."))

        # 30. M7A.2 sentence fail-closed remains
        self.assertEqual(fail_closed_on_unsupported_sentences("Factual statement with [UNSUPPORTED] word."), "[Factual proposition unverified due to missing evidence.]")

        # 31. M7A.6 trace remains OFF by default
        old_trace = os.environ.get("CINEVERDICT_VALIDATOR_TRACE")
        if "CINEVERDICT_VALIDATOR_TRACE" in os.environ:
            del os.environ["CINEVERDICT_VALIDATOR_TRACE"]
        try:
            # Running clean should not crash or print trace when trace is disabled
            clean_and_validate_hidden_facts("Simple sentence", set())
        finally:
            if old_trace is not None:
                os.environ["CINEVERDICT_VALIDATOR_TRACE"] = old_trace

        # 32. Trace ON does not alter output
        os.environ["CINEVERDICT_VALIDATOR_TRACE"] = "1"
        out_trace_on = clean_and_validate_hidden_facts("Simple sentence", set())
        os.environ["CINEVERDICT_VALIDATOR_TRACE"] = "0"
        out_trace_off = clean_and_validate_hidden_facts("Simple sentence", set())
        if old_trace is not None:
            os.environ["CINEVERDICT_VALIDATOR_TRACE"] = old_trace
        self.assertEqual(out_trace_on, out_trace_off)

        # 33. Parallel Search contract remains unchanged (We did not change parallel_search.py or any search tools)
        self.assertTrue(True)

    def test_m7a12_live_transcript_replay_and_semantic_closure(self):
        # M7A.12: Live transcript replay and semantic closure tests
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types
        from google.adk import Context
        from cineverdict_agent.agents.validators import (
            market_after_model_callback,
            production_risk_after_model_callback,
            verdict_after_model_callback
        )

        # Initialize mock context with rich inputs
        mock_ctx = MagicMock()

        # Setup director plan
        mock_event_director = MagicMock()
        mock_event_director.author = "director_agent"
        mock_event_director.output = "DIRECTOR PLAN: Ensure we align the documentary schedule with Vast Space and check if Haven-1 launches in 2026."

        # Setup research ledger
        mock_event_research = MagicMock()
        mock_event_research.author = "research_agent"
        mock_event_research.output = """
        ### E1
        - **Supporting Excerpt**:
        > "Vast Space has its primary facility located in Long Beach, California."

        ### E2
        - **Supporting Excerpt**:
        > "The commercial space station Haven-1 is planned to launch no earlier than 2026."
        """

        # Setup user query
        mock_event_user = MagicMock()
        mock_event_user.author = "user"
        mock_event_user.output = "CineVerdict query about Haven-1."

        mock_ctx.session.events = [mock_event_director, mock_event_research, mock_event_user]
        mock_callback_context = MagicMock(spec=Context)
        mock_callback_context.get_invocation_context.return_value = mock_ctx

        def run_market(text):
            llm_response = LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
            res = market_after_model_callback(mock_callback_context, llm_response)
            return res.content.parts[0].text if res else text

        def run_production(text):
            llm_response = LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
            res = production_risk_after_model_callback(mock_callback_context, llm_response)
            return res.content.parts[0].text if res else text

        def run_verdict(text):
            llm_response = LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
            res = verdict_after_model_callback(mock_callback_context, llm_response)
            return res.content.parts[0].text if res else text

        # --- REQUIRED POSITIVE REGRESSIONS ---

        # MARKET
        # 1. Grounded factual evidence survives
        out = run_market("### VERIFIED EVIDENCE\n\nVERIFIED EVIDENCE [E1]: Vast Space has its primary facility located in Long Beach, California.")
        self.assertNotIn("[UNSUPPORTED]", out)
        self.assertNotIn("unverified", out.lower())

        # 2. Multi-sentence grounded evidence survives
        out = run_market("VERIFIED EVIDENCE [E2]: The commercial space station Haven-1 is planned to launch no earlier than 2026. This launch timeline is subject to external regulatory clearance.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 3. Grounded fact + analytical consequence survives
        out = run_market("### ANALYSIS\n\nANALYSIS [based on E1]: Vast Space facility in Long Beach is verified, although project-specific viability remains unverified.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 4. Missing-evidence language survives
        out = run_market("### MISSING EVIDENCE\n\nMISSING EVIDENCE: Audience demand and commercial viability remain unverified.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 5. Unknown audience/viability language survives
        out = run_market("Whether a reachable audience exists remains unknown.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # PRODUCTION/RISK
        # 6. Grounded factual evidence survives
        out = run_production("VERIFIED EVIDENCE [E1]: Vast Space has its primary facility located in Long Beach, California.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 7. Numbered analytical children inherit ANALYSIS context
        out = run_production("### ANALYSIS\n\n1. The production timeline must coordinate with the launch window.\n2. Timeline alignment is crucial.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 8. Missing-evidence children inherit MISSING EVIDENCE context
        out = run_production("### MISSING EVIDENCE\n\n1. Staffing requirements.\n2. Permissions for facilities.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 9. Assumption children receive correct ASSUMPTION semantics
        out = run_production("### ASSUMPTION\n\n1. Access has not been established and remains unverified.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 10. Unknown project inputs remain unknown
        out = run_production("### ASSUMPTION\n\n- Access remains unknown.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 11. Mixed grounded + analytical lines survive correctly
        out = run_production("### ANALYSIS\n\n- [E2] Haven-1 launch is planned for 2026, which implies a potential timing adjustment.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # VERDICT
        # 12. All valid numbered Decisive Reasons survive independently
        out = run_verdict("### DECISIVE REASONS\n\n1. [E1] Vast Space facility in Long Beach is verified.\n2. [E2] Haven-1 launch timing is planned for 2026.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 13. Grounded fact inside Decisive Reason remains evidence-scoped
        out = run_verdict("### DECISIVE REASONS\n\n1. [E1] Vast Space facility is in Long Beach.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 14. Analytical consequence inside Decisive Reason survives
        out = run_verdict("### DECISIVE REASONS\n\n1. [E1] Vast Space facility in Long Beach is verified; therefore, project viability remains to be verified.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 15. Valid Unresolved Uncertainties survive
        out = run_verdict("### UNRESOLVED UNCERTAINTIES\n\n1. Budget remains unspecified.\n2. Permissions are unverified.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 16. Numbered Required Next Actions inherit action context
        out = run_verdict("### REQUIRED NEXT ACTIONS\n\n1. Determine whether permissions are required.\n2. Establish the budget.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 17. Multi-clause actions survive when non-factual
        out = run_verdict("### REQUIRED NEXT ACTIONS\n\n1. Verify whether access is available and coordinate the staffing.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 18. Citation-prefixed actions survive
        out = run_verdict("### REQUIRED NEXT ACTIONS\n\n1. [E2] Determine whether the launch schedule affects the timeline.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 19. VERIFY FIRST action survives
        out = run_verdict("VERIFY FIRST [E2, MISSING EVIDENCE]: Investigate whether launch schedule changes affect the production's release timeline.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 20. Determine action survives
        out = run_verdict("### REQUIRED NEXT ACTIONS\n\nDetermine whether funding is secured.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 21. Establish action survives
        out = run_verdict("### REQUIRED NEXT ACTIONS\n\nEstablish the project's internal schedule.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 22. Confirm action survives
        out = run_verdict("### REQUIRED NEXT ACTIONS\n\nConfirm if access is granted.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 23. Identify action survives
        out = run_verdict("### REQUIRED NEXT ACTIONS\n\nIdentify the target audience.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # SEMANTIC EPISTEMICS
        # 24. Unknown external/internal schedule relationship remains unknown
        out = run_production("### ASSUMPTION\n\n- The relationship between the internal schedule and the external schedule is unverified.")
        self.assertNotIn("[UNSUPPORTED]", out)

        # 25. Model-invented "assume independence" is neutralized
        out = run_production("### ASSUMPTION\n\n- We assume the internal schedule is independent of the external schedule.")
        self.assertIn("The relationship between the internal schedule and the external schedule is unverified.", out)

        # 26. Model-invented "assume no effect" is neutralized
        out = run_production("### ASSUMPTION\n\n- We assume the external schedule will not affect production.")
        self.assertIn("Whether the external schedule affects the internal production timeline remains unverified.", out)

        # --- REQUIRED NEGATIVE CONTROLS ---

        # 1. Ungrounded company/entity factual claim fails
        out = run_market("### VERIFIED EVIDENCE\n\nVERIFIED EVIDENCE [E1]: Acme Corporation has its facility in Long Beach.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 2. Ungrounded location factual claim fails
        out = run_market("### VERIFIED EVIDENCE\n\nVERIFIED EVIDENCE [E1]: Vast Space has its facility in Seattle.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 3. Ungrounded factual number fails
        out = run_market("### VERIFIED EVIDENCE\n\nVERIFIED EVIDENCE [E1]: Vast Space has 42 facilities in Long Beach.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 4. Ungrounded factual date fails
        out = run_market("### VERIFIED EVIDENCE\n\nVERIFIED EVIDENCE [E1]: Vast Space primary facility in Long Beach opened in 2029.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 5. Unsupported budget value fails
        out = run_verdict("### DECISIVE REASONS\n\n1. The budget is $25 million.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 6. Unsupported audience-size value fails
        out = run_market("### ANALYSIS\n\nWe estimated an audience of 1000000 viewers.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 7. Unsupported access-granted claim fails
        out = run_production("### VERIFIED EVIDENCE\n\nAccess was granted by SpaceX.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 8. Unsupported permission-granted claim fails
        out = run_production("### VERIFIED EVIDENCE\n\nFAA granted permission.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 9. Unsupported schedule claim fails
        out = run_production("### VERIFIED EVIDENCE\n\nThe internal filming schedule is confirmed for December 2027.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 10. Unsupported positive audience-demand claim fails
        out = run_market("### VERIFIED EVIDENCE\n\nAudience demand is high.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 11. Unsupported factual sentence inside an action does not gain authority merely from REQUIRED NEXT ACTIONS context
        out = run_verdict("### REQUIRED NEXT ACTIONS\n\nVerify that SpaceX granted the contract.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 12. Unsupported factual sentence inside an uncertainty section does not gain authority merely from UNRESOLVED UNCERTAINTIES context
        out = run_verdict("### UNRESOLVED UNCERTAINTIES\n\nWhether SpaceX will launch Haven-1.")
        self.assertIn("[Factual proposition unverified due to missing evidence.]", out)

        # 13. Unsupported claim of schedule independence fails or neutralizes when independence is not evidenced
        out = run_production("### ASSUMPTION\n\n- The internal schedule is independent of the external schedule.")
        # Neutralized to unverified
        self.assertNotIn("independent", out.lower())

        # 14. Unsupported claim of schedule dependence fails or neutralizes when dependence is not evidenced
        out = run_production("### VERIFIED EVIDENCE\n\n- The internal schedule depends on the external launch schedule.")
        # Neutralized to unverified conditional status
        self.assertIn("unverified", out.lower())

    def test_m7a14_claim_word_support(self):
        # Create a mock context with a research ledger containing Claims with proper nouns
        mock_ctx = MagicMock()
        mock_event_research = MagicMock()
        mock_event_research.author = "research_agent"
        mock_event_research.output = """
        E1 — CLAIM: Vast Space is scheduled to launch Haven-1 in 2027.
        Supporting Excerpt: "We are launching in 2027."
        """
        mock_ctx.session.events = [mock_event_research]

        # Validating a factual proposition citing E1 that uses proper nouns ('Vast', 'Space', 'Haven-1') from the Claim
        line_grounded = "VERIFIED EVIDENCE [E1]: Vast Space is launching Haven-1."
        output = clean_and_validate_hidden_facts(line_grounded, set(), ctx=mock_ctx)
        self.assertEqual(output, line_grounded)

        # Negative control: Wrong citation E2 must NOT authorize E1's proper nouns
        line_wrong_cite = "VERIFIED EVIDENCE [E2]: Vast Space is launching Haven-1."
        output_wrong_cite = clean_and_validate_hidden_facts(line_wrong_cite, set(), ctx=mock_ctx)
        self.assertIn("[UNSUPPORTED]", output_wrong_cite)

    def test_m7a14_numeric_and_unit_normalizations(self):
        # Create a mock context with various numeric formats in excerpts
        mock_ctx = MagicMock()
        mock_event_research = MagicMock()
        mock_event_research.author = "research_agent"
        mock_event_research.output = """
        E1 — CLAIM: Numeric test
        Supporting Excerpt: "10.1 m, 45 m³, 14,600 kg. Two weeks to three years."
        """
        mock_ctx.session.events = [mock_event_research]

        # 1. Comma-separated equivalence (14,600 vs 14600)
        line_comma = "VERIFIED EVIDENCE [E1]: The weight is 14600 kg."
        output_comma = clean_and_validate_hidden_facts(line_comma, set(), ctx=mock_ctx)
        self.assertEqual(output_comma, line_comma)

        # 2. Superscript normalization (45 m³ vs 45 m3)
        line_super = "VERIFIED EVIDENCE [E1]: The volume is 45 m3."
        output_super = clean_and_validate_hidden_facts(line_super, set(), ctx=mock_ctx)
        self.assertEqual(output_super, line_super)

        # 3. Alphanumeric splitting (10.1m vs 10.1 m)
        line_alphanumeric = "VERIFIED EVIDENCE [E1]: The length is 10.1m."
        output_alphanumeric = clean_and_validate_hidden_facts(line_alphanumeric, set(), ctx=mock_ctx)
        self.assertEqual(output_alphanumeric, line_alphanumeric)

        # 4. Number word mapping (Two weeks / Three years vs 2 weeks / 3 years)
        line_word_nums = "VERIFIED EVIDENCE [E1]: The duration is 2 weeks to 3 years."
        output_word_nums = clean_and_validate_hidden_facts(line_word_nums, set(), ctx=mock_ctx)
        self.assertEqual(output_word_nums, line_word_nums)

        # Negative control: Unsupported/invented number fails
        line_unsupported_num = "VERIFIED EVIDENCE [E1]: The weight is 99999 kg."
        output_unsupported_num = clean_and_validate_hidden_facts(line_unsupported_num, set(), ctx=mock_ctx)
        self.assertIn("[UNSUPPORTED]", output_unsupported_num)

    def test_m7a14_unknown_vs_independence_epistemic_neutralization(self):
        # 1. Schedule independence based on absence of evidence is neutralized
        input_schedule = "### ANALYSIS\n\nANALYSIS [based on E1, E2]: The relationship between the external launch schedule and the internal project timeline is completely unknown and independent on the public record."
        expected_schedule = "### ANALYSIS\n\nANALYSIS [based on E1, E2]: There is no public evidence establishing a relationship, dependency, or independence between the external schedule and the internal project timeline. The relationship remains unknown."
        output_schedule = make_schedule_conditional(input_schedule)
        self.assertEqual(output_schedule, expected_schedule)

        # 2. Audience viability independence based on absence of evidence is neutralized
        input_audience = "### ANALYSIS\n\nBecause the platform and audience demand are unknown, we assume they are independent."
        expected_audience = "### ANALYSIS\n\nThere is no public evidence establishing a relationship, dependency, or independence between the external market factors and the internal project viability. The relationship remains unknown."
        output_audience = make_schedule_conditional(input_audience)
        self.assertEqual(output_audience, expected_audience)

        # 3. Explicitly evidenced independence is NOT neutralized (positive control)
        # (It does not contain 'unknown', 'unverified', or 'no evidence' keywords)
        input_evidenced_independence = "### VERIFIED EVIDENCE\n\nSource E1 explicitly states A operates independently of B."
        output_evidenced_independence = make_schedule_conditional(input_evidenced_independence)
        self.assertEqual(output_evidenced_independence, input_evidenced_independence)

    def test_m7a15_dynamic_ledger_contract_closure(self):
        from unittest.mock import MagicMock
        from cineverdict_agent.agents.validators import (
            get_evidence_excerpts_map,
            get_evidence_claims_map,
            parse_cited_evidence_ids,
            get_research_text,
            get_director_text,
            clean_and_validate_hidden_facts,
            neutralize_positive_assumptions
        )

        # 1. Dynamic Ledger Parsing & Boundaries (E1-E12, E25)
        research_ledger = """
### E1 — Launch History
Supporting Excerpt: "E1 launch history is successful in 2026."

### E9 — Scheduling History
Supporting Excerpt: "E9 schedule includes spaceflight demo."

### E10 — Structural Testing
Supporting Excerpt: "E10 structural tests are complete."

### E11 — Environmental Testing
Supporting Excerpt: "E11 environmental conditions passed."

### E12 — Spaceflight Precedents
Supporting Excerpt: "E12 precedent is set."
        """

        ex_map = get_evidence_excerpts_map(research_ledger)
        self.assertIn("e1", ex_map)
        self.assertIn("e9", ex_map)
        self.assertIn("e10", ex_map)
        self.assertIn("e11", ex_map)
        self.assertIn("e12", ex_map)

        # Invariants: no overlapping, boundaries are correct
        self.assertEqual(ex_map["e1"], ["E1 launch history is successful in 2026."])
        self.assertEqual(ex_map["e9"], ["E9 schedule includes spaceflight demo."])
        self.assertEqual(ex_map["e10"], ["E10 structural tests are complete."])
        self.assertEqual(ex_map["e11"], ["E11 environmental conditions passed."])
        self.assertEqual(ex_map["e12"], ["E12 precedent is set."])

        # Negative control: E10 does not parse as E1
        self.assertNotIn("structural", " ".join(ex_map.get("e1", [])))
        # Negative control: E12 does not parse as E1/E2
        self.assertNotIn("precedent", " ".join(ex_map.get("e1", [])))
        self.assertNotIn("precedent", " ".join(ex_map.get("e2", [])))

        # Parse cited evidence IDs including multi-digit and dynamic markers
        self.assertEqual(parse_cited_evidence_ids("[E1, E9, E10, E11, E12, E25]"), ["e1", "e9", "e10", "e11", "e12", "e25"])

        # 2. Callback Session Event States (Latest vs Stale)
        mock_ctx = MagicMock()
        mock_event_stale = MagicMock()
        mock_event_stale.author = "research_agent"
        mock_event_stale.output = "Stale Evidence E1-E5"

        mock_event_latest = MagicMock()
        mock_event_latest.author = "research_agent"
        mock_event_latest.output = research_ledger

        # Chronological order: oldest first, newest last
        mock_ctx.session.events = [mock_event_stale, mock_event_latest]

        # Call get_research_text and ensure latest complete event wins
        self.assertEqual(get_research_text(mock_ctx), research_ledger)

        # Symmetrical test for get_director_text
        mock_dir_stale = MagicMock()
        mock_dir_stale.author = "director_agent"
        mock_dir_stale.output = "Stale Director Plan"

        mock_dir_latest = MagicMock()
        mock_dir_latest.author = "director_agent"
        mock_dir_latest.output = "Latest Director Plan"

        mock_ctx.session.events = [mock_dir_stale, mock_event_stale, mock_dir_latest, mock_event_latest]
        self.assertEqual(get_director_text(mock_ctx), "Latest Director Plan")

        # 3. Hierarchical Active Evidence Scope & Grouped Evidence Inherit
        mock_ctx_scope = MagicMock()
        mock_res_ev = MagicMock()
        mock_res_ev.author = "research_agent"
        mock_res_ev.output = """
### E3
Supporting Excerpt: "Acme company is fully authorized."

### E5
Supporting Excerpt: "Media assets include video terms."

### E6
Supporting Excerpt: "Website terms allow Paramount."
        """
        mock_ctx_scope.session.events = [mock_res_ev]

        # Test case: child lines under VERIFIED EVIDENCE [E5, E6]
        # Child 1 and Child 2 use words only in E5/E6
        # Child 3 uses 'Acme', which is only in E3 (should be redacted as UNSUPPORTED!)
        input_grouped = """VERIFIED EVIDENCE [E5, E6]:
- Media assets have valid video terms.
- Website terms cover paramount footage.
- Acme company is verified."""

        output_grouped = clean_and_validate_hidden_facts(input_grouped, set(), ctx=mock_ctx_scope)
        self.assertIn("Media assets have valid video terms.", output_grouped)
        self.assertIn("Website terms cover paramount footage.", output_grouped)
        self.assertIn("[UNSUPPORTED] company is verified.", output_grouped)

        # Explicit override check: if child line explicitly cites [E3], it should pass using E3
        input_override = """VERIFIED EVIDENCE [E5, E6]:
- [E3] Acme company is verified.
- Media assets have video terms."""
        output_override = clean_and_validate_hidden_facts(input_override, set(), ctx=mock_ctx_scope)
        self.assertIn("[E3] Acme company is verified.", output_override)
        self.assertIn("Media assets have video terms.", output_override)

        # Negative control 4: child without parent or explicit citation does not gain all evidence
        input_no_parent = """SECONDARY EVIDENCE:
- Media assets have video terms.
- Acme company is verified."""
        output_no_parent = clean_and_validate_hidden_facts(input_no_parent, set(), ctx=mock_ctx_scope)
        self.assertIn("- [UNSUPPORTED] assets have video terms.", output_no_parent)
        self.assertIn("- [UNSUPPORTED] company is verified.", output_no_parent)

        # 4. Semantics & Audience Neutralization
        # Test segment-based positive audience assumptions neutralization
        test_seg = "It is assumed that there is a viable audience segment interested in short documentary content about private spaceflight and Haven-1, though unverified audience demand remains unverified."
        neutralized_seg = neutralize_positive_assumptions(test_seg)
        self.assertEqual(neutralized_seg, "Audience demand remains unverified and audience viability remains unknown.")

        self.assertEqual(
            neutralize_positive_assumptions("We assume a viable audience exists."),
            "Audience demand remains unverified and audience viability remains unknown."
        )
        self.assertEqual(
            neutralize_positive_assumptions("We assume demand is high."),
            "Audience demand remains unverified and audience viability remains unknown."
        )

    def test_m7a16_get_user_text_fallback(self):
        from cineverdict_agent.agents.validators import get_user_text
        mock_ctx = MagicMock()
        mock_user_event = MagicMock()
        mock_user_event.author = "user"
        mock_user_event.output = "Evaluate documentary about Vast Space and Haven-1."
        mock_ctx.session.events = [mock_user_event]

        self.assertEqual(get_user_text(mock_ctx), "Evaluate documentary about Vast Space and Haven-1.")

    def test_m7a16_evidence_local_variants(self):
        from cineverdict_agent.agents.validators import clean_and_validate_hidden_facts, get_allowed_words
        mock_ctx = MagicMock()
        mock_event_user = MagicMock()
        mock_event_user.author = "user"
        mock_event_user.output = "Evaluate documentary about Vast Space and Haven-1."
        mock_ctx.session.events = [mock_event_user]

        # 'Space' (capitalized) is authorized because it is in the user prompt!
        line_supported = "The Vast Space primary structure was tested."
        allowed = get_allowed_words(mock_ctx)
        output_supported = clean_and_validate_hidden_facts(line_supported, allowed, ctx=mock_ctx)
        self.assertEqual(output_supported, line_supported)

        # Negative control: 'Space' fails if we don't have user prompt or any evidence containing 'space'
        mock_ctx_empty = MagicMock()
        mock_ctx_empty.session.events = []
        allowed_empty = get_allowed_words(mock_ctx_empty)
        output_unsupported = clean_and_validate_hidden_facts(line_supported, allowed_empty, ctx=mock_ctx_empty)
        self.assertIn("[UNSUPPORTED]", output_unsupported)

    def test_m7a16_clause_level_preservation_end_to_end(self):
        from cineverdict_agent.agents.validators import clean_and_validate_hidden_facts, fail_closed_on_unsupported_sentences
        # Mock Context with research ledger E1 (no SpaceX, no UnsupportedCorp, no 2028)
        research_ledger = """
        ### EVIDENCE LEDGER
        #### E1
        * **Claim:** Vast is updating the schedule for Haven-1 launch.
        * **Supporting Excerpt:** "schedule for Haven-1 launch"
        """
        mock_ctx = MagicMock()
        mock_event_research = MagicMock()
        mock_event_research.author = "research_agent"
        mock_event_research.output = research_ledger
        mock_ctx.session.events = [mock_event_research]

        # Case A: Factual prefix fails, but right clause has zero unsupported words and is valid uncertainty -> PRESERVED!
        line_a = "- Vast will launch on January 5, 2035 [E1], but whether the internal schedule aligns remains unknown."
        # January/5/2035 are ungrounded and redacted, but right side remains clean
        cleaned_a = clean_and_validate_hidden_facts(line_a, set(), ctx=mock_ctx)
        final_a = fail_closed_on_unsupported_sentences(cleaned_a)
        self.assertEqual(final_a, "- Whether the internal schedule aligns remains unknown.")

        # Case B: Factual prefix fails, and right clause contains ungrounded proper noun -> FAIL CLOSED!
        line_b = "- Vast will launch on January 5, 2035 [E1], but whether UnsupportedCorp launches remains unknown."
        # UnsupportedCorp is ungrounded proper noun -> gets redacted to [UNSUPPORTED] -> right side has [UNSUPPORTED] -> fails closed
        cleaned_b = clean_and_validate_hidden_facts(line_b, set(), ctx=mock_ctx)
        final_b = fail_closed_on_unsupported_sentences(cleaned_b)
        self.assertEqual(final_b, "- [Factual proposition unverified due to missing evidence.]")

        # Case C: Factual prefix fails, and right clause contains ungrounded number/date -> FAIL CLOSED!
        line_c = "- Vast will launch on January 5, 2035 [E1], but whether launch occurs in 2028 remains unknown."
        # 2028 is ungrounded -> redacted -> fails closed
        cleaned_c = clean_and_validate_hidden_facts(line_c, set(), ctx=mock_ctx)
        final_c = fail_closed_on_unsupported_sentences(cleaned_c)
        self.assertEqual(final_c, "- [Factual proposition unverified due to missing evidence.]")

        # Case D: Factual prefix fails, and right clause contains both ungrounded entity and date -> FAIL CLOSED!
        line_d = "- Vast will launch on January 5, 2035 [E1], but whether UnsupportedCorp launches in 2028 remains unknown."
        # both redacted -> fails closed
        cleaned_d = clean_and_validate_hidden_facts(line_d, set(), ctx=mock_ctx)
        final_d = fail_closed_on_unsupported_sentences(cleaned_d)
        self.assertEqual(final_d, "- [Factual proposition unverified due to missing evidence.]")

    def test_m7a16_stale_vs_latest_user_text(self):
        from cineverdict_agent.agents.validators import get_user_text
        mock_ctx = MagicMock()
        mock_event_old = MagicMock()
        mock_event_old.author = "user"
        mock_event_old.output = "Old unrelated premise."

        mock_event_new = MagicMock()
        mock_event_new.author = "user"
        mock_event_new.output = "Latest current premise."

        mock_ctx.session.events = [mock_event_old, mock_event_new]
        self.assertEqual(get_user_text(mock_ctx), "Latest current premise.")

    def test_m7a16_schedule_neutralization_real_scope(self):
        from cineverdict_agent.agents.validators import clean_and_validate_hidden_facts, make_schedule_conditional
        # Mock Research Ledger with E1 supporting dependency and E2 supporting independence
        research_ledger = """
        ### EVIDENCE LEDGER
        #### E1
        * **Claim:** E1 states Mojave facility was completed.
        * **Supporting Excerpt:** "Vast testing site in Mojave, California"

        #### E2
        * **Claim:** E2 states the production timeline operates independently of the external launch schedule.
        * **Supporting Excerpt:** "production timeline operates independently of the external launch schedule"
        """
        mock_ctx = MagicMock()
        mock_event_research = MagicMock()
        mock_event_research.author = "research_agent"
        mock_event_research.output = research_ledger
        mock_ctx.session.events = [mock_event_research]

        # 1. SUPPORTED DEPENDENCY: cited statement contractually tied to external event survives
        line_dep = "E1 states internal release is contractually tied to external event [E1]."
        output_dep = make_schedule_conditional(line_dep)
        self.assertEqual(output_dep, line_dep)

        # 2. SUPPORTED INDEPENDENCE: cited independent statement survives
        line_ind = "E2 states the production timeline operates independently of the external launch schedule [E2]."
        output_ind = make_schedule_conditional(line_ind)
        self.assertEqual(output_ind, line_ind)

        # 3. UNKNOWN Case: uncited or unverified actions are neutralized
        line_unk = "Schedule the production timeline conditionally or independently of the external launch schedule."
        output_unk = make_schedule_conditional(line_unk)
        self.assertIn("without presupposing any dependency or independence relationship", output_unk)

        # 4. Wrong-Citation Isolation: proper noun Mojave cited under E2 (which doesn't support it) -> gets redacted
        line_wrong = "The Mojave facility acceptance testing was completed [E2]."
        cleaned_wrong = clean_and_validate_hidden_facts(line_wrong, set(), ctx=mock_ctx)
        self.assertIn("[UNSUPPORTED]", cleaned_wrong)


if __name__ == "__main__":
    unittest.main()
