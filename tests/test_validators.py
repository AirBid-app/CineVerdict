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
    get_allowed_words
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


if __name__ == "__main__":
    unittest.main()
