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
        mock_event_director.output = "DIRECTOR PLAN - USER Premise: Vast Space film"

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
        # Houston should be redacted, successful neutralized, viable audience neutralized
        self.assertIn("[UNSUPPORTED]", modified_text)
        self.assertNotIn("Houston", modified_text)
        self.assertIn("HYPOTHESIS: an audience may exist", modified_text)
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
        self.assertIn("[UNSUPPORTED]", modified_text_prod)
        self.assertNotIn("Mojave", modified_text_prod)
        self.assertIn("whether coordination with the subject company is possible", modified_text_prod)

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


if __name__ == "__main__":
    unittest.main()
