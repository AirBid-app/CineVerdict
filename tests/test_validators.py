"""Tests for CineVerdict Validator Engine.

Regenerated test suite confirming that the Gemini-led NLP validator engine
preserves all required M7A behavioral contracts. It verifies that unsupported
facts fail closed, assumptions are neutralized, evidence boundaries are enforced,
and structural formatting remains robust.
"""

import unittest
from unittest.mock import MagicMock
import os
import io
from unittest.mock import patch

from cineverdict_agent.agents.validators import (
    fail_closed_on_unsupported_sentences,
    neutralize_positive_assumptions,
    neutralize_audience_assumptions,
    neutralize_production_assumptions,
    make_schedule_conditional,
    clean_and_validate_hidden_facts,
    get_allowed_words,
    verdict_before_model_callback,
    market_after_model_callback,
    extract_supporting_excerpts
)
from google.adk.models.llm_response import LlmResponse

class TestValidators(unittest.TestCase):

    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.session = MagicMock()
        self.mock_ctx.session.events = []

    def _add_event(self, author: str, text: str):
        ev = MagicMock()
        ev.author = author
        ev.output = text
        self.mock_ctx.session.events.append(ev)

    def test_fail_closed_preserves_clean_sentences(self):
        """A clean sentence without UNSUPPORTED markers survives unchanged."""
        text = "This is a completely clean and supported sentence."
        res = fail_closed_on_unsupported_sentences(text)
        self.assertEqual(res, text)

    def test_fail_closed_redacts_unsupported_sentences(self):
        """A sentence with an UNSUPPORTED marker fails closed."""
        text = "The location is at [UNSUPPORTED] in California."
        res = fail_closed_on_unsupported_sentences(text)
        self.assertIn("Evidence is insufficient to verify this factual proposition.", res)
        self.assertNotIn("California", res)

    def test_fail_closed_preserves_valid_clause(self):
        """A compound sentence redacts the unsupported clause but preserves the clean grammatical clause."""
        text = "The location is at [UNSUPPORTED], but the launch will happen in 2026."
        res = fail_closed_on_unsupported_sentences(text)
        self.assertIn("The launch will happen in 2026.", res)
        self.assertNotIn("[UNSUPPORTED]", res)

    def test_neutralize_positive_assumptions_audience(self):
        """Assumptions about audience viability are strictly neutralized."""
        text = "We assume that a viable audience exists."
        res = neutralize_positive_assumptions(text)
        self.assertIn("Audience demand remains unverified", res)

    def test_neutralize_positive_assumptions_access(self):
        """Assumptions about production access are strictly neutralized."""
        text = "The assumption is that access to the facility can be coordinated."
        res = neutralize_positive_assumptions(text)
        self.assertIn("Access has not been established and remains unverified.", res)

    def test_neutralize_positive_assumptions_funding(self):
        """Assumptions about funding are strictly neutralized."""
        text = "It is assumed that budget funding is available."
        res = neutralize_positive_assumptions(text)
        self.assertIn("Funding status is unspecified and remains unverified.", res)

    def test_neutralize_positive_assumptions_rights(self):
        """Assumptions about clearance/rights are strictly neutralized."""
        text = "We assume that licensing rights can be obtained."
        res = neutralize_positive_assumptions(text)
        self.assertIn("Rights/authorization remain to be verified.", res)

    def test_make_schedule_conditional_dependency(self):
        """External schedule events do not implicitly dictate internal schedules."""
        text = "The launch impacts the production schedule."
        res = make_schedule_conditional(text)
        self.assertIn("is an external event; determine whether/how it affects the production schedule", res)

    def test_clean_and_validate_hidden_facts(self):
        """Proper nouns not in the allowed evidence map are replaced with [UNSUPPORTED]."""
        self._add_event("director_agent", "Plan to review the documentary.")
        self._add_event("research_agent", "E1 — Claim: ValidCorp approved.\nSupporting Excerpt: ValidCorp approved.")
        allowed = get_allowed_words(self.mock_ctx)

        # ValidCorp is in allowed words, SecretCorp is not.
        text = "ValidCorp approved the launch, but SecretCorp denied it."
        res = clean_and_validate_hidden_facts(text, allowed, self.mock_ctx)
        
        self.assertIn("ValidCorp", res)
        self.assertIn("[UNSUPPORTED]", res)
        self.assertNotIn("SecretCorp", res)

    def test_verdict_before_callback_binds_evidence(self):
        """The verdict agent is constrained to only use E# citations from the active ledger."""
        self._add_event("research_agent", "Here is E1 and E2 and E3.")
        req = MagicMock()
        req.config.system_instruction = "Base instruction."
        
        verdict_before_model_callback(self.mock_ctx, req)
        
        self.assertIn("E1, E2, E3", req.config.system_instruction)
        self.assertIn("DYNAMIC EVIDENCE LEDGER BINDING CONTRACT", req.config.system_instruction)

    def test_market_after_callback_pipeline(self):
        """The ADK after-model callback correctly applies the full validation pipeline."""
        self._add_event("research_agent", "E1 — Claim: ValidCorp approved.\nSupporting Excerpt: ValidCorp approved.")
        
        mock_response = MagicMock(spec=LlmResponse)
        mock_response.text = "ValidCorp approved it [E1].\nBut SecretCorp denied it [E2].\nAssume audience demand exists."
        
        processed = market_after_model_callback(self.mock_ctx, mock_response)
        
        # ValidCorp survives, SecretCorp becomes unsupported, unsupported fails closed, audience is neutralized.
        self.assertIn("ValidCorp", processed.text)
        self.assertNotIn("SecretCorp", processed.text)
        self.assertIn("Evidence is insufficient to verify", processed.text)
        self.assertIn("audience demand remains unverified", processed.text.lower())

    def test_extract_supporting_excerpts(self):
        """Ensures the regex accurately targets excerpts from the research ledger."""
        ledger = "E1 — Claim: X.\nSupporting Excerpt: This is the excerpt.\nE2 — Claim: Y."
        excerpts = extract_supporting_excerpts(ledger)
        # Verify it captures the text. It might include surrounding spaces depending on regex group.
        self.assertTrue(any("This is the excerpt." in ex for ex in excerpts))

    def test_trace_logging(self):
        """Trace logs write to stderr when enabled."""
        os.environ["CINEVERDICT_VALIDATOR_TRACE"] = "1"
        try:
            stderr_capture = io.StringIO()
            with patch('sys.stderr', stderr_capture):
                self._add_event("research_agent", "E1 — Claim: X\nSupporting Excerpt: X")
                allowed = get_allowed_words(self.mock_ctx)
                clean_and_validate_hidden_facts("Testing trace output.", allowed, self.mock_ctx)
            
            self.assertIn("[CINEVERDICT TRACE]", stderr_capture.getvalue())
        finally:
            del os.environ["CINEVERDICT_VALIDATOR_TRACE"]

if __name__ == '__main__':
    unittest.main()
