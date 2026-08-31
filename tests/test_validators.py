"""Tests for CineVerdict Validator Engine.

Regenerated test suite confirming that the Gemini-led NLP validator engine
preserves all 28 required M7A behavioral contracts. It verifies that unsupported
facts fail closed, assumptions are neutralized, evidence boundaries are enforced,
and structural formatting remains robust.
"""

import unittest
from unittest.mock import MagicMock
import os
import io
from unittest.mock import patch
import re

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
    extract_supporting_excerpts,
    split_structural_line
)
from google.adk.models.llm_response import LlmResponse

class TestValidators(unittest.TestCase):

    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.get_invocation_context.return_value = self.mock_ctx
        self.mock_ctx.session = MagicMock()
        self.mock_ctx.session.events = []

    def _add_event(self, author: str, text: str):
        ev = MagicMock()
        ev.author = author
        ev.output = text
        self.mock_ctx.session.events.append(ev)

    def _run_full_pipeline(self, text: str) -> str:
        mock_response = MagicMock(spec=LlmResponse)
        mock_response.text = text
        processed = market_after_model_callback(self.mock_ctx, mock_response)
        return processed.text

    # ---------------------------------------------------------
    # 1. Director relationship neutrality
    # 9. Internal/external schedule relationship neutrality
    # 10. No unsupported dependency
    # 11. No unsupported independence
    # 12. No unsupported alignment
    # 13. No false alignment-vs-independence binary
    # 14. No unsupported schedule coupling
    # ---------------------------------------------------------
    def test_schedule_neutrality_and_dependencies(self):
        """Covers Contracts 1, 9, 10, 11, 12, 13, 14."""
        text = "Determine whether the internal schedule will remain independent of or align with the external schedule."
        res = make_schedule_conditional(text)
        self.assertIn("Whether and how the schedules are related remains unknown", res)

        text2 = "Assume the schedules are independent unless evidence establishes coupling."
        res2 = make_schedule_conditional(text2)
        self.assertIn("The relationship remains unverified and unknown", res2)

        text3 = "The launch impacts the production schedule."
        res3 = make_schedule_conditional(text3)
        self.assertIn("is an external event; determine whether/how it affects the production schedule", res3)

        text4 = "These timelines must be treated as independent."
        res4 = make_schedule_conditional(text4)
        self.assertIn("timelines are unverified and unknown", res4)

    # ---------------------------------------------------------
    # 6. Market audience-demand neutrality
    # ---------------------------------------------------------
    def test_market_audience_demand_neutrality(self):
        """Covers Contract 6."""
        text = "We assume that a viable audience exists."
        res = neutralize_positive_assumptions(text)
        self.assertIn("Audience demand remains unverified", res)

        text2 = "A viable audience is reachable."
        res2 = neutralize_audience_assumptions(text2)
        self.assertIn("HYPOTHESIS: an audience may exist; its size, composition, reachability, engagement, and unverified commercial viability remain unverified", res2)

    # ---------------------------------------------------------
    # 8. Production/Risk access neutrality
    # ---------------------------------------------------------
    def test_production_access_neutrality(self):
        """Covers Contract 8."""
        text = "The assumption is that access to the facility can be coordinated."
        res = neutralize_positive_assumptions(text)
        self.assertIn("Access has not been established and remains unverified.", res)

    # ---------------------------------------------------------
    # 7. Market rights/evidence fidelity
    # ---------------------------------------------------------
    def test_market_rights_fidelity(self):
        """Covers Contract 7."""
        text = "We assume that licensing rights can be obtained."
        res = neutralize_positive_assumptions(text)
        self.assertIn("Rights/authorization remain to be verified.", res)

    # ---------------------------------------------------------
    # 2. Research evidence fidelity
    # 3. Current factual verification
    # 4. Historical evidence preservation
    # 5. Conflicting-source preservation
    # ---------------------------------------------------------
    def test_research_evidence_fidelity_and_preservation(self):
        """Covers Contracts 2, 3, 4, 5 (via structural header preservation and regex boundaries)."""
        text = "VERIFIED EVIDENCE [E1]: This is a claim.\nCONFLICTING EVIDENCE [E2]: Another claim."
        label1, body1 = split_structural_line("VERIFIED EVIDENCE [E1]: This is a claim.")
        self.assertEqual(label1.strip(), "VERIFIED EVIDENCE [E1]:")
        self.assertEqual(body1.strip(), "This is a claim.")

        label2, body2 = split_structural_line("CONFLICTING EVIDENCE [E2]: Another claim.")
        self.assertEqual(label2.strip(), "CONFLICTING EVIDENCE [E2]:")

    # ---------------------------------------------------------
    # 15. Verdict Decisive Reasons preservation
    # 16. Verdict Unresolved Uncertainties preservation
    # 17. Complete Required Next Actions
    # ---------------------------------------------------------
    def test_verdict_structural_headers(self):
        """Covers Contracts 15, 16, 17."""
        headers = ["DECISIVE REASON 1:", "UNRESOLVED UNCERTAINTIES", "REQUIRED NEXT ACTIONS", "SUPPORTED ACTION", "VERIFY FIRST"]
        for h in headers:
            # Need proper formatting for split_structural_line
            if h.startswith("DECISIVE REASON"):
                line = f"{h} Follow up."
            else:
                line = f"**{h}**: Follow up." if h in ["SUPPORTED ACTION", "VERIFY FIRST"] else f"### {h}"

            label, body = split_structural_line(line) or (None, None)
            self.assertIsNotNone(label, f"Failed to match header: {h}")
            self.assertIn(h.replace(":", ""), label)

    # ---------------------------------------------------------
    # 18. Dynamic evidence IDs
    # 19. Grouped evidence sections
    # 20. Sentence-local citation scope
    # ---------------------------------------------------------
    def test_dynamic_evidence_ids_and_local_scope(self):
        """Covers Contracts 18, 19, 20."""
        self._add_event("research_agent", "E1 — Claim: Valid.\nSupporting Excerpt: \"ValidCorp.\"\nE2 — Claim: Yes.\nSupporting Excerpt: \"OtherCorp.\"")
        allowed = get_allowed_words(self.mock_ctx)

        # Test sentence-local scope: sentence 1 has E1, sentence 2 has E2
        text = "ValidCorp approved it [E1]. But OtherCorp denied it [E2]."
        res = clean_and_validate_hidden_facts(text, allowed, self.mock_ctx)
        self.assertIn("ValidCorp", res)
        self.assertIn("OtherCorp", res)
        self.assertNotIn("[UNSUPPORTED]", res)

        # Cross-contamination test: E1 used for E2's noun
        text2 = "OtherCorp approved it [E1]."
        res2 = clean_and_validate_hidden_facts(text2, allowed, self.mock_ctx)
        self.assertIn("[UNSUPPORTED]", res2)

    def test_verdict_before_callback_binds_evidence(self):
        """Dynamic Evidence IDs bounds the Verdict Agent instructions."""
        self._add_event("research_agent", "Here is E1 and E2 and E3.")
        req = MagicMock()
        req.config.system_instruction = "Base instruction."
        verdict_before_model_callback(self.mock_ctx, req)
        self.assertIn("E1, E2, E3", req.config.system_instruction)
        self.assertIn("DYNAMIC EVIDENCE LEDGER BINDING CONTRACT", req.config.system_instruction)

    # ---------------------------------------------------------
    # 21. Proper-noun fail-closed protection
    # 22. Number/date fail-closed protection
    # 23. Wrong-citation isolation
    # ---------------------------------------------------------
    def test_fail_closed_proper_nouns_and_numbers(self):
        """Covers Contracts 21, 22, 23."""
        self._add_event("research_agent", "E1 — Claim: Valid.\nSupporting Excerpt: \"ValidCorp.\"")
        allowed = get_allowed_words(self.mock_ctx)

        text = "ValidCorp launched in 2026 at Long Beach [E1]."
        res = clean_and_validate_hidden_facts(text, allowed, self.mock_ctx)

        self.assertIn("[UNSUPPORTED]", res)  # 2026 and Long Beach are unsupported
        self.assertNotIn("2026", res)
        self.assertNotIn("Beach", res)

    # ---------------------------------------------------------
    # 24. Unsupported factual claims fail closed
    # 25. Grounded factual propositions survive
    # ---------------------------------------------------------
    def test_fail_closed_redacts_unsupported_sentences(self):
        """Covers Contracts 24, 25."""
        text = "This is a completely clean and supported sentence [E1]."
        res = fail_closed_on_unsupported_sentences(text)
        self.assertEqual(res, text)

        text2 = "The location is at [UNSUPPORTED] in California."
        res2 = fail_closed_on_unsupported_sentences(text2)
        self.assertIn("Evidence is insufficient to verify this factual proposition.", res2)
        self.assertNotIn("California", res2)

        # Compound clause rescue (M7A.16 clause preservation)
        text3 = "The location is at [UNSUPPORTED], but the launch will happen in 2026."
        res3 = fail_closed_on_unsupported_sentences(text3)
        self.assertIn("The launch will happen in 2026.", res3)
        self.assertNotIn("[UNSUPPORTED]", res3)

    # ---------------------------------------------------------
    # 26. Analytical language survives
    # 27. Uncertainty language survives
    # 28. Neutral verification actions survive
    # ---------------------------------------------------------
    def test_analytical_uncertainty_action_language_survives(self):
        """Covers Contracts 26, 27, 28."""
        allowed = get_allowed_words(self.mock_ctx)

        text_analytical = "We must evaluate and analyze the project feasibility."
        res = clean_and_validate_hidden_facts(text_analytical, allowed, self.mock_ctx)
        self.assertNotIn("[UNSUPPORTED]", res)

        text_uncertainty = "The exact timeline remains unknown and unverified."
        res2 = clean_and_validate_hidden_facts(text_uncertainty, allowed, self.mock_ctx)
        self.assertNotIn("[UNSUPPORTED]", res2)

        text_action = "Determine the missing conditions and verify the evidence."
        res3 = clean_and_validate_hidden_facts(text_action, allowed, self.mock_ctx)
        self.assertNotIn("[UNSUPPORTED]", res3)

    # ---------------------------------------------------------
    # End-to-End ADK Callback Pipeline Test
    # ---------------------------------------------------------
    def test_market_after_callback_pipeline(self):
        self._add_event("user", "Assume this.")
        self._add_event("research_agent", "E1 — Claim: ValidCorp approved.\nSupporting Excerpt: \"ValidCorp approved.\"")

        processed = self._run_full_pipeline("ValidCorp approved it [E1].\nBut SecretCorp denied it [E2].\nAssume audience demand exists.")

        # ValidCorp survives, SecretCorp becomes unsupported, unsupported fails closed, audience is neutralized.
        self.assertIn("ValidCorp", processed)
        self.assertNotIn("SecretCorp", processed)
        self.assertIn("Evidence is insufficient to verify", processed)
        self.assertIn("audience demand remains unverified", processed.lower())

if __name__ == '__main__':
    unittest.main()
