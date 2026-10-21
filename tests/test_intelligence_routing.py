"""Regression tests for JagX AI's tool-routing and truthful response handling."""
import unittest

import main


class IntelligenceRoutingTests(unittest.TestCase):
    def test_routes_current_research_to_tools(self):
        self.assertTrue(main.needs_tools("What are the latest AI news today?"))

    def test_routes_image_requests_to_tools(self):
        self.assertTrue(main.needs_tools("Create an image of a red jaguar"))

    def test_routes_artifact_requests_to_tools(self):
        self.assertTrue(main.needs_tools("Generate a PDF resume for a software engineer"))

    def test_routes_explicit_code_execution_to_tools(self):
        self.assertTrue(main.needs_tools("Run this Python code: print(2 + 2)"))

    def test_explanations_do_not_require_tools(self):
        self.assertFalse(main.needs_tools("Explain recursion in simple terms"))

    def test_provider_names_are_not_rewritten(self):
        reply = "This deployment uses Groq with OpenRouter as a fallback."
        self.assertEqual(main.sanitize_identity(reply), reply)


if __name__ == "__main__":
    unittest.main()
