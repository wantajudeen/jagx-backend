"""Regression tests for JagX AI's tool-routing and truthful response handling."""
import unittest
import os
from unittest.mock import patch
from fastapi import HTTPException

import main


class IntelligenceRoutingTests(unittest.TestCase):
    def test_routes_current_research_to_tools(self):
        self.assertTrue(main.needs_tools("What are the latest AI news today?"))

    def test_routes_image_requests_to_tools(self):
        self.assertTrue(main.needs_tools("Create an image of a red jaguar"))
        self.assertTrue(main.needs_tools("Draw a red jaguar logo"))

    def test_routes_artifact_requests_to_tools(self):
        self.assertTrue(main.needs_tools("Generate a PDF resume for a software engineer"))

    def test_routes_explicit_code_execution_to_tools(self):
        self.assertTrue(main.needs_tools("Run this Python code: print(2 + 2)"))

    def test_explanations_do_not_require_tools(self):
        self.assertFalse(main.needs_tools("Explain recursion in simple terms"))

    def test_provider_names_are_not_rewritten(self):
        reply = "This deployment uses Groq with OpenRouter as a fallback."
        self.assertEqual(main.sanitize_identity(reply), reply)

    def test_capabilities_endpoint_is_callable(self):
        result = main.capabilities()
        self.assertTrue(result["chat"])
        self.assertIn("image_generation", result)

    def test_image_endpoint_returns_a_generation_url_without_extra_secrets(self):
        with patch.dict(os.environ, {"REPLICATE_API_TOKEN": "", "REPLICATE_IMAGE_MODEL": ""}):
            result = main.generate_image(main.MediaRequest(prompt="A red jaguar at night", style="Cinematic"), None)
        self.assertEqual(result["attachment"]["type"], "image")
        self.assertTrue(result["attachment"]["url"].startswith("https://image.pollinations.ai/prompt/"))

    def test_video_endpoint_explains_missing_provider_instead_of_404(self):
        with patch.dict(os.environ, {"REPLICATE_API_TOKEN": "", "REPLICATE_VIDEO_MODEL": ""}):
            with self.assertRaises(HTTPException) as raised:
                main.generate_video(main.MediaRequest(prompt="A red jaguar running"), None)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("No video was generated", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
