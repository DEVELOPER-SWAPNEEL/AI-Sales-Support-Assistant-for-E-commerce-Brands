from __future__ import annotations

import unittest

from app.agent import ask
from app.pipeline import main as build_pipeline


class AgentWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        result = build_pipeline()
        if result != 0:
            raise RuntimeError("Pipeline setup failed for tests.")

    def test_retrieval_returns_policy_source(self) -> None:
        answer = ask("What is your return policy?", thread_id="retrieval-test")
        self.assertIn("Return Policy", answer)
        self.assertIn("Confidence: High", answer)
        self.assertIn("Source:", answer)

    def test_memory_persists_user_name(self) -> None:
        ask("My name is Alex", thread_id="memory-test")
        answer = ask("What is my name?", thread_id="memory-test")
        self.assertIn("Alex", answer)

    def test_fallback_is_safe(self) -> None:
        answer = ask("Can you write a poem about mountains?", thread_id="fallback-test")
        self.assertIn("Confidence: Low", answer)
        self.assertIn("General Guidance", answer)

    def test_tool_usage_handles_math(self) -> None:
        answer = ask("calculate 4 * 5", thread_id="tool-test")
        self.assertIn("20", answer)
        self.assertIn("Confidence: Medium", answer)


if __name__ == "__main__":
    unittest.main()
