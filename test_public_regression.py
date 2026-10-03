"""Public, deterministic regression tests using synthetic resume data only."""

import unittest
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient

import api
from agent_loop import (
    AgentState,
    analyze_history,
    answer_contradicts_evidence,
    answer_has_unsupported_positive_claims,
    answer_has_unsupported_relationship_claims,
    complete_answer_with_relevant_evidence,
    ground_final_answer,
)


SYNTHETIC_RESUME = """Alex Rivera
alex@example.com
EXPERIENCE
Example Software Inc.
Software Engineer
Built internal APIs using Python and PostgreSQL.
Maintained the deployment dashboard for the engineering team.
SKILLS
Python, React, PostgreSQL, Docker
"""

EXPERIENCE_EVIDENCE = (
    "Example Software Inc.\n"
    "Software Engineer\n"
    "Built internal APIs using Python and PostgreSQL.\n"
    "Maintained the deployment dashboard for the engineering team."
)
SKILLS_EVIDENCE = "Python, React, PostgreSQL, Docker"


def state_with_tool_results(question, results, resume_text=SYNTHETIC_RESUME):
    messages = [{"role": "user", "content": question}]
    for category, evidence in results:
        messages.append({
            "role": "tool",
            "tool": "search_resume",
            "arguments": {"query": question},
            "result": {"category": category, "results": [evidence]},
        })
    return AgentState(messages=messages, resume_text=resume_text)


def make_synthetic_pdf():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72),
        SYNTHETIC_RESUME,
        fontsize=11,
    )
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


class GroundingRegressionTests(unittest.TestCase):
    def test_supported_relationship_within_same_experience_entry_is_accepted(self):
        question = "Did the candidate use Python at Example Software Inc.?"
        state = state_with_tool_results(
            question,
            [("experience", EXPERIENCE_EVIDENCE)],
        )
        analysis = analyze_history(state.messages, state.resume_text)
        groups = analysis["evidence_groups"]
        experience_group = next(group for group in groups if group["section"] == "experience")
        claim = "The candidate used Python at Example Software Inc."

        self.assertEqual(experience_group["entry_id"], "experience:0")
        self.assertFalse(answer_has_unsupported_relationship_claims(claim, analysis))
        self.assertFalse(answer_contradicts_evidence(claim, analysis))

    def test_skill_listed_separately_does_not_support_employer_relationship(self):
        question = "Did the candidate use React at Example Software Inc.?"
        state = state_with_tool_results(
            question,
            [
                ("experience", EXPERIENCE_EVIDENCE),
                ("skills", SKILLS_EVIDENCE),
            ],
        )
        analysis = analyze_history(state.messages, state.resume_text)
        group_ids = {group["entry_id"] for group in analysis["evidence_groups"]}
        claim = "The candidate used React at Example Software Inc."

        self.assertIn("experience:0", group_ids)
        self.assertIn("skills:0", group_ids)
        self.assertTrue(answer_has_unsupported_relationship_claims(claim, analysis))
        self.assertTrue(answer_contradicts_evidence(claim, analysis))

    def test_unsupported_technology_claim_is_rejected(self):
        question = "Does Alex know Kotlin?"
        state = AgentState(messages=[
            {"role": "user", "content": question},
            {
                "role": "tool",
                "tool": "search_resume",
                "arguments": {"query": "Kotlin"},
                "result": {
                    "category": "skills",
                    "results": [],
                    "not_found_terms": ["Kotlin"],
                },
            },
        ], resume_text=SYNTHETIC_RESUME)
        analysis = analyze_history(state.messages, state.resume_text)
        claim = "Alex used Kotlin to build mobile apps."

        self.assertTrue(answer_has_unsupported_positive_claims(claim, analysis))
        self.assertTrue(answer_contradicts_evidence(claim, analysis))

    def test_completeness_adds_separately_supported_responsibilities(self):
        question = "What responsibilities did Alex have?"
        state = state_with_tool_results(
            question,
            [("experience", EXPERIENCE_EVIDENCE)],
        )
        partial_answer = "The candidate built internal APIs."

        completed = complete_answer_with_relevant_evidence(partial_answer, state)

        self.assertIn("built internal apis", completed.lower())
        self.assertIn("maintained the deployment dashboard", completed.lower())

    def test_unsafe_relationship_falls_back_to_resume_evidence(self):
        question = "Did Alex use React at Example Software Inc.?"
        state = state_with_tool_results(
            question,
            [
                ("experience", EXPERIENCE_EVIDENCE),
                ("skills", SKILLS_EVIDENCE),
            ],
        )
        unsupported_answer = "The candidate used React at Example Software Inc."

        with patch("agent_loop.ask_llm_for_decision", return_value={
            "action": "final",
            "answer": unsupported_answer,
        }):
            fallback = ground_final_answer(state, unsupported_answer)

        self.assertNotIn(unsupported_answer, fallback)
        self.assertIn("Built internal APIs using Python and PostgreSQL", fallback)
        self.assertIn("React", fallback)
        self.assertIn("does not establish the requested relationship", fallback)


class UploadPrivacyAndSizeTests(unittest.TestCase):
    def setUp(self):
        self.previous_active_resume = api.get_active_resume_text()
        api.set_active_resume_text("")

    def tearDown(self):
        api.set_active_resume_text(self.previous_active_resume)

    def test_metadata_only_upload_size_limit_and_rejected_upload_state(self):
        pdf_bytes = make_synthetic_pdf()
        self.assertLess(len(pdf_bytes), api.MAX_UPLOAD_SIZE_BYTES)

        with TestClient(api.app) as client:
            success = client.post(
                "/upload-resume",
                files={"file": ("synthetic-resume.pdf", pdf_bytes, "application/pdf")},
            )
            self.assertEqual(success.status_code, 200, success.text)
            payload = success.json()
            self.assertEqual(set(payload), {"filename", "pages"})
            self.assertEqual(payload["filename"], "synthetic-resume.pdf")
            self.assertIsInstance(payload["pages"], int)
            self.assertNotIn("raw_text", payload)
            self.assertNotIn("text", payload)

            active_resume = api.get_active_resume_text()
            self.assertTrue(active_resume)

            oversized = client.post(
                "/upload-resume",
                files={
                    "file": (
                        "oversized.pdf",
                        b"x" * (api.MAX_UPLOAD_SIZE_BYTES + 1),
                        "application/pdf",
                    )
                },
            )
            self.assertEqual(oversized.status_code, 413, oversized.text)
            self.assertIn("10 MiB", oversized.json()["detail"])
            self.assertEqual(api.get_active_resume_text(), active_resume)


if __name__ == "__main__":
    unittest.main(verbosity=2)
