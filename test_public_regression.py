"""Public, deterministic regression tests using synthetic resume data only."""

import unittest
from unittest.mock import Mock, patch

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
    MODEL_NAME,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_URL,
    OLLAMA_WARMUP_PROMPT,
    OLLAMA_WARMUP_TIMEOUT,
    call_ollama,
    run_agent,
    search_resume,
    warm_up_ollama,
)
from resume_parser import parse_resume


SYNTHETIC_RESUME = """Alex Rivera
alex@example.com
EXPERIENCE
Example Software Inc.
Software Engineer
Built internal APIs using Python and PostgreSQL.
Maintained the deployment dashboard for the engineering team.
SKILLS
Python, React, PostgreSQL, Docker, Flutter
"""

EXPERIENCE_EVIDENCE = (
    "Example Software Inc.\n"
    "Software Engineer\n"
    "Built internal APIs using Python and PostgreSQL.\n"
    "Maintained the deployment dashboard for the engineering team."
)
SKILLS_EVIDENCE = "Python, React, PostgreSQL, Docker, Flutter"
NO_NAME_RESUME = """EDUCATION
Bachelor of Science in Information Systems
Example University
"""


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

    def test_favorite_language_remains_unsupported(self):
        question = "What is the candidate's favorite programming language?"
        result = search_resume(question, SYNTHETIC_RESUME)
        state = AgentState(messages=[
            {"role": "user", "content": question},
            {
                "role": "tool",
                "tool": "search_resume",
                "arguments": {"query": question},
                "result": result,
            },
        ], resume_text=SYNTHETIC_RESUME)
        analysis = analyze_history(state.messages, state.resume_text)
        claim = "Python is the candidate's favorite programming language."

        self.assertIn("favorite", result.get("not_found_terms", []))
        self.assertTrue(answer_has_unsupported_positive_claims(claim, analysis))
        self.assertTrue(answer_contradicts_evidence(claim, analysis))

    def test_flutter_skill_does_not_support_company_relationship(self):
        question = "Did the candidate use Flutter at Example Software Inc.?"
        state = state_with_tool_results(
            question,
            [
                ("experience", EXPERIENCE_EVIDENCE),
                ("skills", SKILLS_EVIDENCE),
            ],
        )
        analysis = analyze_history(state.messages, state.resume_text)
        claim = "The candidate used Flutter at Example Software Inc."

        self.assertTrue(answer_has_unsupported_relationship_claims(claim, analysis))
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


class CandidateNameRoutingTests(unittest.TestCase):
    def assert_name_query_returns_parsed_name(self, resume_text, query):
        parsed_name = parse_resume(resume_text)["candidate_name"]
        self.assertIsNotNone(parsed_name)

        result = search_resume(query, resume_text)

        self.assertIn(f"Candidate name: {parsed_name}", result["results"])
        self.assertNotIn("not_found_terms", result)

    def test_natural_language_candidate_name_queries_use_parsed_identity(self):
        questions = (
            "What is the candidate's name?",
            "What is the candidate's full name?",
            "Who is the candidate?",
            "Can you tell me the candidate's name?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assert_name_query_returns_parsed_name(SYNTHETIC_RESUME, question)

    def test_exact_candidate_name_query_remains_supported_and_generic(self):
        other_resume = SYNTHETIC_RESUME.replace("Alex Rivera", "Casey Morgan")
        for resume_text in (SYNTHETIC_RESUME, other_resume):
            with self.subTest(candidate=parse_resume(resume_text)["candidate_name"]):
                self.assert_name_query_returns_parsed_name(resume_text, "candidate name")

    def test_run_agent_preflights_name_query_without_calling_llm(self):
        parsed_name = parse_resume(SYNTHETIC_RESUME)["candidate_name"]
        with patch("agent_loop.ask_llm_for_decision") as ask_llm:
            outcome = run_agent(
                "What is the candidate's name?",
                SYNTHETIC_RESUME,
            )

        ask_llm.assert_not_called()
        self.assertEqual(outcome["status"], "final")
        self.assertEqual(outcome["iterations"], 1)
        self.assertEqual(outcome["tools_executed"], 1)
        self.assertTrue(outcome["grounding_verified"])
        self.assertIn(parsed_name, outcome["answer"])

    def test_candidate_name_query_fails_closed_when_name_is_absent(self):
        with patch("agent_loop.ask_llm_for_decision") as ask_llm:
            outcome = run_agent(
                "What is the candidate's name?",
                NO_NAME_RESUME,
            )

        ask_llm.assert_not_called()
        self.assertEqual(outcome["tools_executed"], 1)
        self.assertFalse(outcome["grounding_verified"])
        self.assertIn("no evidence", outcome["answer"].lower())

    def test_unrelated_name_question_is_not_routed_to_candidate_identity(self):
        result = search_resume(
            "What does the candidate's name mean for project fit?",
            SYNTHETIC_RESUME,
        )

        self.assertNotIn("Candidate name: Alex Rivera", result["results"])


class OllamaWarmupTests(unittest.TestCase):
    def test_warmup_uses_fixed_minimal_prompt_and_keep_alive(self):
        response = Mock()
        response.json.return_value = {"response": "ready"}
        with patch("agent_loop.requests.post", return_value=response) as post:
            warm_up_ollama()

        args, kwargs = post.call_args
        payload = kwargs["json"]
        self.assertEqual(args[0], OLLAMA_URL)
        self.assertEqual(payload["model"], MODEL_NAME)
        self.assertEqual(payload["prompt"], OLLAMA_WARMUP_PROMPT)
        self.assertNotIn("Alex Rivera", payload["prompt"])
        self.assertNotIn("What responsibilities", payload["prompt"])
        self.assertEqual(payload["options"]["num_predict"], 1)
        self.assertEqual(payload["keep_alive"], OLLAMA_KEEP_ALIVE)
        self.assertEqual(kwargs["timeout"], OLLAMA_WARMUP_TIMEOUT)

    def test_normal_generations_refresh_model_keep_alive(self):
        response = Mock()
        response.json.return_value = {"response": "ready"}
        with patch("agent_loop.requests.post", return_value=response) as post:
            self.assertEqual(call_ollama("synthetic prompt", max_tokens=1), "ready")

        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], OLLAMA_KEEP_ALIVE)

    def test_startup_reports_ready_when_warmup_succeeds(self):
        with patch("api.warm_up_ollama") as warmup:
            with TestClient(api.app) as client:
                health = client.get("/health")

        warmup.assert_called_once_with()
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok", "model_ready": True})

    def test_startup_continues_safely_when_warmup_fails(self):
        sentinel = "synthetic private-data sentinel"
        with patch("api.warm_up_ollama", side_effect=RuntimeError(sentinel)):
            with self.assertLogs("resume_api", level="WARNING") as captured:
                with TestClient(api.app) as client:
                    health = client.get("/health")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok", "model_ready": False})
        self.assertNotIn(sentinel, "\n".join(captured.output))


class UploadPrivacyAndSizeTests(unittest.TestCase):
    def setUp(self):
        self.previous_active_resume = api.get_active_resume_text()
        api.set_active_resume_text("")

    def tearDown(self):
        api.set_active_resume_text(self.previous_active_resume)

    def test_metadata_only_upload_size_limit_and_rejected_upload_state(self):
        pdf_bytes = make_synthetic_pdf()
        self.assertLess(len(pdf_bytes), api.MAX_UPLOAD_SIZE_BYTES)

        with patch("api.warm_up_ollama"), TestClient(api.app) as client:
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
