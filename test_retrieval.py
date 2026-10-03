"""
test_retrieval.py
==================

Tests for the section-aware evidence ranking in agent_loop.search_resume(),
including the GENERIC "entity anchoring" retrieval fix (see
agent_loop.TIER_ENTITY_MATCH / agent_loop._find_entity_anchored_blocks):
when a query names a SPECIFIC entity (a company, employer, etc.) that is
not one of the general CONCEPTS, the whole structured block that entity's
name was found inside is now promoted to the evidence list, instead of
only the one short line the entity's name happened to sit on.

WHY THIS IS A SEPARATE FILE FROM test_resume_parser.py:
    test_resume_parser.py tests resume_parser.py's structured parsing in
    isolation (plain text -> structured dict). This file tests the
    RETRIEVAL layer on top of it: agent_loop.search_resume(), which is
    responsible for deciding what to hand the LLM as evidence, and in what
    order. That is a distinct concern (retrieval/ranking vs. parsing), so
    it gets its own test file, exactly the way the project already keeps
    the parser and the agent loop in separate modules.

WHAT IT CHECKS (mirrors the bug report and the requested behavior):
    1. "What was my capstone project?"      -> project evidence ranks
       ABOVE unrelated internship evidence (the original bug).
    2. "What was my capstone?"              -> same.
    3. "What projects did I work on?"       -> same.
    4. "What was my internship experience?" -> experience is prioritized.
    5. "Where did I study?"                 -> education is prioritized.
    6. "What are my skills?"                -> skills are prioritized.
    7. "What honors did I receive?"         -> honors are prioritized.
    8. A generic query (naming a specific, unrelated technology) still
       uses general/literal retrieval, unaffected by the section-aware
       ranking added for education/project/experience/certification/honors.
    9. A resume with NO projects section does not crash when asked about
       projects or a capstone, and still returns something reasonable.
    10. Existing parser tests (test_resume_parser.py) are unaffected,
        since resume_parser.py itself was NOT modified by this fix - run
        that file directly to confirm (see main() below).
    11. NEW: asking what the candidate did at ONE SPECIFIC, NAMED company
        (e.g. "What did I do at Acme Corp?") returns that company's full
        structured experience block - including the bullet points, not
        just the one line the company name appears on - and does NOT pull
        in the other, unrelated company's block. Proven generically with
        two synthetic (fictional) companies, neither of which is
        hardcoded anywhere in agent_loop.py or resume_parser.py.
    12. NEW: "What companies did I intern at?" still returns evidence for
        BOTH companies (the general "experience" concept still works).
    13. NEW: "What did I do during my internship?" (no company named)
        still returns general internship/experience evidence instead of
        an empty result.
    14. NEW: "What is my capstone project?" still correctly returns the
        project evidence on a resume that also contains multiple,
        differently-named companies (proving the entity-anchoring fix
        does not interfere with concept-based retrieval).

Nothing here is specific to any one person's resume. FICTIONAL_RESUME_TEXT
below is a made-up, generic resume built specifically to reproduce the
reported bug shape: a PROJECT EXPERIENCE section whose real content should
win, and a completely separate INTERNSHIP section that happens to share
several of the "project" concept's generic keywords ("developed", "system",
"designed") - the exact situation that used to let the wrong section's text
outrank (and even bury) the real project evidence.

TWO_COMPANY_RESUME_TEXT (new) is a second, entirely separate fictional
resume built specifically to prove the entity-anchoring fix generalizes to
ANY company name, not just one particular real employer. "Acme Corp" and
"Beta Systems" are placeholder names chosen precisely because they are
generic and interchangeable - the same mechanism would work for any other
company name substituted in their place.

HOW TO RUN:
    python test_retrieval.py

It prints each check as PASS/FAIL and exits with a non-zero status code if
anything failed, so it can be used as a simple CI step alongside
test_resume_parser.py.
"""

import sys

from agent_loop import search_resume


# ============================================================
# FIXTURES
# ============================================================

# A clean, well-structured, entirely fictional resume. Its INTERNSHIP
# section deliberately reuses several of the "project" concept's generic
# keywords ("developed", "system") in a way that has nothing to do with the
# candidate's actual capstone project - this is what used to cause the bug.
FICTIONAL_RESUME_TEXT = """
Jamie Lin
jamie.lin@example.com | (555) 234-5678
Springfield, State

EDUCATION
Bachelor of Science in Computer Science, Example State University
2021 - 2025

HONORS & AWARDS
Dean's List, Example State University, 2023-2024
Magna Cum Laude, Example State University, 2025

PROJECT EXPERIENCE
Campus Marketplace: A Web and Mobile Peer-to-Peer Trading Platform
Example State University
CAPSTONE PROJECT LEAD
Designed the system architecture and implemented the core trading engine.

INTERNSHIP
IT Support Intern
Sample Logistics Inc.
Developed and maintained multiple internal web applications that streamlined company operations and enhanced efficiency, including an Inventory Tracking System, Vendor Management System, and Automated Reporting Portal.
Provided technical support to employees.

SKILLS
Programming languages: Python, JavaScript, SQL
Frameworks and technologies: React, Django
Development tools: Git, Docker

CERTIFICATIONS
AWS Certified Cloud Practitioner
""".strip()

# A resume with NO projects/capstone section at all, used for the
# "must not crash" check.
NO_PROJECTS_RESUME_TEXT = """
Alex Rivera

EDUCATION
Bachelor of Science in Information Systems, Example University

SKILLS
Python, SQL

EXPERIENCE
Support Intern, Example Corp
Assisted with customer support tickets.
""".strip()

# NEW: an entirely separate, fictional resume with TWO NAMED COMPANIES in
# its EXPERIENCE section, used to prove the entity-anchoring fix works for
# ANY company name, generically - nothing about "Acme Corp" or "Beta
# Systems" is referenced anywhere in agent_loop.py or resume_parser.py.
# Each entry has its own distinctive bullet points so a test can check
# that asking about ONE company surfaces ONLY that company's bullets.
TWO_COMPANY_RESUME_TEXT = """
Taylor Morgan
taylor.morgan@example.com | (555) 987-6543
Rivertown, State

EDUCATION
Bachelor of Science in Software Engineering, Example Technical University
2020 - 2024

EXPERIENCE
Software Engineering Intern, Acme Corp, 2022 - 2023
Built automated test suites for the billing platform.
Reduced deployment time by streamlining the CI pipeline.

IT Support Intern, Beta Systems, 2023 - 2024
Provided first-line technical support to end users.
Documented recurring hardware issues for the support team.

PROJECT EXPERIENCE
Weather Dashboard: A Real-Time Weather Visualization Tool
Example Technical University
CAPSTONE PROJECT LEAD
Designed the data pipeline and built the visualization frontend.

SKILLS
Programming languages: Python, Go
Frameworks and technologies: Vue.js, Flask
Development tools: Git, Jenkins

CERTIFICATIONS
CompTIA A+ Certification
""".strip()


class CheckFailure(Exception):
    pass


def check(label, condition):
    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        raise CheckFailure(label)


def _make_safe_check(counter_holder):
    def safe_check(label, condition):
        try:
            check(label, condition)
        except CheckFailure:
            counter_holder[0] += 1
    return safe_check


def _index_of_result_containing(results, substring):
    """Return the index of the first result containing `substring`, or
    None if no result contains it. Used to compare RANKING, not just
    presence."""
    for index, item in enumerate(results):
        if substring in item:
            return index
    return None


# ============================================================
# 1-3. CAPSTONE / PROJECT QUERIES: project evidence must outrank the
#      unrelated internship evidence (the original bug report).
# ============================================================

def run_checks_on_capstone_queries():
    print("=" * 60)
    print("Checks: capstone/project queries rank project evidence first")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    queries = [
        "What was my capstone project?",
        "What was my capstone?",
        "What projects did I work on?",
    ]

    for query in queries:
        result = search_resume(query, resume_text=FICTIONAL_RESUME_TEXT)
        results = result["results"]

        project_index = _index_of_result_containing(results, "Campus Marketplace")
        unrelated_index = _index_of_result_containing(results, "Inventory Tracking System")

        safe_check(
            f"{query!r} -> the real project evidence is present",
            project_index is not None,
        )
        safe_check(
            f"{query!r} -> the unrelated internship evidence is present (not silently dropped)",
            unrelated_index is not None,
        )
        if project_index is not None and unrelated_index is not None:
            safe_check(
                f"{query!r} -> project evidence (index {project_index}) ranks "
                f"ABOVE unrelated internship evidence (index {unrelated_index})",
                project_index < unrelated_index,
            )

    return failures[0]


# ============================================================
# 4. INTERNSHIP / EXPERIENCE QUERY
# ============================================================

def run_checks_on_experience_query():
    print()
    print("=" * 60)
    print("Checks: internship/experience query prioritizes experience")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    result = search_resume("What was my internship experience?", resume_text=FICTIONAL_RESUME_TEXT)
    results = result["results"]

    safe_check("results are non-empty", len(results) > 0)

    experience_index = _index_of_result_containing(results, "Sample Logistics")
    unrelated_index = _index_of_result_containing(results, "Campus Marketplace")

    safe_check(
        "the internship evidence (Sample Logistics) is present",
        experience_index is not None,
    )
    if experience_index is not None:
        safe_check(
            "the internship evidence ranks first (index 0)",
            experience_index == 0,
        )
    if experience_index is not None and unrelated_index is not None:
        safe_check(
            "internship evidence ranks above the unrelated project evidence",
            experience_index < unrelated_index,
        )

    return failures[0]


# ============================================================
# 5. EDUCATION QUERY
# ============================================================

def run_checks_on_education_query():
    print()
    print("=" * 60)
    print("Checks: 'Where did I study?' prioritizes education")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    result = search_resume("Where did I study?", resume_text=FICTIONAL_RESUME_TEXT)
    results = result["results"]

    safe_check("results are non-empty", len(results) > 0)
    safe_check(
        "the top result is about education (Example State University)",
        len(results) > 0 and "Example State University" in results[0],
    )

    return failures[0]


# ============================================================
# 6. SKILLS QUERY
# ============================================================

def run_checks_on_skills_query():
    print()
    print("=" * 60)
    print("Checks: 'What are my skills?' prioritizes skills")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    result = search_resume("What are my skills?", resume_text=FICTIONAL_RESUME_TEXT)
    results = result["results"]

    safe_check("category is 'skills'", result["category"] == "skills")
    safe_check("results are non-empty", len(results) > 0)
    safe_check(
        "programming languages line is present near the top",
        any("Python" in item for item in results[:2]),
    )
    safe_check(
        "unrelated project/internship evidence is NOT present in a skills answer",
        _index_of_result_containing(results, "Campus Marketplace") is None
        and _index_of_result_containing(results, "Sample Logistics") is None,
    )

    return failures[0]


# ============================================================
# 7. HONORS QUERY
# ============================================================

def run_checks_on_honors_query():
    print()
    print("=" * 60)
    print("Checks: 'What honors did I receive?' prioritizes honors")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    result = search_resume("What honors did I receive?", resume_text=FICTIONAL_RESUME_TEXT)
    results = result["results"]

    safe_check("category is 'honors'", result["category"] == "honors")
    safe_check("results are non-empty", len(results) > 0)
    safe_check(
        "the top result contains the honors content (Dean's List / Magna Cum Laude)",
        len(results) > 0 and ("Dean's List" in results[0] or "Magna Cum Laude" in results[0]),
    )
    safe_check(
        "unrelated evidence (project/internship) is not mixed into the honors answer",
        _index_of_result_containing(results, "Campus Marketplace") is None
        and _index_of_result_containing(results, "Sample Logistics") is None,
    )

    return failures[0]


# ============================================================
# 8. GENERIC / LITERAL QUERY STILL WORKS
# ============================================================

def run_checks_on_generic_query():
    print()
    print("=" * 60)
    print("Checks: a generic technology query still uses literal retrieval")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    # "Docker" triggers no section concept at all - it must still be found
    # via the plain literal word search, exactly as before this change.
    result = search_resume("Does the candidate know Docker?", resume_text=FICTIONAL_RESUME_TEXT)
    safe_check(
        "Docker is found via literal/general retrieval",
        any("Docker" in item for item in result["results"]),
    )
    safe_check("category reflects a literal 'search' match", "search" in result["category"])

    # A technology that is NOT in the resume must be correctly reported as
    # not found, not silently hallucinated or matched.
    result_missing = search_resume("Does the candidate know Kubernetes?", resume_text=FICTIONAL_RESUME_TEXT)
    safe_check(
        "an absent technology is reported in not_found_terms",
        "kubernetes" in result_missing.get("not_found_terms", []),
    )

    return failures[0]


# ============================================================
# 9. NO PROJECTS SECTION -> MUST NOT CRASH
# ============================================================

def run_checks_on_missing_projects_section():
    print()
    print("=" * 60)
    print("Checks: asking about projects/capstone on a resume with none")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    try:
        result = search_resume("What was my capstone project?", resume_text=NO_PROJECTS_RESUME_TEXT)
        crashed = False
    except Exception as error:  # noqa: BLE001 - we want to catch anything here
        crashed = True
        result = None
        print(f"  (unexpected exception: {error})")

    safe_check("search_resume() does not raise an exception", not crashed)
    safe_check("a dict with a 'results' list is still returned", isinstance(result, dict) and "results" in result)

    try:
        search_resume("What projects did I work on?", resume_text=NO_PROJECTS_RESUME_TEXT)
        crashed2 = False
    except Exception as error:  # noqa: BLE001
        crashed2 = True
        print(f"  (unexpected exception: {error})")

    safe_check("a second phrasing ('what projects...') also does not crash", not crashed2)

    return failures[0]


# ============================================================
# 11. NEW: COMPANY-SPECIFIC ENTITY-ANCHORED RETRIEVAL
#     (the bug reported: a question about an internship at a named employer
#     only returned the company-name
#     line, not what the candidate actually did there)
#
#     Proven with two SYNTHETIC, made-up companies ("Acme Corp" and "Beta
#     Systems") to demonstrate the fix is fully generic and not hardcoded
#     to any particular real employer.
# ============================================================

def run_checks_on_company_specific_queries():
    print()
    print("=" * 60)
    print("Checks: company-specific queries surface that company's full block")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    # ---- "What did I do at Acme Corp?" -> Acme's bullet points, not just
    #      the company-name line, and NOT Beta Systems' content. ----
    result = search_resume("What did I do at Acme Corp?", resume_text=TWO_COMPANY_RESUME_TEXT)
    results = result["results"]

    safe_check("results are non-empty for 'Acme Corp' query", len(results) > 0)
    safe_check(
        "Acme Corp's own bullet point ('CI pipeline') is present, not just the company-name line",
        any("CI pipeline" in item for item in results),
    )
    safe_check(
        "Beta Systems' unrelated content is NOT pulled into an Acme-specific answer",
        not any("Beta Systems" in item or "first-line technical support" in item for item in results),
    )

    # ---- "What were my responsibilities at Beta Systems?" -> the mirror
    #      image of the check above, proving this isn't hardcoded to one
    #      specific company name. ----
    result2 = search_resume(
        "What were my responsibilities at Beta Systems?", resume_text=TWO_COMPANY_RESUME_TEXT
    )
    results2 = result2["results"]

    safe_check("results are non-empty for 'Beta Systems' query", len(results2) > 0)
    safe_check(
        "Beta Systems' own bullet point ('first-line technical support') is present",
        any("first-line technical support" in item for item in results2),
    )
    safe_check(
        "Acme Corp's unrelated content is NOT pulled into a Beta-specific answer",
        not any("Acme Corp" in item or "CI pipeline" in item for item in results2),
    )

    return failures[0]


# ============================================================
# 12. NEW: "What companies did I intern at?" -> BOTH companies returned
# ============================================================

def run_checks_on_both_companies_query():
    print()
    print("=" * 60)
    print("Checks: 'What companies did I intern at?' still returns both")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    result = search_resume("What companies did I intern at?", resume_text=TWO_COMPANY_RESUME_TEXT)
    results = result["results"]

    safe_check("results are non-empty", len(results) > 0)
    safe_check(
        "Acme Corp's block is present",
        any("Acme Corp" in item for item in results),
    )
    safe_check(
        "Beta Systems' block is present",
        any("Beta Systems" in item for item in results),
    )

    return failures[0]


# ============================================================
# 13. NEW: "What did I do during my internship?" (no company named)
#     -> general internship/experience evidence, not an empty result.
# ============================================================

def run_checks_on_general_internship_query():
    print()
    print("=" * 60)
    print("Checks: 'What did I do during my internship?' returns general evidence")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    result = search_resume("What did I do during my internship?", resume_text=TWO_COMPANY_RESUME_TEXT)
    results = result["results"]

    safe_check(
        "results are non-empty (the query is not silently emptied out)",
        len(results) > 0,
    )
    safe_check(
        "at least one experience entry is present",
        any("Acme Corp" in item or "Beta Systems" in item for item in results),
    )

    return failures[0]


# ============================================================
# 14. NEW: capstone query still works on a resume with multiple companies
# ============================================================

def run_checks_on_capstone_with_multiple_companies():
    print()
    print("=" * 60)
    print("Checks: capstone query unaffected by entity-anchoring on a multi-company resume")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    result = search_resume("What is my capstone project?", resume_text=TWO_COMPANY_RESUME_TEXT)
    results = result["results"]

    safe_check("results are non-empty", len(results) > 0)
    safe_check(
        "the top result is the project evidence (Weather Dashboard)",
        len(results) > 0 and "Weather Dashboard" in results[0],
    )
    safe_check(
        "neither company's experience block outranks the project evidence",
        _index_of_result_containing(results, "Weather Dashboard")
        < (_index_of_result_containing(results, "Acme Corp") or len(results))
        and _index_of_result_containing(results, "Weather Dashboard")
        < (_index_of_result_containing(results, "Beta Systems") or len(results)),
    )

    return failures[0]


def main():
    total_failures = 0
    total_failures += run_checks_on_capstone_queries()
    total_failures += run_checks_on_experience_query()
    total_failures += run_checks_on_education_query()
    total_failures += run_checks_on_skills_query()
    total_failures += run_checks_on_honors_query()
    total_failures += run_checks_on_generic_query()
    total_failures += run_checks_on_missing_projects_section()
    total_failures += run_checks_on_company_specific_queries()
    total_failures += run_checks_on_both_companies_query()
    total_failures += run_checks_on_general_internship_query()
    total_failures += run_checks_on_capstone_with_multiple_companies()

    print()
    print("=" * 60)
    if total_failures == 0:
        print("ALL RETRIEVAL CHECKS PASSED")
    else:
        print(f"{total_failures} CHECK(S) FAILED")
    print("=" * 60)
    print()
    print("NOTE: run 'python test_resume_parser.py' too - resume_parser.py")
    print("was NOT modified by this fix, so those tests should still pass")
    print("unchanged (check #10 from the request).")

    sys.exit(1 if total_failures else 0)


if __name__ == "__main__":
    main()