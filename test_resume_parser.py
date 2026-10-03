"""
test_resume_parser.py
======================

A small, DETERMINISTIC test / CLI demonstration for:
    * resume_parser.normalize_pdf_text()   (pypdf per-character spacing
      artifact normalization)
    * resume_parser.parse_resume()

WHY THIS FILE IS SEPARATE FROM THE PRODUCTION API:
    Per the project requirements, this test must not require Ollama and
    must not touch api.py. It only imports:
        * resume_parser.normalize_pdf_text   (pure text/regex, no network)
        * resume_parser.parse_resume         (pure text/regex, no network)
    Importing agent_loop.py is safe here: its main() only runs when
    agent_loop.py is executed directly (python agent_loop.py), which this
    file never does.

WHAT IT CHECKS:
    * normalize_pdf_text() correctly undoes pypdf's per-character spacing
      artifact on representative examples (headings, mixed-case words,
      emails, URLs, names)
    * normalize_pdf_text() does NOT damage normal text that merely contains
      short tokens or hyphens ("National University - Manila", "Web &
      Mobile Development", "2024 - 2026")
    * a full, realistic PDF-extracted (character-spaced) resume fixture
      round-trips back to its original clean text through
      normalize_pdf_text()
    * parse_resume() produces the SAME structured result whether it is
      given the character-spaced fixture or the original clean text
      (parse_resume() normalizes internally - see resume_parser.py)
    * candidate name extraction (including a case where the name is not
      in the conventional header position)
    * education extraction
    * skills extraction (and that contact info never leaks into skills)
    * experience extraction
    * experience-entry splitting: a single blank-line-free block that
      contains SEVERAL jobs (each ending in a standalone date-range line)
      is split into exactly one entry per job, with each job's duties
      staying with the correct job, a duplicated trailing date staying
      attached to its entry, inline dates and duty lines never splitting,
      and blank-line-separated entries staying separate
    * that OTHER sections (education, projects) are NOT affected by the
      experience-only splitting rule
    * project extraction, including that a following SUMMARY heading
      never leaks into the preceding project's raw_text
    * certification extraction
    * the new dedicated "honors" section
    * section-heading aliasing (PROJECT EXPERIENCE, INTERNSHIP,
      HONORS & AWARDS, SUMMARY, SKILLS, INTERESTS)
    * a compact, realistic fixture modeled on a messy multi-column PDF
      extraction (contact details landing mid-section, a name landing
      inside a project entry, a summary paragraph appearing before any
      heading) - all with generic, fictional content, never a specific
      real person's resume data

HOW TO RUN:
    python test_resume_parser.py

It prints each check as PASS/FAIL and exits with a non-zero status code if
anything failed, so it can also be used in a simple CI step.
"""

import sys

from resume_parser import normalize_pdf_text, parse_resume


# ----------------------------------------------------------------------
# A second, more fully-headed sample resume, used specifically to exercise
# an explicit "EDUCATION" heading (DEFAULT_RESUME_TEXT does not have one -
# the degree line sits in the header with no heading above it, which is
# also tested separately below).
# ----------------------------------------------------------------------
SAMPLE_RESUME_TEXT = """
Maria Dela Cruz
maria.delacruz@example.com | (0917) 123-4567
San Pablo, Calabarzon
linkedin.com/in/mariadelacruz | github.com/mdelacruz

EDUCATION
Bachelor of Science in Computer Science, University of the Philippines
2021 - 2025

TECHNICAL SKILLS
Programming languages: Python, Java, SQL
Frameworks and technologies: React, Django
Development tools: Git, Docker

WORK EXPERIENCE
Software Engineering Intern, Acme Corp, 2024 - 2025
Built REST APIs used by the mobile team.
Fixed bugs in the billing service.

PROJECTS
Campus Event Finder: a web app that lets students discover campus events.

CERTIFICATIONS
AWS Certified Cloud Practitioner

INTERESTS
Chess, hiking
""".strip()


class CheckFailure(Exception):
    """Raised by check() when an assertion fails, so main() can report it
    and keep going instead of crashing on the first failure."""


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


# ============================================================
# NORMALIZATION CHECKS
# ============================================================

def run_checks_on_normalize_pdf_text_examples():
    print("=" * 60)
    print("Checks on normalize_pdf_text() - individual examples")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    # ---- pypdf's per-character-spacing artifact is undone ----
    safe_check(
        "'E D U C A T I O N' -> 'EDUCATION'",
        normalize_pdf_text("E D U C A T I O N") == "EDUCATION",
    )
    safe_check(
        "'I n f o r m a t i o n  T e c h n o l o g y' -> 'Information Technology'",
        normalize_pdf_text("I n f o r m a t i o n  T e c h n o l o g y")
        == "Information Technology",
    )
    safe_check(
        "'M y S Q L' -> 'MySQL' (internal capitalization preserved)",
        normalize_pdf_text("M y S Q L") == "MySQL",
    )
    safe_check(
        "'J a v a S c r i p t' -> 'JavaScript' (internal capitalization preserved)",
        normalize_pdf_text("J a v a S c r i p t") == "JavaScript",
    )
    safe_check(
        "a character-spaced email address is reconstructed with no stray spaces",
        normalize_pdf_text("s a m p l e . u s e r 1 2 3 @ g m a i l . c o m")
        == "sample.user123@gmail.com",
    )
    safe_check(
        "a character-spaced URL is reconstructed with no stray spaces",
        normalize_pdf_text("H T T P S : / / w w w . l i n k e d i n . c o m")
        == "HTTPS://www.linkedin.com",
    )
    safe_check(
        "a character-spaced name reconstructs word boundaries correctly",
        normalize_pdf_text("J a m i e  R i v e r a") == "Jamie Rivera",
    )
    safe_check(
        "a character-spaced multi-word heading with '&' reconstructs correctly",
        normalize_pdf_text("H O N O R S  &  A W A R D S") == "HONORS & AWARDS",
    )

    # ---- normal text must NOT be damaged ----
    safe_check(
        "'National University - Manila' is left unchanged",
        normalize_pdf_text("National University - Manila")
        == "National University - Manila",
    )
    safe_check(
        "'Web & Mobile Development' is left unchanged",
        normalize_pdf_text("Web & Mobile Development") == "Web & Mobile Development",
    )
    safe_check(
        "'2024 - 2026' is left unchanged",
        normalize_pdf_text("2024 - 2026") == "2024 - 2026",
    )
    safe_check(
        "an ordinary sentence is left unchanged",
        normalize_pdf_text("Built internal tools using React and TypeScript.")
        == "Built internal tools using React and TypeScript.",
    )

    # ---- idempotency: running it twice is the same as running it once ----
    twice_clean = normalize_pdf_text(normalize_pdf_text("E D U C A T I O N"))
    safe_check(
        "normalize_pdf_text() is idempotent on already-normalized text",
        twice_clean == "EDUCATION",
    )
    already_clean = "National University - Manila"
    safe_check(
        "normalize_pdf_text() is idempotent on text that was never spaced",
        normalize_pdf_text(already_clean) == already_clean,
    )

    # ---- edge cases: empty / None-like input never crashes ----
    safe_check("normalize_pdf_text('') returns ''", normalize_pdf_text("") == "")

    return failures[0]


# ============================================================
# ROUND-TRIP + STRUCTURED PARSING ON A REALISTIC PDF-EXTRACTED FIXTURE
# ============================================================

# A normal, clean resume used as the SOURCE OF TRUTH for the fixture below.
# It is intentionally generic/fictional (not tied to any real person) and
# only contains a small, deliberately unremarkable set of qualifications,
# so nothing about a real candidate is hardcoded anywhere in this file.
_CLEAN_SOURCE_FOR_PDF_FIXTURE = """
Ana Reyes
ana.reyes@example.com | (0917) 123-4567
San Pablo, Calabarzon
linkedin.com/in/anareyes | github.com/anareyes

EDUCATION
Bachelor of Science in Information Technology, National University - Manila
2022 - 2026

SKILLS
Programming languages: Python, JavaScript, TypeScript
Frameworks and technologies: React, Firebase
Development tools: MySQL, Git

INTERNSHIP
Software Engineering Intern, Bright Labs, 2025 - 2026
Built internal tools using React and TypeScript.

PROJECT EXPERIENCE
Campus Event Finder: a web app that helps students discover events.

CERTIFICATIONS
Google IT Support Certificate

HONORS & AWARDS
Dean's Lister, 2024
""".strip()


def _char_spacify(text):
    """Simulate pypdf's per-character-spacing artifact on normal text, for
    use as a deterministic test fixture: every word becomes its individual
    characters separated by a single space, and word boundaries become a
    run of two spaces. This is the exact pattern normalize_pdf_text() is
    designed to undo (see resume_parser.py)."""
    lines = text.split("\n")
    spaced_lines = []

    for line in lines:
        if not line.strip():
            spaced_lines.append(line)
            continue

        words = [word for word in line.split(" ") if word != ""]
        spaced_words = [" ".join(list(word)) for word in words]
        spaced_lines.append("  ".join(spaced_words))

    return "\n".join(spaced_lines)


# The realistic PDF-extracted fixture: what pypdf might actually hand back
# for the clean resume above, with the per-character-spacing artifact.
PDF_EXTRACTED_RESUME_TEXT = _char_spacify(_CLEAN_SOURCE_FOR_PDF_FIXTURE)


def run_checks_on_pdf_extracted_fixture():
    print()
    print("=" * 60)
    print("Checks on a realistic PDF-extracted (character-spaced) fixture")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    # ---- the fixture really does exhibit the artifact (sanity check on
    #      the test itself, so a broken fixture can't silently pass) ----
    safe_check(
        "the PDF-extracted fixture actually contains character-spaced text",
        "E D U C A T I O N" in PDF_EXTRACTED_RESUME_TEXT,
    )

    # ---- round-trip: normalizing the spaced fixture recovers the original
    #      clean source text EXACTLY ----
    recovered = normalize_pdf_text(PDF_EXTRACTED_RESUME_TEXT)
    safe_check(
        "normalize_pdf_text() recovers the original clean resume text exactly",
        recovered == _CLEAN_SOURCE_FOR_PDF_FIXTURE,
    )

    # ---- parse_resume() gives the SAME structured result whether given the
    #      raw character-spaced fixture or the already-clean text (it
    #      normalizes internally - see resume_parser.parse_resume) ----
    parsed_from_spaced = parse_resume(PDF_EXTRACTED_RESUME_TEXT)
    parsed_from_clean = parse_resume(_CLEAN_SOURCE_FOR_PDF_FIXTURE)
    safe_check(
        "parse_resume(spaced_text) == parse_resume(clean_text)",
        parsed_from_spaced == parsed_from_clean,
    )

    # ---- spot-check the structured fields, using ONLY what is explicitly
    #      present in the fixture above - nothing invented or assumed ----
    safe_check(
        "candidate name extracted correctly from the spaced fixture",
        parsed_from_spaced["candidate_name"] == "Ana Reyes",
    )
    safe_check(
        "email extracted correctly from the spaced fixture",
        parsed_from_spaced["contact"].get("email") == "ana.reyes@example.com",
    )
    safe_check(
        "phone extracted correctly from the spaced fixture",
        "123-4567" in parsed_from_spaced["contact"].get("phone", ""),
    )

    education_text = " ".join(entry["raw_text"] for entry in parsed_from_spaced["education"])
    safe_check(
        "education section detected in the spaced fixture",
        len(parsed_from_spaced["education"]) >= 1,
    )
    safe_check(
        "education content reconstructed with normal spacing (dash preserved)",
        "National University - Manila" in education_text,
    )

    safe_check(
        "skills items reconstructed correctly (JavaScript, TypeScript, MySQL)",
        {"JavaScript", "TypeScript", "MySQL"}.issubset(
            set(parsed_from_spaced["skills"]["items"])
        ),
    )

    experience_text = " ".join(entry["raw_text"] for entry in parsed_from_spaced["experience"])
    safe_check(
        "experience (INTERNSHIP heading) detected in the spaced fixture",
        len(parsed_from_spaced["experience"]) >= 1,
    )
    safe_check(
        "experience content reconstructed correctly",
        "Bright Labs" in experience_text,
    )

    projects_text = " ".join(entry["raw_text"] for entry in parsed_from_spaced["projects"])
    safe_check(
        "projects (PROJECT EXPERIENCE heading) detected in the spaced fixture",
        len(parsed_from_spaced["projects"]) >= 1,
    )
    safe_check(
        "project content reconstructed correctly",
        "Campus Event Finder" in projects_text,
    )

    certifications_text = " ".join(
        entry["raw_text"] for entry in parsed_from_spaced["certifications"]
    )
    safe_check(
        "certifications detected in the spaced fixture",
        len(parsed_from_spaced["certifications"]) >= 1,
    )
    safe_check(
        "certification content reconstructed correctly",
        "Google IT Support Certificate" in certifications_text,
    )

    # ---- HONORS & AWARDS now has its OWN dedicated top-level field ----
    honors_text = " ".join(entry["raw_text"] for entry in parsed_from_spaced["honors"])
    safe_check(
        "HONORS & AWARDS heading produces a dedicated 'honors' entry, reconstructed correctly",
        "Dean's Lister" in honors_text,
    )
    safe_check(
        "honors content is NOT filed under other_sections",
        "honors" not in parsed_from_spaced["other_sections"],
    )

    return failures[0]


# ============================================================
# EXISTING STRUCTURED PARSER CHECKS (unchanged from before)
# ============================================================

def run_checks_on_sample_resume():
    print()
    print("=" * 60)
    print("Checks on SAMPLE_RESUME_TEXT (has explicit section headings)")
    print("=" * 60)

    data = parse_resume(SAMPLE_RESUME_TEXT)
    failures = [0]
    safe_check = _make_safe_check(failures)

    # ---- candidate name ----
    safe_check(
        "candidate name extracted correctly",
        data["candidate_name"] == "Maria Dela Cruz",
    )

    # ---- contact info ----
    safe_check("email extracted", data["contact"].get("email") == "maria.delacruz@example.com")
    safe_check("phone extracted", "123-4567" in data["contact"].get("phone", ""))
    safe_check("linkedin extracted", "linkedin.com/in/mariadelacruz" in data["contact"].get("linkedin", ""))
    safe_check("github extracted", "github.com/mdelacruz" in data["contact"].get("github", ""))

    # ---- education ----
    education_text = " ".join(entry["raw_text"] for entry in data["education"])
    safe_check("education section detected", len(data["education"]) >= 1)
    safe_check(
        "education content preserved verbatim",
        "University of the Philippines" in education_text,
    )

    # ---- skills ----
    safe_check("skills raw_text preserved", "Python" in data["skills"]["raw_text"])
    safe_check(
        "skills split into separable items",
        "Python" in data["skills"]["items"] and "Django" in data["skills"]["items"],
    )

    # ---- experience ----
    experience_text = " ".join(entry["raw_text"] for entry in data["experience"])
    safe_check("experience section detected", len(data["experience"]) >= 1)
    safe_check("experience content preserved verbatim", "Acme Corp" in experience_text)
    safe_check(
        "a single job with only an INLINE date stays exactly one entry",
        len(data["experience"]) == 1,
    )

    # ---- projects ----
    projects_text = " ".join(entry["raw_text"] for entry in data["projects"])
    safe_check("project section detected", len(data["projects"]) >= 1)
    safe_check("project content preserved verbatim", "Campus Event Finder" in projects_text)

    # ---- certifications ----
    certifications_text = " ".join(entry["raw_text"] for entry in data["certifications"])
    safe_check("certification section detected", len(data["certifications"]) >= 1)
    safe_check(
        "certification content preserved verbatim",
        "AWS Certified Cloud Practitioner" in certifications_text,
    )

    # ---- unknown/unrecognized-but-detected section (Interests) ----
    safe_check(
        "INTERESTS heading filed under other_sections (not discarded)",
        "chess" in data["other_sections"].get("interests", "").lower(),
    )

    # ---- Interests must never contain honors, and honors must be empty
    #      here since this resume has no HONORS heading ----
    safe_check(
        "no honors entries fabricated when there is no honors heading",
        data["honors"] == [],
    )

    # ---- nothing invented ----
    safe_check(
        "no fabricated location text",
        data["contact"].get("location", "") in ("", "San Pablo, Calabarzon"),
    )

    return failures[0]


def run_checks_on_generic_skill_fixture():
    print()
    print("=" * 60)
    print("Checks on generic skill parsing, including parenthetical commas")
    print("=" * 60)

    data = parse_resume("""
SKILLS
Database Management (AlphaOne, BetaTwo, GammaThree)
Programming languages: ExampleLang, SampleScript
""".strip())
    failures = [0]
    safe_check = _make_safe_check(failures)

    safe_check(
        "commas inside parentheses stay within one skill item",
        "Database Management (AlphaOne, BetaTwo, GammaThree)" in data["skills"]["items"],
    )
    safe_check(
        "items outside parentheses still split at top-level commas",
        data["skills"]["items"][-2:] == ["ExampleLang", "SampleScript"],
    )

    return failures[0]


def run_checks_on_empty_and_headerless_text():
    print()
    print("=" * 60)
    print("Edge cases: empty text and text with no recognizable sections")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    empty_result = parse_resume("")
    safe_check("empty text -> candidate_name is None", empty_result["candidate_name"] is None)
    safe_check("empty text -> no education entries", empty_result["education"] == [])
    safe_check("empty text -> no fabricated contact info", empty_result["contact"] == {})
    safe_check("empty text -> no fabricated honors", empty_result["honors"] == [])

    no_sections_text = "Just a single unstructured paragraph with no headings at all."
    no_sections_result = parse_resume(no_sections_text)
    safe_check(
        "text with no headings -> nothing invented for candidate_name",
        no_sections_result["candidate_name"] is None,
    )
    safe_check(
        "text with no headings -> content preserved under other_sections, not discarded",
        no_sections_text in no_sections_result["other_sections"].get("header", ""),
    )

    return failures[0]


# ============================================================
# NEW: SECTION-HEADING ALIAS CHECKS
# ============================================================

def run_checks_on_heading_aliases():
    print()
    print("=" * 60)
    print("Checks on section-heading aliases")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    data = parse_resume("PROJECT EXPERIENCE\nSample generic project entry text.")
    projects_text = " ".join(entry["raw_text"] for entry in data["projects"])
    safe_check(
        "'PROJECT EXPERIENCE' maps to the projects section",
        "Sample generic project entry text." in projects_text,
    )

    data = parse_resume("INTERNSHIP\nSample generic internship entry text.")
    experience_text = " ".join(entry["raw_text"] for entry in data["experience"])
    safe_check(
        "'INTERNSHIP' maps to the experience section",
        "Sample generic internship entry text." in experience_text,
    )

    data = parse_resume("HONORS & AWARDS\nSample generic honor line.")
    honors_text = " ".join(entry["raw_text"] for entry in data["honors"])
    safe_check(
        "'HONORS & AWARDS' maps to the honors section",
        "Sample generic honor line." in honors_text,
    )

    data = parse_resume("SUMMARY\nA short generic professional summary sentence.")
    safe_check(
        "'SUMMARY' maps to the summary section",
        "A short generic professional summary sentence." in data["summary"]["raw_text"],
    )

    data = parse_resume("SKILLS\nPython, SQL, Communication")
    safe_check(
        "'SKILLS' maps to the skills section",
        "Python" in data["skills"]["items"],
    )

    data = parse_resume("INTERESTS\nReading, cycling")
    safe_check(
        "'INTERESTS' maps to other_sections['interests']",
        "cycling" in data["other_sections"].get("interests", "").lower(),
    )

    return failures[0]


# ============================================================
# NEW: SUMMARY MUST TERMINATE A PRECEDING PROJECT SECTION
# ============================================================

def run_checks_on_project_summary_boundary():
    print()
    print("=" * 60)
    print("Checks on the SUMMARY-terminates-PROJECT-EXPERIENCE boundary")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    text = """
PROJECT EXPERIENCE
Example Project
Example University
CAPSTONE PROJECT MANAGER
John Doe
SUMMARY
Some summary text
""".strip()

    data = parse_resume(text)
    projects_text = " ".join(entry["raw_text"] for entry in data["projects"])

    safe_check(
        "the project entry contains its own content",
        "Example Project" in projects_text and "CAPSTONE PROJECT MANAGER" in projects_text,
    )
    safe_check(
        "the project raw_text does NOT contain 'SUMMARY'",
        "SUMMARY" not in projects_text,
    )
    safe_check(
        "the summary section correctly picks up the text after the heading",
        "Some summary text" in data["summary"]["raw_text"],
    )
    safe_check(
        "the summary section does not swallow the project content",
        "Example Project" not in data["summary"]["raw_text"],
    )

    return failures[0]


# ============================================================
# NEW: GENERIC CANDIDATE-NAME DETECTION
# ============================================================

def run_checks_on_generic_candidate_name():
    print()
    print("=" * 60)
    print("Checks on generic (non-hardcoded) candidate-name detection")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    # A plausible standalone full name in the conventional header position.
    header_case = "Alex Rivera\nSUMMARY\nA generic summary sentence goes here."
    data = parse_resume(header_case)
    safe_check(
        "a standalone header name is detected generically",
        data["candidate_name"] == "Alex Rivera",
    )

    # A name that ends up INSIDE a project entry due to a scrambled PDF
    # reading order - the same shape of problem seen in real multi-column
    # PDF extractions - with no heading-position header name at all.
    scrambled_case = """
PROJECT EXPERIENCE
Example Project
Example University
CAPSTONE PROJECT MANAGER
Jordan Blake
SUMMARY
Some summary text
""".strip()
    data = parse_resume(scrambled_case)
    safe_check(
        "a name found inside a project entry (not the header) is still detected",
        data["candidate_name"] == "Jordan Blake",
    )

    # Company, school, and job-title lines must never be mistaken for a name.
    safe_check(
        "'Example University' is not mistaken for a name",
        data["candidate_name"] != "Example University",
    )
    safe_check(
        "'CAPSTONE PROJECT MANAGER' (a job title) is not mistaken for a name",
        data["candidate_name"] != "CAPSTONE PROJECT MANAGER",
    )

    org_only_case = "Bright Labs Incorporated\nEDUCATION\nSome degree line here."
    data = parse_resume(org_only_case)
    safe_check(
        "a company name is never mistaken for a candidate name",
        data["candidate_name"] is None,
    )

    return failures[0]


# ============================================================
# NEW: CONTACT INFO MUST BE SEPARATED FROM SKILLS
# ============================================================

def run_checks_on_contact_vs_skills():
    print()
    print("=" * 60)
    print("Checks on contact extraction being separated from skills")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    # Contact details deliberately placed INSIDE the skills section body,
    # in the pipe-separated style commonly produced by PDF extraction.
    text = """
SKILLS
555-000-1111 | Example City, Example Region |
someone@example.com |
https://www.linkedin.com/in/example-person/
Technical: Python, JavaScript, SQL
""".strip()

    data = parse_resume(text)

    safe_check("phone extracted into contact", "1111" in data["contact"].get("phone", ""))
    safe_check("email extracted into contact", data["contact"].get("email") == "someone@example.com")
    safe_check(
        "linkedin extracted into contact",
        "linkedin.com/in/example-person" in data["contact"].get("linkedin", ""),
    )
    safe_check(
        "location extracted into contact",
        data["contact"].get("location") == "Example City, Example Region",
    )

    safe_check(
        "skills raw_text does not contain the email",
        "someone@example.com" not in data["skills"]["raw_text"],
    )
    safe_check(
        "skills raw_text does not contain the phone number",
        "555-000-1111" not in data["skills"]["raw_text"],
    )
    safe_check(
        "skills raw_text does not contain the LinkedIn URL",
        "linkedin.com" not in data["skills"]["raw_text"],
    )
    safe_check(
        "skills items still contain the actual skills",
        {"Python", "JavaScript", "SQL"}.issubset(set(data["skills"]["items"])),
    )

    return failures[0]


# ============================================================
# NEW: HONORS SECTION
# ============================================================

def run_checks_on_honors_section():
    print()
    print("=" * 60)
    print("Checks on the dedicated honors section")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    text = """
HONORS & AWARDS
Dean's Lister
Magna Cum Laude
""".strip()

    data = parse_resume(text)
    honors_text = " ".join(entry["raw_text"] for entry in data["honors"])

    safe_check("honors section is populated", len(data["honors"]) >= 1)
    safe_check("'Dean's Lister' is captured under honors", "Dean's Lister" in honors_text)
    safe_check("'Magna Cum Laude' is captured under honors", "Magna Cum Laude" in honors_text)
    safe_check(
        "honors content is not placed in other_sections['interests']",
        "Dean's Lister" not in data["other_sections"].get("interests", ""),
    )
    safe_check(
        "honors content is not placed in other_sections at all",
        "honors" not in data["other_sections"],
    )

    return failures[0]


# ============================================================
# NEW: INTERESTS SECTION MUST NOT CONTAIN HONORS
# ============================================================

def run_checks_on_interests_only():
    print()
    print("=" * 60)
    print("Checks that INTERESTS contains only interests")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    text = """
INTERESTS
Prompt Engineering
Generative AI
""".strip()

    data = parse_resume(text)
    interests_text = data["other_sections"].get("interests", "")

    safe_check("'Prompt Engineering' is captured under interests", "Prompt Engineering" in interests_text)
    safe_check("'Generative AI' is captured under interests", "Generative AI" in interests_text)
    safe_check(
        "no honors entries are fabricated from an interests-only resume",
        data["honors"] == [],
    )

    return failures[0]


# ============================================================
# NEW: COMPACT REALISTIC FIXTURE (messy multi-column PDF structure)
# ============================================================
# Modeled on the STRUCTURE of a real multi-column PDF extraction (a
# professional-summary paragraph appearing before any heading, a name
# line landing inside a project entry rather than the header, and contact
# details landing mid-document inside another section) - but every
# name, company, and school below is generic/fictional. Nothing about any
# real person's resume is hardcoded here or in resume_parser.py.

REALISTIC_FIXTURE_TEXT = """
Information technology professional with hands-on experience across several
areas of software development, testing, and support.
EDUCATION
Bachelor of Science in Sample Technology
Sample University - Somewhere 2022 - 2026
HONORS & AWARDS
Dean's Lister, Sample University
Magna Cum Laude, Sample University
PROJECT EXPERIENCE
Sample Company: A Sample Capstone Project
Sample University - Somewhere
LEAD DEVELOPER
Jamie Rivera
SUMMARY
SKILLS
555-000-1111 | Sampleton, Someplace |
jamie.rivera@example.com |
https://www.linkedin.com/in/jamierivera/
Technical: Python, JavaScript, SQL
INTERESTS
Reading
Hiking
INTERNSHIP
IT Intern
Sample Corp - Somewhere
Did sample internship work.
""".strip()


def run_checks_on_realistic_fixture():
    print()
    print("=" * 60)
    print("Checks on a compact, realistic (messy-structure) fixture")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    data = parse_resume(REALISTIC_FIXTURE_TEXT)

    # ---- name found inside the project entry, not the header ----
    safe_check(
        "candidate name found even though it sits inside a project entry",
        data["candidate_name"] == "Jamie Rivera",
    )

    # ---- project boundary: SUMMARY heading must not leak in ----
    projects_text = " ".join(entry["raw_text"] for entry in data["projects"])
    safe_check(
        "project content preserved",
        "Sample Capstone Project" in projects_text and "LEAD DEVELOPER" in projects_text,
    )
    safe_check(
        "'SUMMARY' does not leak into the project raw_text",
        "SUMMARY" not in projects_text,
    )

    # ---- contact extracted correctly despite being mid-document ----
    safe_check("phone extracted", "1111" in data["contact"].get("phone", ""))
    safe_check("email extracted", data["contact"].get("email") == "jamie.rivera@example.com")
    safe_check(
        "linkedin extracted",
        "linkedin.com/in/jamierivera" in data["contact"].get("linkedin", ""),
    )
    safe_check(
        "location extracted",
        data["contact"].get("location") == "Sampleton, Someplace",
    )

    # ---- skills contain only skills, not the contact block ----
    safe_check(
        "skills items contain the real skills",
        {"Python", "JavaScript", "SQL"}.issubset(set(data["skills"]["items"])),
    )
    safe_check(
        "skills raw_text excludes the phone/email/linkedin block",
        "@" not in data["skills"]["raw_text"] and "555-000-1111" not in data["skills"]["raw_text"],
    )

    # ---- honors populated correctly ----
    honors_text = " ".join(entry["raw_text"] for entry in data["honors"])
    safe_check("honors section populated", "Dean's Lister" in honors_text)

    # ---- interests isolated ----
    interests_text = data["other_sections"].get("interests", "")
    safe_check(
        "interests contains only the actual interests",
        "Reading" in interests_text and "Hiking" in interests_text,
    )

    # ---- experience (INTERNSHIP heading) ----
    experience_text = " ".join(entry["raw_text"] for entry in data["experience"])
    safe_check(
        "INTERNSHIP heading recognized as experience",
        "Did sample internship work." in experience_text,
    )

    # ---- education ----
    education_text = " ".join(entry["raw_text"] for entry in data["education"])
    safe_check(
        "education section detected",
        "Bachelor of Science in Sample Technology" in education_text,
    )

    return failures[0]


# ============================================================
# NEW: ONE BLANK-LINE-FREE EXPERIENCE BLOCK CONTAINING SEVERAL JOBS
# ============================================================
# Reproduces the exact STRUCTURE of the failing real-PDF case, using only
# fictional companies. The whole INTERNSHIP section arrives as ONE block
# (no blank lines), with:
#   Entry 1 -> title, company, duties, "2024 - 2026"
#   Entry 2 -> company, title, duties, "2025-2026", "2025-2026" (duplicate)
# Note the two entries deliberately use DIFFERENT line orders (title-first
# vs company-first) and the second ends with a DUPLICATED date, because
# that is what the real extraction produced.

MULTI_ENTRY_EXPERIENCE_TEXT = """
INTERNSHIP
IT & Digital Marketing Intern
Northwind Logistics Corporation - Riverside, Metro City
Developed and maintained several internal web applications.
Conducted testing and quality assurance on new releases.
2024 - 2026
Bluepeak Systems Incorporated - Riverside, Metro City
IT & Digital Marketing Intern
Designed and refined a structured recruitment process flow.
Developed and managed job vacancy postings.
2025-2026
2025-2026
""".strip()


def _experience_texts(data):
    return [entry["raw_text"] for entry in data["experience"]]


def run_checks_on_multi_entry_experience():
    print()
    print("=" * 60)
    print("Checks on splitting one experience block into several entries")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    # ------------------------------------------------------------
    # The exact structure from the real failing PDF (fictional names)
    # ------------------------------------------------------------
    data = parse_resume(MULTI_ENTRY_EXPERIENCE_TEXT)
    entries = _experience_texts(data)

    safe_check(
        "exactly TWO experience entries are produced",
        len(entries) == 2,
    )

    if len(entries) == 2:
        first, second = entries
        first_lines = first.splitlines()
        second_lines = second.splitlines()

        # ---- Entry 1: title -> company -> duties -> 2024 - 2026 ----
        safe_check(
            "entry 1 starts with the job title line",
            first_lines[0] == "IT & Digital Marketing Intern",
        )
        safe_check(
            "entry 1 contains its own company line",
            "Northwind Logistics Corporation - Riverside, Metro City" in first_lines,
        )
        safe_check(
            "entry 1 contains ALL of its duty lines",
            "Developed and maintained several internal web applications." in first_lines
            and "Conducted testing and quality assurance on new releases." in first_lines,
        )
        safe_check(
            "entry 1 ends with its own date range '2024 - 2026'",
            first_lines[-1] == "2024 - 2026",
        )
        safe_check(
            "entry 1 does NOT contain the second company or its duties",
            "Bluepeak" not in first
            and "recruitment process flow" not in first
            and "job vacancy" not in first,
        )
        safe_check(
            "entry 1 does NOT contain the second entry's dates",
            "2025-2026" not in first,
        )

        # ---- Entry 2: company -> title -> duties -> 2025-2026 x2 ----
        safe_check(
            "entry 2 starts with the second company line",
            second_lines[0] == "Bluepeak Systems Incorporated - Riverside, Metro City",
        )
        safe_check(
            "entry 2 contains its job title line",
            second_lines[1] == "IT & Digital Marketing Intern",
        )
        safe_check(
            "entry 2 contains ALL of its duty lines",
            "Designed and refined a structured recruitment process flow." in second_lines
            and "Developed and managed job vacancy postings." in second_lines,
        )
        safe_check(
            "entry 2 keeps BOTH duplicate trailing '2025-2026' lines",
            second_lines[-2:] == ["2025-2026", "2025-2026"],
        )
        safe_check(
            "entry 2 does NOT contain the first company or its duties",
            "Northwind" not in second
            and "internal web applications" not in second
            and "quality assurance" not in second,
        )
        safe_check(
            "entry 2 does NOT contain the first entry's date",
            "2024 - 2026" not in second,
        )

    safe_check(
        "no entry consists only of a date range (no date-only third entry)",
        all(
            any(not line.strip()[:1].isdigit() for line in entry.splitlines())
            for entry in entries
        ),
    )

    # ------------------------------------------------------------
    # Same structure, but as raw pypdf character-spaced output: the
    # parse must be identical (normalization runs first).
    # ------------------------------------------------------------
    spaced_data = parse_resume(_char_spacify(MULTI_ENTRY_EXPERIENCE_TEXT))
    safe_check(
        "the character-spaced version yields the same two experience entries",
        _experience_texts(spaced_data) == entries,
    )

    # ------------------------------------------------------------
    # Nothing is lost: joining the entries reproduces the section body
    # ------------------------------------------------------------
    original_body_lines = MULTI_ENTRY_EXPERIENCE_TEXT.splitlines()[1:]
    rejoined_lines = "\n".join(entries).splitlines()
    safe_check(
        "splitting drops and reorders nothing (all original lines preserved in order)",
        rejoined_lines == original_body_lines,
    )

    # ------------------------------------------------------------
    # Inline dates and duty lines must NOT split
    # ------------------------------------------------------------
    inline_text = """
WORK EXPERIENCE
Software Engineering Intern, Acme Corp, 2024 - 2025
Built REST APIs used by the mobile team.
Migrated the reporting module between 2024 - 2025 without downtime.
* 2024 - 2025 Shipped the billing dashboard.
Fixed bugs in the billing service.
""".strip()
    inline_entries = _experience_texts(parse_resume(inline_text))
    safe_check(
        "inline dates and bulleted/duty lines never split an entry",
        len(inline_entries) == 1,
    )

    # ------------------------------------------------------------
    # Blank-line-separated entries are still preserved
    # ------------------------------------------------------------
    blank_separated_text = """
EXPERIENCE
Junior Analyst, First Co
Prepared weekly reports.
2023 - 2024

Senior Analyst, Second Co
Led the reporting team.
2024 - 2025
""".strip()
    blank_entries = _experience_texts(parse_resume(blank_separated_text))
    safe_check(
        "blank-line-separated experience entries stay separate (2 entries)",
        len(blank_entries) == 2
        and "First Co" in blank_entries[0]
        and "Second Co" in blank_entries[1]
        and "Second Co" not in blank_entries[0],
    )

    # A blank-line-separated block that itself contains two jobs: the
    # blank-line split AND the date-boundary split both apply -> 3 entries.
    mixed_text = """
EXPERIENCE
Junior Analyst, First Co
Prepared weekly reports.
2023 - 2024

Alpha Company - City One
Team Lead
Ran the daily standup.
2024 - 2025
Beta Company - City Two
Team Lead
Managed the roadmap.
2025 - Present
""".strip()
    mixed_entries = _experience_texts(parse_resume(mixed_text))
    safe_check(
        "blank-line split and date-boundary split combine correctly (3 entries)",
        len(mixed_entries) == 3
        and "First Co" in mixed_entries[0]
        and "Alpha Company" in mixed_entries[1]
        and "Ran the daily standup." in mixed_entries[1]
        and "Beta Company" in mixed_entries[2]
        and "Managed the roadmap." in mixed_entries[2],
    )

    # ------------------------------------------------------------
    # A duplicated date separated by a BLANK line must not become a
    # date-only entry either
    # ------------------------------------------------------------
    duplicate_after_blank_text = """
EXPERIENCE
Sample Role, Sample Co
Did sample work.
2025-2026

2025-2026
""".strip()
    duplicate_entries = _experience_texts(parse_resume(duplicate_after_blank_text))
    safe_check(
        "a duplicated date after a blank line is attached, not made its own entry",
        len(duplicate_entries) == 1
        and duplicate_entries[0].splitlines()[-2:] == ["2025-2026", "2025-2026"],
    )

    # ------------------------------------------------------------
    # A single job whose block ends with a date: exactly one entry
    # (a trailing date with nothing after it never creates another)
    # ------------------------------------------------------------
    single_trailing_date_text = "EXPERIENCE\nSample Role, Sample Co\nDid sample work.\n2024 - 2026"
    single_entries = _experience_texts(parse_resume(single_trailing_date_text))
    safe_check(
        "a single job ending in a standalone date stays exactly one entry",
        len(single_entries) == 1 and single_entries[0].endswith("2024 - 2026"),
    )

    # ------------------------------------------------------------
    # Date-range shape variants are recognized as boundaries
    # ------------------------------------------------------------
    variant_text = """
EXPERIENCE
Role One
Company One
Jan 2023 - Mar 2024
Role Two
Company Two
2024 \u2013 Present
Role Three
Company Three
2022 to 2023
""".strip()
    variant_entries = _experience_texts(parse_resume(variant_text))
    safe_check(
        "month-name, en-dash/Present, and 'to' date ranges all act as boundaries",
        len(variant_entries) == 3
        and "Company One" in variant_entries[0]
        and "Company Two" in variant_entries[1]
        and "Company Three" in variant_entries[2],
    )

    # ------------------------------------------------------------
    # A block that BEGINS with a date is ambiguous -> left whole
    # ------------------------------------------------------------
    date_first_text = """
EXPERIENCE
2024 - 2026
Company X
Did work at X.
2025 - 2026
Company Y
Did work at Y.
""".strip()
    date_first_entries = _experience_texts(parse_resume(date_first_text))
    safe_check(
        "a block that begins with a standalone date is left whole (no guessing)",
        len(date_first_entries) == 1
        and "Company X" in date_first_entries[0]
        and "Company Y" in date_first_entries[0],
    )

    return failures[0]


# ============================================================
# NEW: THE EXPERIENCE-ONLY SPLITTING RULE MUST NOT AFFECT OTHER SECTIONS
# ============================================================

def run_checks_that_other_sections_are_unchanged():
    print()
    print("=" * 60)
    print("Checks that other sections are NOT split by standalone dates")
    print("=" * 60)

    failures = [0]
    safe_check = _make_safe_check(failures)

    text = """
EDUCATION
Bachelor of Science in Sample Technology
Sample University - Somewhere
2022 - 2026
Certificate in Sample Studies
Sample Institute - Elsewhere
2020-2021

PROJECTS
Sample Project One
Built a sample tool.
2024 - 2025
Sample Project Two
Built another sample tool.
2025-2026

HONORS & AWARDS
Dean's Lister
2023 - 2024
Magna Cum Laude
2025-2026
""".strip()

    data = parse_resume(text)

    safe_check(
        "education with standalone date lines is still ONE blank-line entry",
        len(data["education"]) == 1
        and "Sample Institute - Elsewhere" in data["education"][0]["raw_text"],
    )
    safe_check(
        "projects with standalone date lines are still ONE blank-line entry",
        len(data["projects"]) == 1
        and "Sample Project Two" in data["projects"][0]["raw_text"],
    )
    safe_check(
        "honors with standalone date lines are still ONE blank-line entry",
        len(data["honors"]) == 1
        and "Magna Cum Laude" in data["honors"][0]["raw_text"],
    )

    return failures[0]


def run_checks_on_malformed_pymupdf_lines():
    print()
    print("=" * 60)
    print("Checks that malformed PyMuPDF lines are skipped instead of crashing")
    print("=" * 60)

    from resume_extractor import _collect_lines

    class FakePage:
        def get_text(self, mode):
            return {
                "blocks": [
                    {"type": 0, "lines": [{"bbox": (0, 0, 10, 10), "spans": None}]},
                    {"type": 0, "lines": [{"bbox": (0, 0, 20, 20), "spans": [{"text": "OK", "font": "Arial", "flags": 0}]}]},
                ]
            }

    failures = [0]
    safe_check = _make_safe_check(failures)

    lines = _collect_lines(FakePage())
    safe_check(
        "malformed PyMuPDF lines are ignored while valid lines still load",
        len(lines) == 1 and lines[0].text == "OK",
    )

    return failures[0]


def main():
    total_failures = 0
    total_failures += run_checks_on_normalize_pdf_text_examples()
    total_failures += run_checks_on_pdf_extracted_fixture()
    total_failures += run_checks_on_sample_resume()
    total_failures += run_checks_on_generic_skill_fixture()
    total_failures += run_checks_on_empty_and_headerless_text()
    total_failures += run_checks_on_heading_aliases()
    total_failures += run_checks_on_project_summary_boundary()
    total_failures += run_checks_on_generic_candidate_name()
    total_failures += run_checks_on_contact_vs_skills()
    total_failures += run_checks_on_honors_section()
    total_failures += run_checks_on_interests_only()
    total_failures += run_checks_on_realistic_fixture()
    total_failures += run_checks_on_multi_entry_experience()
    total_failures += run_checks_that_other_sections_are_unchanged()
    total_failures += run_checks_on_malformed_pymupdf_lines()

    print()
    print("=" * 60)
    if total_failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{total_failures} CHECK(S) FAILED")
    print("=" * 60)

    sys.exit(1 if total_failures else 0)


if __name__ == "__main__":
    main()