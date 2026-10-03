"""
resume_parser.py
=================

A small, DETERMINISTIC helper module that turns raw resume text into a
structured Python dictionary, and that deterministically undoes pypdf's
per-character-spacing extraction artifact BEFORE that structured parsing
runs.

WHY THIS IS A SEPARATE FILE (instead of being added to agent_loop.py):
    agent_loop.py is already responsible for the LLM agent loop, the tool
    registry, and all the reliability safeguards. Structured resume parsing
    (and the PDF-text normalization it depends on) is a distinct,
    self-contained concern (plain text -> plain dict, no Ollama, no tools,
    no agent state), so it is kept in its own module and simply imported by
    agent_loop.py and api.py. This also makes it trivial to unit test on
    its own (see test_resume_parser.py), with no dependency on Ollama being
    installed or running.

IMPORTANT DESIGN RULES (matching the project's existing philosophy):
    * NO LLM call of any kind lives in this file. Everything here is plain
      string/regex processing.
    * NOTHING is invented. If a piece of information (a name, an email, a
      section) cannot be found in the actual text, the corresponding field
      is left empty/None rather than guessed. Normalization only ever
      reconstructs spacing from characters that are already present in the
      text - it never adds, removes, or infers a fact.
    * Section content is preserved as the ACTUAL resume text ("raw_text"),
      not rewritten or summarized. This keeps the structured layer grounded
      by construction, the same way search_resume() in agent_loop.py stays
      grounded by only ever returning lines that are really in the resume.
    * No eval(), no exec(), no subprocess, no dynamic imports - just str,
      re, and basic control flow.
    * Nothing in this file is specific to any one person's resume. Every
      heuristic below (heading aliases, contact regexes, the name
      heuristic, the "is this line just contact info" check, the
      experience-entry boundary check, the orphaned-lead-in check, the
      displaced-content check) is phrased in terms of generic patterns -
      never a specific name, company, or school - so the parser keeps
      working on other resumes.

PDF-TEXT NORMALIZATION
-----------------------------------------------
Real PDFs extracted through pypdf can come back with a per-character
spacing artifact: instead of "EDUCATION" or "JavaScript", pypdf sometimes
returns "E D U C A T I O N" or "J a v a S c r i p t" - a single space
between almost every individual character, with real word boundaries
marked by a run of two or more spaces. Left uncorrected, this breaks BOTH
this file's heading detection (HEADING_MAP below) AND agent_loop.py's
plain keyword/skill lookups, because neither can match "J a v a S c r i p
t" against the word "JavaScript".

normalize_pdf_text(text) (section 0 below) detects and undoes this
artifact, line by line, using only plain string/regex processing - no LLM,
no guessing. The algorithm, in short:

    1. For each line, split it into RUNS: a run is either a maximal chunk
       of non-space characters (a "token") or a maximal chunk of spaces.
    2. Decide whether the line LOOKS character-spaced: count how many
       tokens are exactly one character long. If there are at least
       _MIN_TOKENS_FOR_DETECTION tokens and at least
       _SINGLE_CHAR_RATIO_THRESHOLD (60%) of them are single characters,
       the line is treated as character-spaced. This is deliberately
       conservative: normal short sentences/dates ("2024 - 2026", "Web &
       Mobile Development") have far too low a ratio of one-character
       tokens to ever cross this threshold, so they are left untouched.
    3. If a line IS character-spaced, reconstruct it: a run of exactly ONE
       space between two tokens is treated as pypdf's artifact separator
       and removed (the characters are joined directly); a run of TWO OR
       MORE spaces is treated as a genuine word boundary and collapsed to
       a single real space. If a line is NOT character-spaced, it is
       returned completely unchanged.

This function is a no-op (returns the input unchanged) on text that does
not exhibit the artifact, and it is idempotent (normalizing already-clean
text, or normalizing twice, changes nothing further). Because of that, it
is always safe to call - which is why parse_resume() below calls it
unconditionally before doing anything else, and why agent_loop.py also
routes every active resume text through it (see agent_loop._resolve_resume_text).

The architecture is now:

    PDF -> pypdf extraction -> normalize_pdf_text() -> parse_resume() /
    keyword search -> agent tools -> LLM agent

WHAT parse_resume() RETURNS
----------------------------
    {
        "candidate_name": str | None,
        "contact": {
            # only keys that were actually found are present
            "email": str, "phone": str, "location": str,
            "linkedin": str, "github": str, "website": str,
        },
        "education": [ {"raw_text": "..."}, ... ],
        "experience": [ {"raw_text": "..."}, ... ],
        "skills": {"raw_text": "...", "items": [...]},
        "projects": [ {"raw_text": "..."}, ... ],
        "certifications": [ {"raw_text": "..."}, ... ],
        "honors": [ {"raw_text": "..."}, ... ],
        "summary": {"raw_text": "..."},
        # Any recognized heading with no dedicated field above (currently
        # just "interests"), PLUS any leftover text found at the top of the
        # resume that isn't the name or contact info, is kept here instead
        # of being silently discarded.
        "other_sections": {"interests": "...", "header": "...", ...},
    }

HOW SECTIONS ARE DETECTED
--------------------------
This parser recognizes a fixed, explicit list of section headings (see
HEADING_MAP below). A line counts as a heading when, once stripped of
surrounding whitespace and an optional trailing colon, it matches one of
those headings case-insensitively. This is intentionally simple and
predictable rather than a fuzzy "does this line look like a heading"
guess, which would risk misclassifying ordinary resume content. Because
normalize_pdf_text() runs BEFORE heading detection, a character-spaced
heading like "E D U C A T I O N" or "H O N O R S  &  A W A R D S" is
reconstructed to "EDUCATION" / "HONORS & AWARDS" first, so it is matched
exactly the same way a heading typed normally would be.

Every recognized heading - not just the ones with a dedicated top-level
field - terminates whatever section came before it. That is what makes
"SUMMARY" (or any other heading) a hard section boundary: once a heading
line is seen, everything from that point on belongs to the new section,
never the previous one. This is a property of the generic heading-scan
itself, not something bolted on for any particular section name.

HOW EXPERIENCE ENTRIES ARE SPLIT
----------------------------------
Every list-style section is first split into entries on BLANK LINES. That
is enough for most sections, but real PDF extraction frequently produces
an EXPERIENCE/INTERNSHIP section with NO blank lines at all between
several distinct jobs, so the whole section arrives as one block. For the
experience section ONLY, each such block is therefore additionally scanned
line by line (see _split_experience_block below):

    * A line that consists of NOTHING but a date range ("2024 - 2026",
      "2025-2026", "Jan 2024 - Present") is treated as the END of an
      experience entry, provided meaningful (non-blank, non-date) content
      follows it.
    * Dates that are merely part of a longer line ("Intern, Acme Corp,
      2024 - 2025") are never boundaries, and neither are duty/bullet
      lines.
    * Consecutive standalone date lines (a duplicated trailing date) stay
      together at the end of the SAME entry: only the last date line of
      such a run can be a boundary, and only if real content follows it.
      A trailing date with nothing after it never creates an extra entry.
    * A date-only block (separated by blank lines) that merely repeats the
      date line ending the previous entry is appended to that entry
      instead of becoming an entry containing only a date.
    * If a block BEGINS with a standalone date line, the layout is
      ambiguous (the date may lead its entry rather than end it), so the
      block is left whole rather than guessed at.

The rule looks only at the SHAPE of each line, never at any company,
title, or person, so it works on any resume that lists dates on their own
line. Education, projects, skills, honors, and every other section keep
using the plain blank-line split.

WHY A PROJECT CAN LOSE ITS OWN DATE AND DUTIES (ORPHANED LEAD-INS)
--------------------------------------------------------------------
Section detection is purely POSITIONAL: a heading ends the previous
section, and every line after it belongs to the new section. That is
correct when the extracted text follows the visual layout, but some PDFs
extract a role's right-aligned date and its duty bullets AFTER the next
section's heading, even though visually they sit under the previous
section's role/title lines. The result is:

    PROJECT EXPERIENCE
    <role line>
    <project title>
    <school/organization line>
    INTERNSHIP
    - duty ...            <- really belongs to the project above
    - duty ...
    2024 - 2026           <- really the project's date
    <company line>
    <role line>
    ...

Positionally, the duties and the date now sit inside the INTERNSHIP body.
The structural fact that exposes this is generic: an entry always STARTS
with a header line (a title or company line). A section body that begins
with duty bullets (and/or standalone dates) BEFORE any header line cannot
be the start of its own entry - it is the orphaned tail of the entry that
came before it. _reattach_orphan_leads() (section 3c) therefore moves such
a leading fragment back onto the previous role-style section when, and only
when, ALL of these hold:

    * both sections are role-style sections (experience or projects),
    * the fragment begins the section body and contains at least one duty
      bullet (dates alone stay ambiguous and are never moved),
    * real content (an actual header line) follows the fragment, so a
      section whose entire body is bullets is never touched, and
    * the previous section's last block is HEADER-ONLY (it has no duty
      bullets yet), i.e. it is visibly missing exactly what the fragment
      supplies.

WHY THAT WAS NOT ENOUGH (DISPLACED, MARKER-LESS DUTIES)
---------------------------------------------------------
The orphaned-lead-in repair assumes (1) duties carry bullet markers and (2)
the displaced duties sit at the very START of the next section's body.
Real extraction can violate BOTH:

    * The bullet glyphs are frequently dropped by the PDF text layer, so a
      duty arrives as a plain sentence line with no marker at all. The
      bullet check then sees nothing.
    * The displaced duties do not always come first. They can arrive AFTER
      the next section's own first header and duties, in one unbroken run
      with no blank line and no marker between the two owners' duties:

          PROJECT EXPERIENCE
          <project title> / <school> / <role>          <- header only
          INTERNSHIP
          <role> / <company>                           <- entry A header
          <A duty> <A duty> <A duty>
          <project duty> <project duty> <project duty> <- displaced
          2024 - 2026                                  <- project's date
          <company> / <role>                           <- entry B header
          <B duty> <B duty>
          2025-2026                                    <- A's date, displaced
          2025-2026                                    <- B's date

      The date rule in section 3b then closes "entry A" at the project's
      date, so the project's duties and date are glued onto A and A's own
      date ends up duplicated on B.

_repair_displaced_role_content() (section 3d) handles this pattern. Because
the two owners' duties are contiguous, NO line-shape rule can find the
seam between them. The only evidence left in the text is physical: lines
that belong to one text frame wrap at (about) the same width, so a run of
duty lines that cannot all have been wrapped at one width came from two
frames. The repair therefore fires only when EVERY one of these holds:

    * both sections are role-style, and the previous section's last block
      is header-only (no duty sentences, no dates) - it visibly lacks
      duties and a date;
    * the next section's body is exactly: header group, a run of >= 2
      duty units, ONE standalone date, then further header group(s), and
      finally a trailing run of standalone dates;
    * the standalone-date count is one more than the number of header
      groups in that body (one date per company plus the displaced one),
      and the trailing date run is exactly as long as the header-group
      count (so they can be handed back one per entry, in order);
    * the first duty run is NOT consistent with a single wrap width, and
      there is a split point where each side IS consistent (see
      _find_frame_split). If the wrap widths give no evidence of two
      frames, nothing is moved.

When it fires, the duty units after the split point plus the displaced date
are appended to the previous section's header-only block; the remaining
entries are separated by blank lines and each receives one of the trailing
dates, in order. Every line is moved verbatim: nothing is added, dropped,
reordered within a header, or rewritten.

The wrap-width evidence uses characters as a proxy for pixel width, so it
can be wrong for very different fonts; that is why it needs a tolerance,
why it must find positive evidence (silence means "do not move"), and why
the surrounding signature is so strict. Where a PDF gives no such evidence
the only fully reliable fix is upstream (layout-aware extraction that keeps
each text frame's coordinates), which is outside this module.

A NOTE ON MESSY, MULTI-COLUMN PDF TEXT
----------------------------------------
Some real PDFs (especially multi-column resumes) extract with their text
in an order that does not match the visual layout - e.g. a paragraph that
visually sits at the top of the page might extract AFTER a heading that
visually sits lower on the page, or contact details might extract in the
middle of an unrelated section. This module does not try to detect and
"correct" that kind of reordering by guessing the intended visual layout -
doing so would require guessing, and guessing is exactly what this file is
designed to avoid. Apart from the two narrow, shape-based repairs described
above, it only ever looks at the ACTUAL sequence of headings and lines it
is given and applies the same generic rules regardless of where in the
document that content ends up. If a source PDF extracts text in a
scrambled order, the resulting structured sections will faithfully reflect
the actual extracted text, not a hallucinated "fixed" version of it.
"""

import re


# ============================================================
# 0. PDF TEXT NORMALIZATION (pypdf per-character-spacing artifact)
# ============================================================
# See the module docstring above for the full explanation of WHY this
# exists and HOW the algorithm works. This section is pure, deterministic
# string/regex processing - no LLM, no network, no guessing about missing
# content. It only ever reconstructs spacing from characters that are
# already present in the input text.

# Matches either a run of non-whitespace characters (a token) or a run of
# plain spaces/tabs (a spacing run). Used to break a single line into an
# ordered list of alternating tokens and spacing runs, so the reconstruction
# step can tell a one-space "character separator" apart from a two-or-more
# space "real word boundary".
_WS_RUN_RE = re.compile(r'\S+|[ \t]+')

# A line needs at least this many non-space tokens before normalization is
# even considered. This avoids false positives on very short lines (e.g.
# "2024 - 2026", which has only 3 tokens and a low single-character ratio
# anyway, but this adds an extra margin of safety for very short lines in
# general).
_MIN_TOKENS_FOR_DETECTION = 3

# A line is treated as character-spaced only when AT LEAST this fraction of
# its non-space tokens are exactly one character long. Ordinary resume text
# ("National University - Manila", "Web & Mobile Development") has at most
# one or two naturally short tokens (like "-" or "&") out of several
# multi-character words, so its ratio stays well below this threshold and
# the line is left completely alone.
_SINGLE_CHAR_RATIO_THRESHOLD = 0.6

# A spacing run of this many spaces (or more) between two tokens is treated
# as a genuine WORD boundary once a line has been identified as
# character-spaced. A spacing run shorter than this (i.e. exactly one
# space) is treated as pypdf's artifact separator between individual
# characters of the SAME word, and is removed during reconstruction.
_WORD_BOUNDARY_MIN_SPACES = 2


def _line_runs(line):
    """Split one line (no newline characters) into an ordered list of
    runs: each run is either a token (non-space characters) or a spacing
    run (one or more spaces/tabs). Concatenating the runs back together
    reproduces the original line exactly."""
    return _WS_RUN_RE.findall(line)


def _is_char_spaced_line(runs):
    """Decide whether `runs` (the output of _line_runs) looks like pypdf's
    per-character-spacing artifact: mostly single-character tokens, with
    enough tokens overall that this isn't just coincidental short words."""
    tokens = [r for r in runs if not r[0].isspace()]

    if len(tokens) < _MIN_TOKENS_FOR_DETECTION:
        return False

    single_char_tokens = sum(1 for t in tokens if len(t) == 1)
    ratio = single_char_tokens / len(tokens)

    return ratio >= _SINGLE_CHAR_RATIO_THRESHOLD


def _reconstruct_char_spaced_line(runs):
    """Reconstruct a line that _is_char_spaced_line() has already flagged
    as character-spaced: tokens separated by a spacing run of length ONE
    are concatenated directly (the artifact separator is removed); a
    spacing run of length TWO OR MORE marks a real word boundary and
    becomes a single space between the reconstructed words. Nothing is
    invented - every character in the output was already present in one of
    the input tokens."""
    words = []
    buffer = []

    for run in runs:
        if run[0].isspace():
            if len(run) >= _WORD_BOUNDARY_MIN_SPACES:
                if buffer:
                    words.append("".join(buffer))
                    buffer = []
            # A single space between two tokens is the per-character
            # separator artifact itself: skip it (do not start a new word).
        else:
            buffer.append(run)

    if buffer:
        words.append("".join(buffer))

    return " ".join(words)


def normalize_pdf_text(text):
    """Deterministically undo pypdf's per-character-spacing extraction
    artifact, wherever it appears, without an LLM and without inventing
    any content.

    Applied LINE BY LINE (newlines are preserved - this never collapses
    the resume into one paragraph, which matters because parse_resume()
    depends on line structure to find section headings):
      * A line that looks character-spaced ("E D U C A T I O N",
        "J a v a S c r i p t", "t r o y @ g m a i l . c o m") is
        reconstructed: single spaces between tokens are removed (they were
        pypdf's artifact), and runs of two-or-more spaces become the real
        word boundaries.
      * A line that does NOT look character-spaced (ordinary text, however
        many short words or hyphens it contains - "National University -
        Manila", "Web & Mobile Development", "2024 - 2026") is returned
        completely unchanged.

    This function is a no-op on text that never exhibits the artifact, and
    it is idempotent - normalizing already-normalized text, or the same
    text twice, always produces the same result. That makes it always safe
    to call defensively (see parse_resume() below and
    agent_loop._resolve_resume_text()), whether the input is raw pypdf
    output or text that is already clean.

    Blank/whitespace-only lines and non-string/empty input are returned
    unchanged.
    """
    if not text or not isinstance(text, str):
        return text

    lines = text.split("\n")
    normalized_lines = []

    for line in lines:
        # Strip a possible trailing carriage return (e.g. from Windows-style
        # line endings) but otherwise leave the line's own content alone
        # until we know whether it needs reconstruction.
        stripped_trailing = line.rstrip("\r")

        if not stripped_trailing.strip():
            # Blank or whitespace-only line: nothing to normalize, and
            # blank lines are meaningful (they separate resume entries).
            normalized_lines.append(stripped_trailing)
            continue

        runs = _line_runs(stripped_trailing)

        if _is_char_spaced_line(runs):
            normalized_lines.append(_reconstruct_char_spaced_line(runs))
        else:
            normalized_lines.append(stripped_trailing)

    return "\n".join(normalized_lines)


# ============================================================
# 1. KNOWN SECTION HEADINGS
# ============================================================
# Maps a heading's UPPERCASE text to a canonical section key.
# Several headings can map to the same canonical key (e.g. "EXPERIENCE",
# "WORK EXPERIENCE", "INTERNSHIP" and "INTERNSHIPS" are all treated as the
# candidate's experience section; "PROJECTS" and "PROJECT EXPERIENCE" are
# both treated as the candidate's projects section; "HONORS & AWARDS" and
# its common variants are all treated as the candidate's honors section).
#
# "INTERESTS" is recognized as a heading (so its content is correctly cut
# out of whatever section comes before/after it) but has no dedicated
# top-level field in the output dict, so it is filed under
# other_sections["interests"] instead of being invented a field of its
# own. This is a generic, common resume section title (not specific to any
# one person's resume), included so the parser generalizes to real-world
# resumes that use it.
#
# Every entry below is a generic heading label that shows up across many
# different resumes - none of it is tied to any specific person's resume
# content.

HEADING_MAP = {
    "EDUCATION": "education",
    "EXPERIENCE": "experience",
    "WORK EXPERIENCE": "experience",
    "INTERNSHIP": "experience",
    "INTERNSHIPS": "experience",
    "SKILLS": "skills",
    "TECHNICAL SKILLS": "skills",
    "PROJECTS": "projects",
    "PROJECT EXPERIENCE": "projects",
    "CERTIFICATIONS": "certifications",
    "CERTIFICATES": "certifications",
    "INTERESTS": "interests",
    "HONORS": "honors",
    "HONORS & AWARDS": "honors",
    "HONORS AND AWARDS": "honors",
    "AWARDS": "honors",
    "AWARDS & HONORS": "honors",
    "SUMMARY": "summary",
    "PROFESSIONAL SUMMARY": "summary",
    "CAREER SUMMARY": "summary",
    "PROFILE": "summary",
    "OBJECTIVE": "summary",
}

# Canonical keys that get their own top-level LIST field in the output
# (each entry is a preserved raw_text block). "skills" and "summary" are
# handled separately below because they are single free-text fields rather
# than a list of entries.
_LIST_SECTION_KEYS = {"education", "experience", "projects", "certifications", "honors"}

# Role-style sections: sections whose entries are made of a header (a
# title/company/project line), optional dates, and duty bullets. Only these
# take part in the orphaned-lead-in re-attachment (section 3c) and the
# displaced-content repair (section 3d), so a bullets-first or
# sentence-heavy body in, say, a skills or education section is never
# touched.
_ROLE_SECTION_KEYS = {"experience", "projects"}


def _heading_key(line):
    """Return the canonical section key if `line` is a known heading,
    otherwise None. Tolerant of extra whitespace, a trailing colon, and
    any capitalization."""
    candidate = line.strip()

    if not candidate:
        return None

    # Drop one optional trailing colon ("Skills:" -> "Skills").
    if candidate.endswith(":"):
        candidate = candidate[:-1].strip()

    return HEADING_MAP.get(candidate.upper())


# ============================================================
# 2. CONTACT INFORMATION PATTERNS
# ============================================================
# Contact fields are searched for across the WHOLE document (not just the
# header) because some PDF layouts (multi-column resumes especially) can
# extract contact details in the middle of the text rather than at the
# very top. Each regex is still applied line-by-line (never across a
# newline) so that unrelated numbers on separate lines (e.g. two separate
# date ranges) can never be concatenated into a false "phone number".

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Loose but reasonably safe phone pattern: a run of digits/spaces/dashes/
# dots/parentheses that contains at least 7 digits in total. Restricted to
# a single line (see usage below) so it can never span a line break.
_PHONE_RE = re.compile(r"(\+?\d[\d\-\.\s\(\)]{6,}\d)")

_LINKEDIN_RE = re.compile(r"(https?://)?(www\.)?linkedin\.com/\S+", re.IGNORECASE)
_GITHUB_RE = re.compile(r"(https?://)?(www\.)?github\.com/\S+", re.IGNORECASE)

# A generic URL/website pattern, used only as a fallback once linkedin/
# github/email have already been ruled out for that line.
_WEBSITE_RE = re.compile(
    r"(https?://)?(www\.)?[A-Za-z0-9\-]+\.[A-Za-z]{2,}(/[^\s,]*)?", re.IGNORECASE
)

# Generic words that show up in EDUCATION-type lines. Used to keep a
# degree/institution line (e.g. "Bachelor of Science ..., National
# University") from ever being mistaken for a location line just because
# it happens to contain a comma. None of these words are specific to any
# one school.
_EDUCATION_WORDS = {
    "bachelor", "master", "b.s.", "m.s.", "degree", "university",
    "college", "institute", "academy", "diploma", "certificate",
}


def _count_digits(text):
    return sum(character.isdigit() for character in text)


def _looks_like_year_range(candidate):
    """True if `candidate` (a phone-regex match) is really just two
    four-digit years separated by punctuation/whitespace, e.g. "2022 -
    2026" or "2025-2026" - a extremely common resume date-range format
    that would otherwise satisfy the phone-number pattern (7+ digits in a
    digit/dash/space run). This check is generic (it only looks at the
    SHAPE of the digits, not any particular year), so it works for any
    resume's date ranges, not just one person's."""
    digit_groups = re.findall(r"\d+", candidate)
    return len(digit_groups) == 2 and all(len(group) == 4 for group in digit_groups)


def _looks_like_name(line):
    """True if `line` looks like "First Last" / "First Middle Last", with
    no digits, no '@', and 2-4 words that each start with a capital letter.
    This is intentionally conservative: if it does not clearly look like a
    name, we do not guess."""
    candidate = line.strip()

    if not candidate or "@" in candidate:
        return False

    if _count_digits(candidate) > 0:
        return False

    words = candidate.split()

    if not (2 <= len(words) <= 4):
        return False

    name_word = re.compile(r"^[A-Z][a-zA-Z.'\-]*$")
    return all(name_word.match(word) for word in words)


# Generic words that, when they appear in an otherwise name-shaped line,
# mean the line is almost certainly an organization name, a job title, or
# a resume-section-style word rather than a person's name. None of these
# are specific to any one company, school, or resume - they are common
# words that appear across many organizations and job titles in general.
_ORG_KEYWORDS = {
    "university", "college", "institute", "academy", "school",
    "corporation", "corp", "incorporated", "inc", "llc", "ltd",
    "company", "co", "services", "systems", "technologies",
    "technology", "solutions", "group", "enterprises", "agency",
    "foundation", "labs", "laboratory", "partners", "associates",
}
_TITLE_KEYWORDS = {
    "manager", "engineer", "developer", "intern", "director",
    "specialist", "analyst", "coordinator", "assistant", "lead",
    "officer", "consultant", "designer", "architect",
    "administrator", "supervisor", "president", "executive",
    "founder", "owner", "chief", "head", "vp",
}
_OTHER_NON_NAME_WORDS = {
    "project", "projects", "experience", "internship", "summary",
    "objective", "profile", "resume", "curriculum", "vitae", "capstone",
    "education", "skills", "interests", "honors", "awards",
    "certifications", "certificates",
}
_NON_NAME_KEYWORDS = _ORG_KEYWORDS | _TITLE_KEYWORDS | _OTHER_NON_NAME_WORDS


def _looks_like_non_name_line(line):
    """True if `line` contains a generic organization/title/section-style
    word, meaning it should never be treated as a candidate's name even if
    it happens to be shaped like one (e.g. "National University", "Sample
    Corporation", "Lead Engineer"). Purely word-list based - no specific
    company, school, or person is referenced."""
    words = re.findall(r"[A-Za-z]+", line.lower())
    return any(word in _NON_NAME_KEYWORDS for word in words)


def _extract_name(lines, heading_line_indices):
    """Return the first line in the document that clearly looks like a
    person's standalone full name, or None if nothing qualifies. Nothing
    is invented - a name is only returned when a line in the actual text
    passes every check below.

    Because this scans the WHOLE document in order (skipping only
    recognized heading lines), a name that sits in the conventional header
    position is still found first simply because the header comes first in
    the document. But it also generalizes to resumes where PDF extraction
    scrambles the reading order and a standalone name line ends up further
    down (e.g. inside a project or contact block) - a real, common PDF
    text-extraction artifact for multi-column layouts.

    A line only qualifies when it:
      * looks like "First [Middle] Last" (_looks_like_name), and
      * is not written ALL IN CAPS (job titles and some headings commonly
        are, e.g. "CAPSTONE PROJECT MANAGER" - a real person's name line is
        not), and
      * does not contain a generic organization/title/section word
        (_looks_like_non_name_line) - so a school, employer, or job title
        that happens to be name-shaped ("Sample University", "Lead
        Engineer") is never mistaken for a name.

    Nothing about this heuristic depends on any specific name, company, or
    school - it only inspects the generic shape and vocabulary of each
    line.
    """
    for index, line in enumerate(lines):
        if index in heading_line_indices:
            continue

        stripped = line.strip()
        if not stripped:
            continue

        if not _looks_like_name(stripped):
            continue

        if stripped.isupper():
            continue

        if _looks_like_non_name_line(stripped):
            continue

        return stripped

    return None


def _segment_is_hard_contact_match(segment):
    """True if `segment` itself contains an email, a LinkedIn/GitHub URL,
    or a phone number (excluding year-range false positives). This is the
    "high confidence" contact check, used both to decide whether a whole
    line is contact-only noise and to populate the contact dict itself."""
    if _EMAIL_RE.search(segment):
        return True
    if _LINKEDIN_RE.search(segment) or _GITHUB_RE.search(segment):
        return True
    for match in _PHONE_RE.finditer(segment):
        candidate = match.group(0).strip()
        if _count_digits(candidate) >= 7 and not _looks_like_year_range(candidate):
            return True
    return False


def _segment_looks_like_location(segment):
    """Conservative "this short, comma-containing segment looks like a
    location" check (e.g. "Sampaloc, Metro Manila", "San Pablo,
    Calabarzon"). Deliberately narrow: short (<=6 words), few digits,
    doesn't look like a name, and doesn't contain a generic education
    word - so an ordinary comma-joined sentence elsewhere in the resume
    (e.g. "Chess, hiking" under Interests, or a degree line) is not
    mistaken for an address."""
    stripped = segment.strip()
    if "," not in stripped:
        return False
    if _count_digits(stripped) > 4:
        return False
    words = stripped.split()
    if not (1 <= len(words) <= 6):
        return False
    if _looks_like_name(stripped):
        return False
    if any(word.strip(".,").lower() in _EDUCATION_WORDS for word in words):
        return False
    return True


def _line_is_contact_only(line):
    """True if `line` (optionally pipe-separated, e.g. "phone | city,
    region |", a common way contact details are laid out on one resume
    line) consists ENTIRELY of contact-detail segments and nothing else.
    Used to keep contact information that unusual PDF text extraction has
    placed inside the body of some other section (e.g. Skills) out of that
    section's content.

    The location-style check (_segment_looks_like_location) is only ever
    trusted for a segment that sits on a line that ALSO contains a hard
    contact match (email/phone/LinkedIn/GitHub) somewhere among its other
    segments. This is what keeps an unrelated comma-joined line elsewhere
    in the resume (e.g. "Chess, hiking" or "Dean's Lister, State
    University") from ever being stripped out as if it were contact
    information - it is only trusted right next to a real phone number or
    email, exactly the layout produced by a contact-details line.
    """
    segments = [segment.strip() for segment in line.split("|")]
    non_empty_segments = [segment for segment in segments if segment]

    if not non_empty_segments:
        return False

    has_hard_contact_match = any(
        _segment_is_hard_contact_match(segment) for segment in non_empty_segments
    )
    if not has_hard_contact_match:
        return False

    def _segment_is_contact_like(segment):
        return _segment_is_hard_contact_match(segment) or _segment_looks_like_location(segment)

    return all(_segment_is_contact_like(segment) for segment in non_empty_segments)


def _extract_contact(all_lines):
    """Best-effort, whole-document extraction of contact fields. Only
    fields that are clearly present are included - nothing is invented or
    required to match any specific person's actual values."""
    contact = {}

    for line in all_lines:
        if "email" not in contact:
            match = _EMAIL_RE.search(line)
            if match:
                contact["email"] = match.group(0)

        if "linkedin" not in contact:
            match = _LINKEDIN_RE.search(line)
            if match:
                contact["linkedin"] = match.group(0)

        if "github" not in contact:
            match = _GITHUB_RE.search(line)
            if match:
                contact["github"] = match.group(0)

        if "phone" not in contact:
            for match in _PHONE_RE.finditer(line):
                candidate = match.group(0).strip()
                if _count_digits(candidate) >= 7 and not _looks_like_year_range(candidate):
                    contact["phone"] = candidate
                    break

    # A generic website/portfolio link: skip any line that contains the
    # email (to avoid matching the email's own domain) and skip a match
    # that is really the linkedin/github URL already captured above. Lines
    # are also split on "|" (the same contact-details layout handled
    # elsewhere), and a match is only trusted when it covers MOST of its
    # segment - this is what keeps a domain-shaped word mentioned in the
    # middle of an unrelated sentence (e.g. "Node.js" inside a long skills
    # list) from being mistaken for a website link, while still matching a
    # segment that really is just a bare URL/domain.
    for line in all_lines:
        if "@" in line:
            continue
        for segment in line.split("|"):
            stripped = segment.strip().rstrip(".,;")
            if not stripped:
                continue
            match = _WEBSITE_RE.search(stripped)
            if not match:
                continue
            found = match.group(0)
            if found == contact.get("linkedin") or found == contact.get("github"):
                continue
            if "linkedin.com" in found.lower() or "github.com" in found.lower():
                continue
            if len(found) < 0.8 * len(stripped):
                continue
            contact["website"] = found
            break
        if "website" in contact:
            break

    # Location: only trust a comma-containing segment as a location when it
    # passes _segment_looks_like_location. Lines are split on "|" first so
    # a layout like "555-000-1111 | City, Region |" (phone and location
    # packed onto one physical line) doesn't fail the location check just
    # because the *line as a whole* has too many digits.
    for line in all_lines:
        found_location = None
        for segment in line.split("|"):
            stripped = segment.strip()
            if not stripped or "@" in stripped:
                continue
            if _LINKEDIN_RE.search(stripped) or _GITHUB_RE.search(stripped):
                continue
            if _segment_looks_like_location(stripped):
                found_location = stripped
                break
        if found_location:
            contact["location"] = found_location
            break

    return contact


# ============================================================
# 3. SPLITTING SECTION TEXT INTO PRESERVED "raw_text" ENTRIES
# ============================================================
# A section's body is split into one entry per blank-line-separated block,
# so a section that clearly lists several distinct items (e.g. two
# projects separated by a blank line) becomes several entries, while a
# section written as one continuous block (no blank lines) stays as a
# single entry. Either way, the text itself is preserved verbatim.
#
# The EXPERIENCE section additionally gets a second, line-by-line pass
# (section 3b below) because real PDF extraction often delivers several
# jobs as ONE blank-line-free block.

_BLANK_LINE_RE = re.compile(r"\n\s*\n+")


def _split_into_blocks(section_text):
    if not section_text.strip():
        return []

    blocks = _BLANK_LINE_RE.split(section_text.strip())
    return [block.strip() for block in blocks if block.strip()]


def _as_raw_text_entries(section_text):
    return [{"raw_text": block} for block in _split_into_blocks(section_text)]


# ============================================================
# 3b. EXPERIENCE: SPLITTING ONE BLOCK INTO SEVERAL ENTRIES
# ============================================================
# See "HOW EXPERIENCE ENTRIES ARE SPLIT" in the module docstring. Only the
# SHAPE of a line is inspected: a line counts as a "standalone date range"
# only when the WHOLE line is a date range and nothing else. Any line that
# carries other text (a title/company with an inline date, a duty line, a
# bullet) can never match, so inline dates and duty lines are never
# boundaries. No company, title, or person is referenced anywhere.

_MONTH_NAME = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_RANGE_START = rf"(?:{_MONTH_NAME}\s+)?\d{{4}}"
_RANGE_END = rf"(?:{_MONTH_NAME}\s+)?(?:\d{{4}}|present|current|now|ongoing)"

# Whole line == "<start> <separator> <end>", e.g. "2024 - 2026",
# "2025-2026", "2023 – Present", "Jan 2024 - Mar 2025", "2022 to 2024".
_DATE_RANGE_LINE_RE = re.compile(
    rf"^\s*{_RANGE_START}\s*(?:-|\u2013|\u2014|\bto\b)\s*{_RANGE_END}\s*$",
    re.IGNORECASE,
)


def _is_date_range_line(line):
    """True if `line` is NOTHING BUT a date range (see _DATE_RANGE_LINE_RE).
    A date that is only part of a longer line is deliberately not a match."""
    return bool(_DATE_RANGE_LINE_RE.match(line))


def _next_non_blank_line(lines, start_index):
    """Return the first non-blank line at or after `start_index`, or None
    if there is none."""
    for line in lines[start_index:]:
        if line.strip():
            return line
    return None


def _split_experience_block(block_text):
    """Split ONE contiguous (blank-line-free) experience block into one or
    more entry texts by scanning it line by line.

    A standalone date-range line CLOSES the current entry (the date stays
    at the end of the entry it closes) when, and only when, meaningful
    content follows it - i.e. the next non-blank line exists and is not
    itself a standalone date-range line. Consequences:
      * Inline dates and duty/bullet lines never split anything.
      * A run of consecutive date lines (a duplicated trailing date) is
        kept together: every date line except the last one in the run is
        followed by another date line, so none of them is a boundary, and
        the last one is a boundary only if real content comes after it.
      * A date line at the very end of the block has nothing after it, so
        it never creates an extra (date-only) entry.
      * If the block BEGINS with a standalone date line, the date may be
        leading its entry rather than ending it - that is ambiguous, so
        the block is returned whole instead of being guessed at.

    Nothing is added, removed, or rewritten: joining the returned entries
    with newlines reproduces the block's own lines in order.
    """
    lines = block_text.splitlines()

    first_line = _next_non_blank_line(lines, 0)
    if first_line is None:
        return []

    if _is_date_range_line(first_line):
        return [block_text.strip()]

    entries = []
    current = []
    current_has_content = False

    for index, line in enumerate(lines):
        current.append(line)

        if not line.strip():
            continue

        if not _is_date_range_line(line):
            current_has_content = True
            continue

        # `line` is a standalone date range. It ends the current entry only
        # if the entry already has real content AND real content follows.
        if not current_has_content:
            continue

        following = _next_non_blank_line(lines, index + 1)
        if following is None or _is_date_range_line(following):
            continue

        entries.append(current)
        current = []
        current_has_content = False

    if current:
        entries.append(current)

    texts = ["\n".join(entry_lines).strip() for entry_lines in entries]
    return [text for text in texts if text]


def _block_is_only_date_ranges(block_text):
    """True if every non-blank line of `block_text` is a standalone date
    range (and there is at least one)."""
    non_blank = [line for line in block_text.splitlines() if line.strip()]
    return bool(non_blank) and all(_is_date_range_line(line) for line in non_blank)


def _last_non_blank_line(text):
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return None


def _as_experience_entries(section_text):
    """Turn the experience section text into a list of {"raw_text": ...}
    entries: first split on blank lines (existing behavior, so
    blank-line-separated entries stay separate), then split each block at
    standalone date-range boundaries (_split_experience_block).

    A block made up ONLY of date-range lines is not an experience entry.
    When the previous entry already ends with a date-range line, such a
    block is a repeated/duplicated date and is appended to that entry so
    no date-only entry is created. If the previous entry does NOT end with
    a date, it is ambiguous which entry the date belongs to, so the block
    is kept as-is (never discarded, never guessed).
    """
    entries = []

    for block in _split_into_blocks(section_text):
        if entries and _block_is_only_date_ranges(block):
            previous_last_line = _last_non_blank_line(entries[-1]["raw_text"])
            if previous_last_line is not None and _is_date_range_line(previous_last_line):
                entries[-1]["raw_text"] += "\n" + block
                continue

        for text in _split_experience_block(block):
            entries.append({"raw_text": text})

    return entries


# ============================================================
# 3c. ROLE SECTIONS: RE-ATTACHING AN ORPHANED LEAD-IN
# ============================================================
# See "WHY A PROJECT CAN LOSE ITS OWN DATE AND DUTIES" in the module
# docstring. Section detection is positional, so when PDF extraction emits
# a role's duty bullets (and right-aligned date) AFTER the NEXT section's
# heading, they end up at the START of that next section's body, before any
# header line. An entry always starts with a header line, so a body that
# begins with bullets cannot be the start of its own entry: it is the
# orphaned tail of the previous section's last entry.
#
# Only the SHAPE of each line is inspected (bullet marker, standalone date
# range, lowercase-initial wrapped continuation). Nothing is rewritten and
# no company, title, school, or person is referenced.

# A duty/bullet line: starts with a bullet marker character followed by
# content. En/em dashes are deliberately NOT bullet markers here, because
# they appear inside ordinary header lines ("Some University - City") and a
# line that STARTS with one is not a normal bullet.
_BULLET_LINE_RE = re.compile(r"^\s*[-*\u2022\u00b7\u25aa\u25cf]\s*\S")


def _is_bullet_line(line):
    """True if `line` starts with a bullet marker followed by content."""
    return bool(_BULLET_LINE_RE.match(line))


def _is_continuation_line(line):
    """True if `line` looks like the wrapped continuation of a bullet: it
    is not a bullet itself and starts with a lowercase letter. Header lines
    (titles, companies) start with a capital, so they never qualify."""
    stripped = line.strip()
    return bool(stripped) and stripped[0].islower() and not _is_bullet_line(line)


def _split_orphan_lead(lines):
    """Split a section body (list of lines) into (lead_lines, rest_lines),
    where `lead_lines` is the orphaned lead-in: the leading run of lines
    that are duty bullets, standalone date ranges, or wrapped bullet
    continuations, ending at the first line that looks like a header.

    A lead-in is only reported when it contains at least one duty bullet
    (a lone date is ambiguous - it may legitimately lead its own entry -
    so it is never treated as an orphan) AND real content follows it (so a
    section whose whole body is bullets is left untouched). Otherwise
    returns ([], lines) unchanged. Blank lines inside the lead-in are
    dropped so the moved text stays one contiguous block."""
    lead_end = 0
    has_bullet = False
    last_was_bullet = False

    for index, line in enumerate(lines):
        if not line.strip():
            continue

        if _is_date_range_line(line):
            last_was_bullet = False
        elif _is_bullet_line(line):
            has_bullet = True
            last_was_bullet = True
        elif last_was_bullet and _is_continuation_line(line):
            pass
        else:
            break

        lead_end = index + 1

    if not has_bullet:
        return [], lines

    rest = lines[lead_end:]
    if not any(line.strip() for line in rest):
        return [], lines

    lead = [line for line in lines[:lead_end] if line.strip()]
    return lead, rest


def _last_block_is_header_only(lines):
    """True if the LAST blank-line-separated block of `lines` exists and
    contains no duty bullets - i.e. it is visibly a header (role/title/
    organization lines) that is still missing its duties."""
    block = []

    for line in reversed(lines):
        if not line.strip():
            if block:
                break
            continue
        block.append(line)

    if not block:
        return False

    return not any(_is_bullet_line(line) for line in block)


def _reattach_orphan_leads(sections):
    """`sections` is the ordered list of [canonical_key, body_lines] pairs
    (one per heading, in document order). For each role-style section
    whose body begins with an orphaned lead-in (see _split_orphan_lead),
    move that lead-in onto the end of the immediately preceding role-style
    section, provided that section's last block is header-only. Mutates
    `sections` in place. Any section that does not meet every condition is
    left exactly as extracted.

    Returns a list of repair-event dicts, one per reattachment actually
    performed (empty if this repair never fired). This is PURE PROVENANCE
    bookkeeping added for diagnostics (see parse_resume()'s "repair_applied"
    tagging below): it does not change which sections are examined, which
    conditions are checked, or what gets moved - it only records, for each
    move that already happens, which section received the moved lines
    (`key`, matching that section's canonical key) and exactly which lines
    were moved (`lines`), so a later step can tell which specific preserved
    entry the repair actually touched.
    """
    repair_events = []

    for index in range(1, len(sections)):
        previous_key, previous_lines = sections[index - 1]
        key, lines = sections[index]

        if key not in _ROLE_SECTION_KEYS or previous_key not in _ROLE_SECTION_KEYS:
            continue

        lead, rest = _split_orphan_lead(lines)
        if not lead:
            continue

        if not _last_block_is_header_only(previous_lines):
            continue

        trimmed_previous = list(previous_lines)
        while trimmed_previous and not trimmed_previous[-1].strip():
            trimmed_previous.pop()

        sections[index - 1][1] = trimmed_previous + lead
        sections[index][1] = rest

        repair_events.append({
            "type": "orphan_lead_reattach",
            "key": previous_key,
            "lines": list(lead),
        })

    return repair_events


# ============================================================
# 3d. ROLE SECTIONS: DUTIES AND DATES DISPLACED PAST THE NEXT HEADING
# ============================================================
# See "WHY THAT WAS NOT ENOUGH" in the module docstring. This repair works
# WITHOUT bullet markers and WITHOUT assuming the displaced lines come
# first in the next section's body.
#
# Vocabulary used below (all purely about line SHAPE):
#   unit   - one logical line: a line plus any wrapped lowercase-initial
#            continuation lines directly after it.
#   "date" - a unit that is a standalone date range.
#   "duty" - a unit that is bullet-marked, OR that ends in sentence
#            punctuation and has enough words to be a sentence. Headers
#            (titles, organizations, project names) essentially never end
#            in sentence punctuation.
#   "header" - any other unit.

# Punctuation that ends a duty sentence.
_TERMINAL_PUNCTUATION = (".", "!", "?")

# A marker-less unit must have at least this many words to count as a duty,
# so a short organization name ending in a period ("Some Company Inc.") is
# not mistaken for one.
_MIN_WORDS_FOR_MARKERLESS_DUTY = 5

# Wrap-width evidence is measured in CHARACTERS, a proxy for pixel width
# that is off by a few percent for proportional fonts. Two runs are only
# treated as coming from different text frames when their width bounds
# disagree by MORE than this many characters.
_FRAME_WIDTH_TOLERANCE_CHARS = 8

_UNBOUNDED = float("inf")


def _unit_is_duty(unit_lines):
    """True if the unit (list of raw lines) looks like a duty sentence."""
    if _is_bullet_line(unit_lines[0]):
        return True

    if not unit_lines[-1].rstrip().endswith(_TERMINAL_PUNCTUATION):
        return False

    word_count = sum(len(line.split()) for line in unit_lines)
    return word_count >= _MIN_WORDS_FOR_MARKERLESS_DUTY


def _build_units(lines):
    """Group the non-blank `lines` into units (see the vocabulary above).
    Each unit is {"kind": "date"|"duty"|"header", "lines": [raw lines]}.
    Blank lines are ignored here; the ORIGINAL lines are kept verbatim
    inside the units so nothing is ever rewritten."""
    units = []

    for line in lines:
        if not line.strip():
            continue

        if _is_date_range_line(line):
            units.append({"kind": "date", "lines": [line]})
        elif units and units[-1]["kind"] != "date" and _is_continuation_line(line):
            units[-1]["lines"].append(line)
        else:
            units.append({"kind": "text", "lines": [line]})

    for unit in units:
        if unit["kind"] == "text":
            unit["kind"] = "duty" if _unit_is_duty(unit["lines"]) else "header"

    return units


def _flatten_units(units):
    return [line for unit in units for line in unit["lines"]]


def _last_block_lines(lines):
    """The lines of the LAST blank-line-separated block of `lines`."""
    block = []

    for line in reversed(lines):
        if not line.strip():
            if block:
                break
            continue
        block.append(line)

    block.reverse()
    return block


def _unit_width_bounds(unit_lines):
    """Bounds (lower, upper) on the wrap width, in characters, of the text
    frame a duty unit was laid out in.

    * Every line must fit, so the width is at least the longest line.
    * A line that WRAPPED (it is followed by a continuation line) was cut
      because the next word did not fit, so the width is less than that
      line plus a space plus the next line's first word.
    A unit that never wrapped only has a lower bound (upper is unbounded)."""
    stripped = [line.strip() for line in unit_lines]
    lower = max(len(line) for line in stripped)
    upper = _UNBOUNDED

    for current, following in zip(stripped, stripped[1:]):
        first_word_length = len(following.split()[0])
        upper = min(upper, len(current) + 1 + first_word_length)

    return lower, upper


def _frame_is_consistent(bounds):
    """True if all of `bounds` could have come from ONE text frame (allowing
    _FRAME_WIDTH_TOLERANCE_CHARS of slack for the character/pixel proxy)."""
    lower = max(bound[0] for bound in bounds)
    upper = min(bound[1] for bound in bounds)
    return lower <= upper + _FRAME_WIDTH_TOLERANCE_CHARS


def _find_frame_split(duty_units):
    """Find where a contiguous run of duty units switches from one text
    frame to another. Returns the number of units that stay with the
    FIRST owner, or None when there is no evidence of two frames.

    The run as a whole must be inconsistent with a single wrap width, and
    the split must leave BOTH sides consistent. When several split points
    qualify (the units around the seam fit either frame), the LARGEST
    prefix is kept with the first owner, i.e. ambiguous units stay with
    the entry whose header they directly follow."""
    if len(duty_units) < 2:
        return None

    bounds = [_unit_width_bounds(unit["lines"]) for unit in duty_units]

    if _frame_is_consistent(bounds):
        return None

    for split in range(len(bounds) - 1, 0, -1):
        if _frame_is_consistent(bounds[:split]) and _frame_is_consistent(bounds[split:]):
            return split

    return None


def _match_displaced_pattern(units):
    """Match the exact displaced-content signature (see the module
    docstring) against a section body's units. Returns a dict describing
    the pieces, or None if ANY condition fails.

    Expected unit sequence:
        header+  duty{2,}  DATE  header+ duty* [header+ duty*]...  DATE+
    with (number of trailing dates) == (number of header groups)."""
    count = len(units)
    position = 0

    while position < count and units[position]["kind"] == "header":
        position += 1
    if position == 0:
        return None
    first_header = units[:position]

    duty_start = position
    while position < count and units[position]["kind"] == "duty":
        position += 1
    first_duties = units[duty_start:position]
    if len(first_duties) < 2:
        return None

    if position >= count or units[position]["kind"] != "date":
        return None
    displaced_date = units[position]
    remainder = units[position + 1:]

    trailing_start = len(remainder)
    while trailing_start > 0 and remainder[trailing_start - 1]["kind"] == "date":
        trailing_start -= 1
    body = remainder[:trailing_start]
    trailing_dates = remainder[trailing_start:]

    if not body or body[0]["kind"] != "header":
        return None
    if any(unit["kind"] == "date" for unit in body):
        return None

    later_groups = []
    for unit in body:
        if unit["kind"] == "header":
            if later_groups and not later_groups[-1]["duties"]:
                later_groups[-1]["header"].append(unit)
            else:
                later_groups.append({"header": [unit], "duties": []})
        else:
            later_groups[-1]["duties"].append(unit)

    if len(trailing_dates) != 1 + len(later_groups):
        return None

    return {
        "first_header": first_header,
        "first_duties": first_duties,
        "displaced_date": displaced_date,
        "later_groups": later_groups,
        "trailing_dates": trailing_dates,
    }


def _repair_displaced_role_content(sections):
    """For each pair of consecutive role-style sections, if the previous
    section ends in a header-only block and the next section's body
    matches the displaced-content signature (see _match_displaced_pattern)
    AND the wrap widths show two text frames (see _find_frame_split), then:
      * the duty units after the split point, plus the displaced date, are
        appended verbatim to the previous section's header-only block;
      * the remaining entries of the next section are separated by blank
        lines, each followed by one of the trailing dates, in order.
    Mutates `sections` in place. Anything that does not meet every
    condition is left exactly as extracted.

    Returns a list of repair-event dicts, one per block actually rewritten
    by a firing of this repair (empty if this repair never fired). Like
    _reattach_orphan_leads()'s return value, this is PURE PROVENANCE
    bookkeeping for diagnostics (see parse_resume()'s "repair_applied"
    tagging below): it changes no condition and moves nothing extra. Two
    kinds of events are recorded when the repair fires: one for the
    previous section's header-only block (which received the moved duties
    and the displaced date), and one for EACH rebuilt block of the next
    section's body (since every one of those blocks had its own duties and
    date reassembled by this repair, not just passed through unchanged).
    """
    repair_events = []

    for index in range(1, len(sections)):
        previous_key, previous_lines = sections[index - 1]
        key, lines = sections[index]

        if key not in _ROLE_SECTION_KEYS or previous_key not in _ROLE_SECTION_KEYS:
            continue

        previous_units = _build_units(_last_block_lines(previous_lines))
        if not previous_units or any(unit["kind"] != "header" for unit in previous_units):
            continue

        pattern = _match_displaced_pattern(_build_units(lines))
        if pattern is None:
            continue

        split = _find_frame_split(pattern["first_duties"])
        if split is None:
            continue

        own_duties = pattern["first_duties"][:split]
        moved_duties = pattern["first_duties"][split:]
        trailing_dates = pattern["trailing_dates"]

        moved_lines = _flatten_units(moved_duties) + list(pattern["displaced_date"]["lines"])

        trimmed_previous = list(previous_lines)
        while trimmed_previous and not trimmed_previous[-1].strip():
            trimmed_previous.pop()

        blocks = [
            _flatten_units(pattern["first_header"] + own_duties)
            + list(trailing_dates[0]["lines"])
        ]
        for group_number, group in enumerate(pattern["later_groups"], start=1):
            blocks.append(
                _flatten_units(group["header"] + group["duties"])
                + list(trailing_dates[group_number]["lines"])
            )

        new_lines = []
        for block_number, block in enumerate(blocks):
            if block_number:
                new_lines.append("")
            new_lines.extend(block)

        sections[index - 1][1] = trimmed_previous + moved_lines
        sections[index][1] = new_lines

        repair_events.append({
            "type": "displaced_role_content_repair",
            "key": previous_key,
            "lines": list(moved_lines),
        })
        for block in blocks:
            repair_events.append({
                "type": "displaced_role_content_repair",
                "key": key,
                "lines": list(block),
            })

    return repair_events


# ============================================================
# 4. SKILLS: PRESERVE TEXT, ONLY SPLIT INTO ITEMS WHEN CLEARLY SEPARABLE
# ============================================================

def _split_skill_items(section_text):
    """Split the skills section into individual items, but only where the
    text clearly provides separable items (commas, semicolons, bullet
    lines). We do NOT guess a category (programming language vs framework
    vs tool) here - that classification is intentionally left out, per the
    requirement not to label a technology as a programming language just
    because it appears in the skills section."""
    items = []

    for raw_line in section_text.splitlines():
        line = raw_line.strip(" \t-*\u2022\u00b7\u25aa\u25cf")
        if not line:
            continue

        # A line like "Programming languages: Python, JavaScript, SQL"
        # -> only the part after the label is a list of items.
        if ":" in line:
            _, _, remainder = line.partition(":")
        else:
            remainder = line

        remainder = remainder.strip()
        if not remainder:
            continue

        if "," in remainder or ";" in remainder:
            current = []
            depth = 0
            for ch in remainder:
                if ch == "(":
                    depth += 1
                    current.append(ch)
                elif ch == ")":
                    if depth > 0:
                        depth -= 1
                    current.append(ch)
                elif ch in ",;" and depth == 0:
                    piece = "".join(current).strip()
                    if piece and piece not in items:
                        items.append(piece)
                    current = []
                else:
                    current.append(ch)

            tail = "".join(current).strip()
            if tail and tail not in items:
                items.append(tail)
        else:
            # A single item on its own line (no separator) is still a
            # clearly separable item - one line, one item.
            if remainder not in items:
                items.append(remainder)

    return items


def _build_skills_field(section_text):
    text = section_text.strip()
    return {
        "raw_text": text,
        "items": _split_skill_items(text) if text else [],
    }


# ============================================================
# 4b. REPAIR PROVENANCE: TAGGING ENTRIES WITH "repair_applied"
# ============================================================
# This section is pure diagnostics, added on top of the existing repairs
# (_reattach_orphan_leads, _repair_displaced_role_content) without changing
# any repair decision, without making a repair more aggressive, and without
# adding any new repair case. It only reports, for each already-produced
# experience/projects entry, whether one of those two repairs actually
# touched it.
#
# "Touched" is decided the same verbatim-lines way the module already
# reasons about repairs: an entry is tagged with a given repair event when
# at least one of the EXACT lines that event moved (see the "lines" field
# recorded by the repair functions) is present, verbatim, among the
# entry's own lines. Since both repairs only ever move lines verbatim
# (never rewriting or partially editing a line), this exact-line-overlap
# check is a precise, non-guessing way to trace a moved line back to the
# specific preserved entry it ended up in.

def _tag_entry_repair_provenance(entry, events_for_key):
    """Return "repair_applied" for one {"raw_text": ...} entry: the "type"
    of the first repair event (from events_for_key, already filtered to
    the entry's own canonical section key) whose moved lines overlap this
    entry's own lines, or None if no such event exists. Does not modify
    `entry` itself - the caller assigns the returned value."""
    if not events_for_key:
        return None

    entry_lines = set(entry["raw_text"].splitlines())
    if not entry_lines:
        return None

    for event in events_for_key:
        if entry_lines & set(event["lines"]):
            return event["type"]

    return None


# ============================================================
# 4c. ENTRY INTEGRITY DIAGNOSTICS (structural evidence only)
# ============================================================
# This section adds a purely DIAGNOSTIC, additive "structure" field to
# each already-produced experience/projects entry. It reports structural
# evidence that a future ATS analyzer could use to flag a potentially
# merged or structurally ambiguous entry - it does NOT split, merge,
# repair, or otherwise change any entry's raw_text or boundaries. Nothing
# about how entries are produced (sections 3, 3b, 3c, 3d above) changes
# because of this section.
#
# It deliberately reuses the SAME shape-based unit vocabulary already used
# by the displaced-content repair (_build_units, section 3d): every
# non-blank line of an entry is classified, using existing heuristics, as
# a "header" unit (a title/company/project line), a "duty" unit (a bullet
# or sentence-shaped line), or a "date" unit (a standalone date range,
# _is_date_range_line). This keeps the diagnostic layer small, consistent
# with the rest of the file, and grounded in logic that has already been
# validated by the repair functions above - it does not invent a second,
# parallel parsing system.
#
# WHAT COUNTS AS A "DATE GROUP" (conservative, see the module's usage
# instructions): a maximal run of ADJACENT "date" units in the entry's own
# unit sequence counts as exactly ONE date group, no matter how many
# duplicate/consecutive date lines it contains. This is what keeps a
# duplicated trailing date ("2025-2026" appearing twice back to back) from
# ever being counted as two separate date groups - only a date unit (or
# run of date units) that is separated from another by an intervening
# header or duty unit starts a SECOND group. This mirrors, at the
# diagnostic level, the same "duplicated trailing date stays with the
# entry it closes" conservatism the existing _as_experience_entries /
# _block_is_only_date_ranges logic already applies when SPLITTING entries.
#
# WHAT COUNTS AS "headers_after_duties": true only when a header-kind unit
# appears in the sequence AFTER at least one duty-kind unit has already
# been seen. An entry that starts with its header and is followed only by
# duties (the normal, expected shape) never trips this, regardless of how
# many duty lines, wrapped lines, or repeated words it has - only an
# entry whose unit sequence goes back to a header-shaped line after duty
# content has already begun can trip it.
#
# "suspicious" is true when EITHER of the above two conditions holds. No
# other condition (line count, duty count, wrapping, repeated words) is
# used, which is what keeps ordinary entries - however long or
# bullet-heavy - from ever being marked suspicious.

def _date_group_count(units):
    """Number of maximal consecutive runs of "date"-kind units in `units`.
    A run of two or more ADJACENT date units (e.g. a duplicated trailing
    date) still counts as ONE group; only a date unit/run that is
    separated from another by an intervening header or duty unit starts a
    new group. See the section docstring above for why this matters."""
    groups = 0
    previous_was_date = False

    for unit in units:
        is_date = unit["kind"] == "date"
        if is_date and not previous_was_date:
            groups += 1
        previous_was_date = is_date

    return groups


def _headers_after_duties(units):
    """True if a "header"-kind unit appears anywhere after the first
    "duty"-kind unit in `units`. An entry whose header(s) all come before
    its duties (the normal shape, regardless of how many duties it has)
    never trips this."""
    seen_duty = False

    for unit in units:
        if unit["kind"] == "duty":
            seen_duty = True
        elif unit["kind"] == "header" and seen_duty:
            return True

    return False


def _entry_structure_diagnostics(raw_text):
    """Return the "structure" diagnostic dict for one preserved
    experience/projects entry's raw_text. Purely additive/read-only: it
    only inspects raw_text (via the existing _build_units() shape
    classification) and never modifies it or the entry's boundaries.

    Fields:
        header_unit_count   - how many header-shaped units the entry has.
        duty_unit_count     - how many duty-shaped units the entry has.
        date_group_count    - how many separate date groups (see
                               _date_group_count above) the entry has.
        headers_after_duties - True if a header unit appears after duty
                               content has already started (see
                               _headers_after_duties above).
        multiple_date_groups - True if date_group_count > 1.
        suspicious          - True if EITHER headers_after_duties or
                               multiple_date_groups is True. This is the
                               only combination rule used: no line count,
                               duty count, or wording is ever considered,
                               so a normal entry - however long or
                               bullet-heavy - is never marked suspicious
                               on its own.
        reasons             - short, explainable strings, one per
                               triggered condition above (empty when
                               suspicious is False).
    """
    lines = raw_text.splitlines() if raw_text else []
    units = _build_units(lines)

    header_unit_count = sum(1 for unit in units if unit["kind"] == "header")
    duty_unit_count = sum(1 for unit in units if unit["kind"] == "duty")
    date_group_count = _date_group_count(units)

    headers_after_duties = _headers_after_duties(units)
    multiple_date_groups = date_group_count > 1

    reasons = []
    if headers_after_duties:
        reasons.append(
            "A header-like line appears after duty content has already "
            "started, which may indicate a second entry merged into this one."
        )
    if multiple_date_groups:
        reasons.append(
            "More than one separate date group was found in this entry, "
            "which may indicate multiple entries merged into one block."
        )

    return {
        "header_unit_count": header_unit_count,
        "duty_unit_count": duty_unit_count,
        "date_group_count": date_group_count,
        "headers_after_duties": headers_after_duties,
        "multiple_date_groups": multiple_date_groups,
        "suspicious": headers_after_duties or multiple_date_groups,
        "reasons": reasons,
    }


# ============================================================
# 4d. ENTRY FIELD DIAGNOSTICS (organization/role/date/duty relationships)
# ============================================================
# This section adds a second, purely DIAGNOSTIC, additive field
# ("fields") to each already-produced experience/projects entry. It
# reports what deterministic evidence exists for the entry's four core
# components (organization, role, date, duty content) and whether that
# evidence stays coherently associated within the entry - it does NOT
# split, merge, repair, or otherwise change any entry's raw_text or
# boundaries, and it does NOT replace or recompute Goal 2's "structure"
# diagnostic - it simply consumes that diagnostic's already-computed
# "suspicious" signal (via _entry_structure_diagnostics, unchanged) rather
# than inventing a second, competing notion of what is suspicious.
#
# WHAT COUNTS AS "date" evidence (two deterministic, shape-only sources):
#   1. a STANDALONE date unit - exactly the same _build_units() "date" kind
#      Goal 2 already uses (unchanged), and
#   2. a date-RANGE SUFFIX at the end of a HEADER-kind unit's line (see
#      _line_has_date_range_suffix below). On a real PDF a right-aligned
#      date is extracted onto the SAME physical row as its header line
#      ("Some Corporation - City  2025-2026"), which is not a standalone
#      date line, so source 1 alone produced a false "no date". Source 2 is
#      deliberately narrow: it must be a full range (two 4-digit years, or
#      a year and "present"), optionally with month names, preceded by
#      whitespace and ending the line, on a header unit only (a duty
#      sentence that merely mentions years is never a date), with plausible
#      years. It is only ever READ: no line is split, moved, or altered.
#
# WHAT COUNTS AS "duty" evidence: unchanged - the existing _build_units()
# "duty" kind (including its >=5-word rule for marker-less duty lines,
# which this section deliberately does not touch).
#
# ORGANIZATION / ROLE: THE PARSER HAS NO REAL SIGNAL FOR THIS. Both are
# just "header"-kind units under _build_units. The only generic vocabulary
# available is _ORG_KEYWORDS / _TITLE_KEYWORDS (already used by the name
# heuristic). A keyword hit on a line is a WEAK HINT, not identification:
# "University" or "Services" can sit in a project title, and "Corporation"
# proves nothing about which line is the employer. So this diagnostic uses
# three evidence levels per field (public keys organization_evidence /
# role_evidence):
#   "corroborated" - exactly ONE header unit hits only the organization
#                    vocabulary AND exactly ONE different header unit hits
#                    only the role vocabulary (no unit hits both). Two
#                    lines that each match their own vocabulary and
#                    neither the other's is the only structural evidence
#                    available that the lines play different roles. Only
#                    then are organization_present / role_present True.
#   "weak_hint"    - some line matched the vocabulary but the pairing above
#                    does not hold (several lines match, one line matches
#                    both, or only one side matched). present stays False
#                    and the field is listed in unconfirmed_fields, NOT in
#                    missing_expected_fields (something was hinted).
#   "none"         - no line matched at all (listed as missing).
# Relationship fields are only "clear"/"unclear" when BOTH sides are
# established; otherwise "unknown". Vocabulary hits still miss many real
# organizations/titles ("Google", "Founding Member"); that is reported as
# "none"/"weak_hint", never guessed.

# Years outside this window are not treated as plausible resume dates.
_MIN_PLAUSIBLE_YEAR = 1950
_MAX_PLAUSIBLE_YEAR = 2100

# "<whitespace or line start><start> <sep> <end>" at the END of a line,
# built from the same date-range pieces as _DATE_RANGE_LINE_RE above.
_DATE_RANGE_SUFFIX_RE = re.compile(
    rf"(?:^|\s)({_RANGE_START}\s*(?:-|\u2013|\u2014|\bto\b)\s*{_RANGE_END})\s*$",
    re.IGNORECASE,
)


def _line_has_date_range_suffix(line):
    """True if `line` ENDS with a clear date range (e.g. "... Manila
    2025-2026", "... 2025 \u2013 2026", "... May 2025 - August 2026",
    "... 2023 - Present") separated from the preceding text by whitespace
    and with plausible years. Arbitrary trailing numbers never match."""
    match = _DATE_RANGE_SUFFIX_RE.search(line.rstrip())
    if not match:
        return False

    years = [int(year) for year in re.findall(r"\d{4}", match.group(1))]
    return all(_MIN_PLAUSIBLE_YEAR <= year <= _MAX_PLAUSIBLE_YEAR for year in years)


def _classify_header_unit_org_role(unit_lines):
    """Return (looks_like_organization, looks_like_role) for one header
    unit's lines, using the SAME generic, already-defined organization/
    title-keyword vocabularies the module already uses for the name
    heuristic (_ORG_KEYWORDS, _TITLE_KEYWORDS - see
    _looks_like_non_name_line above). A header line can be BOTH (a title
    and organization combined on one row, e.g. "IT Intern at Example
    Corp") or NEITHER (an organization or title that doesn't happen to use
    one of these generic words) - (False, False) is itself honest,
    useful diagnostic information, not a failure of this function."""
    words = set()
    for line in unit_lines:
        words.update(re.findall(r"[A-Za-z]+", line.lower()))

    looks_like_organization = bool(words & _ORG_KEYWORDS)
    looks_like_role = bool(words & _TITLE_KEYWORDS)
    return looks_like_organization, looks_like_role


def _relationship_status(first_level, second_level, structure_suspicious):
    """Shared 3-way classifier for organization_role_relationship and
    role_date_relationship. Each level is "strong" (deterministically
    established), "weak" (vocabulary hint only) or "none".
      * "clear"   - BOTH sides strong and Goal 2 found nothing suspicious.
      * "unclear" - both sides strong but Goal 2 flagged the entry as
                    potentially merged; OR the entry is flagged and both
                    sides have at least a weak hint (so a pairing exists
                    but cannot be trusted).
      * "unknown" - anything else: not enough evidence to assess.
    Never introduces a suspicious trigger of its own."""
    if first_level == "strong" and second_level == "strong":
        return "unclear" if structure_suspicious else "clear"
    if structure_suspicious and first_level != "none" and second_level != "none":
        return "unclear"
    return "unknown"


def _entry_field_diagnostics(raw_text, entry_kind):
    """Return the "fields" diagnostic dict for one preserved
    experience/projects entry's raw_text (see the section docstring
    above). `entry_kind` ("experience" or "projects") only decides whether
    organization is expected: a project is not assumed to need a separate
    organization the way an employment entry does.

    Purely read-only: never modifies raw_text or boundaries, and consumes
    (does not recompute or override) Goal 2's "structure" suspicious flag.

    Keys: organization_present / role_present (True only when
    "corroborated"), organization_evidence / role_evidence
    ("corroborated" | "weak_hint" | "none"), date_present,
    date_evidence ("standalone" | "header_suffix" | "both" | "none"),
    duty_content_present, the two relationship keys, organization_units /
    role_units (count of units with a vocabulary HINT, not confirmed
    identifications), date_units (standalone units + header units carrying
    a date suffix), duty_units, missing_expected_fields (fields with NO
    evidence), unconfirmed_fields (fields with only a weak hint),
    suspicious (Goal 2's flag), reasons.
    """
    lines = raw_text.splitlines() if raw_text else []
    units = _build_units(lines)

    header_units = [unit for unit in units if unit["kind"] == "header"]
    duty_units_count = sum(1 for unit in units if unit["kind"] == "duty")
    standalone_date_units = sum(1 for unit in units if unit["kind"] == "date")

    org_only = role_only = both = 0
    organization_units = role_units = 0
    suffix_date_units = 0

    for unit in header_units:
        is_org, is_role = _classify_header_unit_org_role(unit["lines"])
        organization_units += is_org
        role_units += is_role
        if is_org and is_role:
            both += 1
        elif is_org:
            org_only += 1
        elif is_role:
            role_only += 1

        if any(_line_has_date_range_suffix(line) for line in unit["lines"]):
            suffix_date_units += 1

    corroborated = org_only == 1 and role_only == 1 and both == 0

    def _evidence(hint_units):
        if corroborated:
            return "corroborated"
        return "weak_hint" if hint_units > 0 else "none"

    organization_evidence = _evidence(organization_units)
    role_evidence = _evidence(role_units)

    date_present = (standalone_date_units + suffix_date_units) > 0
    if standalone_date_units and suffix_date_units:
        date_evidence = "both"
    elif standalone_date_units:
        date_evidence = "standalone"
    elif suffix_date_units:
        date_evidence = "header_suffix"
    else:
        date_evidence = "none"

    organization_present = organization_evidence == "corroborated"
    role_present = role_evidence == "corroborated"
    duty_content_present = duty_units_count > 0

    structure_suspicious = _entry_structure_diagnostics(raw_text)["suspicious"]

    level = {"corroborated": "strong", "weak_hint": "weak", "none": "none"}
    date_level = "strong" if date_present else "none"
    organization_role_relationship = _relationship_status(
        level[organization_evidence], level[role_evidence], structure_suspicious
    )
    role_date_relationship = _relationship_status(
        level[role_evidence], date_level, structure_suspicious
    )

    missing_expected_fields = []
    unconfirmed_fields = []
    if role_evidence == "none":
        missing_expected_fields.append("role")
    elif role_evidence == "weak_hint":
        unconfirmed_fields.append("role")
    if not date_present:
        missing_expected_fields.append("date")
    if not duty_content_present:
        missing_expected_fields.append("duty_content")
    if entry_kind == "experience":
        if organization_evidence == "none":
            missing_expected_fields.append("organization")
        elif organization_evidence == "weak_hint":
            unconfirmed_fields.append("organization")

    reasons = []
    if not date_present:
        reasons.append(
            "No standalone date-range line and no date-range suffix on a "
            "header line was found, so date presence could not be confirmed."
        )
    if role_evidence == "none":
        reasons.append("No line matched the generic role/title vocabulary.")
    elif role_evidence == "weak_hint":
        reasons.append(
            "A line matched the generic role/title vocabulary, but this is "
            "only a weak hint (not corroborated by a separate "
            "organization-vocabulary line), so role is not confirmed."
        )
    if organization_evidence == "none":
        reasons.append("No line matched the generic organization vocabulary.")
    elif organization_evidence == "weak_hint":
        reasons.append(
            "A line matched the generic organization vocabulary, but this "
            "is only a weak hint (not corroborated by a separate "
            "role-vocabulary line), so organization is not confirmed."
        )
    if structure_suspicious and "unclear" in (
        organization_role_relationship, role_date_relationship
    ):
        reasons.append(
            "This entry's own structure was flagged as potentially merged "
            "(Goal 2), so field relationships are not confidently clear."
        )

    return {
        "organization_present": organization_present,
        "role_present": role_present,
        "date_present": date_present,
        "duty_content_present": duty_content_present,
        "organization_evidence": organization_evidence,
        "role_evidence": role_evidence,
        "date_evidence": date_evidence,
        "organization_role_relationship": organization_role_relationship,
        "role_date_relationship": role_date_relationship,
        "organization_units": organization_units,
        "role_units": role_units,
        "date_units": standalone_date_units + suffix_date_units,
        "duty_units": duty_units_count,
        "missing_expected_fields": missing_expected_fields,
        "unconfirmed_fields": unconfirmed_fields,
        "suspicious": structure_suspicious,
        "reasons": reasons,
    }


# ============================================================
# 5. THE MAIN ENTRY POINT
# ============================================================

def parse_resume(resume_text):
    """Deterministically parse raw resume text into a structured dict.

    No LLM is used. Nothing is invented: a field is only populated when the
    corresponding information is actually present in resume_text.

    resume_text is passed through normalize_pdf_text() (section 0 above)
    FIRST, before any section/heading detection runs. This makes
    parse_resume() safe and correct to call with EITHER already-clean text
    OR raw pypdf output that exhibits the per-character-spacing artifact -
    normalize_pdf_text() is a no-op on text that is already clean, so
    calling it unconditionally never changes behavior for callers that
    already pass clean text.
    """
    result = {
        "candidate_name": None,
        "contact": {},
        "education": [],
        "experience": [],
        "skills": {"raw_text": "", "items": []},
        "projects": [],
        "certifications": [],
        "honors": [],
        "summary": {"raw_text": ""},
        "other_sections": {},
    }

    if not resume_text or not resume_text.strip():
        return result

    resume_text = normalize_pdf_text(resume_text)

    lines = resume_text.splitlines()

    # ---- Find every heading line and its canonical key ----
    heading_positions = []  # list of (line_index, canonical_key)
    for index, line in enumerate(lines):
        key = _heading_key(line)
        if key:
            heading_positions.append((index, key))

    heading_line_indices = {line_index for line_index, _ in heading_positions}

    # ---- Header portion: everything before the first heading ----
    first_heading_index = heading_positions[0][0] if heading_positions else len(lines)
    header_lines = [line for line in lines[:first_heading_index] if line.strip()]

    # Candidate name and contact info are both searched for across the
    # WHOLE document (not just the header): normal resumes have them in
    # the header, but some PDF layouts extract them elsewhere (e.g. a name
    # line inside a scrambled project block, or contact details that land
    # in the middle of another section). See _extract_name / _extract_contact.
    result["candidate_name"] = _extract_name(lines, heading_line_indices)
    result["contact"] = _extract_contact(lines)

    # Anything in the header that is neither the detected name nor part of
    # the detected contact info is preserved rather than discarded (e.g. a
    # degree line, or a professional-summary paragraph, sitting at the top
    # of the resume with no heading above it).
    contact_values = set(result["contact"].values())
    leftover_header_lines = [
        line for line in header_lines
        if line.strip() != (result["candidate_name"] or "")
        and not any(value and value in line for value in contact_values)
        and not _line_is_contact_only(line)
    ]
    if leftover_header_lines:
        result["other_sections"]["header"] = "\n".join(leftover_header_lines).strip()

    # ---- Collect each section's body lines, in order of appearance ----
    # Any line that is PURELY contact information (see _line_is_contact_only)
    # is dropped from every section body before that body is stored. This
    # is what keeps phone/email/LinkedIn lines that unusual PDF extraction
    # has placed inside, say, the Skills section from ending up mixed into
    # that section's skills.
    ordered_sections = []  # list of [canonical_key, body_lines], in order

    for position, (line_index, key) in enumerate(heading_positions):
        body_start = line_index + 1
        body_end = (
            heading_positions[position + 1][0]
            if position + 1 < len(heading_positions)
            else len(lines)
        )
        body_lines = [
            line for line in lines[body_start:body_end]
            if not _line_is_contact_only(line)
        ]
        ordered_sections.append([key, body_lines])

    # ---- Repair content displaced past the next heading (3c, 3d) ----
    # If extraction placed a role's duties/date after the NEXT heading,
    # move them back to the section they belong to BEFORE the bodies are
    # split into entries. This is what keeps project content out of
    # experience entries (and vice versa). Each repair is a no-op unless
    # its full structural signature is present.
    #
    # Each function's return value is repair-event PROVENANCE only (see
    # their docstrings): it is empty whenever the repair did not fire, and
    # otherwise records exactly which lines it moved into which section.
    # Nothing about the repairs themselves - what they check, what they
    # move - changes because of this; the events are used below purely to
    # tag the resulting preserved entries with "repair_applied" so a future
    # ATS analyzer can tell which specific entries were touched.
    repair_events = (
        _reattach_orphan_leads(ordered_sections)
        + _repair_displaced_role_content(ordered_sections)
    )

    # Sections that share a canonical key (e.g. two "PROJECTS" headings, or
    # "EXPERIENCE" + "INTERNSHIP") have their bodies concatenated with a
    # blank line in between, so _split_into_blocks() still treats them as
    # separate preserved entries.
    section_bodies = {}  # canonical_key -> list of body texts (in order)

    for key, body_lines in ordered_sections:
        body_text = "\n".join(body_lines).strip()
        section_bodies.setdefault(key, []).append(body_text)

    # Repair events, grouped by the canonical section key they targeted, so
    # tagging an entry only ever compares it against events for its OWN
    # section (an experience entry can never be tagged from a projects-only
    # event, and vice versa). Only "experience" and "projects" can ever
    # have entries here, because both repair functions only ever operate on
    # _ROLE_SECTION_KEYS.
    repair_events_by_key = {}
    for event in repair_events:
        repair_events_by_key.setdefault(event["key"], []).append(event)

    # ---- Fill in the output dict from the collected section bodies ----
    for key, bodies in section_bodies.items():
        combined_text = "\n\n".join(body for body in bodies if body)

        if key == "experience":
            # Experience alone gets the extra line-by-line boundary pass
            # (see section 3b); every other list section is unchanged.
            result[key] = _as_experience_entries(combined_text)
        elif key in _LIST_SECTION_KEYS:
            result[key] = _as_raw_text_entries(combined_text)
        elif key == "skills":
            result["skills"] = _build_skills_field(combined_text)
        elif key == "summary":
            result["summary"] = {"raw_text": combined_text}
        else:
            # A recognized heading with no dedicated top-level field
            # (currently just "interests"). Preserved, not discarded.
            if combined_text:
                result["other_sections"][key] = combined_text

        # ---- Tag "repair_applied" (diagnostics only) ----
        # Only "experience" and "projects" entries are ever checked, since
        # _reattach_orphan_leads() and _repair_displaced_role_content()
        # only ever operate on _ROLE_SECTION_KEYS - an education,
        # certifications, or honors entry can never have been touched by
        # either repair, so those entries are left exactly as they already
        # were (no "repair_applied" key added), matching the "smallest
        # additive implementation" this diagnostic is meant to be. This
        # never changes an entry's raw_text or which entries exist - it
        # only adds one extra key to entries that already exist.
        if key in _ROLE_SECTION_KEYS:
            events_for_key = repair_events_by_key.get(key, [])
            for entry in result[key]:
                entry["repair_applied"] = _tag_entry_repair_provenance(entry, events_for_key)

                # ---- Entry integrity diagnostics (Goal 2, diagnostics only) ----
                # Purely additive: computed from the entry's own already-final
                # raw_text via _entry_structure_diagnostics() (section 4c). This
                # never changes raw_text, never changes which entries exist, and
                # never changes repair_applied - it only adds one more read-only
                # key to entries that already exist, exactly like repair_applied
                # itself.
                entry["structure"] = _entry_structure_diagnostics(entry["raw_text"])

                # ---- Entry field diagnostics (Goal 3, diagnostics only) ----
                # Purely additive: computed from the entry's own already-final
                # raw_text (and its own canonical section key, `key`) via
                # _entry_field_diagnostics() (section 4d). This never changes
                # raw_text, never changes which entries exist, and never
                # changes repair_applied or structure - it only adds one more
                # read-only key to entries that already exist, exactly like
                # structure and repair_applied themselves.
                entry["fields"] = _entry_field_diagnostics(entry["raw_text"], key)

    return result