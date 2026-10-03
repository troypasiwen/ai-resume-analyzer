"""
ats_analyzer.py
===============

Goal 4: ATS PARSEABILITY ANALYZER.

One question, answered deterministically and explainably:

    "How reliably can this resume be machine-read and structurally
     interpreted?"

It does NOT answer "is this a good resume?", "will it get hired?", "will it
pass an ATS?" or "what is its ATS score?". Consequently there is NO score,
NO percentage, NO grade, NO ranking and NO overall verdict anywhere in the
output. The result is a flat list of individual, evidence-carrying checks.

INPUTS (consumed as-is, never recomputed)
-----------------------------------------
    extraction : resume_extractor.ExtractionResult (or a mapping with the
                 same keys, incl. "text"). Fields used: text, page_count,
                 method, layout_reordered, columns_detected,
                 pages_with_columns, fallback_used, notes.
    parsed     : the dict returned by resume_parser.parse_resume(). Fields
                 used: candidate_name, contact, education/experience/skills/
                 projects/certifications/honors/summary, other_sections, and
                 for every experience/projects entry the Goal 1/2/3 keys
                 "repair_applied", "structure" and "fields".

Goal 2's and Goal 3's detection logic is NOT duplicated here: this module
only reads the diagnostics those goals already attach to each entry and maps
them onto pass / warning / fail.

INDEPENDENCE
------------
No LLM, no Ollama, no ChromaDB, no embeddings, no job description, no agent
loop. Only the standard library plus resume_parser.HEADING_MAP (used to tell
"recognized heading without a dedicated field" apart from "unmapped text").
PyMuPDF is only needed by analyze_pdf_bytes()/the CLI, and is imported lazily
through resume_extractor.

STATUS MEANING
--------------
    pass    : the available evidence says this aspect was interpreted
              successfully.
    warning : ambiguity, fallback, repair activity, or a limitation - the
              resume can still be interpreted, but not with full confidence.
    fail    : evidence of a meaningful parseability failure (currently: no
              usable text, or text with no recognizable section at all).
Missing OPTIONAL information (a contact field, a section) is never a fail.

WHAT IS NOT A CHECK: NOT-ASSESSED AND UNAVAILABLE
-------------------------------------------------
The statuses are only pass/warning/fail, so a check that cannot honestly be
evaluated is NOT given a made-up status. It is listed instead in:
    diagnostics["not_assessed"] : checks skipped for THIS input (e.g. every
        parser-based check when no text was extracted, or an entry that
        lacks a diagnostics key).
    diagnostics["unavailable"]  : limitations of the CURRENT pipeline that
        apply to every input (image/non-text content cannot be detected,
        unrecognized headings are not reported by the parser, no OCR).

FAIL POLICY FOR FIELD RELATIONSHIPS
-----------------------------------
Goal 3 reports "none" both when a field is truly absent and when it merely
uses words outside the generic vocabulary ("Founding Member"). That cannot
support a claim of serious failure, so field-relationship checks are pass or
warning only.
"""

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field

from resume_parser import HEADING_MAP

# ============================================================
# 1. VOCABULARY
# ============================================================

STATUS_PASS = "pass"
STATUS_WARNING = "warning"
STATUS_FAIL = "fail"
STATUSES = (STATUS_PASS, STATUS_WARNING, STATUS_FAIL)

CAT_EXTRACTION = "EXTRACTION"
CAT_READING_ORDER = "READING_ORDER"
CAT_SECTIONS = "SECTIONS"
CAT_CONTACT = "CONTACT"
CAT_ENTRY_STRUCTURE = "ENTRY_STRUCTURE"
CAT_FIELD_RELATIONSHIPS = "FIELD_RELATIONSHIPS"
CAT_REPAIR_PROVENANCE = "REPAIR_PROVENANCE"
CAT_MACHINE_READABILITY = "MACHINE_READABILITY"
CATEGORIES = (
    CAT_EXTRACTION, CAT_READING_ORDER, CAT_SECTIONS, CAT_CONTACT,
    CAT_ENTRY_STRUCTURE, CAT_FIELD_RELATIONSHIPS, CAT_REPAIR_PROVENANCE,
    CAT_MACHINE_READABILITY,
)

# Fewer non-whitespace characters than this is treated as "barely any text".
# Zero is a fail; between 1 and this value is a warning.
MIN_USABLE_CHARS = 50

# Parser section keys that have a dedicated top-level field.
_LIST_SECTIONS = ("education", "experience", "projects", "certifications", "honors")
_FREE_TEXT_SECTIONS = ("skills", "summary")
_ALL_DEDICATED_SECTIONS = ("summary", "education", "experience", "projects",
                           "skills", "certifications", "honors")

# Canonical keys the parser's heading list can produce. An other_sections key
# in this set is a RECOGNIZED heading kept as free text (e.g. "interests");
# any other key (e.g. "header") is text the parser could not map.
_RECOGNIZED_HEADING_KEYS = frozenset(HEADING_MAP.values())

_ROLE_ENTRY_KINDS = ("experience", "projects")

# Contact fields reported on (the ones resume_parser.parse_resume exposes).
_CONTACT_FIELDS = ("name", "email", "phone", "location", "linkedin")
_CONTACT_EXTRA_FIELDS = ("github", "website")

_LABEL_MAX_CHARS = 80

_MISSING = object()


# ============================================================
# 2. RESULT TYPES
# ============================================================

@dataclass
class AtsCheck:
    id: str
    category: str
    status: str
    detail: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "category": self.category,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass
class AtsAnalysis:
    checks: list
    diagnostics: dict

    def to_dict(self):
        return {
            "checks": [check.to_dict() for check in self.checks],
            "diagnostics": self.diagnostics,
        }

    def get(self, check_id):
        """The check with this id, or None."""
        for check in self.checks:
            if check.id == check_id:
                return check
        return None

    def in_category(self, category):
        return [check for check in self.checks if check.category == category]


# ============================================================
# 3. SMALL HELPERS
# ============================================================

def _read(extraction, name):
    """Read one extractor field from an ExtractionResult-like object or a
    mapping; _MISSING if the field is not there (never guessed)."""
    if extraction is None:
        return _MISSING
    if isinstance(extraction, Mapping):
        return extraction.get(name, _MISSING)
    return getattr(extraction, name, _MISSING)


def _entry_label(raw_text):
    """First non-blank line of an entry, shortened. Identifies the entry in
    evidence using the resume's own text (nothing is hardcoded)."""
    for line in (raw_text or "").splitlines():
        if line.strip():
            label = " ".join(line.split())
            if len(label) > _LABEL_MAX_CHARS:
                label = label[:_LABEL_MAX_CHARS - 3].rstrip() + "..."
            return label
    return ""


def _not_assessed(check_id, category, reason):
    return {"id": check_id, "category": category, "reason": reason}


def _iter_role_entries(parsed):
    """Yield (kind, index, entry) for every experience/projects entry."""
    for kind in _ROLE_ENTRY_KINDS:
        entries = parsed.get(kind) or []
        for index, entry in enumerate(entries):
            if isinstance(entry, Mapping):
                yield kind, index, entry


def _entry_identity(kind, index, entry):
    return {
        "entry": _entry_label(entry.get("raw_text", "")),
        "type": kind,
        "index": index,
    }


# ============================================================
# 4. EXTRACTION / MACHINE-READABILITY / READING-ORDER CHECKS
# ============================================================

def _text_stats(text):
    return {
        "non_whitespace_chars": sum(1 for ch in text if not ch.isspace()),
        "word_count": len(text.split()),
        "line_count": sum(1 for line in text.splitlines() if line.strip()),
    }


def _check_extraction_success(text, stats, extraction, checks):
    chars = stats["non_whitespace_chars"]
    evidence = dict(stats)
    evidence["min_usable_chars"] = MIN_USABLE_CHARS
    page_count = _read(extraction, "page_count")
    if page_count is not _MISSING:
        evidence["page_count"] = page_count

    if chars == 0:
        status = STATUS_FAIL
        detail = "No text was extracted from the document."
    elif chars < MIN_USABLE_CHARS:
        status = STATUS_WARNING
        detail = (f"Only {chars} non-whitespace characters were extracted, "
                  f"below the {MIN_USABLE_CHARS}-character usability floor.")
    else:
        status = STATUS_PASS
        detail = f"Usable text was extracted ({chars} non-whitespace characters)."

    checks.append(AtsCheck("extraction_success", CAT_EXTRACTION, status, detail, evidence))


def _check_machine_readable_text(stats, extraction, checks):
    chars = stats["non_whitespace_chars"]
    evidence = {"non_whitespace_chars": chars, "ocr_available": False}
    for name in ("page_count", "method"):
        value = _read(extraction, name)
        if value is not _MISSING:
            evidence[name] = value

    if chars > 0:
        status = STATUS_PASS
        detail = "The document has an extractable text layer."
    else:
        status = STATUS_FAIL
        detail = ("No extractable text layer was found. The pipeline has no "
                  "OCR, so a scanned or image-only document cannot be read.")

    checks.append(AtsCheck("machine_readable_text", CAT_MACHINE_READABILITY,
                           status, detail, evidence))


def _check_extraction_fallback(extraction, checks, not_assessed):
    fallback_used = _read(extraction, "fallback_used")
    if fallback_used is _MISSING:
        not_assessed.append(_not_assessed(
            "extraction_fallback", CAT_EXTRACTION,
            "The extractor output does not report fallback_used."))
        return

    method = _read(extraction, "method")
    notes = _read(extraction, "notes")
    evidence = {
        "fallback_used": bool(fallback_used),
        "method": None if method is _MISSING else method,
        "notes": [] if notes is _MISSING or notes is None else list(notes),
    }

    if fallback_used:
        status = STATUS_WARNING
        detail = ("A fallback extraction path was used instead of the "
                  "layout-aware extractor.")
    else:
        status = STATUS_PASS
        detail = "Layout-aware extraction succeeded without needing a fallback."

    checks.append(AtsCheck("extraction_fallback", CAT_EXTRACTION, status, detail, evidence))


def _check_reading_order(extraction, has_text, checks, not_assessed):
    if not has_text:
        not_assessed.append(_not_assessed(
            "reading_order", CAT_READING_ORDER, "No text was extracted, so there is no order to assess."))
        return

    fallback_used = _read(extraction, "fallback_used")
    if fallback_used is _MISSING:
        not_assessed.append(_not_assessed(
            "reading_order", CAT_READING_ORDER,
            "The extractor output does not report fallback_used."))
        return

    def value(name, default=None):
        result = _read(extraction, name)
        return default if result is _MISSING or result is None else result

    evidence = {
        "method": value("method"),
        "fallback_used": bool(fallback_used),
        "layout_reordered": value("layout_reordered"),
        "columns_detected": value("columns_detected"),
        "pages_with_columns": list(value("pages_with_columns", [])),
        "notes": list(value("notes", [])),
    }

    if fallback_used:
        status = STATUS_WARNING
        detail = ("A fallback extractor produced this text, so its reading "
                  "order was not rebuilt from layout coordinates.")
    elif evidence["columns_detected"]:
        status = STATUS_WARNING
        detail = ("A multi-column layout was detected. The extractor read "
                  "the columns one after the other, but this order is "
                  "inferred from geometry and other readers may differ.")
    elif evidence["layout_reordered"]:
        status = STATUS_PASS
        detail = ("The PDF's stored text order differed from its visual "
                  "order, and layout-aware extraction rebuilt the visual "
                  "order.")
    else:
        status = STATUS_PASS
        detail = "Stored text order already matched the visual reading order."

    checks.append(AtsCheck("reading_order", CAT_READING_ORDER, status, detail, evidence))


# ============================================================
# 5. PARSER-BASED CHECKS: SECTIONS AND CONTACT
# ============================================================

def _section_is_present(parsed, key):
    value = parsed.get(key)
    if key in _LIST_SECTIONS:
        return bool(value)
    if key == "skills":
        return isinstance(value, Mapping) and bool((value.get("raw_text") or "").strip())
    if key == "summary":
        return isinstance(value, Mapping) and bool((value.get("raw_text") or "").strip())
    return False


def _check_sections(parsed, checks):
    detected = [key for key in _ALL_DEDICATED_SECTIONS if _section_is_present(parsed, key)]
    other = parsed.get("other_sections") or {}
    free_text = sorted(key for key in other if key in _RECOGNIZED_HEADING_KEYS)
    absent = [key for key in _ALL_DEDICATED_SECTIONS if key not in detected]

    evidence = {
        "detected_sections": detected,
        "recognized_free_text_sections": free_text,
        "sections_not_present": absent,
        "sections_not_present_note": "Optional; absence of a section is not a parseability failure.",
    }

    if detected or free_text:
        status = STATUS_PASS
        detail = (f"{len(detected) + len(free_text)} recognized section(s) were "
                  "detected from explicit headings.")
    else:
        status = STATUS_FAIL
        detail = ("Text was extracted but no recognized section heading was "
                  "found, so no content could be assigned to a section.")

    checks.append(AtsCheck("section_detectability", CAT_SECTIONS, status, detail, evidence))


def _check_unrecognized_structure(parsed, checks):
    other = parsed.get("other_sections") or {}
    unmapped = {key: text for key, text in other.items()
                if key not in _RECOGNIZED_HEADING_KEYS}

    if unmapped:
        evidence = {
            "unmapped_sections": sorted(unmapped),
            "line_counts": {key: sum(1 for line in str(text).splitlines() if line.strip())
                            for key, text in unmapped.items()},
        }
        status = STATUS_WARNING
        detail = ("Some content was extracted but not mapped to a recognized "
                  "structure: " + ", ".join(sorted(unmapped)) + ".")
    else:
        evidence = {"unmapped_sections": [], "line_counts": {}}
        status = STATUS_PASS
        detail = ("The parser reported no extracted content outside its "
                  "recognized structure. (It does not report unrecognized "
                  "headings; see diagnostics['unavailable'].)")

    checks.append(AtsCheck("unrecognized_structure", CAT_SECTIONS, status, detail, evidence))


def _check_contact(parsed, checks):
    contact = parsed.get("contact") or {}
    values = {}
    name = parsed.get("candidate_name")
    if name:
        values["name"] = name
    for key in _CONTACT_FIELDS[1:] + _CONTACT_EXTRA_FIELDS:
        if contact.get(key):
            values[key] = contact[key]

    detected = [key for key in _CONTACT_FIELDS + _CONTACT_EXTRA_FIELDS if key in values]
    not_detected = [key for key in _CONTACT_FIELDS if key not in values]
    evidence = {
        "detected": detected,
        "not_detected": not_detected,
        "detected_values": values,
    }

    has_name = "name" in values
    has_way_to_reach = "email" in values or "phone" in values

    if has_name and has_way_to_reach:
        status = STATUS_PASS
        detail = "A name and at least one direct contact method (email or phone) were detected."
    else:
        missing = []
        if not has_name:
            missing.append("a candidate name")
        if not has_way_to_reach:
            missing.append("an email or phone number")
        status = STATUS_WARNING
        detail = ("Not detected: " + " and ".join(missing) +
                  ". This may be absent from the resume or laid out in a way the parser did not recognize.")

    checks.append(AtsCheck("contact_information", CAT_CONTACT, status, detail, evidence))


# ============================================================
# 6. PARSER-BASED CHECKS: ENTRY-LEVEL (GOAL 1 / 2 / 3 DIAGNOSTICS)
# ============================================================

def _check_entry_structure(kind, index, entry, checks, not_assessed):
    check_id = f"entry_structure.{kind}.{index}"
    structure = entry.get("structure")

    if not isinstance(structure, Mapping) or structure.get("suspicious") is None:
        not_assessed.append(_not_assessed(
            check_id, CAT_ENTRY_STRUCTURE,
            "The entry has no Goal 2 'structure' diagnostics."))
        return

    evidence = _entry_identity(kind, index, entry)
    evidence.update({
        "reasons": list(structure.get("reasons") or []),
        "header_unit_count": structure.get("header_unit_count"),
        "duty_unit_count": structure.get("duty_unit_count"),
        "date_group_count": structure.get("date_group_count"),
        "headers_after_duties": structure.get("headers_after_duties"),
        "multiple_date_groups": structure.get("multiple_date_groups"),
    })

    if structure["suspicious"]:
        status = STATUS_WARNING
        detail = ("The entry's structure was flagged as possibly merged or "
                  "ambiguous by the parser's structural diagnostics.")
    else:
        status = STATUS_PASS
        detail = "The parser found no structural anomaly in this entry."

    checks.append(AtsCheck(check_id, CAT_ENTRY_STRUCTURE, status, detail, evidence))


def _relationships_considered(kind):
    # Goal 3 only expects an organization for employment-style entries, so
    # the organization<->role relationship only matters for "experience".
    if kind == "experience":
        return ("organization_role_relationship", "role_date_relationship")
    return ("role_date_relationship",)


def _check_field_relationships(kind, index, entry, checks, not_assessed):
    check_id = f"field_relationships.{kind}.{index}"
    fields = entry.get("fields")

    if not isinstance(fields, Mapping):
        not_assessed.append(_not_assessed(
            check_id, CAT_FIELD_RELATIONSHIPS,
            "The entry has no Goal 3 'fields' diagnostics."))
        return

    missing = list(fields.get("missing_expected_fields") or [])
    unconfirmed = list(fields.get("unconfirmed_fields") or [])
    considered = _relationships_considered(kind)

    issues = [f"missing:{name}" for name in missing]
    issues += [f"unconfirmed:{name}" for name in unconfirmed]
    for relationship in considered:
        status_value = fields.get(relationship)
        if status_value != "clear":
            issues.append(f"{relationship}:{status_value}")

    evidence = _entry_identity(kind, index, entry)
    evidence.update({
        "organization_evidence": fields.get("organization_evidence"),
        "role_evidence": fields.get("role_evidence"),
        "date_evidence": fields.get("date_evidence"),
        "organization_present": fields.get("organization_present"),
        "role_present": fields.get("role_present"),
        "date_present": fields.get("date_present"),
        "duty_content_present": fields.get("duty_content_present"),
        "organization_role_relationship": fields.get("organization_role_relationship"),
        "role_date_relationship": fields.get("role_date_relationship"),
        "relationships_considered": list(considered),
        "missing_expected_fields": missing,
        "unconfirmed_fields": unconfirmed,
        "structure_suspicious": fields.get("suspicious"),
        "issues": issues,
        "reasons": list(fields.get("reasons") or []),
    })

    if issues:
        status = STATUS_WARNING
        detail = ("Field evidence is incomplete or uncertain for this entry: "
                  + ", ".join(issues) + ".")
    else:
        status = STATUS_PASS
        detail = ("Role, date and duty content are established and their "
                  "relationships are clear.")

    checks.append(AtsCheck(check_id, CAT_FIELD_RELATIONSHIPS, status, detail, evidence))


def _check_repair_provenance(parsed, checks, not_assessed):
    assessed = 0
    repaired = []

    for kind, index, entry in _iter_role_entries(parsed):
        if "repair_applied" not in entry:
            continue
        assessed += 1
        if entry["repair_applied"]:
            item = _entry_identity(kind, index, entry)
            item["repair"] = entry["repair_applied"]
            repaired.append(item)

    if assessed == 0:
        not_assessed.append(_not_assessed(
            "repair_provenance", CAT_REPAIR_PROVENANCE,
            "There are no experience/projects entries carrying Goal 1 'repair_applied'."))
        return

    evidence = {"entries_checked": assessed, "repaired_entries": repaired}

    if repaired:
        status = STATUS_WARNING
        detail = (f"{len(repaired)} entr{'y' if len(repaired) == 1 else 'ies'} "
                  "needed content recovered or reattached by the parser. "
                  "This records that recovery was required, not that it failed.")
    else:
        status = STATUS_PASS
        detail = "No entry needed parser repair."

    checks.append(AtsCheck("repair_provenance", CAT_REPAIR_PROVENANCE, status, detail, evidence))


# ============================================================
# 7. DECLARED LIMITATIONS OF THE CURRENT PIPELINE
# ============================================================

def _unavailable_diagnostics():
    return [
        {
            "id": "non_text_content",
            "category": CAT_MACHINE_READABILITY,
            "reason": ("resume_extractor skips image blocks and does not "
                       "report image counts, skipped blocks or a text-coverage "
                       "measure, so text embedded in images (or images "
                       "carrying meaning) cannot be detected."),
        },
        {
            "id": "unrecognized_headings",
            "category": CAT_SECTIONS,
            "reason": ("resume_parser only matches an explicit heading list "
                       "and does not report unrecognized headings; text under "
                       "one is absorbed into the preceding section."),
        },
        {
            "id": "ocr",
            "category": CAT_MACHINE_READABILITY,
            "reason": ("The pipeline has no OCR: a scanned or image-only PDF "
                       "yields no text (reported by the extraction checks)."),
        },
    ]


# ============================================================
# 8. PUBLIC API
# ============================================================

def analyze_ats_parseability(extraction, parsed=None):
    """Return an AtsAnalysis for one resume.

    extraction: resume_extractor.ExtractionResult (or mapping with "text" and
                the same diagnostic keys).
    parsed:     resume_parser.parse_resume() output for extraction's text.
                When None, every parser-based check is reported under
                diagnostics["not_assessed"] rather than parsing here.

    Pure and read-only: neither argument is modified.
    """
    checks = []
    not_assessed = []

    text = _read(extraction, "text")
    text_available = isinstance(text, str)
    stats = _text_stats(text) if text_available else None
    has_text = bool(stats and stats["non_whitespace_chars"] > 0)

    # ---- extraction-level checks ----
    if text_available:
        _check_extraction_success(text, stats, extraction, checks)
        _check_machine_readable_text(stats, extraction, checks)
    else:
        for check_id, category in (("extraction_success", CAT_EXTRACTION),
                                   ("machine_readable_text", CAT_MACHINE_READABILITY)):
            not_assessed.append(_not_assessed(
                check_id, category, "No extractor output with a text field was provided."))

    _check_extraction_fallback(extraction, checks, not_assessed)
    _check_reading_order(extraction, has_text, checks, not_assessed)

    # ---- parser-based checks ----
    parser_checks = (
        ("section_detectability", CAT_SECTIONS),
        ("unrecognized_structure", CAT_SECTIONS),
        ("contact_information", CAT_CONTACT),
        ("entry_structure", CAT_ENTRY_STRUCTURE),
        ("field_relationships", CAT_FIELD_RELATIONSHIPS),
        ("repair_provenance", CAT_REPAIR_PROVENANCE),
    )
    entries_assessed = {kind: 0 for kind in _ROLE_ENTRY_KINDS}

    if not isinstance(parsed, Mapping):
        for check_id, category in parser_checks:
            not_assessed.append(_not_assessed(check_id, category, "No parsed resume was provided."))
    elif not has_text:
        for check_id, category in parser_checks:
            not_assessed.append(_not_assessed(check_id, category, "No text was extracted, so nothing was parsed."))
    else:
        _check_sections(parsed, checks)
        _check_unrecognized_structure(parsed, checks)
        _check_contact(parsed, checks)

        for kind, index, entry in _iter_role_entries(parsed):
            entries_assessed[kind] += 1
            _check_entry_structure(kind, index, entry, checks, not_assessed)
            _check_field_relationships(kind, index, entry, checks, not_assessed)

        if not any(entries_assessed.values()):
            for check_id, category in parser_checks[3:5]:
                not_assessed.append(_not_assessed(
                    check_id, category, "The parsed resume has no experience or projects entries."))

        _check_repair_provenance(parsed, checks, not_assessed)

    # ---- diagnostics ----
    extraction_report = {}
    for name in ("page_count", "method", "layout_reordered", "columns_detected",
                 "pages_with_columns", "fallback_used", "notes"):
        value = _read(extraction, name)
        if value is not _MISSING:
            extraction_report[name] = value
    if stats is not None:
        extraction_report.update(stats)

    diagnostics = {
        "scope": ("Machine-readability and structural interpretability only. "
                  "Not a quality judgement, a hiring prediction, or an ATS "
                  "pass/fail prediction; no score is produced."),
        "extraction": extraction_report,
        "parser": {
            "entries_assessed": entries_assessed,
            "candidate_name_detected": bool(isinstance(parsed, Mapping) and parsed.get("candidate_name")),
        },
        "not_assessed": not_assessed,
        "unavailable": _unavailable_diagnostics(),
    }

    return AtsAnalysis(checks=checks, diagnostics=diagnostics)


def analyze_pdf_bytes(pdf_bytes):
    """Convenience: PDF bytes -> extract -> parse -> analyze. Returns
    (AtsAnalysis, ExtractionResult, parsed). Needs PyMuPDF (via
    resume_extractor) but nothing else external."""
    from resume_extractor import extract_resume_text
    from resume_parser import parse_resume

    extraction = extract_resume_text(pdf_bytes)
    parsed = parse_resume(extraction.text)
    return analyze_ats_parseability(extraction, parsed), extraction, parsed


# ============================================================
# 9. COMMAND LINE
# ============================================================

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    paths = [arg for arg in argv if arg != "--json"]

    if len(paths) != 1:
        print("usage: python ats_analyzer.py <resume.pdf> [--json]")
        return 1

    with open(paths[0], "rb") as handle:
        analysis, _extraction, _parsed = analyze_pdf_bytes(handle.read())

    if as_json:
        print(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False))
        return 0

    for check in analysis.checks:
        print(f"[{check.status.upper():7}] {check.category:20} {check.id}")
        print(f"          {check.detail}")
    if analysis.diagnostics["not_assessed"]:
        print("\nNot assessed:")
        for item in analysis.diagnostics["not_assessed"]:
            print(f"  - {item['id']}: {item['reason']}")
    print("\nUnavailable in the current pipeline:")
    for item in analysis.diagnostics["unavailable"]:
        print(f"  - {item['id']}: {item['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())