"""
resume_extractor.py
===================

One job: PDF bytes -> text in sensible VISUAL reading order.

    PDF bytes
      -> PyMuPDF text lines (each with its bounding box)
      -> (optional) column detection
      -> group lines into visual ROWS (same baseline = same row)
      -> emit rows top-to-bottom, with a blank line at paragraph gaps
      -> text

The output is plain text meant to be fed to the existing
resume_parser.normalize_pdf_text() -> parse_resume() chain. Nothing in this
file knows anything about resumes' CONTENT: no names, companies, schools or
dates. Everything is decided from geometry (x/y coordinates, gaps, widths).

WHY pypdf FAILED
----------------
A PDF does not store text "in reading order". It stores drawing operations
in whatever order the authoring tool emitted them (often one text frame at a
time, in creation order). pypdf's extract_text() walks those operations in
stored order and throws the coordinates away, so a role's duties and
right-aligned date can be emitted after the NEXT section's heading even
though they sit above it on the page. PyMuPDF keeps each line's bounding box,
so we can sort by position instead of trusting the stored order. (PyMuPDF's
own block order is ALSO the stored order, so this module deliberately ignores
block order and works from individual lines.)

HOW THE ORDERING WORKS
----------------------
1. LINES. Every text line PyMuPDF reports (spans merged) with its bbox.
   Image blocks and blank lines are ignored.

2. COLUMN DETECTION (per page). A gutter is a vertical strip of x positions
   that almost no line crosses. A candidate is accepted only if BOTH sides
   hold real body text (several lines of 20+ characters, a meaningful share
   of all characters) and the two sides overlap vertically. That last part is
   what stops a single-column resume with right-aligned dates from being
   mistaken for two columns: the "right side" of such a page is only short
   date fragments, and full-width paragraphs cross every candidate strip.

3. SEGMENTS. With a gutter, lines that cross it (a full-width name/header
   line) are "spanners". Between spanners, the lines are split into a left
   and a right column; each column is read top-to-bottom, left before right.
   Without a gutter the whole page is one segment.

4. ROWS. Inside a segment, lines whose vertical extents overlap by at least
   half of the shorter one are the same visual row and are joined left to
   right (separated by two spaces). This is how a right-aligned date stays
   attached to the title on its own row, no matter where the PDF stored it.
   Rows are NOT formed across the gutter, so two columns never interleave.

5. PARAGRAPH GAPS. A large vertical gap between consecutive rows becomes a
   blank line (the separator resume_parser already uses to split entries).
   Gaps are relative to the page's median line height: a big gap always
   breaks; a moderate gap breaks only when the next row starts bold and the
   previous one does not (see the tunables).

FALLBACKS
---------
If the layout path fails or yields no text, plain PyMuPDF extraction
(sort=True) is tried, then pypdf if it is installed. The result reports
which path produced the text.
"""

import io
import statistics
from dataclasses import dataclass, field

# ---- Tunables (all relative to the page's own geometry, never absolute
# ---- to one document) ----------------------------------------------------

# Two lines are the same visual row if their vertical overlap is at least
# this fraction of the SHORTER line's height.
_ROW_OVERLAP_RATIO = 0.5

# Gaps between rows are measured as a fraction of the page's median line
# height. A gap above _BIG_GAP_RATIO is always a paragraph break (blank line).
# A gap above _MODERATE_GAP_RATIO is a paragraph break only when the next row
# starts with BOLD text and the previous row does not (a new heading / entry
# header after body text). This style check is what separates "gap before a
# new entry" from "slightly loose spacing between a role line and its duties",
# which can be nearly the same size in a real PDF.
_BIG_GAP_RATIO = 1.0
_MODERATE_GAP_RATIO = 0.45

# Text fragments on the same row are joined with this.
_ROW_JOINER = "  "

# Column detection.
_MIN_LINES_FOR_COLUMNS = 8
_GUTTER_SCAN_MARGIN = 0.15      # ignore the outer 15% of the text width
_GUTTER_MAX_CROSSING = 0.15     # a gutter strip may be crossed by <=15% of lines
_GUTTER_MIN_WIDTH_PT = 8.0
_SUBSTANTIAL_LINE_CHARS = 20    # "body text", not a date or a short label
_MIN_SUBSTANTIAL_LINES_PER_SIDE = 3
_MIN_SIDE_CHAR_SHARE = 0.15
_MIN_VERTICAL_OVERLAP = 0.30


class ResumeExtractionError(ValueError):
    """The bytes could not be read as a PDF by any extraction path."""


@dataclass
class ExtractionResult:
    text: str
    page_count: int
    method: str = "layout"            # "layout", "pymupdf_plain" or "pypdf"
    layout_reordered: bool = False    # final order differs from the PDF's stored order
    columns_detected: bool = False    # a two-column gutter was found on any page
    fallback_used: bool = False
    pages_with_columns: list = field(default_factory=list)  # 1-based page numbers
    notes: list = field(default_factory=list)

    def diagnostics(self):
        return {
            "page_count": self.page_count,
            "method": self.method,
            "layout_reordered": self.layout_reordered,
            "columns_detected": self.columns_detected,
            "pages_with_columns": self.pages_with_columns,
            "fallback_used": self.fallback_used,
            "notes": self.notes,
        }


# ============================================================
# Line collection
# ============================================================

class _Line:
    __slots__ = ("text", "x0", "y0", "x1", "y1", "order", "bold")

    def __init__(self, text, bbox, order, bold=False):
        self.text = text
        self.bold = bold
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.order = order  # position in the PDF's stored order

    @property
    def height(self):
        return max(self.y1 - self.y0, 0.01)

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2.0


def _collect_lines(page):
    """Every non-empty text line on the page, with its bbox and its index in
    the PDF's stored order. Image blocks and malformed fragments are skipped."""
    lines = []
    order = 0
    text_dict = page.get_text("dict") or {}
    for block in text_dict.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines") or []:
            if not isinstance(raw_line, dict):
                continue
            raw_bbox = raw_line.get("bbox")
            if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
                continue
            spans = raw_line.get("spans") or []
            if not isinstance(spans, list):
                continue
            cleaned_spans = []
            for span in spans:
                if not isinstance(span, dict):
                    continue
                text_part = span.get("text")
                if text_part is None:
                    continue
                cleaned_spans.append(span)
            if not cleaned_spans:
                continue
            text = "".join(str(span.get("text", "")) for span in cleaned_spans)
            text = text.replace("\u00a0", " ").strip()
            if not text:
                continue
            first = next((sp for sp in cleaned_spans if str(sp.get("text", "")).strip()), {})
            flags = first.get("flags", 0)
            bold = bool(flags & 16) or "bold" in str(first.get("font", "")).lower()
            lines.append(_Line(text, raw_bbox, order, bold))
            order += 1
    return lines


# ============================================================
# Column detection
# ============================================================

def _find_gutter(lines):
    """Return (gutter_x0, gutter_x1) if the lines form two real columns,
    otherwise None. See the module docstring, step 2."""
    if len(lines) < _MIN_LINES_FOR_COLUMNS:
        return None

    xmin = min(l.x0 for l in lines)
    xmax = max(l.x1 for l in lines)
    span = xmax - xmin
    if span < 100:
        return None

    scan_start = int(xmin + _GUTTER_SCAN_MARGIN * span)
    scan_end = int(xmax - _GUTTER_SCAN_MARGIN * span)
    limit = _GUTTER_MAX_CROSSING * len(lines)

    # Find contiguous runs of x positions crossed by few lines.
    runs, run_start = [], None
    for x in range(scan_start, scan_end + 1):
        crossing = sum(1 for l in lines if l.x0 + 1 < x < l.x1 - 1)
        if crossing <= limit:
            if run_start is None:
                run_start = x
        elif run_start is not None:
            runs.append((run_start, x - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, scan_end))

    runs = [r for r in runs if r[1] - r[0] >= _GUTTER_MIN_WIDTH_PT]

    total_chars = sum(len(l.text) for l in lines)
    for g0, g1 in sorted(runs, key=lambda r: r[1] - r[0], reverse=True):
        mid = (g0 + g1) / 2.0
        left = [l for l in lines if l.x1 <= g1 and l.cy is not None and (l.x0 + l.x1) / 2 < mid]
        right = [l for l in lines if l.x0 >= g0 and (l.x0 + l.x1) / 2 >= mid]

        def substantial(side):
            return [l for l in side if len(l.text) >= _SUBSTANTIAL_LINE_CHARS]

        ls, rs = substantial(left), substantial(right)
        if (len(ls) < _MIN_SUBSTANTIAL_LINES_PER_SIDE
                or len(rs) < _MIN_SUBSTANTIAL_LINES_PER_SIDE):
            continue

        if (sum(len(l.text) for l in left) < _MIN_SIDE_CHAR_SHARE * total_chars
                or sum(len(l.text) for l in right) < _MIN_SIDE_CHAR_SHARE * total_chars):
            continue

        # Real columns run side by side: their vertical extents must overlap.
        l0, l1 = min(l.y0 for l in ls), max(l.y1 for l in ls)
        r0, r1 = min(l.y0 for l in rs), max(l.y1 for l in rs)
        overlap = min(l1, r1) - max(l0, r0)
        shorter = min(l1 - l0, r1 - r0)
        if shorter <= 0 or overlap / shorter < _MIN_VERTICAL_OVERLAP:
            continue

        return g0, g1

    return None


# ============================================================
# Ordering
# ============================================================

def _segments(lines, gutter):
    """Split the page's lines into an ordered list of segments. Each segment
    is a list of lines that may share rows with each other."""
    if gutter is None:
        return [sorted(lines, key=lambda l: (l.cy, l.x0))]

    g0, g1 = gutter
    mid = (g0 + g1) / 2.0

    def is_spanner(l):
        return l.x0 < g0 and l.x1 > g1

    segments, left, right = [], [], []

    def flush():
        nonlocal left, right
        if left:
            segments.append(sorted(left, key=lambda l: (l.cy, l.x0)))
        if right:
            segments.append(sorted(right, key=lambda l: (l.cy, l.x0)))
        left, right = [], []

    for l in sorted(lines, key=lambda l: (l.cy, l.x0)):
        if is_spanner(l):
            flush()
            segments.append([l])
        elif (l.x0 + l.x1) / 2.0 < mid:
            left.append(l)
        else:
            right.append(l)
    flush()
    return segments


def _build_rows(segment):
    """Group a segment's lines into visual rows (same baseline band).
    Returns a list of (row_lines_sorted_by_x, y0, y1)."""
    rows = []  # each: {"lines": [...], "y0":, "y1":}
    for l in segment:  # already sorted by vertical centre
        placed = False
        if rows:
            row = rows[-1]
            overlap = min(l.y1, row["y1"]) - max(l.y0, row["y0"])
            shorter = min(l.height, row["y1"] - row["y0"])
            if overlap >= _ROW_OVERLAP_RATIO * shorter:
                row["lines"].append(l)
                row["y0"] = min(row["y0"], l.y0)
                row["y1"] = max(row["y1"], l.y1)
                placed = True
        if not placed:
            rows.append({"lines": [l], "y0": l.y0, "y1": l.y1})

    return [(sorted(r["lines"], key=lambda x: x.x0), r["y0"], r["y1"]) for r in rows]


def _page_text_from_lines(lines):
    """Return (text, columns_detected, reordered) for one page's lines."""
    if not lines:
        return "", False, False

    gutter = _find_gutter(lines)
    median_height = statistics.median(l.height for l in lines)
    big_gap = _BIG_GAP_RATIO * median_height
    moderate_gap = _MODERATE_GAP_RATIO * median_height

    out, emitted = [], []
    for segment in _segments(lines, gutter):
        if out and gutter is not None:
            out.append("")  # a new column/spanner segment starts a new paragraph
        previous_y1, previous_bold = None, False
        for row_lines, y0, y1 in _build_rows(segment):
            row_bold = row_lines[0].bold
            if previous_y1 is not None:
                gap = y0 - previous_y1
                if gap > big_gap or (gap > moderate_gap and row_bold and not previous_bold):
                    out.append("")
            out.append(_ROW_JOINER.join(l.text for l in row_lines))
            emitted.extend(row_lines)
            previous_y1, previous_bold = y1, row_bold

    reordered = [l.order for l in emitted] != sorted(l.order for l in emitted)
    return "\n".join(out).strip("\n"), gutter is not None, reordered


# ============================================================
# Fallbacks
# ============================================================

def _extract_pymupdf_plain(doc):
    parts = [page.get_text("text", sort=True).strip() for page in doc]
    return "\n\n".join(p for p in parts if p).strip()


def _extract_pypdf(pdf_bytes):
    from pypdf import PdfReader  # optional dependency, only for the last resort
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip(), len(reader.pages)


# ============================================================
# Public API
# ============================================================

def extract_resume_text(pdf_bytes):
    """PDF bytes -> ExtractionResult (text in visual reading order plus
    diagnostics). Raises ResumeExtractionError only if NO extraction path can
    open the bytes; an image-only PDF returns an empty `text` instead."""
    if not pdf_bytes:
        raise ResumeExtractionError("Empty PDF data.")

    result = None
    doc = None
    try:
        import pymupdf
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        pymupdf_error = exc
    else:
        pymupdf_error = None

    if doc is not None:
        page_count = doc.page_count
        try:
            page_texts, any_columns, any_reordered, column_pages = [], False, False, []
            for number, page in enumerate(doc, start=1):
                text, columns, reordered = _page_text_from_lines(_collect_lines(page))
                if text:
                    page_texts.append(text)
                any_columns |= columns
                any_reordered |= reordered
                if columns:
                    column_pages.append(number)
            layout_text = "\n\n".join(page_texts).strip()
            if layout_text:
                result = ExtractionResult(
                    text=layout_text, page_count=page_count, method="layout",
                    layout_reordered=any_reordered, columns_detected=any_columns,
                    pages_with_columns=column_pages,
                )
        except Exception as exc:
            pymupdf_error = exc

        if result is None:
            note = (f"layout extraction failed: {pymupdf_error}" if pymupdf_error
                    else "layout extraction found no text")
            try:
                plain = _extract_pymupdf_plain(doc)
            except Exception:
                plain = ""
            result = ExtractionResult(
                text=plain, page_count=page_count, method="pymupdf_plain",
                fallback_used=True, notes=[note],
            )
        doc.close()
        return result

    # PyMuPDF could not open the bytes at all: last resort is pypdf.
    try:
        text, page_count = _extract_pypdf(pdf_bytes)
    except Exception as exc:
        raise ResumeExtractionError(
            f"Could not read the PDF (PyMuPDF: {pymupdf_error}; pypdf: {exc})."
        ) from exc
    return ExtractionResult(
        text=text, page_count=page_count, method="pypdf", fallback_used=True,
        notes=[f"PyMuPDF unavailable or failed: {pymupdf_error}"],
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python resume_extractor.py <file.pdf>")
        sys.exit(1)
    with open(sys.argv[1], "rb") as handle:
        extracted = extract_resume_text(handle.read())
    print(extracted.text)
    print("\n--- diagnostics ---")
    print(extracted.diagnostics())