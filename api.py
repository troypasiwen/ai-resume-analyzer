"""
api.py
======

A small FastAPI backend that exposes the EXISTING agent in agent_loop.py
through HTTP.

There is NO second agent in this file. All the agent logic (the loop, the
tools, the safeguards, the grounding check) stays in agent_loop.py. This file
does four jobs:

    1. RECEIVE  an HTTP request and check that it is well-formed
    2. TRACK    which resume text is currently "active" (see section 2 below)
    3. CALL     agent_loop.run_agent(question, active_resume_text)
    4. RETURN   the agent's result as JSON

NOTE ON THIS VERSION: layout-aware PDF extraction
--------------------------------------------------
POST /upload-resume no longer extracts text with pypdf's
page.extract_text(). pypdf walks a PDF's drawing operations in the order
they are STORED, which is not necessarily the order a reader SEES them (a
role's duties and right-aligned date can be stored after the next
section's heading). The upload now calls
resume_extractor.extract_resume_text(pdf_bytes), which uses PyMuPDF line
coordinates to rebuild the visual reading order. Its `.text` is what gets
normalized and stored. Everything after that step is unchanged.

pypdf is no longer imported by this file. (resume_extractor.py only uses it
as a last-resort fallback if PyMuPDF cannot open the bytes.)

NOTE ON PDF-TEXT NORMALIZATION (Step 21C)
------------------------------------------
Extracted PDF text sometimes carries a per-character spacing artifact -
"EDUCATION" comes back as "E D U C A T I O N", "JavaScript" as
"J a v a S c r i p t". resume_parser.normalize_pdf_text() deterministically
undoes this (see resume_parser.py for the full algorithm; no LLM is
involved).

The extracted text is normalized before its normalized version becomes the active resume.

The same four routes exist, the same in-memory "active resume" model is
used, and agent_loop.run_agent() is still the only thing that ever talks to
Ollama. (agent_loop.py also normalizes defensively wherever it resolves an
active resume text - see agent_loop._resolve_resume_text - so the agent is
correct even if it is ever called with un-normalized text from somewhere
other than this file.)

NOTE ON RESUME-CONTENT VALIDATION
----------------------------------
A PDF can be perfectly valid and contain plenty of text and still not be a
resume ("Hello World / This is a test document."). Before this version,
such a file was accepted and REPLACED the active resume, so /chat would
then answer questions about a document that is not a resume at all.

POST /upload-resume now runs a small, deterministic content check AFTER the
text has been normalized and BEFORE it is made the active resume. The check
(see looks_like_a_resume() in section 3B) looks for simple textual
evidence in three categories - contact details, resume section headings,
and professional/academic terms - and requires evidence from at least TWO
different categories. A PDF that fails the check is rejected with HTTP 400
and the active resume is left exactly as it was.

No LLM, no Ollama, no embeddings, no ChromaDB and no network call is used
for this check - just the standard-library "re" module.

What FastAPI is
---------------
FastAPI is a Python web framework. You describe URLs ("routes") as normal
Python functions, and FastAPI handles the HTTP part for you:
    * it listens for requests (through a server called Uvicorn),
    * it reads the JSON body and validates it with Pydantic models,
    * it calls your function,
    * it turns whatever your function returns back into JSON.
It also creates interactive documentation for free at /docs.

WHAT'S IN "active resume" (in-memory state)
---------------------------------------------------
        * A module-level variable (_active_resume_text, see section 2) holds
            whatever resume text should currently be used by /chat.
        * It starts empty so there is no silent fallback to sample candidate data.
        * When POST /upload-resume successfully extracts text from a PDF, that
            extracted text is NORMALIZED (see above), checked to make sure it
            really looks like a resume (see above), and only then does the
            normalized text REPLACE the active resume text.
        * POST /chat reads the CURRENT active resume text and passes it into
            run_agent(question, active_resume_text), so every answer is grounded
            in whatever was most recently uploaded.

WHY THIS IS IN-MEMORY (not saved to disk):
    This is intentionally a TEMPORARY, in-process cache for this stage of
    the project. It is stored in a plain Python variable inside this
    running server process:
        * It is NOT written to disk anywhere, so the uploaded PDF's text
          (raw or normalized) never touches the filesystem.
                * It does NOT touch resume.txt - that file is left completely
                    alone, in case other parts of the project (or a future step)
                    still rely on it.
                * It resets to an empty string whenever the server restarts, and
                    (in a single-process deployment like `uvicorn api:app`) is shared
                    by every request - there is no per-user/per-session separation yet.
                    That is fine for this single-user prototype stage; a later step
                    (with real users/sessions) would replace this with a proper
                    per-session or per-request store.

How a request travels (POST /chat)
----------------------------------
    Client sends:   POST /chat   {"question": "What programming languages does the candidate know?"}
         |
         v
    FastAPI         matches the URL + method to the chat() function below
         |
         v
    Pydantic        ChatRequest checks the body (not empty, not too long)
         |           -> invalid body = automatic HTTP 422, chat() never runs
         v
    chat()          reads the ACTIVE (normalized) resume text
         |           (get_active_resume_text()) and calls
         |           run_agent(question, active_resume_text)
         v
    run_agent()     runs the loop (Ollama + tools + safeguards), with every
         |           tool searching the ACTIVE resume text, and returns a dict:
         |           {"status", "answer", "iterations", "tools_executed",
         |            "duplicates_skipped", "tools_refused"}
         v
    chat()          turns that dict into a ChatResponse
         |
         v
    FastAPI         sends the ChatResponse back to the client as JSON

How a request travels (POST /upload-resume)
--------------------------------------------
    Client sends:   POST /upload-resume   (multipart/form-data, a PDF file)
         |
         v
    FastAPI         matches the URL + method to the upload_resume() function below
         |
         v
    upload_resume() checks the file LOOKS like a PDF (content type and/or
         |           filename extension) -> not a PDF = HTTP 415
         v
    extract_resume_text()  (resume_extractor.py) PyMuPDF reads the uploaded
         |                 bytes and rebuilds the text in VISUAL reading
         |                 order from line coordinates; the result's .text
         |                 is the extracted text
         v
    upload_resume() if nothing came out (e.g. scanned/image-only PDF)
         |           -> HTTP 400 explaining that
         v
    normalize_pdf_text()   deterministically undoes the per-character
         |                 spacing artifact, if the extracted text shows it
         v
    looks_like_a_resume()  deterministic check (no LLM): does the
         |                 normalized text show evidence from at least TWO
         |                 different categories (contact / section headings
         |                 / professional-academic terms)?
         |                 -> no = HTTP 400, active resume NOT changed
         v
    upload_resume() on success, calls set_active_resume_text(normalized)
         |           so /chat will use the NORMALIZED version of THIS
         |           resume from now on
         v
    FastAPI         sends filename, page count, raw text AND normalized
                     text back as JSON

Nothing here talks to Ollama or any external API during upload. That
endpoint only reads a PDF, extracts its text, normalizes it, validates it,
and updates the in-memory active resume. Ollama and the agent are still
only used by /chat.

How to run it
-------------
    pip install fastapi uvicorn pymupdf python-multipart
    (pypdf is optional now: resume_extractor.py only uses it as a
     last-resort fallback: pip install pypdf)
    uvicorn api:app --reload

Then open:
    http://127.0.0.1:8000/        -> "API is running" message
    http://127.0.0.1:8000/health  -> {"status": "ok"}
    http://127.0.0.1:8000/docs    -> interactive page where you can try
                                      POST /chat and POST /upload-resume

Make sure Ollama is running locally first (the model stays local).
api.py, agent_loop.py, resume_parser.py and resume_extractor.py must be in
the SAME folder.

Everything stays local: no OpenAI, no Anthropic, no Gemini, no paid API.

NOTE ON THE STREAMLIT APP: this file does not change anything about how the
Streamlit app talks to this API. Connecting the upload button in Streamlit
to POST /upload-resume is a later step.
"""


# ============================================================
# 1. IMPORTS
# ============================================================

import asyncio
import logging
import re
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

# resume_extractor.py turns PDF bytes into text in VISUAL reading order
# (PyMuPDF line coordinates instead of pypdf's stored drawing order).
#   extract_resume_text(pdf_bytes) -> ExtractionResult with .text,
#       .page_count, .method, .fallback_used, .notes, ...
#   ResumeExtractionError          -> raised only if NO extraction path can
#                                     open the bytes as a PDF
# An image-only PDF does NOT raise: it returns an empty .text.
from resume_extractor import ResumeExtractionError, extract_resume_text

# These names really exist in agent_loop.py:
#   run_agent            - runs the agent loop for ONE question, grounded in
#                           ONE resume, and returns a dict
#   MAX_QUESTION_LENGTH  - the maximum question length the agent's CLI accepts
#   MODEL_NAME           - the local Ollama model name (used only for the info message)
#
# Importing agent_loop is safe: its main() only runs when you execute
# "python agent_loop.py" directly (it is guarded by if __name__ == "__main__").
from agent_loop import MAX_QUESTION_LENGTH, MODEL_NAME, run_agent, warm_up_ollama

# normalize_pdf_text is the same deterministic, no-LLM helper agent_loop.py
# uses internally (see agent_loop._resolve_resume_text). Importing it
# directly here lets /upload-resume normalize the PDF text once, up front,
# and keep the normalized version as the active resume without returning
# resume contents to the client.
from resume_parser import normalize_pdf_text


# ============================================================
# 2. ACTIVE RESUME STATE (temporary, in-memory)
# ============================================================
# WHAT THIS IS: a single module-level variable holding the resume text that
# /chat should currently use. It lives only in this process's memory for as
# long as the server keeps running. It always holds NORMALIZED text (see
# upload_resume() below) - normalize_pdf_text() is a no-op on text that
# does not show the per-character-spacing artifact, so this is exactly
# equivalent to the raw text whenever that artifact isn't present.
#
# WHY IT IS STORED IN MEMORY (and not on disk):
#   * The task at this stage is explicitly "temporary in-memory storage" -
#     the uploaded PDF's text should become usable by /chat immediately,
#     without being written anywhere permanent.
#   * It must NEVER modify resume.txt, and it doesn't: nothing in this file
#     opens, reads, or writes that file.
#   * A restart of the server naturally clears it back to an empty string,
#     which is the correct fail-closed behavior for a prototype like this.
#
# HOW IT IS PASSED TO THE AGENT:
#   chat() reads it with get_active_resume_text() and passes it straight
#   into agent_loop.run_agent(question, resume_text). run_agent hands that
#   same text to every tool call for that one question (see agent_loop.py,
#   section 16), so the tools search whatever resume is currently active
#   instead of a hardcoded one.
#
# A lock guards reads/writes because FastAPI can run several "def" (sync)
# routes concurrently in its worker thread pool; the lock just makes sure a
# read never sees a half-written value. It is not meant to provide any
# stronger isolation than that (see the module docstring's note about this
# being a single, process-wide value for now, not a per-user one).
_active_resume_lock = threading.Lock()
_active_resume_text = ""


def get_active_resume_text():
    """Return the resume text /chat should currently use."""
    with _active_resume_lock:
        return _active_resume_text


def set_active_resume_text(new_text):
    """Replace the active resume text (called after a successful PDF upload).
    Callers are expected to have already normalized new_text (see
    upload_resume()); this function does not normalize again itself, since
    it may also be used to restore a value that is already known to be
    normalized."""
    global _active_resume_text
    with _active_resume_lock:
        _active_resume_text = new_text


# ============================================================
# 3. LOGGING
# ============================================================
# Real error details (including stack traces) are written to the SERVER log,
# which only you can see. They are never sent to the API client.

logger = logging.getLogger("resume_api")


# ============================================================
# 3B. RESUME-CONTENT VALIDATION (deterministic, no LLM)
# ============================================================
# WHY THIS EXISTS:
#   The PDF checks in upload_resume() only prove that the file is a real PDF
#   and that it contains SOME text. They say nothing about WHAT the text is.
#   Without a content check, a PDF containing "Hello World / This is a test
#   document." would be accepted and would REPLACE the active resume, and
#   /chat would then "answer questions about the candidate" using a document
#   that is not a resume at all.
#
# WHY IT USES DETERMINISTIC CHECKS INSTEAD OF AN LLM:
#   * It runs in microseconds and needs nothing to be running (no Ollama).
#   * It gives the SAME verdict for the SAME text every time, so an upload is
#     never accepted on one attempt and rejected on the next.
#   * It cannot be "talked into" anything by text inside the PDF (an LLM
#     reading uploaded content could be influenced by it).
#   * It keeps the upload endpoint free of any model or network dependency,
#     matching the rule that /upload-resume never talks to Ollama.
#
# WHY MORE THAN ONE CATEGORY IS REQUIRED:
#   Any single indicator on its own is weak evidence. An email address
#   appears in invoices and newsletters; the word "experience" appears in
#   marketing copy; "manager" and "university" appear in news articles. A
#   real resume, however, almost always shows evidence from SEVERAL kinds at
#   once (how to contact the person, resume-style section headings, and
#   job/education vocabulary). Requiring at least TWO different categories
#   rejects random text while staying flexible: no particular layout,
#   heading style, order or exact wording is required, and a resume that
#   omits (say) a phone number or a "Skills" heading still passes.
#
# HOW MATCHING WORKS:
#   Each category is a list of case-insensitive regular expressions. A
#   category "matches" if AT LEAST ONE of its patterns is found ANYWHERE in
#   the text. Word boundaries (\b) are used for the plain words so that, for
#   example, "degree" does not match inside an unrelated longer word. The
#   validation is applied to the NORMALIZED text, so headings that arrived
#   as "E D U C A T I O N" have already been repaired by that point.

# Minimum number of DIFFERENT categories that must match.
MIN_RESUME_CATEGORIES = 2
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024

# Category 1 - contact indicators (how to reach the candidate).
_CONTACT_PATTERNS = [
    # Email address, e.g. name@example.com
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}",
    # LinkedIn profile mention or URL
    r"\blinkedin\b",
    # Phone number: optional +country code, then digit groups separated by
    # spaces, dots or dashes (e.g. +63 917 123 4567, 0917-123-4567,
    # (02) 8123-4567, 555.123.4567) ...
    r"(?<![\w/])(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.\-]\d{3,4}[\s.\-]\d{3,4}(?!\w)",
    # ... or one unbroken run of 10-13 digits, optionally with a leading +
    # (e.g. 09171234567, +639171234567).
    r"(?<![\w.])\+?\d{10,13}(?![\w.])",
]

# Category 2 - resume section indicators (typical section headings).
_SECTION_PATTERNS = [
    r"\beducation\b",
    r"\bacademic\s+background\b",
    r"\bexperience\b",           # also covers "Work / Professional Experience"
    r"\bwork\s+experience\b",
    r"\bemployment\s+history\b",
    r"\bprofessional\s+experience\b",
    r"\bskills?\b",              # also covers "Technical Skills"
    r"\btechnical\s+skills?\b",
    r"\bcore\s+competenc(?:y|ies)\b",
    r"\bprojects?\b",            # also covers "Project Experience"
    r"\bproject\s+experience\b",
    r"\bcertifications?\b",
    r"\bcertificates?\b",
    r"\bachievements?\b",
    r"\bawards?\b",
    r"\bhonou?rs?\b",
]

# Category 3 - professional / academic indicators (job and education words).
_PROFESSIONAL_PATTERNS = [
    r"\binterns?\b",
    r"\binternships?\b",
    r"\bdevelopers?\b",
    r"\bengineer(?:s|ing)?\b",
    r"\bmanagers?\b",
    r"\banalysts?\b",
    r"\bdesigners?\b",
    r"\bspecialists?\b",
    r"\bbachelor(?:'s|s)?\b",
    r"\bmaster(?:'s|s)?\b",
    r"\bdegrees?\b",
    r"\buniversity\b",
    r"\bcollege\b",
]

# Category name -> compiled patterns. Compiled once at import time, so each
# upload only pays for the searches themselves.
_RESUME_INDICATOR_CATEGORIES = {
    "contact": [re.compile(p, re.IGNORECASE) for p in _CONTACT_PATTERNS],
    "resume_sections": [re.compile(p, re.IGNORECASE) for p in _SECTION_PATTERNS],
    "professional_academic": [re.compile(p, re.IGNORECASE) for p in _PROFESSIONAL_PATTERNS],
}


def matched_resume_categories(text):
    """Return the names of the indicator categories that have at least one
    match in `text` (a list, in a fixed order, possibly empty)."""
    matched = []
    for category_name, patterns in _RESUME_INDICATOR_CATEGORIES.items():
        if any(pattern.search(text) for pattern in patterns):
            matched.append(category_name)
    return matched


def looks_like_a_resume(text):
    """Return True if `text` shows evidence from at least
    MIN_RESUME_CATEGORIES different indicator categories, else False.

    Purely deterministic: no LLM, no network, no files - just regular
    expressions over the text that is passed in."""
    return len(matched_resume_categories(text)) >= MIN_RESUME_CATEGORIES


# ============================================================
# 4. REQUEST / RESPONSE MODELS (Pydantic)
# ============================================================
# Pydantic models describe the exact shape of the JSON going in and out.
# FastAPI uses them to:
#   * validate incoming JSON automatically,
#   * shape the outgoing JSON,
#   * generate the documentation at /docs.

class ChatRequest(BaseModel):
    """The JSON body the client must send to POST /chat."""

    # str_strip_whitespace removes spaces around the text BEFORE the length
    # checks run, so a question made only of spaces counts as empty.
    model_config = ConfigDict(str_strip_whitespace=True)

    # min_length=1     -> empty questions are rejected (HTTP 422)
    # max_length=...   -> same limit the terminal version of the agent uses
    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="The question to ask about the candidate's resume.",
        examples=["What programming languages does the candidate know?"],
    )


class ChatResponse(BaseModel):
    """The JSON the client receives from POST /chat."""

    # The agent's answer text.
    answer: str

    # How many times the agent asked the LLM for a decision.
    iterations: int

    # How many tools Python really executed (skipped duplicates and refused
    # requests are NOT counted).
    tools_executed: int

    # True  -> the agent ended normally with a "final" answer, which passed
    #          through the agent's grounding check (ground_final_answer) against
    #          the tool results before it was returned.
    # False -> Python had to stop the agent (iteration limit, repetition loop,
    #          or tools locked). The answer is then a safe, evidence-only
    #          message built by Python instead of a normal final answer.
    grounded: bool

    # Explicit verification flag from the agent itself. This is the canonical
    # grounding result and must not be inferred only from status == "final".
    grounding_verified: bool

    # How the agent run ended: "final", "limit", "loop" or "locked".
    status: str

    # Extra counters from the agent's safeguards (handy while learning).
    duplicates_skipped: int
    tools_refused: int


class UploadResumeResponse(BaseModel):
    """The JSON the client receives from POST /upload-resume."""

    # The original filename the client uploaded (as sent by the client).
    filename: str

    # How many pages the extractor found in the PDF.
    pages: int


# ============================================================
# 5. THE FASTAPI APPLICATION
# ============================================================
# "app" is the object Uvicorn runs (the "api:app" in the uvicorn command means
# "the variable called app inside api.py"). Every route below is registered on it.

_ollama_model_ready = False


@asynccontextmanager
async def lifespan(_application):
    global _ollama_model_ready
    _ollama_model_ready = False
    try:
        await asyncio.to_thread(warm_up_ollama)
    except Exception:
        logger.warning(
            "Ollama model warm-up unavailable; the API is starting without a ready model."
        )
    else:
        _ollama_model_ready = True

    try:
        yield
    finally:
        _ollama_model_ready = False


app = FastAPI(
    title="AI Resume Analyzer API",
    description="HTTP backend for the local Ollama resume agent (agent_loop.py).",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 6. ROUTES
# ============================================================

@app.get("/")
def root():
    """GET /  ->  a simple message showing that the API is up."""
    return {"message": "AI Resume Analyzer API is running"}


@app.get("/health")
def health():
    """GET /health  ->  a tiny endpoint that monitoring tools can poll.

    It reports whether the API process is alive and whether its startup
    Ollama warm-up succeeded. It does NOT call Ollama, so it stays instant.
    """
    return {"status": "ok", "model_ready": _ollama_model_ready}


# This function is a normal "def" (not "async def") on purpose.
# run_agent() is synchronous and can take a while (a local 3B model on CPU is
# slow). FastAPI runs normal "def" routes in a worker thread pool, so one slow
# question does not freeze the whole web server.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """POST /chat  ->  ask the existing agent one question about the
    CURRENTLY ACTIVE resume (the most recently uploaded PDF, or no resume if
    nothing has been uploaded yet).

    By the time this function runs, FastAPI + Pydantic have already:
      * parsed the JSON body,
      * confirmed it has a non-empty "question" string of allowed length.
    (Otherwise the client already received an automatic HTTP 422 error.)
    """
    global _ollama_model_ready
    question = request.question

    # ---- Read whichever resume is currently active ----
    # This is the bridge between /upload-resume and /chat: whatever text was
    # last stored by set_active_resume_text() (see section 2) is what the
    # agent will be grounded in for this question. It is already normalized
    # (see upload_resume() below), so run_agent() and its tools can match
    # keywords against it directly.
    active_resume_text = get_active_resume_text()

    if not active_resume_text or not active_resume_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No resume is loaded. Please upload a PDF resume before asking candidate-specific questions.",
        )

    # ---- Call the EXISTING agent ----
    # This single line is the whole bridge between HTTP and the agent.
    # run_agent() runs the full loop (Ollama + tools + all safeguards),
    # passing the active resume text through to every tool call, and
    # returns a dict. It handles Ollama/decision errors itself and reports
    # them as status "error", but any bug we did not foresee could still
    # raise, so we catch that too.
    try:
        outcome = run_agent(question, active_resume_text)

    except Exception:
        # logger.exception() writes the full stack trace to the SERVER log only.
        logger.exception("Unexpected error while running the agent")
        raise HTTPException(
            status_code=500,
            detail="Internal error while processing the question.",
        )

    # ---- The agent reported a known failure (Ollama down, unusable decision) ----
    # run_agent() returns status "error" instead of raising. The text in
    # outcome["answer"] is an error message, not a resume answer, so we do not
    # return it as a normal 200 response. We log the details for you and send
    # the client a short, safe message.
    if outcome["status"] == "error":
        _ollama_model_ready = False
        logger.error("Agent stopped with an error: %s", outcome["answer"])
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI agent could not complete the request. "
                f"Make sure Ollama is running and the model '{MODEL_NAME}' is available."
            ),
        )

    _ollama_model_ready = True

    # ---- Build the response ----
    # The dict returned by run_agent() is copied into the ChatResponse model.
    # FastAPI then converts it to JSON and sends it to the client.
    return ChatResponse(
        answer=outcome["answer"],
        iterations=outcome["iterations"],
        tools_executed=outcome["tools_executed"],
        grounded=bool(outcome.get("grounding_verified", False)),
        grounding_verified=bool(outcome.get("grounding_verified", False)),
        status=outcome["status"],
        duplicates_skipped=outcome["duplicates_skipped"],
        tools_refused=outcome["tools_refused"],
    )


# This endpoint does not touch Ollama in any way - it only reads a PDF,
# extracts its text, normalizes it, validates that it looks like a resume,
# and (on success) updates the in-memory active resume that /chat reads on
# every request.
@app.post("/upload-resume", response_model=UploadResumeResponse)
def upload_resume(file: UploadFile = File(...)):
    """POST /upload-resume  ->  accept a PDF resume, extract its text in
    visual reading order, NORMALIZE it, check that it really looks like a
    resume, and make the normalized text the ACTIVE resume used by /chat.

    This endpoint is intentionally simple for now:
      1. Check the upload looks like a PDF (content type and/or filename).
      2. Extract the text with resume_extractor.extract_resume_text()
         (layout-aware, PyMuPDF-based).
      3. Normalize the extracted text (undo the per-character-spacing
         artifact, if present - see resume_parser.normalize_pdf_text()).
      4. Validate that the normalized text contains enough resume evidence
         (deterministic check, no LLM - see looks_like_a_resume()).
      5. Make the NORMALIZED text the ACTIVE resume (in memory only).

    It does NOT call Ollama, does NOT call any external API, and does NOT
    save the file to disk - the upload only exists in memory, first for the
    duration of this request; only the normalized text remains active until
    the next successful upload or a server restart. Embeddings/ChromaDB come
    in a later step.
    """

    # ---- 1. Validate that this is actually a PDF ----
    # We check two things because clients are not always consistent:
    #   * the declared content type (what the browser/client says the file is)
    #   * the filename extension (a simple, reliable fallback)
    # If EITHER one clearly says "PDF", we accept it. If NEITHER does, we
    # reject with 415 Unsupported Media Type (the standard status code for
    # "I understand the request, but not this file format").
    content_type_is_pdf = file.content_type == "application/pdf"
    filename_is_pdf = bool(file.filename) and file.filename.lower().endswith(".pdf")

    if not (content_type_is_pdf or filename_is_pdf):
        raise HTTPException(
            status_code=415,
            detail=(
                "Only PDF files are accepted. Please upload a file with "
                "content type 'application/pdf' and/or a '.pdf' extension."
            ),
        )

    # ---- 2. Read the upload with a strict size cap ----
    # Read at most one byte beyond the limit so oversized uploads are rejected
    # without loading their full contents into memory.
    try:
        pdf_bytes = file.file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    except Exception:
        logger.exception("Failed to read the uploaded file")
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded file. Please try again.",
        )
    finally:
        # Always close the upload, whether reading succeeded or not.
        file.file.close()

    if len(pdf_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Uploaded PDF exceeds the maximum allowed size of 10 MiB.",
        )

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # ---- 3. Extract the text in VISUAL reading order ----
    # extract_resume_text() takes the raw PDF bytes directly, so nothing is
    # saved to disk. It uses each text line's coordinates (PyMuPDF) instead
    # of the order the PDF happens to store its drawing operations in, which
    # is what keeps a role's duties and date next to that role. It returns
    # an ExtractionResult; its .text is the extracted text.
    try:
        extraction = extract_resume_text(pdf_bytes)
    except ResumeExtractionError:
        # No extraction path could open the bytes - it isn't a valid PDF.
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as a PDF. It may be corrupted or not a real PDF.",
        )
    except Exception:
        # Anything else unexpected - log full details server-side, send a
        # safe generic message to the client (no stack trace).
        logger.exception("Unexpected error while extracting text from the PDF")
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF could not be processed while extracting text.",
        )

    # If the layout-aware path did not produce the text (PyMuPDF missing or
    # failed, so a plain/pypdf fallback ran), the reading order may not be
    # reliable. The upload still succeeds, but the server log says so.
    if extraction.fallback_used:
        logger.warning(
            "Layout-aware extraction was not used for %r (method=%s, notes=%s)",
            file.filename,
            extraction.method,
            extraction.notes,
        )

    page_count = extraction.page_count

    # Trim outer whitespace so the response is tidy. This is the RAW
    # extractor text (it may still show the per-character-spacing artifact).
    raw_text = extraction.text.strip()

    # ---- 4. Make sure we actually got something useful ----
    # If the extractor found no text, this is very likely a scanned/image-only
    # PDF with no text layer (there is no OCR), so we tell the client
    # clearly what happened instead of returning an empty result. Note that
    # the active resume is NOT changed in this case, so /chat keeps using
    # whatever resume was active before this failed upload.
    if not raw_text:
        raise HTTPException(
            status_code=400,
            detail=(
                "No text could be extracted from this PDF. It may contain "
                "scanned/image-only pages or have no extractable text."
            ),
        )

    # ---- 5. Normalize the per-character-spacing artifact, if present ------
    # normalize_pdf_text() is deterministic, uses no LLM, and only ever
    # reconstructs spacing from characters that are already in raw_text -
    # it never invents missing content. It is also a no-op on text that
    # does not show the artifact, so this is always safe to run.
    normalized_text = normalize_pdf_text(raw_text)

    # ---- 6. Check that the text actually looks like a resume --------------
    # WHY: a PDF can be valid and contain text and still not be a resume
    # (e.g. "Hello World / This is a test document."). Accepting it would
    # make that text the active resume and /chat would answer questions
    # about a document that is not a resume.
    #
    # HOW: looks_like_a_resume() (section 3B) is a deterministic, regex-based
    # check - deliberately NOT an LLM (fast, repeatable, no Ollama needed, and
    # not influenced by instructions hidden inside the uploaded text). It
    # requires evidence from at least TWO different categories (contact
    # details, resume section headings, professional/academic terms), because
    # a single weak indicator (an email address, the word "experience") also
    # shows up in many non-resume documents, while a real resume almost
    # always shows several kinds at once - without requiring any particular
    # layout.
    #
    # WHY THE ACTIVE RESUME IS NOT CHANGED ON FAILURE: this check runs BEFORE
    # set_active_resume_text() below, and raising HTTPException leaves this
    # function immediately, so a rejected non-resume PDF can never replace
    # the resume that /chat is currently using. A previous active resume
    # stays unchanged; if none was uploaded, no resume remains active.
    if not looks_like_a_resume(normalized_text):
        logger.warning(
            "Rejected upload %r: text present but only matched resume "
            "categories %s (need at least %d)",
            file.filename,
            matched_resume_categories(normalized_text),
            MIN_RESUME_CATEGORIES,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded PDF contains text, but it does not appear to "
                "contain enough resume information. Please upload a resume "
                "or CV."
            ),
        )

    # ---- 7. Make the NORMALIZED text the ACTIVE resume ----
    # From this point on, every /chat question is answered against this
    # normalized text, until either another PDF is uploaded or the server
    # restarts. This is the ONLY place in the whole application that
    # changes the active resume. resume.txt is never opened, read, or
    # written here; only normalized_text is retained as active state.
    set_active_resume_text(normalized_text)

    # ---- 8. Return the result ----
    return UploadResumeResponse(
        filename=file.filename,
        pages=page_count,
    )


# ============================================================
# 7. RUNNING THIS FILE DIRECTLY (optional shortcut)
# ============================================================
# "python api.py" works the same as "uvicorn api:app". Without reload mode
# the app object is passed directly, which keeps this shortcut simple.

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)