# AI Resume Analyzer

This project is a local, single-user resume analyzer built around a FastAPI backend, a Streamlit frontend, a PyMuPDF-based PDF extractor, and a local Ollama model call for the reasoning loop.

## Runtime architecture

- API backend: `api.py`
- Agent reasoning loop: `agent_loop.py`
- PDF extraction and visual reading-order reconstruction: `resume_extractor.py`
- Resume normalization and structured parsing: `resume_parser.py`
- Streamlit UX: `app.py`

The app expects an uploaded PDF resume, extracts text using PyMuPDF, normalizes the text, validates that it looks like a resume, and then uses the active resume text as grounded context for agent questions.

## Required tooling

- Python 3.12
- Ollama running locally
- Local model: `llama3.2:3b`

If needed, pull it with:

```bash
ollama pull llama3.2:3b
```

## Setup

```bash
cd ai-resume-analyzer
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run the app

Start the backend:

```bash
. .venv/bin/activate
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Then start the frontend in a second terminal:

```bash
. .venv/bin/activate
streamlit run app.py
```

## Local URLs

- API: http://127.0.0.1:8000/
- Health: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs
- Streamlit UI: http://localhost:8501

## Usage

1. Open the Streamlit app.
2. Upload a resume PDF.
3. Wait for the upload to finish successfully.
4. Ask questions only after the upload has succeeded.
5. The chat route uses the currently active resume in memory; it does not read from disk.

## Validation checks

The public repository includes deterministic checks for retrieval and resume parsing:

```bash
.venv/bin/python test_retrieval.py
.venv/bin/python test_resume_parser.py
```

## Notes

- `resume_extractor.py` uses PyMuPDF for the main PDF path and keeps `pypdf` as an optional last-resort fallback.
- `requirements.txt` lists the direct runtime dependencies used by the app.
