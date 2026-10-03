import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/chat"
UPLOAD_API_URL = "http://127.0.0.1:8000/upload-resume"

# Local 3B models can be slow, so allow plenty of time.
REQUEST_TIMEOUT_SECONDS = 180


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# BACKEND CALL
# ============================================================

def ask_backend(question):
    """
    Send the question to the FastAPI backend.

    Returns a tuple: (data, error_message)
    - On success: (dict, None)
    - On failure: (None, "friendly error message")
    """

    try:

        response = requests.post(
            API_URL,
            json={"question": question},
            timeout=REQUEST_TIMEOUT_SECONDS
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError:

        return None, (
            "Could not connect to the AI backend. "
            "Make sure FastAPI is running at http://127.0.0.1:8000."
        )

    except requests.exceptions.Timeout:

        return None, (
            "The AI backend took too long to respond. "
            "The model may still be loading - please try again."
        )

    except requests.exceptions.HTTPError as error:

        status_code = error.response.status_code

        # FastAPI usually puts its error message in a "detail" field.
        try:
            detail = error.response.json().get("detail", "")
        except ValueError:
            detail = ""

        message = f"The AI backend returned an error (HTTP {status_code})."

        if detail:
            message += f" Details: {detail}"

        return None, message

    except requests.exceptions.RequestException:

        return None, (
            "Something went wrong while contacting the AI backend."
        )

    # Parse the JSON body
    try:

        data = response.json()

    except ValueError:

        return None, "The AI backend did not return valid JSON."

    if not isinstance(data, dict) or "answer" not in data:

        return None, "The AI backend response did not include an answer."

    return data, None


def upload_resume_to_backend(uploaded_file):
    """Send one selected PDF to the existing upload endpoint.

    Returns (response_data, error_message, outcome), where outcome is
    "success", "rejected", or "unknown". Unknown means the server may have
    processed the upload but the client could not verify the response.
    """
    try:
        response = requests.post(
            UPLOAD_API_URL,
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/pdf",
                )
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

    except requests.exceptions.HTTPError as error:
        status_code = error.response.status_code
        try:
            detail = error.response.json().get("detail", "")
        except (ValueError, AttributeError):
            detail = ""

        message = f"The upload failed (HTTP {status_code})."
        if detail:
            message += f" {detail}"
        return None, message, "rejected"

    except requests.exceptions.Timeout:
        return (
            None,
            "The upload request timed out. Its result could not be confirmed.",
            "unknown",
        )

    except requests.exceptions.ConnectionError:
        return (
            None,
            "Could not connect to the AI backend. The upload result could not be confirmed.",
            "unknown",
        )

    except requests.exceptions.RequestException:
        return (
            None,
            "The upload request could not be completed. Its result could not be confirmed.",
            "unknown",
        )

    if response.status_code != 200:
        return (
            None,
            f"The upload returned an unexpected HTTP status ({response.status_code}).",
            "unknown",
        )

    try:
        data = response.json()
    except ValueError:
        return None, "The upload response was not valid JSON.", "unknown"

    if (
        not isinstance(data, dict)
        or not isinstance(data.get("filename"), str)
        or not data["filename"].strip()
        or not isinstance(data.get("pages"), int)
    ):
        return None, "The upload response was missing valid resume details.", "unknown"

    return data, None, "success"


# ============================================================
# TITLE
# ============================================================

st.title("AI Resume Analyzer")

st.write(
    "Ask a question about the candidate's resume. "
    "The question is sent to the local AI agent, which uses "
    "its resume tools to find a grounded answer."
)


# ============================================================
# RESUME UPLOAD
# ============================================================

if "resume_upload_state" not in st.session_state:
    st.session_state.resume_upload_state = "none"
if "active_resume_filename" not in st.session_state:
    st.session_state.active_resume_filename = None

st.subheader("Resume")

with st.form("resume_upload_form", clear_on_submit=False):
    uploaded_file = st.file_uploader(
        "Choose a PDF resume",
        type=["pdf"],
        accept_multiple_files=False,
    )
    upload_clicked = st.form_submit_button("Upload resume")

if upload_clicked:
    if uploaded_file is None:
        st.error("Choose a PDF file before uploading.")
    else:
        with st.spinner("Uploading and validating resume..."):
            upload_data, upload_error, upload_outcome = upload_resume_to_backend(uploaded_file)

        if upload_outcome == "success":
            st.session_state.resume_upload_state = "active"
            st.session_state.active_resume_filename = upload_data["filename"]
            st.success(
                f"Resume uploaded successfully: {upload_data['filename']}. "
                "It is now active for questions."
            )
        elif upload_outcome == "rejected":
            st.error(f"Upload failed. {upload_error}")
            if st.session_state.resume_upload_state == "active":
                st.info(
                    "The previous active resume remains unchanged: "
                    f"{st.session_state.active_resume_filename}."
                )
            else:
                st.info("No resume became active. Upload a valid PDF before asking questions.")
        else:
            st.session_state.resume_upload_state = "unknown"
            st.session_state.active_resume_filename = None
            st.error(f"Upload status unknown. {upload_error}")
            st.warning(
                "The backend may have received the file, but its response could not be verified. "
                "Upload a PDF again before asking questions."
            )

if st.session_state.resume_upload_state == "active":
    st.success(f"Active resume: {st.session_state.active_resume_filename}")
elif st.session_state.resume_upload_state == "unknown":
    st.warning("The active resume could not be confirmed. Upload a PDF before asking questions.")
else:
    st.info("Upload a resume first.")


# ============================================================
# CHAT INPUTS
# ============================================================

can_chat = st.session_state.resume_upload_state == "active"

question = st.text_area(
    "Your Question",
    height=150,
    placeholder="Example: What skills are listed in the resume?",
    disabled=not can_chat,
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("Ask Question", disabled=not can_chat):

    if not question.strip():

        st.error("Please enter a question.")

    else:

        # ----------------------------------------------------
        # CALL FASTAPI
        # ----------------------------------------------------

        with st.spinner("Asking the AI agent..."):

            data, error_message = ask_backend(question.strip())

        if error_message:

            st.error(error_message)
            st.stop()


        # ----------------------------------------------------
        # READ RESPONSE FIELDS
        # ----------------------------------------------------

        answer = data.get("answer", "")
        iterations = data.get("iterations", 0)
        tools_executed = data.get("tools_executed", 0)
        grounded = data.get("grounded", False)
        status = data.get("status", "unknown")
        duplicates_skipped = data.get("duplicates_skipped", 0)
        tools_refused = data.get("tools_refused", 0)


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.divider()

        st.header("Analysis Results")


        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.subheader("Answer")

        if answer.strip():

            st.success(answer)

        else:

            st.info("The AI did not return an answer.")


        # ----------------------------------------------------
        # AGENT INFORMATION
        # ----------------------------------------------------

        st.subheader("Agent Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Iterations", iterations)

        with col2:
            st.metric("Tools Executed", tools_executed)

        with col3:
            st.metric("Grounded", "Yes" if grounded else "No")

        if not grounded:

            st.warning(
                "This answer was not grounded in the resume tools, "
                "so treat it with caution."
            )


        # ----------------------------------------------------
        # EXTRA DETAILS
        # ----------------------------------------------------

        with st.expander("More details"):

            st.write(f"**Status:** {status}")
            st.write(f"**Duplicate tool calls skipped:** {duplicates_skipped}")
            st.write(f"**Tool calls refused:** {tools_refused}")