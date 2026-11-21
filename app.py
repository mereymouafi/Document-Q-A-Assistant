from io import BytesIO
import base64
from pathlib import Path
import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from embed_notes import ask, prepare_document


def clear_document():
    st.session_state.pop("document", None)
    st.session_state.pop("result", None)


st.set_page_config(page_title="Ask your documents", layout="centered")

background_path = Path(__file__).parent / "assets" / "jellyfish.png"

background_image = "none"
if background_path.is_file():
    image_data = base64.b64encode(background_path.read_bytes()).decode("ascii")
    background_image = f"""
        linear-gradient(rgba(24, 47, 66, 0.54), rgba(13, 31, 48, 0.68)),
        url("data:image/png;base64,{image_data}")
    """

st.markdown(
    """
    <style>
    .stApp {
        background-color: #162c3e;
        background-image: __BACKGROUND__;
        background-position: center;
        background-size: cover;
        background-attachment: fixed;
        color: #f4eee5;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 760px;
        margin: clamp(4rem, 12vh, 8rem) auto 3rem;
        padding: clamp(1.25rem, 4vw, 3rem);
        width: calc(100% - 2rem);
        background: rgba(18, 37, 54, 0.88);
        backdrop-filter: blur(12px);
        border-radius: 20px;
    }

    .block-container h1,
    .block-container h2,
    .block-container h3 {
        font-family: Georgia, serif;
        font-weight: 400;
        color: #f4eee5;
    }

    .block-container h1 {
        font-size: clamp(2rem, 5vw, 3.2rem);
        letter-spacing: -0.035em;
        margin: 0 0 0.7rem;
    }

    .block-container p,
    .block-container label {
        color: #e7eef4;
    }

    [data-testid="stCaptionContainer"] p {
        color: #b7cad9;
    }

    [data-testid="stFileUploaderDropzone"],
    [data-testid="stTextInput"] [data-baseweb="input"] {
        background: #263e51;
        border: 1px solid rgba(186, 210, 228, 0.2);
        border-radius: 12px;
    }

    [data-testid="stTextInput"] input {
        color: #f4eee5;
    }

    [data-testid="stForm"] {
        border: 0;
        padding: 0;
    }

    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stFileUploaderDropzone"] button {
        background: #b1cddd;
        color: #102737;
        border: 0;
        border-radius: 10px;
    }

    [data-testid="stButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #d3e4ee;
        color: #102737;
    }

    .block-container button p {
        color: inherit;
    }

    .block-container button:focus-visible {
        outline: 2px solid #f4eee5;
        outline-offset: 3px;
    }
    </style>
    """.replace("__BACKGROUND__", background_image),
    unsafe_allow_html=True,
)

st.title("Ask your documents")

uploaded_files = st.file_uploader(
    "Upload up to 5 PDF or TXT files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    on_change=clear_document,
)

if len(uploaded_files) > 5:
    st.warning("Please remove files until you have 5 or fewer.")
    st.stop()


if uploaded_files and st.button("Index documents"):
    clear_document()
    all_chunks = []
    all_vectors = []
    source_names = []

    with st.spinner("Indexing your documents..."):
        for uploaded_file in uploaded_files:
            try:
                file_bytes = uploaded_file.getvalue()

                if uploaded_file.name.lower().endswith(".pdf"):
                    reader = PdfReader(BytesIO(file_bytes))
                    text = "\n\n".join(
                        (page.extract_text() or "").strip()
                        for page in reader.pages
                    )
                else:
                    text = file_bytes.decode("utf-8-sig")
            except (UnicodeDecodeError, PdfReadError):
                st.error(f"Could not read {uploaded_file.name}.")
                st.stop()

            if not text.strip():
                st.error(f"No readable text found in {uploaded_file.name}.")
                st.stop()

            file_chunks, file_vectors = prepare_document(text)
            all_chunks.extend(file_chunks)
            all_vectors.extend(file_vectors)
            source_names.extend([uploaded_file.name] * len(file_chunks))

    st.session_state.document = (all_chunks, all_vectors, source_names)


if "document" in st.session_state:
    chunks, vectors, source_names = st.session_state.document
    st.caption(
        f"{len(uploaded_files)} files ready · {len(chunks)} passages indexed"
    )

    with st.form("question_form"):
        question = st.text_input("Ask a question about these documents")
        submitted = st.form_submit_button("Ask")

    if submitted:
        question = question.strip()

        if question:
            with st.spinner("Searching your documents..."):
                st.session_state.result = ask(
                    question, chunks, vectors, source_names
                )
        else:
            st.warning("Please enter a question.")

    if "result" in st.session_state:
        answer, passages = st.session_state.result
        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved passages"):
            for passage in passages:
                st.text(passage)