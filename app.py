import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
from google.genai.errors import ClientError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from bm25_search import build_bm25_index
from embed_notes import generate_answer, prepare_document
from hybrid_search import rank_hybrid


def clear_document():
    """Clear the indexes and answer when the uploaded files change."""
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
    """<style>
    [data-testid="stToolbar"],
    [data-testid="stAppViewerBadge"],
    [class*="viewerBadge_container"] {
        display: none !important;
    }

    .stApp {
        background-color: #162c3e;
        background-image: __BACKGROUND__;
        background-position: center;
        background-size: cover;
        background-attachment: fixed;
        color: #f4eee5;
    }

    @keyframes reveal-title {
        from { clip-path: inset(0 100% 0 0); }
        to   { clip-path: inset(0 0 0 0); }
    }

    @media (min-width: 600px) and (prefers-reduced-motion: no-preference) {
        .typing-title {
            display: inline-block;
            animation: reveal-title 0.9s steps(19, end) both;
        }
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

    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] input {
        background-color: #263e51 !important;
        color: #f4eee5 !important;
        caret-color: #f4eee5;
    }

    [data-testid="stTextInput"] input::placeholder {
        color: #b7cad9;
        opacity: 1;
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

animate_title = not st.session_state.get("title_animation_done", False)
title_class = "typing-title" if animate_title else ""
st.session_state.title_animation_done = True

st.markdown(
    f'<h1 class="{title_class}">Ask your documents.</h1>',
    unsafe_allow_html=True,
)
st.markdown("Ask questions about your files and check the passages behind each answer.")

uploaded_files = st.file_uploader(
    "Upload up to 5 PDF or TXT files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    on_change=clear_document,
)

if len(uploaded_files) > 5:
    st.warning("Please remove files until you have 5 or fewer.")
    st.stop()


# 1. Extract and index the user's files.
if uploaded_files and "document" not in st.session_state:
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
                        (page.extract_text() or "").strip() for page in reader.pages
                    )
                else:
                    text = file_bytes.decode("utf-8-sig")
            except (UnicodeDecodeError, PdfReadError):
                st.error(f"Could not read {uploaded_file.name}.")
                st.stop()

            if not text.strip():
                st.error(f"No readable text found in {uploaded_file.name}.")
                st.stop()

            try:
                file_chunks, file_vectors = prepare_document(text)
            except ClientError as error:
                if error.code != 429:
                    raise

                st.warning(
                    "Gemini is still rate-limiting document indexing. "
                    "Wait a minute, then re-upload the file or try a smaller one."
                )
                st.stop()

            # The text, vector, and filename share the same chunk position.
            all_chunks.extend(file_chunks)
            all_vectors.extend(file_vectors)
            source_names.extend([uploaded_file.name] * len(file_chunks))

        if not all_chunks:
            st.error("No passages could be created from the uploaded files.")
            st.stop()

        bm25_index = build_bm25_index(all_chunks)
        st.session_state.document = (
            all_chunks,
            all_vectors,
            source_names,
            bm25_index,
        )


# 2. Reuse the document indexes on later questions.
document = st.session_state.get("document")

if document is not None:
    chunks, vectors, source_names, bm25_index = document

    st.caption(
        f"{len(uploaded_files)} file(s) ready · {len(chunks)} passages indexed"
    )

    with st.form("question_form"):
        question = st.text_input(
            "Ask a question about these documents",
            placeholder="What are the main points?",
        )
        submitted = st.form_submit_button("Ask")

    # 3. Retrieve passages and generate the answer from those passages.
    if submitted:
        question = question.strip()
        st.session_state.pop("result", None)

        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching your documents and writing an answer..."):
                try:
                    indices = rank_hybrid(question, bm25_index, vectors, limit=4)
                    passages = [
                        f"[{rank}] {source_names[index]}\n{chunks[index]}"
                        for rank, index in enumerate(indices, start=1)
                    ]

                    if passages:
                        answer = generate_answer(question, passages)
                    else:
                        answer = "I couldn't find the answer in the uploaded documents."

                    st.session_state.result = (answer, passages)
                except ClientError as error:
                    if error.code != 429:
                        raise

                    st.warning("Gemini is rate-limiting requests. Please try again shortly.")

    # 4. Display the answer and the passages used to generate it.
    if "result" in st.session_state:
        answer, passages = st.session_state.result

        st.subheader("Answer")
        st.write(answer)

        passage_label = "passage" if len(passages) == 1 else "passages"
        st.caption(f"{len(passages)} {passage_label} retrieved")

        if passages:
            with st.expander("Retrieved passages"):
                for passage in passages:
                    st.text(passage)