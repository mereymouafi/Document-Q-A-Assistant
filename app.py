from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from embed_notes import ask, prepare_document


def clear_document():
    st.session_state.pop("document", None)
    st.session_state.pop("result", None)


st.title("Document Q&A Assistant")

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