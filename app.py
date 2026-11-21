from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from embed_notes import ask, prepare_document


def clear_document():
    st.session_state.pop("document", None)
    st.session_state.pop("result", None)


st.title("Document Q&A Assistant")

uploaded_file = st.file_uploader(
    "Upload a text file or PDF",
    type=["txt", "pdf"],
    on_change=clear_document,
)

if uploaded_file is not None and st.button("Index document"):
    clear_document()

    try:
        file_bytes = uploaded_file.getvalue()

        if uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(BytesIO(file_bytes))
            page_texts = [
                (page.extract_text() or "").strip()
                for page in reader.pages
            ]
            text = "\n\n".join(page for page in page_texts if page)
        else:
            text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        st.error("Save your text file as UTF-8 and upload it again.")
    except PdfReadError:
        st.error("Could not read this PDF. Try a different copy.")
    else:
        if text.strip():
            with st.spinner("Indexing your document..."):
                chunks, vectors = prepare_document(text)
            st.session_state.document = (uploaded_file.name, chunks, vectors)
        else:
            st.warning("No readable text found. Scanned PDFs may need OCR.")

if "document" in st.session_state:
    filename, chunks, vectors = st.session_state.document
    st.caption(f"Ready to search {filename} · {len(chunks)} passages")

    with st.form("question_form"):
        question = st.text_input("Ask a question about this document")
        submitted = st.form_submit_button("Ask")

    if submitted:
        question = question.strip()
        if question:
            with st.spinner("Searching your document..."):
                st.session_state.result = ask(question, chunks, vectors, filename)
        else:
            st.warning("Please enter a question.")

    if "result" in st.session_state:
        answer, passages = st.session_state.result
        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved passages"):
            for passage in passages:
                st.text(passage)