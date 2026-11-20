import streamlit as st

from embed_notes import ask

st.title("AWS Notes Assistant")

with st.form("question_form"):
    question = st.text_input("Ask a question about your AWS notes")
    submitted = st.form_submit_button("Ask")

if submitted:
    question = question.strip()

    if question:
        with st.spinner("Searching your notes..."):
            answer, passages = ask(question)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved passages"):
            for passage in passages:
                st.text(passage)
    else:
        st.warning("Please enter a question.")