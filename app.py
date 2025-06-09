# app.py

import streamlit as st
from rag_pipe import answer_query

st.set_page_config(page_title="LLM Text Evaluator", layout="wide")
st.title("🧠 LLM-based Text Evaluation Tool")

# Input Section 
st.header("1. Enter Input for Evaluation")

gen_text = st.text_area("✍️ Input Text", height=200)
context = st.text_area(
    "🎯 Context (Purpose, Audience, Style)",
    placeholder="E.g., High school essay; audience is students; purpose is to educate",
    height=100
)

st.subheader("📂 Choose Reference Input Type")
input_type = st.radio("Reference Source", ["Upload PDF", "Upload TXT", "Paste Text", "From URL"])

uploaded_file = None
reference_text = ""
url = ""

if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
elif input_type == "Upload TXT":
    uploaded_file = st.file_uploader("Upload Text File", type=["txt"])
elif input_type == "Paste Text":
    reference_text = st.text_area("Paste Reference Text", height=200)
elif input_type == "From URL":
    url = st.text_input("Enter Reference URL")

query = st.text_input("Enter your query..")
evaluate = st.button("Evaluate")

if evaluate:
    if not gen_text or not context or not query:
        st.warning("Please provide input text and context.")
    elif input_type in ["Upload PDF", "Upload TXT"] and not uploaded_file:
        st.warning("Please upload a reference file.")
    elif input_type == "From URL" and not url:
        st.warning("Please enter a URL.")
    else:
        with st.spinner("Evaluating..."):
            reference_input = uploaded_file if "Upload" in input_type else reference_text or url
            
            try:
                answer = answer_query(gen_text, context, input_type, reference_input, query)
                st.success(answer)
            except Exception as e:
                st.error(f'Error: , {e}')
