# rag_pipe.py

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from vectorStore import (
    extract_chunks_from_pdf,
    extract_chunks_from_txt,
    extract_chunks_from_text,
    extract_chunks_from_url,
    get_retriever
)

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# define the instructions and structure for the LLM to perform the evaluation.
prompt_template = PromptTemplate(
    template="""You are a expert evaluator. Evaluate the following text based on grammar, coherence, context, and factual accuracy only using the Reference Chunks. If Reference Chunks and Input text is not related, give factual accuracy as 0, do not use your own knowledge. If you dont know something, just say, I dont know.

Input text: {text}
Context: {context}
Reference Chunks: {reference_text}
User Query: {query}

### Instructions:
1. Analyze the text and assign scores out of 100 for:
   - Grammar
   - Coherence
   - Factual Accuracy (dates, numbers)
   - Style
2. Explain each score briefly.
3. Compute a Composite Score (average).
4. Suggest:
   - Corrections (grammar/facts)
   - Additions (for clarity/completeness)

### Output Format:
- Quality Report:
  - Composite Score: <score>/100
  - Grammar: <score>/100 – <feedback>
  - Coherence: <score>/100 – <feedback>
  - Factual Accuracy: <score>/100 – <feedback>
  - Style: <score>/100 – <feedback>
- Suggestions:
  - Correct: <corrected version>
  - Add: <improvement suggestions>
""",
    input_variables=["text", "context", "reference_text", "query"]
)

#Simply convert the LLM's output to a string.
parser = StrOutputParser()                  


# main function called by app.py to initiate the evaluation.
def answer_query(gen_text: str, context: str, reference_type: str, reference_input, query: str):
    # Load and chunk reference content
    if reference_type == "Upload PDF":
        file_bytes = reference_input.read()
        # Extract chunks from reference input
        chunks = extract_chunks_from_pdf(file_bytes)
    elif reference_type == "Upload TXT":
        file_bytes = reference_input.read()
        chunks = extract_chunks_from_txt(file_bytes)
    elif reference_type == "Paste Text":
        chunks = extract_chunks_from_text(reference_input)
    elif reference_type == "From URL":
        chunks = extract_chunks_from_url(reference_input)
    else:
        chunks = []

    # create a FAISS vector store from the extracted chunks and return a retriever object.
    retriever = get_retriever(chunks)

    # The retriever is invoked with a combination of the generated text and the user's query. This helps to retrieve the most relevant chunks from the reference material that are pertinent to both the gen_text and the query.
    retrieved_docs = retriever.invoke(gen_text + " " + query)

    # Extract the actual text content from the retrieved documents.
    relevant_chunks = [doc.page_content for doc in retrieved_docs]



    formatted_prompt = prompt_template.format(
        text=gen_text,
        context=context,
        reference_text="\n".join(relevant_chunks),
        query=query
    )

    # Simple LLm chain
    chain = llm | parser
    return chain.invoke(formatted_prompt)
