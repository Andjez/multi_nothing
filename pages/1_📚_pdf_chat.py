#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Androse_Jes
#
# Created:     28-05-2023
# Copyright:   (c) Androse_Jes 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os, re, time
import streamlit as st
from io import BytesIO
import main, loader, faiss
from pypdf import PdfReader
from langchain import OpenAI
from langchain.llms import OpenAI
from streamlit_chat import message
from typing import Any, Dict, List
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


if "generated_01" not in st.session_state:
    st.session_state["generated_01"] = []

if "past_01" not in st.session_state:
    st.session_state["past_01"] = []
chain = None
pages = None
#st.set_page_config(page_title="PDF Chatbot", page_icon=":zany_face:")

st.markdown("# PDF Chatbot 📕📖")
st.sidebar.markdown("# PDF Chatbot 📕📖")

os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]

user_input = ""

col1,col2 = st.columns(2)
with col1:
    if st.button("Clear All"):
    # Clears all st.cache_resource caches:
        st.session_state["link"] = ""
        st.cache_resource.clear()
        st.session_state["generated_01"] = []
        st.session_state["past_01"] = []
        st.session_state["source"] = []
        st.session_state["time_sec"] = []
        st.session_state["foo_"] = ""
        st.cache_data.clear()
with col2:
    if st.button("Clear History"):
        st.session_state["generated_01"] = []
        st.session_state["past_01"] = []
        st.session_state["source"] = []
        st.session_state["time_sec"] = []
        st.session_state["foo_"] = ""

#new lines

@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

@st.cache_resource
def emb_model():
    inst_embeddings = main.embedding()
    return inst_embeddings
    
@st.cache_resource
def test_embed(_pages):
    instructor_embeddings = emb_model()
    # Indexing
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(pages, instructor_embeddings)
        #store = FAISS.from_documents(documents=texts,embedding=instructor_embeddings)
        retriever = index.as_retriever(search_kwargs={"k": 3})
    st.success("Embeddings done.", icon="✅")

    return retriever

uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])

if uploaded_file:
    name_of_file = uploaded_file.name
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)
    chain = test_embed(pages)
    model = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
else:
    st.session_state["generated_01"] = []
    st.session_state["past_01"] = []
    st.session_state["source"] = []
    st.session_state["time_sec"] = []
    st.session_state["foo_"] = ""

if pages:
    with st.expander("Show Page Content", expanded=False):
        page_sel = st.number_input(label="Select Page", min_value=1, max_value=len(pages), step=1)
        pages[page_sel - 1]
if chain:
    user_input = st.text_input("You: ", key="foo_")

if user_input:
    docs = chain.get_relevant_documents(user_input)
    output = docs[0].page_content
    answer = model.run(input_documents=docs, question=user_input)
    st.session_state.past_01.append(user_input)
    st.session_state.generated_01.append(answer)

if st.session_state["generated_01"]:
    for i in range(len(st.session_state["generated_01"]) - 1, -1, -1):
        message(st.session_state["generated_01"][i],avatar_style="adventurer",seed=122, key=str(i))
        message(st.session_state["past_01"][i], avatar_style="adventurer",seed=121, is_user=True, key=str(i+9999) + "_user")
