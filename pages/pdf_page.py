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

import os, re
import streamlit as st
from langchain import OpenAI
#from apikey import openai_apikey
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
import main
import loader
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains import RetrievalQAWithSourcesChain

import faiss
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


if "generated_01" not in st.session_state:
    st.session_state["generated_01"] = []

if "past_01" not in st.session_state:
    st.session_state["past_01"] = []
st.set_page_config(page_title="PDF Chatbot", page_icon=":zany_face:")
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
with col2:
    if st.button("Clear History"):
        st.session_state["generated_01"] = []
        st.session_state["past_01"] = []
        st.session_state["source"] = []
        st.session_state["time_sec"] = []
        st.session_state["foo"] = ""

@st.cache_resource
def load_chain(texts):
    instructor_embeddings = main.embedding()
    store = FAISS.from_documents(documents=texts,embedding=instructor_embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 3})
    return retriever

#yt_link = st.text_input("enter youtube link here!",key="link")

uploaded_files = st.file_uploader(f"Choose PDF file(s) to Upload", accept_multiple_files=True,key="link_01")

def get_text():
    input_text = st.text_input("You: ", key="foo")
    return input_text
if uploaded_files:
    texts = loader.pdf_loader(uploaded_files)
    chain = load_chain(texts)
    model = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    user_input = get_text()

if user_input:
    #add_video = st.video(yt_link,start_time=0)
    docs = chain.get_relevant_documents(user_input)
    output = docs[0].page_content
    answer = model.run(input_documents=docs, question=user_input)
    st.session_state.past_01.append(user_input)
    st.session_state.generated_01.append(answer)

if st.session_state["generated_01"]:
    for i in range(len(st.session_state["generated_01"]) - 1, -1, -1):
        message(st.session_state["generated_01"][i],avatar_style="adventurer",seed=122, key=str(i))
        message(st.session_state["past_01"][i], avatar_style="adventurer",seed=121, is_user=True, key=str(i+9999) + "_user")
